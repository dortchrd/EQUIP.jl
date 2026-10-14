# Fitting whole datasets.
#
# Data layouts. MRI volumes usually arrive as X×Y×Z×M (NIfTI, MATLAB: observations last), which
# in Julia's column-major order makes each voxel's curve strided by the number of voxels. Instead
# of permuting the whole dataset, each task copies a block of voxels into a small M×chunk buffer
# while reading the source contiguously (a blocked transpose), then fits from the buffer. When
# the data are already observation-major (M×X×Y×Z, e.g. C-ordered NumPy arrays seen from
# Julia), curves are fitted straight from the source with no copy. Inputs may be memory-mapped
# and of any numeric type; conversion to the working precision happens on the fly.
#
# Parallelism: a fixed pool of tasks pulls chunks of the (masked) voxel list from an atomic
# counter, which balances load when the iteration count varies across voxels. Each task
# allocates its scratch buffers once.

"""
    MapResult

Output of [`fit_maps`](@ref). For data with spatial size `sz` and a model with `P` parameters:

- `names::Vector{Symbol}`: parameter names.
- `params::Array{T}`: size `(sz..., P)`; `r[:T2]` returns the T2 map as a view.
- `sse`: weighted residual sum of squares, size `sz`.
- `iterations::Array{Int32}`, `status::Array{UInt8}` ([`FitStatus`](@ref) codes), size `sz`.
- `stderr`: standard-error maps, size `(sz..., P)`, or `nothing`.

Voxels outside the mask hold `NaN` parameters and status 0.
"""
struct MapResult{T<:AbstractFloat,N,M}
    names::Vector{Symbol}
    params::Array{T,N}
    sse::Array{T,M}
    iterations::Array{Int32,M}
    status::Array{UInt8,M}
    stderr::Union{Nothing,Array{T,N}}
end

function MapResult{T}(names, spatial::Dims; stderr::Bool=false) where {T}
    P = length(names)
    N = length(spatial) + 1
    MapResult{T,N,N - 1}(collect(Symbol, names), Array{T}(undef, spatial..., P),
                         Array{T}(undef, spatial...), Array{Int32}(undef, spatial...),
                         Array{UInt8}(undef, spatial...),
                         stderr ? Array{T}(undef, spatial..., P) : nothing)
end

function _param_index(r::MapResult, name)
    j = findfirst(==(Symbol(name)), r.names)
    j === nothing && throw(KeyError(name))
    return j
end
Base.getindex(r::MapResult, name::Union{Symbol,AbstractString}) =
    selectdim(r.params, ndims(r.params), _param_index(r, name))
Base.keys(r::MapResult) = r.names
function Base.show(io::IO, r::MapResult{T}) where {T}
    sz = size(r.sse)
    nfit = count(s -> s != UInt8(NotFitted) && s != UInt8(InvalidData), r.status)
    nconv = count(s -> isconverged(s), r.status)
    print(io, "MapResult{$T}: ", join(sz, "×"), " voxels, parameters ", Tuple(r.names),
          "; fitted $nfit, converged $nconv")
end

# --------------------------------------------------------------------------------------------
# Input preparation

struct VoxelMajor end   # Y is nvox × m
struct ObsMajor end     # Y is m × nvox

_prepare_x(x::AbstractVector{<:Real}, ::Type{T}) where {T} = Vector{T}(x)
_prepare_x(x::AbstractVector{<:StaticVector{K}}, ::Type{T}) where {K,T} = [SVector{K,T}(v) for v in x]
_prepare_x(x::AbstractVector{<:Tuple}, ::Type{T}) where {T} = [SVector(map(T, v)) for v in x]
function _prepare_x(x::AbstractMatrix, ::Type{T}) where {T}
    size(x, 2) == 1 && return Vector{T}(vec(x))
    return _rows_to_svectors(x, T, Val(size(x, 2)))
end
_rows_to_svectors(x, ::Type{T}, ::Val{K}) where {T,K} =
    [SVector{K,T}(ntuple(k -> x[i, k], Val(K))) for i in axes(x, 1)]

_svec(v, ::Val{P}, ::Type{T}) where {P,T} =
    length(v) == P ? SVector{P,T}(ntuple(i -> T(v[i]), Val(P))) :
    throw(ArgumentError("expected $P values, got $(length(v))"))

# Initial guesses
struct ModelInit end
struct ConstInit{P,T}
    θ::SVector{P,T}
end
struct MapInit{A}
    A::A        # nvox × P
end

_prepare_init(::Nothing, ::Val{P}, T, nvox) where {P} = ModelInit()
_prepare_init(s::Symbol, ::Val{P}, T, nvox) where {P} =
    s === :auto ? ModelInit() : throw(ArgumentError("unknown init option :$s"))
function _prepare_init(v::Union{Tuple,AbstractVector,StaticVector}, ::Val{P}, ::Type{T}, nvox) where {P,T}
    ConstInit(_svec(v, Val(P), T))
end
function _prepare_init(A::AbstractArray, ::Val{P}, T, nvox) where {P}
    (length(A) == nvox * P && size(A, ndims(A)) == P) ||
        throw(ArgumentError("init map must have size (spatial..., $P)"))
    MapInit(reshape(A, nvox, P))
end

@inline function _sanitize(θ::SVector{P}, lo::SVector{P,T}, hi::SVector{P,T}) where {P,T}
    # Replace non-finite starting values (e.g. an all-zero voxel) by a point inside the bounds.
    map(θ, lo, hi) do t, l, h
        tt = T(t)
        isfinite(tt) ? tt : (isfinite(l) && isfinite(h) ? (l + h) / 2 :
                             isfinite(l) ? l + one(T) : isfinite(h) ? h - one(T) : one(T))
    end
end

@inline _start(::ModelInit, m, x, y, v, lo, hi) = _sanitize(initial_guess(m, x, y), lo, hi)
@inline _start(c::ConstInit, m, x, y, v, lo, hi) = c.θ
@inline _start(mi::MapInit, m, x, y, v, lo::SVector{P,T}, hi) where {P,T} =
    _sanitize(SVector{P,T}(ntuple(j -> @inbounds(T(mi.A[v, j])), Val(P))), lo, hi)

# Per-voxel covariates
struct Covariates{K,A}
    C::A        # nvox × K
end
_prepare_cov(::Nothing, nvox) = nothing
function _prepare_cov(C::AbstractArray, nvox)
    if length(C) == nvox   # a single map
        return Covariates{1,typeof(vec(C))}(vec(C))
    end
    K = size(C, ndims(C))
    length(C) == nvox * K || throw(ArgumentError("covariates must have size (spatial...) or (spatial..., K)"))
    M = reshape(C, nvox, K)
    return Covariates{K,typeof(M)}(M)
end
@inline _covariate(::Nothing, v, ::Type{T}) where {T} = nothing
@inline _covariate(c::Covariates{1,<:AbstractVector}, v, ::Type{T}) where {T} =
    SVector{1,T}(@inbounds T(c.C[v]))
@inline _covariate(c::Covariates{K}, v, ::Type{T}) where {K,T} =
    SVector{K,T}(ntuple(k -> @inbounds(T(c.C[v, k])), Val(K)))

function _mask_indices(mask, nvox)
    length(mask) == nvox || throw(DimensionMismatch("mask has $(length(mask)) voxels, data has $nvox"))
    idx = Int[]
    sizehint!(idx, nvox)
    for (v, mv) in enumerate(mask)
        iszero(mv) || push!(idx, v)
    end
    return idx
end

# --------------------------------------------------------------------------------------------

"""
    fit_maps(model, x, data; kwargs...) -> MapResult

Fit `model` to every voxel of `data`. `x` holds the acquisition settings (echo times, b-values,
inversion times, flip angles...), one per observation; use an `nobs × k` matrix when each
observation has `k` settings.

Keyword arguments:

- `obsdim = ndims(data)`: dimension of `data` holding the observations. `ndims(data)`
  (X×Y×Z×M, NIfTI/MATLAB) and `1` (M×X×Y×Z) are handled without copying the dataset.
- `mask`: array with the spatial size; voxels where `mask` is zero/false are skipped.
- `init`: `:auto`/`nothing` (per-voxel [`initial_guess`](@ref)), a vector of `P` values used for
  every voxel, or an array of size `(spatial..., P)` (e.g. a previous fit).
- `lower`, `upper`: bounds (length-`P` vectors, ±Inf for none); defaults from the model.
- `covariates`: per-voxel known quantities passed to `model(x, θ, c)`, size `spatial` or
  `(spatial..., K)` (e.g. a relative B1 map for [`SPGR`](@ref)).
- `weights`: per-observation weights (length `nobs`).
- `precision = Float64`: working and output float type. `Float32` halves memory and is the
  natural choice on GPUs.
- `maxiter = 100`, `xtol`, `ftol`, `gtol`, `damping`: see [`LMOptions`](@ref).
- `stderr = false`: also compute standard-error maps.
- `ntasks = Threads.nthreads()`, `chunksize = 128`: CPU threading controls.
- `backend`: a KernelAbstractions backend (e.g. `CUDABackend()`) to fit on a GPU; needs the
  backend package loaded. `batchsize = 2^18` voxels per transfer.

Start Julia with threads (`julia -t auto`) for parallel CPU fitting.
"""
function fit_maps(model::AbstractModel, x, data::AbstractArray; obsdim::Integer=ndims(data),
                  precision::Type{<:AbstractFloat}=Float64, stderr::Bool=false, kwargs...)
    spatial = _spatial_size(size(data), obsdim)
    res = MapResult{precision}(paramnames(model), spatial; stderr)
    return fit_maps!(res, model, x, data; obsdim, kwargs...)
end

_spatial_size(sz::Dims, obsdim) = ntuple(i -> sz[i < obsdim ? i : i + 1], length(sz) - 1)

"""
    fit_maps!(res::MapResult, model, x, data; kwargs...) -> res

In-place version of [`fit_maps`](@ref); `res` may wrap preallocated or memory-mapped arrays.
The working precision is the element type of `res`.
"""
function fit_maps!(res::MapResult{T}, model::AbstractModel{P}, x, data::AbstractArray;
                   obsdim::Integer=ndims(data), mask=nothing, init=nothing, lower=nothing,
                   upper=nothing, covariates=nothing, weights=nothing, maxiter::Integer=100,
                   xtol=nothing, ftol=nothing, gtol=nothing, damping=1e-3,
                   ntasks::Integer=Threads.nthreads(), chunksize::Integer=128, backend=nothing,
                   batchsize::Integer=2^18) where {T,P}
    N = ndims(data)
    1 <= obsdim <= N || throw(ArgumentError("obsdim=$obsdim out of range for $N-d data"))
    nobs = size(data, obsdim)
    spatial = _spatial_size(size(data), obsdim)
    nvox = prod(spatial; init=1)
    size(res.params) == (spatial..., P) ||
        throw(DimensionMismatch("result has size $(size(res.params)), expected $((spatial..., P))"))
    xs = _prepare_x(x, T)
    length(xs) == nobs || throw(DimensionMismatch("x has $(length(xs)) entries, data has $nobs observations"))
    nobs >= P || @warn "fewer observations ($nobs) than parameters ($P)"

    Y, layout = if obsdim == N
        reshape(data, nvox, nobs), VoxelMajor()
    elseif obsdim == 1
        reshape(data, nobs, nvox), ObsMajor()
    else  # rare: one permuted copy
        perm = (ntuple(i -> i < obsdim ? i : i + 1, N - 1)..., obsdim)
        reshape(permutedims(data, perm), nvox, nobs), VoxelMajor()
    end

    lo = _svec(something(lower, lower_bounds(model)), Val(P), T)
    hi = _svec(something(upper, upper_bounds(model)), Val(P), T)
    all(lo .<= hi) || throw(ArgumentError("lower bounds exceed upper bounds"))
    initspec = _prepare_init(init, Val(P), T, nvox)
    covs = _prepare_cov(covariates, nvox)
    w = weights === nothing ? nothing : Vector{T}(weights)
    w === nothing || length(w) == nobs || throw(DimensionMismatch("weights must have length $nobs"))
    idx = mask === nothing ? (1:nvox) : _mask_indices(mask, nvox)
    opts = LMOptions{T}(; maxiter, xtol, ftol, gtol, damping)

    fill!(res.params, T(NaN))
    fill!(res.sse, T(NaN))
    fill!(res.iterations, 0)
    fill!(res.status, UInt8(NotFitted))
    res.stderr === nothing || fill!(res.stderr, T(NaN))

    out = (params=reshape(res.params, nvox, P), sse=vec(res.sse), iterations=vec(res.iterations),
           status=vec(res.status),
           stderr=res.stderr === nothing ? nothing : reshape(res.stderr, nvox, P))

    if backend === nothing
        _fit_threaded!(out, model, xs, Y, layout, idx, initspec, lo, hi, opts, w, covs,
                       ntasks, chunksize)
    else
        Base.get_extension(@__MODULE__, :VoxelFitKernelAbstractionsExt) === nothing &&
            error("`backend` requires KernelAbstractions.jl and a GPU package (e.g. `using CUDA`)")
        _fit_backend!(backend, out, model, xs, Y, layout, idx, initspec, lo, hi, opts, w, covs,
                      batchsize)
    end
    return res
end

function _fit_backend! end   # see ext/VoxelFitKernelAbstractionsExt.jl

# @noinline on the two functions below keeps the compiled solver loop independent of the
# dataset's dimensionality, so the precompiled code is reused for any input shape.
@noinline function _fit_threaded!(out, model, xs, Y, layout, idx, initspec, lo, hi, opts, w, covs,
                                  ntasks, chunksize)
    n = length(idx)
    n == 0 && return out
    nchunks = cld(n, chunksize)
    counter = Threads.Atomic{Int}(1)
    nt = clamp(ntasks, 1, nchunks)
    if nt == 1
        _worker!(out, model, xs, Y, layout, idx, initspec, lo, hi, opts, w, covs, counter,
                 nchunks, chunksize)
    else
        tasks = [Threads.@spawn _worker!(out, model, xs, Y, layout, idx, initspec, lo, hi, opts,
                                         w, covs, counter, nchunks, chunksize) for _ in 1:nt]
        foreach(fetch, tasks)
    end
    return out
end

@noinline function _worker!(out, model, xs, Y, layout, idx, initspec, lo::SVector{P,T}, hi, opts,
                            w, covs, counter, nchunks, chunksize) where {P,T}
    nobs = length(xs)
    buf = layout isa VoxelMajor ? Matrix{T}(undef, nobs, chunksize) : nothing
    ws = workspace(model, T, nobs)
    while true
        c = Threads.atomic_add!(counter, 1)
        c > nchunks && break
        r = ((c - 1) * chunksize + 1):min(c * chunksize, length(idx))
        layout isa VoxelMajor && _gather!(buf, Y, idx, r)
        for (k, kk) in enumerate(r)
            v = @inbounds idx[kk]
            y = layout isa VoxelMajor ? view(buf, :, k) : view(Y, :, v)
            _fit_and_store!(out, v, model, xs, y, initspec, lo, hi, opts, w, covs, ws)
        end
    end
    return nothing
end

# Blocked transpose: the inner loop walks voxels, contiguous in a voxel-major source.
function _gather!(buf::Matrix{T}, Y, idx, r) where {T}
    @inbounds for i in axes(Y, 2)
        for (k, kk) in enumerate(r)
            buf[i, k] = T(Y[idx[kk], i])
        end
    end
    return buf
end

@inline function _allfinite_data(y)
    @inbounds for i in 1:length(y)
        isfinite(y[i]) || return false
    end
    return true
end

# Not inlined: with voxel-major data `y` is always a view of the Matrix{T} buffer, so this
# heavy function is compiled once per model and precision whatever the input element type.
@noinline function _fit_and_store!(out, v, model, xs, y, initspec, lo::SVector{P,T}, hi, opts, w,
                                   covs, ws) where {P,T}
    if !_allfinite_data(y)
        @inbounds out.status[v] = UInt8(InvalidData)
        return nothing
    end
    m = bind_covariates(model, _covariate(covs, v, T))
    θ0 = _start(initspec, m, xs, y, v, lo, hi)
    r = lm_fit(m, xs, y, θ0, lo, hi, opts, w, ws)
    @inbounds begin
        for j in 1:P
            out.params[v, j] = r.θ[j]
        end
        out.sse[v] = r.sse
        out.iterations[v] = r.iterations
        out.status[v] = UInt8(r.status)
        if out.stderr !== nothing
            se = standard_errors(r, length(y))
            for j in 1:P
                out.stderr[v, j] = se[j]
            end
        end
    end
    return nothing
end

# --------------------------------------------------------------------------------------------

"""
    fit_voxel(model, x, y; init=:auto, lower, upper, covariates=nothing, weights=nothing,
              precision=Float64, kwargs...) -> LMResult

Fit a single signal curve `y` measured at `x`. Keyword arguments as in [`fit_maps`](@ref).
"""
function fit_voxel(model::AbstractModel{P}, x, y::AbstractVector; init=nothing, lower=nothing,
                   upper=nothing, covariates=nothing, weights=nothing,
                   precision::Type{<:AbstractFloat}=Float64, maxiter::Integer=100, xtol=nothing,
                   ftol=nothing, gtol=nothing, damping=1e-3) where {P}
    T = precision
    xs = _prepare_x(x, T)
    ys = Vector{T}(y)
    lo = _svec(something(lower, lower_bounds(model)), Val(P), T)
    hi = _svec(something(upper, upper_bounds(model)), Val(P), T)
    c = covariates === nothing ? nothing : SVector{length(covariates),T}(covariates)
    m = bind_covariates(model, c)
    θ0 = _start(_prepare_init(init, Val(P), T, 1), m, xs, ys, 1, lo, hi)
    w = weights === nothing ? nothing : Vector{T}(weights)
    opts = LMOptions{T}(; maxiter, xtol, ftol, gtol, damping)
    return lm_fit(m, xs, ys, θ0, lo, hi, opts, w, workspace(model, T, length(xs)))
end
