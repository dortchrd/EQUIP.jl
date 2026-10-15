# Fitting whole datasets.
#
# Data layouts. MRI volumes usually arrive as X×Y×Z×M (NIfTI, MATLAB: observations last), which
# in Julia's column-major order makes each voxel's curve strided by the number of voxels. Instead
# of permuting the dataset, each task copies a block of voxels into a small M×chunk buffer while
# reading the source contiguously (a blocked transpose), then fits from the buffer. When the
# data are observation-major (M×X×Y×Z, e.g. C-ordered NumPy arrays seen from Julia), curves are
# fitted straight from the source. Inputs may be memory-mapped and of any numeric type.
#
# Parallelism: a fixed pool of tasks, each with its own FitCache, pulls chunks of the (masked)
# voxel list from an atomic counter, which balances load when iteration counts vary.

"""
    MapResult

Output of [`fit_maps`](@ref). For data with spatial size `sz` and `P` estimated parameters:

- `names::Vector{Symbol}`: estimated parameter names.
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

struct VoxelMajor end   # Y is nvox × m
struct ObsMajor end     # Y is m × nvox

_spatial_size(sz::Dims, obsdim) = ntuple(i -> sz[i < obsdim ? i : i + 1], length(sz) - 1)

function _mask_indices(mask, nvox)
    length(mask) == nvox || throw(DimensionMismatch("mask has $(length(mask)) voxels, data have $nvox"))
    idx = Int[]
    sizehint!(idx, nvox)
    for (v, mv) in enumerate(mask)
        iszero(mv) || push!(idx, v)
    end
    return idx
end

function _check_maps(nt::NamedTuple, nvox, kind)
    for (n, x) in pairs(nt)
        if x isa AbstractArray && length(x) != nvox
            throw(DimensionMismatch("$kind map `$n` has $(length(x)) voxels, data have $nvox"))
        end
    end
end

_start_maps(s::NamedTuple) = s
_start_maps(r::MapResult) = NamedTuple{Tuple(r.names)}(Tuple(r[n] for n in r.names))

"""
    fit_maps(model, acq, data; kwargs...) -> MapResult

Fit `model` to every voxel of `data`. `acq` is the protocol: a named tuple such as
`(TI = [...], TD = 2.5)` (vectors per observation, scalars constant), or a plain vector for
one-setting models (`fit_maps(MonoExp(), te, data)`).

Keyword arguments:

- `obsdim = ndims(data)`: dimension of `data` holding the observations. `ndims(data)`
  (X×Y×Z×M, NIfTI/MATLAB) and `1` (M×X×Y×Z) are used without copying the dataset.
- `mask`: array with the spatial size; voxels where `mask` is zero/false are skipped.
- `fixed`: named tuple of parameters to hold fixed: numbers, per-voxel maps (spatial size, e.g.
  `B1 = b1map`), or ties (`R1m = p -> p.R1f`). A non-finite map value marks the voxel invalid.
- `free`: parameters to estimate that the model fixes by default: names (`(:R1m,)`) or
  `name => Estimate(start; lower, upper)` pairs.
- `start`: named tuple of starting values (numbers or maps), or a previous `MapResult`.
- `weights`: per-observation weights.
- `precision = Float64`: working and output float type (`Float32` halves memory, ~1.6× faster).
- `maxiter = 100`, `xtol`, `ftol`, `gtol`, `damping`: see [`LMOptions`](@ref).
- `stderr = false`: also compute standard-error maps.
- `ntasks = Threads.nthreads()`, `chunksize = 128`: threading controls.

Start Julia with threads (`julia -t auto`) for parallel fitting.
"""
function fit_maps(model::AbstractModel, acq, data::AbstractArray; obsdim::Integer=ndims(data),
                  precision::Type{<:AbstractFloat}=Float64, stderr::Bool=false, fixed=(;),
                  free=(), kwargs...)
    spatial = _spatial_size(size(data), obsdim)
    names = freenames(setup(model, precision; fixed, free))
    res = MapResult{precision}(names, spatial; stderr)
    return fit_maps!(res, model, acq, data; obsdim, fixed, free, kwargs...)
end

"""
    fit_maps!(res::MapResult, model, acq, data; kwargs...) -> res

In-place version of [`fit_maps`](@ref); `res` may wrap preallocated or memory-mapped arrays.
The working precision is the element type of `res`.
"""
function fit_maps!(res::MapResult{T}, model::AbstractModel, acq, data::AbstractArray;
                   obsdim::Integer=ndims(data), mask=nothing, fixed=(;), free=(), start=(;),
                   weights=nothing, maxiter::Integer=100, xtol=nothing, ftol=nothing,
                   gtol=nothing, damping=1e-3, ntasks::Integer=Threads.nthreads(),
                   chunksize::Integer=128) where {T}
    N = ndims(data)
    1 <= obsdim <= N || throw(ArgumentError("obsdim=$obsdim out of range for $N-d data"))
    m = size(data, obsdim)
    spatial = _spatial_size(size(data), obsdim)
    nvox = prod(spatial; init=1)
    start = _start_maps(start)
    cache = init(model, acq; fixed, free, start, weights, precision=T, maxiter, xtol, ftol, gtol, damping)
    nobs(cache.acq) == m ||
        throw(DimensionMismatch("protocol has $(nobs(cache.acq)) observations, data have $m"))
    P = nfree(cache.spec)
    size(res.params) == (spatial..., P) ||
        throw(DimensionMismatch("result has size $(size(res.params)), expected $((spatial..., P))"))
    res.names == collect(freenames(cache.spec)) ||
        throw(ArgumentError("result parameters $(res.names) differ from the fit's $(freenames(cache.spec))"))
    _check_maps(cache.spec.fixed, nvox, "fixed")
    _check_maps(cache.spec.start, nvox, "start")
    m >= P || @warn "fewer observations ($m) than estimated parameters ($P)"

    Y, layout = if obsdim == N
        reshape(data, nvox, m), VoxelMajor()
    elseif obsdim == 1
        reshape(data, m, nvox), ObsMajor()
    else  # rare: one permuted copy
        perm = (ntuple(i -> i < obsdim ? i : i + 1, N - 1)..., obsdim)
        reshape(permutedims(data, perm), nvox, m), VoxelMajor()
    end
    idx = mask === nothing ? (1:nvox) : _mask_indices(mask, nvox)

    fill!(res.params, T(NaN))
    fill!(res.sse, T(NaN))
    fill!(res.iterations, 0)
    fill!(res.status, UInt8(NotFitted))
    res.stderr === nothing || fill!(res.stderr, T(NaN))
    out = (params=reshape(res.params, nvox, P), sse=vec(res.sse), iterations=vec(res.iterations),
           status=vec(res.status),
           stderr=res.stderr === nothing ? nothing : reshape(res.stderr, nvox, P))
    _fit_threaded!(out, cache, Y, layout, idx, ntasks, chunksize)
    return res
end

# @noinline keeps the compiled solver loop independent of the dataset's dimensionality, so the
# precompiled code is reused for any input shape.
@noinline function _fit_threaded!(out, cache, Y, layout, idx, ntasks, chunksize)
    n = length(idx)
    n == 0 && return out
    nchunks = cld(n, chunksize)
    counter = Threads.Atomic{Int}(1)
    nt = clamp(ntasks, 1, nchunks)
    if nt == 1
        _worker!(out, _copycache(cache), Y, layout, idx, counter, nchunks, chunksize)
    else
        tasks = [Threads.@spawn _worker!(out, _copycache(cache), Y, layout, idx, counter, nchunks,
                                         chunksize) for _ in 1:nt]
        foreach(fetch, tasks)
    end
    return out
end

@noinline function _worker!(out, c::FitCache{M,A,SP,T}, Y, layout, idx, counter, nchunks,
                            chunksize) where {M,A,SP,T}
    m = length(c.ybuf)
    buf = layout isa VoxelMajor ? Matrix{T}(undef, m, chunksize) : nothing
    while true
        k0 = Threads.atomic_add!(counter, 1)
        k0 > nchunks && break
        r = ((k0 - 1) * chunksize + 1):min(k0 * chunksize, length(idx))
        layout isa VoxelMajor && _gather!(buf, Y, idx, r)
        for (k, kk) in enumerate(r)
            v = @inbounds idx[kk]
            y = layout isa VoxelMajor ? view(buf, :, k) : view(Y, :, v)
            _store!(out, v, _fit_voxel(c, y, v), m)
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

@inline function _store!(out, v, r::VoxelResult{NT,P}, m) where {NT,P}
    @inbounds begin
        θ = Tuple(r.params)
        for j in 1:P
            out.params[v, j] = θ[j]
        end
        out.sse[v] = r.sse
        out.iterations[v] = r.iterations
        out.status[v] = UInt8(r.status)
        if out.stderr !== nothing && r.status != InvalidData
            se = Tuple(standard_errors(r, m))
            for j in 1:P
                out.stderr[v, j] = se[j]
            end
        end
    end
    return nothing
end
