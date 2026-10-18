# =================================================================================================
# The reusable cache and fitting one voxel
#
#   FitSpec (setup.jl)
#        │
#        ▼  init         FitSpec + model + solver options + work buffers
#   FitCache             reusable state, like LsqFit's OnceDifferentiable. Made ONCE (per thread).
#        │
#        ▼  fit!         copy one voxel's data in, pick its map values, run LM (levmar.jl).
#   VoxelResult          Called once per voxel; allocates nothing.
# =================================================================================================

"""
    FitCache

Reusable state for fitting voxels with one model and one set of roles: the model (its
acquisition converted to the working precision), the [`FitSpec`](@ref), solver options and work
buffers. Make it with [`init`](@ref), use it with [`fit!`](@ref). Not thread-safe: use one
cache per thread (`fit` on a dataset does this for you).
"""
struct FitCache{M,S<:FitSpec,T,B}
    model::M            # the model, its acquisition vectors converted to precision T
    spec::S
    opts::SolverOptions{T}
    buf::B              # Vector{Dual{FitTag,T,P}}: the model output, one per observation
    y::Vector{T}        # the current voxel's data
end

function Base.show(io::IO, c::FitCache{M,S,T}) where {M,S,T}
    print(io, "FitCache(", nameof(M), "; free ", freenames(c.spec), ", fixed ", keys(c.spec.fixed),
          ", ", length(c.y), " observations, ", T, ")")
end
# Adding a method to `Base.show` controls how the object prints in the REPL. `nameof(M)` is the
# model's bare type name, e.g. `QMTSIR`.

"""
    init(model; fixed = (;), free = (), start = (;), precision = Float64,
         maxiter = 100, xtol, ftol, gtol, damping) -> FitCache

Set up a reusable cache for fitting single voxels with [`fit!`](@ref).

- `fixed`, `free`, `start`: override the model's parameter roles (see [`setup`](@ref)).
  Maps (arrays) are indexed by the `voxel` keyword of `fit!`.
- `precision`: `Float64` or `Float32`.
- `maxiter`, `xtol`, `ftol`, `gtol`, `damping`: solver settings ([`SolverOptions`](@ref)).

`init` is `CommonSolve.init`, the SciML convention, so it does not clash with other packages.
"""
function CommonSolve.init(model::MRIModel; fixed = (;), free = (), start = (;),
                          precision::Type{<:AbstractFloat} = Float64, kwargs...)
    T = precision
    n = nobs(model)                                # also checks the acquisition's lengths
    spec = setup(model, T; fixed, free, start)    # `; fixed` is short for `; fixed = fixed`
    opts = SolverOptions{T}(; kwargs...)               # remaining keywords go to the solver
    return _cache(_with_precision(model, T), spec, opts, n)
end

# A separate function so that P (the number of free parameters) is available as a type
# parameter, to allocate the dual-number buffer with the right type.
_cache(model, spec::FitSpec{F,P,T}, opts, n) where {F,P,T} =
    FitCache(model, spec, opts, Vector{Dual{FitTag,T,P}}(undef, n), Vector{T}(undef, n))

# Same setup, fresh buffers: one per thread.
_copycache(c::FitCache) = FitCache(c.model, c.spec, c.opts, similar(c.buf), similar(c.y))

# =================================================================================================
# Fitting one voxel
# =================================================================================================

"""
    VoxelResult

Result of one voxel fit: `params` (named tuple of the free parameters), `sse` (sum of squared
residuals), `iterations` and `status` ([`FitStatus`](@ref)).
"""
struct VoxelResult{NT,T}
    params::NT
    sse::T
    iterations::Int32
    status::FitStatus
end

isconverged(r::VoxelResult) = isconverged(r.status)

"""
    fit!(cache, y; voxel = 1) -> VoxelResult

Fit one voxel's data `y` (one value per observation). `voxel` is the voxel's linear index into
any per-voxel maps given as `fixed` or `start` values. Does not allocate.

```julia
cache = init(QMTSIR(TI = TI, TD = TD); fixed = (kmf = 15.0,))
r = fit!(cache, y)
r.params.PSR
```
"""
function fit!(c::FitCache, y::AbstractVector; voxel::Integer = 1)
    length(y) == length(c.y) ||
        throw(DimensionMismatch("data have $(length(y)) values, the model has $(length(c.y)) observations"))
    copyto!(c.y, y)                    # into the cache: contiguous and in precision T
    return _fit_voxel(c, c.spec, voxel)
end
# `fit!` is StatsAPI's function (imported in EQUIP.jl); this adds a method for FitCache.

# Per-voxel values: a map is indexed by the voxel; numbers and ties pass through unchanged.
@inline _pick(x::AbstractArray, v, ::Type{T}) where {T} = @inbounds T(x[v])
@inline _pick(x, v, ::Type{T}) where {T} = x

# NaN/Inf checks that skip ties (functions).
@inline _finite(x::Real) = isfinite(x)
@inline _finite(x) = true

function _invalid(::FitSpec{F,P,T}) where {F,P,T}
    nan = NamedTuple{F}(ntuple(_ -> T(NaN), Val(P)))
    return VoxelResult(nan, T(NaN), Int32(0), InvalidData)
end

# `spec` is passed separately (not read from `c` inside) so that its type parameters F, P, T
# can be named in the method signature and used below.
function _fit_voxel(c::FitCache, spec::FitSpec{F,P,T}, v) where {F,P,T}
    # 1. This voxel's fixed values: (kmf = 12.5, R1m = Fixed(:R1f), Sm = sm_map[v]).
    fixed = map(x -> _pick(x, v, T), spec.fixed)
    # 2. This voxel's starting point, as an SVector in the order F.
    θ0 = SVector{P,T}(Tuple(map(x -> _pick(x, v, T), spec.start)))
    # 3. Bad inputs mark the voxel invalid instead of producing a meaningless fit.
    if !(all(isfinite, c.y) && all(_finite, values(fixed)) && all(isfinite, θ0))
        return _invalid(spec)
    end
    prob = VoxelProblem{F}(c.model, c.y, fixed)
    # 4. Starting point from the data: the model's own estimates, if it has them (`datastart`);
    #    then the best value on the model's grid, if it has one (`startgrid`), or a `Free(:scale)`
    #    parameter at the value that best matches the data's scale.
    θ0 = _data_start(prob, θ0, spec)
    if spec.gridindex > 0
        θ0 = _grid_start(prob, θ0, spec, c.buf)
    elseif spec.scale > 0
        θ0 = _scale_start(prob, θ0, spec.scale, spec.lower, spec.upper, c.buf)
    end
    # 5. Run LM and attach the names: (PSR = θ₁, R1f = θ₂, ...). A model whose labels can swap
    #    picks the ones to report (see `relabel`; the default changes nothing).
    r = levmar(prob, θ0, spec.lower, spec.upper, c.opts, c.buf)
    return VoxelResult(relabel(c.model, NamedTuple{F}(Tuple(r.θ))), r.sse, r.iterations, r.status)
end

# Starting values from the model's own estimates on this voxel's data (see `datastart`), for the
# free parameters without a `start` from the user. A NaN or Inf estimate is skipped (the usual
# start stays); the others are clamped to the bounds. A `:scale` parameter is skipped too: it is
# set afterwards, by least squares given the other starts.
function _data_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec) where {F,P,T}
    d = datastart(prob.model, prob.y)
    d === nothing && return θ0                  # no data start: decided when compiling, no cost
    for (name, x) in pairs(d)
        k = findfirst(==(name), F)              # position among the free parameters, or nothing
        if k !== nothing && k != spec.scale && !spec.userstart[k] && isfinite(x)
            θ0 = Base.setindex(θ0, clamp(T(x), spec.lower[k], spec.upper[k]), k)
        end
    end
    return θ0
end

# Starting point from a grid (see `startgrid`): for each grid value g of free parameter k, evaluate
# the signal (with a `:scale` parameter at its least-squares value for that g) and its SSE; keep
# the best g. Plain numbers only, in the dual buffer as in `_scale_start`; nothing is allocated.
function _grid_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec, buf) where {F,P,T}
    k, sc = spec.gridindex, spec.scale
    best, bestsse = θ0, T(Inf)
    for g in spec.grid
        θ = Base.setindex(θ0, g, k)
        sc > 0 && (θ = Base.setindex(θ, one(T), sc))          # signal at scale 1
        signal!(buf, prob.model, _params(Val(F), θ, prob.fixed))
        sy = ss = yy = zero(T)
        @inbounds for i in eachindex(buf)
            s, y = T(value(buf[i])), T(prob.y[i])
            sy += s * y
            ss += s * s
            yy += y * y
        end
        if sc > 0                                            # best scale a for this g
            a = sy / ss
            a = isfinite(a) && a > 0 ? clamp(a, spec.lower[sc], spec.upper[sc]) : one(T)
            θ = Base.setindex(θ, a, sc)
            sse = yy - 2a * sy + a^2 * ss                    # Σ (a·s - y)²
        else
            sse = yy - 2sy + ss                              # Σ (s - y)²
        end
        if sse < bestsse                                     # NaN never wins
            best, bestsse = θ, sse
        end
    end
    # No usable grid value (e.g. NaN signals): fall back to the usual start.
    isfinite(bestsse) && return best
    return sc > 0 ? _scale_start(prob, θ0, sc, spec.lower, spec.upper, buf) : θ0
end

# Starting value of the scale parameter k (S0, M0, ...). The signal is proportional to it, so with
# s = the signal at scale 1 (θ0 holds 1 there), the best fit to y is  scale = (s·y) / (s·s).
# The model is evaluated with plain numbers; storing them in the dual buffer just sets the
# derivative parts to zero, so no extra buffer is needed.
function _scale_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, k, lo, hi, buf) where {F,P,T}
    signal!(buf, prob.model, _params(Val(F), θ0, prob.fixed))
    sy = zero(T)
    ss = zero(T)
    @inbounds for i in eachindex(buf)
        s = T(value(buf[i]))
        sy += s * T(prob.y[i])
        ss += s * s
    end
    a = sy / ss
    a = isfinite(a) && a > 0 ? a : one(T)     # e.g. zero signal at the start: keep 1
    return Base.setindex(θ0, clamp(a, lo[k], hi[k]), k)   # a copy of θ0 with entry k replaced
end
