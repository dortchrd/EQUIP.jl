# =================================================================================================
# The reusable cache and fitting one voxel
#
#   FitSpec (setup.jl)
#        │
#        ▼  init         FitSpec + model + solver options + work buffers
#   FitCache             reusable state, like LsqFit's OnceDifferentiable. Made ONCE (per thread).
#        │
#        ▼  fit!         copy one voxel's data in, pick its map values, run LM (levmar.jl).
#   FitResult            Called once per voxel; allocates nothing.
# =================================================================================================

"""
    FitCache

Reusable state for fitting voxels with one model and one set of roles: the model (its
acquisition converted to the working precision), the [`FitSpec`](@ref), solver options and work
buffers. Make it with [`init`](@ref), use it with [`fit!`](@ref). Not thread-safe: use one
cache per thread (`fit` on a dataset does this for you).
"""
struct FitCache{M,S<:FitSpec,T,B,K}
    model::M            # the model, its acquisition vectors converted to precision T
    spec::S
    opts::SolverOptions{T}
    buf::B              # Vector{Dual{FitTag,T,P}}: the model output, one per observation
    y::Vector{T}        # the current voxel's data
    settings::K         # the keywords of `init`, passed on to every FitResult
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
    settings = (; fixed, free, start, precision, kwargs...)   # kept for the results
    return _cache(_with_precision(model, T), spec, opts, n, settings)
end

# A separate function so that P (the number of free parameters) is available as a type
# parameter, to allocate the dual-number buffer with the right type.
_cache(model, spec::FitSpec{F,P,T}, opts, n, settings) where {F,P,T} =
    FitCache(model, spec, opts, Vector{Dual{FitTag,T,P}}(undef, n), Vector{T}(undef, n), settings)

# Same setup, fresh buffers: one per thread.
_copycache(c::FitCache) = FitCache(c.model, c.spec, c.opts, similar(c.buf), similar(c.y), c.settings)

# =================================================================================================
# Fitting one voxel
# =================================================================================================

# One voxel's fit, inside the package: `fit!` adds the model and settings to make a FitResult,
# and the dataset loop (fit.jl) stores the fields in maps.
struct _VoxelFit{NT,T}
    params::NT
    sse::T
    iterations::Int32
    status::FitStatus
end

"""
    fit!(cache, y; voxel = 1) -> FitResult

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
    r = _fit1!(c, y, voxel)
    return FitResult(r.params, r.sse, r.iterations, r.status, c.model, c.settings)
end

# Copy one voxel's data into the cache and fit it (the dataset loop calls this directly).
function _fit1!(c::FitCache, y, voxel)
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
    return _VoxelFit(nan, T(NaN), Int32(0), InvalidData)
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
    # 4. Starting point from the data (starts.jl): the model's own estimates, if it has them
    #    (`datastart`); then the best value on the model's grid, if it has one (`startgrid`), or a
    #    `Free(:scale)` parameter at the value that best matches the data's scale.
    θ0 = _data_start(prob, θ0, spec)
    if spec.gridindex > 0
        θ0 = _grid_start(prob, θ0, spec, c.buf)
    elseif spec.scale > 0
        θ0 = _scale_start(prob, θ0, spec.scale, spec.lower, spec.upper, c.buf)
    end
    # 5. Run LM and attach the names: (PSR = θ₁, R1f = θ₂, ...). A model whose labels can swap
    #    picks the ones to report (see `relabel`; the default changes nothing).
    r = levmar(prob, θ0, spec.lower, spec.upper, c.opts, c.buf)
    return _VoxelFit(relabel(c.model, NamedTuple{F}(Tuple(r.θ))), r.sse, r.iterations, r.status)
end
