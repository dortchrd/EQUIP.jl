# =================================================================================================
# Fitting one voxel
#
#   FitCache (cache.jl) + one voxel's data y
#        │
#        ▼  _fit_voxel   copy y in, pick the voxel's map values, set its start (starts.jl and
#                        loglinear.jl), run LM
#   result               (levmar.jl). Allocates nothing.
#
# Three ways in:
#   fit!(cache, y)       for your own loops: reuses the cache, allocates nothing
#   fit(m, y)            one voxel in one call (y a vector): makes a cache with `init`, then `fit!`
#   the dataset loop     dataset.jl calls `_fit_voxel` for every voxel and stores the results in maps
#
# `fit` and `fit!` are not our own functions: they belong to StatsAPI.jl (the interface package
# behind StatsBase, GLM, Distributions). We add methods for our types, so `using EQUIP,
# StatsBase` works without a name clash.
# =================================================================================================

# One voxel's fit, inside the package: `fit!` adds the model and settings to make a FitResult,
# and the dataset loop (dataset.jl) stores the fields in maps.
struct _VoxelFit{NT,T}
    params::NT
    sse::T
    iterations::Int32
    status::FitStatus
end

"""
    fit!(cache, y; voxel = 1) -> FitResult

Fit one voxel's data `y` (one value per observation). `voxel` is the voxel's linear index into
any per-voxel maps given as parameter keywords (fixed values or starts). Does not allocate.

```julia
cache = init(QMTSIR(TI = TI, TD = TD); kmf = 15.0)
r = fit!(cache, y)
r.params.PSR
```
"""
function fit!(c::FitCache, y::AbstractVector; voxel::Integer = 1)
    length(y) == length(c.y) ||
        throw(DimensionMismatch("data have $(length(y)) values, the model has $(length(c.y)) observations"))
    r = _fit_voxel(c, c.spec, y, voxel)
    fixed = _voxel_fixed(c.spec, voxel)
    return FitResult(r.params, fixed, r.sse, r.iterations, r.status, c.model, c.settings)
end
# The fixed values a voxel's fit used: numbers (a map's value at the voxel) and ties, as written.
_voxel_fixed(spec::FitSpec{F,P,T}, v) where {F,P,T} = map(x -> _untie(_pick(x, v, T)), spec.fixed)
# `fit!` is StatsAPI's function (imported in EQUIP.jl); this adds a method for FitCache.

"""
    fit(model, y::AbstractVector; precision, solver, params...) -> FitResult

Fit one voxel: `y` has one value per observation. Returns a [`FitResult`](@ref) with plain
numbers, e.g. `r.params.T1`, `r.status`.

Keywords named after the model's parameters change their roles: a value fixes one
(`kmf = 15.0`), `Free(...)` estimates one, with any start and bounds given
(`kmf = Free()`, `R1f = Free(start = 0.8)`, `Sf = Free(lower = -1.5)`). Options: `precision`
(`Float64` or `Float32`) and `solver` ([`LM`](@ref)`(maxiter = 50, ...)`).

For many single-voxel fits in a loop, make the cache once with `init` and call `fit!` instead.

```julia
m = QMTSIR(TI = [0.010, 0.050, 0.277, 0.843], TD = 2.5)
r = fit(m, y; kmf = 15.0)
r.params.PSR
```
"""
fit(model::MRIModel, y::AbstractVector; kwargs...) = fit!(init(model; kwargs...), y)

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

# Above this many free parameters, the voxel fit's larger pieces are called as functions, not
# inlined (decided when compiling: P is a type parameter): the log start here, and the voxel fit
# itself in the dataset loop (dataset.jl). Inlined at P = 22 (DKI), the compiler made slow code of
# the whole: a dataset fit 83% slower than a loop over `fit!` of the same voxels, its start 70%
# slower (measured 2026-11-08); called as functions, the small models' fits are 4–23% slower, so
# they keep them inlined. The same threshold as crlb's kernel (`_CRLB_INLINE_MAXP`), for the same
# reason (docs/design.md, "crlb at many parameters").
const _FIT_INLINE_MAXP = 8

# Fit voxel `v`, whose data are `y`: the steps both `fit!` and the dataset loop (dataset.jl) run.
# Returns the lean per-voxel result (`_VoxelFit`); the callers check `y`'s length first.
# `spec` is passed separately (not read from `c` inside) so that its type parameters F, P, T
# can be named in the method signature and used below (a "function barrier").
function _fit_voxel(c::FitCache, spec::FitSpec{F,P,T}, y, v) where {F,P,T}
    copyto!(c.y, y)                    # 0. the data, into the cache: contiguous and in precision T
    # 1. This voxel's fixed values: (kmf = 12.5, R1m = Fixed(:R1f), Sm = sm_map[v]).
    fixed = map(x -> _pick(x, v, T), spec.fixed)
    # 2. This voxel's starting point, as an SVector in the order F.
    θ0 = SVector{P,T}(Tuple(map(x -> _pick(x, v, T), spec.start)))
    # 3. Bad inputs mark the voxel invalid instead of producing a meaningless fit.
    if !(all(isfinite, c.y) && all(_finite, values(fixed)) && all(isfinite, θ0))
        return _invalid(spec)
    end
    prob = VoxelProblem{F}(c.model, c.y, fixed, spec.weights)
    # 4. Starting point from the data (starts.jl, loglinear.jl): parameters starting at `:data` from the model's
    #    estimates (`datastart`); those starting at `:loglinear` from the weighted fit of ln y with
    #    ln S; then a `BestOf` parameter at its best value, or else the `:scale` parameter at the
    #    value that best matches the data's scale.
    θ0 = _data_start(prob, θ0, spec)
    θ0 = P > _FIT_INLINE_MAXP ? (@noinline _loglinear_start(prob, θ0, spec, c.buf, c.logpass)) :
                                _loglinear_start(prob, θ0, spec, c.buf, c.logpass)
    if spec.candindex > 0
        θ0 = _best_start(prob, θ0, spec, c.buf)
    elseif spec.scale > 0
        θ0 = _scale_start(prob, θ0, spec.scale, spec.lower, spec.upper, c.buf)
    end
    # 5. Run LM and attach the names: (PSR = θ₁, R1f = θ₂, ...). A model whose labels can swap
    #    picks the ones to report (see `relabel`; the default changes nothing).
    r = levmar(prob, θ0, spec.lower, spec.upper, c.opts, c.buf, c.geo)
    return _VoxelFit(relabel(c.model, NamedTuple{F}(Tuple(r.θ))), r.sse, r.iterations, r.status)
end
