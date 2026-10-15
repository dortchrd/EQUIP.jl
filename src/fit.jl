# =================================================================================================
# From "what the user asked for" to "fitting one voxel"
#
#   parameters(m)  +  fixed / free / start keywords
#        │
#        ▼  setup        merge the user's choices into the model's default roles, then split the
#   FitSpec              parameters into Free (estimated) and Fixed. Done ONCE per fit call.
#        │
#        ▼  init         FitSpec + acquisition + solver options + work buffers
#   FitCache             reusable state, like LsqFit's OnceDifferentiable. Made ONCE (per thread).
#        │
#        ▼  fit!         copy one voxel's data in, pick its map values, run LM.
#   VoxelResult          Called once per voxel; allocates nothing.
# =================================================================================================

"""
    FitSpec{F,P,T}

The resolved parameter roles of one fit (made by `setup`):

- `F`: names of the free parameters, e.g. `(:PSR, :R1f, :Sf, :M0f)`, in model order
- `P = length(F)`, `T`: the working float type
- `lower`, `upper`: bounds of the free parameters (`SVector{P,T}`)
- `start`: starting values of the free parameters (numbers or per-voxel maps), names `F`
- `scale`: position in `F` of the `Free(:scale)` parameter, or 0 if there is none
- `fixed`: the fixed parameters (numbers, per-voxel maps or ties), by name

`F` and `P` are type parameters rather than fields: the compiler then knows the names and the
number of free parameters and generates code specialized for them (fixed-size vectors and
compile-time named tuples instead of name lookups at run time).
"""
struct FitSpec{F,P,T,S,X}
    lower::SVector{P,T}
    upper::SVector{P,T}
    start::S        # NamedTuple with names F (a :scale parameter holds 1 here)
    scale::Int      # which free parameter starts at the least-squares scale (0 = none)
    fixed::X        # NamedTuple with the fixed parameters' names
end

freenames(::FitSpec{F}) where {F} = F       # read the names back out of the type

# --- How each kind of user input becomes a role --------------------------------------------------

# `fixed = (kmf = 15.0, B1 = b1map, R1m = p -> p.R1f)`: plain values become Fixed(value).
_asfixed(x::Fixed) = x
_asfixed(x::Free) = throw(ArgumentError("use `free` (not `fixed`) to estimate a parameter"))
_asfixed(x) = Fixed(x)

# `free` accepts names, `free = (:kmf,)` or `free = :kmf`, or roles, `free = (kmf = Free(...),)`.
# The result is always a named tuple of Free roles.
_asfree(free::Symbol, roles) = _asfree((free,), roles)
_asfree(free::NamedTuple, roles) =
    map(x -> x isa Free ? x : throw(ArgumentError("`free` values must be Free(...), got $x")), free)
function _asfree(free, roles)                    # a tuple or vector of names
    names = Tuple(Symbol(n) for n in free)
    _check_names("free", names, keys(roles))
    return NamedTuple{names}(map(n -> _tofree(roles[n], n), names))
end
# Freeing a parameter by name: a Free one stays as is; a Fixed number becomes the starting value
# (with no bounds; give `name = Free(init; lower, upper)` for bounds); maps and ties need a start.
_tofree(r::Free, n) = r
_tofree(r::Fixed{<:Real}, n) = Free(r.value)
_tofree(r::Fixed, n) = throw(ArgumentError("give `$n` a starting value: free = ($n = Free(init),)"))

# Numbers are converted to the working precision; maps and ties are kept as they are.
_value(x::Real, ::Type{T}) where {T} = T(x)
_value(x, ::Type{T}) where {T} = x

"""
    setup(model, T = Float64; fixed = (;), free = (), start = (;)) -> FitSpec

Combine the model's default roles with the user's choices and split the parameters into free
and fixed:

- `fixed`: named tuple of parameters to hold fixed: numbers, per-voxel maps or ties.
- `free`: parameters to estimate that the model fixes by default: names (`(:kmf,)`) or a
  named tuple of roles (`(kmf = Free(12.5; lower = 0),)`).
- `start`: named tuple of starting values for free parameters: numbers or per-voxel maps.
"""
function setup(model::AbstractModel, ::Type{T} = Float64; fixed = (;), free = (), start = (;)) where {T}
    # 1. The model's defaults, e.g. (PSR = Free(0.1; ...), ..., kmf = Fixed(12.5), ...).
    roles = parameters(model)
    names = keys(roles)

    # 2. Apply `fixed`: `merge(a, b)` replaces a's entries by b's with the same name and keeps
    #    a's order, so (…, kmf = Fixed(12.5), …) becomes (…, kmf = Fixed(15.0), …).
    fixed isa NamedTuple || throw(ArgumentError("`fixed` must be a named tuple, e.g. (kmf = 15.0,)"))
    _check_names("fixed", keys(fixed), names)
    roles = merge(roles, map(_asfixed, fixed))

    # 3. Apply `free` the same way.
    free = _asfree(free, roles)
    _check_names("free", keys(free), names)
    for n in keys(free)
        n in keys(fixed) && throw(ArgumentError("`$n` is in both `fixed` and `free`"))
    end
    roles = merge(roles, free)

    # 4. Split by role, keeping the model's order.
    F = Tuple(n for n in names if roles[n] isa Free)     # e.g. (:PSR, :R1f, :Sf, :M0f)
    X = Tuple(n for n in names if roles[n] isa Fixed)    # e.g. (:kmf, :R1m, :Sm)
    isempty(F) && throw(ArgumentError("no free parameters: everything is fixed"))
    P = length(F)

    # 5. Starting values: each Free role's `init`, overridden by the `start` keyword. A `:scale`
    #    parameter gets the placeholder 1 here; `_fit_voxel` replaces it voxel by voxel.
    start isa NamedTuple || throw(ArgumentError("`start` must be a named tuple, e.g. (T1 = 1.2,)"))
    _check_names("start", keys(start), F)                # only free parameters have starts
    isscale(n) = roles[n].init === :scale && !haskey(start, n)
    count(isscale, F) <= 1 || throw(ArgumentError("only one parameter can start as :scale"))
    scale = something(findfirst(isscale, F), 0)          # findfirst gives `nothing` if none
    init_value(n) = roles[n].init === :scale ? one(T) : T(roles[n].init)
    starts = merge(NamedTuple{F}(map(init_value, F)), map(x -> _value(x, T), start))

    # 6. Bounds as fixed-size vectors, and the fixed values.
    lower = SVector{P,T}(map(n -> T(roles[n].lower), F))
    upper = SVector{P,T}(map(n -> T(roles[n].upper), F))
    all(lower .<= upper) || throw(ArgumentError("a lower bound exceeds its upper bound"))
    fixedvals = NamedTuple{X}(map(n -> _value(roles[n].value, T), X))

    # The names F and their number P go into the type: FitSpec{(:PSR, :R1f, :Sf, :M0f), 4, ...}.
    return FitSpec{F,P,T,typeof(starts),typeof(fixedvals)}(lower, upper, starts, scale, fixedvals)
end

# =================================================================================================
# The reusable cache
# =================================================================================================

"""
    FitCache

Reusable state for fitting voxels with one model, acquisition and set of roles: the
[`FitSpec`](@ref), solver options and work buffers. Make it with [`init`](@ref), use it with
[`fit!`](@ref). Not thread-safe: use one cache per thread (`fit_maps` does this for you).
"""
struct FitCache{M,A,S<:FitSpec,T,B}
    model::M
    acq::A              # acquisition, converted to precision T
    spec::S
    opts::LMOptions{T}
    buf::B              # Vector{Dual{VoxelFitTag,T,P}}: the model output, one per observation
    y::Vector{T}        # the current voxel's data
end

function Base.show(io::IO, c::FitCache{M,A,S,T}) where {M,A,S,T}
    print(io, "FitCache(", c.model, "; free ", freenames(c.spec), ", fixed ", keys(c.spec.fixed),
          ", ", length(c.y), " observations, ", T, ")")
end
# Adding a method to `Base.show` controls how the object prints in the REPL.

"""
    init(model, acq; fixed = (;), free = (), start = (;), precision = Float64,
         maxiter = 100, xtol, ftol, gtol, damping) -> FitCache

Set up a reusable cache for fitting single voxels with [`fit!`](@ref).

- `acq`: the acquisition, a named tuple, e.g. `(TI = [...], TD = [...])`.
- `fixed`, `free`, `start`: override the model's parameter roles (see [`setup`](@ref)).
  Maps (arrays) are indexed by the `voxel` keyword of `fit!`.
- `precision`: `Float64` or `Float32`.
- `maxiter`, `xtol`, `ftol`, `gtol`, `damping`: solver settings ([`LMOptions`](@ref)).

`init` is `CommonSolve.init`, the SciML convention, so it does not clash with other packages.
"""
function CommonSolve.init(model::AbstractModel, acq::NamedTuple; fixed = (;), free = (),
                          start = (;), precision::Type{<:AbstractFloat} = Float64, kwargs...)
    T = precision
    a = _convert_acq(acq, T)
    m = nobs(a)
    spec = setup(model, T; fixed, free, start)    # `; fixed` is short for `; fixed = fixed`
    opts = LMOptions{T}(; kwargs...)               # remaining keywords go to the solver
    return _cache(model, a, spec, opts, m)
end

# A separate function so that P (the number of free parameters) is available as a type
# parameter, to allocate the dual-number buffer with the right type.
_cache(model, acq, spec::FitSpec{F,P,T}, opts, m) where {F,P,T} =
    FitCache(model, acq, spec, opts, Vector{Dual{VoxelFitTag,T,P}}(undef, m), Vector{T}(undef, m))

# Same setup, fresh buffers: one per thread.
_copycache(c::FitCache) = FitCache(c.model, c.acq, c.spec, c.opts, similar(c.buf), similar(c.y))

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
cache = init(SIRFSE(), acq; fixed = (kmf = 15.0,))
r = fit!(cache, y)
r.params.PSR
```
"""
function fit!(c::FitCache, y::AbstractVector; voxel::Integer = 1)
    length(y) == length(c.y) ||
        throw(DimensionMismatch("data have $(length(y)) values, the acquisition has $(length(c.y))"))
    copyto!(c.y, y)                    # into the cache: contiguous and in precision T
    return _fit_voxel(c, c.spec, voxel)
end

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
    # 1. This voxel's fixed values: (kmf = 12.5, R1m = p -> p.R1f, Sm = sm_map[v]).
    fixed = map(x -> _pick(x, v, T), spec.fixed)
    # 2. This voxel's starting point, as an SVector in the order F.
    θ0 = SVector{P,T}(Tuple(map(x -> _pick(x, v, T), spec.start)))
    # 3. Bad inputs mark the voxel invalid instead of producing a meaningless fit.
    if !(all(isfinite, c.y) && all(_finite, values(fixed)) && all(isfinite, θ0))
        return _invalid(spec)
    end
    prob = VoxelProblem{F}(c.model, c.acq, c.y, fixed)
    # 4. A `Free(:scale)` parameter starts at the value that best matches the data's scale.
    if spec.scale > 0
        θ0 = _scale_start(prob, θ0, spec.scale, spec.lower, spec.upper, c.buf)
    end
    # 5. Run LM and attach the names: (PSR = θ₁, R1f = θ₂, ...).
    r = lm_fit(prob, θ0, spec.lower, spec.upper, c.opts, c.buf)
    return VoxelResult(NamedTuple{F}(Tuple(r.θ)), r.sse, r.iterations, r.status)
end

# Starting value of the scale parameter k (S0, M0, ...). The signal is proportional to it, so with
# s = the signal at scale 1 (θ0 holds 1 there), the best fit to y is  scale = (s·y) / (s·s).
# The model is evaluated with plain numbers; storing them in the dual buffer just sets the
# derivative parts to zero, so no extra buffer is needed.
function _scale_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, k, lo, hi, buf) where {F,P,T}
    signal!(buf, prob.model, prob.acq, _params(Val(F), θ0, prob.fixed))
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
