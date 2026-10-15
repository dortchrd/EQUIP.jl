# Fit setup and the reusable cache (`init` / `fit!`).
#
# `init` resolves the parameter roles once (model defaults + the user's `fixed`, `free` and
# `start`), converts the protocol, and allocates the buffers a voxel fit needs. `fit!` then
# fits any number of voxels without allocating. `fit_maps` makes one cache per thread.

"""
    FitSpec

Resolved parameter roles for one fit: names of the estimated parameters `F` (a type parameter,
so they are known to the compiler), their bounds, explicit starting values (numbers or maps),
whether the model's `initial_guess` is needed, and the fixed values (numbers, maps or ties).
"""
struct FitSpec{F,P,T,S,FX}
    lower::SVector{P,T}
    upper::SVector{P,T}
    start::S          # NamedTuple: starting values (numbers or per-voxel maps), subset of F
    needs_auto::Bool  # some estimated parameter has no explicit start
    fixed::FX         # NamedTuple: numbers (T), per-voxel maps, or ties (functions)
end

freenames(::FitSpec{F}) where {F} = F
nfree(::FitSpec{F,P}) where {F,P} = P

_torole(x::Estimate) = throw(ArgumentError("use `free` (not `fixed`) to estimate a parameter"))
_torole(x::Fixed) = x
_torole(x) = Fixed(x)
_freerole(old::Estimate) = old
_freerole(old::Fixed{<:Real}) = Estimate(Float64(old.value))
_freerole(old::Fixed) = Estimate(:auto)

_fixedvalue(x::Real, ::Type{T}) where {T} = T(x)
_fixedvalue(x, ::Type{T}) where {T} = x            # maps and ties

function _check_names(kind, given, known)
    for n in given
        n in known || throw(ArgumentError("unknown parameter `$n` in `$kind`; parameters are $(known)"))
    end
end

"""
    setup(model, T; fixed = (;), free = (), start = (;)) -> FitSpec

Combine the model's default roles with the user's choices:

- `fixed`: named tuple; a number, a per-voxel map, a tie function, or `Fixed(...)`.
- `free`: names to estimate (`:R1m`) or `name => Estimate(...)` pairs to also set the start
  and bounds.
- `start`: named tuple of starting values (numbers or per-voxel maps) for estimated parameters.
"""
function setup(model::AbstractModel, ::Type{T}=Float64; fixed=(;), free=(), start=(;)) where {T}
    roles = parameters(model)
    known = keys(roles)
    fixed isa NamedTuple || throw(ArgumentError("`fixed` must be a named tuple, e.g. (kmf = 14.5,)"))
    _check_names("fixed", keys(fixed), known)
    roles = merge(roles, map(_torole, fixed))
    for f in (free isa Union{Symbol,AbstractString,Pair} ? (free,) : free)
        # a name, `name => Estimate(...)`, or `(name, Estimate(...))` (how Python passes pairs)
        name, role = f isa Union{Pair,Tuple{Any,Estimate}} ? (Symbol(first(f)), last(f)) : (Symbol(f), nothing)
        _check_names("free", (name,), known)
        role = role === nothing ? _freerole(roles[name]) : role
        role isa Estimate || throw(ArgumentError("`free = ($name => ...)` needs an Estimate"))
        roles = merge(roles, NamedTuple{(name,)}((role,)))
    end
    est = Tuple(n for n in keys(roles) if roles[n] isa Estimate)
    fix = Tuple(n for n in keys(roles) if roles[n] isa Fixed)
    isempty(est) && throw(ArgumentError("no parameters to estimate"))
    start isa NamedTuple || throw(ArgumentError("`start` must be a named tuple"))
    _check_names("start", keys(start), est)
    P = length(est)
    # explicit starts: numeric Estimate inits, overridden by `start`
    numeric = Tuple(n for n in est if roles[n].init isa Real)
    starts = merge(NamedTuple{numeric}(map(n -> T(roles[n].init), numeric)),
                   map(v -> v isa Real ? T(v) : v, start))
    needs_auto = any(n -> !haskey(starts, n), est)
    lower = SVector{P,T}(map(n -> T(roles[n].lower), est))
    upper = SVector{P,T}(map(n -> T(roles[n].upper), est))
    all(lower .<= upper) || throw(ArgumentError("lower bounds exceed upper bounds"))
    fixedvals = NamedTuple{fix}(map(n -> _fixedvalue(roles[n].value, T), fix))
    return FitSpec{est,P,T,typeof(starts),typeof(fixedvals)}(lower, upper, starts, needs_auto, fixedvals)
end

# Per-voxel values: maps are indexed by the voxel, numbers and ties pass through.
@inline _pick(x::AbstractArray, v, ::Type{T}) where {T} = @inbounds T(x[v])
@inline _pick(x::Real, v, ::Type{T}) where {T} = T(x)
@inline _pick(x, v, ::Type{T}) where {T} = x
@inline _finite(x::Real) = isfinite(x)
@inline _finite(x) = true
@inline _allfinite_nt(nt::NamedTuple) = all(_finite, values(nt))

# ---------------------------------------------------------------------------------------------

"""
    VoxelResult

Result of one voxel fit: `params` (named tuple of the estimated parameters), `sse` (weighted
residual sum of squares), `JtJ` (Gauss–Newton information matrix at the solution),
`iterations`, `status` ([`FitStatus`](@ref)).
"""
struct VoxelResult{NT,P,T,L}
    params::NT
    sse::T
    JtJ::SMatrix{P,P,T,L}
    iterations::Int32
    status::FitStatus
end

isconverged(r::VoxelResult) = isconverged(r.status)

"""
    standard_errors(r::VoxelResult, nobs) -> NamedTuple

Asymptotic standard errors √diag(σ̂²(JᵀWJ)⁻¹), σ̂² = SSE/(nobs - P), for the estimated
parameters.
"""
function standard_errors(r::VoxelResult{NT,P,T}, nobs::Integer) where {NT,P,T}
    lm = LMResult(SVector{P,T}(Tuple(r.params)), r.sse, r.JtJ, r.iterations, r.status)
    return NamedTuple{keys(r.params)}(Tuple(standard_errors(lm, nobs)))
end

"""
    FitCache

Reusable state for fitting voxels of one model and protocol: resolved roles, solver options and
work buffers. Create with [`init`](@ref), use with [`fit!`](@ref).
"""
struct FitCache{M,A,SP<:FitSpec,T,B,W}
    model::M
    acq::A
    spec::SP
    opts::LMOptions{T}
    buf::B              # Vector{Dual{VoxelFitTag,T,P}}, one per observation
    ybuf::Vector{T}
    w::W
end

Base.show(io::IO, c::FitCache{M,A,SP,T}) where {M,A,SP,T} =
    print(io, "FitCache(", c.model, "; estimate ", freenames(c.spec), ", fixed ", keys(c.spec.fixed),
          ", ", nobs(c.acq), " observations, ", T, ")")

"""
    init(model, acq; fixed = (;), free = (), start = (;), weights = nothing,
         precision = Float64, maxiter = 100, xtol, ftol, gtol, damping = 1e-3) -> FitCache

Set up a reusable cache for fitting voxels: resolves the parameter roles (see `setup`),
converts the protocol `acq` (a named tuple, or a vector for one-setting models) and allocates
work buffers. This is `CommonSolve.init`.
"""
function CommonSolve.init(model::AbstractModel, acq; fixed=(;), free=(), start=(;),
                          weights=nothing, precision::Type{<:AbstractFloat}=Float64,
                          maxiter::Integer=100, xtol=nothing, ftol=nothing, gtol=nothing,
                          damping=1e-3)
    T = precision
    a = prepare_protocol(model, acq, T)
    spec = setup(model, T; fixed, free, start)
    m = nobs(a)
    w = weights === nothing ? nothing : Vector{T}(weights)
    w === nothing || length(w) == m || throw(DimensionMismatch("weights must have length $m"))
    opts = LMOptions{T}(; maxiter, xtol, ftol, gtol, damping)
    return _cache(model, a, spec, opts, w, m)
end

_cache(model, a, spec::FitSpec{F,P,T}, opts, w, m) where {F,P,T} =
    FitCache(model, a, spec, opts, Vector{Dual{VoxelFitTag,T,P}}(undef, m), Vector{T}(undef, m), w)

# Same setup, fresh buffers (one per thread).
_copycache(c::FitCache) = FitCache(c.model, c.acq, c.spec, c.opts, similar(c.buf), similar(c.ybuf), c.w)

"""
    fit!(cache, y; voxel = 1) -> VoxelResult

Fit one voxel's data `y` (one value per observation). `voxel` is the voxel's index into any
per-voxel maps given as fixed values or starting values. Does not allocate.
"""
function fit!(c::FitCache, y::AbstractVector; voxel::Integer=1)
    length(y) == length(c.ybuf) ||
        throw(DimensionMismatch("data have $(length(y)) values, protocol has $(length(c.ybuf))"))
    copyto!(c.ybuf, y)
    return _fit_voxel(c, c.ybuf, voxel)
end

@inline _allfinite_data(y) = all(isfinite, y)

function _invalid(::FitSpec{F,P,T}) where {F,P,T}
    VoxelResult(NamedTuple{F}(ntuple(_ -> T(NaN), Val(P))), T(NaN),
                SMatrix{P,P,T}(ntuple(_ -> T(NaN), Val(P * P))), Int32(0), InvalidData)
end

@inline function _sanitize(θ::SVector{P}, lo::SVector{P,T}, hi::SVector{P,T}) where {P,T}
    # Replace non-finite starting values (e.g. an all-zero voxel) by a point inside the bounds.
    map(θ, lo, hi) do t, l, h
        tt = T(t)
        isfinite(tt) ? tt : (isfinite(l) && isfinite(h) ? (l + h) / 2 :
                             isfinite(l) ? l + one(T) : isfinite(h) ? h - one(T) : one(T))
    end
end

@inline _select(::Val{F}, nt, ::Type{T}) where {F,T} = SVector(map(T, Tuple(NamedTuple{F}(nt))))

@inline function _start(c::FitCache, spec::FitSpec{F,P,T}, y, v, fixedv) where {F,P,T}
    explicit = map(s -> _pick(s, v, T), spec.start)
    θ = if spec.needs_auto
        guess = initial_guess(c.model, c.acq, y, _drop_ties(fixedv))
        _select(Val(F), merge(guess, explicit), T)
    else
        _select(Val(F), explicit, T)
    end
    return _sanitize(θ, spec.lower, spec.upper)
end

# initial_guess sees the voxel's fixed numbers (e.g. B1); ties refer to estimated parameters,
# which are not known yet, so they are left out.
@inline _drop_ties(nt::NamedTuple) = _drop_ties(keys(nt), values(nt))
@inline _drop_ties(::Tuple{}, ::Tuple{}) = (;)
@inline _drop_ties(k::Tuple, v::Tuple) = merge(
    first(v) isa Function ? (;) : NamedTuple{(first(k),)}((first(v),)), _drop_ties(Base.tail(k), Base.tail(v)))

function _fit_voxel(c::FitCache, y, v)
    spec = c.spec
    _allfinite_data(y) || return _invalid(spec)
    return _fit_voxel(c, spec, y, v)
end

@noinline function _fit_voxel(c::FitCache, spec::FitSpec{F,P,T}, y, v) where {F,P,T}
    fixedv = map(x -> _pick(x, v, T), spec.fixed)
    _allfinite_nt(fixedv) || return _invalid(spec)
    θ0 = _start(c, spec, y, v, fixedv)
    prob = VoxelProblem{F}(c.model, c.acq, y, c.w, fixedv)
    r = lm_fit(prob, θ0, spec.lower, spec.upper, c.opts, c.buf)
    return VoxelResult(NamedTuple{F}(Tuple(r.θ)), r.sse, r.JtJ, r.iterations, r.status)
end
