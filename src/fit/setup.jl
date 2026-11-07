# =================================================================================================
# Fit setup: from "what the user asked for" to resolved parameter roles
#
#   parameters(m)  +  keywords named after the parameters (kmf = 15.0, R1f = Free(start = 0.8))
#        │
#        ▼  setup        merge the user's choices into the model's default roles, then split the
#   FitSpec              parameters into Free (estimated) and Fixed. Done ONCE per fit call.
#
# What happens next, file by file:
#   cache.jl    `init` wraps the FitSpec in a FitCache (with the model in the working precision,
#               the solver settings and work buffers)
#   voxel.jl    fitting one voxel with it: `fit!(cache, y)`, and `fit(m, y)` for a vector
#   starts.jl   each voxel's starting point (:data, :loglinear, BestOf, :scale)
#   levmar.jl   the solver (Levenberg–Marquardt) for one voxel
#   dataset.jl  `fit(m, data)`: every voxel of a dataset, spread over threads; `refit`
#   result.jl   FitResult: what `fit` and `fit!` return
# =================================================================================================

"""
    FitSpec{F,P,T}

The resolved parameter roles of one fit (made by `setup`):

- `F`: names of the free parameters, e.g. `(:PSR, :R1f, :Sf, :M0f)`, in model order
- `P = length(F)`, `T`: the working float type
- `lower`, `upper`: bounds of the free parameters (`SVector{P,T}`)
- `start`: starting values of the free parameters (numbers or per-voxel maps), names `F`
- `scale`: position in `F` of the `Free(start = :scale)` parameter, or 0 if there is none
- `fixed`: the fixed parameters (numbers, per-voxel maps or ties), by name
- `candidates`, `candindex`: the [`BestOf`](@ref) values to try for free parameter number
  `candindex` (those within its bounds); `candindex = 0` means none
- `fromdata`: for each free parameter, whether it starts at `:data` (the model's
  [`datastart`](@ref))
- `fromlog`: for each free parameter, whether it starts at `:loglinear` (the weighted fit of
  ln y with ln S, starts.jl)
- `weights`: `nothing` (unweighted), or one weight per observation in precision `T`: the fit
  minimizes ½Σᵢwᵢrᵢ²

`F` and `P` are type parameters rather than fields: the compiler then knows the names and the
number of free parameters and generates code specialized for them (fixed-size vectors and
compile-time named tuples instead of name lookups at run time).
"""
struct FitSpec{F,P,T,S,X,W}
    lower::SVector{P,T}
    upper::SVector{P,T}
    start::S        # NamedTuple with names F (a :scale parameter holds 1 here)
    scale::Int      # which free parameter starts at the least-squares scale (0 = none)
    fixed::X        # NamedTuple with the fixed parameters' names
    candidates::Vector{T}       # BestOf values to try for one free parameter (empty = none)
    candindex::Int              # which free parameter they are for (0 = none)
    fromdata::NTuple{P,Bool}    # true for free parameters that start at :data
    fromlog::NTuple{P,Bool}     # true for free parameters that start at :loglinear
    weights::W                  # nothing, or Vector{T}: one weight per observation
end

freenames(::FitSpec{F}) where {F} = F       # read the names back out of the type

# --- How the keywords become roles ---------------------------------------------------------------

# Keywords of `fit` and `init` that are options, not parameters. A model's parameter cannot have
# one of these names: the keyword would be ambiguous.
const _OPTIONS = (:mask, :precision, :solver)
# `weights` cannot be a parameter's name either, but it is part of what a fit minimizes, not an
# option of how: a scenario holds it (errors/scenario.jl), where the options are refused.
const _RESERVED = (_OPTIONS..., :weights)

# The observations' weights, checked once: one per observation, finite, ≥ 0 (0 leaves an
# observation out), at least one positive; in the working precision. `nothing`: unweighted.
_checked_weights(::Nothing, N, ::Type{T}) where {T} = nothing
function _checked_weights(w, N, ::Type{T}) where {T}
    w isa AbstractVector{<:Real} || throw(ArgumentError("`weights` is one weight per " *
        "observation, a vector of $N numbers (e.g. the numbers of averages), got $(repr(w))"))
    length(w) == N ||
        throw(DimensionMismatch("`weights` has $(length(w)) values; the model has $N observations"))
    all(x -> isfinite(x) && x >= 0, w) ||
        throw(ArgumentError("`weights` must be finite and ≥ 0 (0 leaves an observation out)"))
    any(>(0), w) || throw(ArgumentError("`weights`: at least one must be positive"))
    return Vector{T}(w)
end

# A keyword `name = x` turns the parameter's default role into the fit's role:
#   Free(...)       estimated; whatever it leaves out (start, bounds) keeps the default
#   Fixed(...)      as given
#   anything else   fixed at that value: a number, a per-voxel map, or a tie (`R1m = :R1f`, or a
#                   function of the other parameters)
_role(default, x::Free, name) = _override(default, x, name)
_role(default, x::Fixed, name) = x
_role(default, x, name) = Fixed(x)

_keep(new, default) = new === nothing ? default : new
# Over a default Free: the new start and bounds where given, the default's elsewhere.
_override(d::Free, x::Free, name) = Free(start = _keep(x.start, d.start),
                                         lower = _keep(x.lower, d.lower), upper = _keep(x.upper, d.upper))
# A parameter fixed by default, now estimated: it starts at its fixed value unless given a start;
# no bounds unless given.
function _override(d::Fixed, x::Free, name)
    x.start !== nothing && return x
    d.value isa Real ||
        throw(ArgumentError("`$name` is fixed by default to $(repr(d.value)); give it a start: $name = Free(start = ...)"))
    return Free(start = d.value, lower = x.lower, upper = x.upper)
end

# `:scale` assumes the signal is proportional to the parameter (true for M0, not for T1), so only
# the model can declare it. A fit can turn it off (M0 = Free(start = 1000.0)) but not move it.
# `M` is the model's name, `own` its own roles.
function _check_scale(M::Symbol, own::NamedTuple, n)
    s = _scalename(own)
    s === n && return nothing
    throw(ArgumentError(s === nothing ? "`$n = Free(start = :scale)`: $M has no scale parameter" :
        "`$n = Free(start = :scale)`: :scale is for the parameter the signal is proportional to ($s in $M)"))
end

# Numbers are converted to the working precision; a tie by name becomes a `_Tie` (see
# models/interface.jl); maps and function ties are kept as they are.
_value(x::Real, ::Type{T}) where {T} = T(x)
_value(x::Symbol, ::Type{T}) where {T} = _Tie(x)
_value(x, ::Type{T}) where {T} = x

# The roles with the keywords applied, one parameter each: what `setup` does first, and what
# `parameters(m; changes...)` returns. `merge(a, b)` replaces a's entry by b's with the same name
# and keeps a's order: (…, kmf = Fixed(12.5), …) becomes (…, kmf = Fixed(15.0), …). A start that
# depends on the acquisition, written as a function of the model (`ADC = Free(start = m -> 1 /
# mean(m.b))`), is then computed, once, for this model: the model's own or one given here.
# `unknown(name)` makes the error for a keyword that is not a parameter.
function _apply_roles(model::MRIModel, roles::NamedTuple, changes, unknown)
    for (n, x) in pairs(changes)
        haskey(roles, n) || throw(unknown(n))
        x isa Free && x.start === :scale && _check_scale(nameof(typeof(model)), parameters(model), n)
        roles = merge(roles, NamedTuple{(n,)}((_role(roles[n], x, n),)))
    end
    return map(r -> _at_acquisition(r, model), roles)
end

"""
    setup(model, T = Float64; params...) -> FitSpec

Combine the model's default roles with the fit's keywords, and split the parameters into free
and fixed. Each keyword is named after a parameter of the model:

- a value fixes it: a number, a per-voxel map, or a tie to another parameter, by name
  (`R1m = :R1f`) or as a function (`R1m = p -> 0.8 * p.R1f`);
- `Free(...)` estimates it, with the start (a number, a map, `:data` or `BestOf(values)`) and
  bounds given, and the model's defaults for anything left out: `kmf = Free()`,
  `R1f = Free(start = 0.8)`, `Sf = Free(lower = -1.5)`.

`weights`: one weight per observation (the fit then minimizes ½Σwᵢrᵢ²), or `nothing`.
"""
function setup(model::MRIModel, ::Type{T} = Float64; weights = nothing, params...) where {T}
    # 1. The model's defaults, e.g. (PSR = Free(start = 0.1, ...), ..., kmf = Fixed(12.5), ...).
    roles = parameters(model)
    names = keys(roles)
    for n in names
        n in _RESERVED && throw(ArgumentError("$(nameof(typeof(model))) has a parameter named `$n`, " *
            "which is also an option of fit ($(join(_RESERVED, ", "))); rename the parameter"))
    end

    # 2. Apply the keywords, one parameter each (`_apply_roles`, above).
    roles = _apply_roles(model, roles, params, n ->
        n in (:noise, :averages, :averaging) ? _noise_refused("fit", n) :
        ArgumentError(
        n === :mask ? "`mask` is for datasets: fit(m, data; mask) with `data` holding many voxels" :
        n in fieldnames(LM) ? "`$n` is a solver setting: solver = LM($n = ...)" :
        "`$n` is neither a parameter of $(nameof(typeof(model))) ($(join(names, ", "))) " *
        "nor an option ($(join(_RESERVED, ", ")))"))
    return _spec(model, roles, T, weights)
end

# Steps 3–5 of `setup`, from a complete set of roles (every parameter, in the model's order, each
# start computed for the acquisition): what `setup` makes from the keywords, and what crlb makes
# from a scenario's roles (errors/scenario.jl) without merging them in one by one.
function _spec(model::MRIModel, roles::NamedTuple, ::Type{T}, weights = nothing) where {T}
    names = keys(roles)
    w = _checked_weights(weights, nobs(model), T)
    # 3. Split by role, keeping the model's order.
    F = Tuple(n for n in names if roles[n] isa Free)     # e.g. (:PSR, :R1f, :Sf, :M0f)
    X = Tuple(n for n in names if roles[n] isa Fixed)    # e.g. (:kmf, :R1m, :Sm)
    isempty(F) && throw(ArgumentError("no free parameters: everything is fixed"))
    P = length(F)
    # A tie by name (R1m = :R1f) must name a parameter that is not itself a tie.
    _check_ties(map(r -> r isa Fixed ? r.value : r, roles))

    # 4. Bounds as fixed-size vectors (none = ±Inf), and the fixed values.
    lower = SVector{P,T}(map(n -> T(something(roles[n].lower, -Inf)), F))
    upper = SVector{P,T}(map(n -> T(something(roles[n].upper, Inf)), F))
    all(lower .<= upper) || throw(ArgumentError("a lower bound exceeds its upper bound"))
    fixedvals = NamedTuple{X}(map(n -> _value(roles[n].value, T), X))

    # 5. Starting values. Each Free role's start says where it comes from (see `Free`): a number
    #    or a map, `:scale`, `:data`, `:loglinear`, or `BestOf(values)`.
    M = nameof(typeof(model))
    start(n) = roles[n].start
    for n in F
        start(n) === nothing && throw(ArgumentError("`$n` has no starting value: give one in " *
            "the call, `$n = Free(start = ...)`, or in the model's parameters ($M)"))
    end
    #    `:scale`: at most one, the model's own (checked in step 2).
    count(n -> start(n) === :scale, F) <= 1 || throw(ArgumentError("only one parameter can start as :scale"))
    scale = something(findfirst(n -> start(n) === :scale, F), 0)    # findfirst gives `nothing` if none
    #    `BestOf`: one parameter for now; its values within the bounds.
    best = findall(n -> start(n) isa BestOf, F)
    length(best) <= 1 || throw(ArgumentError("only one parameter can start from BestOf for now " *
        "(several, trying every combination, are on the roadmap): $(join(F[best], ", "))"))
    candindex = isempty(best) ? 0 : only(best)
    candidates = candindex == 0 ? T[] : _candidates(F[candindex], start(F[candindex]), lower[candindex], upper[candindex], T)
    #    `:data`: the model's `datastart` must estimate it. It is asked once here, on placeholder
    #    data (ones), to see which parameters it covers.
    fromdata = map(n -> start(n) === :data, F)
    if any(fromdata)
        d = datastart(model, ones(T, nobs(model)), map(x -> _pick(x, 1, T), fixedvals))
        for n in F
            start(n) === :data || continue
            d !== nothing && haskey(d, n) ||
                throw(ArgumentError("`$n` starts at :data, but $M has no data start for it (see `datastart`)"))
        end
    end
    #    `:loglinear`: the fit's own log-linear start (starts.jl); nothing to check here.
    fromlog = map(n -> start(n) === :loglinear, F)
    #    The values `_fit_voxel` starts from. `:scale` and `:data` hold the placeholder 1, replaced
    #    voxel by voxel; `:loglinear` 0, where its Gauss–Newton steps begin (and where the
    #    parameter stays if the data cannot determine it); a BestOf parameter its middle value,
    #    used if no value can be scored.
    startvalue(n) = start(n) === :scale || start(n) === :data ? one(T) :
                    start(n) === :loglinear ? zero(T) :
                    start(n) isa BestOf ? sort(candidates)[cld(length(candidates), 2)] : _value(start(n), T)
    starts = NamedTuple{F}(map(startvalue, F))
    #    `:loglinear` fits ln S, so the signal where it starts (those parameters at 0, the rest at
    #    their starts and fixed values, the first voxel's of any map) must be positive and finite
    #    in every observation. It is not for a time constant at 0 (T2 = 0 is no signal: the start
    #    stayed at T2's lower bound, and the fit stopped there and reported convergence, measured
    #    2026-11-07) or for a signal that changes sign (a real-valued inversion recovery).
    if any(fromlog)
        s0 = zeros(T, nobs(model))
        signal!(s0, model, _resolve_ties(merge(map(x -> _pick(x, 1, T), starts), map(x -> _pick(x, 1, T), fixedvals))))
        if !all(x -> isfinite(x) && x > 0, s0)
            lognames = join((n for n in F if start(n) === :loglinear), "`, `")
            throw(ArgumentError("`$lognames` start at :loglinear, but $M's signal there (each at 0, " *
                "the other parameters at their starts) is not positive and finite in every " *
                "observation, so its logarithm is undefined. :loglinear is for parameters for which " *
                "0 is a valid value in a signal that stays positive: diffusivities and rates (R2, " *
                "not T2), magnitude data. Give a number or :data instead."))
        end
    end

    # The names F and their number P go into the type: FitSpec{(:PSR, :R1f, :Sf, :M0f), 4, ...}.
    return FitSpec{F,P,T,typeof(starts),typeof(fixedvals),typeof(w)}(lower, upper, starts, scale,
                                                    fixedvals, candidates, candindex, fromdata, fromlog, w)
end

# The BestOf values of parameter `name` within its bounds, in precision T; an error if none is.
function _candidates(name, b::BestOf, lo, hi, ::Type{T}) where {T}
    c = T[T(x) for x in b.values if lo <= x <= hi]
    isempty(c) && throw(ArgumentError("`$name`: no BestOf value within its bounds [$lo, $hi]"))
    return c
end
