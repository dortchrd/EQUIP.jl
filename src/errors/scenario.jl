# =================================================================================================
# Scenario: a truth, and the fit that is judged against it (what crlb and montecarlo take)
#
#   s = Scenario(QMTSIR, θ; kmf = 12.5)
#       truth    every parameter's true value: θ, as for `simulate` (numbers, or ties)
#       roles    the fit, built from the truth:
#                  estimated parameters as the model estimates them, with the model's bounds,
#                  starting at their true values (`starts = :truth`, the default) or by the
#                  model's own rules (`starts = :model`: `:scale`, `BestOf`, `:data`, as `fit`
#                  starts a real fit);
#                  fixed parameters held at their TRUE values;
#                then the changes given, by `fit`'s rules (a value fixes; `Free(...)` estimates;
#                a start or a bound given wins)
#       wrong    the fixed values that differ from the truth's: the source of any bias
#       weights  the fit's weights, one per observation (`weights = nex`), or nothing
#   crlb(m, s; snr = 200), montecarlo(m, s; noise = :rician, snr = 200)        the scenario under protocol m
#
# Why an object: crlb and montecarlo answer "how well does this fit recover this truth?". Built
# from the truth, the fit is unbiased unless a change makes it wrong: `kmf = 12.5` against a true
# kmf of 5 is an assumption you state, and the scenario names it (`s.wrong`). It holds no
# acquisition, so the same scenario is judged under any protocol, as a protocol search needs.
# The keyword forms, `crlb(m, θ; kmf = 12.5)`, build one inside.
#
# The starts: crlb's noise-free fit always starts at the truth, whatever `starts` says (the
# pseudo-true value is the minimum nearest the truth); montecarlo's fits start where the scenario
# says, so `starts = :model` measures the model's starting rules too. In a sweep over true values
# (`vary`), the starts taken from the truth follow each setting's truth, and the fixed values stay
# as they are: vary changes the truth, never the fit. docs/design.md, "Scenario".
# =================================================================================================

"""
    Scenario(M, θ; starts = :truth, weights, changes...)
    Scenario(m, θ; starts = :truth, weights, changes...)

A truth and the fit judged against it: what [`crlb`](@ref) and [`montecarlo`](@ref) take, with
a protocol (a model with its acquisition): `crlb(m, s; snr = 200)`. Made from a model's type `M`
(or a model `m`: only its type is kept, not its acquisition) and the truth θ, every parameter's
true value (numbers, or ties), as for [`simulate`](@ref): e.g. `nominal(QMTSIR, :SIR_WM_I)`.

The fit is built from the truth:

- estimated: the model's free parameters, with the model's bounds, starting at their true values
  (`starts = :truth`, the default) or by the model's own rules (`starts = :model`: `:scale`,
  `BestOf`, `:data`, as [`fit`](@ref) starts a real fit);
- fixed: the model's other parameters, held at their TRUE values, so the fit is unbiased;
- then the changes, keywords named after the parameters, by `fit`'s rules: a value fixes
  (`kmf = 12.5`; a tie, `R1m = :R1f`), `Free(...)` estimates (`kmf = Free()` starts at the true
  kmf, or with `starts = :model` at the model's value, as `fit` would), and a start or a bound
  given wins (`R1f = Free(start = 0.8)`, `PSR = Free(lower = 0.05)`);
- `weights`: the fit's weights, one per observation, as for `fit` (`weights = nex`, the numbers
  of averages): a weighted fit is judged (crlb's sandwich H⁻¹J₀ᵀWΣWJ₀H⁻¹; montecarlo fits with
  them). Checked against the protocol's number of observations where the scenario meets one.

Fields: `model` (the type), `truth` (every parameter in model order, as given), `roles` (every
parameter's role in the fit, [`Free`](@ref) or [`Fixed`](@ref)), `starts`, `wrong` (the fixed
parameters whose value in the fit, a tie followed at the truth, differs from the truth's, each as
`(fit = ..., truth = ...)`; empty when the fit is unbiased), `attruth` (the estimated
parameters that start at their true value) and `weights` (`nothing`, or a vector).

crlb's noise-free fit always starts at the truth, whatever `starts` says; montecarlo's fits
start where the scenario says. In a sweep over true values (`vary`), the starts taken from the
truth follow each setting's truth; the fixed values stay (the fit does not change).

```julia
θ = nominal(QMTSIR, :SIR_WM_I)                         # kmf = 5 in this tissue
s = Scenario(QMTSIR, θ; kmf = 12.5)                    # the fit assumes 12.5: s.wrong.kmf
crlb(QMTSIR(; SIR_4POINT...), s; snr = 200)            # under the 4-point protocol
crlb(QMTSIR(; SIR_5POINT...), Scenario(QMTSIR, θ; kmf = Free()); snr = 200)
montecarlo(QMTSIR(; SIR_4POINT...), Scenario(QMTSIR, θ; kmf = 12.5, starts = :model); noise = :rician, snr = 200)
```
"""
struct Scenario{M,T<:NamedTuple,R<:NamedTuple,W<:NamedTuple,A<:Tuple,V}
    model::Type{M}        # the model's type, without an acquisition (e.g. QMTSIR)
    truth::T              # every parameter's true value, in model order, as given (ties kept)
    roles::R              # every parameter's role in the fit: Free(start, bounds) or Fixed(value)
    starts::Symbol        # :truth or :model: where estimated parameters start, unless given
    wrong::W              # (name = (fit = value, truth = value), ...): the fixed values that are wrong
    attruth::A            # the estimated parameters that start at their true value
    weights::V            # nothing, or the fit's weights (Vector{Float64}, one per observation)
end

Scenario(M::Type{<:MRIModel}, θ::NamedTuple; starts::Symbol = :truth, weights = nothing,
         changes...) =
    _scenario(Base.typename(M).wrapper, _roles(M), θ, starts, changes, weights)
Scenario(m::MRIModel, θ::NamedTuple; starts::Symbol = :truth, weights = nothing, changes...) =
    _scenario(Base.typename(typeof(m)).wrapper, _ownroles(m), θ, starts, changes, weights)
# `Base.typename(T).wrapper` is the type without its parameters: QMTSIR for a
# QMTSIR{Vector{Float64}}, so the scenario fits a QMTSIR with any kind of acquisition vectors.

# A model's own roles: its type's, where the model declares them for its type (a start that
# depends on the acquisition stays a function, computed for each protocol the scenario meets);
# otherwise the model's own method's (`parameters(m::MyModel)`).
_ownroles(m::MRIModel) = applicable(parameters, typeof(m)) ? parameters(typeof(m)) : parameters(m)

function _scenario(M, own::NamedTuple, θ::NamedTuple, starts::Symbol, changes, weights = nothing)
    starts in (:truth, :model) || throw(ArgumentError("`starts` must be :truth (every " *
        "estimated parameter starts at its true value) or :model (the model's own starting " *
        "rules, as `fit` starts a real fit), got $(repr(starts))"))
    name = nameof(M)
    truth = _complete(name, own, θ)           # checked: every parameter, model order, ties kept
    any(values(map(_hasmap, truth))) && throw(ArgumentError("a scenario is one voxel's truth: " *
        "give numbers, not maps. For maps, simulate and fit them, then `compare`"))
    θt = _resolve_ties(map(_tie, truth))      # the true values, ties followed

    # The fit from the truth, parameter by parameter: estimated as the model estimates them, or
    # held at the true value. `map` over named tuples with the same names in the same order
    # calls `base(own[n], truth[n], θt[n])` for each n, and Julia picks the method by the role.
    base(r::Fixed, t, v) = Fixed(t)
    base(r::Free, t, v) = starts === :truth ? Free(start = float(v), lower = r.lower, upper = r.upper) : r
    roles = map(base, own, truth, θt)

    # The changes, by fit's rules (`_role`, fit/setup.jl): a value fixes; Free(...) estimates,
    # keeping the start and bounds it leaves out. A parameter held here and freed by a change
    # starts at its true value (`starts = :truth`) or where `fit` would start it (the model's
    # value, `starts = :model`).
    given = ()                                # the parameters given a start here
    for (n, x) in pairs(changes)
        haskey(roles, n) || throw(n in (:noise, :averages, :averaging) ? _noise_refused("Scenario", n) :
            ArgumentError(n in _OPTIONS ?
                "`$n` is an option of the fit, not a parameter's role: give it to montecarlo" :
                "`$n` is not a parameter of $name; the parameters are $(join(keys(own), ", "))"))
        _hasmap(x) && throw(ArgumentError("`$n`: a scenario is one voxel: give numbers, not maps"))
        x isa Free && x.start === :scale && _check_scale(name, own, n)
        r = if x isa Free && roles[n] isa Fixed
            x.start !== nothing ? x :
            starts === :truth ? Free(start = float(θt[n]), lower = x.lower, upper = x.upper) :
            _override(own[n], x, n)
        else
            _role(roles[n], x, n)
        end
        x isa Free && x.start !== nothing && (given = (given..., n))
        roles = merge(roles, NamedTuple{(n,)}((r,)))
    end
    # The weights, checked as `fit` checks them (finite, ≥ 0, one positive); their number is
    # checked against a protocol's observations when the scenario meets one (`_check_protocol`).
    w = weights === nothing ? nothing : _checked_weights(weights, length(weights), Float64)
    return _checked_scenario(M, truth, θt, roles, starts, given, w)
end

# The rest in a function of its own (a "function barrier"): after the changes the compiler no
# longer knows the roles' type in the function above (a change can make any parameter Free or
# Fixed); here it does again, so the `map`s below compile to straight-line code, not a lookup
# per parameter (the constructor 2.5× faster; docs/design.md, "Scenario").
function _checked_scenario(M, truth, θt, roles::NamedTuple{K}, starts, given, w) where {K}
    # Checked as `setup` checks a fit's roles: something to estimate, ties that name a parameter
    # that is not a tie, bounds in order.
    isfree = map(r -> r isa Free, roles)
    any(values(isfree)) || throw(ArgumentError("no free parameters: everything is fixed"))
    _check_ties(map(r -> r isa Fixed ? r.value : r, roles))
    inverted = map(r -> r isa Free && something(r.lower, -Inf) > something(r.upper, Inf), roles)
    if any(values(inverted))
        n = first(_names(inverted))
        throw(ArgumentError("`$n`: the lower bound $(roles[n].lower) exceeds the upper bound " *
                            "$(roles[n].upper)"))
    end
    fitvalues = map((r, v) -> r isa Fixed ? r.value : v, roles, θt)   # free ones at the truth
    names = NamedTuple{K}(K)                  # each parameter's name, by name
    attruth = starts === :truth ? _names(map((f, n) -> f && !(n in given), isfree, names)) : ()
    return Scenario(M, truth, roles, starts, _wrong(θt, fitvalues), attruth, w)
end

# The names whose flag is true, in order; the entries of `x` with those names.
_names(flags::NamedTuple{K}) where {K} = Tuple(K[i] for i in eachindex(K) if flags[i])
_subset(flags::NamedTuple, x::NamedTuple) = (w = _names(flags); NamedTuple{w}(map(n -> x[n], w)))

# The fixed values that differ from the truth's, each as (fit = ..., truth = ...): the fit's
# value, its ties followed at the truth, against the true value, the truth's own ties followed.
# `θt` is the truth with its ties followed; `fitvalues` every parameter's value in the fit at
# the truth (the free ones θt's; from the fixed values alone: `merge(θt, fixed)`). A tie can be
# wrong too: R1m = :R1f (1.5 here) for a tissue with R1m = 1.0.
function _wrong(θt::NamedTuple, fitvalues::NamedTuple)
    θf = _resolve_ties(map(_tie, fitvalues))
    return _subset(map(!=, θf, θt), map((f, t) -> (fit = f, truth = t), θf, θt))
end

# The scenario's fit under protocol m, for the true values θt (ties followed): a start that
# depends on the acquisition computed for m, and the starts taken from the truth moved to θt's
# values (in a sweep over true values they follow the truth; the fixed values stay).
function _roles_at(s::Scenario, m::MRIModel, θt::NamedTuple)
    at(n, r::Free) = n in s.attruth ? Free(start = float(θt[n]), lower = r.lower, upper = r.upper) :
                                      _at_acquisition(r, m)
    at(n, r) = r
    return NamedTuple{keys(s.roles)}(map(at, keys(s.roles), values(s.roles)))
end

# The protocol must be a model of the scenario's type, with one observation per weight.
function _check_protocol(m::MRIModel, s::Scenario)
    m isa s.model || throw(ArgumentError("the scenario is for $(nameof(s.model)); the model is " *
                                         "a $(nameof(typeof(m)))"))
    s.weights === nothing || length(s.weights) == nobs(m) || throw(DimensionMismatch("the " *
        "scenario's fit has $(length(s.weights)) weights; the protocol has $(nobs(m)) observations"))
    return true
end

# Several scenarios side by side, by name: `(fixed = Scenario(...), free = Scenario(...))`, at
# least one. One more axis of a sweep (crlb, montecarlo).
const _Scenarios = NamedTuple{N,<:Tuple{Scenario,Vararg{Scenario}}} where {N}
_scenarios(s::Scenario) = (s,)
_scenarios(ss::NamedTuple) = values(ss)

# -------------------------------------------------------------------------------------------------
# Display: one row per parameter, its true value and its role in the fit; wrong values marked
# -------------------------------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", s::Scenario)
    cell(x) = x isa Real ? repr(x; context = :compact => true) : _showvalue(x)
    setting(r::Free) = r.start isa Function ? "from the protocol" : cell(r.start)
    setting(r::Fixed) = _istie(r.value) ? cell(r.value) * " (tie)" : cell(r.value)
    bounds(r::Free) = "[" * cell(something(r.lower, -Inf)) * ", " * cell(something(r.upper, Inf)) * "]"
    bounds(r::Fixed) = ""
    print(io, "Scenario for ", nameof(s.model), ": a truth and the fit judged against it")
    print(io, "\nstarts: ", s.starts === :truth ? "the truth" :
          "the model's rules, as `fit` starts a real fit (montecarlo; crlb's noise-free fit " *
          "starts at the truth)")
    rows = [["name", "true", "role", "start/value", "bounds", ""]]
    for (n, r) in pairs(s.roles)
        push!(rows, [string(n), cell(s.truth[n]), r isa Free ? "free" : "fixed", setting(r),
                     bounds(r), haskey(s.wrong, n) ? "← wrong" : ""])
    end
    widths = [maximum(row -> textwidth(row[j]), rows) for j in 1:6]
    for row in rows
        print(io, "\n", rstrip("  " * join(rpad.(row, widths), "  ")))
    end
    print(io, isempty(s.wrong) ? "\nnothing fixed is wrong: the fit is unbiased" :
          "\nwrong: " * join(("$n = $(cell(w.fit)) (true $(cell(w.truth)))" for (n, w) in pairs(s.wrong)), ", ") *
          ": the fit is biased")
    s.weights === nothing || print(io, "\nweighted fit: ", _weightstext(s.weights))
end

# The weights, in words: "weights 1 to 8 (7 observations)".
function _weightstext(w)
    f(x) = repr(round(x; sigdigits = 4); context = :compact => true)
    span = all(==(w[1]), w) ? f(w[1]) : "$(f(minimum(w))) to $(f(maximum(w)))"
    return "weights $span ($(length(w)) observations)"
end

Base.show(io::IO, s::Scenario) = print(io, "Scenario(", nameof(s.model), "; ",
    isempty(s.wrong) ? "unbiased" : "wrong: " * join(keys(s.wrong), ", "), ")")
