# =================================================================================================
# Sweeps: crlb or montecarlo over several noise levels and true values
#
#   crlb(m, s; snr = [100, 200], vary = :kmf => 5:2.5:20)        14 settings, one crlb each
#   montecarlo(m, s; snr = 200, vary = :kmf => 5:2.5:20)          7 settings, one run each
#   crlb(m, s; snr = 200, vary = [:kmf => 5:5:20, :PSR => [0.1, 0.2]])   8 settings
#   crlb(m, (fixed = s1, free = s2); snr = [100, 200])            4 settings: fits side by side
#
# The settings are every combination of the values given, the first keyword varying slowest
# (like nested loops written in that order), and the scenarios side by side fastest, so a table
# shows the fits of one case next to each other. `vary` changes the TRUTH, never the fit (the
# scenario's): with the fit fixing kmf at 12.5 and the true kmf swept from 5 to 20, the result
# is the bias of that assumption against the true value (Dortch et al., MRM 2018;80:1824,
# Fig. 2B). Varying a fitted parameter instead (PSR = 0.05:0.05:0.2) gives the accuracy over a
# tissue range; starts taken from the truth follow it (montecarlo).
# =================================================================================================

"""
    Sweep

What [`crlb`](@ref) and [`montecarlo`](@ref) return for several settings (`snr = [...]`,
`vary = :kmf => 5:2.5:20`):

- `settings`: one named tuple per setting, e.g. `(snr = 100, kmf = 5.0)`
- `results`: the [`CRLBResult`](@ref) or [`MonteCarloResult`](@ref) of each, in the same order;
  `s[i]` is `s.results[i]`, `length(s)` the number of settings, and `for r in s` loops over them
- `rows`: the table, one named tuple per setting and free parameter: the setting's values, then
  `parameter`, `truth` and the numbers (crlb: `noisefree`, the noise-free fit's value, `bias`,
  ...; montecarlo: measured, then `crlb_...` predicted), e.g. `DataFrame(s.rows)` with
  DataFrames.jl. With scenarios side by side, the setting's `fit` column holds the scenario's
  name.
"""
struct Sweep{R}
    settings::Vector{NamedTuple}
    results::Vector{R}
    rows::Vector{NamedTuple}
end
Base.length(s::Sweep) = length(s.results)
Base.getindex(s::Sweep, i) = s.results[i]
Base.iterate(s::Sweep, state...) = iterate(s.results, state...)
Base.eltype(::Type{Sweep{R}}) where {R} = R

_issweep(snr, vary, fits = (;)) = snr isa AbstractVector || !isempty(vary) || !isempty(fits)

# `vary` as a named tuple of axes, from one pair (`:kmf => 5:2.5:20`), several (a vector or tuple
# of pairs: `[:kmf => 5:5:20, :PSR => [0.1, 0.2]]`), or a named tuple (`(kmf = 5:5:20, PSR =
# [0.1, 0.2])`). `(; pairs...)` makes a named tuple from pairs of a name and a value.
_varyaxes(v::NamedTuple) = v
_varyaxes(v::Pair{Symbol}) = NamedTuple{(v.first,)}((v.second,))
function _varyaxes(v::Union{AbstractVector,Tuple})
    all(p -> p isa Pair{Symbol}, v) || throw(ArgumentError(_VARY_FORMS))
    return (; v...)
end
_varyaxes(v) = throw(ArgumentError(_VARY_FORMS))
const _VARY_FORMS = "`vary`: a parameter's name and its true values, `vary = :kmf => 5:2.5:20`, " *
    "or several, `vary = [:kmf => 5:5:20, :PSR => [0.1, 0.2]]`"

# Every combination of the noise levels and the varied values, the first keyword slowest.
# `Iterators.product` varies its FIRST argument fastest, so the axes go in reversed and each
# combination is reversed back.
function _sweep_settings(m::_ModelOrType, snr, vary::NamedTuple, fits::NamedTuple = (;))
    roles = parameters(m)
    for (k, v) in pairs(vary)
        k === :fit && throw(ArgumentError("`vary = :fit => ...`: `fit` names the fits in a " *
            "sweep; for several fits, give scenarios side by side: `crlb(m, (a = s1, b = s2); ...)`"))
        M = m isa Type ? m : typeof(m)
        haskey(roles, k) || k in fromderived(M) || throw(ArgumentError("`vary = :$k => ...`: " *
            "not a parameter of $(_modelname(m)); the parameters are $(keys(roles))" *
            (isempty(fromderived(M)) ? "" :
             ", and the derived quantities it takes, $(join(fromderived(M), ", "))")))
        v isa AbstractVector && !isempty(v) || throw(ArgumentError("`vary = :$k => ...`: give " *
            "the values as a vector or a range, e.g. `:$k => [5.0, 10.0]` or `:$k => 5:2.5:20`"))
    end
    ax = map(collect, vary)                    # the axes: the values of each keyword
    snr isa AbstractVector && (ax = merge((snr = collect(snr),), ax))   # the noise level first
    isempty(fits) || (ax = merge(ax, (fit = collect(keys(fits)),)))     # the scenarios last
    names = keys(ax)
    return NamedTuple[NamedTuple{names}(reverse(c))
                      for c in vec(collect(Iterators.product(reverse(values(ax))...)))]
end

# A setting split into its parts: the noise level (or the one given for every setting), the
# scenario (by name, side by side; or the one for every setting), and the true values it
# changes: the fitted model's truth, or with data from another model (`simulate_with = m2 =>
# θ2`) the data's truth θ2, while `truth` stays the reference the fit is judged against.
_snr_of(setting, snr) = haskey(setting, :snr) ? setting.snr : snr
_scenario_of(setting, s) = haskey(setting, :fit) ? s[setting.fit] : s
_varied(setting) = Base.structdiff(setting, NamedTuple{(:snr, :fit)})
_datamodel(m, ::Nothing) = m
_datamodel(m, s::Pair) = s.first
_truth_of(setting, s::Scenario, ::Pair) = s.truth
function _truth_of(setting, s::Scenario, ::Nothing)
    v = _varied(setting)
    all(n -> haskey(s.truth, n), keys(v)) && return merge(s.truth, v)   # parameters varied
    # A derived quantity varied (`vary = :FA => ...`): each setting's truth made again from the
    # truth as given (`s.given`, e.g. MD, FA and the direction), with the varied values in place,
    # so that what stays fixed is what the truth was given by: with MD and FA given, MD.
    for n in keys(v)
        haskey(s.truth, n) || haskey(s.given, n) || throw(ArgumentError("`vary = :$n => ...`: " *
            "$n is a derived quantity, and the truth was given by its parameters: give it by " *
            "its derived quantities, e.g. `(S0 = ..., MD = ..., FA = ..., direction = ...)` for " *
            "DTI, so that what stays fixed while $n varies is what you gave"))
    end
    return _fromderived(s.model, merge(s.given, v), keys(s.truth))
end
_data_of(setting, ::Nothing) = nothing
_data_of(setting, s::Pair) = s.first => merge(s.second, _varied(setting))   # derived: `_datasource`

# The fixed values of a sweep's fits, one line per scenario side by side, or one for all; with
# the truth where it differs, unless the truth varies.
function _print_sweep_fixed(io::IO, s::Sweep)
    θt(r) = isempty(_varied(s.settings[1])) ? resolve(r.model, r.truth) : nothing
    if haskey(s.settings[1], :fit)
        for name in unique(st.fit for st in s.settings)
            r = s.results[findfirst(st -> st.fit === name, s.settings)]
            isempty(r.fixed) ? print(io, "\nfixed in fit `", name, "`: nothing") :
                               _print_fixed(io, r.fixed, θt(r), "fixed in fit `$name`")
        end
    else
        _print_fixed(io, s.results[1].fixed, θt(s.results[1]))
    end
    d = s.results[1].data                   # with simulate_with: the data's model
    d === nothing ||
        print(io, "\ndata from ", nameof(typeof(d.first)), " (its truth as in each setting)")
end

# The table: one row per setting and free parameter, then per derived quantity (`derived`).
# Each setting's rows come from a function of their own (`_settingrows`, a "function barrier":
# the setting's and the result's types are known there, so each row is built directly rather
# than looked up name by name), with the free parameters' true values found once per setting
# (`_freetruth`, diagnostics.jl). Before: the whole truth resolved for every row, 61% of a crlb
# sweep's time.
function _rows(settings, results)
    rows = NamedTuple[]
    sizehint!(rows, sum(r -> length(r.sd), results))     # one row per setting and parameter
    for (st, r) in zip(settings, results)
        append!(rows, _settingrows(st, r, _freetruth(r)))
    end
    return rows
end

_settingrows(st::NamedTuple, c::CRLBResult, θt) =
    map(p -> (; st..., parameter = p, truth = _truthcell(θt, p, c.fit[p] - c.bias[p]), noisefree = c.fit[p],
              bias = c.bias[p], relbias = c.relbias[p], sd = c.sd[p], relsd = c.relsd[p],
              rmse = c.rmse[p], relrmse = c.relrmse[p]), keys(c.fit))
_settingrows(st::NamedTuple, r::MonteCarloResult, θt) =
    map(p -> (; st..., parameter = p, truth = _truthcell(θt, p, r.mean[p] - r.bias[p]), mean = r.mean[p],
              bias = r.bias[p], relbias = r.relbias[p], sd = r.sd[p], relsd = r.relsd[p],
              rmse = r.rmse[p], relrmse = r.relrmse[p], se_mean = r.se_mean[p],
              se_sd = r.se_sd[p], converged = r.converged, crlb_fit = r.crlb.fit[p],
              crlb_bias = r.crlb.bias[p], crlb_relbias = r.crlb.relbias[p],
              crlb_sd = r.crlb.sd[p], crlb_relsd = r.crlb.relsd[p],
              crlb_relrmse = r.crlb.relrmse[p]), keys(r.sd))

# A row's true value: a free parameter's exactly, as given (`_freetruth`); a derived quantity's
# (after the parameters, `derived`) from the result, the noise-free fit's value (or the mean)
# minus its bias.
_truthcell(θt, p, fromresult) = haskey(θt, p) ? Float64(θt[p]) : fromresult

# -------------------------------------------------------------------------------------------------
# Display: the settings' columns, then the numbers; a setting's values on its first row only
# -------------------------------------------------------------------------------------------------

# A setting's value in a table cell: a fit setting by its name (`free`, not `:free`), a value
# as it is written (a tie keeps its colon: `:R1f`).
_settingvalue(name, x) = name === :fit || (name === :tissue && x isa Symbol) ? string(x) :
                         _showvalue(x)

function _print_table(io::IO, header, rows)
    w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for r in rows
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::Sweep{<:CRLBResult})
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    c1 = s.results[1]
    σ = c1.σ
    noise = !haskey(s.settings[1], :snr) ? ", " * _noisetext(σ, c1.Σ, c1.averages) :
            c1.averages === nothing ? "" : ", " * _averagestext(c1.averages)
    print(io, "crlb sweep for ", nameof(typeof(c1.model)), ", ", length(σ), " observations: ",
          length(s), " settings of ", join(keys(s.settings[1]), ", "), noise, ", covariance: ",
          c1.method)
    _print_sweep_fixed(io, s)                # the fit's assumptions (the truth varies)
    names = collect(keys(s.settings[1]))
    header = (string.(names)..., "parameter", "noise-free fit", "rel bias", "rel SD", "rel RMSE")
    rows = [(ntuple(j -> k == 1 ? _settingvalue(names[j], st[names[j]]) : "", length(names))...,
             string(p), f(c.fit[p]), pct(c.relbias[p]), pct(c.relsd[p]), pct(c.relrmse[p]))
            for (st, c) in zip(s.settings, s.results) for (k, p) in enumerate(keys(c.fit))]
    _print_table(io, header, rows)
end

function Base.show(io::IO, ::MIME"text/plain", s::Sweep{<:MonteCarloResult})
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    f2(x) = isfinite(x) ? string(round(x; sigdigits = 2)) : string(x)
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    r1 = s.results[1]
    noise = !haskey(s.settings[1], :snr) ? ", " * _noisetext(r1.σ, r1.crlb.Σ, r1.averages) :
            r1.averages === nothing ? "" : ", " * _averagestext(r1.averages)
    print(io, "montecarlo sweep for ", nameof(typeof(r1.model)), ", ", length(r1.crlb.σ),
          " observations: ", length(s), " settings of ", join(keys(s.settings[1]), ", "), ", ",
          r1.n, " noisy copies each, ", r1.noise, " noise",
          r1.averaging === nothing ? "" : ", $(r1.averaging) averaging", noise)
    _print_sweep_fixed(io, s)                # the fit's assumptions (the truth varies)
    print(io, "\nmeasured (± standard error) and predicted by crlb (sandwich, Gaussian noise)")
    names = collect(keys(s.settings[1]))
    header = (string.(names)..., "parameter", "rel bias", "crlb", "SD/crlb", "rel RMSE", "crlb",
              "converged")
    rows = [(ntuple(j -> k == 1 ? _settingvalue(names[j], st[names[j]]) : "", length(names))...,
             string(p), pct(r.relbias[p]), pct(r.crlb.relbias[p]),
             f(r.sd[p] / r.crlb.sd[p]) * " ± " * f2(r.se_sd[p] / r.crlb.sd[p]),
             pct(r.relrmse[p]), pct(r.crlb.relrmse[p]),
             k == 1 ? string(round(100 * r.converged; sigdigits = 4), "%") : "")
            for (st, r) in zip(s.settings, s.results) for (k, p) in enumerate(keys(r.sd))]
    _print_table(io, header, rows)
end

function Base.show(io::IO, s::Sweep)
    kind = eltype(s) <: CRLBResult ? "crlb" : eltype(s) <: MonteCarloResult ? "montecarlo" :
           eltype(s) <: Scenario ? "scenario" : "?"
    print(io, "Sweep(", length(s), " ", kind, " results: ", join(keys(s.settings[1]), ", "), ")")
end

# -------------------------------------------------------------------------------------------------
# A set of scenarios: tissues, boxes of true values, or both (what a protocol design is judged
# over; the MATLAB calcVmax's 'single', 'multiple' and 'ranges', Dortch et al. 2018)
#
#   scenarios(QMTSIR, :SIR_WM_I; kmf = 12.5)                         one
#   scenarios(QMTSIR, [:SIR_WM_I, :SIR_GM]; kmf = 12.5)              discrete tissues
#   scenarios(QMTSIR, θ; vary = (PSR = [0.05, 0.2], ...), midpoint = true)   a box: corners + center
#   scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.83, 0.88], Sm = 0.83)   each tissue × each value
#
# A sweep of the truth, as `vary` is for crlb and montecarlo (the same forms and order; the fit
# never changes), whose results are the scenarios themselves rather than their numbers: a
# `Sweep` of `Scenario`s, so `settings` says which tissue and values each one has.
# -------------------------------------------------------------------------------------------------

"""
    scenarios(M, truths; vary, midpoint = false, starts = :truth, changes...) -> Sweep

A set of [`Scenario`](@ref)s, the truths a protocol is judged over: one or several tissues,
every combination of true values within ranges, or both. The result is a [`Sweep`](@ref) whose
results are the scenarios: `s.settings[i]` says which tissue and values scenario `s[i]` has, and
`s.rows` is a table (setting, then every true value). [`ProtocolDesign`](@ref) takes it as it is;
after an evaluation `d.values` holds each scenario's value, so `s.settings[argmax(d.values)]` is
the worst.

- `M`: the model (its type, `QMTSIR`, or a protocol), as for `Scenario`.
- `truths`: a tissue name (`:SIR_WM_I`, [`TISSUES`](@ref)) or a named tuple of true values
  (filled in from the model as [`nominal`](@ref) does), or several: a vector of them, or a named
  tuple of named truths, `(wm = θ1, gm = θ2)`. With several, `settings` has their name (or
  place) as `tissue`, varying slowest.
- `vary`: true values to combine with each truth, as crlb's `vary` (`:Sm => [0.78, 0.88]`,
  `[:kmf => [5, 20], :PSR => [0.05, 0.2]]`, or a named tuple): every combination, the first
  parameter varying slowest. Two values each give the corners of a box; more, a grid.
- `midpoint = true`: also the box's center for each truth (each varied parameter halfway
  between its smallest and largest value, to 12 significant digits), first, as calcVmax's
  `'ranges'` (2⁵ corners + 1 = Li 2010's 33 sets); not repeated when the combinations already
  contain it. Numbers only. Off by default, so that `vary` gives exactly the combinations of
  the values given, as it does for `crlb` and `montecarlo`.
- `starts`, `changes...`: the fit, the same for every scenario, as for `Scenario` (`kmf = 12.5`:
  the fit assumes 12.5 whatever the true kmf; a parameter fixed by default and not given holds
  each scenario's own true value).

Varying a free parameter (PSR) moves the estimator's operating point; varying a fixed one (`Sm`)
with the fit's value given (`Sm = 0.83`) is a misspecification, what makes a scenario biased
(each scenario's `wrong`, the display's last column): the tissues under conditions where the
assumption is wrong, a robust design's set. Without the fit's value given, a varied fixed
parameter follows each truth, and nothing is wrong.

What a set is, and is not:

- **A set of points, not the box.** The corners and the midpoint need not contain the worst
  case: for a nonlinear model (SIR qMT) the largest error can lie inside the box. Check the
  worst case found with a denser grid (three or more values per parameter), or around the
  worst scenario.
- **Equal weights.** With `ProtocolDesign`'s default `aggregate = maximum` the midpoint is one
  more candidate. With `aggregate = mean` every scenario weighs the same: the center as much as
  each corner, a midpoint added to two values a third of the weight. That is a discrete mean,
  not a mean over the continuous box; for one, a regular grid (or quadrature points) with
  `midpoint = false`, or weights, `aggregate = v -> sum(w .* v)`.
- **The scale parameter need not be varied.** With `snr` relative to it (M0f, S0), the relative
  errors do not depend on it (Eq. 5's terms at M0f 0.8, 1.0 and 1.2 agree to 6e-14), so a box
  over SIR's PSR, kmf, R1f and Sf is 2⁴ + 1 = 17 scenarios; with M0f too, twice as many for
  the same answer.

[`crlb`](@ref) and [`montecarlo`](@ref) take a set too: one result per scenario, a `Sweep` with
the set's settings (`crlb(m, robust; snr = 200)`).

```julia
tissues = [:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion]
s4 = scenarios(QMTSIR, tissues; kmf = 12.5)                       # Dortch 2018 (calcVmax 'multiple')
box = scenarios(QMTSIR, nominal(QMTSIR); kmf = 12.5, midpoint = true,   # 'ranges': 2⁴ + 1
                vary = (PSR = [0.05, 0.2], kmf = [5, 20], R1f = [0.5, 1.5], Sf = [-1, -0.8]))
robust = scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.83, 0.88], kmf = 12.5, Sm = 0.83)  # 4 × 3
d = ProtocolDesign(QMTSIR(; SIR_4POINT..., magnitude = false), robust; snr = 200,
                   acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),
                   scantime = m -> sum(m.TI .+ m.TD .+ 0.3))
d(d.model); robust.settings[argmax(d.values)]                     # the worst: which tissue and Sm
```
"""
function scenarios(M::_ModelOrType, truths; vary = (;), midpoint::Bool = false,
                   starts::Symbol = :truth, changes...)
    named = _named_truths(M, truths)               # name => complete truth, or nothing => one
    ax = _varyaxes(vary)
    haskey(ax, :tissue) && throw(ArgumentError("`vary = :tissue => ...`: `tissue` names the truths; give several truths instead"))
    for n in keys(ax)
        haskey(parameters(M), n) || throw(ArgumentError("`scenarios`: `vary = :$n => ...` takes " *
            "parameters; to vary a derived quantity, give crlb or montecarlo the truth by its " *
            "derived quantities: `crlb(m, (S0 = ..., MD = ..., FA = ..., direction = ...); " *
            "vary = :FA => ...)`"))
    end
    # Every combination of the varied values (crlb's order: the first slowest), then the box's
    # center first, as calcVmax evaluates it first.
    combos = isempty(ax) ? NamedTuple[(;)] : _sweep_settings(M, nothing, ax)
    if midpoint && !isempty(ax)
        all(v -> all(x -> x isa Real, v), values(ax)) ||
            throw(ArgumentError("`midpoint`: the varied values must be numbers (a tie has no " *
                                "midpoint); give `midpoint = false`"))
        # Halfway, rounded to 12 significant digits so that the rounding of the sum does not
        # show: (0.78 + 0.88)/2 is 0.8300000000000001 in floating point, which a fit assuming
        # 0.83 would count as wrong. A combination within 1e-12 of it is the center already.
        center = map(v -> round((minimum(v) + maximum(v)) / 2; sigdigits = 12), ax)
        near(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-15)
        any(c -> all(k -> near(c[k], center[k]), keys(ax)), combos) || pushfirst!(combos, center)
    end
    settings = NamedTuple[]
    list = Scenario[]
    for (name, θ) in named, c in combos
        push!(settings, name === nothing ? c : (; tissue = name, c...))
        push!(list, Scenario(M, merge(θ, c); starts, changes...))
    end
    rows = NamedTuple[merge(st, sc.truth) for (st, sc) in zip(settings, list)]
    return Sweep(settings, list, rows)
end

# The truths by name: one (no name: `nothing`), or several, each named by its tissue name, its
# key (a named tuple of truths) or its place; each complete, as `nominal` fills it in.
_named_truths(M, t::Symbol) = [t => nominal(M, t)]
_named_truths(M, t::NamedTuple) = all(x -> x isa NamedTuple, values(t)) && !isempty(t) ?
    [k => nominal(M, v) for (k, v) in pairs(t)] : [nothing => nominal(M, t)]
function _named_truths(M, ts::Union{AbstractVector,Tuple})
    isempty(ts) && throw(ArgumentError("give at least one truth"))
    return [(t isa Symbol ? t : i) => _one_truth(M, t) for (i, t) in enumerate(ts)]
end
_named_truths(M, t) = throw(ArgumentError("a truth is a tissue name (e.g. :SIR_WM_I) or a " *
    "named tuple of true values, or several of them; got $(repr(t))"))
_one_truth(M, t::Union{Symbol,NamedTuple}) = nominal(M, t)
_one_truth(M, t) = _named_truths(M, t)            # the error above

# At the REPL: how many, the fit (if the same for all), and one row per scenario: its setting and
# what the fit gets wrong in it.
function Base.show(io::IO, ::MIME"text/plain", s::Sweep{<:Scenario})
    cell(x) = x isa Real ? repr(x; context = :compact => true) : _showvalue(x)
    sc = s.results
    print(io, length(sc), " scenario", length(sc) == 1 ? "" : "s", " for ", nameof(sc[1].model))
    ks = keys(s.settings[1])
    isempty(ks) || print(io, ": ", join(ks, " × "))
    fixed(x) = NamedTuple(n => r.value for (n, r) in pairs(x.roles) if r isa Fixed)
    free(x) = Tuple(n for (n, r) in pairs(x.roles) if r isa Free)
    if all(x -> free(x) == free(sc[1]) && isequal(fixed(x), fixed(sc[1])), sc)
        print(io, "\nthe fit: ", join(free(sc[1]), ", "), " free; ",
              join(("$n = $(cell(v))" for (n, v) in pairs(fixed(sc[1]))), ", "))
    else
        print(io, "\nthe fits differ (fixed values that follow each truth)")
    end
    header = (string.(ks)..., "wrong in the fit")
    value(k, x) = k === :tissue && x isa Symbol ? string(x) : cell(x)     # WM, not :WM
    rows = [((value(k, st[k]) for k in ks)...,
             isempty(x.wrong) ? "nothing" :
             join(("$n (true $(cell(w.truth)))" for (n, w) in pairs(x.wrong)), ", "))
            for (st, x) in zip(s.settings, sc)]
    w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for r in rows
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end

# crlb and montecarlo over a set of scenarios: one result per scenario (and noise level), the
# set's settings as the sweep's. Here rather than with crlb and montecarlo: a method for a
# `Sweep` of scenarios needs `Sweep` defined, which this file does after them.
function crlb(m::MRIModel, set::Sweep{<:Scenario}; snr = nothing, σ = nothing, Σ = nothing,
              averages = nothing, cov::Symbol = :sandwich, vary = (;), simulate_with = nothing,
              noise = nothing, averaging = nothing)
    noise === nothing || throw(_noise_refused("crlb"))     # crlb assumes Gaussian noise
    averaging === nothing || throw(_noise_refused("crlb", :averaging))    # first order: σ/√NEX
    _check_set(set, vary, simulate_with)
    foreach(x -> _check_protocol(m, x), set.results)
    caches = [_CRLBCache(m, sc, cov) for sc in set.results]          # one per scenario, once
    levels = snr isa AbstractVector ? collect(snr) : [snr]
    settings = _set_settings(set, snr, levels)
    results = [crlb!(caches[i], set[i].truth; snr = x, σ, Σ, averages)
               for x in levels for i in eachindex(caches)]
    return Sweep(settings, results, _rows(settings, results))
end
function montecarlo(m::MRIModel, set::Sweep{<:Scenario}; snr = nothing, σ = nothing,
                    Σ = nothing, n::Integer = 10_000, noise = nothing, averages = nothing,
                    averaging = nothing, rng::AbstractRNG = Random.default_rng(), vary = (;),
                    simulate_with = nothing, solver::LM = LM(),
                    precision::Type{<:AbstractFloat} = Float64)
    _check_set(set, vary, simulate_with)
    foreach(x -> _check_protocol(m, x), set.results)
    n >= 2 || throw(ArgumentError("montecarlo needs at least 2 noisy copies (n = $n)"))
    noise = _checked_kind(noise)             # :rician or :gaussian, given (no default)
    averaging = _checked_averaging(averaging, noise, averages)    # Rician with averages
    seed = rand(rng, UInt64)                 # one seed for every setting (common random numbers)
    options = (; solver, precision)
    levels = snr isa AbstractVector ? collect(snr) : [snr]
    settings = _set_settings(set, snr, levels)
    results = [_montecarlo(m, sc, sc.truth, x, σ, Σ, n, options, noise, averages, averaging,
                           seed, nothing)
               for x in levels for sc in set.results]
    return Sweep(settings, results, _rows(settings, results))
end
_check_set(set, vary, simulate_with) =
    !isempty(vary) ? throw(ArgumentError("`vary`: the set's truths are given; vary them in " *
                                         "`scenarios` (its `vary`)")) :
    simulate_with !== nothing ? throw(ArgumentError("`simulate_with` is for one scenario, not a " *
                                                    "set")) : nothing
# The noise level first (slowest), then the set's own settings.
_set_settings(set, snr, levels) = snr isa AbstractVector ?
    NamedTuple[(; snr = x, st...) for x in levels for st in set.settings] : copy(set.settings)
