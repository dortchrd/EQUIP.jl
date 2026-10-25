# =================================================================================================
# Sweeps: crlb or montecarlo over several noise levels and true values
#
#   crlb(m, θ; snr = [100, 200], vary = (kmf = 5:2.5:20,))       14 settings, one crlb each
#   montecarlo(m, θ; snr = 200, vary = (kmf = 5:2.5:20,))         7 settings, one run each
#
# The settings are every combination of the values given, the first keyword varying slowest
# (like nested loops written in that order), and the fit settings (`fits`) fastest, so a table
# shows the fits of one case next to each other. `vary` changes the TRUTH, never the fit: with the
# fit fixing kmf at 12.5 and the true kmf swept from 5 to 20, the result is the bias of that
# assumption against the true value (Dortch et al., MRM 2018;80:1824, Fig. 2B). Varying a
# fitted parameter instead (PSR = 0.05:0.05:0.2) gives the accuracy over a tissue range.
# =================================================================================================

"""
    Sweep

What [`crlb`](@ref) and [`montecarlo`](@ref) return for several settings (`snr = [...]`,
`vary = (...)`):

- `settings`: one named tuple per setting, e.g. `(snr = 100, kmf = 5.0)`
- `results`: the [`CRLBResult`](@ref) or [`MonteCarloResult`](@ref) of each, in the same order;
  `s[i]` is `s.results[i]`, `length(s)` the number of settings, and `for r in s` loops over them
- `rows`: the table, one named tuple per setting and free parameter: the setting's values, then
  `parameter`, `truth` and the numbers (crlb: `noisefree`, the noise-free fit's value, `bias`,
  ...; montecarlo: measured, then `crlb_...` predicted), e.g. `DataFrame(s.rows)` with
  DataFrames.jl. With `fits`, the setting's `fit` column holds the fit setting's name.
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

# Every combination of the noise levels and the varied values, the first keyword slowest.
# `Iterators.product` varies its FIRST argument fastest, so the axes go in reversed and each
# combination is reversed back.
function _sweep_settings(m::MRIModel, snr, vary::NamedTuple, fits::NamedTuple = (;))
    roles = parameters(m)
    for (k, v) in pairs(vary)
        k === :fit && throw(ArgumentError("`vary = (; fit)`: `fit` names the fit settings in a " *
            "sweep; for several fit settings use `fits`"))
        haskey(roles, k) || throw(ArgumentError("`vary = (; $k)`: not a parameter of " *
            "$(nameof(typeof(m))); the parameters are $(keys(roles))"))
        v isa AbstractVector && !isempty(v) || throw(ArgumentError("`vary = ($k = ...)`: give " *
            "the values as a vector or a range, e.g. `$k = [5.0, 10.0]` or `$k = 5:2.5:20`"))
    end
    ax = map(collect, vary)                    # the axes: the values of each keyword
    snr isa AbstractVector && (ax = merge((snr = collect(snr),), ax))   # the noise level first
    isempty(fits) || (ax = merge(ax, (fit = collect(keys(fits)),)))     # the fit settings last
    names = keys(ax)
    return NamedTuple[NamedTuple{names}(reverse(c))
                      for c in vec(collect(Iterators.product(reverse(values(ax))...)))]
end

# A setting split into its parts: the noise level (or the one given for every setting), the fit
# settings (by name, from `fits`; or the one `fit` for every setting), and the true values it
# changes: the fitted model's truth, or with data from another model (`simulate_with = m2 =>
# θ2`) the data's truth θ2, while `truth` stays the reference the fit is judged against.
_snr_of(setting, snr) = haskey(setting, :snr) ? setting.snr : snr
_fit_of(setting, fit, fits) = haskey(setting, :fit) ? fits[setting.fit] : fit
_varied(setting) = Base.structdiff(setting, NamedTuple{(:snr, :fit)})
_datamodel(m, ::Nothing) = m
_datamodel(m, s::Pair) = s.first
_truth_of(setting, truth, ::Nothing) = merge(truth, _varied(setting))
_truth_of(setting, truth, ::Pair) = truth
_data_of(setting, ::Nothing) = nothing
_data_of(setting, s::Pair) = s.first => merge(s.second, _varied(setting))

# The fit settings of a sweep, one line per named setting (`fits`), or one for all.
function _print_sweep_fixed(io::IO, s::Sweep)
    if haskey(s.settings[1], :fit)
        for name in unique(st.fit for st in s.settings)
            r = s.results[findfirst(st -> st.fit === name, s.settings)]
            isempty(r.fixed) ? print(io, "\nfixed in fit `", name, "`: nothing") :
                               _print_fixed(io, r.fixed, r.fixed, "fixed in fit `$name`")
        end
    else
        _print_fixed(io, s.results[1].fixed, s.results[1].fixed)   # the truth varies: no "true"
    end
    d = s.results[1].data                   # with simulate_with: the data's model
    d === nothing ||
        print(io, "\ndata from ", nameof(typeof(d.first)), " (its truth as in each setting)")
end

# The true value of free parameter p in a result, exactly (the result keeps its truth as given).
_truthof(r, p) = Float64(resolve(r.model, r.truth)[p])

# The table: one row per setting and free parameter.
function _rows(settings, results::Vector{<:CRLBResult})
    return NamedTuple[(; st..., parameter = p, truth = _truthof(c, p), noisefree = c.fit[p],
                       bias = c.bias[p], relbias = c.relbias[p], sd = c.sd[p], relsd = c.relsd[p],
                       rmse = c.rmse[p], relrmse = c.relrmse[p])
                      for (st, c) in zip(settings, results) for p in keys(c.fit)]
end
function _rows(settings, results::Vector{<:MonteCarloResult})
    return NamedTuple[(; st..., parameter = p, truth = _truthof(r, p), mean = r.mean[p],
                       bias = r.bias[p], relbias = r.relbias[p], sd = r.sd[p], relsd = r.relsd[p],
                       rmse = r.rmse[p], relrmse = r.relrmse[p], se_mean = r.se_mean[p],
                       se_sd = r.se_sd[p], converged = r.converged, crlb_fit = r.crlb.fit[p],
                       crlb_bias = r.crlb.bias[p], crlb_relbias = r.crlb.relbias[p],
                       crlb_sd = r.crlb.sd[p], crlb_relsd = r.crlb.relsd[p],
                       crlb_relrmse = r.crlb.relrmse[p])
                      for (st, r) in zip(settings, results) for p in keys(r.sd)]
end

# -------------------------------------------------------------------------------------------------
# Display: the settings' columns, then the numbers; a setting's values on its first row only
# -------------------------------------------------------------------------------------------------

# A setting's value in a table cell: a fit setting by its name (`free`, not `:free`), a value
# as it is written (a tie keeps its colon: `:R1f`).
_settingvalue(name, x) = name === :fit ? string(x) : _showvalue(x)

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
    noise = haskey(s.settings[1], :snr) ? "" :
            all(==(σ[1]), σ) ? ", σ = $(f(σ[1]))" : ", σ per observation"
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
    noise = haskey(s.settings[1], :snr) ? "" : ", σ = $(f(r1.σ))"
    print(io, "montecarlo sweep for ", nameof(typeof(r1.model)), ", ", length(r1.crlb.σ),
          " observations: ", length(s), " settings of ", join(keys(s.settings[1]), ", "), ", ",
          r1.n, " noisy copies each, ", r1.noise, " noise", noise)
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
    kind = eltype(s) <: CRLBResult ? "crlb" : eltype(s) <: MonteCarloResult ? "montecarlo" : "?"
    print(io, "Sweep(", length(s), " ", kind, " results: ", join(keys(s.settings[1]), ", "), ")")
end
