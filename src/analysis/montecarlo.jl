# =================================================================================================
# montecarlo: the bias, SD and RMSE of a fit, measured by simulating and fitting noisy copies
#
#   truth θ ──simulate──▶ n noisy copies of one voxel ──fit (threads)──▶ n estimates
#                                                                           │
#   statistics over the n estimates (mean, bias, SD, RMSE, ± standard errors) ◀┘
#   next to crlb's prediction at the same truth, noise and fit settings
#
# Why both this and `crlb` (analysis/crlb.jl): crlb predicts from derivatives, first order in
# the noise, for Gaussian noise; Monte Carlo shows the actual estimator, with what the
# derivatives leave out: Rician noise (the magnitude floor at low signal), bounds, local minima
# and the fit's own starts (`:scale`, `:data`, `BestOf`), nonlinearity at low SNR. Where both
# apply they should agree within the Monte Carlo's standard errors: Dortch et al., MRM
# 2018;80:1824, Fig. 1 (Rician Monte Carlo against the CRLB at SNR 200); Lankford & Does, MRM
# 2013;69:127 (within 8%, at very high SNR for the 7-parameter mcDESPOT fit).
#
# One case, or a sweep (`snr = [...]`, `vary = :kmf => 5:2.5:20`, sweep.jl), fit settings side
# by side (`fits`) and data from another model (`simulate_with`); docs/design.md, "montecarlo".
# =================================================================================================

"""
    MonteCarloResult

What [`montecarlo`](@ref) returns. Fields, each a named tuple over the fit's free parameters
(`mc.sd.PSR`), with the names and meanings of [`CRLBResult`](@ref)'s so the two line up:

- `mean` (of the estimates), `bias` (mean − truth), `sd`, `rmse`, in the parameters' units
- `relbias`, `relsd`, `relrmse`: the same divided by |truth|
- `se_mean`: standard error of the mean (and of the bias), SD/√n
- `se_sd`: standard error of the SD, SD/√(2(n − 1)) (the Gaussian approximation)

and `converged` (the fraction of the `n` fits that converged), `n`, `noise` (`:rician` or
`:gaussian`), `σ` (the noise SD of each observation), `crlb` (the [`crlb`](@ref) prediction at
the same truth, noise and fit settings), `result` (the [`FitResult`](@ref) of the noisy copies: the estimates themselves,
e.g. `mc.result.params.PSR` for a histogram), `truth`, `fixed` (the fit's fixed values), `model`
and `data` (`nothing`, or the `model => truth` the data came from, with `simulate_with`).
[`vcov`](@ref)`(mc)` is the covariance matrix of the estimates, [`stderror`](@ref)`(mc)` their
SDs (StatsAPI's names; `informationmatrix(mc.crlb)` for the predicted information).
"""
struct MonteCarloResult{M<:MRIModel,N<:NamedTuple,Tr<:NamedTuple,X<:NamedTuple,R<:FitResult,
                        C<:CRLBResult,D}
    mean::N
    bias::N
    sd::N
    rmse::N
    relbias::N
    relsd::N
    relrmse::N
    se_mean::N
    se_sd::N
    converged::Float64
    n::Int
    noise::Symbol
    σ::Vector{Float64}
    crlb::C
    result::R
    truth::Tr
    fixed::X
    model::M
    data::D                    # nothing, or `model => truth` the data came from (simulate_with)
end

"""
    montecarlo(m, truth; snr or σ or Σ, n = 10_000, noise, rng, vary, fits, simulate_with,
               settings...)
        -> MonteCarloResult or Sweep

Simulate `n` noisy copies of one voxel with the values `truth`, fit them, and measure each
fitted parameter's bias, SD and RMSE, with standard errors, next to [`crlb`](@ref)'s prediction
for the same truth, noise and fit settings.

- `truth`: every parameter of `m` (numbers, or ties), as for [`simulate`](@ref): e.g.
  `nominal(m, :SIR_WM_I)`.
- The noise, as for `simulate` and `crlb`: `snr` (relative to the scale parameter), `σ` (a
  number, or one SD per observation) or `Σ` (the noise covariance matrix).
- `n`: the number of noisy copies, fitted as one dataset on Julia's threads (start Julia with
  `-t auto`).
- The fit's settings: the keywords of [`fit`](@ref), as for `fit(m, y; ...)`: parameters'
  roles (`kmf = 5.0`, `kmf = Free()`), `solver`, `precision`; the model's own for the rest. The
  fits start as `fit` would start them, not at the truth, so local minima and the starting
  rules are part of what is measured.
- `noise`: `:rician` (the default for magnitude data) or `:gaussian` (the default when the
  data's model is for signed data, `magnitude = false`); crlb assumes Gaussian noise.
- `rng`: the random number generator; `MersenneTwister(seed)` (from Random) for repeatable data.
  One number is drawn from it, the seed of the noise.
- Sweeps, as for `crlb`: `snr = [50, 100, 200]`, `vary = :kmf => 5:2.5:20` (true values; the
  fit's settings stay), every combination; the result is a [`Sweep`](@ref) of
  `MonteCarloResult`s, with a table in `s.rows`. Every setting's noise comes from the same seed
  ("common random numbers"): the settings see the same noise, so differences between them are
  not noise, and adding a setting does not change the others' results.
- Fit settings side by side, as for `crlb`: `fits = (fixed = (;), free = (; kmf = Free()))`. The
  same noisy copies are fitted each way: a paired comparison.
- Data from another model, as for `crlb`: `simulate_with = m2 => θ2`; the estimates are judged
  against `truth`, your reference values for `m`'s parameters; `snr` is relative to `m2`'s
  scale parameter, and `vary` changes `θ2`.

Statistics use every fitted copy, converged or not, as [`compare`](@ref) does; `mc.converged`
says how many converged, and `mc.result` holds every estimate and status.

```julia
m = QMTSIR(; SIR_4POINT...)
mc = montecarlo(m, nominal(m, :SIR_WM_I); snr = 200)        # the fit fixes kmf at 12.5
mc.bias.PSR, mc.sd.PSR, mc.sd.PSR / mc.crlb.sd.PSR          # measured, and against the CRLB
montecarlo(m, nominal(m, :SIR_WM_I); snr = 200, kmf = 5.0, rng = MersenneTwister(1))
montecarlo(m, nominal(m, :SIR_WM_II); snr = 200, vary = :kmf => 5:2.5:20)    # Fig. 2B
montecarlo(QMTSIR(; SIR_5POINT...), θ; snr = 200, fits = (fixed = (;), free = (; kmf = Free())))
```
"""
function montecarlo(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
                    n::Integer = 10_000, noise = nothing, rng::AbstractRNG = Random.default_rng(),
                    vary = (;), fits::NamedTuple = (;), simulate_with = nothing, settings...)
    # The fit's settings: every keyword that is not one of montecarlo's own, as for `fit`
    # (parameters' roles, and `fit`'s options `solver` and `precision`).
    fit = _fitsettings("montecarlo", m, settings,
                       (:snr, :σ, :Σ, :n, :noise, :rng, :vary, :fits, :simulate_with))
    vary = _varyaxes(vary)
    n >= 2 || throw(ArgumentError("montecarlo needs at least 2 noisy copies (n = $n)"))
    any(_hasmap, values(truth)) && throw(ArgumentError("montecarlo simulates copies of one " *
        "voxel: give numbers. For maps (a phantom), simulate and fit them, then `compare`"))
    any(f -> haskey(f, :mask), (fit, values(fits)...)) &&
        throw(ArgumentError("`mask`: montecarlo fits every copy"))
    _check_fits(fit, fits)
    _check_simulate_with(simulate_with)
    # The noise: as `simulate` would add it to the data's model (Rician for magnitude data).
    noise = something(noise, _default_noise(_datamodel(m, simulate_with)))
    seed = rand(rng, UInt64)                 # the noise's seed: one draw, for every setting
    if _issweep(snr, vary, fits)             # a sweep: one run per setting (sweep.jl)
        settings = _sweep_settings(_datamodel(m, simulate_with), snr, vary, fits)
        results = [_montecarlo(m, _truth_of(st, truth, simulate_with), _snr_of(st, snr), σ, Σ, n,
                               _fit_of(st, fit, fits), noise, seed,
                               _data_of(st, simulate_with)) for st in settings]
        return Sweep(settings, results, _rows(settings, results))
    end
    return _montecarlo(m, truth, snr, σ, Σ, n, fit, noise, seed, simulate_with)
end

# One setting: one truth, one noise level, one set of fit settings. `fit` is an argument here,
# so inside this function the name means the settings; the fits call `_fit_copies` (below),
# which is the function `fit`.
function _montecarlo(m, truth, snr, σ, Σ, n, fit, noise, seed, simulate_with)
    θ = resolve(m, truth)                    # checks the truth: every parameter, no roles

    # The prediction first: it also checks the noise level, the fit's parameter settings and
    # the data's model.
    roles = (; (k => v for (k, v) in pairs(fit) if !(k in _OPTIONS))...)   # crlb: roles only
    c = crlb(m, truth; snr, σ, Σ, simulate_with, roles...)

    # n noisy copies (an n × observations matrix: one voxel per row), from the fitted model at
    # the truth or from the data's model (`simulate_with`), fitted as one dataset.
    dm, dt = simulate_with === nothing ? (m, truth) : (simulate_with.first, simulate_with.second)
    Y = simulate(dm, dt; snr, σ, Σ, noise, n, rng = Xoshiro(seed))
    res = _fit_copies(m, Y; fit...)

    # Statistics over every fitted copy (as `compare`), each against the parameter's true value.
    F = keys(res.params)
    used = (res.status .!= NotFitted) .& (res.status .!= InvalidData)
    k = count(used)
    k >= 2 || throw(ErrorException("montecarlo: fewer than 2 copies could be fitted"))
    stat(f) = NamedTuple{F}(map(p -> f(Float64.(res.params[p][used]), Float64(θ[p])), F))
    meanv = stat((x, t) -> mean(x))
    bias = stat((x, t) -> mean(x) - t)
    sd = stat((x, t) -> std(x))
    rmse = stat((x, t) -> sqrt(mean(abs2, x .- t)))
    rel(q) = NamedTuple{F}(map(p -> q[p] / abs(Float64(θ[p])), F))
    se_mean = map(s -> s / sqrt(k), sd)
    se_sd = map(s -> s / sqrt(2 * (k - 1)), sd)
    return MonteCarloResult(meanv, bias, sd, rmse, rel(bias), rel(sd), rel(rmse), se_mean, se_sd,
                            count(isconverged, res.status) / n, Int(n), noise, c.σ, c, res,
                            truth, c.fixed, m, simulate_with)
end

# StatsAPI's names: the covariance matrix of the estimates (every fitted copy, as the other
# statistics), and their SDs, the free parameters in model order.
function vcov(mc::MonteCarloResult)
    r = mc.result
    used = (r.status .!= NotFitted) .& (r.status .!= InvalidData)
    return cov(reduce(hcat, (Float64.(r.params[p][used]) for p in keys(r.params))))
end
stderror(mc::MonteCarloResult) = collect(values(mc.sd))

# The dataset fit, under a name the keyword `fit` of `montecarlo` does not hide.
_fit_copies(m, Y; kwargs...) = fit(m, Y; kwargs...)

# -------------------------------------------------------------------------------------------------
# Display: measured and predicted side by side, one row per free parameter
# -------------------------------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", mc::MonteCarloResult)
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    f2(x) = isfinite(x) ? string(round(x; sigdigits = 2)) : string(x)      # standard errors
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    c = mc.crlb
    print(io, "montecarlo for ", nameof(typeof(mc.model)), ", ", length(c.σ), " observations, ",
          _noisetext(mc.σ, c.Σ), ", ", mc.noise, " noise: ", mc.n, " noisy copies, ",
          string(round(100 * mc.converged; sigdigits = 4)), "% converged")
    _print_data(io, mc.data)
    _print_fixed(io, mc.fixed, mc.truth)
    print(io, "\nmeasured (± standard error) and predicted by crlb (sandwich, Gaussian noise)")
    header = ("parameter", "true", "mean", "rel bias", "crlb", "SD", "crlb", "SD/crlb",
              "rel RMSE", "crlb")
    rows = map(collect(keys(mc.sd))) do p
        t = mc.mean[p] - mc.bias[p]
        ratio, se_ratio = mc.sd[p] / c.sd[p], mc.se_sd[p] / c.sd[p]
        (string(p), f(t), f(mc.mean[p]) * " ± " * f2(mc.se_mean[p]), pct(mc.relbias[p]),
         pct(c.relbias[p]), f(mc.sd[p]) * " ± " * f2(mc.se_sd[p]), f(c.sd[p]),
         f(ratio) * " ± " * f2(se_ratio), pct(mc.relrmse[p]), pct(c.relrmse[p]))
    end
    w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for r in rows
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end

Base.show(io::IO, mc::MonteCarloResult) =
    print(io, "MonteCarloResult(", nameof(typeof(mc.model)), ": ", join(keys(mc.sd), ", "),
          "; n = ", mc.n, ")")
