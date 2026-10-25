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
# One case for now: one truth, one noise level, one set of fit settings. Sweeps (`snr = [...]`,
# `vary = (kmf = 5:2.5:20,)`), fit settings side by side and data from another model
# (`simulate_with`) come next (docs/design.md, "montecarlo").
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
`:gaussian`), `σ`, `crlb` (the [`crlb`](@ref) prediction at the same truth, noise and fit
settings), `result` (the [`FitResult`](@ref) of the noisy copies: the estimates themselves,
e.g. `mc.result.params.PSR` for a histogram), `truth`, `fixed` (the fit's fixed values) and
`model`.
"""
struct MonteCarloResult{M<:MRIModel,N<:NamedTuple,Tr<:NamedTuple,X<:NamedTuple,R<:FitResult,
                        C<:CRLBResult}
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
    σ::Float64
    crlb::C
    result::R
    truth::Tr
    fixed::X
    model::M
end

"""
    montecarlo(m, truth; snr or σ, n = 10_000, fit = (...), noise, rng) -> MonteCarloResult

Simulate `n` noisy copies of one voxel with the values `truth`, fit them, and measure each
fitted parameter's bias, SD and RMSE, with standard errors, next to [`crlb`](@ref)'s prediction
for the same truth, noise and fit settings.

- `truth`: every parameter of `m` (numbers, or ties), as for [`simulate`](@ref): e.g.
  `nominal(m, :SIR_WM_I)`.
- `snr` (relative to the scale parameter, as for `simulate`) or `σ`, a number.
- `n`: the number of noisy copies, fitted as one dataset on Julia's threads (start Julia with
  `-t auto`).
- `fit`: the fit's settings, the keywords of [`fit`](@ref) in a named tuple (roles, `solver`,
  `precision`); the model's own when left out. The fits start as `fit` would start them, not
  at the truth, so local minima and the starting rules are part of what is measured.
- `noise`: `:rician` (the default for magnitude data) or `:gaussian` (the default when the model
  fits signed data, `magnitude = false`); crlb assumes Gaussian noise.
- `rng`: the random number generator; `MersenneTwister(seed)` (from Random) for repeatable data.

Statistics use every fitted copy, converged or not, as [`compare`](@ref) does; `mc.converged`
says how many converged, and `mc.result` holds every estimate and status.

```julia
m = QMTSIR(; SIR_4POINT...)
mc = montecarlo(m, nominal(m, :SIR_WM_I); snr = 200)        # the fit fixes kmf at 12.5
mc.bias.PSR, mc.sd.PSR, mc.sd.PSR / mc.crlb.sd.PSR          # measured, and against the CRLB
montecarlo(m, nominal(m, :SIR_WM_I); snr = 200, fit = (kmf = 5.0,), rng = MersenneTwister(1))
```
"""
function montecarlo(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing,
                    n::Integer = 10_000, fit::NamedTuple = (;), noise::Symbol = _default_noise(m),
                    rng::AbstractRNG = Random.default_rng())
    # `fit` is a keyword here, so inside this function the name means the settings; the fits
    # call `_fit_copies` (below), which is the function `fit`.
    n >= 2 || throw(ArgumentError("montecarlo needs at least 2 noisy copies (n = $n)"))
    any(_hasmap, values(truth)) && throw(ArgumentError("montecarlo simulates copies of one " *
        "voxel: give numbers. For maps (a phantom), simulate and fit them, then `compare`"))
    σ === nothing || σ isa Real || throw(ArgumentError("montecarlo takes `σ` as a number " *
        "(simulate has no σ per observation yet); crlb takes one per observation"))
    haskey(fit, :mask) && throw(ArgumentError("`fit = (; mask)`: montecarlo fits every copy"))
    θ = resolve(m, truth)                    # checks the truth: every parameter, no roles

    # The prediction first: it also checks the noise level and the fit's parameter settings.
    roles = (; (k => v for (k, v) in pairs(fit) if !(k in _OPTIONS))...)   # crlb: roles only
    c = crlb(m, truth; snr, σ, fit = roles)

    # n noisy copies (an n × observations matrix: one voxel per row), fitted as one dataset.
    Y = simulate(m, truth; snr, σ, noise, n, rng)
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
                            count(isconverged, res.status) / n, Int(n), noise, c.σ[1], c, res,
                            truth, c.fixed, m)
end

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
    print(io, "montecarlo for ", nameof(typeof(mc.model)), ", ", length(c.σ), " observations, σ = ",
          f(mc.σ), ", ", mc.noise, " noise: ", mc.n, " noisy copies, ",
          string(round(100 * mc.converged; sigdigits = 4)), "% converged")
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
