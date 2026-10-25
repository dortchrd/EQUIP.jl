# =================================================================================================
# crlb: the bias, SD and RMSE of a fit, predicted from derivatives (no noisy fits)
#
#   truth θ ──────▶ g_true = signal(m, θ)                 the noise-free data
#   fit settings ─▶ free and fixed parameters, bounds     as `fit` uses them
#   ① noise-free fit  θ̂ = noisefree(m, θ; fit), started at the truth;  b = bias(θ̂, θ) = θ̂ − θ
#   ② derivatives     J₀ = jacobian(θ̂);  H = hessian(θ̂, g_true) = J₀ᵀJ₀ + Σᵢ r₀ᵢ∇²gᵢ
#   ③ covariance      :sandwich  sandwich(H, J₀, Σ) = H⁻¹ J₀ᵀΣJ₀ H⁻¹          (default)
#                     :biased    biasedcrlb(∇b, F) = (I+∇b) F⁻¹ (I+∇b)ᵀ,  F = J_tᵀΣ⁻¹J_t
#   ④ report          SD = √diag, RMSE = √(b² + SD²), absolute and relative to the truth
#
# Each step is a function of its own (derivatives.jl), and `crlb` is their composition, so the
# code reads as the formulas; the allocation-free kernel for the protocol optimizer
# (crlb_kernel.jl) does the same steps without forming the matrices, and is tested against it.
#
# Why a bias at all: a fit that fixes a parameter at a wrong value (kmf = 12.5 for a tissue with
# kmf = 5) cannot reproduce the data, even without noise. Its noise-free answer θ̂ (the
# "pseudo-true" value) differs from the truth by the bias b; noise then spreads the fit around
# θ̂, not around the truth. When nothing fixed is wrong, θ̂ = θ, b = 0, r₀ = 0, H = JᵀJ, and both
# covariances are the classical CRLB σ²(JᵀJ)⁻¹.
#
# The two covariances (both first order in σ; see docs/design.md, "crlb"):
# - the sandwich, H⁻¹(J₀ᵀΣJ₀)H⁻¹: what the least-squares fit itself achieves. Noise ε moves the
#   fit by δ = H⁻¹J₀ᵀε (expand the fit's zero-gradient condition around θ̂), so Cov(δ) is the
#   "meat" J₀ᵀΣJ₀ (how hard the noise pushes) between two slices of "bread" H⁻¹ (how stiff the
#   cost function is). Huber–White (White, Econometrica 1982;50:1), the misspecified CRB
#   (Fortunati et al., IEEE Signal Process Mag 2017;34(6):142).
# - Eq. 3 of Dortch et al., MRM 2018;80:1824, (I + ∇b)F⁻¹(I + ∇b)ᵀ: a bound for any estimator
#   with this bias gradient. Written out, σ²H⁻¹J₀ᵀP_tJ₀H⁻¹ with P_t = J_t(J_tᵀJ_t)⁻¹J_tᵀ, so it
#   is never larger than the sandwich. Equal when the fit is exact (as many observations as
#   free parameters, e.g. the 4-point scheme).
#
# ∇b without extra fits (the slow part of the MATLAB code, which refits perturbed data): the
# fit's gradient G(θ̂, θ) = J₀ᵀ[g(θ̂) − g(θ; true)] is 0 for every truth θ, so differentiating it
# (implicit function theorem) gives ∂θ̂/∂θ = −H⁻¹·∂G/∂θ = H⁻¹J₀ᵀJ_t, one linear solve
# (`biasgradient`).
# =================================================================================================

"""
    CRLBResult

What [`crlb`](@ref) returns. Fields, each a named tuple over the fit's free parameters (`c.sd.PSR`,
like `res.params.PSR` of a [`FitResult`](@ref)):

- `fit`: the noise-free fit's values θ̂ (the "pseudo-true" values, where the noisy fits center)
- `bias` (θ̂ − truth), `sd`, `rmse` (√(bias² + SD²)), in the parameters' units
- `relbias`, `relsd`, `relrmse`: the same divided by |truth|

and `cov` (the covariance matrix, free parameters in model order), `cond` (condition number of
the Jacobian with each column scaled by its parameter's true value: large means the parameters
are hard to tell apart), `onbound` (free parameters on a bound in the noise-free fit: held there,
SD 0), `status` (the noise-free fit's [`FitStatus`](@ref)), `method` (`:sandwich` or `:biased`),
`σ` (the noise SD per observation), `truth`, `fixed` (the fit's fixed values), `model`,
`data` (`nothing`, or the `model => truth` the data came from, with `simulate_with`) and `Σ`
(`nothing`, or the noise covariance matrix given). [`informationmatrix`](@ref)`(c)`,
[`vcov`](@ref)`(c)` and [`stderror`](@ref)`(c)` give the information and covariance matrices
and the SDs, under StatsAPI's names.
"""
struct CRLBResult{M<:MRIModel,N<:NamedTuple,Tr<:NamedTuple,X<:NamedTuple,D,S}
    fit::N
    bias::N
    sd::N
    rmse::N
    relbias::N
    relsd::N
    relrmse::N
    cov::Matrix{Float64}
    cond::Float64
    onbound::Vector{Symbol}
    status::FitStatus
    method::Symbol
    σ::Vector{Float64}
    truth::Tr
    fixed::X
    model::M
    data::D                    # nothing, or `model => truth` the data came from (simulate_with)
    Σ::S                       # nothing, or the noise covariance matrix given (`Σ = ...`)
    information::Matrix{Float64}   # Fisher information at θ̂, J₀ᵀΣ⁻¹J₀
    observed::Matrix{Float64}      # observed information at θ̂: the likelihood's Hessian
end

"""
    crlb(m, truth; snr or σ or Σ, fit = (...), cov = :sandwich, vary, fits, simulate_with)
        -> CRLBResult or Sweep

The bias, SD and RMSE of each fitted parameter predicted for one voxel, from derivatives: the
Cramér–Rao lower bound (CRLB, also CRB), biased when the fit fixes a parameter at a value other
than the truth's. No noisy fits: the bias comes from one fit of the noise-free data, and the
SD from the derivatives at that fit's answer, first order in the noise.

- `truth`: every parameter of `m` (numbers, or ties), as for [`simulate`](@ref): e.g.
  `nominal(m, :SIR_WM_I)` or `nominal(m; PSR = 0.15)`.
- The noise (Gaussian), one of: `snr` (relative to the model's scale parameter, as for
  `simulate`: σ = M0/snr); `σ`, a number or one SD per observation (independent noise, e.g.
  different bandwidths or numbers of averages); `Σ`, the full covariance matrix of the noise
  between observations (correlated noise, e.g. after denoising or a joint reconstruction).
  Chosen by type, so the common case, one σ for every observation ("pulled out" of
  JᵀΣ⁻¹J = JᵀJ/σ²), costs nothing extra.
- `fit`: the fit's settings, the keywords of [`fit`](@ref) in a named tuple; the model's own
  roles when left out. The prediction is for a fit with these settings: `fit = (kmf = Free(),)`
  fits kmf, `fit = (kmf = 15.0,)` fixes it at 15. The bias is zero unless a fixed value differs
  from the truth; the classical CRLB with kmf fixed at its true value is `fit = (kmf = 5.0,)`.
  Bounds are the fit's.
- `cov`: the covariance, `:sandwich` (default) or `:biased`. The sandwich is the covariance of
  the least-squares fit itself (what EQUIP's fit achieves; the misspecified CRB); `:biased` is
  Eq. 3 of Dortch et al., MRM 2018, a bound for any estimator with the same bias gradient. Both
  are the classical CRLB σ²(JᵀJ)⁻¹ when nothing fixed is wrong, and equal each other when the
  fit is exact (as many observations as free parameters); otherwise `:biased` ≤ `:sandwich`.
- Sweeps: `snr = [50, 100, 200]` (several noise levels), `vary = (kmf = 5:2.5:20,)` (several
  true values of a parameter; the fit's settings stay as they are, so this is the bias of a
  wrong assumption against the truth, Dortch 2018 Fig. 2B). Every combination is computed, the
  first keyword varying slowest; the result is a [`Sweep`](@ref), with a table in `s.rows`.
- Fit settings side by side: `fits = (fixed = (;), free = (kmf = Free(),))`, named; one more
  axis of the sweep (the last, so a table has the settings of one case next to each other).
  Instead of `fit`.
- Data from another model: `simulate_with = m2 => θ2` (a model with as many observations, and
  every value of its parameters). The noise-free fit of `m` to `m2`'s signal gives the bias,
  against `truth`, your reference values for `m`'s parameters (e.g. the long T2 of a
  bi-exponential tissue fitted mono-exponentially); `snr` is relative to `m2`'s scale
  parameter; `vary` changes `θ2`. The sandwich only: Eq. 3 needs the true model's derivatives
  with respect to `m`'s parameters.

A free parameter that ends on a bound in the noise-free fit is held there (SD 0; `c.onbound`).
Parameters the data cannot determine (a singular Jacobian) get SD `Inf`. EQUIP's fits are
unweighted: with σ per observation or a full Σ, the sandwich is the SD of the unweighted fit,
H⁻¹J₀ᵀΣJ₀H⁻¹, and `:biased` the bound a weighted fit would reach, (JᵀΣ⁻¹J)⁻¹. Rician noise,
bounds hit by noisy fits and nonlinearity at low SNR are not modelled: simulate and fit for
those.

Each step is a function of its own, for use alone or to check this one: [`noisecovariance`](@ref),
[`noisefree`](@ref), [`bias`](@ref), [`jacobian`](@ref), [`hessian`](@ref),
[`informationmatrix`](@ref), [`biasgradient`](@ref), [`sandwich`](@ref), [`biasedcrlb`](@ref).

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)                          # kmf = 5 in this tissue
c = crlb(m, θ; snr = 200)                          # the fit fixes kmf at 12.5: biased
c.bias.PSR, c.sd.PSR, c.relrmse.PSR
crlb(m, θ; snr = 200, fit = (kmf = 5.0,))          # kmf fixed at the truth: unbiased
crlb.(Ref(m), [nominal(m, t) for t in (:SIR_WM_I, :SIR_GM)]; snr = 200)   # several tissues
crlb(m, θ; snr = 200, vary = (kmf = 5:2.5:20,))    # the bias against the true kmf (a Sweep)
crlb(m, θ; snr = 200, fits = (fixed = (;), free = (kmf = Free(),)))   # side by side
θb = (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)       # bi-exponential data,
crlb(T2Decay(TE = te), (S0 = 1.0, T2 = 0.08); snr = 100,           # fitted mono-exponentially
     simulate_with = T2BiexpDecay(TE = te) => θb)
```
"""
function crlb(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
              fit::NamedTuple = (;), cov::Symbol = :sandwich, vary::NamedTuple = (;),
              fits::NamedTuple = (;), simulate_with = nothing)
    _check_fits(fit, fits)
    _check_simulate_with(simulate_with)
    # A sweep: one call per setting (sweep.jl), each with one noise level, one truth and one set
    # of fit settings.
    if _issweep(snr, vary, fits)
        settings = _sweep_settings(_datamodel(m, simulate_with), snr, vary, fits)
        results = [crlb(m, _truth_of(st, truth, simulate_with); snr = _snr_of(st, snr), σ, Σ,
                        fit = _fit_of(st, fit, fits), cov,
                        simulate_with = _data_of(st, simulate_with)) for st in settings]
        return Sweep(settings, results, _rows(settings, results))
    end
    # `fit` is a keyword here, so inside this function the name means the settings, not the
    # function `fit` (the noise-free fit is `noisefree`).
    cov in (:sandwich, :biased) ||
        throw(ArgumentError("`cov` must be :sandwich or :biased, got $(repr(cov))"))
    _check_fit_settings(fit)
    simulate_with === nothing || cov === :sandwich || throw(ArgumentError("with data from " *
        "another model (`simulate_with`), the sandwich only: Eq. 3 needs the true model's " *
        "derivatives with respect to the fitted model's parameters"))
    # The data: from `m` at `truth`, or from another model (`simulate_with = m2 => θ2`), with
    # `truth` then the reference values the fit is judged against.
    dm, dt = _datasource(m, truth, simulate_with)

    # ① The noise, the noise-free data and their fit θ̂ (started at the truth), and the bias.
    #    `snr` is relative to the data's scale parameter.
    Σn = noisecovariance(dm, dt; snr, σ, Σ)           # N×N
    y = signal(dm, dt)                                 # g_true
    r = noisefree(m, truth; fit, simulate_with)        # θ̂ = r.params
    b = bias(r, truth)
    F = keys(r.params)                                 # the free parameters, model order
    P = length(F)
    # A free parameter on a bound in the noise-free fit is held there (SD 0); the covariance is
    # over the others, U.
    onbound = _onbound(r)
    U = findall(n -> !(n in onbound), collect(F))

    # ② The derivatives at θ̂: J₀ (N×P) and the Hessian of the fit's own cost ½‖g − y‖², the
    #    second term from the residual r₀ = g(θ̂) − g_true: H = J₀ᵀJ₀ + Σᵢ r₀ᵢ∇²gᵢ (P×P).
    J₀ = jacobian(r)
    H = hessian(r, y)
    J₀u, Hu = J₀[:, U], H[U, U]

    # ③ The covariance, if the data can determine the parameters (fewer observations than
    #    parameters to estimate, or nearly dependent columns of J₀: Inf, see `_identifiable`; a
    #    failed Cholesky factorization of H too: no negative variance hidden by abs(), as in the
    #    MATLAB code).
    C = zeros(P, P)
    if !(_identifiable(J₀u' * J₀u, length(y)) && isposdef(Hu))
        C[U, U] .= Inf
    elseif cov === :sandwich
        C[U, U] = sandwich(Hu, J₀u, Σn)                # H⁻¹J₀ᵀΣJ₀H⁻¹
    else
        # Eq. 3: F, the Fisher information of the true signal at the truth, J_tᵀΣ⁻¹J_t (the
        # truth's own ties followed: R1m = R1f moves with R1f, as the MATLAB's `R1 + eps` on
        # both), and ∇b (rows of parameters on a bound: ∂θ̂/∂θ = 0).
        Ft = informationmatrix(m, truth; Σ = Σn, fit)
        if _identifiable(Ft, length(y)) && isposdef(Ft)
            C = biasedcrlb(biasgradient(r, truth), Ft)
        else
            C[U, U] .= Inf
        end
    end

    # ④ Packaging: named tuples by quantity, then parameter.
    θt = resolve(m, truth)
    truthF = [Float64(θt[n]) for n in F]
    sd = sqrt.(diag(C))
    rmse = sqrt.(collect(b) .^ 2 .+ sd .^ 2)
    rel(x) = x ./ abs.(truthF)
    nt(x) = NamedTuple{F}(Tuple(x))
    condJ = isempty(U) ? Inf : _condition(J₀u, truthF[U])
    # The information matrices at θ̂, over every free parameter: the Fisher (expected) J₀ᵀΣ⁻¹J₀,
    # and the observed, the Hessian of the negative log-likelihood of the noise-free data; the
    # two agree when the fit is exact (r₀ = 0). With one σ, the observed one is H/σ².
    info = _sym(J₀' * (Σn \ J₀))
    observed = hessian(r, y; Σ = Σn)
    return CRLBResult(r.params, b, nt(sd), nt(rmse), nt(rel(collect(b))), nt(rel(sd)),
                      nt(rel(rmse)), C, condJ, onbound, r.status, cov, sqrt.(diag(Σn)), truth,
                      r.fixed, m, simulate_with, Σ === nothing ? nothing : Matrix{Float64}(Σ),
                      info, observed)
end

# -------------------------------------------------------------------------------------------------
# StatsAPI's names (as for `fit` and `nobs`): the information matrix, the covariance, the SDs
# -------------------------------------------------------------------------------------------------

"""
    informationmatrix(c::CRLBResult; expected = true) -> Matrix

The information matrix of the fit's free parameters (in model order, `keys(c.fit)`) at the
noise-free fit's values θ̂ (the truth when nothing fixed is wrong), for the noise given to
[`crlb`](@ref). `expected = true`: the Fisher information J₀ᵀΣ⁻¹J₀ (J₀ the Jacobian at θ̂, Σ
the noise covariance; JᵀJ/σ² for one σ). `expected = false`: the observed information, the
Hessian of the negative log-likelihood of the noise-free data at θ̂, which adds the residual's
curvature when an assumption is wrong (with one σ: H/σ²). When nothing fixed is wrong, the
inverse of either is the classical CRLB. At other values, `informationmatrix(m, θ; ...)`.

The sandwich is built from them: `vcov(c) ≈ inv(I_obs) * I * inv(I_obs)` with one σ
(`I = informationmatrix(c)`, `I_obs = informationmatrix(c; expected = false)`).
"""
informationmatrix(c::CRLBResult; expected::Bool = true) = expected ? c.information : c.observed

"""
    vcov(c::CRLBResult) -> Matrix
    stderror(c::CRLBResult) -> Vector

The predicted covariance matrix of the fit's free parameters (`c.cov`: the sandwich, or Eq. 3
with `cov = :biased`) and their SDs (`c.sd` as a vector), in model order. StatsAPI's functions.
"""
vcov(c::CRLBResult) = c.cov
stderror(c::CRLBResult) = collect(values(c.sd))

# -------------------------------------------------------------------------------------------------
# Display: at the REPL a table, one row per free parameter, in the style of `compare`
# -------------------------------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", c::CRLBResult)
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    σ = c.σ
    noise = _noisetext(σ, c.Σ)
    how = c.method === :sandwich ? "sandwich (the least-squares fit)" :
                                   "biased CRLB (Dortch 2018 Eq. 3)"
    print(io, "crlb for ", nameof(typeof(c.model)), ", ", length(σ), " observations, ", noise,
          ", covariance: ", how)
    _print_data(io, c.data)
    # The fit's fixed values, with the truth where it differs: the source of any bias.
    _print_fixed(io, c.fixed, c.truth)
    isconverged(c.status) ||
        print(io, "\nthe noise-free fit did not converge ($(c.status)): the bias may be inaccurate")
    isempty(c.onbound) ||
        print(io, "\non a bound in the noise-free fit (held there, SD 0): ", join(c.onbound, ", "))
    header = ("parameter", "true", "noise-free fit", "bias", "rel bias", "SD", "rel SD", "RMSE",
              "rel RMSE")
    rows = [(string(n), f(c.fit[n] - c.bias[n]), f(c.fit[n]), f(c.bias[n]), pct(c.relbias[n]),
             f(c.sd[n]), pct(c.relsd[n]), f(c.rmse[n]), pct(c.relrmse[n])) for n in keys(c.fit)]
    w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for r in rows
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end

# The fit's fixed values, with the truth where it differs: the source of any bias. Shared with
# `montecarlo`'s display and the sweeps' (one line per fit setting, `label`).
function _print_fixed(io::IO, fixed::NamedTuple, truth::NamedTuple, label = "fixed in the fit")
    isempty(fixed) && return
    item(n, x) = (t = truth[n]; s = "$n = $(_showvalue(x))";
                  x isa Real && t isa Real && x != t ? s * " (true $(_showvalue(t)))" : s)
    print(io, "\n", label, ": ", join((item(n, x) for (n, x) in pairs(fixed)), ", "))
end

# The noise, in words: one σ, σ per observation (its range), or a covariance matrix.
function _noisetext(σ, Σ)
    f(x) = string(round(x; sigdigits = 4))
    range = all(==(σ[1]), σ) ? "σ = $(f(σ[1]))" : "σ = $(f(minimum(σ))) to $(f(maximum(σ)))"
    return Σ === nothing ? range : "$range, correlated (Σ given)"
end

# Where the data came from, when not from the fitted model (`simulate_with`).
_print_data(io::IO, ::Nothing) = nothing
_print_data(io::IO, d::Pair) =
    print(io, "\ndata from ", nameof(typeof(d.first)), ": ",
          join(("$n = $(_showvalue(x))" for (n, x) in pairs(d.second)), ", "))

# Checks shared with `montecarlo`.
_check_fits(fit, fits) = isempty(fit) || isempty(fits) || throw(ArgumentError("give `fit` " *
    "(one set of fit settings) or `fits` (several, side by side, by name), not both"))

Base.show(io::IO, c::CRLBResult) =
    print(io, "CRLBResult(", nameof(typeof(c.model)), ": ", join(keys(c.fit), ", "), ")")
