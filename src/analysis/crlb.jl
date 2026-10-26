# =================================================================================================
# crlb: the bias, SD and RMSE of a fit, predicted from derivatives (no noisy fits)
#
#   truth θ ──────▶ g_true = signal(m, θ)                 the noise-free data
#   fit settings ─▶ free and fixed parameters, bounds     as `fit` uses them (CRLBCache, once)
#   ① noise-free fit  θ̂ = fit of g_true, started at the truth;  b = θ̂ − θ;  r₀ = g(θ̂) − g_true
#   ② derivatives     J₀ = ∂g/∂θ at θ̂ (dual numbers, ties followed);  H = J₀ᵀJ₀ + Σᵢ r₀ᵢ∇²gᵢ
#   ③ covariance      :sandwich  H⁻¹ J₀ᵀΣJ₀ H⁻¹                       (default)
#                     :biased    (I+∇b) F⁻¹ (I+∇b)ᵀ,  I+∇b = H⁻¹J₀ᵀJ_t,  F = J_tᵀΣ⁻¹J_t
#   ④ report          SD = √diag, RMSE = √(b² + SD²), absolute and relative to the truth
#
# Each step is a function of its own (derivatives.jl: `noisefree!`, `hessian!`, `meat`,
# `information`, `crossgram`, `sandwich`, `biasgradient`, `biasedcrlb`), and the kernel
# `_crlb!` below is their composition: static P×P matrices, no allocation, 1–16 µs a call, so
# the protocol optimizer can call it millions of times. `crlb!(cache, θ; noise)` adds the checks
# and the result; `crlb(m, θ; ...)` makes the cache first (most of a one-off call's time).
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
# (implicit function theorem) gives ∂θ̂/∂θ = −H⁻¹·∂G/∂θ = H⁻¹J₀ᵀJ_t, one linear solve.
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
    crlb(m, truth; snr or σ or Σ, cov = :sandwich, vary, fits, simulate_with, settings...)
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
- The fit's settings: keywords named after the parameters, as for [`fit`](@ref)`(m, y; ...)`;
  the model's own roles (`parameters(m)`) for the rest. The prediction is for a fit with these
  settings: `kmf = Free()` fits kmf, `kmf = 15.0` fixes it at 15. The bias is zero unless a
  fixed value differs from the truth; the classical CRLB with kmf fixed at its true value is
  `kmf = 5.0`. Bounds are the fit's. The truth stays `truth`: to change it, change `truth`
  (`nominal(m, :SIR_WM_I; kmf = 10.0)`).
- `cov`: the covariance, `:sandwich` (default) or `:biased`. The sandwich is the covariance of
  the least-squares fit itself (what EQUIP's fit achieves; the misspecified CRB); `:biased` is
  Eq. 3 of Dortch et al., MRM 2018, a bound for any estimator with the same bias gradient. Both
  are the classical CRLB σ²(JᵀJ)⁻¹ when nothing fixed is wrong, and equal each other when the
  fit is exact (as many observations as free parameters); otherwise `:biased` ≤ `:sandwich`.
- Sweeps: `snr = [50, 100, 200]` (several noise levels), `vary = :kmf => 5:2.5:20` (several
  true values of a parameter; several parameters as a vector of pairs, every combination; the
  fit's settings stay as they are, so this is the bias of a
  wrong assumption against the truth, Dortch 2018 Fig. 2B). Every combination is computed, the
  first keyword varying slowest; the result is a [`Sweep`](@ref), with a table in `s.rows`.
- Fit settings side by side: `fits = (fixed = (;), free = (; kmf = Free()))`, named; one more
  axis of the sweep (the last, so a table has the settings of one case next to each other).
  Instead of settings given as keywords.
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

For many truths or noise levels with the same fit settings, make the setup once:
`c = CRLBCache(m; cov, settings...)`, then [`crlb!`](@ref)`(c, θ; snr)` (sweeps do this
themselves).
Each step is a function of its own, for use alone or to check this one: [`noisecovariance`](@ref),
[`noisefree`](@ref), [`bias`](@ref), [`jacobian`](@ref), [`hessian`](@ref),
[`informationmatrix`](@ref), [`biasgradient`](@ref), [`sandwich`](@ref), [`biasedcrlb`](@ref).

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)                          # kmf = 5 in this tissue
c = crlb(m, θ; snr = 200)                          # the fit fixes kmf at 12.5: biased
c.bias.PSR, c.sd.PSR, c.relrmse.PSR
crlb(m, θ; snr = 200, kmf = 5.0)                  # kmf fixed at the truth: unbiased
crlb(m, θ; snr = 200, kmf = Free())                # kmf fitted: unbiased, larger SD
crlb.(Ref(m), [nominal(m, t) for t in (:SIR_WM_I, :SIR_GM)]; snr = 200)   # several tissues
crlb(m, θ; snr = 200, vary = :kmf => 5:2.5:20)     # the bias against the true kmf (a Sweep)
crlb(m, θ; snr = 200, fits = (fixed = (;), free = (; kmf = Free())))  # side by side
θb = (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)       # bi-exponential data,
crlb(T2Decay(TE = te), (S0 = 1.0, T2 = 0.08); snr = 100,           # fitted mono-exponentially
     simulate_with = T2BiexpDecay(TE = te) => θb)
```
"""
function crlb(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
              cov::Symbol = :sandwich, vary = (;), fits::NamedTuple = (;),
              simulate_with = nothing, settings...)
    # The fit's settings: every keyword that is not one of crlb's own, as for `fit`.
    fit = _fitsettings("crlb", m, settings, (:snr, :σ, :Σ, :cov, :vary, :fits, :simulate_with))
    vary = _varyaxes(vary)
    _check_fits(fit, fits)
    _check_simulate_with(simulate_with)
    if _issweep(snr, vary, fits)
        # A sweep (sweep.jl): one cache per fit setting, reused for every noise level and truth.
        settings = _sweep_settings(_datamodel(m, simulate_with), snr, vary, fits)
        caches = isempty(fits) ? (_CRLBCache(m, fit, cov),) : map(f -> _CRLBCache(m, f, cov), fits)
        cacheof(st) = haskey(st, :fit) ? caches[st.fit] : caches[1]
        results = [crlb!(cacheof(st), _truth_of(st, truth, simulate_with); snr = _snr_of(st, snr),
                         σ, Σ, simulate_with = _data_of(st, simulate_with)) for st in settings]
        return Sweep(settings, results, _rows(settings, results))
    end
    return crlb!(_CRLBCache(m, fit, cov), truth; snr, σ, Σ, simulate_with)
end

"""
    crlb!(c::CRLBCache, truth; snr or σ or Σ, simulate_with) -> CRLBResult

[`crlb`](@ref) with the fit settings and covariance already in a [`CRLBCache`](@ref): for many
truths or noise levels with the same fit settings, the setup is made once.

```julia
c = CRLBCache(m)                                   # the model's own fit settings
cs = [crlb!(c, nominal(m, t); snr = 200) for t in (:SIR_WM_I, :SIR_WM_II, :SIR_GM)]
```
"""
function crlb!(c::CRLBCache, truth::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
               simulate_with = nothing)
    _check_simulate_with(simulate_with)
    simulate_with === nothing || !c.biased || throw(ArgumentError("with data from " *
        "another model (`simulate_with`), the sandwich only: Eq. 3 needs the true model's " *
        "derivatives with respect to the fitted model's parameters"))
    # The data: from the model at `truth`, or from another model (`simulate_with = m2 => θ2`),
    # with `truth` then the reference values the fit is judged against; `snr` is relative to
    # the data's scale parameter.
    dm, dt = _datasource(c.model, truth, simulate_with)
    Σn = noisecovariance(dm, dt; snr, σ, Σ)
    return _crlbresult(c, truth, dm, dt, Σn, Σ, simulate_with)
end

"""
    _crlb!(c, m, t, σ²[, dm, dt]) -> (; fit, truth, bias, cov, status, pinned, exact, H, G, ∇b)

The kernel of [`crlb`](@ref), without allocation: the noise-free fit's values θ̂, the true free
values, the bias, the covariance (an `SMatrix`; `Inf` for parameters the data cannot determine,
0 for those on a bound), the noise-free fit's status, which parameters are on a bound (held
there), whether the fit was exact, H and J₀ᵀJ₀ at θ̂, and ∇b (Eq. 3 only; `NaN` otherwise). `m`
may be another protocol than the cache's, with as many observations and Float64 acquisition
values (static vectors work), as a protocol optimizer needs. `t` from `_crlb_truth`, `σ²` a
number, a vector (one variance per observation) or a `_FullNoise` (`_kernelnoise`). With `dm`
and `dt` the data come from another model (the sandwich only).
"""
_crlb!(c::CRLBCache, m::MRIModel, t::NamedTuple, σ2) = _crlb!(c, m, t, σ2, m, t)
function _crlb!(c::CRLBCache{F,P,T}, m::MRIModel, t::NamedTuple, σ2, dm::MRIModel,
                dt::NamedTuple) where {F,P,T}
    # ① The noise-free fit θ̂, from the truth's free values θt. A parameter it leaves on a bound
    #    is held there: masked by u below (u = 0 for it, 1 for the others).
    θ̂, θt, r = noisefree!(c, m, t, dm, dt)
    pinned = map(_near, θ̂, c.lower) .| map(_near, θ̂, c.upper)
    u = map(p -> ifelse(p, zero(T), one(T)), pinned)

    # ② The Hessian of the fit's cost at θ̂ (and J₀ᵀJ₀). When the fit is exact the residual's
    #    curvature is 0 and first-order duals suffice. Held parameters: rows and columns masked,
    #    1 on the diagonal, so H⁻¹ leaves them be.
    exact = r.sse <= _EXACT^2 * sum(abs2, c.y)
    H, G = hessian!(c, m, θ̂, exact)
    Hm = (u * u') .* H + Diagonal(one(T) .- u)
    nan = fill(T(NaN), SMatrix{P,P,T})
    out(C, ∇b) = (fit = θ̂, truth = θt, bias = θ̂ - θt, cov = C, status = r.status, pinned,
                  exact, H, G, ∇b)

    # ③ Can the data determine the parameters? Fewer observations than parameters to estimate,
    #    nearly dependent columns of J₀ (`_determined`), or H not positive definite: Inf (no
    #    negative variance hidden by abs(), as in the MATLAB code).
    N = length(c.y)
    _, okH = _cholesky(Hm)
    (N >= count(!, pinned) && _determined(G, u) && okH) || return out(_infcov(u), nan)

    # ④ The covariance. The sandwich, H⁻¹(J₀ᵀΣJ₀)H⁻¹ ...
    if !c.biased
        M = exact ? meat(c.jbuf, σ2, u) : meat(c.hbuf, σ2, u)
        return out(sandwich(Hm, M), nan)
    end
    #    ... or Eq. 3: J_t (the true signal's Jacobian at the truth, in `c.fitbuf`), the bias
    #    gradient ∇b = H⁻¹J₀ᵀJ_t − I, and F = J_tᵀΣ⁻¹J_t.
    attruth!(c.fitbuf, c, m, t, θt)
    K = exact ? crossgram(c.jbuf, c.fitbuf, u) : crossgram(c.hbuf, c.fitbuf, u)
    ∇b = biasgradient(Hm, K)
    Fm = information(c.fitbuf, σ2)
    _, okF = _cholesky(Fm)
    (N >= P && _determined(Fm, ones(SVector{P,T})) && okF) || return out(_infcov(u), ∇b)
    return out(biasedcrlb(∇b, Fm), ∇b)
end

# The kernel, then its numbers as a CRLBResult: named tuples by quantity, then parameter. A
# separate function (a "function barrier") so that F and P are known types here.
function _crlbresult(c::CRLBCache{F,P,T}, truth, dm, dt, Σn, Σ, simulate_with) where {F,P,T}
    σ2 = _kernelnoise(Σn, P)
    t = _crlb_truth(c.model, truth)
    dmT, dtt = simulate_with === nothing ? (c.mT, t) : (_with_precision(dm, Float64), _crlb_truth(dm, dt))
    k = _crlb!(c, c.mT, t, σ2, dmT, dtt)
    sd = sqrt.(diag(k.cov))
    rmse = sqrt.(k.bias .^ 2 .+ sd .^ 2)
    nt(x) = NamedTuple{F}(Tuple(x))
    rel(x) = x ./ abs.(k.truth)
    # The information matrices at θ̂, over every free parameter: the Fisher (expected)
    # J₀ᵀΣ⁻¹J₀, and the observed, the Hessian of the negative log-likelihood of the noise-free
    # data, which adds Σᵢ wᵢ∇²gᵢ, w = Σ⁻¹r₀; they agree when the fit is exact (r₀ = 0). With one
    # σ, the observed one is H/σ².
    rows = k.exact ? c.jbuf : c.hbuf
    info = Matrix(information(rows, σ2))
    if k.exact
        observed = copy(info)
    else
        w = Σn \ ([_val(d) for d in c.hbuf] .- c.y)          # Σ⁻¹r₀
        observed = Matrix(_sym(info + curvature(c.hbuf, w)))
    end
    onbound = Symbol[F[j] for j in 1:P if k.pinned[j]]
    return CRLBResult(nt(k.fit), nt(k.bias), nt(sd), nt(rmse), nt(rel(k.bias)), nt(rel(sd)),
                      nt(rel(rmse)), Matrix(k.cov), _condition(k.G, k.truth, k.pinned, length(c.y)),
                      onbound, k.status, c.biased ? :biased : :sandwich, sqrt.(diag(Σn)), truth,
                      map(_untie, c.fixed), c.model, simulate_with,
                      Σ === nothing ? nothing : Matrix{Float64}(Σ), info, observed)
end

# The condition number of J₀ with each column scaled by its parameter's true value (relative
# sensitivities, so the units drop out; 1 for a true value of 0), over the parameters not on a
# bound, reported as `c.cond`: √(λmax/λmin) of the scaled J₀ᵀJ₀, P×P (no SVD of the N×P J₀). Inf
# when there are fewer observations than parameters, or J₀ᵀJ₀ is singular.
function _condition(G::SMatrix{P,P}, θ, pinned, N) where {P}
    U = [j for j in 1:P if !pinned[j]]
    (isempty(U) || N < length(U)) && return Inf
    s = [θ[j] == 0 ? 1.0 : abs(θ[j]) for j in U]
    λ = eigvals(Symmetric(s .* Matrix(G)[U, U] .* s'))
    return λ[1] > 0 ? sqrt(λ[end] / λ[1]) : Inf
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
_check_fits(fit, fits) = isempty(fit) || isempty(fits) || throw(ArgumentError("give the fit's " *
    "settings as keywords (one set: `kmf = 5.0`) or `fits` (several, side by side, by name), " *
    "not both"))

Base.show(io::IO, c::CRLBResult) =
    print(io, "CRLBResult(", nameof(typeof(c.model)), ": ", join(keys(c.fit), ", "), ")")
