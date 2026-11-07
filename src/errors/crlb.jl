# =================================================================================================
# crlb: the bias, SD and RMSE of a fit, predicted from derivatives (no noisy fits)
#
#   truth θ ──────▶ g_true = signal(m, θ)                 the noise-free data
#   the fit ──────▶ free and fixed parameters, bounds     a Scenario's (CRLBCache, once)
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
# and the result; `crlb(m, s; ...)` makes the cache first (most of a one-off call's time), and
# `crlb(m, θ; changes...)` the scenario before that (scenario.jl: the fit built from the truth,
# fixed parameters at their true values, so it is unbiased unless a change makes it wrong).
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
like `res.params.PSR` of a [`FitResult`](@ref)), then the quantities the model derives from them
([`derived`](@ref): DTI's `c.sd.FA`, DKI's `c.sd.MK`):

- `fit`: the noise-free fit's values θ̂ (the "pseudo-true" values, where the noisy fits center)
- `bias` (θ̂ − truth), `sd`, `rmse` (√(bias² + SD²)), in the parameters' units
- `relbias`, `relsd`, `relrmse`: the same divided by |truth|

and `cov` (the covariance matrix of the free parameters in model order, the derived quantities
not in it), `onbound` (free parameters
on a bound in the noise-free fit: held there, SD 0), `status` (the noise-free fit's
[`FitStatus`](@ref)), `method` (`:sandwich` or `:biased`), `σ` (the noise SD per observation,
σ/√NEX with averages), `truth`, `fixed` (the fit's fixed values), `model`, `data` (`nothing`,
or the `model => truth` the data came from, with `simulate_with`), `Σ` (`nothing`, or the
noise covariance matrix given), `averages` (`nothing`, or the number of averages of each
observation) and `weights` (`nothing`, or the fit's weights). [`informationmatrix`](@ref)`(c)`, [`vcov`](@ref)`(c)` and [`stderror`](@ref)`(c)` give
the information and covariance matrices and the SDs, under StatsAPI's names.

Diagnostics, computed when called (as `fitted(res)` is for a fit): [`cond`](@ref)`(c)`, how hard
the data make the parameters to tell apart, and [`cor`](@ref)`(c)`, the predicted correlations
of the estimates. For a sweep, `cond.(s.results)`.
"""
struct CRLBResult{M<:MRIModel,N<:NamedTuple,Tr<:NamedTuple,X<:NamedTuple,D,S,G}
    fit::N
    bias::N
    sd::N
    rmse::N
    relbias::N
    relsd::N
    relrmse::N
    cov::Matrix{Float64}
    gram::G                    # J₀ᵀJ₀ at θ̂, a static P×P matrix: what `cond(c)` is computed from
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
    averages::Union{Nothing,Vector{Int}}   # nothing, or the averages of each observation
    weights::Union{Nothing,Vector{Float64}}   # nothing, or the fit's weights (a weighted fit)
end

"""
    crlb(m, s::Scenario; snr or σ or Σ, averages, cov = :sandwich, vary, simulate_with) -> CRLBResult or Sweep
    crlb(m, θ; snr or σ or Σ, averages, cov, vary, simulate_with, changes...)
    crlb(m, (name1 = s1, name2 = s2, ...); ...) -> Sweep
    crlb(m, set; snr or σ or Σ, averages, cov) -> Sweep          (a set from `scenarios`)

The bias, SD and RMSE of each fitted parameter predicted for one voxel, from derivatives: the
Cramér–Rao lower bound (CRLB, also CRB), biased when the fit fixes a parameter at a value other
than the truth's. No noisy fits: the bias comes from one fit of the noise-free data, and the
SD from the derivatives at that fit's answer, first order in the noise.

- `m`: the protocol, a model with its acquisition, e.g. `QMTSIR(; SIR_4POINT...)`.
- `s`: a [`Scenario`](@ref), the truth and the fit judged against it, e.g.
  `Scenario(QMTSIR, nominal(QMTSIR, :SIR_WM_I); kmf = 12.5)`. Or the truth θ (every parameter,
  numbers or ties, as for [`simulate`](@ref)) with the changes as keywords, as for
  [`fit`](@ref), which make `Scenario(m, θ; changes...)` inside: the fit holds the fixed
  parameters at their TRUE values (unbiased); `kmf = 12.5` fixes kmf at 12.5 (wrong for a tissue
  with kmf = 5: biased); `kmf = Free()` estimates it. Bounds are the fit's. The noise-free fit
  starts at the truth, whatever the scenario's `starts`.
- The noise (Gaussian), one of: `snr` (relative to the model's scale parameter, as for
  `simulate`: σ = M0/snr); `σ`, a number or one SD per observation (independent noise, e.g.
  different bandwidths or numbers of averages); `Σ`, the full covariance matrix of the noise
  between observations (correlated noise, e.g. after denoising or a joint reconstruction).
  Chosen by type, so the common case, one σ for every observation ("pulled out" of
  JᵀΣ⁻¹J = JᵀJ/σ²), costs nothing extra.
- `averages`: the number of averages (NEX) of each observation, or one for all, with `snr` or
  `σ` given for one average: σⱼ = σ/√NEXⱼ, as [`simulate`](@ref) makes the data. To first order
  in the noise this holds however the averages were combined, so crlb takes no `averaging`;
  the noise floor that averaging magnitude images keeps is [`montecarlo`](@ref)'s to measure.
- `cov`: the covariance, `:sandwich` (default) or `:biased`. The sandwich is the covariance of
  the least-squares fit itself (what EQUIP's fit achieves; the misspecified CRB); `:biased` is
  Eq. 3 of Dortch et al., MRM 2018, a bound for any estimator with the same bias gradient. Both
  are the classical CRLB σ²(JᵀJ)⁻¹ when nothing fixed is wrong, and equal each other when the
  fit is exact (as many observations as free parameters); otherwise `:biased` ≤ `:sandwich`.
- Sweeps: `snr = [50, 100, 200]` (several noise levels), `vary = :kmf => 5:2.5:20` (several
  true values of a parameter; several parameters as a vector of pairs, every combination). The
  fit stays the scenario's, so with `kmf = 12.5` this is the bias of that assumption against the
  true kmf (Dortch 2018 Fig. 2B). Every combination is computed, the first keyword varying
  slowest; the result is a [`Sweep`](@ref), with a table in `s.rows`.
- Fits side by side: scenarios by name, `crlb(m, (fixed = Scenario(m, θ), free = Scenario(m, θ;
  kmf = Free())); snr = 200)`: one more axis of the sweep (the last, so a table has the fits of
  one case next to each other).
- A set of scenarios from [`scenarios`](@ref) (tissues, a box of true values, or both): one
  result per scenario, the set's settings as the sweep's (`tissue`, the varied values), after
  the noise levels when `snr` is a vector: `crlb(m, scenarios(QMTSIR, tissues; kmf = 12.5);
  snr = 200)`. Its truths are the set's (`vary` and `simulate_with` are not taken).
- The model's derived quantities ([`derived`](@ref): DTI's MD, FA, AD, RD and V1, DKI's MK,
  AK, RK and MKT), after the parameters in each named tuple: g(θ̂) the noise-free fit's value,
  g(θ̂) − g(truth) the bias, the SD by the delta method, √(∇g·C·∇gᵀ) with C the parameters'
  covariance and ∇g from dual numbers through `derived`. First order in the noise, as the
  parameters' (the noise's own bias of a nonlinear g, e.g. FA's at low SNR, is not in it:
  [`montecarlo`](@ref) measures it). Where `derived` has no derivative, NaN: DTI's AD, RD and
  V1, and DKI's AK and RK, where λ1 = λ2 (the principal direction is not defined); DKI's MK and
  RK where two of D's eigenvalues are equal and W is not symmetric about the third axis (the
  truths `nominal` makes from derived quantities are symmetric: finite there).
- Data from another model: `simulate_with = m2 => θ2` (a model with as many observations, and
  every value of its parameters). The noise-free fit of `m` to `m2`'s signal gives the bias,
  against the scenario's truth, your reference values for `m`'s parameters (e.g. the long T2 of
  a bi-exponential tissue fitted mono-exponentially); `snr` is relative to `m2`'s scale
  parameter; `vary` changes `θ2`. The sandwich only: Eq. 3 needs the true model's derivatives
  with respect to `m`'s parameters.

A free parameter that ends on a bound in the noise-free fit is held there (SD 0; `c.onbound`).
Parameters the data cannot determine (a singular Jacobian) get SD `Inf`. The fit is the
scenario's, weighted or not (`weights = nex`, as for `fit`): with σ per observation or a full Σ,
the sandwich is the SD of the unweighted fit, H⁻¹J₀ᵀΣJ₀H⁻¹, or of the weighted one,
H⁻¹J₀ᵀWΣWJ₀H⁻¹ (H with the weights); with W = Σ⁻¹ and nothing fixed wrong, that is the bound
(JᵀΣ⁻¹J)⁻¹, which `:biased` gives for any fit (the weights change its ∇b, not F). Rician noise,
bounds hit by noisy fits and nonlinearity at low SNR are not modelled: simulate and fit for
those ([`montecarlo`](@ref)).

For many noise levels or truths with the same fit, make the setup once:
`c = CRLBCache(m, s; cov)`, then [`crlb!`](@ref)`(c, θ; snr)` (sweeps do this themselves).
Each step is a function of its own, for use alone or to check this one: [`noisecovariance`](@ref),
[`noisefree`](@ref), [`bias`](@ref), [`jacobian`](@ref), [`hessian`](@ref),
[`informationmatrix`](@ref), [`biasgradient`](@ref), [`sandwich`](@ref), [`biasedcrlb`](@ref).

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)                          # kmf = 5 in this tissue
crlb(m, θ; snr = 200)                              # kmf held at its true value: unbiased
c = crlb(m, θ; snr = 200, kmf = 12.5)             # the fit assumes kmf = 12.5: biased
c.bias.PSR, c.sd.PSR, c.relrmse.PSR
s = Scenario(QMTSIR, θ; kmf = 12.5)                # the same, made once,
crlb(m, s; snr = 200)                              # and judged under any protocol
crlb(QMTSIR(; SIR_5POINT...), θ; snr = 200, kmf = Free())   # kmf fitted: unbiased, larger SD
crlb.(Ref(m), [nominal(m, t) for t in (:SIR_WM_I, :SIR_GM)]; snr = 200)   # several tissues
crlb(m, θ; snr = 200, kmf = 12.5, vary = :kmf => 5:2.5:20)  # the bias against the true kmf
crlb(m, (assumed = s, exact = Scenario(m, θ)); snr = 200)   # side by side
θb = (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)       # bi-exponential data,
crlb(T2Decay(TE = te), (S0 = 1.0, T2 = 0.08); snr = 100,           # fitted mono-exponentially
     simulate_with = T2BiexpDecay(TE = te) => θb)
```
"""
function crlb(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
              averages = nothing, cov::Symbol = :sandwich, vary = (;), simulate_with = nothing,
              changes...)
    # The scenario from the truth and the changes: every keyword that is not one of crlb's own,
    # as for `fit`.
    ch = _fitsettings("crlb", m, changes, (:snr, :σ, :Σ, :averages, :cov, :vary, :simulate_with))
    return crlb(m, Scenario(m, θ; ch...); snr, σ, Σ, averages, cov, vary, simulate_with)
end
function crlb(m::MRIModel, s::Union{Scenario,_Scenarios}; snr = nothing, σ = nothing,
              Σ = nothing, averages = nothing, cov::Symbol = :sandwich, vary = (;),
              simulate_with = nothing, noise = nothing, averaging = nothing)
    # `noise` and `averaging` are taken only to say why they are refused: crlb assumes Gaussian
    # noise, first order, where averages give σ/√NEX however they were combined.
    noise === nothing || throw(_noise_refused("crlb"))
    averaging === nothing || throw(_noise_refused("crlb", :averaging))
    foreach(x -> _check_protocol(m, x), _scenarios(s))
    vary = _varyaxes(vary)
    _check_simulate_with(simulate_with)
    fits = s isa Scenario ? (;) : s                # side by side: one more axis, by name
    if _issweep(snr, vary, fits)
        # A sweep (sweep.jl): one cache per scenario, reused for every noise level and truth.
        settings = _sweep_settings(_datamodel(m, simulate_with), snr, vary, fits)
        caches = map(x -> _CRLBCache(m, x, cov), s isa Scenario ? (one = s,) : s)
        cacheof(st) = haskey(st, :fit) ? caches[st.fit] : caches.one
        # Numbers varied in the fitted model's truth: each setting's truth needs no second check
        # (`_crlb_varied!`); otherwise (a tie varied, data from another model) `crlb!` checks it.
        numbers = simulate_with === nothing && all(v -> all(x -> x isa Real, v), values(vary))
        resultof(st) = numbers ?
            _crlb_varied!(cacheof(st), _truth_of(st, _scenario_of(st, s).truth, nothing);
                          snr = _snr_of(st, snr), σ, Σ, averages) :
            crlb!(cacheof(st), _truth_of(st, _scenario_of(st, s).truth, simulate_with);
                  snr = _snr_of(st, snr), σ, Σ, averages,
                  simulate_with = _data_of(st, simulate_with))
        results = [resultof(st) for st in settings]
        return Sweep(settings, results, _rows(settings, results))
    end
    return crlb!(_CRLBCache(m, s, cov), s.truth; snr, σ, Σ, averages, simulate_with)
end

"""
    crlb!(c::CRLBCache, truth; snr or σ or Σ, averages, simulate_with) -> CRLBResult

[`crlb`](@ref) with the fit and the covariance already in a [`CRLBCache`](@ref): for many noise
levels or truths with the same fit, the setup is made once. The fit stays the cache's: with a
truth other than its scenario's, that same fit is judged against it.

```julia
c = CRLBCache(m, nominal(m, :SIR_WM_I); kmf = 12.5)
cs = [crlb!(c, nominal(m, :SIR_WM_I); snr) for snr in (50, 100, 200)]
```
"""
function crlb!(c::CRLBCache, truth::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
               averages = nothing, simulate_with = nothing)
    _check_simulate_with(simulate_with)
    simulate_with === nothing || !c.biased || throw(ArgumentError("with data from " *
        "another model (`simulate_with`), the sandwich only: Eq. 3 needs the true model's " *
        "derivatives with respect to the fitted model's parameters"))
    # The data: from the model at `truth`, or from another model (`simulate_with = m2 => θ2`),
    # with `truth` then the reference values the fit is judged against; `snr` is relative to
    # the data's scale parameter.
    dm, dt = _datasource(c.model, truth, simulate_with)
    t = _crlb_truth(c.model, truth)                  # checked; in the kernel's form
    Σn = noisecovariance(dm, dt; snr, σ, Σ, averages)
    return _crlbresult(c, truth, t, dm, dt, Σn, Σ, simulate_with, averages)
end

# One setting of a sweep: `truth` is a scenario's truth (checked when the scenario was made: in
# model order, every parameter, no roles, no maps) with numbers in place of the varied values,
# so it goes to the kernel's form without being checked again (1.1–1.4 µs a setting). The
# sweep takes this path only when every varied value is a number and the data are the fitted
# model's own.
function _crlb_varied!(c::CRLBCache, truth::NamedTuple; snr, σ, Σ, averages = nothing)
    Σn = noisecovariance(c.model, truth; snr, σ, Σ, averages)
    return _crlbresult(c, truth, map(_tie, truth), c.model, truth, Σn, Σ, nothing, averages)
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
# The kernel's pieces, inlined for models with few free parameters and called as functions for
# many (roadmap items 11 and 17). Inlined, the pieces' unrolled P×P static-array code becomes one
# function, and the compiler's time grows much faster than that code's size: the first crlb took
# 17 s at P = 10, 76 s at P = 13 and did not finish in 23 minutes at P = 22 (DKI), while each
# piece alone compiles in seconds. Called as functions above `_CRLB_INLINE_MAXP`, which costs the
# small models' kernel nothing measurable where it is inlined; P is a type parameter, so the
# choice is made when compiling (the `? :` is folded) and costs nothing at run time.
# Measurements: docs/design.md, "crlb at many parameters".
const _CRLB_INLINE_MAXP = 8

macro _piece(call)
    esc(:(P > _CRLB_INLINE_MAXP ? (@noinline $call) : $call))
end

_crlb!(c::CRLBCache, m::MRIModel, t::NamedTuple, σ2) = _crlb!(c, m, t, σ2, m, t)
function _crlb!(c::CRLBCache{F,P,T}, m::MRIModel, t::NamedTuple, σ2, dm::MRIModel,
                dt::NamedTuple) where {F,P,T}
    # ① The noise-free fit θ̂, from the truth's free values θt. A parameter it leaves on a bound
    #    is held there: masked by u below (u = 0 for it, 1 for the others).
    θ̂, θt, r = @_piece noisefree!(c, m, t, dm, dt)
    pinned = map(_near, θ̂, c.lower) .| map(_near, θ̂, c.upper)
    u = map(p -> ifelse(p, zero(T), one(T)), pinned)

    # ② The Hessian of the fit's cost at θ̂ (and J₀ᵀJ₀). When the fit is exact the residual's
    #    curvature is 0 and first-order duals suffice. Held parameters: rows and columns masked,
    #    1 on the diagonal, so H⁻¹ leaves them be. When the fit stopped at its start with a
    #    residual of exactly 0 (nothing fixed is wrong, the usual unbiased case), the solver's one
    #    evaluation there already holds the rows of J₀ (the same model, point, fixed values and
    #    duals): copied, not evaluated again (the unbiased kernel 25–35% faster, results
    #    identical).
    exact = r.sse <= _EXACT^2 * _sumsq(c.y, c.w)      # (weighted: Σwr² against Σwy²)
    H, G = r.iterations == 0 && iszero(r.sse) && θ̂ == r.θ ? (@_piece _startrows!(c)) : (@_piece hessian!(c, m, θ̂, exact))
    Hm = (u * u') .* H + Diagonal(one(T) .- u)
    nan = fill(T(NaN), SMatrix{P,P,T})
    out(C, ∇b) = (fit = θ̂, truth = θt, bias = θ̂ - θt, cov = C, status = r.status, pinned,
                  exact, H, G, ∇b)

    # ③ Can the data determine the parameters? Fewer observations than parameters to estimate,
    #    nearly dependent columns of J₀ (`_determined`), or H not positive definite: Inf (no
    #    negative variance hidden by abs(), as in the MATLAB code).
    N = _ncounted(c)                                 # (weighted: the positive weights)
    _, okH = @_piece _cholesky(Hm)
    (N >= count(!, pinned) && (@_piece _determined(G, u)) && okH) || return out(_infcov(u), nan)

    # ④ The covariance. The sandwich, H⁻¹(J₀ᵀΣJ₀)H⁻¹ ...
    if !c.biased
        M = exact ? (@_piece meat(c.jbuf, σ2, u, c.w)) : (@_piece meat(c.hbuf, σ2, u, c.w))    # J₀ᵀWΣWJ₀
        return out((@_piece sandwich(Hm, M)), nan)
    end
    #    ... or Eq. 3: J_t (the true signal's Jacobian at the truth, in `c.fitbuf`), the bias
    #    gradient ∇b = H⁻¹J₀ᵀJ_t − I, and F = J_tᵀΣ⁻¹J_t.
    @_piece attruth!(c.fitbuf, c, m, t, θt)
    K = exact ? (@_piece crossgram(c.jbuf, c.fitbuf, u, c.w)) : (@_piece crossgram(c.hbuf, c.fitbuf, u, c.w))
    ∇b = @_piece biasgradient(Hm, K)
    Fm = @_piece information(c.fitbuf, σ2)
    _, okF = @_piece _cholesky(Fm)
    (N >= P && (@_piece _determined(Fm, ones(SVector{P,T}))) && okF) || return out(_infcov(u), ∇b)
    return out((@_piece biasedcrlb(∇b, Fm)), ∇b)
end

# The kernel, then its numbers as a CRLBResult: named tuples by quantity, then parameter. A
# separate function (a "function barrier") so that F and P are known types here.
function _crlbresult(c::CRLBCache{F,P,T}, truth, t, dm, dt, Σn, Σ, simulate_with,
                     averages = nothing) where {F,P,T}
    σ2 = _kernelnoise(Σn, P)
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
    # The model's derived quantities after the parameters (`_derivedcrlb`, below); the
    # parameters alone for a model that derives none.
    q = (fit = nt(k.fit), bias = nt(k.bias), sd = nt(sd), rmse = nt(rmse),
         relbias = nt(rel(k.bias)), relsd = nt(rel(sd)), relrmse = nt(rel(rmse)))
    g = _derivedcrlb(c, k, truth)
    q = g === nothing ? q : map(merge, q, g)
    return CRLBResult(q.fit, q.bias, q.sd, q.rmse, q.relbias, q.relsd, q.relrmse,
                      Matrix(k.cov), k.G, onbound, k.status,
                      c.biased ? :biased : :sandwich, sqrt.(diag(Σn)), truth,
                      map(_untie, c.fixed), c.model, simulate_with,
                      Σ === nothing ? nothing : Matrix{Float64}(Σ), info, observed,
                      averages === nothing ? nothing : _averages(averages, length(c.y)),
                      c.w === nothing ? nothing : copy(c.w))
end

# The model's derived quantities (its `derived` hook, e.g. DTI's MD and FA), each g a function
# of the parameters, as crlb reports the parameters themselves:
#
#   the noise-free fit's value   g(θ̂)            (θ̂ with the fit's fixed values, ties followed)
#   the bias                     g(θ̂) − g(θ)     (θ the truth: exact for what the fit assumes)
#   the SD (the delta method)    √diag(∇g·C·∇gᵀ), ∇g = ∂g/∂θ at θ̂ over the free parameters
#
# C is the parameters' covariance (sandwich or Eq. 3), so the SD is first order in the noise, as
# theirs. So is the bias: the noise's own second-order term, ½tr(∇²g·C) (item 16: FA's noise
# bias at low SNR), is not in it, as the parameters' (Box's) is not in theirs. ∇g by dual
# numbers through `derived` (no allocation: P partials, static matrices); NaN where `derived`
# is not differentiable (a repeated eigenvalue: AD and V1 where λ1 = λ2). The parameters on a
# bound have C's rows 0, so they add nothing; when the data cannot determine the parameters
# (C is Inf), neither the derived quantities (SD Inf). `nothing` for a model that derives none:
# known from the model's type, so the branch costs nothing when compiling for it.
function _derivedcrlb(c::CRLBCache{F,P,T}, k, truth) where {F,P,T}
    g(θ) = derived(c.mT, _params(Val(F), θ, c.fixed))
    ĝ = g(k.fit)
    ĝ === nothing && return nothing
    _check_derivednames(c.model, ĝ)
    G = keys(ĝ)
    gt = map(Float64, derived(c.mT, map(float, resolve(c.model, truth))))   # at the truth
    ∇g = ForwardDiff.jacobian(θ -> SVector(values(g(θ))), k.fit)          # length(G) × P
    sd = all(isfinite, k.cov) ? sqrt.(max.(diag(∇g * k.cov * ∇g'), 0)) :
                                fill(Inf, SVector{length(G),Float64})
    ĝ = map(Float64, ĝ)
    b = map(-, ĝ, gt)
    s = NamedTuple{G}(Tuple(sd))
    r = map((b, s) -> sqrt(b^2 + s^2), b, s)
    rel(x) = map((x, t) -> x / abs(t), x, gt)
    return (fit = ĝ, bias = b, sd = s, rmse = r, relbias = rel(b), relsd = rel(s),
            relrmse = rel(r))
end

# A derived quantity named as one of the model's parameters would take its place in the
# results' named tuples (and its map's file name): refused, naming the model and the names.
function _check_derivednames(m::MRIModel, g::NamedTuple)
    clash = filter(n -> haskey(parameters(m), n), keys(g))
    isempty(clash) || throw(ArgumentError("$(nameof(typeof(m)))'s `derived` gives " *
        "$(join(clash, ", ")), the name of a parameter: give the derived quantity another name"))
    return nothing
end

# The fit's free parameters, in model order: the first P names of a result's named tuples (the
# model's derived quantities come after them), and the rows of `c.cov`. Read from the result's
# type, the names K of its named tuples and P from its static P×P `gram`, so that the compiler
# knows them (`keys(c.fit)[1:P]` with P read at run time would not: the sweep's tables 60%
# slower).
_freenames(::CRLBResult{M,<:NamedTuple{K},Tr,X,D,S,<:StaticMatrix{P}}) where {M,K,Tr,X,D,S,P} =
    ntuple(i -> K[i], Val(P))

# -------------------------------------------------------------------------------------------------
# StatsAPI's names (as for `fit` and `nobs`): the information matrix, the covariance, the SDs
# -------------------------------------------------------------------------------------------------

"""
    informationmatrix(c::CRLBResult; expected = true) -> Matrix

The information matrix of the fit's free parameters (in model order, as `vcov(c)`) at the
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
with `cov = :biased`) and their SDs (the parameters' part of `c.sd`, as a vector), in model
order. StatsAPI's functions: the parameters only, not the model's derived quantities.
"""
vcov(c::CRLBResult) = c.cov
stderror(c::CRLBResult) = [c.sd[p] for p in _freenames(c)]

# -------------------------------------------------------------------------------------------------
# Display: at the REPL a table, one row per free parameter, in the style of `compare`
# -------------------------------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", c::CRLBResult)
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    σ = c.σ
    noise = _noisetext(σ, c.Σ, c.averages)
    how = c.method === :sandwich ? "sandwich (the least-squares fit)" :
                                   "biased CRLB (Dortch 2018 Eq. 3)"
    print(io, "crlb for ", nameof(typeof(c.model)), ", ", length(σ), " observations, ", noise,
          ", covariance: ", how)
    _print_data(io, c.data)
    # The fit's fixed values, with the truth where it differs: the source of any bias.
    _print_fixed(io, c.fixed, resolve(c.model, c.truth))
    c.weights === nothing || print(io, "\nweighted fit: ", _weightstext(c.weights))
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
    P = length(_freenames(c))                # then the model's derived quantities, if any
    for (i, r) in enumerate(rows)
        i == P + 1 && print(io, "\n  derived from the parameters (SD by the delta method):")
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end

# The fit's fixed values, with the truth where it differs (`_wrong`, scenario.jl: a tie
# followed at the truth too, e.g. R1m = :R1f against a true R1m of 1.0): the source of any bias.
# `θt` is the truth with its ties followed, or `nothing` when the truth varies (a sweep). Shared
# with `montecarlo`'s display and the sweeps' (one line per fit, `label`).
function _print_fixed(io::IO, fixed::NamedTuple, θt, label = "fixed in the fit")
    isempty(fixed) && return
    w = θt === nothing ? (;) : _wrong(θt, merge(θt, fixed))
    function item(n, x)
        s = "$n = $(_showvalue(x))"
        haskey(w, n) || return s
        tied = x isa Real ? "" : " = $(round(w[n].fit; sigdigits = 4))"
        return s * tied * " (true $(_showvalue(w[n].truth)))"
    end
    print(io, "\n", label, ": ", join((item(n, x) for (n, x) in pairs(fixed)), ", "))
end

# The noise, in words: one σ, σ per observation (its range), or a covariance matrix; with
# averages, one average's σ (σ·√NEX, the level given) and the averages' range.
function _noisetext(σ, Σ, averages = nothing)
    f(x) = string(round(x; sigdigits = 4))
    # One value, or the range (as printed: σ·√NEX is one average's σ to rounding)
    span(x, g) = (v = g.(x); all(==(v[1]), v) ? v[1] : "$(g(minimum(x))) to $(g(maximum(x)))")
    averages === nothing ||
        return "σ = $(span(σ .* sqrt.(averages), f)) per average, " * _averagestext(averages)
    range = all(==(σ[1]), σ) ? "σ = $(f(σ[1]))" : "σ = $(f(minimum(σ))) to $(f(maximum(σ)))"
    return Σ === nothing ? range : "$range, correlated (Σ given)"
end

# The averages, in words: "4 averages", "1 to 8 averages".
function _averagestext(a)
    x = all(==(a[1]), a) ? string(a[1]) : "$(minimum(a)) to $(maximum(a))"
    return x == "1" ? "1 average" : "$x averages"
end

# Where the data came from, when not from the fitted model (`simulate_with`).
_print_data(io::IO, ::Nothing) = nothing
_print_data(io::IO, d::Pair) =
    print(io, "\ndata from ", nameof(typeof(d.first)), ": ",
          join(("$n = $(_showvalue(x))" for (n, x) in pairs(d.second)), ", "))

Base.show(io::IO, c::CRLBResult) =
    print(io, "CRLBResult(", nameof(typeof(c.model)), ": ", join(keys(c.fit), ", "), ")")
