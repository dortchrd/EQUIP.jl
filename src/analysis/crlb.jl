# =================================================================================================
# crlb: the bias, SD and RMSE of a fit, predicted from derivatives (no noisy fits)
#
#   truth θ ──────▶ g_true = signal(m, θ)                 the noise-free data
#   fit settings ─▶ free and fixed parameters, bounds     as `fit` uses them
#   ① noise-free fit  θ̂ = fit of g_true, started at the truth;  b = θ̂ − θ;  r₀ = g(θ̂) − g_true
#   ② derivatives     J₀ = ∂g/∂θ at θ̂ (ForwardDiff, ties followed);  H = J₀ᵀJ₀ + Σᵢ r₀ᵢ∇²gᵢ
#   ③ covariance      :sandwich  H⁻¹ J₀ᵀΣJ₀ H⁻¹                       (default)
#                     :biased    (I+∇b) F⁻¹ (I+∇b)ᵀ,  I+∇b = H⁻¹J₀ᵀJ_t,  F = J_tᵀΣ⁻¹J_t
#   ④ report          SD = √diag, RMSE = √(b² + SD²), absolute and relative to the truth
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
#
# The code: `crlb` checks its inputs, makes a plan (roles, bounds, buffers) and calls the kernel
# `_crlb!`, which allocates nothing (5–30 µs a call), for the protocol optimizer to call
# millions of times; then it packages the kernel's static arrays into named tuples.
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
    # function `fit` (the kernel runs the noise-free fit with the solver itself, `levmar`).
    cov in (:sandwich, :biased) ||
        throw(ArgumentError("`cov` must be :sandwich or :biased, got $(repr(cov))"))
    for (k, x) in pairs(fit)
        k in _OPTIONS && throw(ArgumentError("`fit = (; $k)`: crlb runs its own noise-free fit; " *
                                             "give only parameter settings"))
        _hasmap(x) &&
            throw(ArgumentError("`fit = (; $k)`: crlb is for one voxel; give numbers, not maps"))
    end
    simulate_with === nothing || cov === :sandwich || throw(ArgumentError("with data from " *
        "another model (`simulate_with`), the sandwich only: Eq. 3 needs the true model's " *
        "derivatives with respect to the fitted model's parameters"))
    # The data: from `m` at `truth`, or from another model (`simulate_with = m2 => θ2`), with
    # `truth` then the reference values the fit is judged against.
    dm, dt = simulate_with === nothing ? (m, truth) : (simulate_with.first, simulate_with.second)
    (any(_hasmap, values(truth)) || any(_hasmap, values(dt))) &&
        throw(ArgumentError("crlb is for one voxel: give numbers, not maps"))
    resolve(m, truth)                        # checks the truth: every parameter, no roles
    θd = resolve(dm, dt)                     # and the data's
    nobs(dm) == nobs(m) || throw(DimensionMismatch("the data's model has $(nobs(dm)) " *
        "observations, the fitted model $(nobs(m))"))
    mT = _with_precision(m, Float64)         # acquisition vectors as Float64, as `fit` does
    plan = _crlb_plan(mT, cov; fit...)
    # The noise, in the kernel's form (a variance, one per observation, or the full covariance
    # by its Cholesky factor); `snr` is relative to the data's scale parameter.
    noise, σv = _crlb_noise(dm, θd, snr, σ, Σ, nobs(m), length(_crlbnames(plan)))
    k = _crlb!(plan, mT, _crlb_truth(m, truth), noise, _with_precision(dm, Float64),
               _crlb_truth(dm, dt))

    # Packaging: named tuples by quantity, then parameter (allocating; the kernel does not).
    F = _crlbnames(plan)
    P = length(F)
    sd = sqrt.(diag(k.cov))
    rmse = sqrt.(k.bias .^ 2 .+ sd .^ 2)
    rel(x) = x ./ abs.(k.truth)
    nt(x) = NamedTuple{F}(Tuple(x))
    U = [j for j in 1:P if !k.pinned[j]]
    J0 = [partials(value(d))[j] for d in plan.hbuf, j in 1:P]   # J₀ at θ̂, the kernel's buffer
    condJ = isempty(U) ? Inf : _condition(J0[:, U], k.truth[U])
    info, observed = _information(plan, noise, J0)
    return CRLBResult(nt(k.fit), nt(k.bias), nt(sd), nt(rmse), nt(rel(k.bias)), nt(rel(sd)),
                      nt(rel(rmse)), Matrix(k.cov), condJ, [F[j] for j in 1:P if k.pinned[j]],
                      k.status, cov, σv, truth, map(_untie, plan.fixed), m, simulate_with,
                      Σ === nothing ? nothing : Matrix{Float64}(Σ), info, observed)
end

# A full noise covariance by its Cholesky factor (Σ = LLᵀ, lower triangle), with a work buffer
# for the kernel's whitening (one row of L⁻¹J per observation).
struct _FullNoise{T,P}
    L::Matrix{T}
    v::Vector{SVector{P,T}}
end


# -------------------------------------------------------------------------------------------------
# StatsAPI's names (as for `fit` and `nobs`): the information matrix, the covariance, the SDs
# -------------------------------------------------------------------------------------------------

"""
    informationmatrix(c::CRLBResult; expected = true) -> Matrix
    informationmatrix(m, truth; expected = true, crlb's keywords...) -> Matrix

The information matrix of the fit's free parameters (in model order, `keys(c.fit)`), at the
noise-free fit's values θ̂ (the truth when nothing fixed is wrong), for the noise given to
[`crlb`](@ref). `expected = true`: the Fisher information J₀ᵀΣ⁻¹J₀ (J₀ the Jacobian at θ̂, Σ
the noise covariance; JᵀJ/σ² for one σ). `expected = false`: the observed information, the
Hessian of the negative log-likelihood of the noise-free data at θ̂, which adds the residual's
curvature when an assumption is wrong (with one σ: H/σ²). When nothing fixed is wrong, the
inverse of either is the classical CRLB. StatsAPI's function, with EQUIP's methods.

The sandwich is built from them: `vcov(c) ≈ inv(I_obs) * I * inv(I_obs)` with one σ
(`I = informationmatrix(c)`, `I_obs = informationmatrix(c; expected = false)`).
"""
informationmatrix(c::CRLBResult; expected::Bool = true) = expected ? c.information : c.observed
informationmatrix(m::MRIModel, truth::NamedTuple; expected::Bool = true, kwargs...) =
    informationmatrix(crlb(m, truth; kwargs...); expected)

"""
    vcov(c::CRLBResult) -> Matrix
    stderror(c::CRLBResult) -> Vector

The predicted covariance matrix of the fit's free parameters (`c.cov`: the sandwich, or Eq. 3
with `cov = :biased`) and their SDs (`c.sd` as a vector), in model order. StatsAPI's functions.
"""
vcov(c::CRLBResult) = c.cov
stderror(c::CRLBResult) = collect(values(c.sd))

# -------------------------------------------------------------------------------------------------
# The plan and the kernel: allocation-free, for the protocol optimizer
#
#   plan = _crlb_plan(m, cov; fit...)        once: roles, bounds, solver settings, buffers
#   t = _crlb_truth(m, truth)                once per truth: complete, ties as _Tie
#   k = _crlb!(plan, m, t, σ²)               per call: no allocation
#
# `m` in `_crlb!` may be a different protocol from the one the plan was made with (new TI and
# TD at every step of an optimizer), with the same number of observations and Float64
# acquisition values (static vectors work: `QMTSIR(TI = SVector(...), TD = SVector(...))`).
# σ² is a number, one value per observation, or a full covariance (`_FullNoise`). `crlb` is
# plan + kernel + packaging, so its tests check the kernel the optimizer uses.
# -------------------------------------------------------------------------------------------------

"""
    CRLBPlan{F,P,T}

The fit's free parameters `F` (`P` of them, in precision `T`), their bounds, the fixed values
(ties as `_Tie`), the solver settings of the noise-free fit, the covariance method and the
buffers the kernel `_crlb!` writes into: the noise-free data, first-order duals (the noise-free
fit, and J_t for Eq. 3) and nested duals (J₀ and the second derivatives at θ̂). Not
thread-safe: one plan per task.
"""
struct CRLBPlan{F,P,T,X,B,H}
    lower::SVector{P,T}
    upper::SVector{P,T}
    fixed::X                   # the fit's fixed values: numbers and ties (`_Tie`)
    opts::SolverOptions{T}
    biased::Bool               # Eq. 3 (`cov = :biased`) rather than the sandwich
    y::Vector{T}               # the noise-free data g_true
    buf::B                     # Vector{Dual{FitTag,T,P}}: the noise-free fit; J_t
    hbuf::H                    # Vector{Dual{CRLBTag,Dual{FitTag,T,P},P}}: J₀ and ∇²g at θ̂
end
_crlbnames(::CRLBPlan{F}) where {F} = F

function _crlb_plan(m::MRIModel, cov::Symbol; params...)
    spec = setup(m, Float64; params...)        # the fit's free and fixed parameters, and bounds
    return _crlb_plan(spec, cov === :biased, nobs(m))
end
function _crlb_plan(spec::FitSpec{F,P,T}, biased::Bool, n::Integer) where {F,P,T}
    D1 = Dual{FitTag,T,P}
    D2 = Dual{CRLBTag,D1,P}
    return CRLBPlan{F,P,T,typeof(spec.fixed),Vector{D1},Vector{D2}}(
        spec.lower, spec.upper, spec.fixed, SolverOptions{T}(_CRLB_SOLVER), biased,
        Vector{T}(undef, n), Vector{D1}(undef, n), Vector{D2}(undef, n))
end

# The truth, checked (every parameter, no roles; see `resolve`), in model order, with ties by
# name as `_Tie` so the kernel reads them as plain field accesses. Done once per truth: a tie's
# type depends on the name it holds, which is only known at run time here.
function _crlb_truth(m::MRIModel, truth::NamedTuple)
    resolve(m, truth)
    names = keys(parameters(m))
    return NamedTuple{names}(map(n -> _tie(truth[n]), names))
end

# A second tag for the outer layer of nested dual numbers (second derivatives), distinct from
# the solver's FitTag (first derivatives).
struct CRLBTag end

# Nested dual numbers: θⱼ seeded twice, so one evaluation of the model gives each observation's
# value gᵢ, its gradient (row i of J₀) and its second derivatives ∇²gᵢ (see `_second`).
@inline function _seed2(θ::SVector{P,T}) where {P,T}
    inner = _seed(θ)                           # Dual{FitTag}: value θⱼ, partials eⱼ
    D = eltype(inner)
    return SVector(ntuple(j -> Dual{CRLBTag}(inner[j], ntuple(k -> ifelse(k == j, one(D), zero(D)),
                                                             Val(P))), Val(P)))
end

# ∇²gᵢ from one nested dual: the outer partial k is ∂gᵢ/∂θₖ, itself a dual whose partial l is
# ∂²gᵢ/∂θₖ∂θₗ. `SMatrix` takes the P² entries column by column.
@inline function _second(d::Dual{CRLBTag,Dual{FitTag,T,P},P}) where {T,P}
    ps = partials(d).values
    return SMatrix{P,P,T}(ntuple(n -> partials(ps[(n - 1) ÷ P + 1]).values[(n - 1) % P + 1],
                                 Val(P * P)))
end

@inline _at(s::Real, i) = s                          # σ²: one number for every observation,
@inline _at(s::AbstractVector, i) = @inbounds s[i]   # or one per observation

# The two sums that involve the noise, by its form (Julia picks the method by the type of σ²):
#   sandwich  H⁻¹(J₀ᵀΣJ₀)H⁻¹ = Σₖ aₖaₖᵀ, aₖ = H⁻¹wₖ: one σ or one per observation, wᵢ = σᵢJ₀ᵢ;
#             a full Σ = LLᵀ, wₖ = row k of LᵀJ₀ = Σᵢ≥ₖ LᵢₖJ₀ᵢ (O(N²P), N ≤ 100)
#   Fisher    J_tᵀΣ⁻¹J_t = Σₖ vₖvₖᵀ: vᵢ = J_tᵢ/σᵢ, or a full Σ, vₖ = row k of L⁻¹J_t (forward
#             substitution, the rows kept in the noise's buffer)
# Sums of squares, so neither gives a negative variance. Rows of parameters on a bound are
# masked by `u` (in the sandwich; the Fisher information is over every free parameter).
@inline function _sandwich(σ2::Union{Real,AbstractVector}, Hinv, plan::CRLBPlan{F,P,T},
                           u) where {F,P,T}
    C = zero(SMatrix{P,P,T})
    @inbounds for i in eachindex(plan.hbuf)
        a = Hinv * (u .* SVector{P,T}(partials(value(plan.hbuf[i])).values))
        C += _at(σ2, i) * (a * a')
    end
    return C
end
function _sandwich(n::_FullNoise, Hinv, plan::CRLBPlan{F,P,T}, u) where {F,P,T}
    L = n.L
    C = zero(SMatrix{P,P,T})
    @inbounds for k in eachindex(plan.hbuf)
        w = zero(SVector{P,T})
        for i in k:length(plan.hbuf)
            w += L[i, k] * SVector{P,T}(partials(value(plan.hbuf[i])).values)
        end
        a = Hinv * (u .* w)
        C += a * a'
    end
    return C
end
@inline function _fisher(σ2::Union{Real,AbstractVector}, plan::CRLBPlan{F,P,T}) where {F,P,T}
    Fm = zero(SMatrix{P,P,T})
    @inbounds for i in eachindex(plan.buf)
        Jt = SVector{P,T}(partials(plan.buf[i]).values)
        Fm += (Jt * Jt') / _at(σ2, i)
    end
    return Fm
end
function _fisher(n::_FullNoise, plan::CRLBPlan{F,P,T}) where {F,P,T}
    L, v = n.L, n.v
    Fm = zero(SMatrix{P,P,T})
    @inbounds for k in eachindex(plan.buf)
        t = SVector{P,T}(partials(plan.buf[k]).values)
        for i in 1:k-1
            t -= L[k, i] * v[i]
        end
        v[k] = t / L[k, k]
        Fm += v[k] * v[k]'
    end
    return Fm
end

# A parameter on a bound in the noise-free fit (within 1e-9, relative for large bounds).
@inline _near(x, bound) = isfinite(bound) && abs(x - bound) <= 1e-9 * max(1, abs(bound))

"""
    _crlb!(plan, m, truth, σ²) -> (; fit, truth, bias, cov, status, pinned)
    _crlb!(plan, m, truth, σ², dm, dtruth)

The kernel of [`crlb`](@ref), without allocation: the noise-free fit's values θ̂, the true free
values, the bias, the covariance (an `SMatrix`; `Inf` for parameters the data cannot determine,
0 for those on a bound), the noise-free fit's status, and which parameters are on a bound
(held there). `truth` from `_crlb_truth`, `σ²` a number, a vector (one variance per
observation) or a `_FullNoise` (the covariance by its Cholesky factor; `_crlb_noise`). With `dm`
and `dtruth` (prepared the same way), the noise-free data come from another model; `truth` is
then the reference the bias is measured against, and the start of the noise-free fit (the
sandwich only).
"""
_crlb!(plan::CRLBPlan, m::MRIModel, truth::NamedTuple, σ2) = _crlb!(plan, m, truth, σ2, m, truth)
function _crlb!(plan::CRLBPlan{F,P,T}, m::MRIModel, truth::NamedTuple, σ2, dm::MRIModel,
                dtruth::NamedTuple) where {F,P,T}
    lo, hi = plan.lower, plan.upper
    y = plan.y

    # ① The noise-free data (from `dm` at `dtruth`: the fitted model and the truth, or another
    #    model's, `simulate_with`), and their fit started at the truth (projected into the
    #    bounds): the package's own LM (`levmar`, levmar.jl), with tight tolerances, since the
    #    bias is a small difference of two values and needs every digit. `relabel` as `fit` does.
    θres = _resolve_ties(truth)
    θt = SVector{P,T}(ntuple(j -> T(getfield(θres, F[j])), Val(P)))
    signal!(y, dm, _resolve_ties(dtruth))
    r = levmar(VoxelProblem{F}(m, y, plan.fixed), θt, lo, hi, plan.opts, plan.buf)
    θ̂ = SVector{P,T}(Tuple(relabel(m, NamedTuple{F}(Tuple(r.θ)))))
    pinned = map(_near, θ̂, lo) .| map(_near, θ̂, hi)   # held on a bound: left out below
    u = map(p -> ifelse(p, zero(T), one(T)), pinned)  # 1 for the others

    # ② One evaluation with nested duals at θ̂: J₀ (rows Jᵢ) and the full Hessian of ½‖r‖²,
    #    H = Σᵢ (JᵢJᵢᵀ + rᵢ∇²gᵢ), the second term from the residual r₀ = g(θ̂) − g_true.
    signal!(plan.hbuf, m, _params(Val(F), _seed2(θ̂), plan.fixed))
    H = zero(SMatrix{P,P,T})
    G = zero(SMatrix{P,P,T})                   # J₀ᵀJ₀, for the identifiability test
    @inbounds for i in eachindex(plan.hbuf)
        d = plan.hbuf[i]
        Ji = SVector{P,T}(partials(value(d)).values)
        H += Ji * Ji' + (value(value(d)) - y[i]) * _second(d)
        G += Ji * Ji'
    end
    # Parameters on a bound: rows and columns masked, 1 on the diagonal, so H⁻¹ leaves them be.
    Hm = (u * u') .* H + Diagonal(one(T) .- u)
    infs = _infcov(u)                          # the answer when the data cannot determine them

    # ③ The covariance. Identifiability first (fewer observations than parameters to estimate,
    #    or nearly dependent columns of J₀): a singular H can look positive definite after
    #    rounding, and the sandwich then gives finite, meaningless values. Then a Cholesky
    #    factorization: failing means not positive definite, also Inf (no negative variance
    #    hidden by abs(), as in the MATLAB code).
    LH, okH = _cholesky(Hm)
    if !(length(y) >= count(!, pinned) && _determined(G, u) && okH)
        return (fit = θ̂, truth = θt, bias = θ̂ - θt, cov = infs, status = r.status, pinned)
    end
    Hinv = _chol_inv(LH)
    if !plan.biased
        # Sandwich, H⁻¹(Σᵢ σᵢ² JᵢJᵢᵀ)H⁻¹, as Σᵢ σᵢ² aᵢaᵢᵀ with aᵢ = H⁻¹Jᵢ: how noise in
        # observation i moves the fit. A sum of squares, so the variances are never negative.
        C = _sandwich(σ2, Hinv, plan, u)
    else
        # Eq. 3, (I + ∇b)F⁻¹(I + ∇b)ᵀ. J_t: the true signal's Jacobian at the truth, the truth's
        # own ties followed (R1m = R1f moves with R1f, as the MATLAB's `R1 + eps` on both).
        # I + ∇b = H⁻¹J₀ᵀJ_t (implicit function theorem), rows on a bound 0.
        signal!(plan.buf, m, _resolve_ties(merge(truth, NamedTuple{F}(Tuple(_seed(θt))))))
        K = zero(SMatrix{P,P,T})               # J₀ᵀJ_t
        @inbounds for i in eachindex(plan.buf)
            Jt = SVector{P,T}(partials(plan.buf[i]).values)
            J0 = u .* SVector{P,T}(partials(value(plan.hbuf[i])).values)
            K += J0 * Jt'
        end
        Fm = _fisher(σ2, plan)                 # F = J_tᵀΣ⁻¹J_t
        LF, okF = _cholesky(Fm)
        if !(length(y) >= P && _determined(Fm, ones(SVector{P,T})) && okF)
            return (fit = θ̂, truth = θt, bias = θ̂ - θt, cov = infs, status = r.status, pinned)
        end
        D = Hinv * K
        X = hcat(ntuple(j -> _fwd_solve(LF, D[j, :]), Val(P))...)   # L⁻¹Dᵀ, F = LLᵀ
        C = X' * X                             # D F⁻¹ Dᵀ, never a negative variance
    end
    return (fit = θ̂, truth = θt, bias = θ̂ - θt, cov = C, status = r.status, pinned)
end

# The information matrices at θ̂, from the kernel's buffers (after `_crlb!`): the Fisher
# (expected) information J₀ᵀΣ⁻¹J₀, and the observed information, the Hessian of the negative
# log-likelihood of the noise-free data, J₀ᵀΣ⁻¹J₀ + Σᵢ wᵢ∇²gᵢ with w = Σ⁻¹r, r = g(θ̂) − g_true:
# the two agree when the fit is exact (r = 0), and differ by the residual's curvature when an
# assumption is wrong. With one σ, the observed information is H/σ², the sandwich's bread.
function _information(plan::CRLBPlan{F,P}, noise, J0) where {F,P}
    r = [value(value(d)) for d in plan.hbuf] .- plan.y
    WJ, w = _apply_inverse(noise, J0), _apply_inverse(noise, r)   # Σ⁻¹J₀, Σ⁻¹r
    info = Matrix(Symmetric(J0' * WJ))
    observed = info + sum(w[i] * _second(plan.hbuf[i]) for i in eachindex(w))
    return info, Matrix(Symmetric(observed))
end
_apply_inverse(σ2::Real, x) = x ./ σ2
_apply_inverse(σ2::AbstractVector, x) = x ./ σ2
_apply_inverse(n::_FullNoise, x) = n.L' \ (n.L \ x)                   # Σ = LLᵀ

# Can the data tell the parameters apart? The Cholesky pivots of J₀ᵀJ₀ with unit diagonal (each
# column of J₀ normalized) are the squared sines of the angles between each column and the span
# of the ones before it. A pivot below `_PIVOT_MIN` (an angle below 1e-6 rad: the condition
# number of J₀ is then at least ~1e6, a relative SD of ~1e6 × noise/signal) means a parameter is
# a combination of the others, to within what rounding in J₀ᵀJ₀ can resolve. A parameter with
# no influence at all (a zero column) is undetermined too. `u` masks parameters on a bound.
@inline function _determined(G::SMatrix{P,P,T}, u::SVector{P,T}) where {P,T}
    g = diag(G)
    any((u .> 0) .& .!(g .> 0)) && return false
    s = map((gj, uj) -> uj > 0 ? inv(sqrt(gj)) : zero(T), g, u)
    L, ok = _cholesky((s * s') .* G + Diagonal(one(T) .- u))
    ok || return false
    return all(j -> u[j] == 0 || L[j, j]^2 >= _PIVOT_MIN, 1:P)
end
const _PIVOT_MIN = 1e-12

# Inf for every pair of parameters not on a bound, 0 for those on one.
@inline _infcov(u::SVector{P,T}) where {P,T} =
    SMatrix{P,P,T}(ntuple(n -> u[(n - 1) % P + 1] > 0 && u[(n - 1) ÷ P + 1] > 0 ? T(Inf) : zero(T),
                          Val(P * P)))

# Inverse from a Cholesky factor, column by column (P is small), and forward substitution
# (L z = b) alone; `_cholesky` and `_chol_solve` are the solver's (levmar.jl).
@inline _chol_inv(L::SMatrix{P,P,T}) where {P,T} =
    hcat(ntuple(j -> _chol_solve(L, SVector{P,T}(ntuple(k -> ifelse(k == j, one(T), zero(T)),
                                                        Val(P)))), Val(P))...)
@inline function _fwd_solve(L::SMatrix{P,P,T}, b::SVector{P,T}) where {P,T}
    z = MVector{P,T}(b)
    @inbounds for i in 1:P
        t = z[i]
        for k in 1:i-1
            t -= L[i, k] * z[k]
        end
        z[i] = t / L[i, i]
    end
    return SVector(z)
end

# The noise-free fit's solver: the step tolerance alone, tight, and room to get there.
const _CRLB_SOLVER = LM(maxiter = 1000, xtol = 1e-13, ftol = 0.0, gtol = 0.0)

# The condition number of J with each column scaled by its parameter's true value (relative
# sensitivities, so the units drop out; 1 for a true value of 0), reported as `c.cond`. Inf
# when there are fewer observations than parameters: `cond` of a wide matrix would ignore the
# missing directions.
_condition(J, θ) = size(J, 1) < size(J, 2) ? Inf :
                   cond(J * Diagonal([x == 0 ? 1.0 : abs(x) for x in θ]))

_hasmap(x::AbstractArray) = true
_hasmap(x::Free) = x.start isa AbstractArray
_hasmap(x::Fixed) = x.value isa AbstractArray
_hasmap(x) = false

# The noise, from `snr` (relative to the scale parameter, as `simulate`), `σ` (a number, or one
# SD per observation) or `Σ` (the covariance matrix), in the kernel's form: a variance (σ²,
# "pulled out"), one per observation, or a `_FullNoise`; and the SD of each observation.
function _crlb_noise(m, θ, snr, σ, Σ, N, P)
    count(!isnothing, (snr, σ, Σ)) == 1 ||
        throw(ArgumentError("give the noise level as `snr`, `σ` or `Σ` (one of them)"))
    if snr !== nothing
        scale = _scalename(m)
        scale === nothing && throw(ArgumentError("$(nameof(typeof(m))) has no " *
                                   "`Free(start = :scale)` parameter to define the SNR; give `σ`"))
        s = Float64(θ[scale]) / snr
        return s^2, fill(s, N)
    elseif σ isa Real
        return Float64(σ)^2, fill(Float64(σ), N)
    elseif σ !== nothing
        σ isa AbstractVector || throw(ArgumentError("`σ` is a number or one SD per observation; " *
                                                    "for a covariance matrix, give `Σ`"))
        length(σ) == N ||
            throw(DimensionMismatch("`σ` has $(length(σ)) values; the model has $N observations"))
        v = Float64.(collect(σ))
        return v .^ 2, v
    end
    return _fullnoise(Σ, N, P), sqrt.(diag(Σ))
end

# Σ checked and factored as `simulate` does (`_noisefactor`, sim/simulate.jl).
_fullnoise(Σ, N, P) =
    _FullNoise{Float64,P}(_noisefactor(Σ, N), Vector{SVector{P,Float64}}(undef, N))

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
_check_simulate_with(::Nothing) = nothing
_check_simulate_with(s) = s isa Pair{<:MRIModel,<:NamedTuple} || throw(ArgumentError(
    "`simulate_with = model => truth`: a model and every value of its parameters, e.g. " *
    "`T2BiexpDecay(TE = te) => (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)`"))

Base.show(io::IO, c::CRLBResult) =
    print(io, "CRLBResult(", nameof(typeof(c.model)), ": ", join(keys(c.fit), ", "), ")")
