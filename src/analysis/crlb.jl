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
`σ` (the noise SD per observation), `truth`, `fixed` (the fit's fixed values) and `model`.
"""
struct CRLBResult{M<:MRIModel,N<:NamedTuple,Tr<:NamedTuple,X<:NamedTuple}
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
end

"""
    crlb(m, truth; snr or σ, fit = (...), cov = :sandwich, vary) -> CRLBResult or Sweep

The bias, SD and RMSE of each fitted parameter predicted for one voxel, from derivatives: the
Cramér–Rao lower bound (CRLB, also CRB), biased when the fit fixes a parameter at a value other
than the truth's. No noisy fits: the bias comes from one fit of the noise-free data, and the
SD from the derivatives at that fit's answer, first order in the noise.

- `truth`: every parameter of `m` (numbers, or ties), as for [`simulate`](@ref): e.g.
  `nominal(m, :SIR_WM_I)` or `nominal(m; PSR = 0.15)`.
- `snr` (relative to the model's scale parameter, as for `simulate`: σ = M0/snr) or `σ`, a
  number or one value per observation (Gaussian noise).
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

A free parameter that ends on a bound in the noise-free fit is held there (SD 0; `c.onbound`).
Parameters the data cannot determine (a singular Jacobian) get SD `Inf`. EQUIP's fits are
unweighted: with σ per observation the sandwich is the SD of the unweighted fit, and `:biased`
the bound a weighted fit would reach. Rician noise, bounds hit by noisy fits and nonlinearity
at low SNR are not modelled: simulate and fit for those.

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)                          # kmf = 5 in this tissue
c = crlb(m, θ; snr = 200)                          # the fit fixes kmf at 12.5: biased
c.bias.PSR, c.sd.PSR, c.relrmse.PSR
crlb(m, θ; snr = 200, fit = (kmf = 5.0,))          # kmf fixed at the truth: unbiased
crlb.(Ref(m), [nominal(m, t) for t in (:SIR_WM_I, :SIR_GM)]; snr = 200)   # several tissues
crlb(m, θ; snr = 200, vary = (kmf = 5:2.5:20,))    # the bias against the true kmf (a Sweep)
```
"""
function crlb(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing, fit::NamedTuple = (;),
              cov::Symbol = :sandwich, vary::NamedTuple = (;))
    # A sweep: one call per setting (sweep.jl), each with one noise level and one truth.
    if _issweep(snr, vary)
        settings = _sweep_settings(m, snr, vary)
        results = [crlb(m, merge(truth, _varied(st)); snr = _snr_of(st, snr), σ, fit, cov)
                   for st in settings]
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
    any(_hasmap, values(truth)) &&
        throw(ArgumentError("crlb is for one voxel: give numbers, not maps"))
    θ = resolve(m, truth)                    # checks the truth: every parameter, no roles
    mT = _with_precision(m, Float64)         # acquisition vectors as Float64, as `fit` does
    plan = _crlb_plan(mT, cov; fit...)
    σv = _crlb_sigma(m, θ, snr, σ, nobs(m))
    k = _crlb!(plan, mT, _crlb_truth(m, truth), σv .^ 2)

    # Packaging: named tuples by quantity, then parameter (allocating; the kernel does not).
    F = _crlbnames(plan)
    P = length(F)
    sd = sqrt.(diag(k.cov))
    rmse = sqrt.(k.bias .^ 2 .+ sd .^ 2)
    rel(x) = x ./ abs.(k.truth)
    nt(x) = NamedTuple{F}(Tuple(x))
    U = [j for j in 1:P if !k.pinned[j]]
    J0 = [partials(value(d))[j] for d in plan.hbuf, j in U]   # J₀ at θ̂, from the kernel's buffer
    condJ = isempty(U) ? Inf : _condition(J0, k.truth[U])
    return CRLBResult(nt(k.fit), nt(k.bias), nt(sd), nt(rmse), nt(rel(k.bias)), nt(rel(sd)),
                      nt(rel(rmse)), Matrix(k.cov), condJ, [F[j] for j in 1:P if k.pinned[j]],
                      k.status, cov, σv, truth, map(_untie, plan.fixed), m)
end

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
# σ² is a number or one value per observation. `crlb` is plan + kernel + packaging, so its
# tests check the kernel the optimizer uses.
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

# A parameter on a bound in the noise-free fit (within 1e-9, relative for large bounds).
@inline _near(x, bound) = isfinite(bound) && abs(x - bound) <= 1e-9 * max(1, abs(bound))

"""
    _crlb!(plan, m, truth, σ²) -> (; fit, truth, bias, cov, status, pinned)

The kernel of [`crlb`](@ref), without allocation: the noise-free fit's values θ̂, the true free
values, the bias, the covariance (an `SMatrix`; `Inf` for parameters the data cannot determine,
0 for those on a bound), the noise-free fit's status, and which parameters are on a bound
(held there). `truth` from `_crlb_truth`, `σ²` a number or a vector.
"""
function _crlb!(plan::CRLBPlan{F,P,T}, m::MRIModel, truth::NamedTuple, σ2) where {F,P,T}
    lo, hi = plan.lower, plan.upper
    y = plan.y

    # ① The noise-free data, and their fit started at the truth (projected into the bounds):
    #    the package's own LM (`levmar`, levmar.jl), with tight tolerances, since the bias is a
    #    small difference of two values and needs every digit. `relabel` as `fit` does.
    θres = _resolve_ties(truth)
    θt = SVector{P,T}(ntuple(j -> T(getfield(θres, F[j])), Val(P)))
    signal!(y, m, θres)
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
        C = zero(SMatrix{P,P,T})
        @inbounds for i in eachindex(plan.hbuf)
            a = Hinv * (u .* SVector{P,T}(partials(value(plan.hbuf[i])).values))
            C += _at(σ2, i) * (a * a')
        end
    else
        # Eq. 3, (I + ∇b)F⁻¹(I + ∇b)ᵀ. J_t: the true signal's Jacobian at the truth, the truth's
        # own ties followed (R1m = R1f moves with R1f, as the MATLAB's `R1 + eps` on both).
        # I + ∇b = H⁻¹J₀ᵀJ_t (implicit function theorem), rows on a bound 0.
        signal!(plan.buf, m, _resolve_ties(merge(truth, NamedTuple{F}(Tuple(_seed(θt))))))
        K = zero(SMatrix{P,P,T})               # J₀ᵀJ_t
        Fm = zero(SMatrix{P,P,T})              # F = J_tᵀΣ⁻¹J_t
        @inbounds for i in eachindex(plan.buf)
            Jt = SVector{P,T}(partials(plan.buf[i]).values)
            J0 = u .* SVector{P,T}(partials(value(plan.hbuf[i])).values)
            K += J0 * Jt'
            Fm += (Jt * Jt') / _at(σ2, i)
        end
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

# The noise SD of each observation: from `snr` (relative to the scale parameter, as `simulate`)
# or from `σ` (a number, or one per observation).
function _crlb_sigma(m, θ, snr, σ, N)
    (snr === nothing) == (σ === nothing) &&
        throw(ArgumentError("give the noise level as `snr` or as `σ` (one of them)"))
    if snr !== nothing
        scale = _scalename(m)
        scale === nothing && throw(ArgumentError("$(nameof(typeof(m))) has no " *
                                   "`Free(start = :scale)` parameter to define the SNR; give `σ`"))
        return fill(Float64(θ[scale]) / snr, N)
    end
    σ isa Real && return fill(Float64(σ), N)
    length(σ) == N ||
        throw(DimensionMismatch("`σ` has $(length(σ)) values; the model has $N observations"))
    return Float64.(collect(σ))
end

# -------------------------------------------------------------------------------------------------
# Display: at the REPL a table, one row per free parameter, in the style of `compare`
# -------------------------------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", c::CRLBResult)
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    σ = c.σ
    noise = all(==(σ[1]), σ) ? "σ = $(f(σ[1]))" : "σ = $(f(minimum(σ))) to $(f(maximum(σ)))"
    how = c.method === :sandwich ? "sandwich (the least-squares fit)" :
                                   "biased CRLB (Dortch 2018 Eq. 3)"
    print(io, "crlb for ", nameof(typeof(c.model)), ", ", length(σ), " observations, ", noise,
          ", covariance: ", how)
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
# `montecarlo`'s display.
function _print_fixed(io::IO, fixed::NamedTuple, truth::NamedTuple)
    isempty(fixed) && return
    item(n, x) = (t = truth[n]; s = "$n = $(_showvalue(x))";
                  x isa Real && t isa Real && x != t ? s * " (true $(_showvalue(t)))" : s)
    print(io, "\nfixed in the fit: ", join((item(n, x) for (n, x) in pairs(fixed)), ", "))
end

Base.show(io::IO, c::CRLBResult) =
    print(io, "CRLBResult(", nameof(typeof(c.model)), ": ", join(keys(c.fit), ", "), ")")
