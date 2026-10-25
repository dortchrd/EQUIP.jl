# =================================================================================================
# The crlb kernel: the same steps as `crlb` (derivatives.jl, crlb.jl), without allocation, for the
# protocol optimizer, which calls it millions of times (5–30 µs a call)
#
#   plan = _crlb_plan(m, cov; fit...)        once: roles, bounds, solver settings, buffers
#   t = _crlb_truth(m, truth)                once per truth: complete, ties as _Tie
#   k = _crlb!(plan, m, t, σ²)               per call: no allocation
#
# What makes it fast is what makes it harder to read: J is never formed. One evaluation of the
# model with nested dual numbers gives each observation's value, gradient (a row of J₀) and
# second derivatives, and the sums JᵀJ, J₀ᵀJ_t, H and the covariance are accumulated from them
# row by row into static matrices on the stack, as the solver's `normal_equations` does
# (levmar.jl). Parameters on a bound are masked (rows and columns 0, 1 on the diagonal) rather
# than removed, so every matrix keeps its static size. The readable functions are the reference:
# the tests check the kernel against `crlb`.
#
# `m` in `_crlb!` may be a different protocol from the one the plan was made with (new TI and
# TD at every step of an optimizer), with the same number of observations and Float64
# acquisition values (static vectors work: `QMTSIR(TI = SVector(...), TD = SVector(...))`).
# σ² is a number, one value per observation, or a full covariance (`_FullNoise`);
# `_kernelnoise(noisecovariance(m, θ; ...), P)` gives it in that form.
# =================================================================================================

# A full noise covariance by its Cholesky factor (Σ = LLᵀ, lower triangle), with a work buffer
# for the kernel's whitening (one row of L⁻¹J per observation).
struct _FullNoise{T,P}
    L::Matrix{T}
    v::Vector{SVector{P,T}}
end

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

# Can the data tell the parameters apart? `_identifiable` (derivatives.jl) on static matrices:
# the Cholesky pivots of J₀ᵀJ₀ with unit diagonal, each at least `_PIVOT_MIN`, and no column of
# zeros. `u` masks parameters on a bound (the observation count is checked by the caller).
@inline function _determined(G::SMatrix{P,P,T}, u::SVector{P,T}) where {P,T}
    g = diag(G)
    any((u .> 0) .& .!(g .> 0)) && return false
    s = map((gj, uj) -> uj > 0 ? inv(sqrt(gj)) : zero(T), g, u)
    L, ok = _cholesky((s * s') .* G + Diagonal(one(T) .- u))
    ok || return false
    return all(j -> u[j] == 0 || L[j, j]^2 >= _PIVOT_MIN, 1:P)
end

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

# Σ checked and factored as `simulate` does (`_noisefactor`, sim/simulate.jl).
_fullnoise(Σ, N, P) =
    _FullNoise{Float64,P}(_noisefactor(Σ, N), Vector{SVector{P,Float64}}(undef, N))

# The noise in the kernel's form, from `noisecovariance`: one variance for every observation
# (σ² "pulled out" of the sums, the common case), one per observation, or the full covariance
# by its Cholesky factor, with a work buffer.
_kernelnoise(Σ::Diagonal, P) = allequal(Σ.diag) ? Σ.diag[1] : Vector{Float64}(Σ.diag)
_kernelnoise(Σ::AbstractMatrix, P) = _fullnoise(Σ, size(Σ, 1), P)
