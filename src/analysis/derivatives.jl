# =================================================================================================
# crlb's pieces, one function each, fast: a cache made once per fit setting, and functions that
# sum static P×P matrices over the observations and allocate nothing
#
#   c = CRLBCache(m; fit, cov)          once: the fit's free parameters (in the type), bounds, fixed
#                                       values, buffers; what `init` is for fitting
#   noisefree!(c, m, t)                 θ̂: the fit of the noise-free data, started at the truth
#   hessian!(c, m, θ̂, exact)            H = Σᵢ JᵢJᵢᵀ + rᵢ∇²gᵢ and G = Σᵢ JᵢJᵢᵀ, at θ̂
#   attruth!(buf, c, m, t, θ)           the rows Jᵢ of the true signal's Jacobian, at the truth
#   meat(rows, σ², u)                   J₀ᵀΣJ₀
#   information(rows, σ²)               JᵀΣ⁻¹J, the Fisher information (Dortch 2018, Eq. 4)
#   crossgram(rows₀, rowsₜ, u)           J₀ᵀJ_t
#   curvature(rows, w)                  Σᵢ wᵢ∇²gᵢ
#   sandwich(H, M)                      H⁻¹MH⁻¹
#   biasgradient(H, K)                  ∇b = H⁻¹K − I, K = J₀ᵀJ_t (the implicit function theorem)
#   biasedcrlb(∇b, F)                   (I + ∇b)F⁻¹(I + ∇b)ᵀ (Dortch et al., MRM 2018, Eq. 3)
#
# How: one evaluation of the model with dual numbers gives every observation's value gᵢ with its
# gradient Jᵢ, a row of J (with nested duals, ∇²gᵢ too), and each sum is accumulated into a static
# P×P matrix one row at a time, as the solver's `normal_equations` sums JᵀJ (levmar.jl). The N×P
# matrix J is never stored, nothing is allocated, and the compiler knows every size (P is in the
# cache's type). `crlb`'s kernel (crlb.jl) is their composition. The textbook versions
# (ForwardDiff.jacobian and ForwardDiff.hessian on `signal`, plain matrices, `\`) are in the
# tests, as the reference these are checked against. Measurements: docs/design.md, "crlb".
#
# From a model and values, one call each (a cache made inside; allocating results, for the REPL):
#
#   noisecovariance(m, θ; snr | σ | Σ)     Σ (N×N)
#   noisefree(m, θ; fit)                   θ̂, a FitResult;  bias(m, θ; fit): θ̂ − θ
#   jacobian(m, θ; fit), jacobian(r)       J (N×P)
#   hessian(m, θ, y; fit, Σ), hessian(r, y)
#   informationmatrix(m, θ; noise, fit)    JᵀΣ⁻¹J, or the observed information
#   biasgradient(m, θ; fit, method)        ∇b (P×P)
#
# θ is always every parameter's value (numbers, or ties), as for `signal`: e.g.
# `nominal(m, :SIR_WM_I)`. `fit` is a fit's settings, the keywords of `fit` in a named tuple
# (`kmf = Free()` frees kmf, `kmf = 15.0` fixes it). What runs a fit (`noisefree`, `bias`,
# `biasgradient`) uses all of it: the fixed values are the fit's assumptions, and the truth gives
# only the start. The derivatives (`jacobian`, `hessian`, `informationmatrix`) are taken AT θ:
# `fit` says which parameters are free (the columns of J), and every value is θ's. So with a fit
# that fixes kmf at 12.5 for a tissue with kmf = 5, `jacobian(m, θ; fit)` is the true signal's
# J_t (kmf = 5) and `jacobian(noisefree(m, θ; fit))` the fit's J₀ at its answer (kmf = 12.5).
# =================================================================================================

# The noise-free fit's solver: the step tolerance alone, and room to get there. 1e-10 gives θ̂ to
# a few 1e-9 (relative), about as well as the flat minimum defines it; 1e-13 took up to 2.6 times
# the iterations for digits that were not real (docs/design.md, "crlb").
const _CRLB_SOLVER = LM(maxiter = 1000, xtol = 1e-10, ftol = 0.0, gtol = 0.0)

# The noise-free fit is exact (it reproduces the data) when ‖r‖ ≤ 1e-12 ‖y‖: every case where
# nothing fixed is wrong, and schemes with as many observations as free parameters (4-point SIR).
# Then H = JᵀJ (the curvature term Σᵢ rᵢ∇²gᵢ is 0), and first-order duals suffice.
const _EXACT = 1e-12

# -------------------------------------------------------------------------------------------------
# The cache
# -------------------------------------------------------------------------------------------------

"""
    CRLBCache(m; fit = (...), cov = :sandwich)

What [`crlb`](@ref) needs that depends on the fit settings but not on the truth or the noise,
made once: the fit's free parameters (in the type, so the compiler knows how many), their
bounds, the fixed values as written (`kmf = 12.5`, `R1m = :R1f`), the solver settings of the
noise-free fit and the buffers. What [`init`](@ref) is for fitting. Reuse it with
[`crlb!`](@ref) for many truths or noise levels (several tissues, a sweep): the setup is most
of a one-off call's time. Not thread-safe: one cache per task.

```julia
c = CRLBCache(m; fit = (kmf = Free(),))
[crlb!(c, nominal(m, t); snr = 200) for t in (:SIR_WM_I, :SIR_GM)]
```
"""
struct CRLBCache{F,P,T,M,MT,X,B1,B2}
    model::M                   # the model, as given (kept in the results)
    mT::MT                     # the same, acquisition in precision T (what is computed with)
    lower::SVector{P,T}
    upper::SVector{P,T}
    fixed::X                   # the fit's fixed values: numbers and ties (`_Tie`)
    opts::SolverOptions{T}
    biased::Bool               # Eq. 3 (`cov = :biased`) rather than the sandwich
    y::Vector{T}               # the noise-free data
    fitbuf::B1                 # Vector{Dual{FitTag,T,P}}: the noise-free fit; then J_t
    jbuf::B1                   # J₀ at θ̂ (first-order duals), when the fit is exact
    hbuf::B2                   # Vector{Dual{CRLBTag,Dual{FitTag,T,P},P}}: J₀ and ∇²g at θ̂
end

function CRLBCache(m::MRIModel; fit::NamedTuple = (;), cov::Symbol = :sandwich)
    cov in (:sandwich, :biased) ||
        throw(ArgumentError("`cov` must be :sandwich or :biased, got $(repr(cov))"))
    _check_fit_settings(fit)
    mT = _with_precision(m, Float64)           # acquisition vectors as Float64, as `fit` does
    return _crlbcache(m, mT, setup(mT, Float64; fit...), cov === :biased)
end
# A separate function so that P is a type parameter, to make the buffers' types.
function _crlbcache(m, mT, spec::FitSpec{F,P,T}, biased::Bool) where {F,P,T}
    n = nobs(mT)
    D1 = Dual{FitTag,T,P}
    D2 = Dual{CRLBTag,D1,P}
    return CRLBCache{F,P,T,typeof(m),typeof(mT),typeof(spec.fixed),Vector{D1},Vector{D2}}(
        m, mT, spec.lower, spec.upper, spec.fixed, SolverOptions{T}(_CRLB_SOLVER), biased,
        Vector{T}(undef, n), Vector{D1}(undef, n), Vector{D1}(undef, n), Vector{D2}(undef, n))
end
_crlbnames(::CRLBCache{F}) where {F} = F

Base.show(io::IO, c::CRLBCache{F}) where {F} =
    print(io, "CRLBCache(", nameof(typeof(c.model)), ": ", join(F, ", "), "; ",
          c.biased ? "Eq. 3" : "sandwich", ")")

# The truth, checked (every parameter, no roles; see `resolve`), in model order, with ties by
# name as `_Tie` so the functions below read them as plain field accesses. Done once per truth: a
# tie's type depends on the name it holds, which is only known at run time here.
function _crlb_truth(m::MRIModel, truth::NamedTuple)
    resolve(m, truth)
    names = keys(parameters(m))
    return NamedTuple{names}(map(n -> _tie(truth[n]), names))
end

# The free values of a truth from `_crlb_truth`, as a static vector in the order F.
@inline function _freevalues(c::CRLBCache{F,P,T}, t::NamedTuple) where {F,P,T}
    θres = _resolve_ties(t)
    return SVector{P,T}(ntuple(j -> T(getfield(θres, F[j])), Val(P)))
end

# -------------------------------------------------------------------------------------------------
# Dual numbers: one evaluation of the model gives the rows of J (and ∇²g)
# -------------------------------------------------------------------------------------------------

# A second tag for the outer layer of nested dual numbers (second derivatives), distinct from
# the solver's FitTag (first derivatives).
struct CRLBTag end

# Nested dual numbers: θⱼ seeded twice, so one evaluation of the model gives each observation's
# value gᵢ, its gradient Jᵢ and its second derivatives ∇²gᵢ (see `_second`).
@inline function _seed2(θ::SVector{P,T}) where {P,T}
    inner = _seed(θ)                           # Dual{FitTag}: value θⱼ, partials eⱼ
    D = eltype(inner)
    return SVector(ntuple(j -> Dual{CRLBTag}(inner[j], ntuple(k -> ifelse(k == j, one(D), zero(D)),
                                                             Val(P))), Val(P)))
end
# The seed for a buffer's kind of dual: first order, or nested.
@inline _seedfor(::AbstractVector{<:Dual{FitTag}}, θ) = _seed(θ)
@inline _seedfor(::AbstractVector{<:Dual{CRLBTag}}, θ) = _seed2(θ)

# From one dual: the value gᵢ, the row Jᵢ = ∇gᵢ, and (nested) ∇²gᵢ. For ∇²gᵢ, the outer partial
# k is ∂gᵢ/∂θₖ, itself a dual whose partial l is ∂²gᵢ/∂θₖ∂θₗ; `SMatrix` takes the P² entries
# column by column.
@inline _val(d::Dual{FitTag}) = value(d)
@inline _val(d::Dual{CRLBTag}) = value(value(d))
@inline _row(d::Dual{FitTag,T,P}) where {T,P} = SVector{P,T}(partials(d).values)
@inline _row(d::Dual{CRLBTag}) = _row(value(d))
@inline function _second(d::Dual{CRLBTag,Dual{FitTag,T,P},P}) where {T,P}
    ps = partials(d).values
    return SMatrix{P,P,T}(ntuple(n -> partials(ps[(n - 1) ÷ P + 1]).values[(n - 1) % P + 1],
                                 Val(P * P)))
end
# A zero P×P matrix for a buffer's rows.
@inline _zeroPP(::AbstractVector{Dual{FitTag,T,P}}) where {T,P} = zero(SMatrix{P,P,T})
@inline _zeroPP(::AbstractVector{Dual{CRLBTag,Dual{FitTag,T,P},P}}) where {T,P} = zero(SMatrix{P,P,T})

# Evaluate the model into `buf` (first-order or nested duals) at the free values θ, the other
# values the fit's (`atfit!`: its fixed values and ties, as in the fit) or the truth's
# (`attruth!`: its own ties followed, so R1m = :R1f in the truth moves with R1f, as the MATLAB's
# `R1 + eps` on both).
@inline atfit!(buf, c::CRLBCache{F}, m, θ) where {F} =
    signal!(buf, m, _params(Val(F), _seedfor(buf, θ), c.fixed))
@inline attruth!(buf, c::CRLBCache{F}, m, t, θ) where {F} =
    signal!(buf, m, _resolve_ties(merge(t, NamedTuple{F}(Tuple(_seedfor(buf, θ))))))

# -------------------------------------------------------------------------------------------------
# ① The noise-free fit, ② the Hessian of its cost
# -------------------------------------------------------------------------------------------------

"""
    noisefree!(c, m, t, dm, dt) -> (θ̂, θt, r)

The noise-free data g(θ) into `c.y` (from `dm` at `dt`: the fitted model `m` and its truth `t`,
or another model's, `simulate_with`), and their fit: the package's solver ([`levmar`](@ref))
started at the truth's free values θt, with the fit's fixed values and bounds. Returns θ̂
(`relabel`ed, as `fit` does), θt and the solver's result (`r.sse`, `r.status`). Truths from
`_crlb_truth`. No allocation.
"""
@inline function noisefree!(c::CRLBCache{F,P,T}, m, t, dm, dt) where {F,P,T}
    signal!(c.y, dm, _resolve_ties(dt))
    θt = _freevalues(c, t)
    r = levmar(VoxelProblem{F}(m, c.y, c.fixed), θt, c.lower, c.upper, c.opts, c.fitbuf)
    θ̂ = SVector{P,T}(Tuple(relabel(m, NamedTuple{F}(Tuple(r.θ)))))
    return θ̂, θt, r
end

"""
    hessian!(c, m, θ̂, exact) -> (H, G)

The Hessian of the fit's cost ½‖g − y‖² at θ̂, H = Σᵢ (JᵢJᵢᵀ + rᵢ∇²gᵢ), rᵢ = gᵢ(θ̂) − yᵢ, and its
Gauss–Newton part G = Σᵢ JᵢJᵢᵀ = J₀ᵀJ₀ (static P×P). The second term, the residual's curvature,
is what a wrong assumption adds; when the fit is `exact` (r = 0) it is zero, H = G, and the
model is evaluated with first-order duals (into `c.jbuf`) instead of nested ones (`c.hbuf`,
3–6× the cost). The rows J₀ᵢ stay in that buffer for `meat` and `crossgram`. No allocation.
"""
@inline function hessian!(c::CRLBCache{F,P,T}, m, θ̂, exact::Bool) where {F,P,T}
    if exact
        atfit!(c.jbuf, c, m, θ̂)
        G = information(c.jbuf, one(T))
        return G, G
    end
    atfit!(c.hbuf, c, m, θ̂)
    H = zero(SMatrix{P,P,T})
    G = zero(SMatrix{P,P,T})
    @inbounds for i in eachindex(c.hbuf)
        d = c.hbuf[i]
        Jᵢ = _row(d)
        G += Jᵢ * Jᵢ'
        H += Jᵢ * Jᵢ' + (_val(d) - c.y[i]) * _second(d)
    end
    return H, G
end

# -------------------------------------------------------------------------------------------------
# The sums that involve the noise, by its form (Julia picks the method by the type of σ²): one
# variance for every observation (a number: "pulled out"), one per observation (a vector), or a
# full covariance Σ = LLᵀ (`_FullNoise`, whitening the rows). Sums of squares, so neither gives
# a negative variance. `u` masks the rows of parameters held on a bound (1 for the others).
# -------------------------------------------------------------------------------------------------

# The scale parameter's value (M0, S0): read directly when given as a number; otherwise (a tie,
# or missing: then `resolve` says what is wrong) through `resolve`, which checks every value.
_scalevalue(m, θ, scale) =
    haskey(θ, scale) && θ[scale] isa Real ? Float64(θ[scale]) : Float64(resolve(m, θ)[scale])

# A full noise covariance by its Cholesky factor (Σ = LLᵀ, lower triangle), with a work buffer
# for the whitening (one row of L⁻¹J per observation).
struct _FullNoise{T,P}
    L::Matrix{T}
    v::Vector{SVector{P,T}}
end

@inline _at(s::Real, i) = s                          # σ²: one number for every observation,
@inline _at(s::AbstractVector, i) = @inbounds s[i]   # or one per observation

"""
    meat(rows, σ², u) -> SMatrix{P,P}

The sandwich's "meat" J₀ᵀΣJ₀: Σᵢ σᵢ² JᵢJᵢᵀ, or with a full Σ = LLᵀ, Σₖ wₖwₖᵀ with wₖ = row k of
LᵀJ₀ = Σᵢ≥ₖ LᵢₖJᵢ (O(N²P), N ≤ 100). `rows`: a buffer evaluated by `hessian!`.
"""
@inline function meat(buf, σ2::Union{Real,AbstractVector}, u::SVector{P,T}) where {P,T}
    M = zero(SMatrix{P,P,T})
    @inbounds for i in eachindex(buf)
        Jᵢ = u .* _row(buf[i])
        M += _at(σ2, i) * (Jᵢ * Jᵢ')
    end
    return M
end
function meat(buf, n::_FullNoise, u::SVector{P,T}) where {P,T}
    L = n.L
    M = zero(SMatrix{P,P,T})
    @inbounds for k in eachindex(buf)
        w = zero(SVector{P,T})
        for i in k:length(buf)
            w += L[i, k] * _row(buf[i])
        end
        w = u .* w
        M += w * w'
    end
    return M
end

"""
    information(rows, σ²) -> SMatrix{P,P}

The Fisher information JᵀΣ⁻¹J (Dortch et al., MRM 2018, Eq. 4): Σᵢ JᵢJᵢᵀ/σᵢ², or with a full
Σ = LLᵀ, Σₖ vₖvₖᵀ with vₖ = row k of L⁻¹J (forward substitution, the rows kept in the noise's
buffer). With σ² = 1, the Gauss–Newton matrix JᵀJ.
"""
@inline function information(buf, σ2::Union{Real,AbstractVector})
    Fm = _zeroPP(buf)
    @inbounds for i in eachindex(buf)
        Jᵢ = _row(buf[i])
        Fm += (Jᵢ * Jᵢ') / _at(σ2, i)
    end
    return Fm
end
function information(buf, n::_FullNoise)
    L, v = n.L, n.v
    Fm = _zeroPP(buf)
    @inbounds for k in eachindex(buf)
        t = _row(buf[k])
        for i in 1:k-1
            t -= L[k, i] * v[i]
        end
        v[k] = t / L[k, k]
        Fm += v[k] * v[k]'
    end
    return Fm
end

"""
    crossgram(rows₀, rowsₜ, u) -> SMatrix{P,P}

K = J₀ᵀJ_t = Σᵢ J₀ᵢJ_tᵢᵀ: the fit's Jacobian at θ̂ against the true signal's at the truth (J₀'s
rows masked by `u`), for the bias gradient.
"""
@inline function crossgram(b0, bt, u::SVector{P,T}) where {P,T}
    K = zero(SMatrix{P,P,T})
    @inbounds for i in eachindex(b0)
        K += (u .* _row(b0[i])) * _row(bt[i])'
    end
    return K
end

"""
    curvature(rows, w) -> SMatrix{P,P}

Σᵢ wᵢ∇²gᵢ, from nested duals: the second term of a Hessian (w = r for the fit's cost, w = Σ⁻¹r
for the negative log-likelihood).
"""
@inline function curvature(buf, w)
    S = _zeroPP(buf)
    @inbounds for i in eachindex(buf)
        S += w[i] * _second(buf[i])
    end
    return S
end

# -------------------------------------------------------------------------------------------------
# ③ The formulas: on static P×P matrices (the kernel), and on ordinary matrices (at the REPL)
# -------------------------------------------------------------------------------------------------

"""
    sandwich(H, M) -> matrix
    sandwich(H, J, Σ) -> Matrix

The covariance of a least-squares fit, first order in the noise: H⁻¹MH⁻¹ with the "meat"
M = JᵀΣJ, J the Jacobian at the fit's answer, H the Hessian of its cost ½‖g − y‖² there
([`hessian`](@ref)) and Σ the noise covariance ([`noisecovariance`](@ref)). Noise ε moves the
fit by δ = −H⁻¹Jᵀε (expand the fit's zero-gradient condition), so Cov(δ) is the meat (how hard
the noise pushes) between two slices of "bread" H⁻¹ (how stiff the cost is). Huber–White
(White, Econometrica 1982;50:1); the misspecified CRB (Fortunati et al., IEEE Signal Process
Mag 2017;34(6):142). With H = JᵀJ (the fit exact) and Σ = σ²I: the classical σ²(JᵀJ)⁻¹.
"""
@inline function sandwich(H::SMatrix{P,P,T}, M::SMatrix{P,P,T}) where {P,T}
    Hi = _chol_inv(first(_cholesky(H)))
    C = Hi * M * Hi
    return (C + C') / 2
end
sandwich(H::AbstractMatrix, M::AbstractMatrix) = _sym(H \ M / H)
sandwich(H::AbstractMatrix, J::AbstractMatrix, Σ) = sandwich(H, J' * Σ * J)

"""
    biasgradient(H, K) -> matrix

The bias gradient from the derivatives at the noise-free fit's answer θ̂: the fit's gradient
G(θ̂, θ) = J₀ᵀ[g(θ̂) − g(θ)] is 0 for every truth θ, so differentiating it (the implicit
function theorem) gives I + ∇b = ∂θ̂/∂θ = H⁻¹K, K = J₀ᵀJ_t ([`crossgram`](@ref)), and
∇b = H⁻¹K − I. One linear solve, no refits of perturbed data. A parameter held on a bound (its
row and column of H masked to the identity, its row of K 0) gets the row −eⱼ: ∂θ̂ⱼ/∂θ = 0.
"""
@inline biasgradient(H::SMatrix{P,P,T}, K::SMatrix{P,P,T}) where {P,T} =
    _chol_inv(first(_cholesky(H))) * K - one(SMatrix{P,P,T})
biasgradient(H::AbstractMatrix, K::AbstractMatrix) = H \ K - I

"""
    biasedcrlb(∇b, F) -> matrix

The covariance bound of Dortch et al., MRM 2018, Eq. 3, for any estimator with the bias
gradient ∇b ([`biasgradient`](@ref)): (I + ∇b)F⁻¹(I + ∇b)ᵀ, with F the Fisher information at
the truth ([`informationmatrix`](@ref)). With ∇b = 0, the classical F⁻¹. Computed as XᵀX,
X = L⁻¹(I + ∇b)ᵀ with F = LLᵀ: a sum of squares, so never a negative variance.
"""
@inline function biasedcrlb(∇b::SMatrix{P,P,T}, F::SMatrix{P,P,T}) where {P,T}
    L = first(_cholesky(F))
    D = one(SMatrix{P,P,T}) + ∇b
    X = hcat(ntuple(j -> _fwd_solve(L, D[j, :]), Val(P))...)   # L⁻¹Dᵀ
    return X' * X
end
function biasedcrlb(∇b::AbstractMatrix, F::AbstractMatrix)
    X = cholesky(Symmetric(F)).L \ (I + ∇b)'
    return X' * X
end

# Symmetric part, for sums that come out symmetric only to rounding.
_sym(A) = (A + A') / 2

# Can the data tell the parameters apart? The Cholesky pivots of G = JᵀJ (or JᵀΣ⁻¹J) with unit
# diagonal (each column of J normalized) are the squared sines of the angles between each column
# and the span of the ones before it. A pivot below `_PIVOT_MIN` (an angle below 1e-6 rad: the
# condition number of J is then at least ~1e6, a relative SD of ~1e6 × noise/signal) means a
# parameter is a combination of the others, to within what rounding in JᵀJ can resolve. A
# parameter with no influence at all (a zero column) is undetermined too. Why test: a singular
# H can look positive definite after rounding, and the covariance then has finite, meaningless
# values. `u` masks parameters on a bound (the observation count is checked by the caller).
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

# A parameter on a bound (within 1e-9, relative for large bounds).
@inline _near(x, bound) = isfinite(bound) && abs(x - bound) <= 1e-9 * max(1, abs(bound))

# -------------------------------------------------------------------------------------------------
# The noise
# -------------------------------------------------------------------------------------------------

"""
    noisecovariance(m, θ; snr or σ or Σ) -> Diagonal or Matrix

The covariance Σ of Gaussian noise between the observations of `m` (N×N), from one of:

- `snr`: relative to the model's scale parameter at the values θ, as for [`simulate`](@ref):
  σ = M0/snr, the same for every observation, independent: Σ = σ²I (a `Diagonal`)
- `σ`: a number (Σ = σ²I), or one SD per observation (Σ = Diagonal(σ²): independent noise, e.g.
  different numbers of averages)
- `Σ`: the covariance matrix itself (correlated noise, e.g. after denoising), checked
  (symmetric, positive definite) and returned

```julia
noisecovariance(m, nominal(m, :SIR_WM_I); snr = 200)       # 4×4 Diagonal, (M0f/200)²
```
"""
function noisecovariance(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, Σ = nothing)
    N = nobs(m)
    count(!isnothing, (snr, σ, Σ)) == 1 ||
        throw(ArgumentError("give the noise level as `snr`, `σ` or `Σ` (one of them)"))
    if snr !== nothing
        scale = _scalename(m)
        scale === nothing && throw(ArgumentError("$(nameof(typeof(m))) has no " *
                                   "`Free(start = :scale)` parameter to define the SNR; give `σ`"))
        s = _scalevalue(m, θ, scale) / snr            # σ = M0/snr
        return Diagonal(fill(s^2, N))
    elseif σ isa Real
        return Diagonal(fill(Float64(σ)^2, N))
    elseif σ !== nothing
        σ isa AbstractVector || throw(ArgumentError("`σ` is a number or one SD per observation; " *
                                                    "for a covariance matrix, give `Σ`"))
        length(σ) == N ||
            throw(DimensionMismatch("`σ` has $(length(σ)) values; the model has $N observations"))
        return Diagonal(Float64.(collect(σ)) .^ 2)
    end
    _noisefactor(Σ, N)                    # the checks `simulate` makes (sim/simulate.jl)
    return Σ
end

# Σ checked and factored as `simulate` does (`_noisefactor`, sim/simulate.jl).
_fullnoise(Σ, N, P) =
    _FullNoise{Float64,P}(_noisefactor(Σ, N), Vector{SVector{P,Float64}}(undef, N))

# The noise in the form the sums take, from `noisecovariance`: one variance for every
# observation (σ² "pulled out", the common case), one per observation, or the full covariance.
_kernelnoise(Σ::Diagonal, P) = allequal(Σ.diag) ? Float64(Σ.diag[1]) : Vector{Float64}(Σ.diag)
_kernelnoise(Σ::AbstractMatrix, P) = _fullnoise(Σ, size(Σ, 1), P)
_kernelnoise(Σ::UniformScaling, P) = Float64(Σ.λ)

# -------------------------------------------------------------------------------------------------
# From a model and values, one call each (a cache made inside; allocating results)
# -------------------------------------------------------------------------------------------------

"""
    noisefree(m, θ; fit = (...), simulate_with) -> FitResult

The fit of the noise-free data `signal(m, θ)` with the fit settings `fit` (the keywords of
[`fit`](@ref) in a named tuple; the model's own roles when left out), started at θ's free
values, with the fit's bounds and a tight tolerance. Its values θ̂ (`r.params`) are where noisy
fits center (the "pseudo-true" values): θ itself unless a fixed value of the fit differs from
θ's (`fit = (kmf = 12.5,)` for a tissue with kmf = 5), or the data come from another model
(`simulate_with = m2 => θ2`, as for [`crlb`](@ref)). A [`FitResult`](@ref): [`bias`](@ref),
[`jacobian`](@ref) and [`hessian`](@ref) take it.
"""
function noisefree(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;), simulate_with = nothing)
    dm, dθ = _datasource(m, θ, simulate_with)
    c = CRLBCache(m; fit)
    dmT = simulate_with === nothing ? c.mT : _with_precision(dm, Float64)
    return _noisefree(c, _crlb_truth(m, θ), dmT, _crlb_truth(dm, dθ), fit)
end
function _noisefree(c::CRLBCache{F}, t, dm, dt, fit) where {F}
    θ̂, _, r = noisefree!(c, c.mT, t, dm, dt)
    settings = (; precision = Float64, solver = _CRLB_SOLVER, fit...)   # as `fit` keeps them
    return FitResult(NamedTuple{F}(Tuple(θ̂)), map(_untie, c.fixed), r.sse, r.iterations,
                     r.status, c.mT, settings)
end

"""
    bias(m, θ; fit = (...), simulate_with) -> named tuple
    bias(r::FitResult, θ) -> named tuple

The bias b = θ̂ − θ of each free parameter of the fit: the noise-free fit's values
([`noisefree`](@ref)) minus the truth θ. Zero unless the fit fixes a parameter at a wrong value
or the data come from another model (`simulate_with`). The second form takes the noise-free fit
itself.
"""
bias(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;), simulate_with = nothing) =
    bias(noisefree(m, θ; fit, simulate_with), θ)
function bias(r::FitResult, θ::NamedTuple)
    θt = resolve(r.model, θ)
    F = keys(r.params)
    return NamedTuple{F}(map(n -> r.params[n] - θt[n], F))
end

# The cache of a one-voxel fit's own settings: its free parameters, fixed values and bounds.
function _cacheof(r::FitResult)
    _onevoxel(r) || throw(ArgumentError("give a one-voxel FitResult"))
    return CRLBCache(r.model; fit = Base.structdiff(r.settings, NamedTuple{_OPTIONS}))
end
# A fit's values in the cache's order, as a static vector.
_freevalues(c::CRLBCache{F,P,T}, r::FitResult) where {F,P,T} =
    SVector{P,T}(ntuple(j -> T(getfield(r.params, F[j])), Val(P)))

# A buffer's rows as an N×P matrix.
function _matrix(buf)
    P = length(_row(buf[1]))
    return [_row(buf[i])[j] for i in eachindex(buf), j in 1:P]
end

"""
    jacobian(m, θ; fit = (...)) -> Matrix (N×P)
    jacobian(r::FitResult) -> Matrix

The Jacobian J = ∂g/∂θ of the signal g = `signal(m, θ)`: row i observation i, column j the
j-th free parameter of a fit with settings `fit` (in model order, as `keys(r.params)`; the
model's free parameters when left out). At the values θ, every one of them θ's: `fit` chooses
the columns, not the point. Ties in θ are followed: with `R1m = :R1f`, the R1f column includes
R1m's change. Dual numbers, exact to rounding.

The second form: at a one-voxel fit's answer, with its free parameters, fixed values and ties
(e.g. `jacobian(noisefree(m, θ; fit))`, the J₀ of [`crlb`](@ref)).

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)
J = jacobian(m, θ)                           # 4×4: PSR, R1f, Sf, M0f
J5 = jacobian(m, θ; fit = (kmf = Free(),))   # 4×5: kmf too
```

Also exported by Zygote and DifferentiationInterface: with either loaded, write
`EQUIP.jacobian`.
"""
jacobian(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;)) =
    _jacobian(CRLBCache(m; fit), _crlb_truth(m, θ))
jacobian(r::FitResult) = _jacobian(_cacheof(r), r)
function _jacobian(c::CRLBCache, t::NamedTuple)
    attruth!(c.fitbuf, c, c.mT, t, _freevalues(c, t))
    return _matrix(c.fitbuf)
end
function _jacobian(c::CRLBCache, r::FitResult)
    atfit!(c.jbuf, c, c.mT, _freevalues(c, r))
    return _matrix(c.jbuf)
end

"""
    hessian(m, θ, y; fit = (...), Σ = I) -> Matrix (P×P)
    hessian(r::FitResult, y; Σ = I) -> Matrix

The Hessian of the cost ½ rᵀΣ⁻¹r, r = g(θ) − y the residual of the data `y`, with respect to the
free parameters (as for [`jacobian`](@ref)), at θ:

    H = JᵀΣ⁻¹J + Σᵢ (Σ⁻¹r)ᵢ ∇²gᵢ

With `Σ = I` (the default), the cost the fit minimizes, ½‖g − y‖², and H = JᵀJ + Σᵢ rᵢ∇²gᵢ:
JᵀJ is the Gauss–Newton part the solver uses; the second, the residual's curvature, is zero
when the model fits `y` exactly and not when an assumption is wrong. With Σ the noise
covariance, the negative log-likelihood's Hessian: the observed information. Nested dual
numbers, exact to rounding.

The second form: at a one-voxel fit's answer, e.g. `hessian(noisefree(m, θ; fit), signal(m, θ))`,
the H of [`crlb`](@ref).
"""
hessian(m::MRIModel, θ::NamedTuple, y::AbstractVector; fit::NamedTuple = (;), Σ = I) =
    _hessian(CRLBCache(m; fit), _crlb_truth(m, θ), y, Σ)
hessian(r::FitResult, y::AbstractVector; Σ = I) = _hessian(_cacheof(r), r, y, Σ)
function _hessian(c::CRLBCache{F,P}, at, y, Σ) where {F,P}
    length(y) == length(c.y) ||
        throw(DimensionMismatch("`y` has $(length(y)) values; the model has $(length(c.y)) observations"))
    at isa FitResult ? atfit!(c.hbuf, c, c.mT, _freevalues(c, at)) :
                       attruth!(c.hbuf, c, c.mT, at, _freevalues(c, at))
    r = [_val(c.hbuf[i]) - y[i] for i in eachindex(y)]
    H = information(c.hbuf, _kernelnoise(Σ, P)) + curvature(c.hbuf, Σ \ r)
    return Matrix(_sym(H))
end

"""
    informationmatrix(m, θ; snr or σ or Σ, fit = (...), expected = true, y) -> Matrix (P×P)

The information matrix of the free parameters (as for [`jacobian`](@ref)) at the values θ, for
Gaussian noise (the noise as for [`noisecovariance`](@ref)). `expected = true`: the Fisher
information F = JᵀΣ⁻¹J (Dortch et al., MRM 2018, Eq. 4: JᵀJ/σ² for one σ); its inverse is the
CRLB of an unbiased estimator, and at the truth it is the F of Eq. 3. `expected = false`: the
observed information, the Hessian of the negative log-likelihood of the data `y`
([`hessian`](@ref) with Σ), which adds the residual's curvature; `y` is the noise-free signal at
θ when left out, and then the two are equal. StatsAPI's function, with EQUIP's methods.

```julia
F = informationmatrix(m, θ; snr = 200)       # at the truth
sqrt.(diag(inv(F)))                          # the classical CRLB of each SD
```
"""
function informationmatrix(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
                           fit::NamedTuple = (;), expected::Bool = true, y = nothing)
    Σn = noisecovariance(m, θ; snr, σ, Σ)
    expected || return hessian(m, θ, something(y, signal(m, θ)); fit, Σ = Σn)
    return _information(CRLBCache(m; fit), _crlb_truth(m, θ), Σn)
end
function _information(c::CRLBCache{F,P}, t, Σn) where {F,P}
    attruth!(c.fitbuf, c, c.mT, t, _freevalues(c, t))
    return Matrix(information(c.fitbuf, _kernelnoise(Σn, P)))
end

"""
    biasgradient(m, θ; fit = (...), method = :implicit, steps = 3) -> Matrix (P×P)

The bias gradient ∇b = ∂b/∂θ: how the bias of each free parameter of the fit (row) changes with
the true value of each (column), in Eq. 3 of Dortch et al., MRM 2018 (see [`biasedcrlb`](@ref)).
Zero when nothing fixed is wrong. Two ways, which agree (tested):

- `:implicit` (default): the implicit function theorem at the noise-free fit's answer θ̂,
  ∇b = H⁻¹J₀ᵀJ_t − I (see [`biasgradient(H, K)`](@ref biasgradient)): `crlb`'s own kernel, with
  `cov = :biased`. A parameter that ends on a bound stays there: its row of I + ∇b is 0.
- `:unrolled`: ForwardDiff carried through `steps` Gauss–Newton steps of the noise-free fit
  from the truth, θ̂ ← θ̂ − (JᵀJ)⁻¹Jᵀ(g(θ̂) − g(θ)): the derivative of the iterations
  themselves. No bounds; accurate when a few steps reach the fit (3 for the 4-point scheme,
  kmf assumed 12.5: to 1e-10; the 5- and 16-point schemes need about 10). The form to use where
  derivatives with respect to the protocol are needed too.

`NaN` when the data cannot determine the parameters (`:implicit`).
"""
function biasgradient(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;), method::Symbol = :implicit,
                      steps::Integer = 3)
    method in (:implicit, :unrolled) ||
        throw(ArgumentError("`method` must be :implicit or :unrolled, got $(repr(method))"))
    method === :unrolled && return _biasgradient_unrolled(m, θ, fit, steps)
    return _biasgradient(CRLBCache(m; fit, cov = :biased), _crlb_truth(m, θ))
end
_biasgradient(c::CRLBCache{F,P,T}, t) where {F,P,T} = Matrix(_crlb!(c, c.mT, t, one(T)).∇b)

# ∂θ̂/∂θ by differentiating `steps` Gauss–Newton iterations from the truth: the noise-free data
# g(θ) from the truth's free values x (the truth's own fixed values and ties), fitted with the
# fit's fixed values. Nested ForwardDiff on the allocating `signal`: the outer duals carry ∂/∂x
# through every step. A cross-check, and the pattern for derivatives with respect to the protocol.
function _biasgradient_unrolled(m::MRIModel, θ::NamedTuple, fit::NamedTuple, steps::Integer)
    spec = setup(m, Float64; fit...)
    F = freenames(spec)
    θfit = merge(θ, map(_untie, spec.fixed))     # the values the fit assumes
    g(vals, x) = signal(m, merge(vals, NamedTuple{F}(Tuple(x))))
    function θ̂(x)
        y = g(θ, x)                              # the noise-free data, true values x
        φ = x                                    # the fit, started at the truth
        for _ in 1:steps
            J = ForwardDiff.jacobian(p -> g(θfit, p), φ)
            φ = φ - (J' * J) \ (J' * (g(θfit, φ) - y))
        end
        return φ
    end
    θt = resolve(m, θ)
    return ForwardDiff.jacobian(θ̂, [Float64(θt[n]) for n in F]) - I
end

# -------------------------------------------------------------------------------------------------
# Checks shared by crlb, montecarlo and the functions above
# -------------------------------------------------------------------------------------------------

_hasmap(x::AbstractArray) = true
_hasmap(x::Free) = x.start isa AbstractArray
_hasmap(x::Fixed) = x.value isa AbstractArray
_hasmap(x) = false

function _check_fit_settings(fit::NamedTuple)
    for (k, x) in pairs(fit)
        k in _OPTIONS && throw(ArgumentError("`fit = (; $k)`: the noise-free fit sets its own " *
                                             "options; give only parameter settings"))
        _hasmap(x) &&
            throw(ArgumentError("`fit = (; $k)`: crlb is for one voxel; give numbers, not maps"))
    end
end

# Where the data come from: the fitted model at θ, or another model (`simulate_with = m2 =>
# θ2`), with as many observations; no maps. The values themselves are checked once, by
# `_crlb_truth` (every parameter, no roles), which every caller runs next.
function _datasource(m::MRIModel, θ::NamedTuple, simulate_with)
    _check_simulate_with(simulate_with)
    if simulate_with === nothing
        _nomaps(θ)
        return m, θ
    end
    dm, dθ = simulate_with.first, simulate_with.second
    _nomaps(θ)
    _nomaps(dθ)
    nobs(dm) == nobs(m) || throw(DimensionMismatch("the data's model has $(nobs(dm)) " *
        "observations, the fitted model $(nobs(m))"))
    return dm, dθ
end
function _nomaps(θ::NamedTuple)
    any(_hasmap, values(θ)) && throw(ArgumentError("crlb is for one voxel: give numbers, not maps"))
    return nothing
end
_check_simulate_with(::Nothing) = nothing
_check_simulate_with(s) = s isa Pair{<:MRIModel,<:NamedTuple} || throw(ArgumentError(
    "`simulate_with = model => truth`: a model and every value of its parameters, e.g. " *
    "`T2BiexpDecay(TE = te) => (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)`"))
