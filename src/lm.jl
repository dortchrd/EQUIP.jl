# Levenberg–Marquardt for small, dense problems (P ≲ 10 parameters, m ≲ a few hundred points).
#
# Design choices, all aimed at running millions of independent fits:
# - JᵀWJ (P×P), JᵀWr (P) and the SSE are accumulated observation by observation into
#   StaticArrays, so the m×P Jacobian is never stored and nothing touches the heap.
# - Every trial point is evaluated with derivatives. Most LM steps are accepted, and an
#   accepted step then already has its normal equations, so each iteration costs one pass.
# - Marquardt scaling with MINPACK's monotone diagonal D = max(D, diag(JᵀJ)) makes damping
#   invariant to parameter units (S0 ~ 1e3 next to T2 ~ 1e-2 is the norm in MRI).
# - Nielsen's damping update (Madsen, Nielsen & Tingleff 2004, Alg. 3.16).
# - Box constraints by projection; parameters held at a bound by the gradient are removed from
#   the step (projected-gradient active set), which avoids the stalls of plain clamping.
# - No exceptions, no allocation, no dynamic dispatch: the same code compiles for GPU kernels.

"""
    FitStatus

Per-voxel outcome, stored as `UInt8` in the status map:

| code | name            | meaning                                            |
|-----:|:----------------|:---------------------------------------------------|
| 0    | `NotFitted`     | outside the mask                                   |
| 1    | `ConvergedX`    | relative parameter step below `xtol`               |
| 2    | `ConvergedF`    | relative SSE reduction below `ftol`                |
| 3    | `ConvergedG`    | scaled gradient below `gtol` (or SSE == 0)         |
| 4    | `MaxIterations` | hit `maxiter`                                      |
| 5    | `Failed`        | model or derivatives not finite at the start point |
| 6    | `InvalidData`   | data contain NaN or Inf                            |
"""
@enum FitStatus::UInt8 begin
    NotFitted = 0
    ConvergedX = 1
    ConvergedF = 2
    ConvergedG = 3
    MaxIterations = 4
    Failed = 5
    InvalidData = 6
end

isconverged(s::FitStatus) = s == ConvergedX || s == ConvergedF || s == ConvergedG
isconverged(s::Integer) = isconverged(FitStatus(s))

"""
    LMOptions{T}(; maxiter=100, xtol, ftol, gtol, damping=1e-3)

Solver settings. Tolerances default by precision: `xtol = 1e-8, ftol = 1e-12, gtol = 1e-10`
for `Float64` and `xtol = 1e-5, ftol = 1e-6, gtol = 1e-5` for `Float32`.

- `xtol`: stop when ‖D·step‖ ≤ xtol·‖D·θ‖ (scaled, so unit-independent).
- `ftol`: stop when the actual and predicted relative SSE reductions are both ≤ `ftol`.
- `gtol`: stop when max_j |Jⱼᵀr| / (‖Jⱼ‖‖r‖) ≤ `gtol` (MINPACK's scale-free gradient test).
- `damping`: initial Marquardt parameter relative to diag(JᵀJ).

For parameter maps, `xtol = 1e-4` is usually plenty and saves iterations.
"""
struct LMOptions{T<:AbstractFloat}
    maxiter::Int32
    xtol::T
    ftol::T
    gtol::T
    damping::T
end

_default_tols(::Type{Float64}) = (1e-8, 1e-12, 1e-10)
_default_tols(::Type{Float32}) = (1f-5, 1f-6, 1f-5)
_default_tols(::Type{T}) where {T} = (sqrt(eps(T)), eps(T)^(3 // 4), sqrt(eps(T)))

function LMOptions{T}(; maxiter::Integer=100, xtol=nothing, ftol=nothing, gtol=nothing,
                      damping=1e-3) where {T}
    dx, df, dg = _default_tols(T)
    LMOptions{T}(Int32(maxiter), T(something(xtol, dx)), T(something(ftol, df)),
                 T(something(gtol, dg)), T(damping))
end

"""
    LMResult

Result of one fit: `θ` (SVector), `sse` (weighted sum of squared residuals), `JtJ` (Gauss–Newton
Hessian at `θ`), `iterations`, and `status::FitStatus`.
"""
struct LMResult{P,T,L}
    θ::SVector{P,T}
    sse::T
    JtJ::SMatrix{P,P,T,L}
    iterations::Int32
    status::FitStatus
end

isconverged(r::LMResult) = isconverged(r.status)

# ---------------------------------------------------------------------------------------------
# Normal equations: (SSE, JᵀWJ, JᵀWr) at θ, with r = f(θ) - y.

@inline _weight(::Nothing, i, ::Type{T}) where {T} = one(T)
@inline _weight(w, i, ::Type{T}) where {T} = @inbounds T(w[i])

@inline normal_equations(m::AbstractModel, x, y, w, θ, ws) =
    _normal_equations(modelstyle(m), m, x, y, w, θ, ws)

@inline function _normal_equations(::Pointwise, m::AbstractModel{P}, x, y, w,
                                   θ::SVector{P,T}, ws) where {P,T}
    A = zero(SMatrix{P,P,T})
    g = zero(SVector{P,T})
    sse = zero(T)
    @inbounds for i in 1:length(y)
        f, J = value_gradient(m, x[i], θ)
        wi = _weight(w, i, T)
        r = f - T(y[i])
        wJ = wi * J
        A += wJ * J'
        g += r * wJ
        sse += wi * r * r
    end
    return sse, A, g
end

@inline function _normal_equations(::Curve, m::AbstractModel{P}, x, y, w,
                                   θ::SVector{P,T}, ws) where {P,T}
    _curve!(ws, m, x, _seed(θ))
    A = zero(SMatrix{P,P,T})
    g = zero(SVector{P,T})
    sse = zero(T)
    @inbounds for i in 1:length(y)
        f, J = _vg(ws[i], Val(P), T)
        wi = _weight(w, i, T)
        r = f - T(y[i])
        wJ = wi * J
        A += wJ * J'
        g += r * wJ
        sse += wi * r * r
    end
    return sse, A, g
end

# ---------------------------------------------------------------------------------------------
# Small dense SPD solves without exceptions (non-SPD input returns ok = false).

@inline function _cholesky(M::SMatrix{P,P,T}) where {P,T}
    L = MMatrix{P,P,T}(M)
    @inbounds for j in 1:P
        s = L[j, j]
        for k in 1:j-1
            s -= L[j, k] * L[j, k]
        end
        s > zero(T) || return SMatrix(L), false   # also catches NaN
        ljj = @fastmath sqrt(s)
        L[j, j] = ljj
        il = inv(ljj)
        for i in j+1:P
            t = L[i, j]
            for k in 1:j-1
                t -= L[i, k] * L[j, k]
            end
            L[i, j] = t * il
        end
    end
    return SMatrix(L), true
end

@inline function _chol_solve(L::SMatrix{P,P,T}, b::SVector{P,T}) where {P,T}
    z = MVector{P,T}(b)
    @inbounds for i in 1:P              # L z = b
        t = z[i]
        for k in 1:i-1
            t -= L[i, k] * z[k]
        end
        z[i] = t / L[i, i]
    end
    @inbounds for i in P:-1:1           # Lᵀ h = z
        t = z[i]
        for k in i+1:P
            t -= L[k, i] * z[k]
        end
        z[i] = t / L[i, i]
    end
    return SVector(z)
end

# ---------------------------------------------------------------------------------------------

@inline _allfinite(v::StaticArray) = all(isfinite, v)

# Parameters pinned at a bound with the descent direction pointing outward are not free.
@inline _free(θ, g, lo, hi) = map((t, gi, l, h) -> !((t <= l && gi > 0) || (t >= h && gi < 0)), θ, g, lo, hi)

@inline function _gradient_small(A, g, free, sse, gtol)
    sse > 0 || return true
    T = typeof(sse)
    gmax = zero(T)
    @inbounds for j in eachindex(g)
        a = A[j, j]
        if free[j] && a > 0
            gmax = max(gmax, abs(g[j]) / @fastmath(sqrt(a * sse)))
        end
    end
    return gmax <= gtol
end

@inline function _xsmall(s, θ, d, xtol)
    ds = @fastmath sqrt(sum(d .* s .* s))     # d > 0, so the sums are non-negative
    dθ = @fastmath sqrt(sum(d .* θ .* θ))
    return ds <= xtol * (dθ + xtol)
end

"""
    lm_fit(model, x, y, θ0, lower, upper, opts::LMOptions[, w, workspace]) -> LMResult

Levenberg–Marquardt fit of one signal curve. All vectors of length P are `SVector{P,T}`; `y` and
`w` (weights, or `nothing`) are any indexable vectors of length `length(x)`. Allocation-free and
GPU-compatible for pointwise models.
"""
function lm_fit(m::AbstractModel{P}, x, y, θ0::SVector{P,T}, lo::SVector{P,T}, hi::SVector{P,T},
                opts::LMOptions{T}, w=nothing, ws=nothing) where {P,T}
    θ = clamp.(θ0, lo, hi)
    sse, A, g = normal_equations(m, x, y, w, θ, ws)
    if !(isfinite(sse) && _allfinite(g) && _allfinite(diag(A)))
        return LMResult(θ, sse, A, Int32(0), Failed)
    end
    floorT = floatmin(T) / eps(T)
    d = max.(diag(A), floorT)
    μ = opts.damping
    ν = T(2)
    iter = Int32(0)
    status = MaxIterations
    if _gradient_small(A, g, _free(θ, g, lo, hi), sse, opts.gtol)
        return LMResult(θ, sse, A, iter, ConvergedG)
    end
    while iter < opts.maxiter
        iter += Int32(1)
        free = _free(θ, g, lo, hi)
        fm = map(b -> ifelse(b, one(T), zero(T)), free)
        # Damped normal equations restricted to the free parameters.
        M = (fm * fm') .* (A + Diagonal(μ * d)) + Diagonal(one(T) .- fm)
        L, ok = _cholesky(M)
        if !ok
            μ *= ν
            ν *= 2
            continue
        end
        h = _chol_solve(L, -(fm .* g))
        θn = clamp.(θ + h, lo, hi)
        s = θn - θ
        if _xsmall(s, θ, d, opts.xtol)
            status = ConvergedX
            break
        end
        ssen, An, gn = normal_equations(m, x, y, w, θn, ws)
        pred = -(2 * dot(g, s) + dot(s, A * s))        # predicted SSE reduction
        actual = sse - ssen
        if isfinite(ssen) && _allfinite(gn) && _allfinite(diag(An)) && pred > 0 && actual > 0
            ρ = actual / pred
            fsmall = actual <= opts.ftol * sse && pred <= opts.ftol * sse
            θ, sse, A, g = θn, ssen, An, gn
            d = max.(d, diag(A))
            μ = max(μ * max(T(1 / 3), 1 - (2ρ - 1)^3), eps(T))
            ν = T(2)
            if fsmall
                status = ConvergedF
                break
            elseif _gradient_small(A, g, _free(θ, g, lo, hi), sse, opts.gtol)
                status = ConvergedG
                break
            end
        else
            if pred > 0 && pred <= opts.ftol * sse
                status = ConvergedF
                break
            end
            μ *= ν
            ν *= 2
        end
    end
    return LMResult(θ, sse, A, iter, status)
end

"""
    standard_errors(r::LMResult, nobs) -> SVector

Asymptotic standard errors √diag(σ̂² (JᵀWJ)⁻¹) with σ̂² = SSE / (nobs - P). Returns NaN where
the information matrix is singular. With known noise σ and unit weights, the Cramér–Rao bound is
σ·√diag((JᵀJ)⁻¹).
"""
function standard_errors(r::LMResult{P,T}, nobs::Integer) where {P,T}
    L, ok = _cholesky(r.JtJ)
    ok || return SVector{P,T}(ntuple(_ -> T(NaN), Val(P)))
    σ2 = nobs > P ? r.sse / (nobs - P) : T(NaN)
    return SVector{P,T}(ntuple(Val(P)) do j
        e = SVector{P,T}(ntuple(k -> T(k == j), Val(P)))
        v = σ2 * _chol_solve(L, e)[j]
        ifelse(v >= 0, @fastmath(sqrt(abs(v))), T(NaN))
    end)
end
