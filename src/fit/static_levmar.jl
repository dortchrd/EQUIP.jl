# =================================================================================================
# static_levmar: Levenberg–Marquardt (LM) for ONE voxel, with static (fixed-size) arrays
#
# "static" as in StaticArrays: the number of free parameters P is part of the types, so the
# compiler knows every array size and unrolls the arithmetic. ("levmar" is the usual short name
# for Levenberg–Marquardt; LsqFit.jl's general solver is called `levenberg_marquardt`.)
#
# Scaling, measured (DKI-like model, 3 observations per parameter, no allocation at any size):
#   P = 7: 2.4 µs per fit,  P = 15: 25 µs,  P = 22 (full DKI): 64 µs,  P = 28: 74 µs,
#   3.5–19× faster than LsqFit on the same problems. What grows is compile time (≈1 s at P = 7,
#   ≈8 s at P = 28, once per session per model and set of free parameters). Beyond ~30–40 free
#   parameters a solver with preallocated ordinary arrays (`mul!`, `cholesky!`), a future
#   `levmar`, would be the better choice.
#
# Problem: find the P free parameters θ that minimize  SSE(θ) = Σᵢ rᵢ(θ)²,  rᵢ = fᵢ(θ) − yᵢ,
# where f is the model signal and y the voxel's data (m observations).
#
# Each LM iteration solves the damped "normal equations"
#
#       (JᵀJ + μ·D) h = −Jᵀr
#
# for a step h, where J is the m×P Jacobian (∂fᵢ/∂θⱼ), D a diagonal scaling and μ ≥ 0 the
# damping. Small μ gives a Gauss–Newton step (fast near the solution); large μ a short
# gradient-descent step (safe far from it). μ shrinks after good steps and grows after bad ones.
#
# Why a custom solver: the fit is tiny (P < 10, m < 100) but repeated millions of times, so the
# overheads of general-purpose solvers (allocation, generic linear algebra) dominate. Here:
#
# - Derivatives come from dual numbers (ForwardDiff): the model's `signal!` is evaluated once
#   with duals and every output carries its value AND its P partial derivatives.
# - JᵀJ (P×P), Jᵀr (P) and SSE are summed directly from those outputs into StaticArrays, which
#   live on the stack. The m×P Jacobian is never stored and nothing is allocated on the heap.
# - Marquardt scaling D = diag(JᵀJ) (never decreasing, as in MINPACK) makes the damping
#   independent of parameter units (M0 ~ 1000 next to T2 ~ 0.05 is the norm in MRI).
# - Bounds by projection: a step that leaves the box is clipped onto it, and a parameter
#   sitting on a bound that the gradient pushes outward is frozen for that step.
# =================================================================================================

"""
    FitStatus

Outcome of one voxel fit, stored as a `UInt8` code in the `status` map:

| code | name            | meaning                                               |
|-----:|:----------------|:------------------------------------------------------|
| 0    | `NotFitted`     | outside the mask                                      |
| 1    | `ConvergedX`    | the parameter step became tiny (`xtol`)               |
| 2    | `ConvergedF`    | the SSE stopped decreasing (`ftol`)                   |
| 3    | `ConvergedG`    | the gradient became tiny (`gtol`), or SSE == 0        |
| 4    | `MaxIterations` | reached `maxiter` without converging                  |
| 5    | `Failed`        | model or derivatives not finite at the starting point |
| 6    | `InvalidData`   | data, a fixed-map value or a start value is NaN/Inf   |

`FitStatus(code)` converts a code back to its name.
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
# `@enum` makes a type whose values are named constants; `::UInt8` stores them in one byte.

"""
    isconverged(status) -> Bool

True for the three "converged" codes (1–3). Accepts a `FitStatus`, a code, or a `VoxelResult`.
"""
isconverged(s::FitStatus) = s == ConvergedX || s == ConvergedF || s == ConvergedG
isconverged(s::Integer) = isconverged(FitStatus(s))

"""
    SolverOptions{T}(; maxiter = 100, xtol, ftol, gtol, damping = 1e-3)

Solver settings (keywords of `init` and `fit`). Default tolerances depend on precision:
`xtol = 1e-8, ftol = 1e-12, gtol = 1e-10` for Float64; `1e-5, 1e-6, 1e-5` for Float32.

- `xtol`: stop when the (scaled) step is smaller than `xtol` × the (scaled) parameters.
- `ftol`: stop when the relative SSE decrease, actual and predicted, is below `ftol`.
- `gtol`: stop when the scale-free gradient max_j |Jⱼᵀr| / (‖Jⱼ‖‖r‖) is below `gtol`.
- `damping`: starting μ, relative to diag(JᵀJ).
"""
struct SolverOptions{T<:AbstractFloat}
    maxiter::Int32
    xtol::T
    ftol::T
    gtol::T
    damping::T
end

_default_tols(::Type{Float64}) = (1e-8, 1e-12, 1e-10)
_default_tols(::Type{Float32}) = (1f-5, 1f-6, 1f-5)

# Keyword constructor. `something(a, b)` returns `a` unless it is `nothing`, then `b`.
function SolverOptions{T}(; maxiter::Integer = 100, xtol = nothing, ftol = nothing, gtol = nothing,
                      damping = 1e-3) where {T}
    dx, df, dg = _default_tols(T)
    return SolverOptions{T}(Int32(maxiter), T(something(xtol, dx)), T(something(ftol, df)),
                        T(something(gtol, dg)), T(damping))
end

# Internal result of `static_levmar` (the public one, `VoxelResult`, adds parameter names).
struct SolverResult{P,T}
    θ::SVector{P,T}
    sse::T
    iterations::Int32
    status::FitStatus
end

# -------------------------------------------------------------------------------------------------
# One voxel's problem, and its normal equations
# -------------------------------------------------------------------------------------------------

"""
    VoxelProblem{F}(model, y, fixed)

Everything needed to evaluate one voxel's residuals: the free parameter names `F` (a tuple of
Symbols, stored in the TYPE so the compiler knows them), the model (with its acquisition), the
data `y` and this voxel's fixed values (numbers and ties).
"""
struct VoxelProblem{F,M,Y,X}
    model::M
    y::Y
    fixed::X
end
# Constructor that takes F explicitly and lets Julia infer the other type parameters.
VoxelProblem{F}(model::M, y::Y, fixed::X) where {F,M,Y,X} = VoxelProblem{F,M,Y,X}(model, y, fixed)

# A unique "tag" type for our dual numbers, so they never mix with duals from other code.
struct FitTag end

# Turn θ = (θ₁, …, θ_P) into dual numbers seeded with the identity: dual j has value θⱼ and
# partials eⱼ (1 in slot j, 0 elsewhere). Any function evaluated on them then returns
# value f(θ) with partials (∂f/∂θ₁, …, ∂f/∂θ_P): one pass gives a full row of the Jacobian.
@inline function _seed(θ::SVector{P,T}) where {P,T}
    return SVector(ntuple(j -> Dual{FitTag}(θ[j], ntuple(k -> ifelse(k == j, one(T), zero(T)), Val(P))),
                          Val(P)))
end
# `ntuple(f, Val(P))` builds the tuple (f(1), …, f(P)) with P known at compile time, so it is
# fully unrolled with no allocation.

# Build the complete parameter set for the model from the free values θ and the fixed values:
#   NamedTuple{F}(Tuple(θ))   -> (PSR = θ₁, R1f = θ₂, Sf = θ₃, M0f = θ₄)
#   merge(..., fixed)         -> (PSR = θ₁, ..., kmf = 12.5, R1m = Fixed(:R1f), Sm = 0.83)
#   _resolve_ties(...)        -> ties evaluated: R1m = θ₂
# Because F is a compile-time constant (passed as Val(F)), this compiles to moving numbers into
# a stack-allocated record; no names are looked up at run time.
@inline _params(::Val{F}, θ, fixed) where {F} = _resolve_ties(merge(NamedTuple{F}(Tuple(θ)), fixed))

"""
    normal_equations(prob, θ, buf) -> (sse, JᵀJ, Jᵀr)

Evaluate the model at θ with dual numbers into `buf` (one dual per observation) and sum the
SSE, JᵀJ (P×P) and Jᵀr (P) from it.
"""
@inline function normal_equations(prob::VoxelProblem{F}, θ::SVector{P,T}, buf) where {F,P,T}
    p = _params(Val(F), _seed(θ), prob.fixed)   # every parameter; the free ones are duals
    signal!(buf, prob.model, p)                 # the model fills buf[i] = (fᵢ, ∂fᵢ/∂θ)
    A = zero(SMatrix{P,P,T})                     # JᵀJ
    g = zero(SVector{P,T})                       # Jᵀr
    sse = zero(T)
    y = prob.y
    @inbounds for i in eachindex(buf)
        d = buf[i]
        J = SVector{P,T}(partials(d).values)     # row i of the Jacobian
        r = T(value(d)) - T(y[i])                # residual i
        A += J * J'                              # outer product, P×P
        g += r * J
        sse += r * r
    end
    return sse, A, g
end

# -------------------------------------------------------------------------------------------------
# Small linear solves: Cholesky factorization of a P×P symmetric positive-definite matrix.
# Written out by hand so that a non-positive-definite matrix returns `ok = false` instead of
# throwing an exception (exceptions are slow and we want the fit to just carry on).
# -------------------------------------------------------------------------------------------------

@inline function _cholesky(M::SMatrix{P,P,T}) where {P,T}
    L = MMatrix{P,P,T}(M)                        # mutable copy, still on the stack
    @inbounds for j in 1:P
        s = L[j, j]
        for k in 1:j-1
            s -= L[j, k] * L[j, k]
        end
        s > zero(T) || return SMatrix(L), false  # not positive definite (also catches NaN)
        ljj = sqrt(s)
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
    return SMatrix(L), true                      # lower triangle holds L, with M = L Lᵀ
end

# Solve (L Lᵀ) x = b by forward then backward substitution.
@inline function _chol_solve(L::SMatrix{P,P,T}, b::SVector{P,T}) where {P,T}
    z = MVector{P,T}(b)
    @inbounds for i in 1:P                       # L z = b
        t = z[i]
        for k in 1:i-1
            t -= L[i, k] * z[k]
        end
        z[i] = t / L[i, i]
    end
    @inbounds for i in P:-1:1                    # Lᵀ x = z
        t = z[i]
        for k in i+1:P
            t -= L[k, i] * z[k]
        end
        z[i] = t / L[i, i]
    end
    return SVector(z)
end

# -------------------------------------------------------------------------------------------------
# Small helpers for the main loop
# -------------------------------------------------------------------------------------------------

@inline _allfinite(v::StaticArray) = all(isfinite, v)

# Which parameters may move this step: not those sitting on a bound with the gradient pointing
# outward (the descent direction is −g, so g > 0 at the lower bound pushes further down).
@inline _movable(θ, g, lo, hi) = map((t, gi, l, h) -> !((t <= l && gi > 0) || (t >= h && gi < 0)), θ, g, lo, hi)

# Scale-free gradient test (MINPACK): max_j |Jⱼᵀr| / (‖Jⱼ‖ ‖r‖) ≤ gtol, over movable parameters.
@inline function _gradient_small(A, g, movable, sse, gtol)
    sse > 0 || return true                       # perfect fit
    T = typeof(sse)
    gmax = zero(T)
    @inbounds for j in eachindex(g)
        a = A[j, j]                              # ‖Jⱼ‖²
        if movable[j] && a > 0
            gmax = max(gmax, abs(g[j]) / sqrt(a * sse))
        end
    end
    return gmax <= gtol
end

# Step test in scaled units: ‖D^½ s‖ ≤ xtol (‖D^½ θ‖ + xtol).
@inline function _step_small(s, θ, d, xtol)
    ds = sqrt(sum(d .* s .* s))
    dθ = sqrt(sum(d .* θ .* θ))
    return ds <= xtol * (dθ + xtol)
end

# -------------------------------------------------------------------------------------------------
# The solver
# -------------------------------------------------------------------------------------------------

"""
    static_levmar(prob, θ0, lower, upper, opts, buf) -> SolverResult

Levenberg–Marquardt fit of one voxel from the start `θ0`, within `lower ≤ θ ≤ upper` (all
`SVector{P,T}`). `buf` holds one `Dual{FitTag,T,P}` per observation. Does not allocate.
"""
function static_levmar(prob::VoxelProblem, θ0::SVector{P,T}, lo::SVector{P,T}, hi::SVector{P,T},
                       opts::SolverOptions{T}, buf) where {P,T}
    # --- start: project into the box and evaluate -----------------------------------------------
    θ = clamp.(θ0, lo, hi)
    sse, A, g = normal_equations(prob, θ, buf)
    if !(isfinite(sse) && _allfinite(g) && _allfinite(diag(A)))
        return SolverResult(θ, sse, Int32(0), Failed)
    end
    d = max.(diag(A), floatmin(T) / eps(T))     # Marquardt scaling (kept strictly positive)
    μ = opts.damping                            # damping
    ν = T(2)                                    # growth factor of μ after rejected steps
    iter = Int32(0)
    status = MaxIterations
    _gradient_small(A, g, _movable(θ, g, lo, hi), sse, opts.gtol) &&
        return SolverResult(θ, sse, iter, ConvergedG)

    while iter < opts.maxiter
        iter += Int32(1)

        # --- solve the damped normal equations for the movable parameters ---------------------
        # `fm` is 1 for movable, 0 for frozen parameters. Masking rows/columns of the matrix
        # and putting 1 on the frozen diagonal gives step 0 for frozen parameters.
        fm = map(b -> ifelse(b, one(T), zero(T)), _movable(θ, g, lo, hi))
        M = (fm * fm') .* (A + Diagonal(μ * d)) + Diagonal(one(T) .- fm)
        L, ok = _cholesky(M)
        if !ok                                   # numerically singular: damp more, retry
            μ *= ν
            ν *= 2
            continue
        end
        h = _chol_solve(L, -(fm .* g))
        θn = clamp.(θ + h, lo, hi)               # trial point, projected onto the bounds
        s = θn - θ
        if _step_small(s, θ, d, opts.xtol)
            status = ConvergedX
            break
        end

        # --- evaluate the trial point and compare actual vs predicted SSE decrease ------------
        ssen, An, gn = normal_equations(prob, θn, buf)
        pred = -(2 * dot(g, s) + dot(s, A * s))  # decrease predicted by the linear model
        actual = sse - ssen
        if isfinite(ssen) && _allfinite(gn) && _allfinite(diag(An)) && pred > 0 && actual > 0
            # accept the step
            ρ = actual / pred                    # ≈ 1 when the linear model is accurate
            fsmall = actual <= opts.ftol * sse && pred <= opts.ftol * sse
            θ, sse, A, g = θn, ssen, An, gn
            d = max.(d, diag(A))
            μ = max(μ * max(T(1 / 3), 1 - (2ρ - 1)^3), eps(T))   # Nielsen's update: shrink μ
            ν = T(2)
            if fsmall
                status = ConvergedF
                break
            elseif _gradient_small(A, g, _movable(θ, g, lo, hi), sse, opts.gtol)
                status = ConvergedG
                break
            end
        else
            # reject the step: keep θ, damp more (shorter, more gradient-like steps)
            if pred > 0 && pred <= opts.ftol * sse   # nothing left to gain
                status = ConvergedF
                break
            end
            μ *= ν
            ν *= 2
        end
    end
    return SolverResult(θ, sse, iter, status)
end
