# =================================================================================================
# levmar: Levenberg–Marquardt (LM) for ONE voxel
#
# One function, `levmar`, whose method is chosen by the types of its arguments (multiple
# dispatch), like `cholesky` in Julia's LinearAlgebra, which has one method for StaticArrays
# matrices and another for ordinary ones. The method below is for STATIC arrays (`SVector`,
# `SMatrix` from StaticArrays): the number of free parameters P is part of the types, so the
# compiler knows every array size and unrolls the arithmetic; nothing is allocated. ("levmar" is
# the usual short name for Levenberg–Marquardt; LsqFit.jl's general solver is called
# `levenberg_marquardt`.)
#
# Scaling of this static method, measured (DKI-like model, 3 observations per parameter, no
# allocation at any size): P = 7: 2.4 µs per fit, P = 15: 25 µs, P = 22 (full DKI): 64 µs,
# P = 28: 74 µs; 3.5–19× faster than LsqFit on the same problems. What grows is compile time
# (≈1 s at P = 7, ≈8 s at P = 28, once per session per model and set of free parameters).
# Beyond ~30–40 free parameters, a second method for ordinary arrays (`Vector`/`Matrix` with a
# preallocated workspace, `mul!`, `cholesky!`) would be better; `setup` would choose the array
# type from P once per fit (see the roadmap in docs/design.md).
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
# - Steps into a "dead zone" are rejected (EQUIP's own guard; see `_lost_influence` below): a
#   step after which some parameter no longer affects the signal at all, e.g. a time constant
#   pushed to ~0, where every sample has fully decayed and its derivative is zero.
# =================================================================================================

"""
    FitStatus

Outcome of one voxel fit: `res.status` of a [`FitResult`](@ref) (a map of them for a dataset,
1 byte per voxel). `UInt8(status)` gives the code, e.g. `UInt8.(res.status)` to write a file:

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

True for the three "converged" codes (1–3). Accepts a `FitStatus`, a code, or a
[`FitResult`](@ref) (for a dataset: a map of true/false).
"""
isconverged(s::FitStatus) = s == ConvergedX || s == ConvergedF || s == ConvergedG
isconverged(s::Integer) = isconverged(FitStatus(s))

"""
    LM(; maxiter = 100, xtol, ftol, gtol, damping = 1e-3)

Settings of the Levenberg–Marquardt solver, for the `solver` keyword of [`fit`](@ref) and
[`init`](@ref): `fit(m, data; solver = LM(maxiter = 50))`. Most fits need none of them.

- `maxiter`: the most iterations per voxel (then status `MaxIterations`).
- `xtol`: stop when the (scaled) step is smaller than `xtol` × the (scaled) parameters.
- `ftol`: stop when the relative SSE decrease, actual and predicted, is below `ftol`.
- `gtol`: stop when the scale-free gradient max_j |Jⱼᵀr| / (‖Jⱼ‖‖r‖) is below `gtol`.
- `damping`: the starting μ, relative to diag(JᵀJ).

Tolerances left out depend on the precision: `xtol = 1e-8, ftol = 1e-12, gtol = 1e-10` for
Float64, `1e-5, 1e-6, 1e-5` for Float32.
"""
Base.@kwdef struct LM
    maxiter::Int = 100
    xtol::Union{Nothing,Float64} = nothing          # nothing: the default for the precision
    ftol::Union{Nothing,Float64} = nothing
    gtol::Union{Nothing,Float64} = nothing
    damping::Float64 = 1e-3
end

# The solver's own copy of the settings, in the working precision T, with the defaults filled in.
struct SolverOptions{T<:AbstractFloat}
    maxiter::Int32
    xtol::T
    ftol::T
    gtol::T
    damping::T
end

_default_tols(::Type{Float64}) = (1e-8, 1e-12, 1e-10)
_default_tols(::Type{Float32}) = (1f-5, 1f-6, 1f-5)

# `something(a, b)` returns `a` unless it is `nothing`, then `b`.
function SolverOptions{T}(s::LM) where {T}
    dx, df, dg = _default_tols(T)
    return SolverOptions{T}(Int32(s.maxiter), T(something(s.xtol, dx)), T(something(s.ftol, df)),
                            T(something(s.gtol, dg)), T(s.damping))
end

# Internal result of `levmar` (the public one, `FitResult`, adds names, the model and settings).
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
    VoxelProblem{F}(model, y, fixed[, w])

Everything needed to evaluate one voxel's residuals: the free parameter names `F` (a tuple of
Symbols, stored in the TYPE so the compiler knows them), the model (with its acquisition), the
data `y`, this voxel's fixed values (numbers and ties) and the observations' weights `w`
(`nothing`: unweighted; else one weight per observation, the fit minimizing ½Σwᵢrᵢ²).
"""
struct VoxelProblem{F,M,Y,X,W}
    model::M
    y::Y
    fixed::X
    w::W
end
# Constructor that takes F explicitly and lets Julia infer the other type parameters.
VoxelProblem{F}(model::M, y::Y, fixed::X, w::W = nothing) where {F,M,Y,X,W} =
    VoxelProblem{F,M,Y,X,W}(model, y, fixed, w)

# Observation i's weight, in precision T: 1 for an unweighted fit. Every sum multiplies by it
# (the solver's, the starts', crlb's); x·1 is exact (and the compiler drops the multiplication),
# so an unweighted fit gives the same numbers, bit for bit, as before weights existed. Julia picks
# the method by the type of `w`, when it compiles: no test at run time.
@inline _wt(::Nothing, i, ::Type{T}) where {T} = one(T)
@inline _wt(w::AbstractVector, i, ::Type{T}) where {T} = T(@inbounds w[i])

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
    normal_equations(prob, θ, buf) -> (sse, JᵀWJ, JᵀWr)

Evaluate the model at θ with dual numbers into `buf` (one dual per observation) and sum the
SSE, JᵀJ (P×P) and Jᵀr (P) from it, each term times the observation's weight wᵢ (1 when the
fit is unweighted): Σwᵢrᵢ², JᵀWJ and JᵀWr.
"""
# One function for weighted and unweighted fits: each term times the observation's weight,
# `_wt(prob.w, i, T)`, which is 1 for an unweighted fit (exact: the same numbers, bit for bit, as
# without it). Not inlined into `levmar`: inlined there (twice, at the start and at each trial
# point), the weighted loop made the whole solver slower for the bigger models with the same
# iterations (weights all 1: 39% for the 16-point SIR with kmf free, 18% for the biexponential),
# though the sum alone costs the same; as a call, weighted and unweighted fits run at the same
# speed (docs/design.md, "Weighted fits").
@noinline function normal_equations(prob::VoxelProblem{F}, θ::SVector{P,T}, buf) where {F,P,T}
    p = _params(Val(F), _seed(θ), prob.fixed)   # every parameter; the free ones are duals
    signal!(buf, prob.model, p)                 # the model fills buf[i] = (fᵢ, ∂fᵢ/∂θ)
    A = zero(SMatrix{P,P,T})                     # JᵀWJ
    g = zero(SVector{P,T})                       # JᵀWr
    sse = zero(T)                                # Σwr²
    y = prob.y
    @inbounds for i in eachindex(buf)
        d = buf[i]
        J = SVector{P,T}(partials(d).values)     # row i of the Jacobian
        r = T(value(d)) - T(y[i])                # residual i
        w = _wt(prob.w, i, T)                    # its weight (1: unweighted)
        wJ = w * J
        A += wJ * J'                             # outer product, P×P
        g += r * wJ
        sse += w * r * r
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

# Does the trial point leave a movable parameter with no influence on the signal?
#
# ‖Jⱼ‖² = A[j, j] (the diagonal of JᵀJ) measures how strongly the signal responds to parameter j.
# If it collapses from a positive value to (almost) zero, the parameter has entered a "dead zone":
# e.g. a time constant T pushed to ~0 by a step projected onto its lower bound, where
# exp(-t/T) = 0 at every sample. Its derivative is then exactly zero, so the solver could never
# move it again, yet such a step can still lower the SSE (the other parameters adapt), so the
# usual tests accept it and the fit stops there, reporting convergence.
#
# Rejecting the step (as any bad step: more damping, a shorter step next) avoids this. Bounds
# where the parameter still matters (R2 = 0, a flat curve; PSR = 0) are unaffected.
#
# EQUIP's own guard: we know of no standard solver with exactly this rule. Related: Transtrum &
# Sethna's "parameter evaporation" and their acceptance test on geodesic acceleration; SciPy's
# trust-region-reflective method, which keeps iterates strictly inside the bounds; lmfit/MINUIT,
# which remove bounds by a change of variables. The usual acceptance threshold (actual/predicted
# decrease > η) does not catch it: tested with η up to 0.25. Measurements: docs/design.md.
@inline function _lost_influence(A, An, fm)
    a, an = diag(A), diag(An)
    return any((fm .> 0) .& (a .> 0) .& (an .<= eps(eltype(a)) .* a))
end

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
    levmar(prob, θ0, lower, upper, opts, buf) -> SolverResult

Levenberg–Marquardt fit of one voxel from the start `θ0`, within `lower ≤ θ ≤ upper`.

This method is for static arrays: `θ0`, `lower` and `upper` are `SVector{P,T}`, and `buf` holds
one `Dual{FitTag,T,P}` per observation. Does not allocate.
"""
function levmar(prob::VoxelProblem, θ0::SVector{P,T}, lo::SVector{P,T}, hi::SVector{P,T},
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
        if isfinite(ssen) && _allfinite(gn) && _allfinite(diag(An)) && pred > 0 && actual > 0 &&
           !_lost_influence(A, An, fm)                      # not into a dead zone (see above)
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
