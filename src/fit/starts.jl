# =================================================================================================
# Starting values: where each voxel's starting point comes from
#
# Each free parameter's start says where it comes from (see `Free`). θ0, the voxel's starting
# point (one value per free parameter), already holds the numbers and this voxel's value of any
# start map; then, per voxel, in this order (called from `_fit_voxel` in voxel.jl):
#   1. parameters starting at `:data`: the model's estimates from the data (`datastart`)   _data_start
#   2. parameters starting at `:loglinear`: the weighted fit of ln y with ln S         _loglinear_start
#   3. a parameter starting from `BestOf(values)`: the value that fits best              _best_start
#      or else the `:scale` parameter (M0, S0, ...) at its best value                     _scale_start
#
# Step 3 rests on one calculation, `_score`: the signal at a candidate starting point with the
# scale parameter solved exactly, and the SSE there. BestOf scores each of its values; the scale
# start scores the one starting point.
# =================================================================================================

# Score a candidate starting point θ. The signal is proportional to the scale parameter (number
# `sc` among the free parameters), so with s = the signal at scale 1 (its "shape"), the scale that
# best matches the data y is the least-squares slope through the origin, a = (s·y) / (s·s), and
# the SSE there is Σ (a·s - y)² = y·y - 2a·(s·y) + a²·(s·s). Returns θ with that scale, and the
# SSE. Without a scale parameter (sc = 0): θ unchanged, and its SSE. A weighted fit weights each
# product by wᵢ (Σwsy, Σws², Σwy²: the weighted slope and SSE); unweighted, wᵢ = 1.
#
# If a is unusable (zero signal at θ, a NaN, or a ≤ 0, e.g. noise-only data) the scale stays at 1.
# The model is evaluated with plain numbers; storing them in the dual buffer just sets the
# derivative parts to zero, so no extra buffer is needed and nothing is allocated.
# `@inline`: BestOf calls this once per value (30 times per voxel for T1IR), and Julia does not
# inline a function this size by itself; without it, T1IR fits were 8% slower (measured).
@inline function _score(prob::VoxelProblem{F}, θ::SVector{P,T}, sc, lo, hi, buf) where {F,P,T}
    sc > 0 && (θ = Base.setindex(θ, one(T), sc))           # the signal at scale 1: its shape
    signal!(buf, prob.model, _params(Val(F), θ, prob.fixed))
    sy = ss = yy = zero(T)
    @inbounds for i in eachindex(buf)
        s, y = T(value(buf[i])), T(prob.y[i])
        w = _wt(prob.w, i, T)
        ws = w * s
        sy += ws * y
        ss += ws * s
        yy += w * y * y
    end
    sc > 0 || return θ, yy - 2 * sy + ss                     # Σ (s - y)²
    a = sy / ss
    a = clamp(isfinite(a) && a > 0 ? a : one(T), lo[sc], hi[sc])
    return Base.setindex(θ, a, sc), yy - 2 * a * sy + a^2 * ss    # θ with its scale, Σ (a·s - y)²
end

# The scale parameter `sc` at its best value, given the other starts.
_scale_start(prob, θ0, sc, lo, hi, buf) = first(_score(prob, θ0, sc, lo, hi, buf))

# The best of the BestOf values (see `BestOf`) for free parameter k: score each value (the scale
# solved for each) and keep the lowest SSE. If none is usable (e.g. NaN signals), fall back to the
# scale start from θ0, which holds the middle value.
function _best_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec, buf) where {F,P,T}
    k, sc = spec.candindex, spec.scale
    best, bestsse = θ0, T(Inf)
    for g in spec.candidates
        θ, sse = _score(prob, Base.setindex(θ0, g, k), sc, spec.lower, spec.upper, buf)
        if sse < bestsse                                     # NaN never wins
            best, bestsse = θ, sse
        end
    end
    isfinite(bestsse) && return best
    return sc > 0 ? _scale_start(prob, θ0, sc, spec.lower, spec.upper, buf) : θ0
end

# Starting values from the model's own estimates on this voxel's data (see `datastart`), for the
# free parameters that start at `:data`; clamped to the bounds. `setup` checked that `datastart`
# covers them. A NaN estimate stays NaN, and the fit then stops at once with status Failed.
function _data_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec) where {F,P,T}
    any(spec.fromdata) || return θ0             # no :data start in this fit
    d = datastart(prob.model, prob.y, prob.fixed)
    d === nothing && return θ0                  # no data start: decided when compiling, no cost
    for (name, x) in pairs(d)
        k = findfirst(==(name), F)              # position among the free parameters, or nothing
        if k !== nothing && spec.fromdata[k]
            θ0 = Base.setindex(θ0, clamp(T(x), spec.lower[k], spec.upper[k]), k)
        end
    end
    return θ0
end

# -------------------------------------------------------------------------------------------------
# The log-linear start (`Free(start = :loglinear)`), from the model's own signal
#
# Many diffusion and relaxation signals are exponentials whose logarithm is linear in the
# parameters: ln S = ln S0 − b·gᵀDg for DTI, ln S = ln S0 − b·ADC for ADCDecay. Their standard
# start is the linear fit of ln y, weighted by y² (the noise in ln y is about σ/y; Basser et al.,
# J Magn Reson B 1994). Here it is computed from the model's `signal!` rather than written again
# for each model: Gauss–Newton steps on the residuals ln S(θ) − ln y, with the Jacobian
# ∂ln S/∂θ = (∂S/∂θ)/S from the same dual numbers the solver uses. Where ln S is linear in the
# parameters, the first step lands on the weighted log-linear fit exactly, from any point (S0's
# column, 1/S0, is the same for every image: it is the fit's intercept); where it is nearly linear
# (DKI: the kurtosis term is multiplied by MD²), a few steps reach it.
#
# What moves: the parameters starting at `:loglinear` (from 0, their placeholder), and the
# `:scale` parameter (the intercept: without it, the other parameters would absorb the signal's
# level). The rest stay at their starts. The scale multiplies the signal, so ln S is linear in
# ln S0, not in S0: its step is taken on ln S0, S0·exp(ΔS0/S0) (Gauss–Newton's ΔS0/S0 is the step
# in ln S0), which keeps the first step exact for DTI. Stepped in S0 itself, it crept from its
# placeholder 1 towards 1000 over all ten steps (measured: the start took 48 µs instead of 11 for
# 132 images). The scale is then set again by least squares on y, as for every start
# (`_scale_start`), so the log fit's S0 is only a step on the way.
#
# The pieces, each as in the solver or in DTI's former `datastart` (docs/design.md, "DTI built
# in"; scratch pivots.jl for the threshold):
#   - weights y²·w (w the fit's own observation weights), y divided by its largest value since
#     only the ratios matter; images with y ≤ 0 or S ≤ 0 skipped (no logarithm);
#   - sums in Float64 at least (Float32 fits too); the system scaled to a unit diagonal, then the
#     solver's `_cholesky`, which reports failure instead of throwing;
#   - a parameter the data do not measure at this point (a zero column: DKI's W while D = 0, a
#     direction no image has) does not move this step; a system the data cannot determine at all
#     (one shell and no b = 0; fewer directions than elements) shows as a pivot below eps^(1/4),
#     and the start stays where it is;
#   - a step that makes the weighted log-SSE larger is halved (backtracking); it stops when the
#     predicted decrease is negligible, or after `_LOGLINEAR_STEPS` steps; each step is clamped to
#     the bounds.
# Allocates nothing: the buffer is the solver's, the sums static arrays.
# -------------------------------------------------------------------------------------------------

const _LOGLINEAR_STEPS = 10
# PROTOTYPE switch (2026-11-07, measured, not decided; docs/design.md, "Predicted-signal weights"):
# the weights of the log fit. :data, y² (the default); :twopass, an unweighted step then one
# weighted by the predicted signal², stop (DIPY's WLS, for DTI); :predicted, the predicted signal²
# at every step until nothing is left; :predictedstop, the same, stopping after a step that gains
# under 1% of the weighted log-SSE.
const _LOGLINEAR_WEIGHTS = Ref(:data)

function _loglinear_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec, buf) where {F,P,T}
    any(spec.fromlog) || return θ0               # no :loglinear start in this fit: no cost
    S = promote_type(Float64, T)                 # the precision of the sums and the solve
    ymax = S(maximum(prob.y))
    ymax > 0 || return θ0
    moves = ntuple(k -> spec.fromlog[k] || k == spec.scale, Val(P))   # what the log fit moves
    θ, q = θ0, S(Inf)                            # the current point and its weighted log-SSE
    θprev = θ0
    mode = _LOGLINEAR_WEIGHTS[]
    predicted = mode !== :data
    for step in 1:_LOGLINEAR_STEPS
        qn, A, g = _log_normal_equations(prob, θ, buf, ymax, S, predicted)
        isfinite(qn) || return θprev
        if !predicted && qn > q                  # worse than the last point: halve the step
            θ = (θprev + θ) / 2
            continue
        end
        θprev, q = θ, qn
        Δ, ok = _loglinear_step(A, g, moves)
        ok || return θ                           # the data cannot determine the parameters
        pred = -dot(g, Δ) / 2                    # the decrease the linear model predicts
        if mode === :predictedstop                # take this step, and stop if it gains little
            θ = clamp.(_logstep(θ, SVector{P,T}(Δ), spec.scale), spec.lower, spec.upper)
            pred <= q / 100 && return θ
            continue
        end
        pred <= sqrt(eps(S)) * q + eps(S) && return θ
        θ = clamp.(_logstep(θ, SVector{P,T}(Δ), spec.scale), spec.lower, spec.upper)
        mode === :twopass && step == 2 && return θ
    end
    return θprev
end

# θ + Δ, with the scale parameter (number `sc`, 0 for none) stepped on its logarithm.
@inline function _logstep(θ::SVector{P,T}, Δ::SVector{P,T}, sc) where {P,T}
    θn = θ + Δ
    (sc > 0 && θ[sc] > 0) || return θn
    return Base.setindex(θn, θ[sc] * exp(Δ[sc] / θ[sc]), sc)
end

# The weighted normal equations of the log fit at θ: Σw·r², JᵀWJ and JᵀWr, with r = ln S − ln y,
# J = ∂ln S/∂θ and w = (y/ymax)²·(the fit's weight), summed in precision S.
@noinline function _log_normal_equations(prob::VoxelProblem{F}, θ::SVector{P,T}, buf, ymax,
                                         ::Type{S}, predicted = false) where {F,P,T,S}
    signal!(buf, prob.model, _params(Val(F), _seed(θ), prob.fixed))   # S and ∂S/∂θ, as the solver
    A = zero(SMatrix{P,P,S})
    g = zero(SVector{P,S})
    q = zero(S)
    @inbounds for i in eachindex(buf)
        y, f = S(prob.y[i]), S(value(buf[i]))
        (y > 0 && f > 0) || continue
        J = SVector{P,S}(partials(buf[i]).values) * inv(f)               # ∂ln S/∂θ
        r = log(f / y)                                                   # ln S − ln y
        w = _wt(prob.w, i, S) * ((predicted ? f : y) / ymax)^2
        wJ = w * J
        A += wJ * J'
        g += r * wJ
        q += w * r^2
    end
    return q, A, g
end

# The Gauss–Newton step for the parameters that move (and that the data measure at this point,
# A[k, k] > 0), from the system scaled to a unit diagonal; the others get step 0 (masked as in
# the solver). `ok = false` when the data cannot determine them: a pivot below eps^(1/4).
@inline function _loglinear_step(A::SMatrix{P,P,S}, g::SVector{P,S}, moves) where {P,S}
    on = SVector{P,Bool}(ntuple(k -> moves[k] && A[k, k] > 0, Val(P)))
    any(on) || return zero(SVector{P,S}), false
    sc = map((o, a) -> o ? 1 / sqrt(a) : zero(S), on, diag(A))   # the scaling; 0 masks
    M = (sc * sc') .* A + Diagonal(map(o -> ifelse(o, zero(S), one(S)), on))   # 1 where masked
    L, ok = _cholesky(M)
    (ok && minimum(diag(L)) > sqrt(sqrt(eps(S)))) || return zero(SVector{P,S}), false
    return sc .* _chol_solve(L, -(sc .* g)), true
end
