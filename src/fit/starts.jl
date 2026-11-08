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
# start is the weighted linear fit of ln y. Here it is computed from the model's `signal!` rather
# than written again for each model: Gauss–Newton passes on the residuals ln S(θ) − ln y, with
# the Jacobian ∂ln S/∂θ = (∂S/∂θ)/S from the same dual numbers the solver uses. Where ln S is
# linear in the parameters, a pass lands on the weighted log-linear fit from any point (S0's
# column, 1/S0, is the same for every image: it is the fit's intercept).
#
# The weights and the passes: the noise in ln y is about σ/S, so each image is weighted by S²,
# the predicted signal squared. Exactly two passes, OLS then WLS (Chung et al., NeuroImage 2006;
# DIPY's default WLS): the first unweighted (only the fit's own observation weights, if any),
# the second weighted by the first's prediction. The first is unweighted by rule, not because the
# signal at the placeholder start happens to be flat: a parameter started from a map shapes that
# signal, and weighting by it would make the first pass something other than OLS. For a model
# whose ln S is linear
# in its parameters the start is then DIPY's WLS (to 1e-13, measured). A fixed number of passes,
# not a convergence test (decided 2026-11-07, after a review the owner brought: each pass changes
# the weights and so the objective, and a test on the log-misfit, which is mostly noise, depends
# on the SNR; the LM fit that follows is what converges). For a model whose ln S is not linear
# in its parameters the two passes are two Gauss–Newton steps, an approximate start that LM
# refines: write such a model so that it is linear (DKI in V = MD²·W), or use `:data`. Weighting
# by the data, y², instead (Basser et al., J Magn Reson B 1994; this start's first version, and
# DTI's former `datastart`) puts the noise into the weights: its MD came out 2.0% low at
# b = 2500, against 0.2% for the two passes (docs/design.md, "Predicted-signal weights").
#
# Time constants (T2: no signal at 0) start very long instead and are stepped on their reciprocal,
# for which ln S is linear: ln S = ln S0 − TE·(1/T2). The Jacobian's column for u = 1/θ is
# ∂ln S/∂u = −θ²·∂ln S/∂θ, and the step lands at θ = 1/(1/θ + Δu); a step that would make u ≤ 0
# (data that do not decay) leaves θ where it is. Found by `setup` (`reciprocal`). So T2Decay
# starts from the same two passes as R2Decay, with no start of its own (it had a one-pass fit
# weighted by the data, `_loglinear_rate`, until 2026-11-08).
#
# What moves: the parameters starting at `:loglinear` (from 0, their placeholder), and the
# `:scale` parameter (the intercept: without it, the other parameters would absorb the signal's
# level). The rest stay where they are, fixed or at their own starts. The scale multiplies the
# signal, so ln S is linear in ln S0, not in S0: its step is taken on ln S0, S0·exp(ΔS0/S0)
# (Gauss–Newton's ΔS0/S0 is the step in ln S0), which keeps a pass exact for DTI; stepped in S0
# itself, it crept from its placeholder 1 towards 1000 over many steps (measured). The scale is
# then set again by least squares on y, as for every start (`_scale_start`), so the log fit's S0
# is only a step on the way.
#
# The pieces, each as in the solver or in DTI's former `datastart` (docs/design.md, "DTI built
# in"; scratch pivots.jl for the threshold):
#   - weights S²·w (w the fit's own observation weights), S divided by the data's largest value
#     since only the ratios matter; images with y ≤ 0 or S ≤ 0 skipped (no logarithm);
#   - sums in Float64 at least (Float32 fits too); the system scaled to a unit diagonal, then the
#     solver's `_cholesky`, which reports failure instead of throwing;
#   - a parameter the data do not measure at the current point (a zero column: a direction no
#     image has) does not move in that pass; a system the data cannot determine at all (one
#     shell and no b = 0; fewer directions than elements) shows as a pivot below eps^(1/4), and
#     the start stays where it is;
#   - each pass clamped to the bounds.
# Allocates nothing: the buffer is the solver's, the sums static arrays.
# -------------------------------------------------------------------------------------------------

const _LOGLINEAR_PASSES = 2          # OLS, then WLS weighted by OLS's predicted signal

function _loglinear_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec, buf) where {F,P,T}
    any(spec.fromlog) || return θ0               # no :loglinear start in this fit: no cost
    S = promote_type(Float64, T)                 # the precision of the sums and the solve
    ymax = S(maximum(prob.y))
    ymax > 0 || return θ0
    moves = ntuple(k -> spec.fromlog[k] || k == spec.scale, Val(P))   # what the log fit moves
    θ = θ0
    for pass in 1:_LOGLINEAR_PASSES
        q, A, g = _log_normal_equations(prob, θ, buf, ymax, S, pass > 1, spec.reciprocal)   # OLS, then S(θ)²
        isfinite(q) || return θ
        Δ, ok = _loglinear_step(A, g, moves)
        ok || return θ                           # the data cannot determine the parameters
        θ = clamp.(_logstep(θ, SVector{P,T}(Δ), spec.scale, spec.reciprocal), spec.lower, spec.upper)
    end
    return θ
end

# θ + Δ, with the scale parameter (number `sc`, 0 for none) stepped on its logarithm, and a time
# constant (`recip`) on its reciprocal: 1/(1/θ + Δ), or unchanged where that is not positive.
@inline function _logstep(θ::SVector{P,T}, Δ::SVector{P,T}, sc, recip) where {P,T}
    step(k) = recip[k] ? (u = inv(θ[k]) + Δ[k]; u > 0 ? inv(u) : θ[k]) : θ[k] + Δ[k]
    θn = SVector{P,T}(ntuple(step, Val(P)))
    (sc > 0 && θ[sc] > 0) || return θn
    return Base.setindex(θn, θ[sc] * exp(Δ[sc] / θ[sc]), sc)
end

# The weighted normal equations of the log fit at θ: Σw·r², JᵀWJ and JᵀWr, with r = ln S − ln y,
# J = ∂ln S/∂θ (∂ln S/∂(1/θ) = −θ²·∂ln S/∂θ for a time constant, `recip`) and w the fit's
# weight, times (S/ymax)² when `predicted` (S the signal at θ); summed in precision S.
@noinline function _log_normal_equations(prob::VoxelProblem{F}, θ::SVector{P,T}, buf, ymax,
                                         ::Type{S}, predicted::Bool, recip) where {F,P,T,S}
    signal!(buf, prob.model, _params(Val(F), _seed(θ), prob.fixed))   # S and ∂S/∂θ, as the solver
    d = SVector{P,S}(ntuple(k -> recip[k] ? -S(θ[k])^2 : one(S), Val(P)))   # the chain rule
    A = zero(SMatrix{P,P,S})
    g = zero(SVector{P,S})
    q = zero(S)
    @inbounds for i in eachindex(buf)
        y, f = S(prob.y[i]), S(value(buf[i]))
        (y > 0 && f > 0) || continue
        J = SVector{P,S}(partials(buf[i]).values) .* d * inv(f)          # ∂ln S/∂θ (or ∂/∂(1/θ))
        r = log(f / y)                                                   # ln S − ln y
        w = _wt(prob.w, i, S) * (predicted ? (f / ymax)^2 : one(S))      # the predicted signal², or 1
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
