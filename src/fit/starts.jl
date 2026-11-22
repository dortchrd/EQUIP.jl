# =================================================================================================
# Starting values: where each voxel's starting point comes from
#
# Each free parameter's start says where it comes from (see `Free`). θ0, the voxel's starting
# point (one value per free parameter), already holds the numbers and this voxel's value of any
# start map; then, per voxel, in this order (called from `_fit_voxel` in voxel.jl):
#   1. parameters starting at `:data`: the model's estimates from the data (`datastart`)   _data_start
#   2. parameters starting at `:loglinear`: the weighted fit of ln y with ln S         _loglinear_start
#      (in loglinear.jl, as the part of the start that needs most explaining)
#   3. parameters starting from `BestOf(values)`: the values that fit best               _best_start
#      (from a dictionary, the candidates' signals made once per fit,     _dictionary_start
#      when they are the same in every voxel: dictionary.jl)
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

# The best of the BestOf values (see `BestOf`), for the free parameters `spec.candindex`: score
# each combination (a column of `spec.candidates`; the scale solved for each) and keep the lowest
# SSE. If none is usable (e.g. NaN signals), fall back to the scale start from θ0, which holds
# the middle values. This evaluates the signal once per combination in every voxel; when the
# candidates' signals are the same in every voxel, `_dictionary_start` (dictionary.jl) scores
# them from signals computed once per fit instead.
function _best_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec, buf) where {F,P,T}
    ks, sc, C = spec.candindex, spec.scale, spec.candidates
    best, bestsse = θ0, T(Inf)
    for j in axes(C, 2)
        θ = θ0
        for q in eachindex(ks)                               # this combination's values
            θ = Base.setindex(θ, @inbounds(C[q, j]), @inbounds(ks[q]))
        end
        θ, sse = _score(prob, θ, sc, spec.lower, spec.upper, buf)
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
