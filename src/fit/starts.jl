# =================================================================================================
# Starting values: where each voxel's starting point comes from
#
# Per voxel, in this order (called from `_fit_voxel` in cache.jl):
#   1. the model's own estimates from the data, if it has them (`datastart`)      _data_start
#   2. the best value on the model's grid, if it has one (`startgrid`)             _grid_start
#      or else the `Free(:scale)` parameter (M0, S0, ...) at its best value        _scale_start
#
# Step 2 rests on one calculation, `_score`: the signal at a candidate starting point with the
# scale parameter solved exactly, and the SSE there. The grid scores each of its values; the scale
# start scores the one starting point.
# =================================================================================================

# Score a candidate starting point θ. The signal is proportional to the scale parameter (number
# `sc` among the free parameters), so with s = the signal at scale 1 (its "shape"), the scale that
# best matches the data y is the least-squares slope through the origin, a = (s·y) / (s·s), and
# the SSE there is Σ (a·s - y)² = y·y - 2a·(s·y) + a²·(s·s). Returns θ with that scale, and the
# SSE. Without a scale parameter (sc = 0): θ unchanged, and its SSE.
#
# If a is unusable (zero signal at θ, a NaN, or a ≤ 0, e.g. noise-only data) the scale stays at 1.
# The model is evaluated with plain numbers; storing them in the dual buffer just sets the
# derivative parts to zero, so no extra buffer is needed and nothing is allocated.
# `@inline`: the grid calls this once per grid value (30 times per voxel), and Julia does not
# inline a function this size by itself; without it, T1IR fits were 8% slower (measured).
@inline function _score(prob::VoxelProblem{F}, θ::SVector{P,T}, sc, lo, hi, buf) where {F,P,T}
    sc > 0 && (θ = Base.setindex(θ, one(T), sc))           # the signal at scale 1: its shape
    signal!(buf, prob.model, _params(Val(F), θ, prob.fixed))
    sy = ss = yy = zero(T)
    @inbounds for i in eachindex(buf)
        s, y = T(value(buf[i])), T(prob.y[i])
        sy += s * y
        ss += s * s
        yy += y * y
    end
    sc > 0 || return θ, yy - 2 * sy + ss                     # Σ (s - y)²
    a = sy / ss
    a = clamp(isfinite(a) && a > 0 ? a : one(T), lo[sc], hi[sc])
    return Base.setindex(θ, a, sc), yy - 2 * a * sy + a^2 * ss    # θ with its scale, Σ (a·s - y)²
end

# The scale parameter `sc` at its best value, given the other starts.
_scale_start(prob, θ0, sc, lo, hi, buf) = first(_score(prob, θ0, sc, lo, hi, buf))

# The best value on the model's grid (see `startgrid`) for free parameter k: score each grid value
# (the scale solved for each) and keep the lowest SSE. If none is usable (e.g. NaN signals), fall
# back to the scale start.
function _grid_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec, buf) where {F,P,T}
    k, sc = spec.gridindex, spec.scale
    best, bestsse = θ0, T(Inf)
    for g in spec.grid
        θ, sse = _score(prob, Base.setindex(θ0, g, k), sc, spec.lower, spec.upper, buf)
        if sse < bestsse                                     # NaN never wins
            best, bestsse = θ, sse
        end
    end
    isfinite(bestsse) && return best
    return sc > 0 ? _scale_start(prob, θ0, sc, spec.lower, spec.upper, buf) : θ0
end

# Starting values from the model's own estimates on this voxel's data (see `datastart`), for the
# free parameters not given a start in the call. A NaN or Inf estimate is skipped (the usual
# start stays); the others are clamped to the bounds. A `:scale` parameter is skipped too: it is
# set afterwards, by least squares given the other starts.
function _data_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec) where {F,P,T}
    d = datastart(prob.model, prob.y)
    d === nothing && return θ0                  # no data start: decided when compiling, no cost
    for (name, x) in pairs(d)
        k = findfirst(==(name), F)              # position among the free parameters, or nothing
        if k !== nothing && k != spec.scale && !spec.userstart[k] && isfinite(x)
            θ0 = Base.setindex(θ0, clamp(T(x), spec.lower[k], spec.upper[k]), k)
        end
    end
    return θ0
end
