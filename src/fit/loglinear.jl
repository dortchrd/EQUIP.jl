# =================================================================================================
# The log-linear start, `Free(start = :loglinear)`: the weighted fit of ln y with ln S, from the
# model's own signal (step 2 of each voxel's start, starts.jl)
#
#   _loglinear_start     the two passes, OLS then WLS weighted by the predicted signal²
#   _logpass             the first pass, made once per fit when it is the same in every voxel
#   _warn_loglinear      `init`'s warning where ln S is not linear in those parameters
#
# Which parameters start here, and which are time constants fitted through 1/θ, is found by
# `setup` (setup.jl: `fromlog`, `reciprocal`). Here since 2026-11-12; in starts.jl before.
# =================================================================================================

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
# starts from the same two passes as DTI, with no start of its own (it had a one-pass fit
# weighted by the data, `_loglinear_rate`, until 2026-11-08; R2Decay and ADCDecay keep their
# constant starts, from which their rate steps land in a few iterations: t2_decay.jl).
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
#
# The first pass, made once per fit (`_logpass`): it starts every voxel from the same point (the
# placeholders, and every other parameter's number) when nothing in the start varies by voxel,
# no map among the starts or the fixed values and no `:data` start. The signal there and ∂ln S/∂θ
# are then the same for every voxel, so they are computed when the cache is made, and a voxel's
# first pass sums its own data against them; the second pass, at the voxel's own point, still
# evaluates the model. The same arithmetic, row by row, so the same starts bit for bit (checked:
# T2Decay, R2Decay, ADCDecay, DTI, DKI, both precisions). Measured (2026-11-08, scratch llt/
# share.jl, one session): the start 20–32% faster (T2Decay at 32 echoes 2.92 → 2.35 µs, DTI
# 4.38 → 3.70, DKI 56.6 → 38.4), the whole fit 5–8%. What it cannot share is most of the rest:
# each voxel's logarithms, in both passes, and the second pass at the voxel's own point.
# -------------------------------------------------------------------------------------------------

const _LOGLINEAR_PASSES = 2          # OLS, then WLS weighted by OLS's predicted signal

function _loglinear_start(prob::VoxelProblem{F}, θ0::SVector{P,T}, spec, buf, logpass = nothing) where {F,P,T}
    any(spec.fromlog) || return θ0               # no :loglinear start in this fit: no cost
    S = promote_type(Float64, T)                 # the precision of the sums and the solve
    ymax = S(maximum(prob.y))
    ymax > 0 || return θ0
    moves = ntuple(k -> spec.fromlog[k] || k == spec.scale, Val(P))   # what the log fit moves
    θ = θ0
    for pass in 1:_LOGLINEAR_PASSES
        q, A, g = pass == 1 && logpass !== nothing ?
            _log_normal_equations(prob, logpass) :                       # the first pass, made once
            _log_normal_equations(prob, θ, buf, ymax, S, pass > 1, spec.reciprocal)   # OLS, then S(θ)²
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

# The first pass at the placeholder start, the same in every voxel: the signal f and ∂ln S/∂θ
# there (∂ln S/∂(1/θ) for a time constant), made when the cache is made; `nothing` when the start
# varies by voxel (a map among the starts or the fixed values, a `:data` start) or there is no
# `:loglinear` start. Computed exactly as `_log_normal_equations` computes them.
struct _LogPass{S,P}
    f::Vector{S}
    J::Vector{SVector{P,S}}
end
function _logpass(model, spec::FitSpec{F,P,T}, n) where {F,P,T}
    any(spec.fromlog) || return nothing
    (any(spec.fromdata) || any(x -> x isa AbstractArray, values(spec.start)) ||
     any(x -> x isa AbstractArray, values(spec.fixed))) && return nothing
    S = promote_type(Float64, T)
    θ = SVector{P,T}(Tuple(spec.start))
    buf = Vector{Dual{FitTag,T,P}}(undef, n)
    signal!(buf, model, _params(Val(F), _seed(θ), map(x -> _pick(x, 1, T), spec.fixed)))
    d = SVector{P,S}(ntuple(k -> spec.reciprocal[k] ? -S(θ[k])^2 : one(S), Val(P)))
    f = [S(value(b)) for b in buf]
    J = [SVector{P,S}(partials(b).values) .* d * inv(S(value(b))) for b in buf]
    return _LogPass{S,P}(f, J)
end

# A voxel's first pass from the made one: the OLS normal equations of its data (the fit's own
# observation weights only), as `_log_normal_equations` sums them.
@noinline function _log_normal_equations(prob::VoxelProblem, lp::_LogPass{S,P}) where {S,P}
    A = zero(SMatrix{P,P,S})
    g = zero(SVector{P,S})
    q = zero(S)
    @inbounds for i in eachindex(lp.f)
        y, f = S(prob.y[i]), lp.f[i]
        (y > 0 && f > 0) || continue
        J = lp.J[i]
        r = log(f / y)
        w = _wt(prob.w, i, S) * one(S)
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

# -------------------------------------------------------------------------------------------------
# Is the start the log-linear fit? A warning when it is not (the owner, 2026-11-12: "keep
# :loglinear and add the setup warning")
#
# The two passes are the weighted linear regression of ln y only when ln S is linear in what they
# move: each `:loglinear` parameter (through 1/θ for a time constant), with the scale as ln S0.
# Then ∂ln S/∂φ (φ = θ, or 1/θ) does not depend on φ: the same at any two points. ADCDecay's ADC,
# DTI, DKI in V = MD²·W and T2Decay's T2 (through 1/T2) are; ADCKurtosis with ADC and K both at
# `:loglinear` is not (ln S = ln S0 − b·ADC + b²·ADC²·K/6: ∂ln S/∂K = b²·ADC²/6 is 0 at the
# start, ADC = 0, and grows with ADC). There the passes are two Gauss–Newton steps on ln S, a
# start that LM refines, and the name would mislead, so the fit says so once (`init`).
#
# The two points: the start (the placeholders; every other parameter at its start or fixed
# value, the first voxel's of a map), and the start with each `:loglinear` parameter moved by
# 1/max|∂ln S/∂φ|, the step that changes ln S by about 1 in its most sensitive observation (so
# the test needs no unit), clamped to the bounds. The scale is not moved: by the `:scale` rule
# the signal is proportional to it, so it adds ln S0 and changes no other derivative. Observations
# whose signal is not positive at either point (no logarithm) are left out, as the passes leave
# them out. A derivative counts as changed when it moves by more than 1e-6 of its largest value:
# measured (2026-11-12, scratch llwarn/margin.jl), at most 3.5e-16 for the linear models above
# (T2Decay's 1/T2 the largest; DKI 1.4e-16 at 63 images), 1.0 for ADCKurtosis's K, both
# precisions. Computed in Float64 whatever the fit's precision: linearity is the model's, not
# the arithmetic's.
#
# Returns `nothing`, or the name of the parameter whose derivative changed most and by how much
# (a fraction of the derivative's largest value). Cost, once per fit set up (measured): 1.3 µs
# for T2Decay, 4.5 µs for DTI, 25 µs for DKI, about 12% of a one-voxel `fit(m, y)` (most of it
# `init` itself); nothing per voxel.
# -------------------------------------------------------------------------------------------------

const _LOGLINEAR_TOL = 1e-6

function _loglinear_nonlinearity(model, spec::FitSpec{F,P}) where {F,P}
    any(spec.fromlog) || return nothing
    θ0 = SVector{P,Float64}(map(x -> _pick(x, 1, Float64), Tuple(spec.start)))
    fixed = map(x -> _pick(x, 1, Float64), spec.fixed)
    n = nobs(model)
    b0, b1 = Vector{Dual{FitTag,Float64,P}}(undef, n), Vector{Dual{FitTag,Float64,P}}(undef, n)
    signal!(b0, model, _params(Val(F), _seed(θ0), fixed))
    s0 = zero(SVector{P,Float64})                # max |∂ln S/∂φ| at the start, by parameter
    for s in b0
        J = _logderivative(s, θ0, spec.reciprocal)
        J === nothing || (s0 = max.(s0, abs.(J)))
    end
    θ1 = _logmove(θ0, s0, spec)
    signal!(b1, model, _params(Val(F), _seed(θ1), fixed))
    big = zero(SVector{P,Float64})               # max |∂ln S/∂φ| at either point
    dif = zero(SVector{P,Float64})               # max change between them
    for i in 1:n
        J0 = _logderivative(b0[i], θ0, spec.reciprocal)
        J1 = _logderivative(b1[i], θ1, spec.reciprocal)
        (J0 === nothing || J1 === nothing) && continue
        big = max.(big, abs.(J0), abs.(J1))
        dif = max.(dif, abs.(J1 .- J0))
    end
    worst, change = 0, 0.0
    for k in 1:P                                 # a parameter no observation measures: skipped
        spec.fromlog[k] && big[k] > 0 && dif[k] / big[k] > change && ((worst, change) = (k, dif[k] / big[k]))
    end
    return change > _LOGLINEAR_TOL ? (F[worst], change) : nothing
end

# The second point: each `:loglinear` parameter moved by 1/s (s its largest |∂ln S/∂φ|; through
# 1/θ for a time constant), clamped to its bounds. A function of its own, not a closure: `s0` is
# reassigned in the loop above, and a closure capturing it would box it (measured: 25 kB
# allocated and 11 µs for DTI's check, against 8 kB, the two buffers, and 4.5 µs).
function _logmove(θ0::SVector{P,Float64}, s0, spec) where {P}
    moved(k) = !spec.fromlog[k] || s0[k] == 0 ? θ0[k] :
               clamp(spec.reciprocal[k] ? inv(inv(θ0[k]) + 1 / s0[k]) : θ0[k] + 1 / s0[k],
                     spec.lower[k], spec.upper[k])
    return SVector{P,Float64}(ntuple(moved, Val(P)))
end

# ∂ln S/∂φ of one observation (∂S/∂θ / S, times −θ² for a time constant: ∂/∂(1/θ)), or
# `nothing` where its signal is not positive (no logarithm).
@inline function _logderivative(s::Dual, θ::SVector{P,T}, recip) where {P,T}
    f = value(s)
    (isfinite(f) && f > 0) || return nothing
    d = SVector{P,T}(ntuple(k -> recip[k] ? -θ[k]^2 : one(T), Val(P)))
    return SVector{P,T}(partials(s).values) .* d ./ f
end

# The warning, once per model and set of `:loglinear` parameters in a session (`maxlog`, counted
# by `_id`): a dataset, a montecarlo or a sweep sets up its fit once each, and would say it again.
function _warn_loglinear(model, spec::FitSpec{F}) where {F}
    r = _loglinear_nonlinearity(model, spec)
    r === nothing && return nothing
    name, change = r
    M = nameof(typeof(model))
    lognames = Tuple(n for (n, l) in zip(F, spec.fromlog) if l)
    k = findfirst(==(name), F)
    d = spec.reciprocal[k] ? "∂ln S/∂(1/$name)" : "∂ln S/∂$name"
    @warn("`$(join(lognames, "`, `"))` start at :loglinear, but the logarithm of $M's signal is " *
          "not linear in them ($d changed by $(round(100 * change; sigdigits = 2))% between two " *
          "points): the start is then two Gauss–Newton steps on ln S, which the fit refines, not " *
          "the weighted log-linear fit. It is that fit for a signal whose logarithm is linear " *
          "(ADCDecay's ADC, DTI, DKI in V = MD²·W, T2Decay's T2 through 1/T2); for this one, " *
          "write the signal so, or start these parameters otherwise (a number, :data, BestOf).",
          maxlog = 1, _id = (:loglinear, M, lognames))
    return nothing
end
