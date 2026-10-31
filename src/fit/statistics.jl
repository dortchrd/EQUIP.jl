# =================================================================================================
# Fit statistics: how good a fit is, and how uncertain its values, from the fit itself
#
#   nobs, dof, dof_residual, rss        the counts: N observations, P free parameters (+ σ)
#   residualsd                          σ̂ = √(rss/(N − P)): the noise, estimated from the residuals
#   vcov, stderror, cor                 σ̂²(JᵀJ)⁻¹ at the fit's answer (J the Jacobian there),
#                                       its square-rooted diagonal, its correlations
#   loglikelihood, aic, aicc, bic       Gaussian, σ estimated: for choosing between models fitted
#                                       to the same data (mono- against bi-exponential)
#   onbound                             the parameters the fit left on a bound
#
# StatsAPI's names (GLM's and StatsBase's), with one convention: `dof` counts the free
# parameters and σ (as GLM does for a linear model), `dof_residual` is N − P. For one voxel the
# values are numbers; for a dataset, maps with the spatial size of the data (each voxel its own
# fit), shaped like `r.params`. All assume Gaussian noise, the same for every observation: for
# magnitude data at low SNR (Rician) the residual SD and the standard errors are biased.
#
# The covariance is the classical one for a least-squares fit, first order in the noise:
# Cov(θ̂) ≈ σ²(JᵀJ)⁻¹, with σ the noise SD (estimated by σ̂, or given: `σ = ...`). It is what
# crlb gives at the fit's answer when nothing fixed is wrong (tested). A parameter on a bound was
# not estimated freely: its row is held (as in crlb), and its SD reported as NaN.
# =================================================================================================

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

# A parameter on a bound (within 1e-9, relative for large bounds).
@inline _near(x, bound) = isfinite(bound) && abs(x - bound) <= 1e-9 * max(1, abs(bound))

"""
    nobs(r::FitResult) -> Int
    dof(r::FitResult) -> Int
    dof_residual(r::FitResult) -> Int
    rss(r::FitResult) -> number or map

The number of observations N (per voxel, for a dataset), the degrees of freedom the fit uses
(its P free parameters and the noise σ: P + 1, as GLM counts), the residual degrees of freedom
N − P, and the residual sum of squares (`r.sse`). StatsAPI's names.
"""
nobs(r::FitResult) = nobs(r.model)
dof(r::FitResult) = length(r.params) + 1
dof_residual(r::FitResult) = nobs(r) - length(r.params)
rss(r::FitResult) = r.sse

"""
    residualsd(r::FitResult) -> number or map

The noise SD estimated from the residuals, σ̂ = √(rss/(N − P)), voxel by voxel: unbiased for σ²
when the model is right and the noise Gaussian. `NaN` when N = P (as many observations as free
parameters: the fit passes through every point, the residuals say nothing about the noise;
give the noise to `stderror` instead, `σ = ...`).
"""
residualsd(r::FitResult) = _onevoxel(r) ? _residualsd(r.sse, dof_residual(r)) :
                           _residualsd.(r.sse, dof_residual(r))
_residualsd(sse, ν) = ν > 0 ? sqrt(Float64(sse) / ν) : NaN

"""
    loglikelihood(r::FitResult) -> number or map
    aic(r::FitResult), aicc(r::FitResult), bic(r::FitResult) -> number or map

The Gaussian log-likelihood at the fit, with σ at its maximum-likelihood value (rss/N):
ℓ = −(N/2)(log(2π·rss/N) + 1); and the information criteria from it, AIC = −2ℓ + 2k,
AICc = AIC + 2k(k + 1)/(N − k − 1), BIC = −2ℓ + k·log N, with k = `dof(r)` (the free parameters
and σ). For choosing between models fitted to the same data: the lower, the better (a
difference above about 2 for AIC, 6 for BIC, is notable). Voxel by voxel for a dataset.
AICc is the one to use with few observations per parameter (Burnham & Anderson: N/k below
about 40, so nearly every voxel-wise fit). `NaN` when the residuals say nothing: all four when
N ≤ P (the fit passes through every point, as the 4-point SIR scheme's does), and AICc when
N ≤ k + 1 (its correction undefined).

```julia
r1 = fit(T2Decay(TE = te), y)
r2 = fit(T2BiexpDecay(TE = te), y)
bic(r2) < bic(r1)              # the second pool is worth its two parameters
```
"""
loglikelihood(r::FitResult) = _voxelwise(sse -> _loglik(sse, nobs(r), dof_residual(r)), r)
aic(r::FitResult) = _voxelwise(sse -> -2 * _loglik(sse, nobs(r), dof_residual(r)) + 2 * dof(r), r)
function aicc(r::FitResult)
    N, k = nobs(r), dof(r)
    # The correction 2k(k + 1)/(N − k − 1) is defined for N > k + 1 only: at N = k + 1 it is
    # infinite, below it negative (it would reward the parameters it is meant to penalize).
    N > k + 1 || return _voxelwise(sse -> NaN, r)
    return _voxelwise(sse -> -2 * _loglik(sse, N, dof_residual(r)) + 2k + 2k * (k + 1) / (N - k - 1), r)
end
bic(r::FitResult) =
    _voxelwise(sse -> -2 * _loglik(sse, nobs(r), dof_residual(r)) + log(nobs(r)) * dof(r), r)
# ℓ needs residuals: with N ≤ P (ν = N − P ≤ 0) the fit passes through every point, the SSE is
# rounding and its log meaningless (it made AIC and BIC huge and negative): NaN, as `residualsd`.
_loglik(sse, N, ν) = ν > 0 ? -N / 2 * (log(2π * Float64(sse) / N) + 1) : NaN
_voxelwise(f, r::FitResult) = _onevoxel(r) ? f(r.sse) : map(f, r.sse)

"""
    onbound(r::FitResult) -> named tuple of Bool (or of maps)

Which free parameters the fit left on a bound (within 1e-9, relative for large bounds), voxel
by voxel for a dataset: a value there was held by the bound, not determined by the data alone
(a sign the bounds, the model or the starts need a look). Their standard errors are `NaN`.
"""
function onbound(r::FitResult)
    c = _statscache(r)
    F, P = freenames(c.spec), length(r.params)
    lo, hi = c.spec.lower, c.spec.upper
    at(v, j) = (x = _paramvalue(r, F[j], v); _near(x, lo[j]) || _near(x, hi[j]))
    _onevoxel(r) && return NamedTuple{F}(ntuple(j -> at(1, j), P))
    return NamedTuple{F}(ntuple(j -> reshape(BitVector(at(v, j) for v in 1:length(r.sse)), size(r.sse)), P))
end

"""
    vcov(r::FitResult; σ) -> Matrix (one voxel) or array of matrices (a dataset)
    stderror(r::FitResult; σ) -> named tuple of numbers (or of maps)
    cor(r::FitResult) -> Matrix (or array of matrices)

The covariance of the fitted free parameters (model order, `keys(r.params)`), first order in
the noise: σ²(JᵀJ)⁻¹, J the Jacobian at the fit's answer; their standard errors, by name, like
`r.params` (`stderror(res).T2` a map for a dataset); and their correlations. σ: the noise SD,
from the residuals by default ([`residualsd`](@ref), voxel by voxel), or given: `σ = 10.0` (a
number, or a map), as from an image's background or a known SNR, which also works when N = P
(the 4-point SIR scheme: four observations, four free parameters).

A parameter on a bound ([`onbound`](@ref)) is held there (its row of JᵀJ masked, as in
`crlb`): its SD is `NaN`, the others as if it were fixed. Parameters the data cannot determine
(dependent columns of J, or fewer observations than parameters): `Inf`. Voxels that were not
fitted: `NaN`. When nothing fixed is wrong this is `crlb`'s SD at the fit's values, so the two
can be compared: what the data say against what the protocol predicts.

```julia
res = fit(T2Decay(TE = te), data; mask)
se = stderror(res).T2               # a map of the T2 standard error
stderror(res; σ = 12.0).T2          # with the noise known instead
```
"""
vcov(r::FitResult; σ = nothing) = _onevoxel(r) ? Matrix(first(_covariances(r, σ))) : _covariances(r, σ)
function stderror(r::FitResult; σ = nothing)
    C = _covariances(r, σ)
    F = keys(r.params)
    sd(c, j) = (v = c[j, j]; isnan(v) || v >= 0 ? sqrt(v) : NaN)
    _onevoxel(r) && return NamedTuple{F}(ntuple(j -> sd(C[1], j), length(F)))
    return NamedTuple{F}(ntuple(j -> map(c -> sd(c, j), C), length(F)))
end
function cor(r::FitResult)
    R = map(_correlation, _covariances(r, nothing; unit = true))
    return _onevoxel(r) ? Matrix(first(R)) : R
end
# Cᵢⱼ/(sᵢsⱼ), 1 on the diagonal (as Statistics' cor, also for a held parameter), NaN beside a
# held or undetermined one.
function _correlation(C::SMatrix{P,P,T}) where {P,T}
    s = sqrt.(diag(C))
    return SMatrix{P,P,T}(ntuple(n -> (i = (n - 1) % P + 1; j = (n - 1) ÷ P + 1;
                                       i == j ? one(T) : clamp(C[i, j] / (s[i] * s[j]), -one(T), one(T))),
                                 Val(P * P)))
end

# The covariance of every voxel, as static P×P matrices (an array of the maps' size; one entry
# for one voxel). `unit = true`: with σ = 1 (for correlations, which do not depend on σ).
function _covariances(r::FitResult, σ; unit::Bool = false)
    c = _statscache(r)
    return _covariances(c, c.spec, r, σ, unit)
end
function _covariances(c::FitCache, spec::FitSpec{F,P,T}, r, σ, unit) where {F,P,T}
    nv = _onevoxel(r) ? 1 : length(r.sse)
    σ isa AbstractArray && size(σ) != size(r.sse) &&
        throw(DimensionMismatch("`σ` is a map of size $(size(σ)); the fit's maps are $(size(r.sse))"))
    ν = dof_residual(r)
    N = length(c.y)
    fill!(c.y, zero(T))                              # JᵀJ does not depend on the data
    out = Vector{SMatrix{P,P,T,P * P}}(undef, nv)
    for v in 1:nv
        θ = SVector{P,T}(ntuple(j -> T(_paramvalue(r, F[j], v)), Val(P)))
        s2 = unit ? one(T) : σ === nothing ? T(_residualsd(_sse(r, v), ν))^2 : T(_pick(σ, v, T))^2
        # This voxel's fixed values: as the fit used them (`r.fixed`: numbers and ties, or maps
        # for a dataset), in the cache's form (`_value`: a tie by name as a `_Tie`).
        fixed = map(x -> _pick(_value(x, T), v, T), r.fixed)
        out[v] = _covariance(c, spec, θ, s2, fixed, N)
    end
    return _onevoxel(r) ? out : reshape(out, size(r.sse))
end
# One voxel: JᵀJ at θ (the solver's own sums, with this voxel's fixed values), parameters on a
# bound held (masked to the identity, as crlb does), then σ²(JᵀJ)⁻¹; NaN rows for the held ones.
function _covariance(c, spec::FitSpec{F,P,T}, θ, s2, fixed, N) where {F,P,T}
    nan = fill(T(NaN), SMatrix{P,P,T})
    all(isfinite, θ) && isfinite(s2) || return nan                       # not fitted
    _, A, _ = normal_equations(VoxelProblem{F}(c.model, c.y, fixed), θ, c.buf)
    pinned = map(_near, θ, spec.lower) .| map(_near, θ, spec.upper)
    u = map(p -> ifelse(p, zero(T), one(T)), pinned)
    keep(C) = SMatrix{P,P,T}(ntuple(n -> (i = (n - 1) % P + 1; j = (n - 1) ÷ P + 1;
                                          pinned[i] || pinned[j] ? T(NaN) : C[i, j]), Val(P * P)))
    Am = (u * u') .* A + Diagonal(one(T) .- u)
    L, ok = _cholesky(Am)
    (N >= count(!, pinned) && ok && _determined(A, u)) ||
        return keep(fill(T(Inf), SMatrix{P,P,T}))
    Ai = hcat(ntuple(j -> _chol_solve(L, SVector{P,T}(ntuple(k -> ifelse(k == j, one(T), zero(T)),
                                                              Val(P)))), Val(P))...)
    return keep(s2 .* (Ai + Ai') ./ 2)
end

# The fit's own setup again (its free parameters, bounds and fixed values, maps included), in
# Float64 whatever the fit's precision: the statistics are sums and an inverse, worth the digits.
function _statscache(r::FitResult)
    kw = Base.structdiff(r.settings, NamedTuple{(:mask, :precision)})
    return init(r.model; kw..., precision = Float64)
end
_paramvalue(r::FitResult, n, v) = _onevoxel(r) ? r.params[n] : r.params[n][v]
_sse(r::FitResult, v) = _onevoxel(r) ? r.sse : r.sse[v]
