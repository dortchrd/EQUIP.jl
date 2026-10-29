# =================================================================================================
# Diagnostics: where a fit's errors come from, computed from a result when called
#
#   cond(c)     how hard the data make the free parameters to tell apart (crlb's result)
#   cor(c)      the correlations of the estimates, predicted (crlb) ...
#   cor(mc)     ... and measured (montecarlo)
#
# Functions of a result rather than fields of it, as `fitted` and `residuals` are for a fit: a
# `crlb!` call does not pay for them (cond's eigenvalues, 1.5–2 µs, were a quarter of a 4-point
# `crlb!`), and each is computed from what the result keeps. Methods of LinearAlgebra's `cond`
# and Statistics' `cor`, re-exported. Next (docs/design.md, "Diagnostics"): the leakage of each
# fixed parameter into the fitted ones, dθ̂/dc, and sensitivity maps.
# =================================================================================================

"""
    cond(c::CRLBResult) -> Float64

How hard the data make the fit's free parameters to tell apart: the condition number of the
Jacobian J₀ (at the noise-free fit's values θ̂) with each column scaled by its parameter's true
value (relative sensitivities θⱼ·∂g/∂θⱼ, so the units drop out), over the parameters not on a
bound. 1 when the scaled columns are orthogonal and of equal length; large when some
combination of the parameters barely changes the signal, so that noise moves the estimates
along it (their SDs and correlations grow with it). It depends on the protocol and the truth,
not on the noise level. `Inf` when there are fewer observations than parameters to estimate, or
the columns are dependent. For a sweep, `cond.(s.results)`.

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)
cond(crlb(m, θ; snr = 200))                  # 23.5
cond(crlb(m, θ; snr = 200, kmf = Free()))    # kmf estimated too, 5 parameters from 4 points: Inf
```
"""
cond(c::CRLBResult) =
    _condition(c.gram, _freetruth(c), map(in(c.onbound), keys(c.fit)), length(c.σ))

# √(λmax/λmin) of the scaled J₀ᵀJ₀, P×P (no SVD of the N×P J₀): the scaled J₀'s singular values
# are the square roots of its eigenvalues. Each column scaled by its parameter's true value θ
# (1 for a true value of 0); the parameters on a bound (`pinned`) left out. Inf when there are
# fewer observations N than parameters, or the scaled J₀ᵀJ₀ is singular.
function _condition(G::SMatrix{P,P}, θ, pinned, N) where {P}
    U = any(pinned) ? [j for j in 1:P if !pinned[j]] : (1:P)    # usually none on a bound
    (isempty(U) || N < length(U)) && return Inf
    s = [θ[j] == 0 ? 1.0 : abs(θ[j]) for j in U]
    λ = eigvals(Symmetric(s .* Matrix(G)[U, U] .* s'))
    return λ[1] > 0 ? sqrt(λ[end] / λ[1]) : Inf
end

# The true value of each free parameter of a result (crlb's or montecarlo's), exactly (the
# result keeps its truth as given): read directly when a number, through `resolve` when the
# truth ties it to another parameter (rare: a tie is usually on a fixed parameter, R1m = :R1f).
# Also the sweep's table (sweep.jl).
function _freetruth(r)
    F = keys(r.sd)
    given = map(p -> r.truth[p], NamedTuple{F}(F))
    all(x -> x isa Real, values(given)) && return map(Float64, given)
    θt = resolve(r.model, r.truth)
    return NamedTuple{F}(map(p -> Float64(θt[p]), F))
end

"""
    cor(c::CRLBResult) -> Matrix

The predicted correlations of the estimates of the fit's free parameters (in model order,
`keys(c.fit)`), from the covariance [`vcov`](@ref)`(c)`: Cᵢⱼ / (SDᵢ·SDⱼ), 1 on the diagonal (as
Statistics' `cor`). Near ±1: noise moves the two estimates together (what makes `cond(c)`
large). `NaN` off the diagonal for a parameter held on a bound (SD 0) or not determined by the
data (SD `Inf`). [`cor`](@ref)`(mc)` measures the same from a Monte Carlo's estimates.

```julia
c = crlb(m, θ; snr = 200, kmf = 12.5)
keys(c.fit)                      # (:PSR, :R1f, :Sf, :M0f), the order of the rows and columns
cor(c)[1, 2]                     # PSR with R1f: -0.71
```
"""
function cor(c::CRLBResult)
    C = c.cov
    sd = sqrt.(diag(C))
    R = clamp.(C ./ (sd .* sd'), -1.0, 1.0)     # rounding can leave |r| just above 1, as in `cor`
    R[diagind(R)] .= 1.0
    return R
end

"""
    cor(mc::MonteCarloResult) -> Matrix

The measured correlations of the estimates of the fit's free parameters (in model order), over
every fitted copy, as `vcov(mc)`: Statistics' `cor` of the estimates. The prediction, from
derivatives: [`cor`](@ref)`(mc.crlb)`.
"""
cor(mc::MonteCarloResult) = cor(_estimates(mc))
