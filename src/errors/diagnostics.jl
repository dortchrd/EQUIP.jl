# =================================================================================================
# Diagnostics: where a fit's errors come from, computed from a result when called
#
#   cond(c)     how hard the data make the free parameters to tell apart (crlb's result)
#   cor(c)      the correlations of the estimates, predicted (crlb) ...
#   cor(mc)     ... and measured (montecarlo)
#   leakage(m, s)   how an error in each fixed parameter moves the fitted ones, dθ̂/dc
#   sensitivityefficiency(m, θ; duration)   what each observation tells about each
#                   parameter per √s of scan (Eq. 6)
#
# Functions of a result rather than fields of it, as `fitted` and `residuals` are for a fit: a
# `crlb!` call does not pay for them (cond's eigenvalues, 1.5–2 µs, were a quarter of a 4-point
# `crlb!`), and each is computed from what the result keeps. Methods of LinearAlgebra's `cond`
# and Statistics' `cor`, re-exported. `leakage` needs no crlb result (no noise): it takes the
# protocol and a scenario, as crlb does; `sensitivityefficiency` the protocol and the truth.
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

# -------------------------------------------------------------------------------------------------
# Leakage: how an error in each fixed parameter moves the fitted ones
#
#   the fit's gradient  G(θ̂; c) = J₀ᵀ r = 0   for every value c the fit assumes for a fixed
#                                              parameter (r = g(θ̂; c) − y, y the true signal)
#   so (implicit function theorem)   dθ̂/dc = −H_θθ⁻¹ H_θc
#   with the blocks of the cost's Hessian over (θ, c):  H_θθ = Σᵢ JᵢJᵢᵀ + rᵢ∇²_θθ gᵢ (crlb's H)
#                                                       H_θc = Σᵢ Jᵢ(∂gᵢ/∂c)ᵀ + rᵢ∂²gᵢ/∂θ∂c
#   at the truth (r = 0): −(JᵀJ)⁻¹Jᵀ∂g/∂c, the linear leakage; away from it (a fixed value
#   already wrong) the curvature terms are 21–26% of the answer (docs/design.md, "Diagnostics")
#
# Times an error Δc, dθ̂/dc is the first-order bias, b ≈ (dθ̂/dc)·Δc, for every fixed parameter at
# once and without a fit per assumed value. Both blocks come from one evaluation of the model:
# a second cache in which the fixed numbers are free parameters too (`joint`), so `hessian!`
# seeds them with the free ones, as it seeds the free ones for crlb.
# -------------------------------------------------------------------------------------------------

# Made once per fit: crlb's cache (the noise-free fit, θ̂) and the joint one (the free parameters
# and the fixed numbers C, in model order, all seeded: H over (θ, c)). `C` (their names) and `K`
# (their number) are type parameters, so the kernel's matrices have static sizes.
struct LeakageCache{C,K,A<:CRLBCache,B<:CRLBCache}
    crlb::A
    joint::B
end

function _LeakageCache(m::MRIModel, s::Scenario)
    c = _CRLBCache(m, s, :sandwich)               # checks the protocol against the scenario
    # The fixed parameters held at a number (a tie, R1m = :R1f, follows the parameter it names:
    # its own leakage, of a deviation from the tie, is for later).
    C = Tuple(n for n in keys(s.roles) if s.roles[n] isa Fixed && s.roles[n].value isa Real)
    isempty(C) && throw(ArgumentError("leakage: the fit fixes no parameter at a number"))
    # The joint fit: the same roles with those fixed numbers free. Nothing is fitted with it, so
    # the starts are placeholders (a number each, so that no `:data` or `BestOf` is worked out).
    joint = map(n -> n in C ? Free(start = 0.0) :
                     s.roles[n] isa Free ? Free(start = 0.0, lower = s.roles[n].lower,
                                                upper = s.roles[n].upper) : s.roles[n],
                NamedTuple{keys(s.roles)}(keys(s.roles)))
    a = _crlbcache(m, c.mT, _spec(c.mT, joint, Float64), false)
    return LeakageCache{C,length(C),typeof(c),typeof(a)}(c, a)
end

# Where each of `names` sits in the joint cache's free parameters (model order).
@inline _positions(names::NTuple{P,Symbol}, among::Tuple) where {P} =
    SVector{P,Int}(ntuple(j -> something(findfirst(==(names[j]), among)), Val(P)))

"""
    _leakage!(lc, m, t) -> (; fit, truth, D, status, pinned, exact)

The kernel of [`leakage`](@ref), without allocation: the noise-free fit θ̂ (as crlb's), the
true free values, D = dθ̂/dc (P×K, a static matrix: column k for fixed parameter C[k]; 0 for a
free parameter held on a bound, NaN when the data cannot determine the free parameters), the
noise-free fit's status, which free parameters are on a bound, and whether the fit was exact.
`t` from `_crlb_truth`; `m` the cache's protocol or another with as many observations.
"""
function _leakage!(lc::LeakageCache{C,K}, m::MRIModel, t::NamedTuple) where {C,K}
    c, a = lc.crlb, lc.joint
    F, FA = _crlbnames(c), _crlbnames(a)
    P, PA = length(F), length(FA)
    T = Float64

    # ① The noise-free fit, as crlb's: θ̂, and the parameters it leaves on a bound (held there).
    θ̂, θt, r = noisefree!(c, m, t, m, t)
    pinned = map(_near, θ̂, c.lower) .| map(_near, θ̂, c.upper)
    u = map(p -> ifelse(p, zero(T), one(T)), pinned)
    exact = r.sse <= _EXACT^2 * sum(abs2, c.y)

    # ② The cost's Hessian over (θ, c) at θ̂ and the values the fit assumes, for the same data:
    #    the joint cache, with the free values θ̂ and the fixed numbers in model order. When the
    #    fit is exact (r = 0) the curvature terms are 0 and first-order duals suffice.
    copyto!(a.y, c.y)
    at = merge(c.fixed, NamedTuple{F}(Tuple(θ̂)))
    x = SVector{PA,T}(ntuple(j -> T(getfield(at, FA[j])), Val(PA)))
    H, G = hessian!(a, m, x, exact)
    iF, iC = _positions(F, FA), _positions(C, FA)
    Hθθ, Hθc, Gθθ = H[iF, iF], H[iF, iC], G[iF, iF]

    # ③ Can the data determine the free parameters (as crlb's ③)? Otherwise NaN.
    Hm = (u * u') .* Hθθ + Diagonal(one(T) .- u)     # held parameters: identity rows
    L, ok = _cholesky(Hm)
    N = length(c.y)
    nan = fill(T(NaN), SMatrix{P,K,T})
    out(D) = (fit = θ̂, truth = θt, D, status = r.status, pinned, exact)
    (N >= count(!, pinned) && _determined(Gθθ, u) && ok) || return out(nan)

    # ④ dθ̂/dc = −H_θθ⁻¹H_θc, one column per fixed parameter; a held parameter's row is 0.
    return out(-hcat(ntuple(k -> _chol_solve(L, u .* Hθc[:, k]), Val(K))...))
end

"""
    LeakageResult

What [`leakage`](@ref) returns. Three named tuples, by fixed parameter and then free parameter
(`l.absolute.kmf.PSR`), each the slope dθ̂ⱼ/dc of the noise-free fit's free parameter θ̂ⱼ against
the value c the fit assumes for a fixed parameter:

- `absolute`: dθ̂ⱼ/dc, in θⱼ's units per unit of c (PSR per s⁻¹ of kmf)
- `relative`: the same divided by |θⱼ| (the true value), per unit of c: times an error Δc, the
  first-order relative bias (as crlb's `relbias`)
- `elasticity`: (|c|/|θⱼ|)·dθ̂ⱼ/dc, the % change of θ̂ⱼ per % change of c: comparable between
  fixed parameters with different units (kmf in s⁻¹, Sm without); 0 for a c of 0

and `fit` (the noise-free fit θ̂), `truth`, `fixed` (the fit's fixed values: where the slopes
are taken), `onbound` (free parameters on a bound in the noise-free fit: held there, slope 0),
`status` (the noise-free fit's) and `model`. `NaN` when the data cannot determine the free
parameters (as crlb's SD `Inf`).
"""
struct LeakageResult{M<:MRIModel,A<:NamedTuple,N<:NamedTuple,Tr<:NamedTuple,X<:NamedTuple}
    absolute::A
    relative::A
    elasticity::A
    fit::N
    truth::Tr
    fixed::X
    onbound::Vector{Symbol}
    status::FitStatus
    model::M
end

"""
    leakage(m, s::Scenario) -> LeakageResult
    leakage(m, θ; changes...)

How much an error in each fixed parameter leaks into the fitted ones: for each parameter the
fit holds at a number c (kmf, Sm, ... in qMT), dθ̂/dc, how the noise-free fit's answer moves per
unit change in the value it assumes, at that fit (the truth when nothing fixed is wrong). Times
an error Δc it is the first-order bias, b ≈ (dθ̂/dc)·Δc (for the exact one, [`crlb`](@ref) at
the wrong value): which assumptions matter for a protocol and tissue, and by how much, without a
fit per assumed value. The slope of Dortch et al., MRM 2018;80:1824, Fig. 2B.

- `m`: the protocol, a model with its acquisition.
- `s`: the scenario, the truth and the fit; or the truth θ with the changes as keywords, as for
  `crlb` (`kmf = 12.5`: the slopes where the fit assumes 12.5).

From derivatives, no fits beyond crlb's noise-free one: dθ̂/dc = −H_θθ⁻¹H_θc, the blocks of the
fit's cost's Hessian over the free parameters θ and the fixed ones c (the implicit function
theorem: the fit's gradient is 0 whatever c it assumes); at the truth, −(JᵀJ)⁻¹Jᵀ∂g/∂c. No
noise: leakage is a property of the protocol, the truth and the fit. Ties (`R1m = :R1f`) follow
the parameter they name, and are not among the fixed parameters reported.

```julia
m = QMTSIR(; SIR_5POINT...)
l = leakage(m, nominal(m, :SIR_WM_I))       # kmf fixed at the truth's 5 s⁻¹
l.relative.kmf.PSR                          # −0.119: PSR 11.9% low per s⁻¹ kmf is assumed high
l.elasticity.kmf.PSR                        # % of PSR per % of kmf
l.absolute.kmf.PSR * 0.25                   # the first-order bias of assuming 5.25: crlb's, nearly
leakage(m, nominal(m, :SIR_WM_I); kmf = 12.5)   # the slopes where the fit assumes 12.5
```
"""
function leakage(m::MRIModel, θ::NamedTuple; changes...)
    ch = _fitsettings("leakage", m, changes, ())
    return leakage(m, Scenario(m, θ; ch...))
end
function leakage(m::MRIModel, s::Scenario)
    lc = _LeakageCache(m, s)
    return _leakageresult(lc, s.truth, _crlb_truth(m, s.truth))
end

# The kernel's numbers as a LeakageResult: by fixed parameter, then free parameter. A separate
# function (a function barrier) so that the names and sizes are known types here.
function _leakageresult(lc::LeakageCache{C,K}, truth, t) where {C,K}
    c = lc.crlb
    F = _crlbnames(c)
    P = length(F)
    k = _leakage!(lc, c.mT, t)
    cv = map(n -> Float64(getfield(c.fixed, n)), C)            # the values the fit assumes
    bytable(f) = NamedTuple{C}(ntuple(j -> NamedTuple{F}(ntuple(i -> f(k.D[i, j], k.truth[i], cv[j]),
                                                                Val(P))), Val(K)))
    return LeakageResult(bytable((d, θ, x) -> d), bytable((d, θ, x) -> d / abs(θ)),
                         bytable((d, θ, x) -> d * abs(x) / abs(θ)), NamedTuple{F}(Tuple(k.fit)),
                         truth, map(_untie, c.fixed), Symbol[F[j] for j in 1:P if k.pinned[j]],
                         k.status, c.model)
end

# At the REPL: the relative slopes (in %) and the elasticities (a ratio, % per %), one row per
# fixed parameter. A slope of zero is shown as 0 (not -0.0).
function Base.show(io::IO, ::MIME"text/plain", l::LeakageResult)
    num(x) = isfinite(x) ? string(iszero(x) ? 0 : round(x; sigdigits = 3)) : string(x)
    pct(x) = isfinite(x) ? num(100 * x) * "%" : string(x)
    print(io, "leakage for ", nameof(typeof(l.model)), ", ", nobs(l.model), " observations: ",
          "how the noise-free fit moves per unit of error in each fixed value")
    _print_fixed(io, l.fixed, resolve(l.model, l.truth))
    isconverged(l.status) ||
        print(io, "\nthe noise-free fit did not converge ($(l.status)): the slopes may be inaccurate")
    isempty(l.onbound) ||
        print(io, "\non a bound in the noise-free fit (held there, slope 0): ", join(l.onbound, ", "))
    F = keys(l.fit)
    for (title, tab, f) in (("relative: % of each true value per unit of the fixed value", l.relative, pct),
                            ("elasticity: % of each true value per 1% of the fixed value", l.elasticity, num))
        print(io, "\n", title)
        header = ("fixed", "value", string.(F)...)
        rows = [(string(n), _showvalue(l.fixed[n]), (f(tab[n][p]) for p in F)...) for n in keys(tab)]
        w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
        print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
        for r in rows
            print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
        end
    end
end

# -------------------------------------------------------------------------------------------------
# Sensitivity efficiency: how much each observation tells about each parameter, per √s of scan
# (Dortch et al., MRM 2018;80:1824, Eq. 6)
#
#   S_eff = (∂S/∂θₗ) / √t          t: the observation's time (TI + TD + t_FSE for SIR)
#
# large where an observation measures θₗ precisely for its time, small where the signal is blind
# to it. The paper evaluates it on a grid of TI and TD (Fig. 2A) and contours each map at 85% of
# its largest value; both are its plot's choices, not part of the quantity, so neither is built
# in here: the observations are any protocol's (a grid is one, made in three lines: see the
# docstring), and thresholds are the reader's.
# -------------------------------------------------------------------------------------------------

"""
    SensitivityEfficiency

What [`sensitivityefficiency`](@ref) returns: two named tuples by parameter (`e.relative.PSR`),
each a vector with one value per observation of the protocol:

- `absolute`: (∂S/∂θ)/√t, Dortch et al. 2018, Eq. 6 (t each observation's `duration`): the
  signal's units per unit of θ, per √s
- `relative`: θ·(∂S/∂θ)/(scale·√t), the signal's change for a relative change of θ
  (θ → θ(1 + ε), either sign of θ), as a fraction of the model's scale parameter (M0f, S0; 1 if
  it has none): comparable between parameters (the paper's maps, in %)

and `duration` (each observation's time), `truth` and `model` (the protocol).
"""
struct SensitivityEfficiency{M<:MRIModel,N<:NamedTuple,Tr<:NamedTuple}
    absolute::N
    relative::N
    duration::Vector{Float64}
    truth::Tr
    model::M
end

"""
    sensitivityefficiency(m, θ; duration, parameters) -> SensitivityEfficiency

How much each observation of protocol `m` tells about each parameter per √s of scan time, the
sensitivity efficiency of Dortch et al., MRM 2018;80:1824, Eq. 6: S_eff = (∂S/∂θ)/√t at the true
values θ. Which observations inform which parameter, and where they overlap: the 4-point SIR
scheme measures PSR, R1f, Sf and M0f near their most efficient observations while avoiding
kmf's (so fixing kmf costs little).

- `m`: the protocol: its observations are the candidates (a published scheme, or a grid of
  acquisition values, below).
- `θ`: the true values, every parameter (numbers or ties), one voxel, as for [`signal`](@ref).
- `duration`: the time each observation takes: a function of the protocol
  (`m -> m.TI .+ m.TD .+ 0.3`, TI + TD + t_FSE for SIR, as `ProtocolDesign`'s `scantime` without
  the sum), a vector or a number. (The sensitivity alone, ∂S/∂θ for the fit's free parameters,
  is [`jacobian`](@ref)'s.)
- `parameters`: the parameters to differentiate, by name; default every parameter given as a
  number (ties follow the parameter they name: with `R1m = :R1f`, R1f moves both pools).

The derivatives are exact (dual numbers, through the model's own `signal!`: for magnitude data,
of the magnitude, as the paper's script takes `abs` of the signal).

A grid of candidate observations is a protocol with every combination as its observations; the
values reshape to the grid (the first axis varying fastest), e.g. Fig. 2A's maps, with the
paper's script's values and grid:

```julia
θ = nominal(QMTSIR; PSR = 0.12, R1f = 1.0, Sf = -1.0, kmf = 12.5)
ti = exp10.(range(log10(0.01), log10(5), 1000)); td = exp10.(range(log10(0.01), log10(5), 100))
points = vec(collect(Iterators.product(ti, td)))               # every (TI, TD), TI fastest
e = sensitivityefficiency(QMTSIR(TI = first.(points), TD = last.(points)), θ;
                          duration = m -> m.TI .+ m.TD .+ 0.3, parameters = (:PSR, :R1f, :Sf, :M0f, :kmf))
reshape(e.relative.PSR, length(ti), length(td))                # the PSR map (×100 for %)
e4 = sensitivityefficiency(QMTSIR(; SIR_4POINT...), θ; duration = m -> m.TI .+ m.TD .+ 0.3)
abs.(e4.relative.PSR) ./ maximum(abs, e.relative.PSR)          # the scheme against the grid's best
```
"""
function sensitivityefficiency(m::MRIModel, θ::NamedTuple; duration, parameters = nothing)
    names = _efficiencynames(m, θ, parameters)
    t = _durations(m, duration)
    # One ForwardDiff Jacobian of the signal at every observation (dual numbers through
    # `signal`, ties followed), then each column divided by √t (and scaled by θ/scale).
    x0 = [Float64(θ[n]) for n in names]
    J = ForwardDiff.jacobian(x -> signal(m, merge(θ, NamedTuple{names}(Tuple(x)))), x0)
    w = inv.(sqrt.(t))
    scale = _scalename(m)
    sv = scale === nothing ? 1.0 : Float64(resolve(m, θ)[scale])
    col(f) = NamedTuple{names}(ntuple(f, length(names)))
    return SensitivityEfficiency(col(k -> J[:, k] .* w), col(k -> J[:, k] .* w .* (x0[k] / sv)),
                                 t, θ, m)
end

# The parameters to differentiate: given, or every one given a number in θ (in model order).
function _efficiencynames(m, θ, parameters)
    resolve(m, θ)                                      # θ complete, numbers or ties
    any(x -> x isa AbstractArray, values(θ)) &&
        throw(ArgumentError("sensitivityefficiency is for one voxel: give numbers (or ties), not maps"))
    parameters === nothing && return Tuple(n for n in keys(θ) if θ[n] isa Real)
    names = Tuple(parameters)
    for n in names
        haskey(θ, n) || throw(ArgumentError("`$n` is not a parameter of $(nameof(typeof(m)))"))
        θ[n] isa Real || throw(ArgumentError("`$n` is a tie in θ ($(repr(θ[n]))); untie it " *
                                             "(a number) to differentiate it"))
    end
    return names
end

# Each observation's time: from a function of the protocol, a vector or a number; positive.
_durations(m, d::Function) = _durations(m, d(m))
_durations(m, d::Real) = _durations(m, fill(Float64(d), nobs(m)))
function _durations(m, d::AbstractVector)
    length(d) == nobs(m) || throw(DimensionMismatch("`duration` has $(length(d)) values; the " *
                                                    "protocol has $(nobs(m)) observations"))
    all(>(0), d) || throw(ArgumentError("`duration` must be positive"))
    return Vector{Float64}(d)
end

# At the REPL: the relative efficiencies at each observation (one row per parameter), or, for
# many observations (a grid), each parameter's largest and the observation it is at.
function Base.show(io::IO, ::MIME"text/plain", e::SensitivityEfficiency)
    num(x) = iszero(x) ? "0" : string(round(x; sigdigits = 3))
    pct(x) = iszero(x) ? "0%" : num(100 * x) * "%"
    names = keys(e.relative)
    scale = _scalename(e.model)
    N = nobs(e.model)
    print(io, "sensitivity efficiency for ", nameof(typeof(e.model)), ", ", N,
          " observations (Dortch 2018 Eq. 6)")
    print(io, "\nat ", join(("$n = $(_showvalue(x))" for (n, x) in pairs(e.truth)), ", "))
    print(io, "\nrelative: θ·∂S/∂θ per √s, in ", scale === nothing ? "the signal's units" : "% of $scale")
    # The acquisition at observation i: the model's vector fields (TI, TD, TE, ...).
    acq = NamedTuple(k => getfield(e.model, k) for k in fieldnames(typeof(e.model))
                     if getfield(e.model, k) isa AbstractVector)
    at(i) = join(("$k = $(num(v[i]))" for (k, v) in pairs(acq)), ", ")
    if N <= 12
        header = ("parameter", ("$i" for i in 1:N)...)
        rows = [(string(n), (pct(v) for v in e.relative[n])...) for n in names]
    else
        header = ("parameter", "largest", "at")
        rows = [(i = argmax(abs.(e.relative[n])); (string(n), pct(e.relative[n][i]), at(i)))
                for n in names]
    end
    w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for r in rows
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end
