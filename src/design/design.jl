# =================================================================================================
# ProtocolDesign: the objective of a protocol search, made once, evaluated many times
#
#   d = ProtocolDesign(m, scenarios; snr, acquisition, constraints, scantime)
#   d(x)            the objective at the acquisition values x (one vector: the TIs, then the TDs)
#   d(m)            the same at a protocol, a model (e.g. a published scheme)
#
# The default objective is Eq. 5 of Dortch et al., MRM 2018;80:1824 (after Li et al., MRM
# 2010;64:491, and Cercignani & Alexander, MRM 2006;56:803): the worst scenario's sum of relative
# mean squared errors, times the scan time,
#
#   v(x) = maxₗ Σⱼ Wⱼ (bⱼₗ² + σⱼₗ²)/θⱼₗ² · T(x)
#
# with bⱼₗ and σⱼₗ the bias and SD of free parameter j in scenario l (a tissue, and what the fit
# assumes: crlb's kernel), θⱼₗ its true value, Wⱼ its weight (1 each by default) and T the scan
# time. A wrong assumption in a scenario (kmf = 12.5 for a tissue with kmf = 5) counts as the bias
# it causes. Options: the weights, a fixed time budget instead of × T, and how the scenarios are
# combined (the maximum, minimax, by default).
#
# Why an object: a search evaluates the objective tens of thousands of times. What does not
# change between protocols is made once: one CRLBCache per fit (scenarios that differ only in
# the truth share it), each scenario's truth in the kernel's form and its noise. Each evaluation
# then builds the protocol with static vectors and calls the allocation-free kernel once per
# scenario: no allocation (tested), 11.5 µs for the 4-point scheme over 4 tissues.
# docs/design.md, "The plan: ProtocolDesign".
# =================================================================================================

# The scenarios that share a fit: the cache, and each one's truth, noise and place in the list.
# One group per fit and kind of truth, so the vectors below have concrete element types.
struct _Group{C<:CRLBCache,TT,P,S}
    cache::C                       # the fit these scenarios share
    truths::Vector{TT}             # their truths, in the kernel's form (`_crlb_truth`)
    σ2::Vector{S}                  # their noise variances: a number each (one σ for every
                                   # observation), or a vector each (σ per observation, averages)
    weights::SVector{P,Float64}    # each free parameter's weight, in the cache's order
    index::Vector{Int}             # their places among the design's scenarios
end

"""
    ProtocolDesign(m, scenarios; snr or σ, averages, acquisition, scantime, constraints, shared,
                   weights, budget, aggregate, cov)

The objective of a protocol search, made once: called with the varied acquisition values
`d(x)`, or with a protocol `d(m)`, it gives Eq. 5 of Dortch et al., MRM 2018 (after Li et al.,
MRM 2010), the worst scenario's sum of relative mean squared errors times the scan time:

    v = maxₗ Σⱼ Wⱼ (bⱼₗ² + σⱼₗ²)/θⱼₗ² · T

(bias, SD and true value of free parameter j in scenario l, from [`crlb`](@ref)'s kernel).
[`optimize_protocol`](@ref) searches it.

- `m`: the protocol to start from, a model with its acquisition: it sets the number of points
  and every field that is not varied (`magnitude`, ...).
- `scenarios`: a [`Scenario`](@ref) or several (a vector, or a set from [`scenarios`](@ref)),
  each a truth and the fit judged against it (a tissue, and what the fit assumes):
  `scenarios(QMTSIR, tissues; kmf = 12.5)`. A wrong assumption counts as the bias it causes;
  uncertain conditions are more scenarios, the truth changed and the fit kept:
  `scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.83, 0.88], kmf = 12.5, Sm = 0.83)`. After an
  evaluation, `d.values` holds each scenario's value.
- The noise, independent: `snr` (relative to each scenario's scale parameter, as for `crlb`)
  or `σ` (a number, or one per point); with `averages` (the number of averages of each point,
  fixed in the search), one average's, σ/√NEX for each point. Count the averages in
  `scantime` too: `m -> sum(nex .* (m.TI .+ m.TD .+ 0.3))`.
- `acquisition`: the acquisition fields to optimize, with their bounds (s): `(TI = (0.01, 10.0),
  TD = (0.01, 10.0))`. `x` is their values one after the other (the TIs, then the TDs). With
  `shared = (:TD,)`, one TD for every point (one value in `x`). (Not `vary`, which elsewhere,
  in `crlb`, `montecarlo` and `scenarios`, changes true parameter values.)
- `scantime`: the scan time of a protocol (s), a function: for SIR-FSE Σ(TI + TD + t_FSE),
  `m -> sum(m.TI .+ m.TD .+ 0.3)` (t_FSE is the readout's, which the model does not hold).
- `constraints`: a function of the protocol whose values must all be ≥ 0, or `nothing`: e.g.
  `m -> m.TI .+ m.TD .- 0.5` (TR ≥ 0.8 s with t_FSE = 0.3 s, for SAR). `d(x)` is `Inf` where a
  constraint is broken.
- `weights`: per free parameter, by name (default 1 each, as Eq. 5); `weights = (PSR = 1,
  R1f = 1, Sf = 0, M0f = 0)`.
- `budget`: `nothing` (default, × T: precision per unit time, Eq. 5), or a scan time B (s):
  Σⱼ Wⱼ(bⱼ² + σⱼ²·T/B)/θⱼ², averaging to fill the budget shrinks the variance, never the bias.
- `aggregate`: how the scenarios are combined, a function of the vector of their values:
  `maximum` (default, minimax), `mean`, `v -> quantile(v, 0.9)`, ...
- `cov`: `:sandwich` (default) or `:biased` (Eq. 3), as for `crlb`.

Not thread-safe (the caches' buffers): [`optimize_protocol`](@ref) gives each start a copy.

```julia
tissues = (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)
ss = scenarios(QMTSIR, tissues; kmf = 12.5)                 # the four tissues, kmf assumed 12.5
m = QMTSIR(; SIR_4POINT..., magnitude = false)
d = ProtocolDesign(m, ss; snr = 200, acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),
                   scantime = m -> sum(m.TI .+ m.TD .+ 0.3), constraints = m -> m.TI .+ m.TD .- 0.5)
d(m)                                                        # Eq. 5 of the Table 2 scheme
d([0.01, 0.01, 0.278, 1.007, 4.171, 0.684, 2.73, 0.01])    # the same, as x
```
"""
struct ProtocolDesign{N,V,W,M,X,O,H,G<:Tuple,S,C,A}
    model::M                       # the protocol given (the template)
    fields::X                      # its fields, vectors as SVector{N,Float64}: the base of every protocol
    lower::Vector{Float64}         # the bounds of x
    upper::Vector{Float64}
    offsets::O                     # where each varied field starts in x (a tuple of Ints)
    shared::H                      # whether it is one value for every point (a tuple of Bools)
    groups::G                      # the scenarios, by fit: `_Group`s
    scenarios::Vector{Scenario}
    values::Vector{Float64}        # each scenario's value at the last evaluation
    scantime::S
    constraints::C
    aggregate::A
    budget::Union{Nothing,Float64}
    cov::Symbol
end
# The type parameters: N, the number of points; V, the names of the varied fields (e.g. (:TI,
# :TD)); W, the model's type without its parameters (e.g. QMTSIR), to build each protocol. As
# type parameters, the compiler knows them, so building a protocol from x is static code.

function ProtocolDesign(m::MRIModel, scenarios; snr = nothing, σ = nothing,
                        averages = nothing, acquisition = nothing, scantime = nothing,
                        constraints = nothing, shared = (), weights::NamedTuple = (;),
                        budget = nothing, aggregate = maximum, cov::Symbol = :sandwich,
                        vary = nothing, noise = nothing, averaging = nothing)
    # `noise` and `averaging` are taken only to say why they are refused (the objective is
    # crlb's: Gaussian noise, first order), `vary` only to say where its meaning went:
    # elsewhere it changes true values.
    noise === nothing || throw(_noise_refused("ProtocolDesign"))
    averaging === nothing || throw(_noise_refused("ProtocolDesign", :averaging))
    vary === nothing || throw(ArgumentError("`vary` changes true parameter values (crlb, " *
        "montecarlo, scenarios); the acquisition to optimize is `acquisition = (TI = (0.01, " *
        "10.0), TD = (0.01, 10.0))`, each field with its bounds"))
    acquisition isa NamedTuple && !isempty(acquisition) ||
        throw(ArgumentError("give `acquisition`, the fields to optimize with their bounds (s), " *
                            "e.g. `acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0))`"))
    ss = scenarios isa Scenario ? Scenario[scenarios] : Scenario[s for s in scenarios]
    isempty(ss) && throw(ArgumentError("give at least one scenario"))
    foreach(s -> _check_protocol(m, s), ss)
    scantime === nothing && throw(ArgumentError("give `scantime = m -> ...`, the scan time of " *
        "a protocol (s): for SIR-FSE Σ(TI + TD + t_FSE), e.g. `m -> sum(m.TI .+ m.TD .+ 0.3)`"))
    _check_cov(cov)
    (snr === nothing) + (σ === nothing) == 1 ||
        throw(ArgumentError("give the noise as `snr` or `σ` (one of them)"))
    budget === nothing || budget > 0 || throw(ArgumentError("`budget` is a scan time (s), > 0"))

    # The protocol space: the varied fields, their bounds, and where each sits in x.
    N = nobs(m)
    names = fieldnames(typeof(m))
    for (f, b) in pairs(acquisition)
        f in names || throw(ArgumentError("`acquisition`: $(nameof(typeof(m))) has no field `$f`; " *
            "its fields are $(join(names, ", "))"))
        getfield(m, f) isa AbstractVector{<:Real} && length(getfield(m, f)) == N ||
            throw(ArgumentError("`acquisition`: `$f` is not a per-observation field (a vector of $N numbers)"))
        b isa Tuple{Real,Real} && isfinite(b[1]) && isfinite(b[2]) && b[1] < b[2] ||
            throw(ArgumentError("`acquisition`: the bounds of `$f` as (lower, upper), finite, e.g. `$f = (0.01, 10.0)`"))
    end
    for f in shared
        haskey(acquisition, f) || throw(ArgumentError("`shared`: `$f` is not in `acquisition`"))
    end
    V = keys(acquisition)
    sh = map(f -> f in shared, V)
    len = map(s -> s ? 1 : N, sh)
    offsets = ntuple(j -> sum(len[1:j-1]; init = 0), length(V))
    lower = reduce(vcat, (fill(Float64(acquisition[f][1]), len[j]) for (j, f) in enumerate(V)))
    upper = reduce(vcat, (fill(Float64(acquisition[f][2]), len[j]) for (j, f) in enumerate(V)))
    fields = NamedTuple{names}(map(x -> _static(x, Val(N)), _fields(m)))

    # The scenarios, by fit: one cache for each, shared by the scenarios that differ only in the
    # truth (and the kind of truth, so each group's truths have one type).
    keyof(s) = (map(_fitkey, s.roles), typeof(_crlb_truth(m, s.truth)), s.weights)  # the weights too
    keys_ = Any[]
    members = Vector{Int}[]
    for (i, s) in enumerate(ss)
        k = keyof(s)
        g = findfirst(x -> isequal(x, k), keys_)
        g === nothing ? (push!(keys_, k); push!(members, [i])) : push!(members[g], i)
    end
    free = unique(n for s in ss for (n, r) in pairs(s.roles) if r isa Free)
    for n in keys(weights)
        n in free || throw(ArgumentError("`weights`: `$n` is not a free parameter of the scenarios' fits ($(join(free, ", ")))"))
        weights[n] >= 0 || throw(ArgumentError("`weights`: `$n` must be ≥ 0"))
    end
    groups = Tuple(_group(m, ss, idx, snr, σ, averages, weights, cov) for idx in members)

    W = Base.typename(typeof(m)).wrapper
    d = ProtocolDesign{N,V,W,typeof(m),typeof(fields),typeof(offsets),typeof(sh),typeof(groups),
                       typeof(scantime),typeof(constraints),typeof(aggregate)}(
        m, fields, lower, upper, offsets, sh, groups, ss, zeros(length(ss)), scantime,
        constraints, aggregate, budget === nothing ? nothing : Float64(budget), cov)
    d(m)                   # once, so a mistake in scantime or constraints shows here
    return d
end

# A field of the model as every protocol holds it: a vector of N numbers as a static vector;
# anything else (a flag, a scalar) as it is.
_static(x::AbstractVector{<:Real}, ::Val{N}) where {N} =
    length(x) == N && !(x isa AbstractVector{Bool}) ? SVector{N,Float64}(x) : x
_static(x, ::Val{N}) where {N} = x

# What makes two scenarios' fits the same, for the cache: which parameters are free, their
# bounds, and the fixed values (starts aside: the kernel starts at the truth).
_fitkey(r::Free) = (:free, r.lower, r.upper)
_fitkey(r::Fixed) = (:fixed, r.value)

# One group: the cache of the first scenario's fit, and every member's truth, noise and weights.
function _group(m, ss, idx, snr, σ, averages, weights, cov)
    # The Rician fits' crlb sets its noise and weights from each truth's signal, which here would
    # change with every protocol tried: not yet in the design's allocation-free kernel.
    any(i -> ss[i].likelihood === :rician, idx) && throw(ArgumentError("a protocol design for " *
        "fits with likelihood = :rician: not yet (crlb and montecarlo of such fits are)"))
    return _groupof(_CRLBCache(m, ss[idx[1]], cov), m, ss, idx, snr, σ, averages, weights)
end
# A separate function so that F and P are known here (a function barrier).
function _groupof(c::CRLBCache{F,P}, m, ss, idx, snr, σ, averages, weights) where {F,P}
    truths = [_crlb_truth(m, ss[i].truth) for i in idx]
    # Each scenario's noise in the kernel's form: one variance (a number), or one per point (a
    # vector: σ per point, or averages that differ), so that the group's vector has one
    # concrete element type, a number each or a vector each.
    s2 = [_kernelnoise(noisecovariance(m, ss[i].truth; snr, σ, averages), P) for i in idx]
    σ2 = all(x -> x isa Real, s2) ? Float64[x for x in s2] :
         [x isa Real ? fill(Float64(x), length(c.y)) : Vector{Float64}(x) for x in s2]
    w = SVector{P,Float64}(map(n -> Float64(get(weights, n, 1.0)), F))
    for t in truths, j in 1:P
        w[j] > 0 && _freevalues(c, t)[j] == 0 && throw(ArgumentError("`$(F[j])` is 0 in a " *
            "scenario's truth: its relative error is not defined; give it weight 0"))
    end
    return _Group(c, truths, σ2, w, collect(idx))
end

# -------------------------------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------------------------------

# The protocol for the values x: the model's fields, the varied ones from x, as static vectors.
# `merge` replaces the varied fields and keeps the order, so the fields go back to the model's
# positional constructor (`W(...)`, e.g. QMTSIR(TI, TD, magnitude)) in order.
@inline function _protocol(d::ProtocolDesign{N,V,W}, x::AbstractVector) where {N,V,W}
    varied = NamedTuple{V}(ntuple(j -> _segment(x, d.offsets[j], d.shared[j], Val(N)), Val(length(V))))
    return W(values(merge(d.fields, varied))...)
end
@inline _segment(x, off, shared, ::Val{N}) where {N} =
    SVector{N,Float64}(ntuple(i -> @inbounds(x[shared ? off + 1 : off + i]), Val(N)))

"""
    (d::ProtocolDesign)(x) -> Float64
    (d::ProtocolDesign)(m::MRIModel) -> Float64

The objective at the acquisition values `x` (the varied fields one after the other, as
`acquisition` lists them) or at a protocol `m` (a model of the same kind, with as many points). `Inf`
where a constraint is broken or the data cannot determine the parameters. `x` is not checked
against the bounds (the search keeps it within them).
"""
(d::ProtocolDesign)(x::AbstractVector{<:Real}) = _objective(d, _protocol(d, x))
(d::ProtocolDesign)(m::MRIModel) = _objective(d, _checked(d, m))

function _checked(d::ProtocolDesign{N,V,W}, m) where {N,V,W}
    m isa W && nobs(m) == N || throw(ArgumentError("give a $(nameof(W)) with $N points"))
    return _with_precision(m, Float64)
end

_objective(d::ProtocolDesign, p) = _objective(_rank(d, p))
_objective(r::Tuple) = r[1] > 0 ? Inf : r[2]          # Inf where a constraint is broken

# (how much the constraints are broken, the objective): the order the search compares points
# in, feasibility first (Deb, Comput Methods Appl Mech Eng 2000;186:311). Tuples compare
# element by element, so `isless` on them is exactly that rule.
_rank(d::ProtocolDesign, x::AbstractVector{<:Real}) = _rank(d, _protocol(d, x))
function _rank(d::ProtocolDesign, p::MRIModel)
    v = _violation(d.constraints, p)
    v > 0 && return (v, Inf)
    T = Float64(d.scantime(p))
    τ = d.budget === nothing ? 1.0 : T / d.budget        # the variance's share of the budget
    _scenariovalues!(d.values, d.groups, p, τ)
    a = Float64(d.aggregate(d.values))
    return (0.0, d.budget === nothing ? a * T : a)
end

# How much the constraints are broken: the sum of the negative parts of their values.
_violation(::Nothing, p) = 0.0
_violation(g, p) = _negpart(g(p))
_negpart(g::Real) = max(0.0, -Float64(g))
_negpart(g) = sum(x -> max(0.0, -Float64(x)), g; init = 0.0)

# Each scenario's Σⱼ Wⱼ(bⱼ² + τσⱼ²)/θⱼ², group by group. The groups are a tuple of different
# types (different fits), so they are taken one at a time by recursion (`first`, `Base.tail`),
# which the compiler unrolls: each group's loop is compiled for its own types.
@inline _scenariovalues!(v, ::Tuple{}, p, τ) = nothing
@inline function _scenariovalues!(v, groups::Tuple, p, τ)
    g = first(groups)
    @inbounds for i in eachindex(g.truths)
        k = _crlb!(g.cache, p, g.truths[i], g.σ2[i])
        e = sum(g.weights .* (k.bias .^ 2 .+ τ .* diag(k.cov)) ./ k.truth .^ 2)
        v[g.index[i]] = isfinite(e) ? e : Inf            # undetermined parameters: Inf
    end
    _scenariovalues!(v, Base.tail(groups), p, τ)
end

# A copy with buffers of its own (the caches and the values), for another task.
function Base.copy(d::ProtocolDesign{N,V,W,M,X,O,H,G,S,C,A}) where {N,V,W,M,X,O,H,G,S,C,A}
    groups = map(g -> _Group(_copy(g.cache), g.truths, g.σ2, g.weights, g.index), d.groups)
    return ProtocolDesign{N,V,W,M,X,O,H,G,S,C,A}(d.model, d.fields, d.lower, d.upper, d.offsets,
        d.shared, groups, d.scenarios, zeros(length(d.values)), d.scantime, d.constraints,
        d.aggregate, d.budget, d.cov)
end
# (The weights are read only, so the copies share them; the buffers are each task's own.)
_copy(c::CRLBCache{F,P,T,M,MT,X,B1,B2,W}) where {F,P,T,M,MT,X,B1,B2,W} =
    CRLBCache{F,P,T,M,MT,X,B1,B2,W}(c.model, c.mT, c.lower, c.upper, c.fixed, c.opts, c.biased,
                                    similar(c.y), similar(c.fitbuf), similar(c.jbuf), similar(c.hbuf),
                                    c.w, c.rician)

# The values x of a protocol (a shared field: its first value), within the bounds.
function _x(d::ProtocolDesign{N,V}, m) where {N,V}
    x = reduce(vcat, (d.shared[j] ? [Float64(getfield(m, f)[1])] : Float64.(collect(getfield(m, f)))
                      for (j, f) in enumerate(V)))
    return clamp.(x, d.lower, d.upper)
end

function Base.show(io::IO, d::ProtocolDesign{N,V}) where {N,V}
    b(j, f) = "$f ∈ [$(d.lower[d.offsets[j]+1]), $(d.upper[d.offsets[j]+1])]" * (d.shared[j] ? " (shared)" : "")
    print(io, "ProtocolDesign(", nameof(typeof(d.model)), ", ", N, " points: ",
          join((b(j, f) for (j, f) in enumerate(V)), ", "), "; ", length(d.scenarios),
          " scenario", length(d.scenarios) == 1 ? "" : "s", ")")
end
