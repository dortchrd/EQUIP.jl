# =================================================================================================
# Fitting magnitude data with Rician noise: `fit(m, data; likelihood = :rician, σ, method)`
#
#   likelihood = :rician    the noise model the fit assumes: magnitude data, σ the noise SD of
#                           each of the real and imaginary channels (a number, or a map with one
#                           value per voxel, read voxel by voxel as a fixed parameter's map is)
#   method = :ml            the full Rician model: maximum likelihood (the default)
#            :mean          least squares on the Rician mean, E|y| = ricianmean(A, σ)
#            :wmean         the same, each observation weighted by 1/Var|y| (`ricianvar`)
#
# Decided with the owner, 2026-11-09 (docs/design.md, "Tier 2, bias from noise at low SNR: the
# plan", Decisions 1): `likelihood` names the noise model, `method` how the fit uses it.
#
# Each voxel is fitted twice, in one `fit!`:
#
#   1. as without the keyword: least squares on the model itself, from the model's own starts
#      (`:loglinear`, `datastart`, `:scale`, `BestOf`), which assume the signal itself and work
#      on the data as they are (measured, step 0: started this way, the Rician fits converged);
#   2. from that result, with the Rician noise model:
#      - `:mean`: the model's signal through the Rician mean (`_RicianMeanSignal`), least squares;
#      - `:wmean`: the same, weighted by 1/ricianvar(A, σ), the weights from the previous pass's
#        values (iteratively reweighted least squares, quasi-likelihood: Wedderburn, Biometrika
#        1974;61:439), `_WMEAN_PASSES` passes, the first from the Gaussian fit's values;
#      - `:ml`: the model's magnitude (`_RicianLikelihood`), and in place of the sum of squares the
#        Rician negative log-likelihood, minimized by Fisher scoring inside the same
#        Levenberg–Marquardt solver (its own `normal_equations`, below; Sijbers et al., IEEE Trans
#        Med Imaging 1998;17:357; Karlsen et al., Magn Reson Med 1999;41:614).
#
# The result is a FitResult of the model itself: the same maps, derived quantities, `refit`. Its
# `sse` is Σ(y − E|y|)², the data against their expected value under the fit, for every method;
# its `iterations` both fits'. The standard errors use the known σ: the inverse Rician
# information for `:ml`, the sandwich with the Rician variance for `:mean`, (JᵀV⁻¹J)⁻¹ for
# `:wmean` (statistics.jl's `_covariances`, below). Not yet: weights given by the user, averages,
# σ fitted, the Rician log-likelihood and AIC/BIC of a fit (they need the data).
# =================================================================================================

const _LIKELIHOODS = (:gaussian, :rician)
const _METHODS = (:ml, :mean, :wmean)
const _WMEAN_PASSES = 2      # the weights from the Gaussian fit's values, then from the first pass's

# -------------------------------------------------------------------------------------------------
# The models the second fit sees: the model's signal through the Rician mean, or its magnitude.
# σ is one of their parameters (in the fit's fixed values, `p.σ`), so that a σ map is read voxel by
# voxel like any fixed parameter's.
struct _RicianMeanSignal{M<:MRIModel} <: MRIModel
    model::M
end
struct _RicianLikelihood{M<:MRIModel} <: MRIModel
    model::M
end
nobs(w::Union{_RicianMeanSignal,_RicianLikelihood}) = nobs(w.model)
# The model's magnitude |S| with the ordinary least-squares sums: what crlb's prediction of the
# :ml fit fits, weighted by the Rician information (src/errors/crlb.jl, `_rician!`).
struct _Magnitude{M<:MRIModel} <: MRIModel
    model::M
end
nobs(w::_Magnitude) = nobs(w.model)
relabel(w::_Magnitude, θ) = relabel(w.model, θ)
derived(w::_Magnitude, p) = derived(w.model, p)
function signal!(out, w::_Magnitude, p)
    signal!(out, w.model, p)
    @inbounds for j in eachindex(out)
        out[j] = abs(out[j])
    end
    return out
end

function signal!(out, w::_RicianMeanSignal, p)
    signal!(out, w.model, p)
    σ = p.σ
    @inbounds for j in eachindex(out)
        out[j] = ricianmean(abs(out[j]), σ)
    end
    return out
end
function signal!(out, w::_RicianLikelihood, p)
    signal!(out, w.model, p)
    @inbounds for j in eachindex(out)
        out[j] = abs(out[j])                  # on a dual number, the derivatives' sign follows
    end
    return out
end

# The Rician fit's sums, for Levenberg–Marquardt as Fisher scoring. With Aⱼ the model's magnitude
# (its derivatives Jⱼ from dual numbers) and zⱼ = yⱼAⱼ/σ², the negative log-likelihood is, up to
# terms without θ,
#
#   Σⱼ (yⱼ − Aⱼ)²/2σ² − log(e^(−zⱼ)·I₀(zⱼ))         ((yⱼ² + Aⱼ²)/2σ² − log I₀(zⱼ), written stably)
#
# its gradient Σⱼ (Aⱼ − yⱼ·I₁/I₀(zⱼ))/σ² · Jⱼ, and its expected Hessian (the Fisher information)
# Σⱼ R(Aⱼ/σ)/σ² · JⱼJⱼᵀ (`ricianinformation`'s factor, from its table). The solver's quantities
# are then: "sse" = 2 × that objective, g = the gradient, A = the information, so that its step
# (A + μD)h = −g and its predicted decrease −(2gᵀs + sᵀAs) are Fisher scoring's, damped. At
# high SNR (I₁/I₀ → 1, R → 1) these are least squares' sums divided by σ²: the same fit.
@noinline function normal_equations(prob::VoxelProblem{F,<:_RicianLikelihood}, θ::SVector{P,T},
                                    buf) where {F,P,T}
    p = _params(Val(F), _seed(θ), prob.fixed)
    signal!(buf, prob.model, p)                  # |signal|, with its derivatives
    σ2 = T(prob.fixed.σ)^2
    A = zero(SMatrix{P,P,T})
    g = zero(SVector{P,T})
    q = zero(T)
    y = prob.y
    @inbounds for i in eachindex(buf)
        d = buf[i]
        J = SVector{P,T}(partials(d).values)
        a = T(value(d))
        yi = T(y[i])
        i0, i1 = _i01e(yi * a / σ2)
        ρ = i0 > 0 ? i1 / i0 : zero(T)           # I₁/I₀(z), 0 at z = 0
        q += (yi - a)^2 / σ2 - 2 * log(i0)
        g += ((a - yi * ρ) / σ2) * J
        A += (T(_ricianRtable(a / sqrt(σ2))) / σ2) * (J * J')
    end
    return q, A, g
end

# -------------------------------------------------------------------------------------------------
"""
    RicianFitCache

What [`init`](@ref) makes for a fit with `likelihood = :rician`: the cache of the fit of the
model itself (the first fit, its starts) and the cache of the fit with the Rician noise model
(the second, from the first's result), with σ among the second's fixed values. Used as a
[`FitCache`](@ref) is: [`fit!`](@ref)`(cache, y; voxel)`. Not thread-safe.
"""
struct RicianFitCache{G<:FitCache,R<:FitCache,M,S<:FitSpec,T,K}
    gauss::G                # the first fit: the model itself
    rician::R               # the second: its Rician mean, or its magnitude with the likelihood
    method::Symbol
    model::M                # the model, in the working precision (the results' model)
    spec::S                 # the second fit's spec: the free parameters, bounds, σ among the fixed
    y::Vector{T}            # (the first fit's data buffer: the observations' count)
    w::Vector{T}            # :wmean's weights for this voxel
    s::Vector{T}            # the signal, in plain numbers
    settings::K
end

function Base.show(io::IO, c::RicianFitCache)
    print(io, "RicianFitCache(", nameof(typeof(c.model)), "; method :", c.method, ", free ",
          freenames(c.spec), ", ", length(c.y), " observations, ", eltype(c.y), ")")
end

_precision(::RicianFitCache{G,R,M,S,T}) where {G,R,M,S,T} = T
function _copycache(c::RicianFitCache)
    g = _copycache(c.gauss)
    return RicianFitCache(g, _copycache(c.rician), c.method, c.model, c.spec, g.y, similar(c.w),
                          similar(c.s), c.settings)
end

# `init(m; likelihood = :rician, σ, method, ...)`: the two caches. The first is `init` without
# the Rician keywords; the second shares its spec (free parameters, bounds), with σ added to the
# fixed values and none of the starts (it starts from the first fit's values).
function _rician_init(model::MRIModel, method, precision, solver, weights, params)
    method === nothing && (method = :ml)
    method in _METHODS || throw(ArgumentError("`method` is one of $(join(repr.(_METHODS), ", ")) " *
        "(:ml, the full Rician model, by default), got $(repr(method))"))
    haskey(params, :σ) || throw(ArgumentError("likelihood = :rician needs the noise SD, `σ`: a " *
        "number, or a map with one value per voxel (σ of each channel, as `simulate`'s)"))
    :σ in keys(parameters(model)) && throw(ArgumentError("$(nameof(typeof(model))) has a parameter " *
        "named `σ`, which likelihood = :rician uses for the noise; rename the parameter"))
    weights === nothing || throw(ArgumentError("`weights` with likelihood = :rician: not yet " *
        "(method = :wmean weights by the Rician variance)"))
    σ = params[:σ]
    if σ isa Real
        isfinite(σ) && σ > 0 || throw(ArgumentError("`σ` must be finite and > 0, got $σ"))
    elseif !(σ isa AbstractArray{<:Real})
        throw(ArgumentError("`σ` is a number or a map (an array of numbers), got $(repr(σ))"))
    end
    T = precision
    rest = Base.structdiff((; params...), NamedTuple{(:σ,)})
    gauss = init(model; precision, solver, rest...)
    gs = gauss.spec
    fixed = merge(gs.fixed, (σ = _value(σ, T),))
    P = length(freenames(gs))
    nostart = ntuple(_ -> false, P)
    spec = FitSpec{freenames(gs),P,T,typeof(gs.start),typeof(fixed),Nothing}(
        gs.lower, gs.upper, gs.start, 0, fixed, T[], 0, nostart, nostart, nostart, nothing)
    rmodel = method === :ml ? _RicianLikelihood(gauss.model) : _RicianMeanSignal(gauss.model)
    n = length(gauss.y)
    rician = FitCache(rmodel, spec, gauss.opts, similar(gauss.buf), Vector{T}(undef, n),
                      gauss.settings, nothing)
    settings = (; precision, solver, likelihood = :rician, method, params...)
    return RicianFitCache(gauss, rician, method, gauss.model, spec, gauss.y, Vector{T}(undef, n),
                          Vector{T}(undef, n), settings)
end

# One voxel: the first fit, then the second from its values. Returns the lean result, as
# `_fit_voxel` of a FitCache does (dataset.jl stores it in maps; `fit!` makes a FitResult).
function _fit_voxel(c::RicianFitCache, spec::FitSpec{F,P,T}, y, v) where {F,P,T}
    # (above `_FIT_INLINE_MAXP` free parameters called as a function, as dataset.jl does: voxel.jl)
    r1 = P > _FIT_INLINE_MAXP ? (@noinline _fit_voxel(c.gauss, c.gauss.spec, y, v)) :
                                _fit_voxel(c.gauss, c.gauss.spec, y, v)
    (r1.status == InvalidData || r1.status == Failed) && return r1
    rc = c.rician
    copyto!(rc.y, y)
    fixed = map(x -> _pick(x, v, T), spec.fixed)
    σ = fixed.σ
    # Magnitudes are ≥ 0, and σ a noise SD (a map's background, NaN or 0, is not fitted)
    (all(>=(0), rc.y) && isfinite(σ) && σ > 0) || return _invalid(spec)
    θ0 = SVector{P,T}(Tuple(r1.params))
    if c.method === :wmean
        θ, iters, status = θ0, Int32(0), MaxIterations
        for _ in 1:_WMEAN_PASSES
            _rician_weights!(c, θ, fixed, Val(F))
            r = levmar(VoxelProblem{F}(rc.model, rc.y, fixed, c.w), θ, spec.lower, spec.upper,
                       rc.opts, rc.buf)
            θ, iters, status = r.θ, iters + r.iterations, r.status
            status == Failed && break
        end
    else
        r = levmar(VoxelProblem{F}(rc.model, rc.y, fixed), θ0, spec.lower, spec.upper, rc.opts,
                   rc.buf)
        θ, iters, status = r.θ, r.iterations, r.status
    end
    sse = _rician_sse(c, θ, fixed, Val(F))
    return _VoxelFit(relabel(c.model, NamedTuple{F}(Tuple(θ))), sse, r1.iterations + iters, status)
end

# The model's signal at θ (plain numbers) into c.s; `fixed` holds σ, which the model ignores.
@inline function _signal_at!(c::RicianFitCache, θ, fixed, ::Val{F}) where {F}
    signal!(c.s, c.model, _params(Val(F), θ, fixed))
    return c.s
end
# :wmean's weights at θ: 1/Var|y|, the Rician variance at the model's magnitude there.
function _rician_weights!(c::RicianFitCache, θ, fixed, ::Val{F}) where {F}
    s = _signal_at!(c, θ, fixed, Val(F))
    @inbounds for j in eachindex(c.w, s)
        c.w[j] = 1 / ricianvar(abs(s[j]), fixed.σ)
    end
    return c.w
end
# Σ(y − E|y|)² at θ: the data against their mean under the fit, the same for every method.
function _rician_sse(c::RicianFitCache, θ, fixed, ::Val{F}) where {F}
    s = _signal_at!(c, θ, fixed, Val(F))
    T = eltype(c.s)
    sse = zero(T)
    @inbounds for j in eachindex(s, c.rician.y)
        sse += (c.rician.y[j] - ricianmean(abs(s[j]), fixed.σ))^2
    end
    return sse
end

function fit!(c::RicianFitCache, y::AbstractVector; voxel::Integer = 1)
    length(y) == length(c.y) ||
        throw(DimensionMismatch("data have $(length(y)) values, the model has $(length(c.y)) observations"))
    r = _fit_voxel(c, c.spec, y, voxel)
    fixed = _voxel_fixed(c.gauss.spec, voxel)
    # the voxel's σ (a map's value there), so that the result stands alone, as its fixed values do
    settings = merge(c.settings, (σ = _pick(c.spec.fixed.σ, voxel, eltype(c.y)),))
    return FitResult(r.params, fixed, r.sse, r.iterations, r.status, c.model, settings)
end

# -------------------------------------------------------------------------------------------------
# A Rician fit's statistics (statistics.jl dispatches here): σ is the fit's, not estimated.

_isrician(r::FitResult) = get(r.settings, :likelihood, :gaussian) === :rician
_norician(r::FitResult, what) = _isrician(r) && throw(ArgumentError("$what of a fit with " *
    "likelihood = :rician: not yet (the Rician log-likelihood needs the data; on the roadmap). " *
    "Its σ is the fit's, `res.settings.σ`; its standard errors: `stderror(res)`"))

# σ of voxel v, from the settings: a number or a map.
_resultσ(r::FitResult, v, ::Type{T}) where {T} = _pick(_value(r.settings.σ, T), v, T)

function _covariances(c::RicianFitCache, spec::FitSpec{F,P,T}, r, σ, unit) where {F,P,T}
    σ === nothing || throw(ArgumentError("`σ`: a fit with likelihood = :rician has its own " *
        "(`res.settings.σ`), which its standard errors use"))
    nv = _onevoxel(r) ? 1 : length(r.sse)
    N = nobs(r)
    rc = c.rician
    fill!(rc.y, zero(T))                         # the information does not depend on the data
    out = Vector{SMatrix{P,P,T,P * P}}(undef, nv)
    for v in 1:nv
        θ = SVector{P,T}(ntuple(j -> T(_paramvalue(r, F[j], v)), Val(P)))
        fixed = merge(map(x -> _pick(_value(x, T), v, T), r.fixed), (σ = _resultσ(r, v, T),))
        out[v] = _rician_covariance(c, spec, θ, fixed, N)
    end
    return _onevoxel(r) ? out : reshape(out, size(r.sse))
end

# One voxel. `:ml`: the inverse Rician information at θ (its sums, `normal_equations` above,
# already divided by σ²). `:wmean`: (JᵀV⁻¹J)⁻¹, J of the Rician mean, V its variance at θ.
# `:mean`: the sandwich (JᵀJ)⁻¹JᵀVJ(JᵀJ)⁻¹. Parameters on a bound held, as for a Gaussian fit.
function _rician_covariance(c::RicianFitCache, spec::FitSpec{F,P,T}, θ, fixed, N) where {F,P,T}
    nan = fill(T(NaN), SMatrix{P,P,T})
    (all(isfinite, θ) && isfinite(fixed.σ) && fixed.σ > 0) || return nan
    rc = c.rician
    c.method === :ml && return _covariance(rc, spec, θ, one(T), fixed, N)
    s = _signal_at!(c, θ, fixed, Val(F))
    @inbounds for j in eachindex(c.w, s)
        c.w[j] = ricianvar(abs(s[j]), fixed.σ)     # V, the variance of each observation
    end
    if c.method === :wmean
        c.w .= inv.(c.w)
        return _covariance(rc, spec, θ, one(T), fixed, N, c.w)
    end
    # :mean, the sandwich: A = JᵀJ, B = JᵀVJ; A⁻¹BA⁻¹ over the parameters not on a bound
    _, A, _ = normal_equations(VoxelProblem{F}(rc.model, rc.y, fixed), θ, rc.buf)
    _, B, _ = normal_equations(VoxelProblem{F}(rc.model, rc.y, fixed, c.w), θ, rc.buf)
    pinned = map(_near, θ, spec.lower) .| map(_near, θ, spec.upper)
    u = map(p -> ifelse(p, zero(T), one(T)), pinned)
    keep(C) = SMatrix{P,P,T}(ntuple(n -> (i = (n - 1) % P + 1; j = (n - 1) ÷ P + 1;
                                          pinned[i] || pinned[j] ? T(NaN) : C[i, j]), Val(P * P)))
    L, ok = _cholesky((u * u') .* A + Diagonal(one(T) .- u))
    (N >= count(!, pinned) && ok && _determined(A, u)) || return keep(fill(T(Inf), SMatrix{P,P,T}))
    Ai = hcat(ntuple(j -> _chol_solve(L, SVector{P,T}(ntuple(k -> ifelse(k == j, one(T), zero(T)),
                                                              Val(P)))), Val(P))...)
    Bm = (u * u') .* B
    C = Ai * Bm * Ai
    return keep((C + C') ./ 2)
end
