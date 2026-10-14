# Built-in MRI signal models.
#
# Units are whatever you pass in: time constants come out in the units of `x` (ms in, ms out),
# diffusivities in the inverse units of b. Initial guesses use closed-form linearizations
# (log-linear or DESPOT1-style regressions) computed in a single pass without allocation, so they
# also run inside GPU kernels. Time constants have a small positive lower bound (1e-6) so that
# t/T stays finite.

const _TMIN = 1e-6

# Weighted simple linear regression accumulated one point at a time (immutable, GPU-friendly).
struct _LinReg{T}
    sw::T
    sx::T
    sy::T
    sxx::T
    sxy::T
end
_LinReg(::Type{T}) where {T} = _LinReg{T}(zero(T), zero(T), zero(T), zero(T), zero(T))
@inline _push(a::_LinReg{T}, x, y, w) where {T} =
    _LinReg{T}(a.sw + w, a.sx + w * x, a.sy + w * y, a.sxx + w * x * x, a.sxy + w * x * y)
@inline function _slope_intercept(a::_LinReg{T}) where {T}
    den = a.sw * a.sxx - a.sx * a.sx
    slope = (a.sw * a.sxy - a.sx * a.sy) / den
    return slope, (a.sy - slope * a.sx) / a.sw
end

@inline _ftype(x, y) = float(promote_type(eltype(x), eltype(y)))
@inline _ok(v) = isfinite(v) && v > 0

@inline function _maxabs(y, ::Type{T}) where {T}
    s = zero(T)
    @inbounds for i in 1:length(y)
        s = max(s, abs(T(y[i])))
    end
    return s
end

# Log-linear fit of log(y) = a + b·x with weights y² (corrects for the log's noise amplification).
@inline function _loglinear(x, y, ::Type{T}) where {T}
    a = _LinReg(T)
    @inbounds for i in 1:length(y)
        yi = T(y[i])
        if yi > 0
            a = _push(a, T(x[i]), log(yi), yi * yi)
        end
    end
    return _slope_intercept(a)
end

# ----------------------------------------------------------------------------------- MonoExp

"""
    MonoExp()

Mono-exponential decay `S(t) = S0·exp(-t/T2)`; parameters `(:S0, :T2)`. Use for T2, T2*, or any
decay time constant (`x` = echo times).
"""
struct MonoExp <: AbstractModel{2} end
@inline (::MonoExp)(t, θ) = θ[1] * exp(-t / θ[2])
@inline function value_gradient(::MonoExp, t, θ::SVector{2,T}) where {T}
    e = exp(-T(t) / θ[2])
    f = θ[1] * e
    return f, SVector(e, f * T(t) / (θ[2] * θ[2]))
end
paramnames(::MonoExp) = (:S0, :T2)
lower_bounds(::MonoExp) = (0.0, _TMIN)
function initial_guess(::MonoExp, t, y)
    T = _ftype(t, y)
    slope, icpt = _loglinear(t, y, T)
    T2 = -1 / slope
    S0 = exp(icpt)
    tmax = zero(T)
    @inbounds for i in 1:length(t)
        tmax = max(tmax, abs(T(t[i])))
    end
    return SVector(_ok(S0) ? S0 : _maxabs(y, T), _ok(T2) ? T2 : max(tmax, one(T)))
end

"""
    MonoExpOffset()

`S(t) = S0·exp(-t/T2) + C`, parameters `(:S0, :T2, :C)`; `C ≥ 0` absorbs a noise floor or a very
long component.
"""
struct MonoExpOffset <: AbstractModel{3} end
@inline (::MonoExpOffset)(t, θ) = θ[1] * exp(-t / θ[2]) + θ[3]
paramnames(::MonoExpOffset) = (:S0, :T2, :C)
lower_bounds(::MonoExpOffset) = (0.0, _TMIN, 0.0)
function initial_guess(::MonoExpOffset, t, y)
    θ = initial_guess(MonoExp(), t, y)
    return SVector(θ[1], θ[2], zero(θ[1]))
end

"""
    BiExp()

Bi-exponential decay `S(t) = A1·exp(-t/T2a) + A2·exp(-t/T2b)`; parameters
`(:A1, :T2a, :A2, :T2b)`. The initial guess peels the slow component from the later half of the
curve, then fits the fast component to what is left of the early half; it is a heuristic, so
pass `init` when you have priors (e.g. myelin water fractions).
"""
struct BiExp <: AbstractModel{4} end
@inline (::BiExp)(t, θ) = θ[1] * exp(-t / θ[2]) + θ[3] * exp(-t / θ[4])
paramnames(::BiExp) = (:A1, :T2a, :A2, :T2b)
lower_bounds(::BiExp) = (0.0, _TMIN, 0.0, _TMIN)
function initial_guess(::BiExp, t, y)
    T = _ftype(t, y)
    n = length(y)
    mono = initial_guess(MonoExp(), t, y)
    h = n ÷ 2
    # Slow component from the later echoes.
    a = _LinReg(T)
    @inbounds for i in h+1:n
        yi = T(y[i])
        yi > 0 && (a = _push(a, T(t[i]), log(yi), yi * yi))
    end
    sl, ic = _slope_intercept(a)
    T2b, A2 = -1 / sl, exp(ic)
    (_ok(T2b) && _ok(A2)) || return SVector(mono[1] / 2, mono[2] / 3, mono[1] / 2, 2 * mono[2])
    # Fast component from the residual of the early echoes.
    b = _LinReg(T)
    @inbounds for i in 1:h
        ri = T(y[i]) - A2 * exp(-T(t[i]) / T2b)
        ri > 0 && (b = _push(b, T(t[i]), log(ri), ri * ri))
    end
    sl, ic = _slope_intercept(b)
    T2a, A1 = -1 / sl, exp(ic)
    if !(_ok(T2a) && _ok(A1)) || T2a >= T2b
        T2a, A1 = T2b / 5, max(mono[1] - A2, mono[1] / 10)
    end
    return SVector(A1, T2a, A2, T2b)
end

# ----------------------------------------------------------------------------------- Diffusion

"""
    ADC()

`S(b) = S0·exp(-b·D)`; parameters `(:S0, :D)`.
"""
struct ADC <: AbstractModel{2} end
@inline (::ADC)(b, θ) = θ[1] * exp(-b * θ[2])
@inline function value_gradient(::ADC, b, θ::SVector{2,T}) where {T}
    e = exp(-T(b) * θ[2])
    return θ[1] * e, SVector(e, -T(b) * θ[1] * e)
end
paramnames(::ADC) = (:S0, :D)
lower_bounds(::ADC) = (0.0, 0.0)
function initial_guess(::ADC, b, y)
    T = _ftype(b, y)
    slope, icpt = _loglinear(b, y, T)
    S0 = exp(icpt)
    return SVector(_ok(S0) ? S0 : _maxabs(y, T), _ok(-slope) ? -slope : zero(T))
end

"""
    Kurtosis()

Diffusion kurtosis along one direction (or powder-averaged):
`S(b) = S0·exp(-b·D + b²·D²·K/6)`; parameters `(:S0, :D, :K)`.
"""
struct Kurtosis <: AbstractModel{3} end
@inline (::Kurtosis)(b, θ) = θ[1] * exp(-b * θ[2] + b * b * θ[2] * θ[2] * θ[3] / 6)
paramnames(::Kurtosis) = (:S0, :D, :K)
lower_bounds(::Kurtosis) = (0.0, 0.0, 0.0)
upper_bounds(::Kurtosis) = (Inf, Inf, 10.0)
function initial_guess(::Kurtosis, b, y)
    T = _ftype(b, y)
    # Weighted quadratic regression of log(y) on b: c0 + c1·b + c2·b².
    A = zero(SMatrix{3,3,T})
    r = zero(SVector{3,T})
    @inbounds for i in 1:length(y)
        yi = T(y[i])
        yi > 0 || continue
        bi = T(b[i])
        v = SVector(one(T), bi, bi * bi)
        A += (yi * yi) * (v * v')
        r += (yi * yi * log(yi)) * v
    end
    L, ok = _cholesky(A)
    if ok
        c = _chol_solve(L, r)
        S0, D = exp(c[1]), -c[2]
        K = _ok(D) ? 6 * c[3] / (D * D) : T(NaN)
        if _ok(S0) && _ok(D) && isfinite(K)
            return SVector(S0, D, clamp(K, zero(T), T(3)))
        end
    end
    θ = initial_guess(ADC(), b, y)
    return SVector(θ[1], θ[2], one(T))
end

"""
    IVIM(; bthresh=200)

Intravoxel incoherent motion: `S(b) = S0·(f·exp(-b·Dp) + (1-f)·exp(-b·D))`; parameters
`(:S0, :f, :Dp, :D)` with `Dp` the pseudo-diffusion coefficient. The initial guess is the usual
segmented fit: `D` from b ≥ `bthresh`, then `f` and `Dp` from the low-b residual.
"""
Base.@kwdef struct IVIM <: AbstractModel{4}
    bthresh::Float64 = 200.0
end
@inline (::IVIM)(b, θ) = θ[1] * (θ[2] * exp(-b * θ[3]) + (1 - θ[2]) * exp(-b * θ[4]))
paramnames(::IVIM) = (:S0, :f, :Dp, :D)
lower_bounds(::IVIM) = (0.0, 0.0, 0.0, 0.0)
upper_bounds(::IVIM) = (Inf, 1.0, Inf, Inf)
function initial_guess(m::IVIM, b, y)
    T = _ftype(b, y)
    bt = T(m.bthresh)
    # S0: signal at the smallest b.
    bmin, S0 = T(Inf), zero(T)
    @inbounds for i in 1:length(y)
        bi = T(b[i])
        if bi < bmin
            bmin, S0 = bi, T(y[i])
        end
    end
    _ok(S0) || (S0 = _maxabs(y, T))
    hi = _LinReg(T)
    @inbounds for i in 1:length(y)
        yi, bi = T(y[i]), T(b[i])
        (bi >= bt && yi > 0) && (hi = _push(hi, bi, log(yi), yi * yi))
    end
    sl, ic = _slope_intercept(hi)
    D = _ok(-sl) ? -sl : T(1e-3)
    S0D = exp(ic)
    f = _ok(S0D) ? clamp(1 - S0D / S0, T(0.01), T(0.5)) : T(0.1)
    lo = _LinReg(T)
    @inbounds for i in 1:length(y)
        bi = T(b[i])
        bi < bt || continue
        ri = T(y[i]) - S0 * (1 - f) * exp(-bi * D)
        ri > 0 && (lo = _push(lo, bi, log(ri), ri * ri))
    end
    sl, _ = _slope_intercept(lo)
    Dp = (_ok(-sl) && -sl > D) ? -sl : 10 * D
    return SVector(S0, f, Dp, D)
end

# ----------------------------------------------------------------------------------- Relaxometry

"""
    InversionRecovery(; magnitude=true)

`S(TI) = |S0·(1 - B·exp(-TI/T1))|` (or without `| |` for phase-corrected real data); parameters
`(:S0, :B, :T1)` where `B ≈ 2` for a perfect inversion (the RD-NLS model of Barral et al. 2010).
The initial T1 comes from the null point, `TI_null/ln 2`.
"""
Base.@kwdef struct InversionRecovery <: AbstractModel{3}
    magnitude::Bool = true
end
@inline function (m::InversionRecovery)(ti, θ)
    s = θ[1] * (1 - θ[2] * exp(-ti / θ[3]))
    return m.magnitude ? abs(s) : s
end
paramnames(::InversionRecovery) = (:S0, :B, :T1)
lower_bounds(::InversionRecovery) = (0.0, 0.0, _TMIN)
upper_bounds(::InversionRecovery) = (Inf, 3.0, Inf)
function initial_guess(::InversionRecovery, ti, y)
    T = _ftype(ti, y)
    ymin, tnull, S0, tlast = T(Inf), zero(T), zero(T), T(-Inf)
    @inbounds for i in 1:length(y)
        yi, t = abs(T(y[i])), T(ti[i])
        if yi < ymin
            ymin, tnull = yi, t
        end
        if t > tlast           # S0 from the longest TI
            tlast, S0 = t, yi
        end
    end
    S0 = max(S0, _maxabs(y, T))
    T1 = tnull > 0 ? tnull / log(T(2)) : max(tlast, one(T)) / 2
    return SVector(S0, T(2), T1)
end

"""
    SaturationRecovery()

`S(t) = S0·(1 - exp(-t/T1))`; parameters `(:S0, :T1)`.
"""
struct SaturationRecovery <: AbstractModel{2} end
@inline (::SaturationRecovery)(t, θ) = θ[1] * (1 - exp(-t / θ[2]))
paramnames(::SaturationRecovery) = (:S0, :T1)
lower_bounds(::SaturationRecovery) = (0.0, _TMIN)
function initial_guess(::SaturationRecovery, t, y)
    T = _ftype(t, y)
    S0 = T(1.05) * _maxabs(y, T)
    # log(S0 - y) = log(S0) - t/T1
    a = _LinReg(T)
    @inbounds for i in 1:length(y)
        r = S0 - T(y[i])
        r > 0 && (a = _push(a, T(t[i]), log(r), r * r))
    end
    sl, _ = _slope_intercept(a)
    T1 = -1 / sl
    tmax = zero(T)
    @inbounds for i in 1:length(t)
        tmax = max(tmax, T(t[i]))
    end
    return SVector(S0, _ok(T1) ? T1 : max(tmax, one(T)) / 2)
end

"""
    SPGR(; TR)

Spoiled gradient echo / variable flip angle T1 mapping:
`S(α) = M0·sin(α)·(1 - E1)/(1 - cos(α)·E1)`, `E1 = exp(-TR/T1)`; parameters `(:M0, :T1)`.
`x` holds flip angles **in degrees**; `T1` is in the units of `TR`. Pass a relative B1 map as a
covariate (`covariates = b1map`) to use the actual flip angle `B1·α`. The initial guess is the
DESPOT1 linear regression.
"""
Base.@kwdef struct SPGR <: AbstractModel{2}
    TR::Float64
end
@inline function _spgr(TR, α, θ)
    E1 = exp(-TR / θ[2])
    s, c = sincos(α)
    return θ[1] * s * (1 - E1) / (1 - c * E1)
end
@inline (m::SPGR)(fa, θ) = _spgr(oftype(fa, m.TR), deg2rad(fa), θ)
@inline (m::SPGR)(fa, θ, c) = _spgr(oftype(fa, m.TR), deg2rad(fa) * c[1], θ)
paramnames(::SPGR) = (:M0, :T1)
lower_bounds(::SPGR) = (0.0, _TMIN)
initial_guess(m::SPGR, fa, y) = _spgr_init(m, fa, y, one(_ftype(fa, y)))
initial_guess(m::SPGR, fa, y, c) = _spgr_init(m, fa, y, _ftype(fa, y)(c[1]))
function _spgr_init(m::SPGR, fa, y, b1::T) where {T}
    # y/sin α = E1·(y/tan α) + M0·(1 - E1)
    a = _LinReg(T)
    @inbounds for i in 1:length(y)
        s, c = sincos(deg2rad(T(fa[i])) * b1)
        yi = T(y[i])
        s > 0 && (a = _push(a, yi * c / s, yi / s, one(T)))
    end
    E1, icpt = _slope_intercept(a)
    TR = T(m.TR)
    if 0 < E1 < 1 && icpt > 0
        return SVector(icpt / (1 - E1), -TR / log(E1))
    end
    return SVector(10 * _maxabs(y, T), 50 * TR)
end

# ----------------------------------------------------------------------------------- Registry

"""
    BUILTIN_MODELS

Built-in models by name, as accepted by [`parse_model`](@ref) (CLI, Python and MATLAB).
"""
const BUILTIN_MODELS = Dict{String,Any}(
    "MonoExp" => MonoExp,
    "MonoExpOffset" => MonoExpOffset,
    "BiExp" => BiExp,
    "ADC" => ADC,
    "Kurtosis" => Kurtosis,
    "IVIM" => IVIM,
    "InversionRecovery" => InversionRecovery,
    "SaturationRecovery" => SaturationRecovery,
    "SPGR" => SPGR,
)

_literal(x::Union{Number,Bool,String}) = x
_literal(x::QuoteNode) = _literal(x.value)
_literal(x::Symbol) = x === :true ? true : x === :false ? false : x === :Inf ? Inf :
                      throw(ArgumentError("unsupported value `$x` in model spec"))
function _literal(x::Expr)
    x.head === :call && x.args[1] === :- && length(x.args) == 2 && return -_literal(x.args[2])
    throw(ArgumentError("unsupported value `$x` in model spec (literals only)"))
end

"""
    parse_model(spec::AbstractString) -> AbstractModel

Build a built-in model from a string such as `"MonoExp"`, `"SPGR(TR=15)"` or
`"IVIM(bthresh=250)"`. Only literal arguments are accepted, nothing is evaluated.
"""
function parse_model(spec::AbstractString)
    ex = Meta.parse(strip(spec))
    name, args, kwargs = if ex isa Symbol
        String(ex), Any[], Pair{Symbol,Any}[]
    elseif ex isa Expr && ex.head === :call && ex.args[1] isa Symbol
        pos, kw = Any[], Pair{Symbol,Any}[]
        for a in ex.args[2:end]
            if a isa Expr && a.head === :kw
                push!(kw, a.args[1] => _literal(a.args[2]))
            elseif a isa Expr && a.head === :parameters
                for p in a.args
                    push!(kw, p.args[1] => _literal(p.args[2]))
                end
            else
                push!(pos, _literal(a))
            end
        end
        String(ex.args[1]), pos, kw
    else
        throw(ArgumentError("cannot parse model spec \"$spec\""))
    end
    haskey(BUILTIN_MODELS, name) ||
        throw(ArgumentError("unknown model \"$name\"; available: $(join(sort!(collect(keys(BUILTIN_MODELS))), ", "))"))
    return BUILTIN_MODELS[name](args...; kwargs...)
end
