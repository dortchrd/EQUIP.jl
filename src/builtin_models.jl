# Built-in MRI signal models.
#
# Units pass through: time constants come out in the units of the protocol's times, rates in
# their inverse. Starting values for `Estimate(:auto)` parameters come from closed-form
# linearizations (log-linear or DESPOT1-style regressions) computed in one pass without
# allocation. Time constants have a small positive lower bound (1e-6) so that t/T stays finite.

const _TMIN = 1e-6

# Weighted simple linear regression accumulated one point at a time.
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
    @inbounds for i in eachindex(y)
        s = max(s, abs(T(y[i])))
    end
    return s
end

@inline function _maxval(x, ::Type{T}) where {T}
    s = zero(T)
    @inbounds for i in eachindex(x)
        s = max(s, abs(T(x[i])))
    end
    return s
end

# Log-linear fit of log(y) = a + b·x with weights y² (corrects for the log's noise amplification).
@inline function _loglinear(x, y, ::Type{T}) where {T}
    a = _LinReg(T)
    @inbounds for i in eachindex(y)
        yi = T(y[i])
        if yi > 0
            a = _push(a, T(x[i]), log(yi), yi * yi)
        end
    end
    return _slope_intercept(a)
end

function _monoexp_guess(t, y)
    T = _ftype(t, y)
    slope, icpt = _loglinear(t, y, T)
    T2, S0 = -1 / slope, exp(icpt)
    return (S0 = _ok(S0) ? S0 : _maxabs(y, T), T2 = _ok(T2) ? T2 : max(_maxval(t, T), one(T)))
end

# ------------------------------------------------------------------------------- Relaxation

"""
    MonoExp()

Mono-exponential decay `S = S0·exp(-TE/T2)`. Protocol: `TE`. Parameters `S0`, `T2`. Use for
T2, T2*, or any decay time constant.
"""
struct MonoExp <: AbstractModel end
parameters(::MonoExp) = (S0 = Estimate(:auto; lower=0), T2 = Estimate(:auto; lower=_TMIN))
protocol(::MonoExp) = (:TE,)
function signal!(out, ::MonoExp, acq, p)
    (; S0, T2) = p
    TE = acq.TE
    @inbounds for i in eachindex(out)
        out[i] = S0 * exp(-TE[i] / T2)
    end
    return out
end
initial_guess(::MonoExp, acq, y, p) = _monoexp_guess(acq.TE, y)

"""
    MonoExpOffset()

`S = S0·exp(-TE/T2) + C`. Protocol: `TE`. Parameters `S0`, `T2`, `C` (a noise floor or very
long component, `C ≥ 0`).
"""
struct MonoExpOffset <: AbstractModel end
parameters(::MonoExpOffset) =
    (S0 = Estimate(:auto; lower=0), T2 = Estimate(:auto; lower=_TMIN), C = Estimate(0.0; lower=0))
protocol(::MonoExpOffset) = (:TE,)
function signal!(out, ::MonoExpOffset, acq, p)
    (; S0, T2, C) = p
    TE = acq.TE
    @inbounds for i in eachindex(out)
        out[i] = S0 * exp(-TE[i] / T2) + C
    end
    return out
end
initial_guess(::MonoExpOffset, acq, y, p) = _monoexp_guess(acq.TE, y)

"""
    BiExp()

`S = A1·exp(-TE/T2a) + A2·exp(-TE/T2b)`. Protocol: `TE`. Parameters `A1`, `T2a`, `A2`, `T2b`.
The starting values peel the slow component from the later echoes, then fit the fast one to
the remainder of the early echoes; a heuristic, so give `start` values when you have priors.
"""
struct BiExp <: AbstractModel end
parameters(::BiExp) = (A1 = Estimate(:auto; lower=0), T2a = Estimate(:auto; lower=_TMIN),
                       A2 = Estimate(:auto; lower=0), T2b = Estimate(:auto; lower=_TMIN))
protocol(::BiExp) = (:TE,)
function signal!(out, ::BiExp, acq, p)
    (; A1, T2a, A2, T2b) = p
    TE = acq.TE
    @inbounds for i in eachindex(out)
        out[i] = A1 * exp(-TE[i] / T2a) + A2 * exp(-TE[i] / T2b)
    end
    return out
end
function initial_guess(::BiExp, acq, y, p)
    t = acq.TE
    T = _ftype(t, y)
    n = length(y)
    mono = _monoexp_guess(t, y)
    h = n ÷ 2
    a = _LinReg(T)
    @inbounds for i in h+1:n
        yi = T(y[i])
        yi > 0 && (a = _push(a, T(t[i]), log(yi), yi * yi))
    end
    sl, ic = _slope_intercept(a)
    T2b, A2 = -1 / sl, exp(ic)
    if !(_ok(T2b) && _ok(A2))
        return (A1 = mono.S0 / 2, T2a = mono.T2 / 3, A2 = mono.S0 / 2, T2b = 2 * mono.T2)
    end
    b = _LinReg(T)
    @inbounds for i in 1:h
        ri = T(y[i]) - A2 * exp(-T(t[i]) / T2b)
        ri > 0 && (b = _push(b, T(t[i]), log(ri), ri * ri))
    end
    sl, ic = _slope_intercept(b)
    T2a, A1 = -1 / sl, exp(ic)
    if !(_ok(T2a) && _ok(A1)) || T2a >= T2b
        T2a, A1 = T2b / 5, max(mono.S0 - A2, mono.S0 / 10)
    end
    return (A1 = A1, T2a = T2a, A2 = A2, T2b = T2b)
end

"""
    InversionRecovery(; magnitude = true)

`S = |S0·(1 - B·exp(-TI/T1))|` (without `| |` for phase-corrected data). Protocol: `TI`.
Parameters `S0`, `B` (≈ 2 for a perfect inversion), `T1`. The starting T1 comes from the null
point, `TI_null/ln 2`.
"""
Base.@kwdef struct InversionRecovery <: AbstractModel
    magnitude::Bool = true
end
parameters(::InversionRecovery) =
    (S0 = Estimate(:auto; lower=0), B = Estimate(2.0; lower=0, upper=3), T1 = Estimate(:auto; lower=_TMIN))
protocol(::InversionRecovery) = (:TI,)
function signal!(out, m::InversionRecovery, acq, p)
    (; S0, B, T1) = p
    TI = acq.TI
    @inbounds for i in eachindex(out)
        s = S0 * (1 - B * exp(-TI[i] / T1))
        out[i] = m.magnitude ? abs(s) : s
    end
    return out
end
function initial_guess(::InversionRecovery, acq, y, p)
    ti = acq.TI
    T = _ftype(ti, y)
    ymin, tnull, S0, tlast = T(Inf), zero(T), zero(T), T(-Inf)
    @inbounds for i in eachindex(y)
        yi, t = abs(T(y[i])), T(ti[i])
        if yi < ymin
            ymin, tnull = yi, t
        end
        if t > tlast
            tlast, S0 = t, yi
        end
    end
    S0 = max(S0, _maxabs(y, T))
    T1 = tnull > 0 ? tnull / log(T(2)) : max(tlast, one(T)) / 2
    return (S0 = S0, T1 = T1)
end

"""
    SaturationRecovery()

`S = S0·(1 - exp(-t/T1))`. Protocol: `t` (time after saturation). Parameters `S0`, `T1`.
"""
struct SaturationRecovery <: AbstractModel end
parameters(::SaturationRecovery) = (S0 = Estimate(:auto; lower=0), T1 = Estimate(:auto; lower=_TMIN))
protocol(::SaturationRecovery) = (:t,)
function signal!(out, ::SaturationRecovery, acq, p)
    (; S0, T1) = p
    t = acq.t
    @inbounds for i in eachindex(out)
        out[i] = S0 * (1 - exp(-t[i] / T1))
    end
    return out
end
function initial_guess(::SaturationRecovery, acq, y, p)
    t = acq.t
    T = _ftype(t, y)
    S0 = T(1.05) * _maxabs(y, T)
    a = _LinReg(T)                                 # log(S0 - y) = log(S0) - t/T1
    @inbounds for i in eachindex(y)
        r = S0 - T(y[i])
        r > 0 && (a = _push(a, T(t[i]), log(r), r * r))
    end
    sl, _ = _slope_intercept(a)
    T1 = -1 / sl
    return (S0 = S0, T1 = _ok(T1) ? T1 : max(_maxval(t, T), one(T)) / 2)
end

"""
    SPGR()

Spoiled gradient echo / variable flip angle T1 mapping:
`S = M0·sin(α)·(1 - E1)/(1 - cos(α)·E1)`, `E1 = exp(-TR/T1)`, `α = B1·FA`.
Protocol: `FA` (flip angles in degrees) and `TR`. Parameters `M0`, `T1` (in the units of TR)
and `B1` (relative flip-angle scale, fixed at 1 by default; pass a map with
`fixed = (B1 = b1map,)`). Starting values from the DESPOT1 linear regression.
"""
struct SPGR <: AbstractModel end
parameters(::SPGR) = (M0 = Estimate(:auto; lower=0), T1 = Estimate(:auto; lower=_TMIN), B1 = Fixed(1.0))
protocol(::SPGR) = (:FA, :TR)
function signal!(out, ::SPGR, acq, p)
    (; M0, T1, B1) = p
    FA = acq.FA
    E1 = exp(-acq.TR / T1)
    @inbounds for i in eachindex(out)
        s, c = sincos(B1 * deg2rad(FA[i]))
        out[i] = M0 * s * (1 - E1) / (1 - c * E1)
    end
    return out
end
function initial_guess(::SPGR, acq, y, p)
    fa = acq.FA
    T = _ftype(fa, y)
    b1 = haskey(p, :B1) ? T(p.B1) : one(T)
    a = _LinReg(T)                                 # y/sin α = E1·(y/tan α) + M0·(1 - E1)
    @inbounds for i in eachindex(y)
        s, c = sincos(deg2rad(T(fa[i])) * b1)
        yi = T(y[i])
        s > 0 && (a = _push(a, yi * c / s, yi / s, one(T)))
    end
    E1, icpt = _slope_intercept(a)
    TR = T(acq.TR)
    if 0 < E1 < 1 && icpt > 0
        return (M0 = icpt / (1 - E1), T1 = -TR / log(E1))
    end
    return (M0 = 10 * _maxabs(y, T), T1 = 50 * TR)
end

"""
    SIRFSE(; magnitude = true)

Selective inversion recovery with a fast spin-echo readout (two-pool qMT; Gochberg & Gore). The
readout leaves both pools saturated; they recover over `TD`, the free pool is selectively
inverted, and the signal is read at `TI` (the `M_zf` component of Eq. 1 in Dortch et al.,
Magn Reson Med 2018;80:1824). Protocol: `TI`, `TD` (seconds; either may be a vector or one
constant). Parameters: `pmf` (pool size ratio M0m/M0f, "PSR"), `R1f`, `Sf` (free-pool
inversion efficiency), `M0f` estimated; `kmf` fixed at 12.5 s⁻¹ (the constrained model for
human brain at 3T in that paper; estimate it with `free = (:kmf,)`), `R1m` tied to `R1f`, and
`Sm` fixed at 0.83 (1-ms block inversion pulse; compute it for your pulse).
"""
Base.@kwdef struct SIRFSE <: AbstractModel
    magnitude::Bool = true
end
parameters(::SIRFSE) = (
    pmf = Estimate(0.1; lower=0, upper=1),
    R1f = Estimate(1.0; lower=0.05, upper=10),
    Sf = Estimate(-0.95; lower=-1.2, upper=0),
    M0f = Estimate(:auto; lower=0),
    kmf = Fixed(12.5),
    R1m = Fixed(p -> p.R1f),
    Sm = Fixed(0.83),
)
protocol(::SIRFSE) = (:TI, :TD)
function signal!(M, m::SIRFSE, acq, p)
    (; pmf, R1f, Sf, M0f, kmf, R1m, Sm) = p
    kfm = kmf * pmf                                   # mass balance
    ΔR1 = sqrt((R1f - R1m + kfm - kmf)^2 + 4 * kfm * kmf)
    R1⁺ = (R1f + R1m + kfm + kmf + ΔR1) / 2
    R1⁻ = R1⁺ - ΔR1
    invΔR1 = 1 / ΔR1
    # Recovery from saturation over TD: amplitudes of the two exponentials
    bf_td⁺ = -(R1f - R1⁻) * invΔR1
    bf_td⁻ = (R1f - R1⁺) * invΔR1
    bm_td⁺ = -(R1m - R1⁻) * invΔR1
    bm_td⁻ = (R1m - R1⁺) * invΔR1
    @inbounds for j in eachindex(M)
        (; TI, TD) = obs(acq, j)
        E_td⁺, E_td⁻ = exp(-R1⁺ * TD), exp(-R1⁻ * TD)
        E_ti⁺, E_ti⁻ = exp(-R1⁺ * TI), exp(-R1⁻ * TI)
        Mf_td = bf_td⁺ * E_td⁺ + bf_td⁻ * E_td⁻ + 1
        Mm_td = bm_td⁺ * E_td⁺ + bm_td⁻ * E_td⁻ + 1
        bf1 = Sf * Mf_td - 1
        bf2 = (Sf * Mf_td - Sm * Mm_td) * kfm
        bf_ti⁺ = (bf1 * (R1f - R1⁻) + bf2) * invΔR1
        bf_ti⁻ = -(bf1 * (R1f - R1⁺) + bf2) * invΔR1
        s = (bf_ti⁺ * E_ti⁺ + bf_ti⁻ * E_ti⁻ + 1) * M0f
        M[j] = m.magnitude ? abs(s) : s
    end
    return M
end
initial_guess(::SIRFSE, acq, y, p) = (M0f = _maxabs(y, _ftype(acq.TI, y)),)

# ------------------------------------------------------------------------------- Diffusion

"""
    ADC()

`S = S0·exp(-b·D)`. Protocol: `b`. Parameters `S0`, `D`.
"""
struct ADC <: AbstractModel end
parameters(::ADC) = (S0 = Estimate(:auto; lower=0), D = Estimate(:auto; lower=0))
protocol(::ADC) = (:b,)
function signal!(out, ::ADC, acq, p)
    (; S0, D) = p
    b = acq.b
    @inbounds for i in eachindex(out)
        out[i] = S0 * exp(-b[i] * D)
    end
    return out
end
function initial_guess(::ADC, acq, y, p)
    T = _ftype(acq.b, y)
    slope, icpt = _loglinear(acq.b, y, T)
    S0 = exp(icpt)
    return (S0 = _ok(S0) ? S0 : _maxabs(y, T), D = _ok(-slope) ? -slope : zero(T))
end

"""
    Kurtosis()

Diffusion kurtosis along one direction (or powder-averaged):
`S = S0·exp(-b·D + b²·D²·K/6)`. Protocol: `b`. Parameters `S0`, `D`, `K`.
"""
struct Kurtosis <: AbstractModel end
parameters(::Kurtosis) =
    (S0 = Estimate(:auto; lower=0), D = Estimate(:auto; lower=0), K = Estimate(:auto; lower=0, upper=10))
protocol(::Kurtosis) = (:b,)
function signal!(out, ::Kurtosis, acq, p)
    (; S0, D, K) = p
    b = acq.b
    @inbounds for i in eachindex(out)
        out[i] = S0 * exp(-b[i] * D + b[i]^2 * D^2 * K / 6)
    end
    return out
end
function initial_guess(::Kurtosis, acq, y, p)
    b = acq.b
    T = _ftype(b, y)
    A = zero(SMatrix{3,3,T})                       # weighted quadratic fit of log(y) on b
    r = zero(SVector{3,T})
    @inbounds for i in eachindex(y)
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
        _ok(S0) && _ok(D) && isfinite(K) && return (S0 = S0, D = D, K = clamp(K, zero(T), T(3)))
    end
    g = initial_guess(ADC(), acq, y, p)
    return (S0 = g.S0, D = g.D, K = one(T))
end

"""
    IVIM(; bthresh = 200)

Intravoxel incoherent motion: `S = S0·(f·exp(-b·Dp) + (1-f)·exp(-b·D))`. Protocol: `b`.
Parameters `S0`, `f`, `Dp` (pseudo-diffusion), `D`. Starting values from the usual segmented
fit: `D` from b ≥ `bthresh`, then `f` and `Dp` from the low-b remainder.
"""
Base.@kwdef struct IVIM <: AbstractModel
    bthresh::Float64 = 200.0
end
parameters(::IVIM) = (S0 = Estimate(:auto; lower=0), f = Estimate(:auto; lower=0, upper=1),
                      Dp = Estimate(:auto; lower=0), D = Estimate(:auto; lower=0))
protocol(::IVIM) = (:b,)
function signal!(out, ::IVIM, acq, p)
    (; S0, f, Dp, D) = p
    b = acq.b
    @inbounds for i in eachindex(out)
        out[i] = S0 * (f * exp(-b[i] * Dp) + (1 - f) * exp(-b[i] * D))
    end
    return out
end
function initial_guess(m::IVIM, acq, y, p)
    b = acq.b
    T = _ftype(b, y)
    bt = T(m.bthresh)
    bmin, S0 = T(Inf), zero(T)
    @inbounds for i in eachindex(y)
        bi = T(b[i])
        if bi < bmin
            bmin, S0 = bi, T(y[i])
        end
    end
    _ok(S0) || (S0 = _maxabs(y, T))
    hi = _LinReg(T)
    @inbounds for i in eachindex(y)
        yi, bi = T(y[i]), T(b[i])
        (bi >= bt && yi > 0) && (hi = _push(hi, bi, log(yi), yi * yi))
    end
    sl, ic = _slope_intercept(hi)
    D = _ok(-sl) ? -sl : T(1e-3)
    S0D = exp(ic)
    f = _ok(S0D) ? clamp(1 - S0D / S0, T(0.01), T(0.5)) : T(0.1)
    lo = _LinReg(T)
    @inbounds for i in eachindex(y)
        bi = T(b[i])
        bi < bt || continue
        ri = T(y[i]) - S0 * (1 - f) * exp(-bi * D)
        ri > 0 && (lo = _push(lo, bi, log(ri), ri * ri))
    end
    sl, _ = _slope_intercept(lo)
    Dp = (_ok(-sl) && -sl > D) ? -sl : 10 * D
    return (S0 = S0, f = f, Dp = Dp, D = D)
end

# ------------------------------------------------------------------------------- Registry

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
    "SIRFSE" => SIRFSE,
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

Build a built-in model from a string such as `"MonoExp"`, `"SIRFSE(magnitude=false)"` or
`"IVIM(bthresh=250)"`. Only literal arguments are accepted; nothing is evaluated.
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
                for q in a.args
                    push!(kw, q.args[1] => _literal(q.args[2]))
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
