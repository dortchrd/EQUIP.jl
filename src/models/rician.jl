# =================================================================================================
# Rician noise, the parts a fit uses: what taking the magnitude of noisy complex data does to its
# mean, its variance and the information it carries about the signal, and a model fitted to the
# mean
#
#   ricianmean(A, σ)            E|y|: above A at low SNR (the noise floor, σ√(π/2) at A = 0)
#   ricianvar(A, σ)             Var|y|
#   ricianinformation(A, σ)     the Fisher information about A in one magnitude value, R(A/σ)/σ²
#   RicianMean(m; σ, ...)       a model whose signal is the mean of the magnitude data
#
# (The Rician bound, `ricianbound`, is in src/sim/rician.jl; crlb of magnitude data is crlb's
# `noise = :rician` and `likelihood = :rician`, src/errors/crlb.jl.
# These moved here from there on 2026-11-09, tier 2 step 1, so that the fit can use them: this
# file is included before src/fit/.)
#
# A is the true signal amplitude, σ the noise SD of each of the real and imaginary channels, y the
# measured magnitude (Gudbjartsson & Patz, Magn Reson Med 1995;34:910). With averages: averaged
# as complex data, the magnitude of the mean is Rician with σ/√NEX (the floor falls); averaged as
# magnitudes (a spherical mean over directions is one), the mean of NEX Rician values has the
# same mean E|y| at σ (the floor stays) and the variance Var|y|/NEX.
#
# Speed (docs/design.md, the tier 2 plan, step 1). Everything here rests on two Bessel functions,
# scaled: e^(−x)·I₀(x) and e^(−x)·I₁(x) (`_i01e`). SpecialFunctions computes them with the AMOS
# Fortran library, 100–460 ns a call, and on dual numbers (a fit's derivatives) differentiates
# through more calls: measured, a fit of the Rician mean was 7–105× slower than the Gaussian fit,
# nearly all of it in these calls. So here:
#   - they are Chebyshev series, fitted when EQUIP is precompiled to SpecialFunctions' `besselix`
#     itself (no constants typed in: the method of Cephes and SciPy's i0e and i1e), equal to it
#     within 3e-14 (relative, x from 0 to 10⁶; tested);
#   - on dual numbers they follow their exact derivatives, d/dx[e^(−x)I₀] = e^(−x)(I₁ − I₀) and
#     d/dx[e^(−x)I₁] = e^(−x)(I₀ − I₁ − I₁/x) (from I₀′ = I₁, I₁′ = I₀ − I₁/x), computed from the
#     same two values, so a dual number costs one evaluation, at any nesting (crlb's Hessians);
#   - R(s), the information's factor, is tabulated once from its quadrature.
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# The scaled modified Bessel functions e^(−x)·I₀(x), e^(−x)·I₁(x), for x ≥ 0

# Chebyshev coefficients of f on [−1, 1] from its values at n Chebyshev points: interpolation,
# which for a smooth f converges as fast as the best polynomial of that degree (Trefethen,
# Approximation Theory and Approximation Practice, SIAM 2013, ch. 4). Run while EQUIP is
# precompiled; the coefficients are then constants of the package.
function _chebcoeffs(f, n)
    θ = [π * (j + 0.5) / n for j in 0:n-1]
    fx = f.(cos.(θ))
    c = [2 / n * sum(fx[j+1] * cos(k * θ[j+1]) for j in 0:n-1) for k in 0:n-1]
    c[1] /= 2
    return Tuple(c)
end

# Two Chebyshev series at one point t ∈ [−1, 1], by Clenshaw's recurrence, in one loop.
@inline function _clenshaw2(a::NTuple{N,Float64}, b::NTuple{N,Float64}, t::Float64) where {N}
    a1, a2, b1, b2 = 0.0, 0.0, 0.0, 0.0
    t2 = 2t
    @inbounds for k in N:-1:2
        a1, a2 = muladd(t2, a1, -a2) + a[k], a1
        b1, b2 = muladd(t2, b1, -b2) + b[k], b1
    end
    return muladd(t, a1, -a2) + a[1], muladd(t, b1, -b2) + b[1]
end

# x ∈ [0, 8] as t = x/4 − 1, with I₁ as e^(−x)·I₁(x)/x (so that it is accurate relative to its
# size near 0, where I₁ ≈ x/2); x ∈ [8, ∞) as t = 16/x − 1, with √x·e^(−x)·I(x), which tends to
# 1/√(2π). 26 and 24 terms: more gave no better agreement (measured, 2026-11-09).
const _I0E_LO = _chebcoeffs(t -> besselix(0, 4(t + 1)), 26)
const _I1E_LO = _chebcoeffs(t -> (x = 4(t + 1); besselix(1, x) / x), 26)
const _I0E_HI = _chebcoeffs(t -> (x = 16 / (t + 1); sqrt(x) * besselix(0, x)), 24)
const _I1E_HI = _chebcoeffs(t -> (x = 16 / (t + 1); sqrt(x) * besselix(1, x)), 24)

# (e^(−x)·I₀(x), e^(−x)·I₁(x)) for x ≥ 0; as SpecialFunctions' (besselix(0, x), besselix(1, x)).
# Computed in Float64 and returned in x's precision, as SpecialFunctions does for Float32, so a
# Float32 value is the series' rounded. (Float32 fits of the Rician mean differ from the old
# besselix path by at most 0.007 of their SD, no more than Float32 fits differ from Float64 ones,
# 0.0085: measured 2026-11-09, five models, 500 voxels each, SNR 10.)
function _i01e(x::Real)
    x < 0 && throw(DomainError(x, "the scaled Bessel functions are used here for x ≥ 0 only"))
    T = typeof(float(x))
    z = Float64(x)
    if z <= 8
        p, q = _clenshaw2(_I0E_LO, _I1E_LO, z / 4 - 1)
        return T(p), T(z * q)
    end
    p, q = _clenshaw2(_I0E_HI, _I1E_HI, 16 / z - 1)
    s = 1 / sqrt(z)
    return T(s * p), T(s * q)
end

# On a dual number (ForwardDiff): the values at its value, and the exact derivatives above times
# its partials. Its value may itself be a dual number (nested, for second derivatives): the same
# method then runs on it. I₁(x)/x is 1/2 at x = 0.
function _i01e(x::Dual{Tag}) where {Tag}
    v = value(x)
    i0, i1 = _i01e(v)
    r = iszero(ForwardDiff.value(v)) ? one(i1) / 2 : i1 / v
    p = partials(x)
    return Dual{Tag}(i0, (i1 - i0) * p), Dual{Tag}(i1, (i0 - i1 - r) * p)
end

# -------------------------------------------------------------------------------------------------
"""
    ricianmean(A, σ) -> E|y|

The mean of the magnitude of a complex signal of amplitude `A` with Gaussian noise of SD `σ` in
each channel (Rician; Gudbjartsson & Patz, Magn Reson Med 1995):

    E|y| = σ·√(π/2)·L₁/₂(−A²/2σ²) = σ·√(π/2)·[(1 + ν)·I₀(ν/2) + ν·I₁(ν/2)]·e^(−ν/2),  ν = A²/2σ²

(L₁/₂ a Laguerre function, I₀ and I₁ modified Bessel functions, computed scaled: no overflow at
high SNR). σ√(π/2) at A = 0 (the noise floor), A + σ²/(2A) at high SNR (above A/σ = 1000, its
expansion A + σ²/(2A) + σ⁴/(8A³), exact there to rounding: the Bessel functions' argument would
be out of their range). Differentiable in A and σ (dual numbers, with the Bessel functions'
exact derivatives), so it can be part of a signal model ([`RicianMean`](@ref)).
"""
function ricianmean(A, σ)
    abs(A) > 1000σ && return abs(A) + σ^2 / (2abs(A)) + σ^4 / (8abs(A)^3)
    x = A^2 / (4σ^2)                                    # ν/2
    i0, i1 = _i01e(x)
    return σ * sqrt(oftype(float(x), π) / 2) * ((1 + 2x) * i0 + 2x * i1)
end

"""
    ricianvar(A, σ) -> Var|y|

The variance of the magnitude of a complex signal of amplitude `A` with Gaussian noise of SD `σ`
in each channel: A² + 2σ² − (E|y|)² ([`ricianmean`](@ref)). (2 − π/2)·σ² at A = 0, σ² at high
SNR (above A/σ = 1000, its expansion σ²·(1 − σ²/2A²): the difference loses precision there).
"""
function ricianvar(A, σ)
    s = abs(A) / σ
    s > 1000 && return σ^2 * (1 - 1 / (2s^2))
    return A^2 + 2σ^2 - ricianmean(A, σ)^2
end

"""
    ricianinformation(A, σ) -> R(A/σ)/σ²

The Fisher information about the amplitude `A` in one Rician magnitude value: R(s)/σ², s = A/σ,
with R(s) = E[u²·(I₁(us)/I₀(us))²] − s² over u, the magnitude in units of σ (the factor R(S, σ) of
Devan et al., Magn Reson Imaging 2020;74:56, Appendix, in units of σ; Karlsen et al., Magn Reson
Med 1999). R is 0 at A = 0 (no information), about 0.52 at s = 1, 0.94 at s = 3, and
1 − 1/(2s²) − 1/(4s⁴) at high SNR (Gaussian: 1/σ²). From a table of R made once by quadrature
over u (`_ricianR`: s from 0 to 40 in steps of 0.01, cubic interpolation, within 1e-8 of the
quadrature, 2e-9 above s = 0.5), the expansion above s = 40.
"""
ricianinformation(A::Real, σ::Real) = _ricianRtable(abs(A) / σ) / σ^2

# R(s) by quadrature over u (Simpson, 1201 points over s ± 12, where the density lies); the
# expansion above s = 40. The reference the table is made from (and tested against).
function _ricianR(s::Real)
    s <= 0 && return 0.0
    s > 40 && return 1 - 1 / (2s^2) - 1 / (4s^4)
    n = 1201
    lo, hi = max(0.0, s - 12), s + 12
    h = (hi - lo) / (n - 1)
    total = 0.0
    for i in 1:n
        u = lo + (i - 1) * h
        z = u * s
        i0, i1 = _i01e(z)
        r = z == 0 ? 0.0 : i1 / i0                             # I₁/I₀
        pdf = u * i0 * exp(-(u - s)^2 / 2)                     # the Rician density (σ = 1)
        w = i == 1 || i == n ? 1 : (iseven(i) ? 4 : 2)
        total += w * u^2 * r^2 * pdf
    end
    return total * h / 3 - s^2
end

# R(s) tabulated at s = 0, 0.01, ..., 40 (made while EQUIP is precompiled, about 0.3 s), and read
# by cubic (four-point Lagrange) interpolation: fast enough for a solver or a design's kernel,
# where the quadrature (1201 points) is not.
const _RSTEP = 0.01
const _RTABLE = [_ricianR(k * _RSTEP) for k in 0:4000]

function _ricianRtable(s::Real)
    s <= 0 && return zero(float(s))
    s > 40 && return 1 - 1 / (2s^2) - 1 / (4s^4)
    u = s / _RSTEP
    k = clamp(floor(Int, u), 1, length(_RTABLE) - 3)      # nodes k-1, k, k+1, k+2 (from 0)
    t = u - k                                             # where s lies, from node k
    @inbounds y0, y1, y2, y3 = _RTABLE[k], _RTABLE[k+1], _RTABLE[k+2], _RTABLE[k+3]
    return (-t * (t - 1) * (t - 2) / 6) * y0 + ((t + 1) * (t - 1) * (t - 2) / 2) * y1 +
           (-(t + 1) * t * (t - 2) / 2) * y2 + ((t + 1) * t * (t - 1) / 6) * y3
end

# ρ = I₁(z)/I₀(z) and its first two derivatives, ρ′ = 1 − ρ/z − ρ², ρ″ = −ρ′/z + ρ/z² − 2ρρ′,
# accurate where those closed forms cancel (ρ″ is formed from terms 1/z² times larger than
# itself near 0, and z³ times larger at high z: at z = 1e4 the closed form's ρ″ was off 0.4%):
#
#   z < 0.1   the Taylor series ρ = Σₖ cₖz²ᵏ⁺¹, c = 1/2, −1/16, 1/96, −11/6144, 19/61440
#   z > 50    the asymptotic series ρ = Σₖ qₖtᵏ, t = 1/z, q = 1, −1/2, −1/8, −1/8, −25/128,
#             −13/32, −1073/1024, −103/32, −375733/32768, −23797/512
#   between   the closed forms, from `_i01e`
#
# Both series are I₁'s divided by I₀'s (their power series, and their asymptotic ones, Abramowitz
# and Stegun 9.6.10 and 9.7.1), divided in exact rationals; ρ′ and ρ″ are their term-by-term
# derivatives. Checked against I₀ and I₁ summed in 320-bit arithmetic from z = 0.001 to 1e5: ρ
# within 4e-15, ρ′ and ρ″ within 3e-10 (relative), at worst just below the switches.
const _RHO_TAYLOR = (1 / 2, -1 / 16, 1 / 96, -11 / 6144, 19 / 61440)
const _RHO_ASYMPTOTIC = (1.0, -1 / 2, -1 / 8, -1 / 8, -25 / 128, -13 / 32, -1073 / 1024, -103 / 32,
                         -375733 / 32768, -23797 / 512)
function _besselratio(z::Float64)
    ρ = ρ1 = ρ2 = 0.0
    if z < 0.1
        for (k, c) in enumerate(_RHO_TAYLOR)
            n = 2k - 1                                   # the power of z
            ρ += c * z^n
            ρ1 += n * c * z^(n - 1)
            n > 1 && (ρ2 += n * (n - 1) * c * z^(n - 2))   # (the z term's is 0: none at z = 0)
        end
        return ρ, ρ1, ρ2
    elseif z > 50
        t = 1 / z
        for (k, q) in enumerate(_RHO_ASYMPTOTIC)
            n = k - 1                                    # the power of t; d/dz = −t² d/dt
            ρ += q * t^n
            ρ1 -= n * q * t^(n + 1)
            ρ2 += n * (n + 1) * q * t^(n + 2)
        end
        return ρ, ρ1, ρ2
    end
    i0, i1 = _i01e(z)
    ρ = i1 / i0
    ρ1 = 1 - ρ / z - ρ^2
    return ρ, ρ1, -ρ1 / z + ρ / z^2 - 2ρ * ρ1
end

# The Rician log-likelihood's first three derivatives in the amplitude, and their expectations
# over the data: what the maximum-likelihood fit's second-order bias needs (crlb's `secondorder =
# true` with `likelihood = :rician`; src/errors/crlb.jl, `_mlnoisebias`). For one magnitude value
# y, ℓ(A) = −A²/2σ² + log I₀(yA/σ²) + (terms without A). In units of σ (x = y/σ, a = A/σ the
# amplitude the fit evaluates, as = A★/σ the data's true one) and with ρ = I₁/I₀ at z = x·a:
#
#   u = σ·∂ℓ/∂A    = −a + x·ρ(z)
#   v = σ²·∂²ℓ/∂A² = −1 + x²·ρ′(z),     ρ′ = 1 − ρ/z − ρ²            (from I₀′ = I₁, I₁′ = I₀ − I₁/z)
#   w = σ³·∂³ℓ/∂A³ = x³·ρ″(z),          ρ″ = −ρ′/z + ρ/z² − 2ρρ′
#
# Returned over x ~ Rice(as, 1): (u = E[u], v = E[v], w = E[w], uu = Var[u], vu = Cov[v, u]); in
# A's units divide by σ, σ², σ³, σ² and σ³. With a = as (the model right) Bartlett's identities
# hold: E[u] = 0 and Var[u] = −E[v] = R(a), the information (`_ricianR`). Gaussian noise would
# give (0, −1, 0, 1, 0), the limit at high SNR. By quadrature, as R (Simpson, 1201 points over
# as ± 12, where the density lies, normalized by its own sum); ρ, ρ′ and ρ″ from `_besselratio`.
function _ricianmoments(a::Real, as::Real)
    n = 1201
    lo, hi = max(0.0, as - 12), as + 12
    h = (hi - lo) / (n - 1)
    s0 = su = sv = sw = suu = svu = 0.0
    for i in 1:n
        x = lo + (i - 1) * h
        i0s, _ = _i01e(x * as)
        pdf = x * i0s * exp(-(x - as)^2 / 2)                   # Rice(as, 1): x e^(−(x²+as²)/2) I₀(x·as)
        ρ, ρ1, ρ2 = _besselratio(x * a)
        u, v, w = -a + x * ρ, -1 + x^2 * ρ1, x^3 * ρ2
        c = (i == 1 || i == n ? 1 : (iseven(i) ? 4 : 2)) * pdf
        s0 += c; su += c * u; sv += c * v; sw += c * w; suu += c * u^2; svu += c * v * u
    end
    Eu, Ev = su / s0, sv / s0
    return (u = Eu, v = Ev, w = sw / s0, uu = suu / s0 - Eu^2, vu = svu / s0 - Ev * Eu)
end

# -------------------------------------------------------------------------------------------------
"""
    RicianMean(m; σ, averages = nothing, averaging = :complex)

A model whose signal is the mean of the magnitude data that model `m` gives with Rician noise:
[`ricianmean`](@ref) of `m`'s signal (its magnitude), observation by observation. The same
parameters and acquisition as `m`, and its derived quantities. `σ` is the noise SD of one image,
a number or one per observation. With `averages`: complex averaging (the scanner's, usually)
divides σ by √NEX, so the floor falls; `:magnitude` (a spherical mean over directions is one)
leaves the mean, so the floor, as it is.

As the data a fit is judged against, it gives the bias of a least-squares fit of magnitude data
(the noise floor's): what `crlb(m, θ; snr, noise = :rician)` uses. Fitted itself, it is least
squares on the expected magnitude, a correction of the floor for any model (with σ known):
`fit(m, data; likelihood = :rician, σ, method = :mean)` does that.
"""
struct RicianMean{M<:MRIModel,S<:AbstractVector} <: MRIModel
    model::M
    σ::S                                # each observation's σ, after complex averaging
end

function RicianMean(m::MRIModel; σ, averages = nothing, averaging::Symbol = :complex)
    averaging in (:complex, :magnitude) || throw(ArgumentError("RicianMean: `averaging` is " *
        ":complex (the floor falls with the averages) or :magnitude (it stays), got :$averaging"))
    n = nobs(m)
    σv = σ isa Real ? fill(Float64(σ), n) : Float64.(collect(σ))
    length(σv) == n || throw(DimensionMismatch("RicianMean: $(length(σv)) σ for $n observations"))
    if averages !== nothing && averaging === :complex
        nex = averages isa Real ? fill(averages, n) : collect(averages)
        length(nex) == n || throw(DimensionMismatch("RicianMean: $(length(nex)) averages for " *
                                                    "$n observations"))
        σv = σv ./ sqrt.(nex)
    end
    return RicianMean(m, σv)
end

# The wrapped model's parameters, observations and hooks (its derived quantities too, so that a
# fit of the Rician mean of DTI gives FA and MD); the signal through the Rician mean.
parameters(w::RicianMean) = parameters(w.model)
nobs(w::RicianMean) = nobs(w.model)
datastart(w::RicianMean, y, p) = datastart(w.model, y, p)
relabel(w::RicianMean, θ) = relabel(w.model, θ)
derived(w::RicianMean, p) = derived(w.model, p)
_with_precision(w::RicianMean, ::Type{T}) where {T} =
    RicianMean(_with_precision(w.model, T), Vector{T}(w.σ))
function signal!(out, w::RicianMean, p)
    signal!(out, w.model, p)
    @inbounds for j in eachindex(out, w.σ)
        out[j] = ricianmean(abs(out[j]), w.σ[j])
    end
    return out
end
