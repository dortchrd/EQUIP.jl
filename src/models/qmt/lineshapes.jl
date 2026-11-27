# =================================================================================================
# Absorption lineshapes: how strongly RF at an offset Δ saturates a pool
#
#   W = π·ω1²·g(Δ)      the saturation rate (1/s) of a pool under RF of amplitude ω1 = γ·B1 (rad/s)
#                       at an offset Δ (Hz) from the pool's resonance (Henkelman et al., Magn Reson
#                       Med 1993)
#
# g(Δ) is in seconds, normalized over the angular offset, ∫ g d(2πΔ) = 1:
#
#   Lorentzian        g = (T2/π) / (1 + (2πΔ·T2)²)             the free pool (liquid)
#   Gaussian          g = (T2/√(2π))·exp(−(2πΔ·T2)²/2)         a solid; finite on resonance
#   super-Lorentzian  g = √(2/π) ∫₀¹ T2/|3u² − 1| · exp(−2(2πΔ·T2/|3u² − 1|)²) du
#                                                              the macromolecular pool of tissue:
#                                                              Gaussians averaged over the
#                                                              orientations u = cos θ of a
#                                                              liquid crystal (Morrison &
#                                                              Henkelman, Magn Reson Med 1995)
#
# The pulse that inverts the free pool in SIR is on resonance, so SIR's Sm needs the lineshape at
# Δ = 0 (pulses.jl): the Gaussian there, as Dortch et al. 2018 (g(0) = T2m/√(2π); Sm = 0.83 for a
# 1-ms block at T2m = 15 µs). The super-Lorentzian diverges on resonance (at the magic angle,
# |3u² − 1| → 0, as ln(1/Δ)); it is for the MT pulses of pulsed MT, kHz off resonance. Its value
# near resonance by a cutoff (Gloor et al., Magn Reson Med 2008: extrapolated inside 1.5 kHz) is
# to come, with the models that need it.
#
# The super-Lorentzian has no closed form. It is T2 times a function of one variable,
#
#   g(Δ) = T2·G(x),   x = 2π·|Δ|·T2,   G(x) = √(2/π) ∫₀¹ exp(−2x²/s²)/s du,   s = |3u² − 1|,
#
# so G is tabulated once (src/tables.jl), as ln(G·e^(x²/2)) against ln x (close to a line at both
# ends: G grows as ln(1/x) at small x and falls as e^(−x²/2)/x² at large x), with its first two
# derivatives, at spacing 0.01 from x = e^(−14) ≈ 8e-7 to e^(3.5) ≈ 33: from 13 Hz at T2 = 10 µs,
# to G ≈ 1e-237 (no saturation). Within 1e-13 of the quadrature, its first derivative 6e-12 and
# its second 1e-9 (relative; measured 2026-11-26; spacing 0.02 gave 5e-12, 2e-10 and 1e-8). Each
# value is then about 45 ns, with derivatives in T2 and Δ (a fitted T2m; a B0 map) from the
# table's chain rule, where the integral by quadrature is about 4 µs. Outside the table, G by
# quadrature: on each side of the magic angle u₀ = 1/√3, in w = ln|u − u₀| (the
# integrand e^(−2x²/s²)/(3(2u₀ ± d)) is then smooth and bounded, where in u it has a peak of
# width x near u₀), by 128-point Gauss–Legendre, from where the integrand has fallen e^(−80)
# below its largest. Within 1e-15 of a 25-digit adaptive quadrature (BigFloat, QuadGK) from
# x = 1e-6 to 20 (measured 2026-11-26); MATLAB's rectangle rule with du = 5e-4 (both endpoints
# counted in full) was 1e-4 to 3e-2 too high, worst at large offsets.
# =================================================================================================

"""
    lorentzian(T2, Δ)
    gaussian(T2, Δ)
    superlorentzian(T2, Δ)

A pool's absorption lineshape g(Δ) (s), at an offset `Δ` (Hz) from its resonance, for its `T2`
(s); normalized over the angular offset (∫ g d(2πΔ) = 1). RF of amplitude ω1 = γ·B1 (rad/s)
saturates the pool at the rate W = π·ω1²·g(Δ) (1/s).

- `lorentzian`: (T2/π)/(1 + (2πΔT2)²), a liquid (the free pool)
- `gaussian`: (T2/√(2π))·exp(−(2πΔT2)²/2), a solid; finite on resonance (SIR's Sm)
- `superlorentzian`: the Gaussian averaged over the orientations of a liquid crystal, the
  macromolecular pool of tissue (Morrison & Henkelman, Magn Reson Med 1995); infinite on
  resonance (Δ = 0), so for MT pulses off resonance

They take dual numbers (a fitted T2m; a B0 map's offsets), and are what a model's `lineshape`
holds: `saturationefficiency(pulse, T2m; lineshape = superlorentzian)`.

```julia
gaussian(15e-6, 0.0)               # 5.98e-6 s: T2m/√(2π)
superlorentzian(12e-6, 2000.0)     # the bound pool 2 kHz from its resonance
```
"""
@inline lorentzian(T2, Δ) = T2 / π / (1 + (2π * Δ * T2)^2)
@inline gaussian(T2, Δ) = T2 / sqrt(2π) * exp(-(2π * Δ * T2)^2 / 2)
@inline superlorentzian(T2, Δ) = T2 * _slG(2π * abs(Δ) * T2)
@doc (@doc lorentzian) gaussian
@doc (@doc lorentzian) superlorentzian

# --- G(x) by quadrature -------------------------------------------------------------------------
const _MAGIC_U = 1 / sqrt(3)                                    # cos of the magic angle, u₀
const _SL_Z, _SL_W = _gausslegendre01(128)
# G(x)·e^(x²/2) = √(2/π)·Σ over the two sides of u₀ of ∫ e^(−2x²(1/s² − 1/4))·d/s dw, d = |u − u₀| = e^w,
# with s = 3d(2u₀ − d) below u₀ (to u = 0, d = u₀, s = 1) and 3d(2u₀ + d) above (to u = 1, s = 2);
# d/s = 1/(3(2u₀ ± d)), written so. Scaled by e^(x²/2), G's fall at large x (s = 2, u = 1), so
# that nothing underflows (G is 1e-152 at x = 26, and the table's second derivative of ln G,
# made through G², had overflowed): the scaled G falls as 1/x² there. The nodes depend on x's
# plain value only (the range), so a dual x carries its partials through the exponentials alone.
# x = 0: +Inf (the divergence).
function _slGscaled(x)
    px = _plain(x)
    T = typeof(exp(-2 * (x / 1.0)^2))
    px == 0 && return T(Inf)
    tot = zero(T)
    for (side, dmax, smax) in ((-1, _MAGIC_U, 1.0), (1, 1 - _MAGIC_U, 2.0))
        # the smallest s that matters: where e^(−2x²/s²) is e^(−80) below its largest, e^(−2x²/smax²)
        slo = 1 / sqrt(40 / px^2 + 1 / smax^2)
        dlo = (slo / 3) / (_MAGIC_U + sqrt(_MAGIC_U^2 + side * slo / 3))     # s = 3d(2u₀ ± d), solved
        a = log(min(dlo, dmax))
        L = log(dmax) - a
        acc = zero(T)
        @inbounds for k in eachindex(_SL_Z)
            d = exp(a + L * _SL_Z[k])
            s = 3d * (2_MAGIC_U + side * d)
            acc += _SL_W[k] * exp(-2 * x^2 * (1 / s^2 - 0.25)) / (3 * (2_MAGIC_U + side * d))
        end
        tot += L * acc
    end
    return sqrt(2 / π) * tot
end
_slGexact(x) = _slGscaled(x) * exp(-x^2 / 2)

# --- G(x) from the table ---------------------------------------------------------------------------
# ln(G·e^(x²/2)) against t = ln x, with its derivatives in t (nested dual numbers through the
# quadrature), at t = −14, −13.99, ..., 3.5 (1751 points, 42 kB): close to a line at both ends
# (ln ln(1/x) at small x, −2t at large)
function _slnode(t)
    f(t) = log(_slGscaled(exp(t)))
    f1(t) = ForwardDiff.derivative(f, t)
    return (f(t), f1(t), ForwardDiff.derivative(f1, t))
end
const _SL_TABLE = _QuinticTable(_slnode, -14.0, 0.01, 1751)
const _SL_TTOP = _tabletop(_SL_TABLE)                           # 3.5
# G from the table inside it, by quadrature outside (x < 8e-7, x ≥ 33, x = 0, NaN), so the table
# is never read outside its range
@inline function _slG(x)
    px = _plain(x)
    if px > 0
        t = log(px)
        _SL_TABLE.x0 <= t < _SL_TTOP && return exp(_interp(_SL_TABLE, log(x)) - x^2 / 2)
    end
    return _slGexact(x)
end
