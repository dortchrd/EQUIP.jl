# =================================================================================================
# Two-pool qMT under continuous-wave (CW) saturation: the steady state, and the family's parameters
#
#   QMTCWHenkelman   CW off-resonance irradiation until the steady state (Henkelman et al., Magn
#                    Reson Med 1993): the free pool's Mz in closed form
#
# The two-pool model under saturation, which the pulsed-MT models approximate (qmt_spgr.jl:
# Yarnykh's, Ramani's): its parameters, their roles and the R1f tie are defined here, and those
# models take them as theirs.
#
# The steady state under constant saturation W = diag(Wf, Wm), (W − A)·Mss = R1·M0 (A twopool.jl's
# relaxation and exchange; Mss in units of M0f), with each pool's saturation rate W = π·ω1²·g(Δ),
# g the free pool's Lorentzian (T2f) or the macromolecular pool's lineshape (T2m). Only the free
# pool's Mz is needed, so the code writes it as Henkelman's closed form, with his R = kmf
# (M0f = 1), M0B = PSR, R_A = R1f, R_B = R1m:
#
#   Mzf/M0f = [R1m·(kmf·PSR/R1f) + Wm + R1m + kmf] / [(kmf·PSR/R1f)·(R1m + Wm) + (1 + Wf/R1f)·(Wm + R1m + kmf)]
#
# (his free-pool term is (ω1/2πΔ)²/(R_A·T2A), the Lorentzian rate far from resonance, where
# 2πΔ·T2f ≫ 1; here the Lorentzian in full, which for the free pool alone gives the Bloch
# equations' own steady state, (1 + (2πΔT2)²)/(1 + (2πΔT2)² + ω1²T1T2), and with exchange, which
# acts on Mz alone, the same). Tested against the 2×2 solve and the Bloch steady state
# (test/unit/models.jl).
# =================================================================================================

# The proton's gyromagnetic ratio, γ/2π = 42.577478518 MHz/T (CODATA 2018), as rad/s per µT
# (the MATLAB's 42.58·2π differs by 5e-5)
const _GAMMA_H = 2π * 42.577478518

"""
    QMTCWHenkelman(; offset, B1cw, lineshape = superlorentzian)

Two-pool qMT under continuous-wave (CW) off-resonance saturation, at the steady state
(Henkelman et al., Magn Reson Med 1993): free water (f) and macromolecules (m) exchanging, both
saturated by the irradiation at the rates π·ω1²·g(Δ) (the free pool's Lorentzian, the
macromolecular pool's `lineshape`), the signal the free pool's Mz:

    S = M0f·Mzf,   (W − A)·Mss = R1·M0   (twopool.jl: the steady state under constant saturation)

equal to Henkelman's closed form (the free pool's Lorentzian in full, not only far from
resonance). The two-pool model under saturation: the pulsed-MT models, [`QMTSPGRYarnykh`](@ref)
and [`QMTSPGRRamani`](@ref), approximate it and take its parameters.

- acquisition: `offset` (Hz), the irradiation's offset from water, per image; `B1cw` (µT), its
  amplitude, one number for all images or one per image (0: no irradiation, the reference image
  that gives M0f)
- option: `lineshape`, the macromolecular pool's (`superlorentzian`, the default, for tissue;
  `gaussian`, `lorentzian`, or a function `(T2, Δ) -> g` of your own); the free pool's is
  Lorentzian
- free by default: `PSR` (pool size ratio M0m/M0f), `kmf` (1/s, exchange m→f; kfm = kmf·PSR),
  `T2f` (s), `T2m` (s), `M0f`
- fixed by default: `R1m = 1` 1/s; `R1f` tied to `R1obs`, the R1 a separate (single-pool) R1 map
  measures, so that the two pools' slower recovery rate is R1obs:
  R1f = R1obs − kfm·(R1m − R1obs)/(R1m + kmf − R1obs); `R1obs = 1.0` 1/s: give your map,
  `fit(m, data; R1obs = r1map)`, or fit R1f, `R1f = Free(start = 1.0)`; `B1 = 1` (relative:
  scales the RF, here the irradiation), `B0 = 0` (Hz: the local water resonance's shift, which
  moves every offset)

```julia
m = QMTCWHenkelman(offset = [1e3, 2e3, 4e3, 8e3, 16e3, 0], B1cw = [fill(2.0, 5); 0])
```
"""
struct QMTCWHenkelman{A<:AbstractVector,L} <: MRIModel
    offset::A           # the irradiation's offset from water (Hz)
    B1cw::A             # its amplitude (µT); 0: none
    lineshape::L        # the macromolecular pool's lineshape, (T2, Δ) -> g
end
function QMTCWHenkelman(; offset, B1cw, lineshape = superlorentzian)
    n = length(offset)
    b = B1cw isa Real ? fill(Float64(B1cw), n) : length(B1cw) == n ? Float64.(collect(B1cw)) :
        throw(DimensionMismatch("QMTCWHenkelman: B1cw has $(length(B1cw)) values for $n offsets (one number, or one per image)"))
    all(>=(0), b) || throw(ArgumentError("QMTCWHenkelman: B1cw ≥ 0 (µT); got $(b)"))
    return QMTCWHenkelman(Float64.(collect(offset)), b, lineshape)
end

# The parameters of the two-pool model under saturation, which the pulsed models take as theirs
# (qmt_spgr.jl: `parameters(::Type{<:QMTSPGRYarnykh}) = parameters(QMTCWHenkelman)`, and Ramani's).
# Julia has no subclasses of a concrete type: a model "inherits" by delegating to this method.
parameters(::Type{<:QMTCWHenkelman}) = (
    PSR = Free(start = 0.15, lower = 0, upper = 1),
    kmf = Free(start = 20.0, lower = 0),            # 1/s; kfm = kmf·PSR
    T2f = Free(start = 0.04, lower = 1e-3, upper = 0.5),
    T2m = Free(start = 12e-6, lower = 1e-6, upper = 50e-6),
    M0f = Free(start = :scale, lower = 0),
    R1f = Fixed(_r1ffromobserved),                  # tie: the slower recovery rate is R1obs
    R1m = Fixed(1.0; lower = 0),                    # 1/s, the pulsed-MT convention
    R1obs = Fixed(1.0),                             # 1/s: an R1 map's value (give the map)
    B1 = Fixed(1.0; lower = 0),                     # relative: the RF (here the irradiation; the MT pulse and the excitation in the pulsed models)
    B0 = Fixed(0.0),                                # Hz: the local water resonance's shift
)

# R1f from the observed R1: the two pools' slower recovery rate (the smaller eigenvalue of −A,
# what a single-pool R1 map measures) set to R1obs. From det(−A − R1obs·I) = 0,
# (R1f + kfm − R1obs)·(R1m + kmf − R1obs) = kfm·kmf, so
#   R1f = R1obs − kfm·(R1m − R1obs)/(R1m + kmf − R1obs)
# (R1m = R1obs: R1f = R1obs, whatever the exchange)
_r1ffromobserved(p) = p.R1obs - p.kmf * p.PSR * (p.R1m - p.R1obs) / (p.R1m + p.kmf - p.R1obs)

# Henkelman et al. 1993's closed form, in EQUIP's notation (his R = kmf with M0f = 1, M0B = PSR,
# R_A = R1f, R_B = R1m, R_RFB = Wm; his free-pool term (ω1/2πΔ)²/(R_A·T2A) as Wf/R1f, the
# Lorentzian rate in full):
#
#   Mzf/M0f = [R1m·q + Wm + R1m + kmf] / [q·(R1m + Wm) + (1 + Wf/R1f)·(Wm + R1m + kmf)],   q = kmf·PSR/R1f
#
# written with its numerator and denominator times R1f, so it divides once (kfm = kmf·PSR = q·R1f):
#
#   Mzf/M0f = [kfm·R1m + R1f·(R1m + kmf + Wm)] / [kfm·(R1m + Wm) + (R1f + Wf)·(R1m + kmf + Wm)]
#
# (each division of dual numbers is the quotient rule for every partial: with three, as Henkelman
# wrote it, a fit's signal was 1.5× the 2×2 solve's, bench/qmt/cw_form.jl). No saturation where
# the amplitude is 0, whatever the lineshape gives at that offset: Mzf/M0f = 1.
function signal!(out, m::QMTCWHenkelman, p)
    (; offset, B1cw, lineshape) = m
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = p
    kfm = kmf * PSR                                         # exchange f→m (Henkelman's R·M0B)
    @inbounds for j in eachindex(out, offset, B1cw)
        ω1² = (oftype(B1cw[j], _GAMMA_H) * B1 * B1cw[j])^2   # the irradiation's, rad²/s²
        Δ = offset[j] - B0
        gf, gm = lorentzian(T2f, Δ), lineshape(T2m, Δ)
        Wf = iszero(ω1²) ? zero(gf) * ω1² : π * ω1² * gf     # each pool's saturation rate
        Wm = iszero(ω1²) ? zero(gm) * ω1² : π * ω1² * gm
        Mzf = (kfm * R1m + R1f * (R1m + kmf + Wm)) / (kfm * (R1m + Wm) + (R1f + Wf) * (R1m + kmf + Wm))
        out[j] = M0f * Mzf
    end
    return out
end
