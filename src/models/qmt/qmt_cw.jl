# =================================================================================================
# Two-pool qMT under continuous-wave (CW) saturation, and Ramani's pulsed-MT approximation by it
#
#   QMTCWHenkelman   CW off-resonance irradiation until the steady state (Henkelman et al., Magn
#                    Reson Med 1993): the free pool's Mz in closed form
#   QMTSPGRRamani    pulsed MT with SPGR, the MT pulse replaced by CW of the same power averaged
#                    over TR (its continuous-wave power equivalent, CWPE) and the excitations
#                    neglected (Ramani et al., Magn Reson Imaging 2002;20:721-731): Henkelman's
#                    steady state at that amplitude
#
# The steady state under constant saturation W = diag(Wf, Wm) (twopool.jl's
# `_saturatedsteadystate`, (W − A)·Mss = R1·M0, Mss in units of M0f), with each pool's saturation
# rate W = π·ω1²·g(Δ), g the free pool's Lorentzian (T2f) or the macromolecular pool's lineshape
# (T2m). Only the free pool's Mz is needed, so the code writes it as Henkelman's closed form, with
# his R = kmf (M0f = 1), M0B = PSR, R_A = R1f, R_B = R1m:
#
#   Mzf/M0f = [R1m·(kmf·PSR/R1f) + Wm + R1m + kmf] / [(kmf·PSR/R1f)·(R1m + Wm) + (1 + Wf/R1f)·(Wm + R1m + kmf)]
#
# (his free-pool term is (ω1/2πΔ)²/(R_A·T2A), the Lorentzian rate far from resonance, where
# 2πΔ·T2f ≫ 1; here the Lorentzian in full, which for the free pool alone gives the Bloch
# equations' own steady state, (1 + (2πΔT2)²)/(1 + (2πΔT2)² + ω1²T1T2), and with exchange, which
# acts on Mz alone, the same). Tested against the 2×2 solve and the Bloch steady state
# (test/unit/models.jl).
#
# Ramani's CWPE amplitude: ω1,CWPE² = (1/TR)·∫ω1²dt over the MT pulse = (tm/TR)·ω1e², with ω1e the
# pulse's power-equivalent amplitude over its own duration (QMTSPGRYarnykh's B1e). Its
# approximations against Yarnykh's model (the same acquisition, qmt_spgr.jl): the saturation
# spread over the whole TR (exact only for TR short against the exchange and relaxation), and the
# excitations' effect on the free pool left out (small for small flip angles; the MT-off image
# then gives M0f). So its images are all at one flip angle and TR: VFA images in the same fit
# (as QMTSPGRYarnykh takes) are not for it.
# =================================================================================================

"""
    QMTCWHenkelman(; offset, B1cw, lineshape = superlorentzian)

Two-pool qMT under continuous-wave (CW) off-resonance saturation, at the steady state
(Henkelman et al., Magn Reson Med 1993): free water (f) and macromolecules (m) exchanging, both
saturated by the irradiation at the rates π·ω1²·g(Δ) (the free pool's Lorentzian, the
macromolecular pool's `lineshape`), the signal the free pool's Mz:

    S = M0f·Mzf,   (W − A)·Mss = R1·M0   (twopool.jl: the steady state under constant saturation)

equal to Henkelman's closed form (the free pool's Lorentzian in full, not only far from
resonance).

- acquisition: `offset` (Hz), the irradiation's offset from water, per image; `B1cw` (µT), its
  amplitude, one number for all images or one per image (0: no irradiation, the reference image
  that gives M0f)
- option: `lineshape`, as `QMTSPGRYarnykh`'s
- parameters: `QMTSPGRYarnykh`'s (free: PSR, kmf, T2f, T2m, M0f; fixed: R1f tied to R1obs,
  R1m = 1, R1obs = 1.0, B1 = 1 (scales the irradiation), B0 = 0 Hz)

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

"""
    QMTSPGRRamani(; offset, B1e, tm, ts, TR, FA, lineshape = superlorentzian)
    QMTSPGRRamani(; offset, mtpulse, mtflip, ts, TR, FA, lineshape = superlorentzian)

Pulsed MT with a spoiled gradient echo, Ramani et al.'s approximation (Magn Reson Imaging 2002):
the MT pulse replaced by continuous-wave saturation of the same power averaged over TR, its
continuous-wave power equivalent ω1,CWPE² = (tm/TR)·(γ·B1·B1e)², and the excitations neglected,
so the signal is [`QMTCWHenkelman`](@ref)'s steady state at that amplitude: S = M0f·Mzf. The
acquisition as [`QMTSPGRYarnykh`](@ref)'s (the same keywords; `ts` and `FA` taken and unused:
the approximation leaves them out), so the two compare on one protocol. Its images are at one
flip angle and TR, with an image without the MT pulse (`B1e = 0`) for M0f; VFA images in the
same fit, as `QMTSPGRYarnykh` takes them, are not for it. The parameters: `QMTSPGRYarnykh`'s.
No exponentials: a 2×2 solve an image.
"""
struct QMTSPGRRamani{A<:AbstractVector,L} <: MRIModel
    offset::A           # the MT pulse's offset from water (Hz)
    B1e::A              # its power-equivalent amplitude over its duration (µT); 0, no MT pulse
    tm::A               # its duration (s)
    ts::A               # from its end to the excitation (s): not used (Ramani's approximation)
    TR::A               # repetition time (s)
    FA::A               # the excitation's flip angle (degrees): not used
    lineshape::L        # the macromolecular pool's lineshape, (T2, Δ) -> g
end
QMTSPGRRamani(; kw...) = QMTSPGRRamani(_pulsedspgr(QMTSPGRRamani; kw...)...)

# The same parameters, roles and R1f tie as QMTSPGRYarnykh's (B1 scales the irradiation)
parameters(::Type{<:QMTCWHenkelman}) = parameters(QMTSPGRYarnykh)
parameters(::Type{<:QMTSPGRRamani}) = parameters(QMTSPGRYarnykh)

# The free pool's Mz (in units of M0f) at the steady state under saturation at ω1² (rad²/s²), the
# offset Δ (Hz) from the local water resonance: Henkelman et al. 1993's closed form, in EQUIP's
# notation (his R = kmf with M0f = 1, M0B = PSR, R_A = R1f, R_B = R1m, R_RFB = Wm; his free-pool
# term (ω1/2πΔ)²/(R_A·T2A) as Wf/R1f, the Lorentzian rate in full),
#
#   Mzf/M0f = [R1m·q + Wm + R1m + kmf] / [q·(R1m + Wm) + (1 + Wf/R1f)·(Wm + R1m + kmf)],   q = kmf·PSR/R1f
#
# its numerator and denominator times R1f, so one division (with kfm = kmf·PSR = q·R1f):
#
#   Mzf/M0f = [kfm·R1m + R1f·(R1m + kmf + Wm)] / [kfm·(R1m + Wm) + (R1f + Wf)·(R1m + kmf + Wm)]
#
# (each division of dual numbers is the quotient rule for every partial: as written by Henkelman,
# with three, a fit's signal was 1.5× the 2×2 solve's; measured, bench/qmt/cw_form.jl). No
# saturation where ω1² is 0 (whatever the lineshape gives at its offset): Mzf/M0f = 1.
@inline function _cwfree(PSR, kmf, R1f, R1m, ω1², Δ, T2f, T2m, lineshape)
    gf, gm = lorentzian(T2f, Δ), lineshape(T2m, Δ)
    Wf = iszero(ω1²) ? zero(gf) * ω1² : π * ω1² * gf             # each pool's saturation rate
    Wm = iszero(ω1²) ? zero(gm) * ω1² : π * ω1² * gm
    kfm = kmf * PSR                                               # exchange f→m (Henkelman's R·M0B)
    return (kfm * R1m + R1f * (R1m + kmf + Wm)) / (kfm * (R1m + Wm) + (R1f + Wf) * (R1m + kmf + Wm))
end

function signal!(out, m::QMTCWHenkelman, p)
    (; offset, B1cw, lineshape) = m
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = p
    @inbounds for j in eachindex(out, offset, B1cw)
        ω1² = (oftype(B1cw[j], _GAMMA_H) * B1 * B1cw[j])^2
        out[j] = M0f * _cwfree(PSR, kmf, R1f, R1m, ω1², offset[j] - B0, T2f, T2m, lineshape)
    end
    return out
end

function signal!(out, m::QMTSPGRRamani, p)
    (; offset, B1e, tm, TR, lineshape) = m
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = p
    @inbounds for j in eachindex(out, offset, B1e, tm, TR)
        ω1² = (tm[j] / TR[j]) * (oftype(B1e[j], _GAMMA_H) * B1 * B1e[j])^2   # the CWPE: the power over TR
        out[j] = M0f * _cwfree(PSR, kmf, R1f, R1m, ω1², offset[j] - B0, T2f, T2m, lineshape)
    end
    return out
end
