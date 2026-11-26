# =================================================================================================
# Pulsed MT with a spoiled gradient echo (SPGR): two-pool qMT from off-resonance MT pulses
#
#   QMTSPGRYarnykh   Yarnykh's model: the MT pulse as a rectangular pulse of the same energy over
#                    its duration (its power-equivalent amplitude B1e), each pool saturated at a
#                    constant rate during it; the sequence's steady state in closed form
#
# References:
#   Yarnykh, Magn Reson Med 2002;47:929-939             the model (pulsed Z-spectroscopy)
#   Yarnykh & Yuan, NeuroImage 2004;23:409-424          cross-relaxation imaging with it
#   Henkelman et al., Magn Reson Med 1993;29:759-766    the two pools, saturation rates W = π·ω1²·g
#   Morrison & Henkelman, Magn Reson Med 1995;33:475-482  the super-Lorentzian lineshape
# Written from the lab's MATLAB, yarnykh_pulseMT.m (A. Smith, 2013), and checked against it
# (test/unit/models.jl). Differences from it: the exponentials in closed form (twopool.jl), not
# by `eig`; the MT-off image fitted as one more observation, with M0f, not divided into the
# others (its noise would be in every point); B1 and B0 as parameters (maps); R1f from an
# observed R1 as a tie; the excitation's flip angle in degrees (the MATLAB's comment says
# degrees, its code takes radians).
#
# One TR, in the order the sequence plays it, from just before the excitation:
#
#   1. excitation, α = B1·FA: the free pool's Mz → cos α·Mz; the macromolecular pool untouched
#      (its saturation by the short excitation neglected): C = diag(cos α, 1)
#   2. free recovery for tr = TR − tm − ts:                  Er = exp(A·tr)
#   3. the MT pulse, tm long, both pools saturated:           Em = exp((A − W)·tm), toward Mss
#   4. free recovery for ts, until the next excitation:       Es = exp(A·ts)
#
# with A, W and Mss as in twopool.jl, W = π·(γ·B1·B1e)²·diag(gf, gm), gf the free pool's
# Lorentzian (T2f) and gm the macromolecular pool's lineshape (T2m; the super-Lorentzian by
# default), at the MT pulse's offset from the local water resonance, offset − B0. The steady
# state just before the excitation, M = Es·[Em·(Er·C·M + (I − Er)·M0) + (I − Em)·Mss] + (I − Es)·M0
# solved for M (the MATLAB's form, and Yarnykh 2002's):
#
#   M = (I − Es·Em·Er·C)⁻¹·{[Es·Em·(I − Er) + I − Es]·M0 + Es·(I − Em)·Mss}
#
# and the signal is the free pool's Mz excited: S = M0f·sin α·Mzf (M in units of M0f).
# =================================================================================================

# The proton's gyromagnetic ratio, γ/2π = 42.577478518 MHz/T (CODATA 2018), as rad/s per µT
# (the MATLAB's 42.58·2π differs by 5e-5)
const _GAMMA_H = 2π * 42.577478518

"""
    QMTSPGRYarnykh(; offset, B1e, tm, ts, TR, FA, lineshape = superlorentzian)
    QMTSPGRYarnykh(; offset, mtpulse, mtflip, ts, TR, FA, lineshape = superlorentzian)

Pulsed magnetization transfer with a spoiled gradient echo (SPGR), Yarnykh's two-pool model
(Yarnykh, Magn Reson Med 2002; Yarnykh & Yuan, NeuroImage 2004): free water (f) and
macromolecules (m) exchanging magnetization; each TR an excitation, then an off-resonance MT
pulse that saturates both pools (the macromolecular one far more), its energy spread evenly over
its duration `tm`. The steady state in closed form, per image:

    M = (I − Es·Em·Er·C)⁻¹·{[Es·Em·(I − Er) + I − Es]·M0 + Es·(I − Em)·Mss}
    S = M0f·sin(B1·FA)·Mzf

with Er, Em, Es the pools' relaxation and exchange over the TR's three parts (Em with the MT
pulse's saturation, π·ω1²·g per pool), C the excitation (the free pool only) and Mss the steady
state under saturation (Henkelman et al., MRM 1993).

- acquisition, one number for all images or one per image:
  - `offset` (Hz): the MT pulse's frequency offset from water, per image
  - the MT pulse, in either form:
    - `B1e` (µT), its power-equivalent amplitude √(∫B1²dt/tm), and `tm` (s), its duration;
    - or `mtpulse`, its shape as an [`RFPulse`](@ref) (its duration is tm), and `mtflip`
      (degrees), its nominal flip angle per image (the scanner's), which scales the shape
  - `ts` (s): from the MT pulse's end to the excitation; `TR` (s), at least tm + ts
  - `FA` (degrees): the excitation's flip angle
- an image without the MT pulse: `B1e = 0` (or `mtflip = 0`), its offset anything. Include
  one: it fixes M0f, and is fitted as any other image (not divided into the others)
- option: `lineshape`, the macromolecular pool's (`superlorentzian`, the default, for tissue;
  `gaussian`, `lorentzian`, or a function `(T2, Δ) -> g` of your own); the free pool's is
  Lorentzian
- free by default: `PSR` (pool size ratio M0m/M0f), `kmf` (1/s, exchange m→f; kfm = kmf·PSR),
  `T2f` (s), `T2m` (s), `M0f`
- fixed by default: `R1m = 1` 1/s; `R1f` tied to `R1obs`, the R1 a separate (single-pool) R1 map
  measures, so that the two pools' slower recovery rate is R1obs:
  R1f = R1obs − kfm·(R1m − R1obs)/(R1m + kmf − R1obs); `R1obs = 1.0` 1/s: give your map,
  `fit(m, data; R1obs = r1map)`, or fit R1f, `R1f = Free(start = 1.0)` (images at several
  flip angles or TRs determine it); `B1 = 1` (relative: scales the excitation and the MT pulse),
  `B0 = 0` (Hz: the local water resonance's shift, which moves every offset)

```julia
offsets = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
m = QMTSPGRYarnykh(offset = [offsets; offsets; 0], B1e = [fill(4.0, 6); fill(8.0, 6); 0],
                   tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
p = RFPulse(fill(1.0, 64), fill(0.0, 64), 0.012)                       # a 12-ms MT pulse's shape
m = QMTSPGRYarnykh(offset = [offsets; 0], mtpulse = p, mtflip = [fill(600, 6); 0], ts = 0.002, TR = 0.030, FA = 10)
```
"""
struct QMTSPGRYarnykh{A<:AbstractVector,L} <: MRIModel
    offset::A           # the MT pulse's offset from water (Hz)
    B1e::A              # its power-equivalent amplitude (µT): √(∫B1²dt/tm); 0, no MT pulse
    tm::A               # its duration (s)
    ts::A               # from its end to the excitation (s)
    TR::A               # repetition time (s)
    FA::A               # the excitation's flip angle (degrees)
    lineshape::L        # the macromolecular pool's lineshape, (T2, Δ) -> g
end

function QMTSPGRYarnykh(; offset, ts, TR, FA, B1e = nothing, tm = nothing, mtpulse = nothing, mtflip = nothing,
                        lineshape = superlorentzian)
    n = length(offset)
    # one number for every image, or one per image
    per(x, name) = x isa Real ? fill(Float64(x), n) : length(x) == n ? Float64.(collect(x)) :
        throw(DimensionMismatch("QMTSPGRYarnykh: $name has $(length(x)) values for $n offsets (one number, or one per image)"))
    if mtpulse === nothing
        (B1e === nothing || tm === nothing) &&
            throw(ArgumentError("QMTSPGRYarnykh: give the MT pulse as `B1e` and `tm`, or as `mtpulse` and `mtflip`"))
        mtflip === nothing || throw(ArgumentError("QMTSPGRYarnykh: `mtflip` goes with `mtpulse`; with `B1e` and `tm`, leave it out"))
        b, t = per(B1e, :B1e), per(tm, :tm)
    else
        (B1e === nothing && tm === nothing) ||
            throw(ArgumentError("QMTSPGRYarnykh: the MT pulse as `mtpulse` and `mtflip`, or as `B1e` and `tm`, not both"))
        mtflip === nothing && throw(ArgumentError("QMTSPGRYarnykh: `mtpulse` needs `mtflip`, its flip angle per image (degrees)"))
        t = fill(sum(mtpulse.duration), n)
        b = per(mtflip, :mtflip) .* _b1eperdegree(mtpulse)
    end
    m = QMTSPGRYarnykh(Float64.(collect(offset)), b, t, per(ts, :ts), per(TR, :TR), per(FA, :FA), lineshape)
    for j in 1:n
        m.TR[j] >= m.tm[j] + m.ts[j] ||
            throw(ArgumentError("QMTSPGRYarnykh: TR must hold the MT pulse and ts, TR ≥ tm + ts; image $j: " *
                                "TR = $(m.TR[j]), tm + ts = $(m.tm[j] + m.ts[j])"))
        m.tm[j] > 0 && m.ts[j] >= 0 && m.B1e[j] >= 0 ||
            throw(ArgumentError("QMTSPGRYarnykh: image $j needs tm > 0, ts ≥ 0 and B1e ≥ 0"))
    end
    return m
end

# An MT pulse's power-equivalent amplitude per degree of its nominal flip angle (µT/degree): its
# shape, as given, has ∫ω1²dt = Σ αₖ²/τₖ over its duration tm (pulses.jl) and the net flip angle
# |Σ αₖ·e^(iφₖ)| (its rotation on resonance); scaled to flip angle θ, ω1e = (θ/net)·√(∫ω1²dt/tm),
# and B1e = ω1e/γ
function _b1eperdegree(p::RFPulse)
    net = abs(sum(((α, φ),) -> α * cis(deg2rad(φ)), zip(p.flip, p.phase)))
    net > 0 || throw(ArgumentError("QMTSPGRYarnykh: the MT pulse's shape has no net flip angle, so `mtflip` cannot scale it; give B1e and tm"))
    return sqrt(_rfenergy(p) / sum(p.duration)) / _GAMMA_H / net
end

parameters(::Type{<:QMTSPGRYarnykh}) = (
    PSR = Free(start = 0.15, lower = 0, upper = 1),
    kmf = Free(start = 20.0, lower = 0),            # 1/s; kfm = kmf·PSR
    T2f = Free(start = 0.04, lower = 1e-3, upper = 0.5),
    T2m = Free(start = 12e-6, lower = 1e-6, upper = 50e-6),
    M0f = Free(start = :scale, lower = 0),
    R1f = Fixed(_r1ffromobserved),                  # tie: the slower recovery rate is R1obs
    R1m = Fixed(1.0; lower = 0),                    # 1/s, the pulsed-MT convention
    R1obs = Fixed(1.0),                             # 1/s: an R1 map's value (give the map)
    B1 = Fixed(1.0; lower = 0),                     # relative: the excitation and the MT pulse
    B0 = Fixed(0.0),                                # Hz: the local water resonance's shift
)

# R1f from the observed R1: the two pools' slower recovery rate (the smaller eigenvalue of −A,
# what a single-pool R1 map measures) set to R1obs. From det(−A − R1obs·I) = 0,
# (R1f + kfm − R1obs)·(R1m + kmf − R1obs) = kfm·kmf, so
#   R1f = R1obs − kfm·(R1m − R1obs)/(R1m + kmf − R1obs)
# (R1m = R1obs: R1f = R1obs, whatever the exchange)
_r1ffromobserved(p) = p.R1obs - p.kmf * p.PSR * (p.R1m - p.R1obs) / (p.R1m + p.kmf - p.R1obs)

function signal!(out, m::QMTSPGRYarnykh, p)
    (; offset, B1e, tm, ts, TR, FA, lineshape) = m
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = p
    kfm = kmf * PSR                                         # exchange f→m (mass balance)
    A = _relaxexchange(R1f, R1m, kfm, kmf)
    M0 = SVector(one(PSR), PSR)                             # equilibrium, in units of M0f
    r = SVector(R1f, R1m * PSR)                             # R1·M0
    @inbounds for j in eachindex(out, offset, B1e, tm, ts, TR, FA)
        α = B1 * deg2rad(FA[j])
        ω1² = (oftype(B1e[j], _GAMMA_H) * B1 * B1e[j])^2   # the MT pulse's, rad²/s² (γ in the acquisition's precision)
        Δ = offset[j] - B0
        gf, gm = lorentzian(T2f, Δ), lineshape(T2m, Δ)
        # no MT pulse (B1e = 0): no saturation, whatever the lineshape gives at its offset
        Wf = iszero(B1e[j]) ? zero(gf) * ω1² : π * ω1² * gf
        Wm = iszero(B1e[j]) ? zero(gm) * ω1² : π * ω1² * gm
        Er = _expm2(A * (TR[j] - tm[j] - ts[j]))
        Em = _expm2((A - SMatrix{2,2}(Wf, 0, 0, Wm)) * tm[j])
        Es = _expm2(A * ts[j])
        Mss = _saturatedsteadystate(A, Wf, Wm, r)
        C = SMatrix{2,2}(cos(α), 0, 0, 1)                   # the excitation: the free pool only
        EsEm = Es * Em
        rhs = (EsEm * (I - Er) + I - Es) * M0 + Es * ((I - Em) * Mss)
        Mz = (I - EsEm * Er * C) \ rhs                      # just before the excitation
        out[j] = M0f * sin(α) * Mz[1]
    end
    return out
end
