# =================================================================================================
# Pulsed MT with a spoiled gradient echo (SPGR): two-pool qMT from off-resonance MT pulses
#
#   QMTSPGRYarnykh   Yarnykh's model: the MT pulse as a rectangular pulse of the same energy over
#                    its duration (its power-equivalent amplitude B1e), each pool saturated at a
#                    constant rate during it; the sequence's steady state in closed form
#   QMTSPGRRamani    Ramani's: the MT pulse replaced by continuous-wave saturation of the same
#                    power averaged over TR (its continuous-wave power equivalent, CWPE), the
#                    excitations neglected (Ramani et al., Magn Reson Imaging 2002;20:721-731):
#                    Henkelman's CW steady state (qmt_cw.jl) at that amplitude
#
# Both approximate the two-pool model under saturation, QMTCWHenkelman (qmt_cw.jl), and take its
# parameters. Their acquisition is one (a shared constructor, `_pulsedspgr`), so they compare on
# one protocol.
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
- option: `lineshape`, the macromolecular pool's, as [`QMTCWHenkelman`](@ref)'s
  (`superlorentzian` by default)
- the parameters: `QMTCWHenkelman`'s, the two-pool model under saturation. Free by default:
  `PSR` (pool size ratio M0m/M0f), `kmf` (1/s, exchange m→f; kfm = kmf·PSR), `T2f` (s), `T2m`
  (s), `M0f`. Fixed by default: `R1m = 1` 1/s; `R1f` tied to `R1obs`, the R1 a separate
  (single-pool) R1 map measures (`fit(m, data; R1obs = r1map)`), or fitted,
  `R1f = Free(start = 1.0)` (images at several flip angles or TRs determine it); `B1 = 1`
  (relative: scales the excitation and the MT pulse), `B0 = 0` (Hz: the local water
  resonance's shift, which moves every offset)

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

QMTSPGRYarnykh(; kw...) = QMTSPGRYarnykh(_pulsedspgr(QMTSPGRYarnykh; kw...)...)

# A pulsed-MT SPGR acquisition from its keywords, checked, as the fields in order (offset, B1e,
# tm, ts, TR, FA, lineshape): the MT pulse as B1e and tm, or as its shape and flip angle; each
# field one number for every image or one per image. Shared by QMTSPGRYarnykh and QMTSPGRRamani
# (below), `M` naming the model in the messages.
function _pulsedspgr(M; offset, ts, TR, FA, B1e = nothing, tm = nothing, mtpulse = nothing, mtflip = nothing,
                     lineshape = superlorentzian)
    name = nameof(M)
    n = length(offset)
    # one number for every image, or one per image
    per(x, k) = x isa Real ? fill(Float64(x), n) : length(x) == n ? Float64.(collect(x)) :
        throw(DimensionMismatch("$name: $k has $(length(x)) values for $n offsets (one number, or one per image)"))
    if mtpulse === nothing
        (B1e === nothing || tm === nothing) &&
            throw(ArgumentError("$name: give the MT pulse as `B1e` and `tm`, or as `mtpulse` and `mtflip`"))
        mtflip === nothing || throw(ArgumentError("$name: `mtflip` goes with `mtpulse`; with `B1e` and `tm`, leave it out"))
        b, t = per(B1e, :B1e), per(tm, :tm)
    else
        (B1e === nothing && tm === nothing) ||
            throw(ArgumentError("$name: the MT pulse as `mtpulse` and `mtflip`, or as `B1e` and `tm`, not both"))
        mtflip === nothing && throw(ArgumentError("$name: `mtpulse` needs `mtflip`, its flip angle per image (degrees)"))
        t = fill(sum(mtpulse.duration), n)
        b = per(mtflip, :mtflip) .* _b1eperdegree(mtpulse, name)
    end
    tsv, TRv = per(ts, :ts), per(TR, :TR)
    for j in 1:n
        TRv[j] >= t[j] + tsv[j] ||
            throw(ArgumentError("$name: TR must hold the MT pulse and ts, TR ≥ tm + ts; image $j: " *
                                "TR = $(TRv[j]), tm + ts = $(t[j] + tsv[j])"))
        t[j] > 0 && tsv[j] >= 0 && b[j] >= 0 ||
            throw(ArgumentError("$name: image $j needs tm > 0, ts ≥ 0 and B1e ≥ 0"))
    end
    return (Float64.(collect(offset)), b, t, tsv, TRv, per(FA, :FA), lineshape)
end

# An MT pulse's power-equivalent amplitude per degree of its nominal flip angle (µT/degree): its
# shape, as given, has ∫ω1²dt = Σ αₖ²/τₖ over its duration tm (pulses.jl) and the net flip angle
# |Σ αₖ·e^(iφₖ)| (its rotation on resonance); scaled to flip angle θ, ω1e = (θ/net)·√(∫ω1²dt/tm),
# and B1e = ω1e/γ (`name`, the model's, for the message). A shape whose segments cancel (a 1-1̄
# binomial, 90° at 0° then 90° at 180°) has no net flip angle; in floating point its sum is not
# exactly 0 (cis(π) = −1 + 1.2e-16i), so the test is relative to the flip angles' total
function _b1eperdegree(p::RFPulse, name)
    net = abs(sum(((α, φ),) -> α * cis(deg2rad(φ)), zip(p.flip, p.phase)))
    net > sqrt(eps()) * sum(abs, p.flip) ||
        throw(ArgumentError("$name: the MT pulse's shape has no net flip angle, so `mtflip` cannot scale it; give B1e and tm"))
    return sqrt(_rfenergy(p) / sum(p.duration)) / _GAMMA_H / net
end

# The parameters, their roles and the R1f tie: QMTCWHenkelman's (qmt_cw.jl), the two-pool model
# under saturation that the pulsed models approximate (the method delegates: Julia's way to share
# a definition between concrete types)
parameters(::Type{<:QMTSPGRYarnykh}) = parameters(QMTCWHenkelman)

function signal!(out, m::QMTSPGRYarnykh, p)
    (; offset, B1e, tm, ts, TR, FA, lineshape) = m
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = p
    kfm = kmf * PSR                                         # exchange f→m (mass balance)
    A = SMatrix{2,2}(-(R1f + kfm), kfm, kmf, -(R1m + kmf))  # relaxation and exchange, by column (twopool.jl)
    M0 = SVector(one(PSR), PSR)                             # equilibrium, in units of M0f
    r = SVector(R1f, R1m * PSR)                             # R1·M0
    isempty(out) && return out
    # The free recoveries, Er = exp(A·tr) and Es = exp(A·ts), depend on the parameters and the
    # timings alone, not on the MT pulse's offset or power: made for the first image, and again
    # only where an image's tr or ts differs from the one before (once per signal for a protocol
    # with one TR, tm and ts, as usual; per change for several). Each image then makes only its
    # MT pulse's exponential, Em. The same numbers as made per image, to rounding (1e-15:
    # StaticArrays' 2×2 products are muladds, which the compiler fuses or not as the code around
    # them allows).
    j1 = firstindex(out)
    trp, tsp = TR[j1] - tm[j1] - ts[j1], ts[j1]
    Er, Es = _expm2(A * trp), _expm2(A * tsp)
    @inbounds for j in eachindex(out, offset, B1e, tm, ts, TR, FA)
        trj = TR[j] - tm[j] - ts[j]
        if trj != trp
            Er, trp = _expm2(A * trj), trj
        end
        if ts[j] != tsp
            Es, tsp = _expm2(A * ts[j]), ts[j]
        end
        α = B1 * deg2rad(FA[j])
        ω1² = (oftype(B1e[j], _GAMMA_H) * B1 * B1e[j])^2   # the MT pulse's, rad²/s² (γ in the acquisition's precision)
        Δ = offset[j] - B0
        gf, gm = lorentzian(T2f, Δ), lineshape(T2m, Δ)
        # no MT pulse (B1e = 0): no saturation, whatever the lineshape gives at its offset
        Wf = iszero(B1e[j]) ? zero(gf) * ω1² : π * ω1² * gf
        Wm = iszero(B1e[j]) ? zero(gm) * ω1² : π * ω1² * gm
        Em = _expm2((A - SMatrix{2,2}(Wf, 0, 0, Wm)) * tm[j])
        Mss = (SMatrix{2,2}(Wf, 0, 0, Wm) - A) \ r           # the steady state under the MT pulse, (W − A)·Mss = R1·M0
        C = SMatrix{2,2}(cos(α), 0, 0, 1)                   # the excitation: the free pool only
        EsEm = Es * Em
        rhs = (EsEm * (I - Er) + I - Es) * M0 + Es * ((I - Em) * Mss)
        Mz = (I - EsEm * Er * C) \ rhs                      # just before the excitation
        out[j] = M0f * sin(α) * Mz[1]
    end
    return out
end

# --- Ramani's approximation ---------------------------------------------------------------------
# The CWPE amplitude: ω1,CWPE² = (1/TR)·∫ω1²dt over the MT pulse = (tm/TR)·ω1e², with ω1e the
# pulse's power-equivalent amplitude over its own duration (Yarnykh's B1e). Its approximations
# against Yarnykh's model (the same acquisition): the saturation spread over the whole TR (exact
# only for TR short against the exchange and relaxation), and the excitations' effect on the free
# pool left out (small for small flip angles; the MT-off image then gives M0f). So its images are
# all at one flip angle and TR: VFA images in the same fit (as QMTSPGRYarnykh takes) are not for it.

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
same fit, as `QMTSPGRYarnykh` takes them, are not for it. The parameters: `QMTCWHenkelman`'s.
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

# The parameters: QMTCWHenkelman's, as QMTSPGRYarnykh's
parameters(::Type{<:QMTSPGRRamani}) = parameters(QMTCWHenkelman)

# Henkelman's steady state (QMTCWHenkelman's, qmt_cw.jl) at the continuous-wave power equivalent,
# ω1² = (tm/TR)·ω1e²: the MT pulse's power spread over TR
function signal!(out, m::QMTSPGRRamani, p)
    (; offset, B1e, tm, TR, lineshape) = m
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = p
    kfm = kmf * PSR                                         # exchange f→m
    @inbounds for j in eachindex(out, offset, B1e, tm, TR)
        ω1² = (tm[j] / TR[j]) * (oftype(B1e[j], _GAMMA_H) * B1 * B1e[j])^2   # the CWPE: the power over TR
        Δ = offset[j] - B0
        gf, gm = lorentzian(T2f, Δ), lineshape(T2m, Δ)
        Wf = iszero(ω1²) ? zero(gf) * ω1² : π * ω1² * gf     # each pool's saturation rate
        Wm = iszero(ω1²) ? zero(gm) * ω1² : π * ω1² * gm
        Mzf = (kfm * R1m + R1f * (R1m + kmf + Wm)) / (kfm * (R1m + Wm) + (R1f + Wf) * (R1m + kmf + Wm))
        out[j] = M0f * Mzf
    end
    return out
end
