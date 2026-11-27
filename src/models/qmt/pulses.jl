# =================================================================================================
# RF pulses, and what one does to each pool of a two-pool (qMT) system
#
#   the free pool (f):            rotated, segment by segment, about the pulse's effective field:
#                                 B1 scales the amplitude, an offset Δ (B0) tilts the axis toward
#                                 z. Its Mz/M0 after the pulse, from Mz = M0, is SIR's Sf, the
#                                 inversion efficiency (−1: a perfect inversion).
#   the macromolecular pool (m):  saturated at the rate W(t) = π·ω1(t)²·g(Δ) (lineshapes.jl), its
#                                 transverse magnetization not followed (T2m ≈ 10–15 µs, far
#                                 shorter than the pulse): Mz/M0 = exp(−π·g(Δ)·∫ω1² dt), SIR's
#                                 Sm (1: untouched).
#
# Both neglect relaxation and exchange during the pulse (the pulse, about a millisecond, short
# against T1 and 1/kmf). On resonance the macromolecular pool's lineshape is the Gaussian by
# default, finite there (g(0) = T2m/√(2π)); the super-Lorentzian diverges on resonance
# (lineshapes.jl). As Dortch et al., Magn Reson Med 2018 (calcSm_blockPulse.m,
# calcSm_compPulse.m): the 1-ms block inversion gives Sm = 0.83 at T2m = 15 µs, the paper's
# value. The same saturation rate W, off resonance, is what an MT pulse does in pulsed MT.
#
# A pulse is a list of segments of constant amplitude and phase (a block: one; the composite
# 90x-180y-90x: three; a shaped pulse from the scanner: its samples), each a flip angle (degrees,
# as EQUIP's flip angles elsewhere), a phase (degrees: 0 along x, 90 along y) and a duration (s).
# Segment k's amplitude is ω1 = B1·α_k/τ_k (rad/s), so ∫ω1² dt = B1²·Σ α_k²/τ_k.
# =================================================================================================

"""
    RFPulse(flip, phase, duration)
    blockpulse(flip, duration; phase = 0)

An RF pulse as segments of constant amplitude: per segment, its flip angle (degrees, at B1 = 1),
its phase (degrees: 0 along x, 90 along y) and its duration (s). `duration` is a vector (per
segment) or a number (the whole pulse's, split evenly: a shaped pulse's samples).

```julia
blockpulse(180, 1e-3)                                            # SIR's 1-ms inversion
RFPulse([90, 180, 90], [0, 90, 0], [0.5e-3, 1e-3, 0.5e-3])       # the composite 90x-180y-90x
RFPulse(flips, phases, 4e-3)                                     # a shaped pulse's samples, 4 ms
```

What it does to each pool: [`inversionefficiency`](@ref) (the free pool's Sf) and
[`saturationfraction`](@ref) (the macromolecular pool's Sm).
"""
struct RFPulse
    flip::Vector{Float64}       # degrees, per segment (at B1 = 1)
    phase::Vector{Float64}      # degrees
    duration::Vector{Float64}   # s
    function RFPulse(flip::AbstractVector{<:Real}, phase::AbstractVector{<:Real}, duration::AbstractVector{<:Real})
        length(flip) == length(phase) == length(duration) ||
            throw(ArgumentError("RFPulse: flip ($(length(flip))), phase ($(length(phase))) and duration " *
                                "($(length(duration))) need one value per segment"))
        isempty(flip) && throw(ArgumentError("RFPulse: no segments"))
        all(τ -> isfinite(τ) && τ > 0, duration) ||
            throw(ArgumentError("RFPulse: every duration must be > 0 (s); got $(duration)"))
        return new(Float64.(flip), Float64.(phase), Float64.(duration))
    end
end
RFPulse(flip::AbstractVector{<:Real}, phase::AbstractVector{<:Real}, duration::Real) =
    RFPulse(flip, phase, fill(duration / length(flip), length(flip)))
blockpulse(flip::Real, duration::Real; phase::Real = 0) = RFPulse([flip], [phase], [duration])
@doc (@doc RFPulse) blockpulse

Base.show(io::IO, p::RFPulse) = print(io, "RFPulse(", length(p.flip), length(p.flip) == 1 ? " segment, " : " segments, ",
                                      round(sum(p.duration) * 1e3; sigdigits = 4), " ms)")

# ∫ω1² dt at B1 = 1 (rad²/s): Σ α_k²/τ_k
_rfenergy(p::RFPulse) = sum(((α, τ),) -> deg2rad(α)^2 / τ, zip(p.flip, p.duration))

"""
    saturationfraction(pulse, T2m; B1 = 1, Δ = 0, lineshape = gaussian)

The fraction of the macromolecular pool's Mz left after `pulse` (an [`RFPulse`](@ref)), Mz/M0
from Mz = M0 (1: untouched; 0: fully saturated): SIR's `Sm`, as [`inversionefficiency`](@ref)
is its `Sf`.
Saturated at the rate π·ω1²·g(Δ), so Sm = exp(−π·g(Δ)·B1²·∫ω1² dt), with g the pool's
lineshape (`gaussian`, finite on resonance; [`lorentzian`](@ref); `superlorentzian` off
resonance only) for its T2, `T2m` (s), at the pulse's offset `Δ` (Hz) from the pool's resonance
(a B0 map). `B1` scales the pulse's amplitude (a B1 map's relative value). Relaxation and
exchange during the pulse are neglected. Dortch et al., Magn Reson Med 2018: the 1-ms block
inversion gives Sm = 0.83 at T2m = 15 µs.

```julia
saturationfraction(blockpulse(180, 1e-3), 15e-6)                 # 0.831
Sm = [saturationfraction(p, 15e-6; B1 = b) for b in b1map]      # a map, for fit(...; Sm = Sm)
```
"""
saturationfraction(p::RFPulse, T2m; B1 = 1, Δ = 0, lineshape = gaussian) =
    exp(-π * lineshape(T2m, Δ) * B1^2 * _rfenergy(p))

"""
    inversionefficiency(pulse; B1 = 1, Δ = 0)

The free pool's Mz/M0 after `pulse` (an [`RFPulse`](@ref)), from Mz = M0: SIR's `Sf` (−1: a
perfect inversion). Each segment rotates the magnetization about its effective field,
(ω1·cos φ, ω1·sin φ, 2πΔ) with ω1 = B1·α/τ, by its magnitude times τ: `B1` scales the amplitude
(a B1 map's relative value) and `Δ` is the offset (Hz) of the pool's resonance from the pulse's
frequency (a B0 map). Relaxation and exchange during the pulse are neglected.

```julia
inversionefficiency(blockpulse(180, 1e-3); B1 = 0.9)          # cos(162°) = −0.951
inversionefficiency(RFPulse([90, 180, 90], [0, 90, 0], [0.5e-3, 1e-3, 0.5e-3]); B1 = 0.9)   # −0.999
```
"""
function inversionefficiency(p::RFPulse; B1 = 1, Δ = 0)
    T = float(promote_type(typeof(B1), typeof(Δ)))
    M = SVector{3,T}(0, 0, 1)
    @inbounds for k in eachindex(p.flip)
        τ, φ = p.duration[k], deg2rad(p.phase[k])
        ω1 = B1 * deg2rad(p.flip[k]) / τ
        Ω = SVector(ω1 * cos(φ), ω1 * sin(φ), 2π * Δ)       # the effective field (rad/s)
        ω = sqrt(inner(Ω, Ω))
        ω == 0 && continue                                   # no field: no rotation
        n, θ = Ω / ω, ω * τ
        # a rotation about n by θ (Rodrigues): M cos θ + (n × M) sin θ + n (n·M)(1 − cos θ)
        M = M * cos(θ) + cross(n, M) * sin(θ) + n * (inner(n, M) * (1 - cos(θ)))
    end
    return M[3]
end
