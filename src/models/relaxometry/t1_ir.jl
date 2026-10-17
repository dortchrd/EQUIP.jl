# =================================================================================================
# Inversion recovery
#
#   T1IR, R1IR       long repetition time: S = M0·(1 - B·exp(-TI/T1))
#   T1IRTD, R1IRTD   any delay TD between readout and inversion, modelled explicitly
#                    (Li et al., Magn Reson Med 2010;64:491, Eq. 1)
#
# Fit the quantity you will report, so its standard errors and CRLB come out directly. Tested from
# the default starts (T1 = 1 s, R1 = 1/s), 200 voxels at SNR 100:
# - Magnitude data can end in a wrong local minimum (the fit settles on the wrong side of the
#   signal null), in both forms, for T1 very short or very long compared with the TIs: T1IR at
#   0.05 and 4 s (even with TIs from 10 ms to 12.8 s), T1IRTD at 0.05-0.1 s. The T1 form also at
#   T1IR 0.1 s and T1IRTD 0.2-0.3 s, where the R1 form succeeded. Polarity restoration (planned)
#   is the usual remedy; until then prefer the R1 form there, or give a start near the expected
#   T1, e.g. `start = (T1 = 0.2,)`.
# - Long T1, low SNR (T1IR, T1 = 4 s, SNR 20): the two forms ended at different answers in 361
#   of 2000 voxels, and the T1 form had the better fit in 332 of them.
# =================================================================================================

"""
    T1IR(; TI, magnitude = true)

Inversion recovery with a long repetition time: `S = M0·(1 - B·exp(-TI/T1))`, its absolute
value for magnitude data. `B` absorbs imperfect inversion (B = 2 for a perfect one).

- acquisition: `TI` (s), one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- parameters: `M0`, `B`, `T1` (s)

Rate version: [`R1IR`](@ref).
"""
@kwdef struct T1IR{A<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    magnitude::Bool = true              # a default value makes the keyword optional
end

parameters(::T1IR) = (
    M0 = Free(:scale; lower = 0),
    B = Free(2.0; lower = 0, upper = 3),
    T1 = Free(1.0; lower = _TMIN),
)

function signal!(out, m::T1IR, p)
    (; TI, magnitude) = m
    (; M0, B, T1) = p
    R1 = 1 / T1                         # once per call, so the loop multiplies instead of dividing
    @inbounds for j in eachindex(out, TI)
        s = M0 * (1 - B * exp(-R1 * TI[j]))
        out[j] = magnitude ? abs(s) : s
    end
    return out
end

"""
    R1IR(; TI, magnitude = true)

[`T1IR`](@ref) with the rate R1 = 1/T1: `S = M0·(1 - B·exp(-R1·TI))`.

- acquisition: `TI` (s), one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- parameters: `M0`, `B`, `R1` (1/s)
"""
@kwdef struct R1IR{A<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    magnitude::Bool = true
end

parameters(::R1IR) = (
    M0 = Free(:scale; lower = 0),
    B = Free(2.0; lower = 0, upper = 3),
    R1 = Free(1.0; lower = 0),
)

function signal!(out, m::R1IR, p)
    (; TI, magnitude) = m
    (; M0, B, R1) = p
    @inbounds for j in eachindex(out, TI)
        s = M0 * (1 - B * exp(-R1 * TI[j]))
        out[j] = magnitude ? abs(s) : s
    end
    return out
end

# -------------------------------------------------------------------------------------------------
"""
    T1IRTD(; TI, TD, magnitude = true)

Inversion recovery with the delay `TD` between the readout and the next inversion modelled
explicitly, so TD can be short and can vary (Li et al., Magn Reson Med 2010;64:491, Eq. 1):

    S = M0·[Sf·Mz(TD)·exp(-TI/T1) + 1 - exp(-TI/T1)]
    Mz(TD) = 1 - (1 - Mz_td0)·exp(-TD/T1)

The readout (e.g. a spin echo or fast spin echo) is assumed to leave Mz = 0; Mz recovers for TD,
the inversion pulse multiplies it by Sf (-1 = perfect inversion), and it recovers for TI. With
Sf = -1 this is the familiar `M0·(1 - 2·exp(-TI/T1) + exp(-(TI+TD)/T1))`. For a long TD it
reduces to [`T1IR`](@ref) with B = 1 - Sf.

`Mz_td0` (an extension of Li's equation, which is the case `Mz_td0 = 0`) is Mz at the start of
TD as a fraction of M0. Use it to test the assumption that the readout saturates Mz, as
`Mzf_td0` does for [`QMTSIR`](@ref).

- acquisition: `TI` (s), one per observation; `TD` (s), one number for all observations or one
  per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- free by default: `M0`, `Sf` (inversion efficiency, -1 = perfect), `T1` (s)
- fixed by default: `Mz_td0 = 0`

Rate version: [`R1IRTD`](@ref).

```julia
m = T1IRTD(TI = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4], TD = [0.5, 3.0, 1.0, 0.2, 2.0, 0.05])
m = T1IRTD(TI = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4], TD = 1.0)     # one TD for all
```
"""
@kwdef struct T1IRTD{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
end

T1IRTD(TI, TD::Real, magnitude) = T1IRTD(TI, fill(TD, length(TI)), magnitude)   # one TD for all

parameters(::T1IRTD) = (
    M0 = Free(:scale; lower = 0),
    Sf = Free(-0.95; lower = -1.2, upper = 0),
    T1 = Free(1.0; lower = _TMIN),
    Mz_td0 = Fixed(0.0),                # Mz at the start of TD, as a fraction of M0: 0 = saturated
)

function signal!(out, m::T1IRTD, p)
    (; TI, TD, magnitude) = m
    (; M0, Sf, T1, Mz_td0) = p
    R1 = 1 / T1
    @inbounds for j in eachindex(out, TI, TD)
        Mz_td = 1 - (1 - Mz_td0) * exp(-R1 * TD[j])        # Mz at the end of TD (fraction of M0)
        E_ti = exp(-R1 * TI[j])
        s = M0 * (Sf * Mz_td * E_ti + 1 - E_ti)             # Li 2010, Eq. 1
        out[j] = magnitude ? abs(s) : s
    end
    return out
end

"""
    R1IRTD(; TI, TD, magnitude = true)

[`T1IRTD`](@ref) with the rate R1 = 1/T1:

    S = M0·[Sf·Mz(TD)·exp(-R1·TI) + 1 - exp(-R1·TI)]
    Mz(TD) = 1 - (1 - Mz_td0)·exp(-R1·TD)

- acquisition: `TI` (s), one per observation; `TD` (s), one number or one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- free by default: `M0`, `Sf`, `R1` (1/s)
- fixed by default: `Mz_td0 = 0`
"""
@kwdef struct R1IRTD{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
end

R1IRTD(TI, TD::Real, magnitude) = R1IRTD(TI, fill(TD, length(TI)), magnitude)   # one TD for all

parameters(::R1IRTD) = (
    M0 = Free(:scale; lower = 0),
    Sf = Free(-0.95; lower = -1.2, upper = 0),
    R1 = Free(1.0; lower = 0),
    Mz_td0 = Fixed(0.0),
)

function signal!(out, m::R1IRTD, p)
    (; TI, TD, magnitude) = m
    (; M0, Sf, R1, Mz_td0) = p
    @inbounds for j in eachindex(out, TI, TD)
        Mz_td = 1 - (1 - Mz_td0) * exp(-R1 * TD[j])
        E_ti = exp(-R1 * TI[j])
        s = M0 * (Sf * Mz_td * E_ti + 1 - E_ti)
        out[j] = magnitude ? abs(s) : s
    end
    return out
end
