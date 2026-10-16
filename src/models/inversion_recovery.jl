# =================================================================================================
# Inversion recovery (T1)
# =================================================================================================

"""
    InversionRecovery(; TI, magnitude = true)

Inversion recovery with a long repetition time: `S = M0·(1 - B·exp(-TI/T1))`, its absolute
value for magnitude data. `B` absorbs imperfect inversion (B = 2 for a perfect one).

- acquisition: `TI` (s), one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- parameters: `M0`, `B`, `T1` (s)
"""
@kwdef struct InversionRecovery{A<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    magnitude::Bool = true              # a default value makes the keyword optional
end

parameters(::InversionRecovery) = (
    M0 = Free(:scale; lower = 0),
    B = Free(2.0; lower = 0, upper = 3),
    T1 = Free(1.0; lower = _TMIN),
)

function signal!(out, m::InversionRecovery, p)
    (; TI, magnitude) = m
    (; M0, B, T1) = p
    @inbounds for j in eachindex(out, TI)
        s = M0 * (1 - B * exp(-TI[j] / T1))
        out[j] = magnitude ? abs(s) : s
    end
    return out
end
