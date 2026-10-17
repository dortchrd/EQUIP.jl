# =================================================================================================
# Inversion recovery: T1IR (time constant) and R1IR (rate)
#
# Fit the quantity you will report, so its standard errors and CRLB come out directly. In tests,
# the T1 form was the more robust: with T1 = 4 s at SNR 20, the two forms ended at different
# answers in 361 of 2000 voxels and the T1 form had the better fit in 332 of them.
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
