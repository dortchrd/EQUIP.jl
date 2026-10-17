# =================================================================================================
# Saturation recovery: T1SR (time constant) and R1SR (rate)
#
# Tested from the default starts (T1 = 1 s, R1 = 1/s): the T1 form failed silently in most voxels
# for T1 ≤ 0.2 s (a step overshoots to a tiny T1, where the gradient vanishes); the R1 form did
# not. Give a start for short T1, e.g. `start = (T1 = 0.2,)`.
# =================================================================================================

"""
    T1SR(; TD)

Saturation recovery: `S = M0·(1 - exp(-TD/T1))`.

- acquisition: `TD` (s), the delay after saturation, one per observation
- parameters: `M0`, `T1` (s)

Rate version: [`R1SR`](@ref).
"""
@kwdef struct T1SR{A<:AbstractVector} <: MRIModel
    TD::A                               # delays after saturation (s)
end

parameters(::T1SR) = (
    M0 = Free(:scale; lower = 0),
    T1 = Free(1.0; lower = _TMIN),
)

function signal!(out, m::T1SR, p)
    (; TD) = m
    (; M0, T1) = p
    R1 = 1 / T1                         # once per call, so the loop multiplies instead of dividing
    @inbounds for j in eachindex(out, TD)
        out[j] = M0 * (1 - exp(-R1 * TD[j]))
    end
    return out
end

"""
    R1SR(; TD)

[`T1SR`](@ref) with the rate R1 = 1/T1: `S = M0·(1 - exp(-R1·TD))`.

- acquisition: `TD` (s), the delay after saturation, one per observation
- parameters: `M0`, `R1` (1/s)
"""
@kwdef struct R1SR{A<:AbstractVector} <: MRIModel
    TD::A                               # delays after saturation (s)
end

parameters(::R1SR) = (
    M0 = Free(:scale; lower = 0),
    R1 = Free(1.0; lower = 0),
)

function signal!(out, m::R1SR, p)
    (; TD) = m
    (; M0, R1) = p
    @inbounds for j in eachindex(out, TD)
        out[j] = M0 * (1 - exp(-R1 * TD[j]))
    end
    return out
end
