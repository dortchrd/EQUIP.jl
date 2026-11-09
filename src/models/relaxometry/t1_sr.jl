# =================================================================================================
# Saturation recovery: T1SR (time constant) and R1SR (rate)
# The equation is written once, in R1SR; T1SR converts and calls it (see builtin.jl).
#
# Tested from the default starts (T1 = 1 s, R1 = 1/s), 200 voxels at SNR 100: both forms recover
# T1 from 0.05 to 4 s. (The T1 form used to fail for T1 ≤ 0.2 s, stepping to T1 ≈ 0 where the
# gradient vanishes; the solver now rejects such steps, see src/fit/levmar.jl.)
# =================================================================================================

"""
    T1SR(; TD)

Saturation recovery: `S = M0·(1 - exp(-TD/T1))`.

- acquisition: `TD` (s), the delay after saturation, one per observation
- parameters: `M0`, `T1` (s, at most 5, as [`T1IR`](@ref))

Rate version: [`R1SR`](@ref).
"""
@kwdef struct T1SR{A<:AbstractVector} <: MRIModel
    TD::A                               # delays after saturation (s)
end

parameters(::Type{<:T1SR}) = (
    M0 = Free(start = :scale, lower = 0),
    T1 = Free(start = 1.0, lower = _TMIN, upper = _T1MAX),
)

# The T1 form is the R1 form with R1 = 1/T1: convert, then R1SR's signal (below; see builtin.jl
# for why the time form calls the rate form).
signal!(out, m::T1SR, p) = signal!(out, R1SR(m.TD), (; p.M0, R1 = 1 / p.T1))

"""
    R1SR(; TD)

[`T1SR`](@ref) with the rate R1 = 1/T1: `S = M0·(1 - exp(-R1·TD))`.

- acquisition: `TD` (s), the delay after saturation, one per observation
- parameters: `M0`, `R1` (1/s, at least 0.2)
"""
@kwdef struct R1SR{A<:AbstractVector} <: MRIModel
    TD::A                               # delays after saturation (s)
end

parameters(::Type{<:R1SR}) = (
    M0 = Free(start = :scale, lower = 0),
    R1 = Free(start = 1.0, lower = _R1MIN),
)

function signal!(out, m::R1SR, p)
    (; TD) = m
    (; M0, R1) = p
    @inbounds for j in eachindex(out, TD)
        out[j] = M0 * (1 - exp(-R1 * TD[j]))
    end
    return out
end
