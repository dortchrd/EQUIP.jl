# =================================================================================================
# Saturation recovery (T1)
# =================================================================================================

"""
    T1SR(; TD)

Saturation recovery: `S = M0·(1 - exp(-TD/T1))`.

- acquisition: `TD` (s), the delay after saturation, one per observation
- parameters: `M0`, `T1` (s)
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
    @inbounds for j in eachindex(out, TD)
        out[j] = M0 * (1 - exp(-TD[j] / T1))
    end
    return out
end
