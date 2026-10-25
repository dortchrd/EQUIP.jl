# =================================================================================================
# Diffusion
# =================================================================================================

"""
    ADCDecay(; b)

Mono-exponential diffusion decay: `S = S0·exp(-b·ADC)`, one apparent diffusion coefficient per
voxel (e.g. from trace-weighted images). The same equation as [`R2Decay`](@ref), with
diffusion's names and units.

- acquisition: `b` (s/mm²), one per observation; include b = 0 or not
- free by default: `S0` (the signal at b = 0), `ADC` (mm²/s; about 0.7e-3 in brain tissue,
  3e-3 in free water at body temperature)

Starts at `ADC = 1/mean(b)`, which does not depend on the units of `b`.

```julia
m = ADCDecay(b = [0, 250, 500, 750, 1000])
```
"""
@kwdef struct ADCDecay{A<:AbstractVector} <: MRIModel
    b::A                                # b-values (s/mm²)
end

parameters(::Type{<:ADCDecay}) = (
    S0 = Free(start = :scale, lower = 0),
    ADC = Free(start = m -> 1 / mean(m.b), lower = 0),   # from the b-values, once per fit
)

function signal!(out, m::ADCDecay, p)
    (; b) = m
    (; S0, ADC) = p
    @inbounds for j in eachindex(out, b)
        out[j] = S0 * exp(-ADC * b[j])
    end
    return out
end
