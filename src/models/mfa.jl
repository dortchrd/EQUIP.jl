# =================================================================================================
# Multiple flip angle (variable flip angle) T1: the spoiled gradient echo signal
# =================================================================================================

"""
    MFA(; FA, TR)

Multiple flip angle T1 mapping with a spoiled gradient echo:

    S = M0·sin(α)·(1 - E1) / (1 - cos(α)·E1),   E1 = exp(-TR/T1),   α = B1·FA

- acquisition: `FA` (degrees), one per observation; `TR` (s), one number for all observations
  or one per observation
- parameters: `M0`, `T1` (s), and `B1` (relative flip angle), fixed at 1 by default. Pass a
  B1 map with `fixed = (B1 = b1map,)`.

```julia
m = MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
```
"""
@kwdef struct MFA{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    FA::A                               # flip angles (degrees)
    TR::B                               # repetition times (s)
end

# One TR for every flip angle. `@kwdef`'s keyword constructor passes the fields on to the
# positional constructor `MFA(FA, TR)`; when TR is a single number, Julia picks this method
# (the most specific match), which repeats it once per flip angle.
MFA(FA, TR::Real) = MFA(FA, fill(TR, length(FA)))

parameters(::MFA) = (
    M0 = Free(:scale; lower = 0),
    T1 = Free(1.0; lower = _TMIN),
    B1 = Fixed(1.0),
)

function signal!(out, m::MFA, p)
    (; FA, TR) = m
    (; M0, T1, B1) = p
    @inbounds for j in eachindex(out, FA, TR)
        sinα, cosα = sincos(B1 * deg2rad(FA[j]))   # actual flip angle in radians
        E1 = exp(-TR[j] / T1)
        out[j] = M0 * sinα * (1 - E1) / (1 - cosα * E1)
    end
    return out
end
