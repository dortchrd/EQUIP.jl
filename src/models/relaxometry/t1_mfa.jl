# =================================================================================================
# Multiple flip angle (variable flip angle) T1: the spoiled gradient echo signal
# T1MFA (time constant) and R1MFA (rate)
# The equation is written once, in R1MFA; T1MFA converts and calls it (see builtin.jl).
#
# Fit the quantity you will report. Tested from the default starts (T1 = 1 s, R1 = 1/s), 200
# voxels at SNR 100: both forms recover T1 from 0.05 to 4 s (the T1 form used to fail at 0.05
# and 0.2 s, stepping to T1 ≈ 0; the solver now rejects such steps). The R1 form needs 1.5-2×
# more iterations for long T1 (4-20 s).
# =================================================================================================

"""
    T1MFA(; FA, TR)

Multiple flip angle T1 mapping with a spoiled gradient echo:

    S = M0·sin(α)·(1 - E1) / (1 - cos(α)·E1),   E1 = exp(-TR/T1),   α = B1·FA

- acquisition: `FA` (degrees), one per observation; `TR` (s), one number for all observations
  or one per observation
- parameters: `M0`, `T1` (s), and `B1` (relative flip angle), fixed at 1 by default. Pass a
  B1 map with `fit(m, data; B1 = b1map)`.

Rate version: [`R1MFA`](@ref).

```julia
m = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
```
"""
@kwdef struct T1MFA{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    FA::A                               # flip angles (degrees)
    TR::B                               # repetition times (s)
end

# One TR for every flip angle. `@kwdef`'s keyword constructor passes the fields on to the
# positional constructor `T1MFA(FA, TR)`; when TR is a single number, Julia picks this method
# (the most specific match), which repeats it once per flip angle.
T1MFA(FA, TR::Real) = T1MFA(FA, fill(TR, length(FA)))

parameters(::T1MFA) = (
    M0 = Free(start = :scale, lower = 0),
    T1 = Free(start = 1.0, lower = _TMIN),
    B1 = Fixed(1.0),
)

# The T1 form is the R1 form with R1 = 1/T1: convert, then R1MFA's signal (below; see builtin.jl
# for why the time form calls the rate form).
signal!(out, m::T1MFA, p) = signal!(out, R1MFA(m.FA, m.TR), (; p.M0, R1 = 1 / p.T1, p.B1))

"""
    R1MFA(; FA, TR)

[`T1MFA`](@ref) with the rate R1 = 1/T1 (`E1 = exp(-R1·TR)`).

- acquisition: `FA` (degrees), one per observation; `TR` (s), one number for all observations
  or one per observation
- parameters: `M0`, `R1` (1/s), and `B1` (relative flip angle), fixed at 1 by default

```julia
m = R1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
```
"""
@kwdef struct R1MFA{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    FA::A                               # flip angles (degrees)
    TR::B                               # repetition times (s)
end

R1MFA(FA, TR::Real) = R1MFA(FA, fill(TR, length(FA)))    # one TR for every flip angle

parameters(::R1MFA) = (
    M0 = Free(start = :scale, lower = 0),
    R1 = Free(start = 1.0, lower = 0),
    B1 = Fixed(1.0),
)

function signal!(out, m::R1MFA, p)
    (; FA, TR) = m
    (; M0, R1, B1) = p
    @inbounds for j in eachindex(out, FA, TR)
        sinα, cosα = sincos(B1 * deg2rad(FA[j]))   # actual flip angle in radians
        E1 = exp(-R1 * TR[j])
        out[j] = M0 * sinα * (1 - E1) / (1 - cosα * E1)
    end
    return out
end
