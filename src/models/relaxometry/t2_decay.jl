# =================================================================================================
# Mono-exponential T2 decay, in two parametrizations, with or without an offset
#
#   T2Decay         S = S0·exp(-TE/T2)          time constant T2
#   R2Decay         S = S0·exp(-R2·TE)          rate R2 = 1/T2
#   T2DecayOffset   S = S0·exp(-TE/T2) + C      plus a constant offset C
#   R2DecayOffset   S = S0·exp(-R2·TE) + C
#
# The same models fit T2* (R2*) from gradient-echo data in the simple mono-exponential case.
# Diffusion has its own model (ADCDecay in diffusion/adc_decay.jl): the same equation with b and ADC.
#
# Why two parametrizations: fit the one you will report, so standard errors, CRLB and Monte Carlo
# come out directly for it. (Each voxel's estimate converts exactly, R2 = 1/T2, but averages over
# voxels do not: mean(T2) ≠ 1/mean(R2).)
#
# Which is more robust: the rate form. In tests, the time form failed silently when the true
# decay was much faster than its start: it stepped to a tiny T2 where the signal at every TE is
# zero and the gradient vanishes. On nearly flat curves the time form runs off to huge T2, while
# the rate form settles near R2 = 0. With the time form, give a start when you know the typical
# value, e.g. `start = (T2 = 0.05,)`.
#
# Default starts come from the acquisition: T2 = mean(TE), R2 = 1/mean(TE) (a decay comparable to
# the sampled range). S0 starts at the least-squares scale.
#
# The offset versions are an approximation people commonly use: a constant C for the noise
# floor of magnitude images. The real noise floor (Rician bias) depends on the signal level, so
# a constant is not the correct model; Rician-likelihood fitting is on the roadmap.
# =================================================================================================

# -------------------------------------------------------------------------------------------------
"""
    T2Decay(; TE)

Mono-exponential decay with a time constant: `S = S0·exp(-TE/T2)`. Also T2* from gradient-echo
data (the simple mono-exponential case).

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `T2` (s)

Starts at `T2 = mean(TE)`. Prefer [`R2Decay`](@ref) when decays can be much faster than the
echo times, or give a start, e.g. `start = (T2 = 0.05,)`. With an offset:
[`T2DecayOffset`](@ref).

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2Decay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(m::T2Decay) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(mean(m.TE); lower = _TMIN),
)

function signal!(out, m::T2Decay, p)
    (; TE) = m                          # unpack by name: TE = m.TE
    (; S0, T2) = p                      # S0 = p.S0, T2 = p.T2
    R2 = 1 / T2                         # once per call, so the loop multiplies instead of dividing
    @inbounds for j in eachindex(out, TE)
        out[j] = S0 * exp(-R2 * TE[j])
    end
    return out
end
# `eachindex(out, TE)` checks ONCE, before the loop, that `out` and `TE` have the same indices
# (a DimensionMismatch error otherwise), and then loops over them. That makes `@inbounds` safe:
# it skips the per-element bounds checks inside the loop, whoever calls `signal!` and with
# whatever `out`. List every per-observation field: `eachindex(out, TI, TD)` for QMTSIR.

# -------------------------------------------------------------------------------------------------
"""
    R2Decay(; TE)

Mono-exponential decay with a rate: `S = S0·exp(-R2·TE)`. Also R2* from gradient-echo data (the
simple mono-exponential case).

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `R2` (1/s)

Starts at `R2 = 1/mean(TE)`. The more robust of the two forms. With an offset:
[`R2DecayOffset`](@ref).

```julia
m = R2Decay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2Decay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(m::R2Decay) = (
    S0 = Free(:scale; lower = 0),
    R2 = Free(1 / mean(m.TE); lower = 0),
)

function signal!(out, m::R2Decay, p)
    (; TE) = m
    (; S0, R2) = p
    @inbounds for j in eachindex(out, TE)
        out[j] = S0 * exp(-R2 * TE[j])
    end
    return out
end

# -------------------------------------------------------------------------------------------------
"""
    T2DecayOffset(; TE)

[`T2Decay`](@ref) plus a constant offset: `S = S0·exp(-TE/T2) + C`.

A common approximation for the noise floor of magnitude images, which keeps the signal from
decaying to zero. It is not the correct model: the noise floor (Rician bias) depends on the
signal level, not a constant.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `T2` (s), `C` (fix a known offset with `fixed = (C = value,)`)

```julia
m = T2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2DecayOffset{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(m::T2DecayOffset) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(mean(m.TE); lower = _TMIN),
    C = Free(0.0),                      # offset, in signal units
)

function signal!(out, m::T2DecayOffset, p)
    (; TE) = m
    (; S0, T2, C) = p
    R2 = 1 / T2
    @inbounds for j in eachindex(out, TE)
        out[j] = S0 * exp(-R2 * TE[j]) + C
    end
    return out
end

# -------------------------------------------------------------------------------------------------
"""
    R2DecayOffset(; TE)

[`R2Decay`](@ref) plus a constant offset: `S = S0·exp(-R2·TE) + C`.

A common approximation for the noise floor of magnitude images, which keeps the signal from
decaying to zero. It is not the correct model: the noise floor (Rician bias) depends on the
signal level, not a constant.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `R2` (1/s), `C` (fix a known offset with `fixed = (C = value,)`)

```julia
m = R2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2DecayOffset{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(m::R2DecayOffset) = (
    S0 = Free(:scale; lower = 0),
    R2 = Free(1 / mean(m.TE); lower = 0),
    C = Free(0.0),                      # offset, in signal units
)

function signal!(out, m::R2DecayOffset, p)
    (; TE) = m
    (; S0, R2, C) = p
    @inbounds for j in eachindex(out, TE)
        out[j] = S0 * exp(-R2 * TE[j]) + C
    end
    return out
end
