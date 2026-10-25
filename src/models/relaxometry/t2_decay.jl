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
# voxels do not: mean(T2) ≠ 1/mean(R2).) The equations are written once, in the R2 versions; the
# T2 versions convert and call them (see builtin.jl).
#
# Which is more robust: the rate form. On nearly flat curves (T2 much longer than the echo
# times) the time form runs off to huge T2 and stops at max iterations, whatever its start, while
# the rate form settles near R2 = 0. (The time form also used to fail when the decay was much
# faster than its start, stepping to a tiny T2 where the gradient vanishes; the solver now rejects
# such steps, see src/fit/levmar.jl.)
#
# Starting values: T2Decay starts each voxel from a log-linear fit to its data
# (`T2 = Free(start = :data)`; `datastart`, after its `signal!`), which makes it much faster when T2 is far from the echo
# times. The others start from the acquisition: R2 = 1/mean(TE), T2 = mean(TE) (a decay comparable
# to the sampled range). S0 starts at the least-squares scale.
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

Each voxel starts from a log-linear fit to its data (a weighted straight line through log(y)
against TE), or from `T2 = mean(TE)` if the data do not decay; a start in the call
(`T2 = Free(start = ...)`) overrides it. Prefer [`R2Decay`](@ref) when T2 can be much longer
than the echo times: on nearly flat curves the T2 form runs off to huge T2 and stops at max
iterations, while R2 settles near 0. With an offset: [`T2DecayOffset`](@ref).

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2Decay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:T2Decay}) = (
    S0 = Free(start = :scale, lower = 0),
    T2 = Free(start = :data, lower = _TMIN),          # a log-linear fit (datastart, below)
)

# The T2 form is the R2 form with R2 = 1/T2: convert, then R2Decay's signal (below; see
# builtin.jl for why the time form calls the rate form).
signal!(out, m::T2Decay, p) = signal!(out, R2Decay(m.TE), (; p.S0, R2 = 1 / p.T2))

# Each voxel starts from a log-linear fit (a weighted straight line through log(y) against TE; see
# `_loglinear_rate` in builtin.jl), or from mean(TE) when the data do not decay (the rate is then
# NaN). S0 is set afterwards by least squares, given this T2.
function datastart(m::T2Decay, y, p)
    R2 = _loglinear_rate(m.TE, y)
    return (T2 = isnan(R2) ? mean(m.TE) : 1 / R2,)
end

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

parameters(::Type{<:R2Decay}) = (
    S0 = Free(start = :scale, lower = 0),
    R2 = Free(start = m -> 1 / mean(m.TE), lower = 0),   # from the echo times, once per fit
)

function signal!(out, m::R2Decay, p)
    (; TE) = m                          # unpack by name: TE = m.TE
    (; S0, R2) = p                      # S0 = p.S0, R2 = p.R2
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
    T2DecayOffset(; TE)

[`T2Decay`](@ref) plus a constant offset: `S = S0·exp(-TE/T2) + C`.

A common approximation for the noise floor of magnitude images, which keeps the signal from
decaying to zero. It is not the correct model: the noise floor (Rician bias) depends on the
signal level, not a constant.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `T2` (s), `C` (fix a known offset with `C = value`)

```julia
m = T2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2DecayOffset{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:T2DecayOffset}) = (
    S0 = Free(start = :scale, lower = 0),
    T2 = Free(start = m -> mean(m.TE), lower = _TMIN),
    C = Free(start = 0.0),              # offset, in signal units
)

# The T2 form: R2 = 1/T2, then R2DecayOffset's signal (below).
signal!(out, m::T2DecayOffset, p) = signal!(out, R2DecayOffset(m.TE), (; p.S0, R2 = 1 / p.T2, p.C))

# -------------------------------------------------------------------------------------------------
"""
    R2DecayOffset(; TE)

[`R2Decay`](@ref) plus a constant offset: `S = S0·exp(-R2·TE) + C`.

A common approximation for the noise floor of magnitude images, which keeps the signal from
decaying to zero. It is not the correct model: the noise floor (Rician bias) depends on the
signal level, not a constant.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `R2` (1/s), `C` (fix a known offset with `C = value`)

```julia
m = R2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2DecayOffset{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:R2DecayOffset}) = (
    S0 = Free(start = :scale, lower = 0),
    R2 = Free(start = m -> 1 / mean(m.TE), lower = 0),
    C = Free(start = 0.0),              # offset, in signal units
)

function signal!(out, m::R2DecayOffset, p)
    (; TE) = m
    (; S0, R2, C) = p
    @inbounds for j in eachindex(out, TE)
        out[j] = S0 * exp(-R2 * TE[j]) + C
    end
    return out
end
