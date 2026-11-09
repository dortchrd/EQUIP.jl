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
# times) the time form runs off to huge T2, while the rate form heads for R2 = 0. Every model here
# bounds T2 at 2.5 s and R2 at 0.4 s⁻¹ (`_T2MAX`, `_R2MIN`, builtin.jl), so such voxels end on the
# bound and `onbound` flags them, the same in both forms; before the bound (2026-11-08) T2Decay
# reported convergence at about 1e10 s. (The time form also used to fail when the decay was much
# faster than its start, stepping to a tiny T2 where the gradient vanishes; the solver now rejects
# such steps, see src/fit/levmar.jl.)
#
# Starting values: T2Decay starts each voxel from the log-linear fit to its data, the same two
# passes as DTI's, on 1/T2 (`T2 = Free(start = :loglinear)`; starts.jl). The others start from the
# acquisition: R2 = 1/mean(TE), T2 = mean(TE) (a decay comparable to the sampled range). S0
# starts at the least-squares scale.
#
# Why the time form needs a data start and the rate form does not: the solver's steps are
# linearizations in the parameter it fits. ln S is a straight line in R2, so a step from any
# constant lands near the answer (R2Decay: 4–8 iterations from 1/mean(TE)). In T2 a step goes
# through R2 = 1/T2: ΔT2 = −T2²·ΔR2, which gives T2_new = T2·(2 − T2/T2*) for an exact rate step
# (T2* the answer). Towards a longer T2 each step at most doubles it (from mean(TE) = 0.165 s to
# T2 = 10 s at 32 echoes: 33 iterations, against R2Decay's 6, measured); towards a shorter one a
# step from beyond 2·T2* overshoots below zero, into the dead zone the solver rejects. A start
# near the answer avoids both (docs/design.md, "T2Decay on the shared log-linear start").
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

Each voxel starts from the log-linear fit to its data (`:loglinear`, run on 1/T2: OLS, then WLS
weighted by the predicted signal); a start in the call (`T2 = Free(start = ...)`) overrides it.
The time form needs it: its steps go through 1/T2, so from a constant start they crawl towards
a long T2 and overshoot a short one, where R2Decay's do not.

T2 is bounded above at 2.5 s, longer than any tissue's (CSF's is about 2 s). Data that do not
decay are best fitted by T2 → ∞; they end on the bound instead, where [`onbound`](@ref) flags
them (`onbound(res).T2`), as R2Decay's R2 at its bound 0.4 s⁻¹ is. For longer T2s (phantoms of
water), widen it: `fit(m, data; T2 = Free(upper = 10))`. Prefer [`R2Decay`](@ref) when T2 can be
much longer than the echo times: its steps are straight lines in ln S. With an offset:
[`T2DecayOffset`](@ref).

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2Decay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:T2Decay}) = (
    S0 = Free(start = :scale, lower = 0),
    T2 = Free(start = :loglinear, lower = _TMIN, upper = _T2MAX),  # the log-linear fit, on 1/T2 (starts.jl);
                                                                   # at most 2.5 s (builtin.jl)
)

# The T2 form is the R2 form with R2 = 1/T2: convert, then R2Decay's signal (below; see
# builtin.jl for why the time form calls the rate form).
signal!(out, m::T2Decay, p) = signal!(out, R2Decay(m.TE), (; p.S0, R2 = 1 / p.T2))

# -------------------------------------------------------------------------------------------------
"""
    R2Decay(; TE)

Mono-exponential decay with a rate: `S = S0·exp(-R2·TE)`. Also R2* from gradient-echo data (the
simple mono-exponential case).

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `R2` (1/s)

Starts at `R2 = 1/mean(TE)`. The more robust of the two forms. R2 is bounded below at 0.4 s⁻¹
(T2 at most 2.5 s, as [`T2Decay`](@ref)): data that do not decay end there, flagged by
[`onbound`](@ref); widen it with `fit(m, data; R2 = Free(lower = 0))`. With an offset:
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
    R2 = Free(start = m -> 1 / mean(m.TE), lower = _R2MIN),   # from the echo times, once per fit;
                                                               # at least 0.4/s (builtin.jl)
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
- free by default: `S0`, `T2` (s, at most 2.5 s as [`T2Decay`](@ref)'s), `C` (fix a known
  offset with `C = value`)

```julia
m = T2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2DecayOffset{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:T2DecayOffset}) = (
    S0 = Free(start = :scale, lower = 0),
    T2 = Free(start = m -> mean(m.TE), lower = _TMIN, upper = _T2MAX),
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
- free by default: `S0`, `R2` (1/s, at least 0.4 s⁻¹ as [`R2Decay`](@ref)'s), `C` (fix a known
  offset with `C = value`)

```julia
m = R2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2DecayOffset{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:R2DecayOffset}) = (
    S0 = Free(start = :scale, lower = 0),
    R2 = Free(start = m -> 1 / mean(m.TE), lower = _R2MIN),
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
