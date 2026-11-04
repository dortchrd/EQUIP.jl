# =================================================================================================
# T2 decay: one or two exponential components, with or without an offset, in two parametrizations
#
#   S = S0·[(1 − fshort)·exp(−TE/T2) + fshort·exp(−TE/T2short)] + C          time constants
#   S = S0·[(1 − fshort)·exp(−R2·TE) + fshort·exp(−R2short·TE)] + C          rates R2 = 1/T2
#
# One signal equation with every part (src/models/blocks.jl, "Building blocks"); the parts off by
# default are fixed at values that remove them: the short component's fraction fshort = 0 (its T2
# Inf, its rate 0) and the offset C = 0. The named models differ only in what is on by default:
#
#   T2Decay        R2Decay        S0, T2           mono-exponential
#   T2DecayOffset  R2DecayOffset  S0, T2, C        plus a constant offset
#   T2BiexpDecay   R2BiexpDecay   S0, fshort, T2short, T2     two components (e.g. myelin water)
#
# and any part can be switched on or off in a call: `fit(T2Decay(TE = te), y; C = Free())` is
# T2DecayOffset's fit (with T2Decay's starts), `fit(m, y; fshort = Free(), T2short = Free())` the
# bi-exponential one, both together a bi-exponential decay with an offset, which had no model before.
# The same models fit T2* (R2*) from gradient-echo data in the simple cases. Diffusion has its own
# model (ADCDecay in diffusion/adc_decay.jl): the same equation with b and ADC.
#
# Names (changed 2026-11-03, with the building blocks): the main component is `T2` (`R2`) in every
# model, the bi-exponential's long component included (it was `T2long`, `R2long`); the second
# component is `fshort` and `T2short` (`R2short`).
#
# Why two parametrizations: fit the one you will report, so standard errors, CRLB and Monte Carlo
# come out directly for it. (Each voxel's estimate converts exactly, R2 = 1/T2, but averages over
# voxels do not: mean(T2) ≠ 1/mean(R2).) The equation is written once, in the rate form; the time
# form converts and calls it (see builtin.jl).
#
# Which is more robust: the rate form. On nearly flat curves (T2 much longer than the echo times)
# the time form runs off to huge T2 and stops at max iterations, whatever its start, while the
# rate form settles near R2 = 0. (The time form also used to fail when the decay was much faster
# than its start, stepping to a tiny T2 where the gradient vanishes; the solver now rejects such
# steps, see src/fit/levmar.jl.)
#
# Starting values: T2Decay starts each voxel from a log-linear fit to its data (`T2 = Free(start =
# :data)`; `datastart`, below), which makes it much faster when T2 is far from the echo times.
# The others start from the acquisition: R2 = 1/mean(TE), T2 = mean(TE) (a decay comparable to
# the sampled range); the bi-exponential at T2short = mean(TE)/4, T2 = 2·mean(TE), fshort = 0.2
# (the rate forms at the same values as rates). S0 starts at the least-squares scale.
#
# The offset is an approximation people commonly use: a constant C for the noise floor of
# magnitude images. The real noise floor (Rician bias) depends on the signal level, so a constant
# is not the correct model (`RicianMean` is; Rician-likelihood fitting is on the roadmap).
#
# The bi-exponential is a hard fit: four strongly correlated parameters, and a mono-exponential
# curve has many descriptions (with fshort = 0, T2short can be anything; with T2short = T2, fshort
# can). The two components can also swap: (fshort, T2short, T2) and (1 − fshort, T2, T2short) give
# the same signal. The fit itself does not keep T2short < T2 (from the default starts the time
# form swapped them in 196 of 200 voxels at SNR 100, the rate form in none), so after the fit the
# labels are put in order: the shorter T2 is reported as T2short (`relabel`, below).
# =================================================================================================

"""
    AbstractT2Decay

The T2 decays with time constants: [`T2Decay`](@ref), [`T2DecayOffset`](@ref),
[`T2BiexpDecay`](@ref). One signal,

    S = S0·[(1 − fshort)·exp(−TE/T2) + fshort·exp(−TE/T2short)] + C,

and the same parameters, `S0`, `fshort`, `T2short`, `T2`, `C`; the models differ in which are
on by default. Any can be switched on or off in a call (`C = Free()`, `fshort = Free()` with
`T2short = Free()`, `C = 0.0`).
"""
abstract type AbstractT2Decay <: MRIModel end

"""
    AbstractR2Decay

The T2 decays with rates: [`R2Decay`](@ref), [`R2DecayOffset`](@ref), [`R2BiexpDecay`](@ref).
One signal, `S = S0·[(1 − fshort)·exp(−R2·TE) + fshort·exp(−R2short·TE)] + C`, and the same
parameters, `S0`, `fshort`, `R2short`, `R2`, `C`, switched on and off as for
[`AbstractT2Decay`](@ref).
"""
abstract type AbstractR2Decay <: MRIModel end

# Every part, with what is on by default for the mono-exponential (T2Decay, R2Decay); the named
# models change some roles. The order is the order of the outputs and of the free parameters in
# the solver: S0, fshort, T2short, T2, C keeps each model's free parameters in the order they had
# (S0, T2; S0, T2, C; S0, fshort, T2short, T2), so fits are unchanged to the last bit.
function _t2_parts()
    return (
        S0 = Free(start = :scale, lower = 0),
        fshort = Fixed(0.0; free = Free(start = 0.2, lower = 0, upper = 1)),   # off: no short component
        T2short = Fixed(Inf; free = Free(start = m -> mean(m.TE) / 4, lower = _TMIN)),
        T2 = Free(start = :data, lower = _TMIN),                               # a log-linear fit (datastart)
        C = Fixed(0.0; free = Free(start = 0.0)),                              # off: no offset
    )
end
function _r2_parts()
    return (
        S0 = Free(start = :scale, lower = 0),
        fshort = Fixed(0.0; free = Free(start = 0.2, lower = 0, upper = 1)),
        R2short = Fixed(0.0; free = Free(start = m -> 4 / mean(m.TE), lower = 0)),   # rate 0: none
        R2 = Free(start = m -> 1 / mean(m.TE), lower = 0),                     # from the echo times
        C = Fixed(0.0; free = Free(start = 0.0)),
    )
end

# The signal, written once (rates): one observation from the blocks, then the loop. With the parts
# off, (1 − 0)·e + 0 + 0 = e: the mono-exponential's numbers exactly. The short component is
# skipped when its fraction is fixed at 0 (`_skip`, blocks.jl): its zero is then made from the
# types alone (`zero(fshort * R2short * TE)`, no exp), so the sum has one type either way.
@inline function _t2obs(TE, S0, fshort, R2short, R2, C)
    main = (1 - fshort) * decay(TE, R2)
    short = _skip(fshort) ? zero(fshort * R2short * TE) : fshort * decay(TE, R2short)
    return S0 * (main + short) + C
end

function _t2signal!(out, TE, S0, fshort, R2short, R2, C)
    @inbounds for j in eachindex(out, TE)
        out[j] = _t2obs(TE[j], S0, fshort, R2short, R2, C)
    end
    return out
end
# `eachindex(out, TE)` checks ONCE, before the loop, that `out` and `TE` have the same indices
# (a DimensionMismatch error otherwise), and then loops over them. That makes `@inbounds` safe:
# it skips the per-element bounds checks inside the loop, whoever calls `signal!` and with
# whatever `out`. List every per-observation field: `eachindex(out, TI, TD)` for QMTSIR.

signal!(out, m::AbstractR2Decay, p) = _t2signal!(out, m.TE, p.S0, p.fshort, p.R2short, p.R2, p.C)
# The time form: R2 = 1/T2 for each component (1/Inf = 0: a component switched off stays off),
# then the same signal (see builtin.jl for why time converts to rate). A model written per echo
# as `decay(TE, 1/T2)` (Blocks) gives the same numbers bit for bit, and is no slower: 1/T2 does
# not change from echo to echo, so the compiler computes it once, outside the loop (measured,
# examples/building_blocks.jl).
signal!(out, m::AbstractT2Decay, p) =
    _t2signal!(out, m.TE, p.S0, p.fshort, 1 / p.T2short, 1 / p.T2, p.C)

# Each voxel starts from a log-linear fit (a weighted straight line through log(y) against TE; see
# `_loglinear_rate` in builtin.jl), or from mean(TE) when the data do not decay (the rate is then
# NaN): for T2's `:data` start, T2Decay's default. S0 is set afterwards by least squares.
function datastart(m::AbstractT2Decay, y, p)
    R2 = _loglinear_rate(m.TE, y)
    return (T2 = isnan(R2) ? mean(m.TE) : 1 / R2,)
end

# The components can swap; report the shorter T2 (the larger rate) as the short one. Only when
# fshort, the short and the main component are all free: a fixed one pins the labels, and
# swapping would change the signal.
function relabel(::AbstractT2Decay, θ)
    haskey(θ, :fshort) && haskey(θ, :T2short) && haskey(θ, :T2) || return θ
    θ.T2short <= θ.T2 && return θ
    return merge(θ, (fshort = 1 - θ.fshort, T2short = θ.T2, T2 = θ.T2short))
end
function relabel(::AbstractR2Decay, θ)
    haskey(θ, :fshort) && haskey(θ, :R2short) && haskey(θ, :R2) || return θ
    θ.R2short >= θ.R2 && return θ
    return merge(θ, (fshort = 1 - θ.fshort, R2short = θ.R2, R2 = θ.R2short))
end

# -------------------------------------------------------------------------------------------------
"""
    T2Decay(; TE)

Mono-exponential decay with a time constant: `S = S0·exp(-TE/T2)`. Also T2* from gradient-echo
data (the simple mono-exponential case).

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `T2` (s)
- off by default: the offset `C` and a short component (`fshort`, `T2short`): switch them on in
  the call, `fit(m, y; C = Free())`, `fit(m, y; fshort = Free(), T2short = Free())`
  ([`AbstractT2Decay`](@ref)), or use [`T2DecayOffset`](@ref), [`T2BiexpDecay`](@ref)

Each voxel starts from a log-linear fit to its data (a weighted straight line through log(y)
against TE), or from `T2 = mean(TE)` if the data do not decay; a start in the call
(`T2 = Free(start = ...)`) overrides it. Prefer [`R2Decay`](@ref) when T2 can be much longer
than the echo times: on nearly flat curves the T2 form runs off to huge T2 and stops at max
iterations, while R2 settles near 0.

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2Decay{A<:AbstractVector} <: AbstractT2Decay
    TE::A                               # echo times (s)
end
parameters(::Type{<:T2Decay}) = _t2_parts()

"""
    T2DecayOffset(; TE)

[`T2Decay`](@ref) with its offset on: `S = S0·exp(-TE/T2) + C`.

A common approximation for the noise floor of magnitude images, which keeps the signal from
decaying to zero. It is not the correct model: the noise floor (Rician bias) depends on the
signal level, not a constant.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `T2` (s), `C` (fix a known offset with `C = value`)

Starts at `T2 = mean(TE)`, `C = 0`.

```julia
m = T2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2DecayOffset{A<:AbstractVector} <: AbstractT2Decay
    TE::A                               # echo times (s)
end
parameters(::Type{<:T2DecayOffset}) = merge(_t2_parts(), (
    T2 = Free(start = m -> mean(m.TE), lower = _TMIN),
    C = Free(start = 0.0),              # offset, in signal units
))

"""
    T2BiexpDecay(; TE)

[`T2Decay`](@ref) with its short component on: bi-exponential decay with time constants, e.g.
myelin water (short T2) and intra- and extracellular water (the main component, `T2`):

    S = S0·[(1 - fshort)·exp(-TE/T2) + fshort·exp(-TE/T2short)]

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `fshort` (fraction of S0 in the short-T2 component, 0 to 1),
  `T2short`, `T2` (s; the long component, called `T2long` before 2026-11-03)
- off by default: the offset `C` (`C = Free()` adds it)

Starts at `T2short = mean(TE)/4`, `T2 = 2·mean(TE)`, `fshort = 0.2`. The two components can swap
(the same signal with `1 - fshort`), so after the fit the shorter T2 is reported as `T2short`,
when `fshort`, `T2short` and `T2` are all free. Rate version: [`R2BiexpDecay`](@ref).

```julia
m = T2BiexpDecay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2BiexpDecay{A<:AbstractVector} <: AbstractT2Decay
    TE::A                               # echo times (s)
end
parameters(::Type{<:T2BiexpDecay}) = merge(_t2_parts(), (
    fshort = Free(start = 0.2, lower = 0, upper = 1),     # fraction of S0 in the short-T2 component
    T2short = Free(start = m -> mean(m.TE) / 4, lower = _TMIN),    # from the echo times
    T2 = Free(start = m -> 2 * mean(m.TE), lower = _TMIN),
))

# -------------------------------------------------------------------------------------------------
"""
    R2Decay(; TE)

Mono-exponential decay with a rate: `S = S0·exp(-R2·TE)`. Also R2* from gradient-echo data (the
simple mono-exponential case).

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `R2` (1/s)
- off by default: the offset `C` and a short component (`fshort`, `R2short`), switched on as for
  [`T2Decay`](@ref) ([`AbstractR2Decay`](@ref))

Starts at `R2 = 1/mean(TE)`. The more robust of the two forms. With an offset:
[`R2DecayOffset`](@ref).

```julia
m = R2Decay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2Decay{A<:AbstractVector} <: AbstractR2Decay
    TE::A                               # echo times (s)
end
parameters(::Type{<:R2Decay}) = _r2_parts()

"""
    R2DecayOffset(; TE)

[`R2Decay`](@ref) with its offset on: `S = S0·exp(-R2·TE) + C`.

A common approximation for the noise floor of magnitude images, which keeps the signal from
decaying to zero. It is not the correct model: the noise floor (Rician bias) depends on the
signal level, not a constant.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `R2` (1/s), `C` (fix a known offset with `C = value`)

```julia
m = R2DecayOffset(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2DecayOffset{A<:AbstractVector} <: AbstractR2Decay
    TE::A                               # echo times (s)
end
parameters(::Type{<:R2DecayOffset}) = merge(_r2_parts(), (C = Free(start = 0.0),))

"""
    R2BiexpDecay(; TE)

[`T2BiexpDecay`](@ref) with rates, R2short = 1/T2short and R2 = 1/T2:

    S = S0·[(1 - fshort)·exp(-R2·TE) + fshort·exp(-R2short·TE)]

`R2short` is the rate of the short-T2 component (the larger rate); after the fit the larger
rate is reported as `R2short`, as for the T2 version.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `fshort` (0 to 1), `R2short`, `R2` (1/s; called `R2long` before
  2026-11-03)

Starts at `R2short = 4/mean(TE)`, `R2 = 1/(2·mean(TE))`, `fshort = 0.2`.

```julia
m = R2BiexpDecay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2BiexpDecay{A<:AbstractVector} <: AbstractR2Decay
    TE::A                               # echo times (s)
end
parameters(::Type{<:R2BiexpDecay}) = merge(_r2_parts(), (
    fshort = Free(start = 0.2, lower = 0, upper = 1),
    R2short = Free(start = m -> 4 / mean(m.TE), lower = 0),  # rate of the short-T2 component
    R2 = Free(start = m -> 1 / (2 * mean(m.TE)), lower = 0),
))
