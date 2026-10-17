# =================================================================================================
# Mono-exponential decays: T2 (spin echo), T2* (gradient echo), diffusion (ADC)
#
# All three are the same equation, S = S0·exp(-R·x) + C, with a different measured variable x
# (TE or b) and decay rate R (1/T2, 1/T2*, ADC). So the loop is written once, in `_monoexp!`, and
# each model only declares its acquisition field and its parameters: names, starting values and
# bounds suited to its physics, which also gives each output map its proper name.
#
# - Offset: C (default fixed at 0) is a constant added to the signal, e.g. the noise floor of
#   magnitude images. Estimate it with `free = (:C,)`; "with or without offset" is the same model.
# - Rates: fitting R2 or T2 gives the same least-squares answer, so R2 maps are `1 ./ T2` after
#   the fit. A model fitted directly in rate form (e.g. R2*, which is better behaved for nearly
#   flat curves) is a few lines in the same pattern, to add when needed.
# =================================================================================================

# The one loop for every mono-exponential decay: S0·exp(-R·x) + C at every observation.
function _monoexp!(out, x, S0, R, C)
    if C isa AbstractFloat && iszero(C)         # no offset (C fixed at 0): skip adding it
        @inbounds for j in eachindex(out, x)
            out[j] = S0 * exp(-R * x[j])
        end
    else
        @inbounds for j in eachindex(out, x)
            out[j] = S0 * exp(-R * x[j]) + C
        end
    end
    return out
end
# Both loops give the same values; the check only skips useless work. Adding C to every
# observation cost ~8% of a whole fit, so it is skipped when C is a plain number equal to 0
# (the default). A FREE C is a dual number during the fit, never an AbstractFloat, so it always
# takes the second loop and keeps its derivative, even when its value is 0 (its starting value).
#
# `eachindex(out, x)` checks ONCE, before the loop, that `out` and `x` have the same indices
# (a DimensionMismatch error otherwise), and then loops over them. That makes `@inbounds` safe:
# it skips the per-element bounds checks inside the loop, whoever calls `signal!` and with
# whatever `out`. List every per-observation field: `eachindex(out, TI, TD)` for SIR.
#
# R is computed once per call (1/T2 for the time-constant models), so the loop multiplies
# instead of dividing every TE by T2: whole fits measured ~5-15% faster than dividing.

# -------------------------------------------------------------------------------------------------
"""
    T2Decay(; TE)

Spin-echo T2 decay: `S = S0·exp(-TE/T2) + C`.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `T2` (s)
- fixed by default: `C = 0` (signal offset, e.g. the noise floor; estimate it with
  `free = (:C,)`)

For an R2 map, use `1 ./ res.params.T2`: the same least-squares fit.

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2Decay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::T2Decay) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(0.05; lower = _TMIN),
    C = Fixed(0.0),                     # signal offset
)

function signal!(out, m::T2Decay, p)
    (; TE) = m                          # unpack by name: TE = m.TE
    (; S0, T2, C) = p                   # S0 = p.S0, T2 = p.T2, C = p.C
    return _monoexp!(out, TE, S0, 1 / T2, C)    # decay rate R2 = 1/T2
end

# -------------------------------------------------------------------------------------------------
"""
    T2StarDecay(; TE)

Gradient-echo T2* decay: `S = S0·exp(-TE/T2star) + C`.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `T2star` (s)
- fixed by default: `C = 0` (signal offset; estimate it with `free = (:C,)`)

For an R2* map, use `1 ./ res.params.T2star`. No correction yet for signal loss from
macroscopic field gradients across the voxel (planned as a separate model with a sinc term).

```julia
m = T2StarDecay(TE = 0.003:0.003:0.036)
```
"""
@kwdef struct T2StarDecay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::T2StarDecay) = (
    S0 = Free(:scale; lower = 0),
    T2star = Free(0.03; lower = _TMIN),
    C = Fixed(0.0),                     # signal offset
)

function signal!(out, m::T2StarDecay, p)
    (; TE) = m
    (; S0, T2star, C) = p
    return _monoexp!(out, TE, S0, 1 / T2star, C)    # decay rate R2* = 1/T2*
end

# -------------------------------------------------------------------------------------------------
"""
    DiffusionDecay(; b)

Mono-exponential diffusion decay: `S = S0·exp(-b·ADC) + C`, one apparent diffusion coefficient
per voxel (e.g. from trace-weighted images).

- acquisition: `b` (s/mm²), one per observation; include b = 0 or not
- free by default: `S0` (the signal at b = 0), `ADC` (mm²/s; about 0.7e-3 in brain tissue,
  3e-3 in free water at body temperature)
- fixed by default: `C = 0` (signal offset; estimate it with `free = (:C,)`)

```julia
m = DiffusionDecay(b = [0, 250, 500, 750, 1000])
```
"""
@kwdef struct DiffusionDecay{A<:AbstractVector} <: MRIModel
    b::A                                # b-values (s/mm²)
end

parameters(::DiffusionDecay) = (
    S0 = Free(:scale; lower = 0),
    ADC = Free(1e-3; lower = 0),        # mm²/s
    C = Fixed(0.0),                     # signal offset
)

function signal!(out, m::DiffusionDecay, p)
    (; b) = m
    (; S0, ADC, C) = p
    return _monoexp!(out, b, S0, ADC, C)    # the ADC is already a rate (per unit b)
end
