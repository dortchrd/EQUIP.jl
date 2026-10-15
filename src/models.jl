# =================================================================================================
# Built-in models
#
# Units: all times in SECONDS (TE, TI, TD, TR) and rates in 1/s. The default starting values
# below assume this (e.g. T1 starts at 1 s). Flip angles are in degrees.
#
# Default starting values are fixed numbers (no model-specific, data-driven guesses yet), except
# for the parameter that scales the whole signal (S0, M0, M0f): it is declared `Free(:scale)` and
# starts, voxel by voxel, at the least-squares scale given the other starting values. So raw
# data (values in the hundreds or thousands) need no normalization. Override any start with the
# `start` keyword (numbers or per-voxel maps).
#
# Each model follows the same pattern:
#
#   1. @kwdef struct    fields = the acquisition (one value per observation) and options.
#                       `@kwdef` gives the keyword constructor, e.g. `SIRFSE(TI = ..., TD = ...)`.
#                       Each vector field gets its own type parameter (A, B, ...), so a field can
#                       be any kind of vector (Vector, range, ...) and Julia still knows its
#                       exact type. `init` converts them to plain float vectors for fitting.
#   2. fill method      (optional) lets one number stand for every observation, e.g. one TR.
#   3. parameters(m)    every parameter with its default role.
#   4. signal!(out, m, p)
#                       unpack the model and the parameters by name, then loop over observations.
# =================================================================================================

const _TMIN = 1e-6    # lower bound for time constants, so that t/T stays finite

# -------------------------------------------------------------------------------------------------
# Exponential decay (T2, T2*)
# -------------------------------------------------------------------------------------------------

"""
    ExpDecay(; TE)

Mono-exponential decay `S = S0·exp(-TE/T2)` (T2, T2* or any decay time constant).

- acquisition: `TE` (s), one per observation
- parameters: `S0`, `T2` (s)

```julia
m = ExpDecay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct ExpDecay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::ExpDecay) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(0.05; lower = _TMIN),
)

function signal!(out, m::ExpDecay, p)
    (; TE) = m                          # unpack by name: TE = m.TE
    (; S0, T2) = p                      # S0 = p.S0, T2 = p.T2
    @inbounds for j in eachindex(out)
        out[j] = S0 * exp(-TE[j] / T2)
    end
    return out
end
# `@inbounds` skips index bounds checks inside the loop (safe: `init` checked the lengths).

# -------------------------------------------------------------------------------------------------
# Inversion recovery (T1)
# -------------------------------------------------------------------------------------------------

"""
    InversionRecovery(; TI, magnitude = true)

Inversion recovery with a long repetition time: `S = M0·(1 - B·exp(-TI/T1))`, its absolute
value for magnitude data. `B` absorbs imperfect inversion (B = 2 for a perfect one).

- acquisition: `TI` (s), one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- parameters: `M0`, `B`, `T1` (s)
"""
@kwdef struct InversionRecovery{A<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    magnitude::Bool = true              # a default value makes the keyword optional
end

parameters(::InversionRecovery) = (
    M0 = Free(:scale; lower = 0),
    B = Free(2.0; lower = 0, upper = 3),
    T1 = Free(1.0; lower = _TMIN),
)

function signal!(out, m::InversionRecovery, p)
    (; TI, magnitude) = m
    (; M0, B, T1) = p
    @inbounds for j in eachindex(out)
        s = M0 * (1 - B * exp(-TI[j] / T1))
        out[j] = magnitude ? abs(s) : s
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# Saturation recovery (T1)
# -------------------------------------------------------------------------------------------------

"""
    SaturationRecovery(; TD)

Saturation recovery: `S = M0·(1 - exp(-TD/T1))`.

- acquisition: `TD` (s), the delay after saturation, one per observation
- parameters: `M0`, `T1` (s)
"""
@kwdef struct SaturationRecovery{A<:AbstractVector} <: MRIModel
    TD::A                               # delays after saturation (s)
end

parameters(::SaturationRecovery) = (
    M0 = Free(:scale; lower = 0),
    T1 = Free(1.0; lower = _TMIN),
)

function signal!(out, m::SaturationRecovery, p)
    (; TD) = m
    (; M0, T1) = p
    @inbounds for j in eachindex(out)
        out[j] = M0 * (1 - exp(-TD[j] / T1))
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# Multiple flip angle (variable flip angle) T1: the spoiled gradient echo signal
# -------------------------------------------------------------------------------------------------

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
    @inbounds for j in eachindex(out)
        sinα, cosα = sincos(B1 * deg2rad(FA[j]))   # actual flip angle in radians
        E1 = exp(-TR[j] / T1)
        out[j] = M0 * sinα * (1 - E1) / (1 - cosα * E1)
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# Selective inversion recovery with fast spin echo readout (SIR-FSE), two-pool qMT
# -------------------------------------------------------------------------------------------------

"""
    SIRFSE(; TI, TD, magnitude = true)

Selective inversion recovery with a fast spin echo readout: the two-pool (free f, macromolecular
m) magnetization transfer model, Eq. 1 of Dortch et al., Magn Reson Med 2018;80:1824. The FSE
readout saturates both pools; they recover for `TD`; the free pool is inverted; the signal is
the free pool's magnetization after `TI`.

- acquisition: `TI` (s), one per observation; `TD` (s), one number for all observations or
  one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- free by default: `PSR` (pool size ratio M0m/M0f), `R1f` (1/s), `Sf` (free-pool inversion
  efficiency, -1 = perfect inversion), `M0f`
- fixed by default: `kmf = 12.5` 1/s (exchange rate m→f; estimate it with `free = (:kmf,)`),
  `R1m` tied to `R1f`, `Sm = 0.83` (macromolecular saturation for a 1-ms inversion pulse)

```julia
m = SIRFSE(TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])
m = SIRFSE(TI = [0.010, 0.050, 0.277, 1.0], TD = 2.5)        # one TD for all
```
"""
@kwdef struct SIRFSE{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
end

# One TD for every TI (see the MFA fill method above).
SIRFSE(TI, TD::Real, magnitude) = SIRFSE(TI, fill(TD, length(TI)), magnitude)

parameters(::SIRFSE) = (
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),
    kmf = Fixed(12.5),
    R1m = Fixed(p -> p.R1f),            # tie: R1m equals R1f (the paper's assumption)
    Sm = Fixed(0.83),
)

# The two pools exchange magnetization, so after a perturbation each pool recovers as a sum of
# two exponentials with rates R1⁺ and R1⁻ (the eigenvalues of the exchange matrix). Below:
#   1. the two rates (same for every observation),
#   2. the amplitudes of recovery from saturation over TD,
#   3. per observation: both pools after TD, then inversion, then the free pool after TI.
function signal!(out, m::SIRFSE, p)
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf, R1m, Sm) = p
    kfm = kmf * PSR                                   # exchange rate f→m (mass balance)

    # 1. Rates of the two exponentials.
    ΔR1 = sqrt((R1f - R1m + kfm - kmf)^2 + 4 * kfm * kmf)
    R1⁺ = (R1f + R1m + kfm + kmf + ΔR1) / 2
    R1⁻ = R1⁺ - ΔR1
    invΔR1 = 1 / ΔR1

    # 2. Recovery from saturation (both pools start at 0): amplitudes of the two exponentials.
    bf_td⁺ = -(R1f - R1⁻) * invΔR1                    # free pool
    bf_td⁻ = (R1f - R1⁺) * invΔR1
    bm_td⁺ = -(R1m - R1⁻) * invΔR1                    # macromolecular pool
    bm_td⁻ = (R1m - R1⁺) * invΔR1

    @inbounds for j in eachindex(out)
        E_td⁺, E_td⁻ = exp(-R1⁺ * TD[j]), exp(-R1⁻ * TD[j])
        E_ti⁺, E_ti⁻ = exp(-R1⁺ * TI[j]), exp(-R1⁻ * TI[j])

        # 3a. Each pool after recovering for TD (relative to its equilibrium value).
        Mf_td = bf_td⁺ * E_td⁺ + bf_td⁻ * E_td⁻ + 1
        Mm_td = bm_td⁺ * E_td⁺ + bm_td⁻ * E_td⁻ + 1

        # 3b. The inversion scales the pools by Sf and Sm; then both recover for TI. These are
        #     the amplitudes of the free pool's two exponentials during TI.
        bf1 = Sf * Mf_td - 1
        bf2 = (Sf * Mf_td - Sm * Mm_td) * kfm
        bf_ti⁺ = (bf1 * (R1f - R1⁻) + bf2) * invΔR1
        bf_ti⁻ = -(bf1 * (R1f - R1⁺) + bf2) * invΔR1

        # 3c. The free pool's magnetization at TI.
        s = (bf_ti⁺ * E_ti⁺ + bf_ti⁻ * E_ti⁻ + 1) * M0f
        out[j] = magnitude ? abs(s) : s
    end
    return out
end
