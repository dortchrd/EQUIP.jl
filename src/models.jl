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
#   1. struct            (fields = options that are the same for every voxel)
#   2. parameters(m)     (every parameter with its default role)
#   3. signal!(out, m, acq, p)
# =================================================================================================

const _TMIN = 1e-6    # lower bound for time constants, so that t/T stays finite

# -------------------------------------------------------------------------------------------------
# Exponential decay (T2, T2*)
# -------------------------------------------------------------------------------------------------

"""
    ExpDecay()

Mono-exponential decay `S = S0·exp(-TE/T2)` (T2, T2* or any decay time constant).

- acquisition: `TE` (s)
- parameters: `S0`, `T2` (s)
"""
struct ExpDecay <: AbstractModel end    # no options, so no fields

parameters(::ExpDecay) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(0.05; lower = _TMIN),
)

function signal!(out, ::ExpDecay, acq, p)
    (; S0, T2) = p                      # unpack by name: S0 = p.S0, T2 = p.T2
    @inbounds for j in eachindex(out)
        (; TE) = obs(acq, j)            # this observation's echo time
        out[j] = S0 * exp(-TE / T2)
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# Inversion recovery (T1)
# -------------------------------------------------------------------------------------------------

"""
    InversionRecovery(; magnitude = true)

Inversion recovery with a long repetition time: `S = M0·(1 - B·exp(-TI/T1))`, its absolute
value for magnitude data. `B` absorbs imperfect inversion (B = 2 for a perfect one).

- acquisition: `TI` (s)
- parameters: `M0`, `B`, `T1` (s)
"""
struct InversionRecovery <: AbstractModel
    magnitude::Bool                     # true: fit |S| (magnitude images); false: signed S
end
InversionRecovery(; magnitude = true) = InversionRecovery(magnitude)
# The default constructor is `InversionRecovery(true)` (positional). This keyword method lets
# you write `InversionRecovery()` or `InversionRecovery(magnitude = false)`.

parameters(::InversionRecovery) = (
    M0 = Free(:scale; lower = 0),
    B = Free(2.0; lower = 0, upper = 3),
    T1 = Free(1.0; lower = _TMIN),
)

function signal!(out, m::InversionRecovery, acq, p)
    (; M0, B, T1) = p
    @inbounds for j in eachindex(out)
        (; TI) = obs(acq, j)
        s = M0 * (1 - B * exp(-TI / T1))
        out[j] = m.magnitude ? abs(s) : s
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# Saturation recovery (T1)
# -------------------------------------------------------------------------------------------------

"""
    SaturationRecovery()

Saturation recovery: `S = M0·(1 - exp(-TD/T1))`.

- acquisition: `TD` (s), the delay after saturation
- parameters: `M0`, `T1` (s)
"""
struct SaturationRecovery <: AbstractModel end

parameters(::SaturationRecovery) = (
    M0 = Free(:scale; lower = 0),
    T1 = Free(1.0; lower = _TMIN),
)

function signal!(out, ::SaturationRecovery, acq, p)
    (; M0, T1) = p
    @inbounds for j in eachindex(out)
        (; TD) = obs(acq, j)
        out[j] = M0 * (1 - exp(-TD / T1))
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# Multiple flip angle (variable flip angle) T1: the spoiled gradient echo signal
# -------------------------------------------------------------------------------------------------

"""
    MFA()

Multiple flip angle T1 mapping with a spoiled gradient echo:

    S = M0·sin(α)·(1 - E1) / (1 - cos(α)·E1),   E1 = exp(-TR/T1),   α = B1·FA

- acquisition: `FA` (degrees), `TR` (s; a scalar, or one per observation)
- parameters: `M0`, `T1` (s), and `B1` (relative flip angle), fixed at 1 by default. Pass a
  B1 map with `fixed = (B1 = b1map,)`.
"""
struct MFA <: AbstractModel end

parameters(::MFA) = (
    M0 = Free(:scale; lower = 0),
    T1 = Free(1.0; lower = _TMIN),
    B1 = Fixed(1.0),
)

function signal!(out, ::MFA, acq, p)
    (; M0, T1, B1) = p
    @inbounds for j in eachindex(out)
        (; FA, TR) = obs(acq, j)                  # TR may be one number for all observations
        sinα, cosα = sincos(B1 * deg2rad(FA))     # actual flip angle in radians
        E1 = exp(-TR / T1)
        out[j] = M0 * sinα * (1 - E1) / (1 - cosα * E1)
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# Selective inversion recovery with fast spin echo readout (SIR-FSE), two-pool qMT
# -------------------------------------------------------------------------------------------------

"""
    SIRFSE(; magnitude = true)

Selective inversion recovery with a fast spin echo readout: the two-pool (free f, macromolecular
m) magnetization transfer model, Eq. 1 of Dortch et al., Magn Reson Med 2018;80:1824. The FSE
readout saturates both pools; they recover for `TD`; the free pool is inverted; the signal is
the free pool's magnetization after `TI`.

- acquisition: `TI`, `TD` (s), vectors or (TD) one number for all observations
- free by default: `PSR` (pool size ratio M0m/M0f), `R1f` (1/s), `Sf` (free-pool inversion
  efficiency, -1 = perfect inversion), `M0f`
- fixed by default: `kmf = 12.5` 1/s (exchange rate m→f; estimate it with `free = (:kmf,)`),
  `R1m` tied to `R1f`, `Sm = 0.83` (macromolecular saturation for a 1-ms inversion pulse)
"""
struct SIRFSE <: AbstractModel
    magnitude::Bool
end
SIRFSE(; magnitude = true) = SIRFSE(magnitude)

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
function signal!(M, m::SIRFSE, acq, p)
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

    @inbounds for j in eachindex(M)
        (; TI, TD) = obs(acq, j)
        E_td⁺, E_td⁻ = exp(-R1⁺ * TD), exp(-R1⁻ * TD)
        E_ti⁺, E_ti⁻ = exp(-R1⁺ * TI), exp(-R1⁻ * TI)

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
        M[j] = m.magnitude ? abs(s) : s
    end
    return M
end
