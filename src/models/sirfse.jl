# =================================================================================================
# Selective inversion recovery with fast spin echo readout (SIR-FSE), two-pool qMT
# =================================================================================================

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

# One TD for every TI (the same pattern as the single-TR method in mfa.jl).
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
