# =================================================================================================
# Selective inversion recovery (SIR), two-pool quantitative magnetization transfer (qMT)
#
# References (the equation numbers in the comments below refer to these):
#   Li et al., Magn Reson Med 2010;64:491-500      closed-form solution, Eqs. 2-6 (used here)
#   Dortch et al., Magn Reson Med 2018;80:1824-1835 the same model in matrix form, Eq. 1
# =================================================================================================

"""
    QMTSIR(; TI, TD, magnitude = true)

Selective inversion recovery (SIR), a quantitative magnetization transfer (qMT) model with two
pools that exchange magnetization: free water (f) and macromolecules (m). For each observation:

1. the readout leaves both pools saturated, Mz = 0 (e.g. the fast spin echo train in SIR-FSE,
   or a saturation pulse);
2. both pools recover for `TD`;
3. a low-power inversion pulse inverts the free pool and barely affects the macromolecular pool;
4. both pools recover for `TI`; the signal is the free pool's Mz.

In matrix form, with Mz = [Mzf, Mzm] (Dortch et al., Magn Reson Med 2018, Eq. 1):

    Mz(TI, TD) = { exp(A·TI)·S·[I - exp(A·TD)] + [I - exp(A·TI)] }·M0

    A  = [ -(R1f + kfm)       kmf      ]     relaxation and exchange; kfm = kmf·PSR
         [      kfm      -(R1m + kmf)  ]
    S  = diag(Sf, Sm)                         the inversion pulse's effect on each pool
    M0 = [M0f, PSR·M0f]                       equilibrium magnetization
    I  = the 2×2 identity matrix

Step 1 is an assumption. `Mzf_td0` and `Mzm_td0` relax it: each pool's Mz at the start of TD, as
a fraction of its equilibrium value (default 0). Then `[I - exp(A·TD)]` above becomes
`[I - exp(A·TD)·(I - Z)]` with `Z = diag(Mzf_td0, Mzm_td0)`. Use them to test how sensitive the
fitted parameters are to the assumption, e.g. simulate with `Mzf_td0 = 0.05` and fit with the
default.

- acquisition: `TI` (s), one per observation; `TD` (s), one number for all observations or
  one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- free by default: `PSR` (pool size ratio M0m/M0f), `R1f` (1/s), `Sf` (free-pool inversion
  efficiency, -1 = perfect inversion), `M0f`
- fixed by default: `kmf = 12.5` 1/s (exchange rate m→f; estimate it with `free = (:kmf,)`),
  `R1m` tied to `R1f`, `Sm = 0.83` (macromolecular saturation for a 1-ms inversion pulse),
  `Mzf_td0 = Mzm_td0 = 0`

`kmf` must be > 0. With `kmf = 0` there is no exchange: PSR has no effect on the signal, and the
two recovery rates are equal, so the closed-form solution divides by zero and the signal is NaN.
To simulate no exchange, use a tiny value such as `kmf = 1e-12`: it gives single-pool inversion
recovery (accurate down to at least `kmf = 1e-100`).

```julia
m = QMTSIR(TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])
m = QMTSIR(TI = [0.010, 0.050, 0.277, 1.0], TD = 2.5)        # one TD for all
```
"""
@kwdef struct QMTSIR{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
end

# One TD for every TI (the same pattern as the single-TR method of T1MFA).
QMTSIR(TI, TD::Real, magnitude) = QMTSIR(TI, fill(TD, length(TI)), magnitude)

parameters(::QMTSIR) = (
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),
    kmf = Fixed(12.5),
    R1m = Fixed(:R1f),                  # tie: R1m equals R1f
    Sm = Fixed(0.83),
    Mzf_td0 = Fixed(0.0),               # each pool's Mz at the start of TD, as a fraction of its
    Mzm_td0 = Fixed(0.0),               # equilibrium value: 0 = saturated by the readout
)

# How the code computes the signal (Li et al. 2010, without matrices).
#
# Work with each pool's Mz as a fraction of its equilibrium value (M0f; Li calls it Mf∞), so 1 is
# equilibrium, 0 saturated and -1 inverted. With no RF pulses, each pool recovers from a starting
# value Mf(0) as a sum of two exponentials, with a fast rate R1⁺ and a slow rate R1⁻ (Li Eq. 3):
#
#     Mf(t) = bf⁺·exp(-R1⁺·t) + bf⁻·exp(-R1⁻·t) + 1
#
# The amplitudes depend on where BOTH pools start, because they exchange magnetization (Li Eq. 4):
#
#     bf⁺ = [ (Mf(0) - 1)·(R1f - R1⁻) + (Mf(0) - Mm(0))·kfm ] / (R1⁺ - R1⁻)
#     bf⁻ = (Mf(0) - 1) - bf⁺          (Eq. 3 at t = 0: bf⁺ + bf⁻ + 1 = Mf(0))
#
# The macromolecular pool's are the same with f and m swapped. The signal applies these twice:
# recovery over TD (from Mzf_td0, Mzm_td0), the inversion pulse (Li Eq. 6: Mf → Sf·Mf,
# Mm → Sm·Mm), then recovery over TI.
#
# Li writes bf⁻ in full, like bf⁺; using bf⁺ + bf⁻ = Mf(0) - 1 gives the same value with fewer
# operations (whole fits measured ~7% faster). Checked against the matrix form above, a numerical
# solution of Li's Eq. 2, and finite-difference derivatives (see test/runtests.jl).
function signal!(out, m::QMTSIR, p)
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf, R1m, Sm, Mzf_td0, Mzm_td0) = p
    kfm = kmf * PSR                                     # exchange rate f→m (mass balance)

    # 1. The fast and slow recovery rates (Li Eq. 4; -R1⁺ and -R1⁻ are the eigenvalues of A).
    ΔR1 = sqrt((R1f - R1m + kfm - kmf)^2 + 4 * kfm * kmf)    # R1⁺ - R1⁻
    R1⁺ = (R1f + R1m + kfm + kmf + ΔR1) / 2
    R1⁻ = R1⁺ - ΔR1
    invΔR1 = 1 / ΔR1                                    # multiply instead of dividing in the loop

    # 2. Amplitudes for the recovery over TD (Li Eq. 4), from the start values. They are the same
    #    for every observation. With the default start values 0 they reduce to Li Eq. 5.
    bf_td⁺ = ((Mzf_td0 - 1) * (R1f - R1⁻) + (Mzf_td0 - Mzm_td0) * kfm) * invΔR1
    bf_td⁻ = (Mzf_td0 - 1) - bf_td⁺
    bm_td⁺ = ((Mzm_td0 - 1) * (R1m - R1⁻) + (Mzm_td0 - Mzf_td0) * kmf) * invΔR1
    bm_td⁻ = (Mzm_td0 - 1) - bm_td⁺

    @inbounds for j in eachindex(out, TI, TD)
        E_td⁺, E_td⁻ = exp(-R1⁺ * TD[j]), exp(-R1⁻ * TD[j])
        E_ti⁺, E_ti⁻ = exp(-R1⁺ * TI[j]), exp(-R1⁻ * TI[j])

        # 3a. Both pools at the end of TD (Li Eq. 3).
        Mzf_td = bf_td⁺ * E_td⁺ + bf_td⁻ * E_td⁻ + 1
        Mzm_td = bm_td⁺ * E_td⁺ + bm_td⁻ * E_td⁻ + 1

        # 3b. The inversion pulse scales each pool (Li Eq. 6): Sf ≈ -1 inverts the free pool,
        #     Sm ≈ 0.83 partly saturates the macromolecular pool.
        Mzf_inv = Sf * Mzf_td
        Mzm_inv = Sm * Mzm_td

        # 3c. Amplitudes for the free pool's recovery over TI, starting from just after the pulse
        #     (Li Eq. 4). Only the free pool is measured, so the macromolecular pool's are not needed.
        bf_ti⁺ = ((Mzf_inv - 1) * (R1f - R1⁻) + (Mzf_inv - Mzm_inv) * kfm) * invΔR1
        bf_ti⁻ = (Mzf_inv - 1) - bf_ti⁺

        # 3d. The free pool at TI (Li Eq. 3), still as a fraction of its equilibrium value.
        Mzf_ti = bf_ti⁺ * E_ti⁺ + bf_ti⁻ * E_ti⁻ + 1

        # 4. The signal: the free pool's Mz at TI in signal units. M0f is its equilibrium Mz and also
        #    absorbs everything else that scales the signal (coil sensitivity, receiver gain, T2
        #    decay during the readout), so it is in arbitrary units.
        s = M0f * Mzf_ti
        out[j] = magnitude ? abs(s) : s                 # magnitude images lose the sign
    end
    return out
end

"""
    SIR_5POINT, SIR_4POINT, SIR_16POINT

SIR-FSE sampling schemes (TI and TD in s), e.g. `QMTSIR(; SIR_5POINT...)`:

- `SIR_5POINT`: optimized for the unconstrained model, kmf estimated
  (`fit(...; free = (:kmf,))`); Dortch et al., Magn Reson Med 2018;80:1824, Table 2
- `SIR_4POINT`: optimized for the constrained model, kmf fixed at 12.5 s⁻¹ (QMTSIR's default);
  same source
- `SIR_16POINT`: the earlier, longer scheme those were compared with: 15 TIs log-spaced from
  10 ms to 2 s and rounded to the ms (0.010, 0.015, 0.021, 0.031, 0.045, 0.066, 0.097, 0.141,
  0.206, 0.301, 0.440, 0.643, 0.938, 1.370, 2.000 s), then TI = 10 s, all with TD = 2.5 s;
  kmf estimated. Described in Dortch et al. 2018, Methods
"""
const SIR_5POINT = (TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])
const SIR_4POINT = (TI = [0.010, 0.010, 0.278, 1.007], TD = [0.684, 4.171, 2.730, 0.010])
const SIR_16POINT = (TI = [round.(exp.(range(log(0.010), log(2.0); length = 15)); digits = 3); 10.0],
                     TD = fill(2.5, 16))
