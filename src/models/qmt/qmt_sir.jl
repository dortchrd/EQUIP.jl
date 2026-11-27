# =================================================================================================
# Selective inversion recovery (SIR), two-pool quantitative magnetization transfer (qMT)
#
#   QMTSIR          the general model: any R1m, any Mz at the start of TD (Li's closed form)
#   QMTSIREqualR1   its two default assumptions built in (R1m = R1f, both pools saturated at the
#                   start of TD): the signal in one line
#
# References (the equation numbers in the comments below refer to these):
#   Li et al., Magn Reson Med 2010;64:491-500      closed-form solution, Eqs. 2-6 (used here)
#   Dortch et al., Magn Reson Med 2018;80:1824-1835 the same model in matrix form, Eq. 1
# =================================================================================================

"""
    QMTSIR(; TI, TD, magnitude = true, pulse = nothing)

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
- option: `pulse`, the inversion pulse (an [`RFPulse`](@ref), e.g. `blockpulse(180, 1e-3)`):
  given, `Sm` is not a parameter but made from it, `saturationefficiency(pulse, T2m; B1, Δ = B0)`,
  and `T2m` (15 µs), `B1` (1) and `B0` (0 Hz) take its place, fixed: give the maps
  (`fit(m, data; B1 = b1map)`), and their errors reach crlb, montecarlo and `leakage` by name
  (`uncertain = (B1 = 0.03,)`). Sf stays fitted
- free by default: `PSR` (pool size ratio M0m/M0f), `R1f` (1/s), `Sf` (free-pool inversion
  efficiency, -1 = perfect inversion), `M0f`
- fixed by default: `kmf = 12.5` 1/s (exchange rate m→f; estimate it with `kmf = Free()`, ≥ 0),
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
@kwdef struct QMTSIR{A<:AbstractVector,B<:AbstractVector,P<:Union{Nothing,RFPulse}} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
    pulse::P = nothing                  # the inversion pulse: given, Sm is made from it (below)
end

# One TD for every TI (the same pattern as the single-TR method of T1MFA).
QMTSIR(TI, TD::Real, magnitude, pulse = nothing) = QMTSIR(TI, fill(TD, length(TI)), magnitude, pulse)

parameters(::Type{<:QMTSIR}) = (
    PSR = Free(start = 0.1, lower = 0, upper = 1),
    R1f = Free(start = 1.0, lower = 0.05, upper = 10),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    M0f = Free(start = :scale, lower = 0),
    kmf = Fixed(12.5; lower = 0),       # an exchange rate: ≥ 0 when estimated (kmf = Free())
    R1m = Fixed(:R1f),                  # tie: R1m equals R1f
    Sm = Fixed(0.83),
    Mzf_td0 = Fixed(0.0),               # each pool's Mz at the start of TD, as a fraction of its
    Mzm_td0 = Fixed(0.0),               # equilibrium value: 0 = saturated by the readout
)

# With the inversion pulse given (`pulse = blockpulse(180, 1e-3)`), Sm is not a parameter: it is
# the pulse's saturation of the macromolecular pool, `saturationefficiency(pulse, T2m; B1, Δ = B0)`
# (pulses.jl: exp(−π·g(Δ)·B1²·∫ω1²dt), the Gaussian lineshape on resonance), and T2m, B1 and B0
# take its place, fixed: maps, whose errors then reach the fit by name (`leakage`, crlb's
# `uncertain = (B1 = 0.03,)`). Sf stays fitted: it takes the free pool's part of a B1 or B0 error.
# T2m = 15 µs gives Sm = 0.831 for the 1-ms block, Dortch et al. 2018's 0.83. (A model type's
# parameters can depend on its type parameters: this method is the more specific one for a
# QMTSIR whose third type parameter, the pulse's, is an RFPulse. Written with the other two's
# bounds, `<:AbstractVector`: with `<:Any`, or as `QMTSIR{A,B,RFPulse} where {A,B}`, Julia ranks
# the general method above it.)
parameters(::Type{<:QMTSIR{<:AbstractVector,<:AbstractVector,RFPulse}}) = (
    PSR = Free(start = 0.1, lower = 0, upper = 1),
    R1f = Free(start = 1.0, lower = 0.05, upper = 10),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    M0f = Free(start = :scale, lower = 0),
    kmf = Fixed(12.5; lower = 0),
    R1m = Fixed(:R1f),
    T2m = Fixed(15e-6; lower = 0),      # s: the macromolecular pool's T2, for Sm
    B1 = Fixed(1.0; lower = 0),         # relative: scales the inversion pulse (Sm)
    B0 = Fixed(0.0),                    # Hz: the pulse's offset from the resonance (Sm)
    Mzf_td0 = Fixed(0.0),
    Mzm_td0 = Fixed(0.0),
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
    (; PSR, R1f, Sf, M0f, kmf, R1m, Mzf_td0, Mzm_td0) = p
    Sm = _sirSm(m, p)                                   # the parameter, or made from the pulse
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
  (`fit(...; kmf = Free())`); Dortch et al., Magn Reson Med 2018;80:1824, Table 2
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
# A docstring documents the one expression right after it, here SIR_5POINT; the other two take
# the same text (`@doc` with the docstring of SIR_5POINT), so `?SIR_4POINT` shows it too.
@doc (@doc SIR_5POINT) SIR_4POINT
@doc (@doc SIR_5POINT) SIR_16POINT


"""
    QMTSIREqualR1(; TI, TD, magnitude = true, pulse = nothing)

[`QMTSIR`](@ref) with its two default assumptions built in, so that the signal is one line:
the macromolecular pool relaxes as the free pool does (R1m = R1f), and the readout leaves both
pools saturated (Mz = 0 at the start of TD). Then, with three exponentials per observation,

    S = M0f·{1 + E_i·[(1 − E_d)·(a + b·E_k) − 1]}
    E_i = e^(−R1f·TI),  E_d = e^(−R1f·TD),  E_k = e^(−k·TI)
    k = kmf·(1 + PSR),  a = (Sf + PSR·Sm)/(1 + PSR),  b = PSR·(Sf − Sm)/(1 + PSR)  (a + b = Sf)

This is inversion recovery with T1 = 1/R1f ([`T1IRTD`](@ref), Li et al. 2010 Eq. 1), except that
the inversion efficiency is not constant: it is a + b·E_k, which during TI relaxes from Sf
toward a, the pools' weighted mean, at the rate k = kfm + kmf; b is the part that exchange
takes away. With PSR = 0, or Sf = Sm, it is inversion recovery. During TD both pools recover
together, at R1f, so exchange plays no part there: the magnetization transfer is seen only
through TI.

The same signal as `QMTSIR` with its defaults, to rounding (checked against Dortch et al.'s
matrix form), at half to two thirds of the cost per evaluation, and defined at kmf = 0
(inversion recovery; `QMTSIR` divides by zero there). For data where those assumptions do not
hold (R1m fitted or fixed apart from R1f, Mz left by the readout): `QMTSIR`.

- acquisition: `TI` (s), one per observation; `TD` (s), one number for all observations or
  one per observation; the SIR protocols apply (`QMTSIREqualR1(; SIR_5POINT...)`)
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- option: `pulse`, the inversion pulse (an [`RFPulse`](@ref), e.g. `blockpulse(180, 1e-3)`):
  given, `Sm` is not a parameter but made from it, `saturationefficiency(pulse, T2m; B1, Δ = B0)`,
  and `T2m` (15 µs), `B1` (1) and `B0` (0 Hz) take its place, fixed: give the maps
  (`fit(m, data; B1 = b1map)`), and their errors reach crlb, montecarlo and `leakage` by name
  (`uncertain = (B1 = 0.03,)`). Sf stays fitted
- free by default: `PSR` (pool size ratio M0m/M0f), `R1f` (1/s), `Sf` (free-pool inversion
  efficiency, -1 = perfect inversion), `M0f`
- fixed by default: `kmf = 12.5` 1/s (exchange rate m→f; estimate it with `kmf = Free()`, ≥ 0),
  `Sm = 0.83` (macromolecular saturation for a 1-ms inversion pulse), as `QMTSIR`

```julia
m = QMTSIREqualR1(; SIR_4POINT...)
r = fit(m, data; mask)                                    # kmf fixed at 12.5
r = fit(QMTSIREqualR1(; SIR_5POINT...), data; mask, kmf = Free())    # kmf estimated
```
"""
@kwdef struct QMTSIREqualR1{A<:AbstractVector,B<:AbstractVector,P<:Union{Nothing,RFPulse}} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
    pulse::P = nothing                  # the inversion pulse: given, Sm is made from it (as QMTSIR's)
end

QMTSIREqualR1(TI, TD::Real, magnitude, pulse = nothing) = QMTSIREqualR1(TI, fill(TD, length(TI)), magnitude, pulse)

# QMTSIR's parameters without R1m (= R1f) and the start values of TD (0), with the same roles.
parameters(::Type{<:QMTSIREqualR1}) = (
    PSR = Free(start = 0.1, lower = 0, upper = 1),
    R1f = Free(start = 1.0, lower = 0.05, upper = 10),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    M0f = Free(start = :scale, lower = 0),
    kmf = Fixed(12.5; lower = 0),       # ≥ 0 when estimated, as QMTSIR's
    Sm = Fixed(0.83),
)
# With the inversion pulse: Sm made from it, T2m, B1 and B0 in its place (as QMTSIR's, above)
parameters(::Type{<:QMTSIREqualR1{<:AbstractVector,<:AbstractVector,RFPulse}}) = (
    PSR = Free(start = 0.1, lower = 0, upper = 1),
    R1f = Free(start = 1.0, lower = 0.05, upper = 10),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    M0f = Free(start = :scale, lower = 0),
    kmf = Fixed(12.5; lower = 0),
    T2m = Fixed(15e-6; lower = 0),
    B1 = Fixed(1.0; lower = 0),
    B0 = Fixed(0.0),
)

# Sm: the parameter, without a pulse; with one, the pulse's saturation of the macromolecular pool
# (the type parameter says which when compiling, so neither model pays for the other's case)
@inline _sirSm(::Union{QMTSIR{<:AbstractVector,<:AbstractVector,Nothing},
                        QMTSIREqualR1{<:AbstractVector,<:AbstractVector,Nothing}}, p) = p.Sm
@inline _sirSm(m::Union{QMTSIR,QMTSIREqualR1}, p) = saturationefficiency(m.pulse, p.T2m; B1 = p.B1, Δ = p.B0)

# Where the one line comes from (Li et al. 2010, in its notation; derived 2026-11-20).
#
# 1. Li's Eq. 2 for each pool's departure from equilibrium, x = Mf/Mf∞ − 1 and y = Mm/Mm∞ − 1,
#    with p = PSR (so kfm = p·kmf) and R1f = R1m = R1:
#        dx/dt = −R1·x − kfm·(x − y),    dy/dt = −R1·y − kmf·(y − x)
#    i.e. dz/dt = −(R1·I + K)·z, with K the exchange terms alone. R1·I commutes with K, so
#    relaxation and exchange separate: K's eigenvalues are 0 and k = kfm + kmf = kmf·(1 + p),
#    and Li's Eq. 4 rates become R1⁻ = R1 and R1⁺ = R1 + k (its square root is kfm + kmf).
# 2. In the coordinates that separate them:
#        u = (x + p·y)/(1 + p), the pools' weighted mean: exchange conserves it, u(t) = u₀·e^(−R1·t)
#        d = x − y, their difference:                                       d(t) = d₀·e^(−(R1 + k)·t)
#    and back, x = u + p/(1 + p)·d: Li's Eq. 3 with bf⁻ = u₀ and bf⁺ = p/(1 + p)·d₀.
# 3. TD (Li's Eq. 5): both pools start saturated, x₀ = y₀ = −1, so d₀ = 0 and u₀ = −1: both
#    recover as 1 − e^(−R1·TD), together, with no exchange.
# 4. The inversion (Li's Eq. 6), with E_d = e^(−R1·TD): x₀ = Sf·(1 − E_d) − 1 and
#    y₀ = Sm·(1 − E_d) − 1, so d₀ = (Sf − Sm)·(1 − E_d) and u₀ = (Sf + p·Sm)/(1 + p)·(1 − E_d) − 1.
# 5. At TI, Mf/Mf∞ = 1 + u + p/(1 + p)·d. With a = (Sf + p·Sm)/(1 + p) and b = p·(Sf − Sm)/(1 + p),
#    u = a·(1 − E_d) − 1 and p/(1 + p)·d = b·(1 − E_d), so Mf/Mf∞ = 1 + E_i·[(1 − E_d)·(a + b·E_k) − 1]
#    (E_i·E_k = e^(−(R1 + k)·TI), the fast rate, needs no exponential of its own). The factored
#    form is the owner's (2026-11-20), as are a and b.
#
# Checked against the matrix form (Dortch et al. 2018, Eq. 1) and QMTSIR's closed form in
# test/unit/models.jl. Measured 2026-11-20 (one signal, 4 and 16 points, plain and with the fit's
# dual numbers): k, a and b computed once, before the loop, and each exponential once, as here,
# about 0.55–0.7× QMTSIR's general form (4 points: 47 against 70 ns plain, 77 against 140 with
# dual numbers); the same equation written out in one line, every exponential where it appears,
# was up to 1.6× slower (the compiler does not merge the repeated e^(−R1f·TI) or move the
# constants out). Other forms of the same arithmetic measured within the noise. `1 − E_d` is not
# `-expm1(-R1f·TD)`: that keeps 1 − E_d's relative precision when R1f·TD is small, but the signal
# adds it to terms of order 1, so its error is a rounding of M0f either way, and expm1 cost 5–15%
# with dual numbers.
function signal!(out, m::QMTSIREqualR1, p)
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf) = p
    Sm = _sirSm(m, p)                       # the parameter, or made from the pulse
    k = kmf * (1 + PSR)                     # kfm + kmf: the rate at which exchange evens out the pools
    a = (Sf + PSR * Sm) / (1 + PSR)         # the pools' weighted mean inversion, where it relaxes to
    b = PSR * (Sf - Sm) / (1 + PSR)         # the part that exchange takes away (a + b = Sf)
    @inbounds for j in eachindex(out, TI, TD)
        Ei, Ed, Ek = exp(-R1f * TI[j]), exp(-R1f * TD[j]), exp(-k * TI[j])
        s = M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
        out[j] = magnitude ? abs(s) : s     # magnitude images lose the sign
    end
    return out
end
