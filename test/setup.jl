# =================================================================================================
# test/setup.jl: what every test file uses: the packages, the test models (a model for
# `datastart`, a probe for subnormal flushing, @mrimodel versions of built-in models, a
# directional model written by hand), the zero-allocation check `@test_noalloc`, the shared
# cases (CASES: every built-in model with true values for every parameter) and protocols, and
# two independent references: SIR's matrix form (Dortch et al., MRM 2018, Eq. 1) and
# Gauss–Legendre nodes for spherical means. Included once, by test/runtests.jl.
# =================================================================================================
using EQUIP
using EQUIP: setup, freenames      # internal functions that the tests inspect
using LinearAlgebra
using StaticArrays: SVector, SMatrix
using Random
using Statistics: mean, std, var, cov, median
using Test
import NIfTI, JSON                  # io tests: NIfTI files with a chosen geometry; sidecars
import SpecialFunctions, ForwardDiff   # SMT: erf, and derivatives of the stick's spherical mean
using MAT                           # io tests: MATLAB files (MRImages' extension)
using EQUIP.MRImages: samegeometry, readmat, writemat

# A model for testing `datastart`: T2 starts at 0.05·c, from the fixed parameter c (as DESPOT1
# needs the voxel's B1), and at NaN when c < 0, which must make that voxel's fit fail.
struct DataStartTest <: MRIModel
    t::Vector{Float64}
end
EQUIP.parameters(::DataStartTest) = (S0 = Free(start = :scale, lower = 0),
                                     T2 = Free(start = :data, lower = 1e-6), c = Fixed(1.0))
function EQUIP.signal!(out, m::DataStartTest, p)
    for j in eachindex(out, m.t)
        out[j] = p.S0 * abs(p.c) * exp(-m.t[j] / p.T2)
    end
    return out
end
EQUIP.datastart(::DataStartTest, y, p) = (T2 = p.c < 0 ? NaN : 0.05 * p.c,)

# A model with a parameter named like a keyword of `fit` (`weights`): refused, with a message.
struct WeightsNamed <: MRIModel
    t::Vector{Float64}
end
EQUIP.parameters(::WeightsNamed) = (S0 = Free(start = :scale), weights = Free(start = 1.0))
EQUIP.signal!(out, m::WeightsNamed, p) = (out .= p.S0 .* exp.(-m.t .* p.weights); out)

# The bytes one `fit!` allocates, measured inside a function (from top-level code Julia 1.11
# counts the call's boxing too, 128 bytes; 1.13 does not).
fitallocs(c, y) = @allocated fit!(c, y)
# Zero-allocation checks, `@test_noalloc expr`: a `@test` (the same expression, the same line in
# a failure's message), skipped when bounds checks are forced on everywhere. `Pkg.test` on Julia
# 1.11 starts the tests with --check-bounds=yes (1.13's no longer does), which turns off the
# code's `@inbounds`, and the allocation-free paths then allocate: CI's first run on 1.11.9
# failed 48 of these checks (fit! 96–576 bytes per fit by model, crlb's kernel, the design's
# d(x), leakage's kernel, up to 6560 bytes), and on 1.13 with the flag fit! allocates too, for
# QMTSIR (576) and R1IRTD in Float32 (192). Users run with the bounds checks the code sets, where
# fit! allocates nothing on 1.11 and 1.13 alike (measured 2026-11-08), so the allocations are
# checked only then; the rest of the suite, the bounds checks included, runs either way.
const BOUNDS_FORCED = Base.JLOptions().check_bounds == 1
macro test_noalloc(ex)
    t = Expr(:macrocall, Symbol("@test"), __source__, ex)
    s = Expr(:macrocall, Symbol("@test_skip"), __source__, ex)
    return esc(:($BOUNDS_FORCED ? $s : $t))
end

# A T2 decay that counts its evaluations with subnormal numbers flushed to zero, and without.
# Dataset fits call it from several threads at once, so the counts are atomic: not a lock, which
# could make a task yield when contended and resume on another thread, whose mode is not the one
# its chunk set (the mode belongs to the thread; the fit assumes a chunk never yields). With a
# lock, CI's first run (Julia 1.13.3) failed here, and this machine in 1–2 of 50 Float32 checks
# and 13–14 of 50 Float64 ones (4 and 16 threads); with the counts, 0 of 100 (2026-11-08).
const NFLUSHED = Threads.Atomic{Int}(0)
const NKEPT = Threads.Atomic{Int}(0)
countreset!() = (NFLUSHED[] = 0; NKEPT[] = 0; nothing)
struct FlushProbe <: MRIModel
    TE::Vector{Float64}
end
EQUIP.parameters(::FlushProbe) = (S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05, lower = 1e-6))
function EQUIP.signal!(out, m::FlushProbe, p)
    Threads.atomic_add!(get_zero_subnormals() ? NFLUSHED : NKEPT, 1)   # this thread's mode; never yields
    out .= p.S0 .* exp.(-m.TE ./ p.T2)
    return out
end

# A model still being written, with no `parameters` method yet: it must still print.
struct NoParametersYet <: MRIModel
    TE::Vector{Float64}
    n::Int
end

# Models made with @mrimodel, with the equations of built-in models written per observation, to
# compare their fits (they must be the same, bit for bit), and models for its other features.
"""T2 decay with an offset, made with @mrimodel: the same as T2DecayOffset."""
@mrimodel MacroT2Offset(TE; S0 = Free(start = :scale, lower = 0),
                            T2 = Free(start = mean(TE), lower = 1e-6),
                            C = Free(start = 0.0)) =
    S0 * exp(-(1 / T2) * TE) + C

# T1MFA's equation; TR may be one number for every flip angle.
@mrimodel function MacroMFA(FA, TR; M0 = Free(start = :scale, lower = 0),
                            T1 = Free(start = 1.0, lower = 1e-6), B1 = 1.0)
    R1 = 1 / T1
    sinα, cosα = sincos(B1 * deg2rad(FA))
    E1 = exp(-R1 * TR)
    M0 * sinα * (1 - E1) / (1 - cosα * E1)
end

# QMTSIR's equation, per observation (the built-in computes the rates once per voxel); the same
# roles: kmf (≥ 0 if freed, as `Fixed(...)` written in the macro), Sm and the TD start values
# fixed, R1m tied to R1f by naming it.
@mrimodel function MacroSIR(TI, TD, magnitude = true;
                            PSR = Free(start = 0.1, lower = 0, upper = 1),
                            R1f = Free(start = 1.0, lower = 0.05, upper = 10),
                            Sf = Free(start = -0.95, lower = -1.2, upper = 0),
                            M0f = Free(start = :scale, lower = 0),
                            kmf = Fixed(12.5; lower = 0), R1m = R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)
    kfm = kmf * PSR
    ΔR1 = sqrt((R1f - R1m + kfm - kmf)^2 + 4 * kfm * kmf)
    R1⁺ = (R1f + R1m + kfm + kmf + ΔR1) / 2
    R1⁻ = R1⁺ - ΔR1
    invΔR1 = 1 / ΔR1
    bf_td⁺ = ((Mzf_td0 - 1) * (R1f - R1⁻) + (Mzf_td0 - Mzm_td0) * kfm) * invΔR1
    bf_td⁻ = (Mzf_td0 - 1) - bf_td⁺
    bm_td⁺ = ((Mzm_td0 - 1) * (R1m - R1⁻) + (Mzm_td0 - Mzf_td0) * kmf) * invΔR1
    bm_td⁻ = (Mzm_td0 - 1) - bm_td⁺
    E_td⁺, E_td⁻ = exp(-R1⁺ * TD), exp(-R1⁻ * TD)
    E_ti⁺, E_ti⁻ = exp(-R1⁺ * TI), exp(-R1⁻ * TI)
    Mzf_td = bf_td⁺ * E_td⁺ + bf_td⁻ * E_td⁻ + 1
    Mzm_td = bm_td⁺ * E_td⁺ + bm_td⁻ * E_td⁻ + 1
    Mzf_inv = Sf * Mzf_td
    Mzm_inv = Sm * Mzm_td
    bf_ti⁺ = ((Mzf_inv - 1) * (R1f - R1⁻) + (Mzf_inv - Mzm_inv) * kfm) * invΔR1
    bf_ti⁻ = (Mzf_inv - 1) - bf_ti⁺
    Mzf_ti = bf_ti⁺ * E_ti⁺ + bf_ti⁻ * E_ti⁻ + 1
    s = M0f * Mzf_ti
    return magnitude ? abs(s) : s
end

# A tie by a formula of another parameter, and a free parameter with no start (given in the call).
# T2Decay's equation written as a user writes it, relaxation as its equation, T2 fitted
@mrimodel BlockT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05, lower = 1e-6)) =
    S0 * exp(-TE / T2)

# The same acquisition given as one 3-vector per image: refused, with what to write instead
Base.@kwdef struct DirVectors{A,G} <: MRIModel
    b::A
    g::G
end
EQUIP.parameters(::Type{<:DirVectors}) = (S0 = Free(start = :scale, lower = 0),)
EQUIP.signal!(out, m::DirVectors, p) = (out .= p.S0; out)

# Gauss–Legendre nodes and weights on [0, 1] (n of them, exact for polynomials of degree 2n − 1),
# by Newton's method on P_n from the Chebyshev guesses: the tests average a compartment's
# directional view over ξ ∈ [0, 1] with them (its spherical mean)
function gausslegendre01(n)
    x, w = zeros(n), zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        for _ in 1:100
            p0, p1 = 1.0, z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = n * (z * p1 - p0) / (z^2 - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        p0, p1 = 1.0, z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = n * (z * p1 - p0) / (z^2 - 1)
        x[i] = (1 + z) / 2                       # [−1, 1] → [0, 1]
        w[i] = 1 / ((1 - z^2) * dp^2)            # 2/((1 − z²)P'²), halved for [0, 1]
    end
    return x, w
end
const GL01 = gausslegendre01(120)

@mrimodel TieFormula(t; A = Free(start = :scale), k1 = Free(start = 1.0), k2 = 0.5 * k1) =
    A * (exp(-k1 * t) + exp(-k2 * t))
@mrimodel NoStart(t; A = Free(start = :scale), k) = A * exp(-k * t)

# Derived quantities (the `derived` hook) on a T2 decay: R2 = 1/T2, whose SD by the delta method
# is SD(T2)/T2² exactly; and a model whose derived quantity takes a parameter's name (refused).
@mrimodel DerivedT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05, lower = 1e-6)) =
    S0 * exp(-TE / T2)
EQUIP.derived(::DerivedT2, p) = (R2 = 1 / p.T2,)
@mrimodel ClashT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05, lower = 1e-6)) =
    S0 * exp(-TE / T2)
EQUIP.derived(::ClashT2, p) = (T2 = p.T2, R2 = 1 / p.T2)

# A SIR protocol used throughout (s): the optimized 5-point scheme of Dortch et al., MRM 2018
const TI5 = [0.010, 0.050, 0.056, 0.277, 0.843]
const TD5 = [3.270, 4.489, 1.652, 2.922, 0.010]
# An inversion recovery protocol with a varying delay TD (s)
const TI6 = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4]
const TD6 = [0.5, 3.0, 1.0, 0.2, 2.0, 0.05]

# A diffusion scheme for the multi-compartment models: b = 0 and two shells of 30 directions
const DWI_ACQ = let g = directionset(30)
    diffusionencoding([0.0; fill(1000.0, 30); fill(2500.0, 30)], reduce(hcat, [[[0.0, 0.0, 0.0]]; collect.(g); collect.(g)]))
end

# Each case: a model (with its acquisition), and the true values of EVERY parameter.
const CASES = [
    (T2Decay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, T2 = 0.08)),
    (R2Decay(TE = collect(0.003:0.003:0.036)), (S0 = 1000.0, R2 = 33.0)),
    (T2DecayOffset(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, T2 = 0.08, C = 50.0)),
    (R2DecayOffset(TE = collect(0.003:0.003:0.036)), (S0 = 1000.0, R2 = 33.0, C = 50.0)),
    (ADCDecay(b = [0.0, 250, 500, 750, 1000, 1500, 2000]), (S0 = 1000.0, ADC = 0.8e-3)),
    (ADCKurtosis(b = [0.0, 250, 500, 1000, 1500, 2000, 2500]), (S0 = 1000.0, ADC = 0.8e-3, K = 1.0)),
    (T2BiexpDecay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08)),
    (R2BiexpDecay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, fshort = 0.15, R2short = 50.0, R2long = 12.5)),
    (T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]), (M0 = 1000.0, Sf = -0.9, T1 = 0.9)),
    (T1SR(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]), (M0 = 1000.0, T1 = 1.2)),
    (T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015), (M0 = 1000.0, T1 = 1.4, B1 = 1.0)),
    (R1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]), (M0 = 1000.0, Sf = -0.9, R1 = 1 / 0.9)),
    (T1IRTD(TI = TI6, TD = TD6), (M0 = 1000.0, Sf = -0.9, T1 = 1.3, Mz_td0 = 0.0)),
    (R1IRTD(TI = TI6, TD = TD6), (M0 = 1000.0, Sf = -0.9, R1 = 1 / 1.3, Mz_td0 = 0.0)),
    (R1SR(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]), (M0 = 1000.0, R1 = 1 / 1.2)),
    (R1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015), (M0 = 1000.0, R1 = 1 / 1.4, B1 = 1.0)),
    (QMTSIR(TI = TI5, TD = TD5),
     (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, R1m = 1.0, Sm = 0.83,
      Mzf_td0 = 0.0, Mzm_td0 = 0.0)),
    (QMTSIREqualR1(TI = TI5, TD = TD5), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, Sm = 0.83)),
    (IVIM(b = [0.0, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400, 600, 800]), (S0 = 1000.0, f = 0.1, Dstar = 0.02, D = 1.0e-3)),
    (StickZeppelin(; DWI_ACQ...), (S0 = 1000.0, vax = 0.6, Dax = 1.9e-3, theta = 0.7, phi = 0.3)),
    (BallStick(; DWI_ACQ...), (S0 = 1000.0, f = 0.5, d = 1.2e-3, theta = 0.7, phi = 0.3)),
    (NODDI(; DWI_ACQ...), (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3.0e-3)),
    (BinghamNODDI(; DWI_ACQ...), (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, betafraction = 0.5, theta = 0.7, phi = 0.3, psi = 0.8, Dpar = 1.7e-3, Diso = 3.0e-3)),
]

# SIR signal straight from the matrix form (Dortch et al., MRM 2018, Eq. 1), with matrix
# exponentials, extended to start TD at Mz = Z·M0 instead of 0, with Z = diag(Mzf_td0, Mzm_td0):
#   Mz(TI, TD) = {exp(A·TI)·S·[I − exp(A·TD)·(I − Z)] + [I − exp(A·TI)]}·M0.
# An independent check of the closed-form solution in `signal!(out, ::QMTSIR, p)`.
function sir_matrix(TI, TD; PSR, R1f, Sf, M0f, kmf, R1m, Sm, Mzf_td0 = 0.0, Mzm_td0 = 0.0,
                    magnitude = true)
    kfm = kmf * PSR
    A = [-(R1f + kfm) kmf; kfm -(R1m + kmf)]
    M0 = [M0f, PSR * M0f]
    S = Diagonal([Sf, Sm])
    Z = Diagonal([Mzf_td0, Mzm_td0])
    return map(TI, TD) do ti, td
        Mz = (exp(A * ti) * S * (I - exp(A * td) * (I - Z)) + (I - exp(A * ti))) * M0
        magnitude ? abs(Mz[1]) : Mz[1]
    end
end
