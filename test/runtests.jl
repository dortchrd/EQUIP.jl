# Tests for EQUIP. Run with threads so the parallel path is exercised:
#     julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
using EQUIP
using EQUIP: setup, freenames      # internal functions that the tests inspect
using LinearAlgebra
using Random
using Statistics: mean, std, cov, median
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

# A T2 decay that records whether subnormal numbers are flushed to zero while it is evaluated.
# Dataset fits call it from several threads at once, so the record is kept under a lock (pushing
# to one vector from several threads is a data race: Julia 1.13 stops it with an error).
const FLUSHED = Bool[]
const FLUSHED_LOCK = ReentrantLock()
struct FlushProbe <: MRIModel
    TE::Vector{Float64}
end
EQUIP.parameters(::FlushProbe) = (S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05, lower = 1e-6))
function EQUIP.signal!(out, m::FlushProbe, p)
    flushed = get_zero_subnormals()                       # this thread's mode, read before the lock
    lock(() -> push!(FLUSHED, flushed), FLUSHED_LOCK)
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
# roles: kmf, Sm and the TD start values fixed, R1m tied to R1f by naming it.
@mrimodel function MacroSIR(TI, TD, magnitude = true;
                            PSR = Free(start = 0.1, lower = 0, upper = 1),
                            R1f = Free(start = 1.0, lower = 0.05, upper = 10),
                            Sf = Free(start = -0.95, lower = -1.2, upper = 0),
                            M0f = Free(start = :scale, lower = 0),
                            kmf = 12.5, R1m = R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)
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
@mrimodel TieFormula(t; A = Free(start = :scale), k1 = Free(start = 1.0), k2 = 0.5 * k1) =
    A * (exp(-k1 * t) + exp(-k2 * t))
@mrimodel NoStart(t; A = Free(start = :scale), k) = A * exp(-k * t)

# A SIR protocol used throughout (s): the optimized 5-point scheme of Dortch et al., MRM 2018
const TI5 = [0.010, 0.050, 0.056, 0.277, 0.843]
const TD5 = [3.270, 4.489, 1.652, 2.922, 0.010]
# An inversion recovery protocol with a varying delay TD (s)
const TI6 = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4]
const TD6 = [0.5, 3.0, 1.0, 0.2, 2.0, 0.05]

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

@testset "EQUIP" begin

    @testset "noise-free recovery from default starts: $(nameof(typeof(m)))" for (m, truth) in CASES
        y = signal(m, truth)
        for T in (Float64, Float32)
            r = fit!(init(m; precision = T), y)
            @test isconverged(r)
            for (name, v) in pairs(r.params)        # only the free parameters are returned
                @test v ≈ truth[name] rtol = (T == Float64 ? 1e-6 : 1e-3)
            end
        end
    end

    @testset "signal! checks lengths when called directly: $(nameof(typeof(m)))" for (m, truth) in CASES
        n = nobs(m)
        @test signal!(zeros(n), m, truth) ≈ signal(m, truth)
        @test_throws DimensionMismatch signal!(zeros(n + 1), m, truth)   # would read past the end
        @test_throws DimensionMismatch signal!(zeros(n - 1), m, truth)
    end

    @testset "fit! does not allocate" begin
        for (m, truth) in CASES, T in (Float64, Float32)
            y = signal(m, truth) .* (1 .+ 0.01 .* randn(MersenneTwister(0), nobs(m)))
            c = init(m; precision = T)
            fit!(c, y)                              # first call compiles
            @test (@allocated fit!(c, y)) == 0
            @inferred fit!(c, y)                    # the result's type is known when compiling
        end
    end

    @testset "Float32 dataset fits flush subnormal numbers to zero, and restore the mode" begin
        mp = FlushProbe(collect(0.01:0.01:0.08))
        Y = simulate(mp, (S0 = 1000.0, T2 = 0.05); σ = 5.0, noise = :gaussian, n = 40, rng = MersenneTwister(1))
        was = get_zero_subnormals()
        empty!(FLUSHED); fit(mp, Float32.(Y); precision = Float32)
        @test !isempty(FLUSHED) && all(FLUSHED) && get_zero_subnormals() == was
        empty!(FLUSHED); fit(mp, Y)
        @test !isempty(FLUSHED) && !any(FLUSHED)                 # Float64: left as it is
        empty!(FLUSHED); fit(mp, Float32.(Y[1, :]); precision = Float32)
        @test !any(FLUSHED)                                      # one voxel: not switched per call
        # in the 5-point SIR's lesion the derivatives reach 4e-40, subnormal in Float32 (1.2e-38):
        # flushed, the fit agrees with Float64's to Float32's precision
        m5 = QMTSIR(; SIR_5POINT...)
        Yl = simulate(m5, nominal(m5, :SIR_lesion); snr = 200, noise = :rician, n = 50, rng = MersenneTwister(2))
        r64, r32 = fit(m5, Yl; kmf = Free()), fit(m5, Float32.(Yl); kmf = Free(), precision = Float32)
        ok = isconverged.(r64.status) .& isconverged.(r32.status)
        @test count(ok) >= 45
        @test maximum(abs.(r32.params.PSR[ok] .- r64.params.PSR[ok])) < 1e-3
    end

    @testset "models: keyword construction, single TD/TR, lengths, precision" begin
        # one number stands for every observation
        m = T1MFA(FA = [2, 5, 10], TR = 0.015)
        @test m.TR == [0.015, 0.015, 0.015] && nobs(m) == 3
        @test QMTSIR(TI = [0.01, 0.1], TD = 2.5).TD == [2.5, 2.5]
        @test QMTSIR(TI = [0.01, 0.1], TD = 2.5).magnitude                      # default option
        @test !QMTSIR(TI = [0.01, 0.1], TD = 2.5, magnitude = false).magnitude
        # ranges and integer vectors are accepted ...
        @test nobs(T2Decay(TE = 0.01:0.01:0.1)) == 10
        # nobs allocates nothing (crlb calls it for every noise level)
        mq = QMTSIR(; SIR_4POINT...)
        nobs(mq)
        @test @allocated(nobs(mq)) == 0
        @test occursin("different lengths, [2, 3]", sprint(showerror, try nobs(QMTSIR(TI = [0.01, 0.1], TD = [1.0, 2.0, 3.0])) catch e; e end))
        # ... and init converts the acquisition to plain vectors of the working precision
        c = init(QMTSIR(TI = 0.01:0.01:0.05, TD = 2); precision = Float32)
        @test c.model.TI isa Vector{Float32} && c.model.TD == fill(2.0f0, 5) && c.model.magnitude
        # mistakes are caught
        @test_throws UndefKeywordError QMTSIR(TD = 2.5)                         # TI missing
        @test_throws DimensionMismatch init(QMTSIR(TI = [0.1, 1.0], TD = [1.0, 2.0, 3.0]))
        @test_throws DimensionMismatch fit!(init(CASES[1][1]), ones(5))         # wrong data length
    end

    @testset "QMTSIR matches the matrix form, also with nonzero start values" begin
        TI = [0.010, 0.015, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0, 2.0, 5.0]
        θ = (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 14.5, R1m = 1.2, Sm = 0.83,
             Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        for TD in (0.01, 1.0, 2.5, 5.0), mag in (true, false)
            @test signal(QMTSIR(TI = TI, TD = TD, magnitude = mag), θ) ≈
                  sir_matrix(TI, fill(TD, length(TI)); θ..., magnitude = mag)
        end
        # random parameters, including Mz left over from the readout (start values ≠ 0)
        rng = MersenneTwister(7)
        for _ in 1:200
            θr = (PSR = 0.3 * rand(rng), R1f = 0.3 + 2 * rand(rng), Sf = -1.1 * rand(rng), M0f = 1000 * rand(rng),
                  kmf = 1 + 40 * rand(rng), R1m = 0.3 + 3 * rand(rng), Sm = rand(rng),
                  Mzf_td0 = 0.4 * rand(rng) - 0.2, Mzm_td0 = 0.4 * rand(rng) - 0.2)
            @test signal(QMTSIR(TI = TI5, TD = TD5, magnitude = false), θr) ≈
                  sir_matrix(TI5, TD5; θr..., magnitude = false) rtol = 1e-10
        end
        # special cases with known answers
        m = QMTSIR(TI = TI5, TD = TD5, magnitude = false)
        θ0 = merge(defaults(m), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0))
        # no macromolecular pool (PSR = 0): single-pool inversion recovery (Li et al. 2010, Eq. 1)
        @test signal(m, merge(θ0, (PSR = 0.0, Mzf_td0 = 0.1))) ≈
              @. 1000 * (-0.95 * (1 - 0.9 * exp(-TD5)) * exp(-TI5) + 1 - exp(-TI5))
        # exchange off: a tiny kmf gives single-pool inversion recovery; kmf = 0 gives NaN
        @test signal(m, merge(θ0, (kmf = 1e-12,))) ≈
              @. 1000 * (-0.95 * (1 - exp(-TD5)) * exp(-TI5) + 1 - exp(-TI5))
        @test all(isnan, signal(m, merge(θ0, (kmf = 0.0,))))
        # start at equilibrium and no inversion: nothing changes
        @test signal(m, merge(θ0, (Sf = 1.0, Sm = 1.0, Mzf_td0 = 1.0, Mzm_td0 = 1.0))) ≈ fill(1000.0, 5)
    end

    @testset "signal needs every value; defaults(m) are merged in explicitly" begin
        m = QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)
        @test defaults(m) == (kmf = 12.5, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        @test defaults(T1MFA(FA = [2, 10], TR = 0.015)) == (B1 = 1.0,)
        @test defaults(T2Decay(TE = [0.01])) == (;)                        # nothing fixed
        θfree = (PSR = 0.12, R1f = 1.1, Sf = -0.95, M0f = 1.0)
        everything = merge(θfree, (kmf = 12.5, R1m = 1.1, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0))
        @test signal(m, merge(defaults(m), θfree)) == signal(m, everything)
        @test signal(m, merge(defaults(m), θfree, (kmf = 20.0,))) != signal(m, everything)
        # nothing is filled in silently: a missing value is an error that lists every one missing
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        @test_throws ArgumentError signal(m, θfree)                          # fixed ones missing
        @test_throws ArgumentError simulate(m, θfree; noise = :rician, snr = 100)
        @test occursin("missing: kmf, R1m, Sm, Mzf_td0, Mzm_td0", msg(() -> signal(m, θfree)))
        @test occursin("start from `nominal(m)`", msg(() -> signal(m, θfree)))
        @test occursin("nominal(m; kmf = ...)", msg(() -> signal(m, θfree)))
        @test occursin("missing: R1f, Sf, M0f", msg(() -> signal(m, merge(defaults(m), (PSR = 0.1,)))))
        @test_throws ArgumentError signal(m, merge(everything, (nope = 1.0,)))   # unknown name
        @test_throws DimensionMismatch signal(QMTSIR(TI = [0.1, 1.0], TD = [1.0, 2.0, 3.0]), everything)
        # roles are not values: passing parameters(m), or a Fixed(...), is a clear error
        @test occursin("is a role", msg(() -> signal(m, parameters(m))))
        @test occursin("is a role", msg(() -> simulate(m, parameters(m); noise = :rician, snr = 100)))
        @test occursin("write `kmf = 10.0`", msg(() -> signal(m, merge(everything, (kmf = Fixed(10.0),)))))
        # truth gives every value, the defaults included, so it simulates as it is
        t = nominal(m, :SIR_WM_II)
        @test keys(t) == keys(parameters(m)) && t.R1m == :R1f && t.Mzf_td0 == 0
        @test signal(m, t) == signal(m, merge(defaults(m), (PSR = 0.12, R1f = 1.0, Sf = -1.0, M0f = 1.0,
                                                             kmf = 10.0)))
    end

    @testset "nominal(m) and resolve" begin
        m = QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        # every value from parameters(m): starts, 1 for the scale, fixed values; ties kept
        @test nominal(m) == (PSR = 0.1, R1f = 1.0, Sf = -0.95, M0f = 1.0, kmf = 12.5, R1m = :R1f, Sm = 0.83,
                           Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        p = nominal(m; PSR = 0.15, kmf = 10.0)
        @test p.PSR == 0.15 && p.kmf == 10.0 && p.R1m === :R1f && keys(p) == keys(parameters(m))
        @test nominal(T1MFA(FA = [2, 10], TR = 0.015)) == (M0 = 1.0, T1 = 1.0, B1 = 1.0)
        # a kept tie follows later changes; resolve gives numbers, in model order
        @test resolve(m, (; nominal(m)..., R1f = 0.7)).R1m == 0.7
        @test resolve(m, nominal(m)) == (PSR = 0.1, R1f = 1.0, Sf = -0.95, M0f = 1.0, kmf = 12.5, R1m = 1.0,
                                       Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        n = nominal(m)
        @test resolve(m, NamedTuple{reverse(keys(n))}(reverse(values(n)))) == resolve(m, n)
        @test resolve(m, nominal(m; R1m = q -> 0.8 * q.R1f)).R1m ≈ 0.8
        # signal = signal! on resolved values; simulate takes nominal(m)
        @test signal(m, nominal(m)) == signal!(zeros(3), m, resolve(m, nominal(m)))
        @test size(simulate(m, nominal(m); noise = :rician, snr = 100, n = 2)) == (2, 3)
        # no number to take: must be given, and the error says why
        te = collect(0.01:0.01:0.32)
        @test occursin("`T2` (it starts from the data)", msg(() -> nominal(T2Decay(TE = te))))
        @test nominal(T2Decay(TE = te); T2 = 0.08) == (S0 = 1.0, T2 = 0.08)
        @test occursin("`T1` (it starts from BestOf values)", msg(() -> nominal(T1IR(TI = [0.1, 1.0]))))
        # ... and lists what truth takes for every parameter, the keywords' values included
        e = msg(() -> nominal(T2Decay(TE = te)))
        @test occursin("\n  S0    free  :scale       [0.0, Inf]     1.0\n", e)
        @test occursin(r"\n  T2 +free +:data +\[1.0e-6, Inf\] +required$", e)
        e = msg(() -> nominal(T1IR(TI = [0.1, 1.0]); M0 = 5.0))
        @test occursin(r"\n  M0 +free +:scale +\[0.0, Inf\] +5.0\n", e)
        # mistakes
        @test occursin("unknown parameter `kfm`", msg(() -> nominal(m; kfm = 3.0)))
        @test occursin("is a role", msg(() -> nominal(m; kmf = Fixed(10.0))))
        @test occursin("itself a tie", msg(() -> nominal(m; Sm = :R1m)))
        @test occursin("missing: Sf", msg(() -> resolve(m, Base.structdiff(nominal(m), NamedTuple{(:Sf,)}))))
        # with a tissue, a free parameter never takes its start: WM (T1, T2) does not fit QMTSIR
        e = msg(() -> nominal(m, :WM))
        @test occursin("no value for `PSR`, `R1f`, `Sf`, free parameters the tissue does not give", e)
        @test occursin("`nominal(m, :WM; PSR = value, R1f = value, Sf = value)`", e)
        @test occursin(r"\n  name +role +start/value +bounds +nominal\(m, :WM\)\n", e)
        @test occursin(r"\n  PSR +free +0.1 +\[0.0, 1.0\] +required\n", e)       # the start is not used
        @test occursin(r"\n  M0f +free +:scale +\[0.0, Inf\] +1.0\n", e)          # the scale is 1
        @test occursin("`nominal(m, tissue; Sf = value)`", msg(() -> nominal(T1IR(TI = [0.1]), (T1 = 1.0,))))
        @test occursin("is a role", msg(() -> nominal(m, :SIR_WM_II; kmf = Fixed(10.0))))
        @test nominal(m, :SIR_WM_II; PSR = 0.15).PSR == 0.15                     # keywords win
    end

    @testset "roles print readably" begin
        @test repr(Free()) == "Free()"
        @test repr(Free(start = 0.1, lower = 0, upper = 1)) == "Free(start = 0.1, lower = 0.0, upper = 1.0)"
        @test repr(Free(start = :scale, lower = 0)) == "Free(start = :scale, lower = 0.0)"
        @test repr(Free(start = [0.1, 0.2])) == "Free(start = 2-element Vector{Float64})"
        @test repr(Fixed(12.5)) == "Fixed(12.5)"
        @test repr(Fixed(:R1f)) == "Fixed(:R1f)"
        @test repr(Fixed(p -> p.R1f)) == "Fixed(<function>)"
        @test repr(Free(start = BestOf([0.3, 0.8]))) == "Free(start = BestOf([0.3, 0.8]))"
    end

    @testset "parameters on the model's type: no acquisition needed" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        m = QMTSIR(; SIR_4POINT...)
        @test parameters(QMTSIR) == parameters(m)          # no QMTSIR start depends on TI or TD
        @test nominal(QMTSIR, :SIR_WM_I) == nominal(m, :SIR_WM_I)
        @test nominal(QMTSIR; PSR = 0.15) == nominal(m; PSR = 0.15) && defaults(QMTSIR) == defaults(m)
        @test occursin("nominal(QMTSIR, :WM; PSR = value", msg(() -> nominal(QMTSIR, :WM)))
        # a start computed from the acquisition: a function for the type, its value for a model
        b = [0.0, 500, 1000]
        @test parameters(ADCDecay).ADC.start isa Function
        @test parameters(ADCDecay(b = b)).ADC.start == 1 / mean(b) == nominal(ADCDecay(b = b)).ADC
        @test occursin("`ADC` (its start is computed from the acquisition)", msg(() -> nominal(ADCDecay)))
        @test occursin("`nominal(ADCDecay; ADC = value)`", msg(() -> nominal(ADCDecay)))
        @test nominal(ADCDecay; ADC = 1e-3) == (S0 = 1.0, ADC = 1e-3) == nominal(ADCDecay, (ADC = 1e-3,))
        ti = [0.1, 0.5, 1.0]
        @test parameters(T1IR(TI = ti)).T1.start.values == EQUIP._t1grid(ti)   # BestOf from the TIs
        # a function start given in a fit is computed for that model too
        te = collect(0.01:0.01:0.32)
        mt = T2Decay(TE = te)
        y = signal(mt, (S0 = 1000.0, T2 = 0.08))
        @test fit(mt, y; T2 = Free(start = q -> mean(q.TE)), solver = LM(maxiter = 0)).params.T2 == mean(te)
        @test fit(mt, y; T2 = Free(start = q -> mean(q.TE))).params.T2 ≈ 0.08 rtol = 1e-6
        # a model whose roles are defined on the instance (the other way) still works, but its
        # type cannot be asked
        @test parameters(DataStartTest([0.01, 0.02])).c == Fixed(1.0)
        @test occursin("give a model", msg(() -> nominal(DataStartTest)))
        @test occursin("give a model", msg(() -> defaults(DataStartTest)))
        @test occursin("a function of the model", msg(() -> Free(start = "x")))
    end

    @testset "models print at the REPL" begin
        txt(x) = sprint(show, MIME"text/plain"(), x)     # what typing `m` at the REPL shows
        s = txt(QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5))
        @test startswith(s, "QMTSIR with 3 observations\n  TI = [0.01, 0.1, 1.0]\n" *
                            "  TD = 2.5, every observation\n  magnitude = true\nParameters:\n" *
                            "  name     role   start/value  bounds        nominal(m)\n")
        @test occursin("\n  PSR      free   0.1          [0.0, 1.0]    0.1\n", s)
        @test occursin("\n  M0f      free   :scale       [0.0, Inf]    1.0\n", s)
        @test occursin("\n  R1m      fixed  :R1f (tie)                 :R1f\n", s)
        @test endswith(s, "\n  Mzm_td0  fixed  0.0                        0.0")   # nothing to give
        s = txt(T2Decay(TE = 0.01:0.01:0.32))
        @test occursin("\n  TE = 0.01:0.01:0.32\n", s) && occursin(r"\n  T2 +free +:data .+ required\n", s)
        @test endswith(s, "\n`nominal(m)` needs a value for T2: `nominal(m; T2 = value)`")
        @test occursin("TE = 32 values from 0.01 to 0.32", txt(T2Decay(TE = collect(0.01:0.01:0.32))))
        @test occursin(r"T1 +free +BestOf\(30 values from 0.02 to 5.0\)", txt(T1IR(TI = [0.1, 1.0])))
        # a function tie, and a free parameter without a start (no built-in model has either)
        t = sprint(EQUIP._print_parameters, (a = Free(), b = Fixed(p -> 2 * p.a)), (a = nothing, b = :x), "v")
        @test occursin(r"\n  a +free +\[-Inf, Inf\] +required\n  b +fixed +<function> \(tie\) +:x$", t)
        # a custom model prints the same way; also before it has `parameters`, or with
        # acquisition vectors of different lengths (no count then)
        @test startswith(txt(DataStartTest([0.01, 0.02])),
                         "DataStartTest with 2 observations\n  t = [0.01, 0.02]\nParameters:")
        @test txt(NoParametersYet([0.01, 0.02], 3)) ==
              "NoParametersYet with 2 observations\n  TE = [0.01, 0.02]\n  n = 3"
        m = QMTSIR(TI = [0.01, 0.1], TD = [1.0, 2.0, 3.0], magnitude = true)
        @test startswith(txt(m), "QMTSIR\n  TI =")
        # one line (in a vector, interpolated): Julia's default, the constructor call
        @test !occursin("\n", repr(QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)))
    end

    @testset "ties: by name (Fixed(:R1f)) and by function" begin
        m = QMTSIR(TI = TI5, TD = TD5)
        @test parameters(m).R1m == Fixed(:R1f)                       # the default for QMTSIR
        @test setup(m).fixed.R1m isa EQUIP._Tie{:R1f}       # name known to the compiler
        θfree = merge(defaults(m), (PSR = 0.12, R1f = 1.1, Sf = -0.95, M0f = 1000.0))
        # simulation: tie by name == tie by function == the value written out
        @test signal(m, θfree) == signal(m, merge(θfree, (R1m = p -> p.R1f,))) ==
              signal(m, merge(θfree, (R1m = 1.1,)))
        # fitting: the override forms give the same fit as the default
        y = signal(m, θfree)
        r = fit(m, y)
        @test r.params == fit(m, y; R1m = :R1f).params
        @test r.params.PSR ≈ fit(m, y; R1m = p -> p.R1f).params.PSR rtol = 1e-10
        # a tie can follow a per-voxel map: an R1 map fixes R1f, and R1m follows it
        r1 = [0.7, 1.0, 1.4]
        Y = reduce(vcat, [permutedims(signal(m, merge(θfree, (R1f = v,)))) for v in r1])
        res = fit(m, Y; R1f = r1)
        @test keys(res.params) == (:PSR, :Sf, :M0f)
        @test res.params.PSR ≈ fill(0.12, 3) rtol = 1e-6
        # mistakes are caught before fitting
        @test_throws ArgumentError setup(m; R1m = :R1F)   # not a parameter
        @test_throws ArgumentError setup(m; R1m = :R1m)   # tied to itself
        @test_throws ArgumentError setup(m; Sm = :R1m)    # tied to a tie
        @test_throws ArgumentError signal(m, merge(θfree, (R1m = :nope,)))
    end

    @testset "parameter roles (setup): keywords named after the parameters" begin
        m = QMTSIR(TI = TI5, TD = TD5)
        s = setup(m)
        @test freenames(s) == (:PSR, :R1f, :Sf, :M0f)
        @test keys(s.fixed) == (:kmf, :R1m, :Sm, :Mzf_td0, :Mzm_td0) && s.fixed.kmf == 12.5
        @test s.lower == [0, 0.05, -1.2, 0] && s.start.PSR == 0.1
        # a value fixes a parameter that is free by default, or changes a fixed value
        @test freenames(setup(m; PSR = 0.1)) == (:R1f, :Sf, :M0f)
        @test setup(m; kmf = 15.0).fixed.kmf == 15.0
        @test setup(m; kmf = Fixed(15.0)).fixed.kmf == 15.0
        # Free() estimates a fixed parameter: its fixed value becomes the start, no bounds
        s = setup(m; kmf = Free())
        @test freenames(s) == (:PSR, :R1f, :Sf, :M0f, :kmf) && s.start.kmf == 12.5
        @test s.lower[end] == -Inf && s.upper[end] == Inf
        s = setup(m; kmf = Free(start = 20.0, lower = 5, upper = 50))   # with start and bounds
        @test s.lower[end] == 5 && s.upper[end] == 50 && s.start.kmf == 20
        # Free(...) over a free parameter keeps whatever it leaves out
        s = setup(m; Sf = Free(lower = -1.5))
        @test s.lower[3] == -1.5 && s.upper[3] == 0 && s.start.Sf == -0.95
        s = setup(m; R1f = Free(start = 0.8))
        @test s.start.R1f == 0.8 && s.lower[2] == 0.05 && s.fromdata == (false, false, false, false)
        @test Free(start = 0.8).start === 0.8 && Free(start = 1).start === 1.0   # numbers: Float64
        @test Free(start = :data).start === :data && Free(start = BestOf([0.3, 0.8])).start isa BestOf
        @test_throws ArgumentError Free(start = :auto)                  # not a start option
        # the start is a keyword, like the bounds: a positional one says how to write it
        @test_throws "write Free(start = 0.8)" Free(0.8)
        @test_throws "write Free(start = :scale, lower = 0)" Free(:scale; lower = 0)
        @test_throws "write Free(start = 0.1, lower = 0, upper = 1)" Free(0.1, 0, 1)
        # starts and precision; an explicit start replaces :scale
        mfa = T1MFA(FA = [2, 10], TR = 0.015)
        s = setup(mfa, Float32; T1 = Free(start = 2))
        @test s.start == (M0 = 1.0f0, T1 = 2.0f0)          # M0 = 1: placeholder for :scale
        @test s.scale == 1                                  # M0 is the :scale parameter
        @test setup(mfa; M0 = Free(start = 500.0)).scale == 0
        @test setup(m).scale == 4                           # M0f
        # mistakes are caught, with the parameters and options listed
        @test_throws ArgumentError setup(m; kmff = 1.0)                  # misspelled
        @test_throws ArgumentError setup(m; R1m = Free())                # a tie freed needs a start
        @test_throws ArgumentError setup(m; mask = trues(3))             # mask is for datasets
        @test_throws "solver = LM(maxiter" setup(m; maxiter = 50)        # solver settings go in LM
        ir = T1IR(TI = [0.1, 0.5, 1.0])
        @test_throws "M0 in T1IR" setup(ir; T1 = Free(start = :scale))                # only the model's scale
        @test_throws "M0 in T1IR" setup(ir; T1 = Free(start = :scale), M0 = Free(start = 1000.0))
        @test setup(ir; M0 = Free(start = 1000.0)).scale == 0                        # turned off: allowed
        @test setup(ir; M0 = Free(start = :scale)).scale == 1                        # the model's own: allowed
        @test_throws ArgumentError setup(T2Decay(TE = [0.01]); S0 = 1.0, T2 = 0.1)   # nothing free
        e = try setup(m; kmff = 1.0) catch err err end
        @test occursin("PSR, R1f", e.msg) && occursin("mask, precision, solver", e.msg)
    end

    @testset "QMTSIR: kmf fixed vs free, tie, per-voxel Sm map" begin
        m = QMTSIR(TI = TI5, TD = TD5)
        truth = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0, R1m = 1.0, Sm = 0.83,
                 Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        y = signal(m, truth)
        # kmf estimated: exact
        r = fit!(init(m; kmf = Free(start = 12.5, lower = 1, upper = 50)), y)
        @test r.params.kmf ≈ 15 rtol = 1e-5
        @test r.params.PSR ≈ 0.12 rtol = 1e-5
        # kmf assumed correctly: exact; assumed wrongly (default 12.5): PSR is biased
        @test fit!(init(m; kmf = 15.0), y).params.PSR ≈ 0.12 rtol = 1e-6
        @test abs(fit!(init(m), y).params.PSR - 0.12) > 1e-4
        # untie R1m by fixing it to a number
        y2 = signal(m, merge(truth, (kmf = 12.5, R1m = 2.0)))
        @test fit!(init(m; R1m = 2.0), y2).params.R1f ≈ 1.0 rtol = 1e-6
        # Mz left over from the readout biases the fit, unless the start values are given
        y3 = signal(m, merge(truth, (Mzf_td0 = 0.05, Mzm_td0 = 0.05)))
        @test fit!(init(m; kmf = 15.0, Mzf_td0 = 0.05, Mzm_td0 = 0.05), y3).params.PSR ≈ 0.12 rtol = 1e-6
        @test abs(fit!(init(m; kmf = 15.0), y3).params.PSR - 0.12) > 1e-4
        # per-voxel Sm map through fit on a dataset (data: voxels × observations)
        n = 50
        sm = 0.75 .+ 0.15 .* rand(MersenneTwister(4), n)
        Y = reduce(vcat, [signal(m, merge(truth, (kmf = 12.5, Sm = sm[v])))' for v in 1:n])
        res = fit(m, Y; Sm = sm)
        @test res.params.PSR ≈ fill(0.12, n) rtol = 1e-5
        # a NaN in a map marks only that voxel invalid
        sm[7] = NaN
        res = fit(m, Y; Sm = sm)
        @test res.status[7] == EQUIP.InvalidData && isnan(res.params.PSR[7])
        @test all(isconverged, res.status[setdiff(1:n, 7)])
    end

    @testset "fit on a dataset: 3-D data, mask, threads, invalid data, start maps" begin
        rng = MersenneTwister(1)
        nx, ny, nz = 20, 16, 4
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        T2 = 0.04 .+ 0.08 .* rand(rng, nx, ny, nz)
        S0 = 800 .+ 400 .* rand(rng, nx, ny, nz)
        data = S0 .* exp.(-reshape(te, 1, 1, 1, :) ./ T2) .+ 2 .* randn(rng, nx, ny, nz, length(te))
        mask = rand(rng, nx, ny, nz) .> 0.3
        data[1, 1, 1, 5] = NaN; mask[1, 1, 1] = true            # one bad voxel inside the mask

        res = fit(m, data; mask)
        @test res isa FitResult && keys(res.params) == (:S0, :T2)
        @test size(res.params.T2) == size(res.iterations) == size(res.status) == (nx, ny, nz)
        @test res.status[1, 1, 1] == EQUIP.InvalidData
        @test all(isnan, res.params.T2[.!mask]) && all(==(EQUIP.NotFitted), res.status[.!mask])
        @test all(==(0), res.iterations[.!mask]) && all(>(0), res.iterations[mask .& isconverged(res)])
        good = mask .& (res.status .!= EQUIP.InvalidData)
        @test all(isconverged, res.status[good])
        @test maximum(abs.(res.params.T2[good] .- T2[good]) ./ T2[good]) < 0.05

        # the threaded result equals fitting voxel by voxel with one cache
        c = init(m)
        for I in CartesianIndices(mask)[good][1:20]
            r = fit!(c, data[I, :])
            @test r.params.T2 == res.params.T2[I] && r.iterations == res.iterations[I]
        end

        # Float32 data and outputs; starting from the previous result's maps
        res32 = refit(res, Float32.(data); precision = Float32)
        @test eltype(res32.params.T2) == Float32
        @test res32.params.T2[good] ≈ res.params.T2[good] rtol = 1e-3

        # size mistakes are caught
        @test_throws DimensionMismatch fit(T2Decay(TE = te[1:5]), data)
        @test_throws DimensionMismatch fit(m, data; mask = mask[:, :, 1])
        @test_throws DimensionMismatch fit(m, data; T2 = Free(start = T2[:, :, 1]))
        # a map must have the data's spatial size, not just its number of voxels: a reshaped map
        # (or mask) would put values in the wrong voxels; trailing length-1 dimensions are ignored
        @test_throws DimensionMismatch fit(m, data; T2 = Free(start = reshape(T2, nx * ny, nz)))
        @test_throws DimensionMismatch fit(m, data; mask = reshape(mask, ny, nx, nz))
        d1 = data[:, :, 1:1, :]                                    # one slice: nx × ny × 1 × M
        @test size(fit(m, d1; T2 = Free(start = T2[:, :, 1])).params.T2) == (nx, ny, 1)
        @test size(fit(m, reshape(data, :, length(te)); mask = vec(mask)).params.T2) == (nx * ny * nz,)
    end

    @testset "refit: starts and bounds" begin
        rng = MersenneTwister(9)
        td = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
        m = T1SR(TD = td)                                          # T1's default start: 1.0
        T1 = 0.2 .+ 0.3 .* rand(rng, 10)
        Y = reduce(vcat, [permutedims(signal(m, (M0 = 1000.0, T1 = t)) .+ randn(rng, 6)) for t in T1])
        res = fit(m, Y; T1 = Free(lower = 0.05))
        start(r; kw...) = refit(r, Y; solver = LM(maxiter = 0), kw...).params.T1   # the start used
        @test start(res) == res.params.T1                          # previous values
        @test start(res; T1 = Free(upper = 5.0)) == res.params.T1  # a bounds-only change keeps them
        @test start(res; T1 = Free(start = 0.7)) == fill(0.7, 10)  # an explicit start wins
        @test start(res; T1 = Free(upper = 0.25)) == min.(res.params.T1, 0.25)   # moved onto a new bound
        r2 = refit(res, Y; T1 = Free(upper = 5.0))
        @test r2.settings.T1.lower == 0.05 && r2.settings.T1.upper == 5.0   # stored bound kept
        @test r2.params.T1 ≈ res.params.T1 rtol = 1e-6
    end

    @testset "mono-exponential: time vs rate form, offset versions, diffusion" begin
        te = collect(0.01:0.01:0.32)
        rng = MersenneTwister(4)
        # the same data in both forms: the same least-squares answer, R2 = 1/T2, voxel by voxel
        mT, mR = T2Decay(TE = te), R2Decay(TE = te)
        for _ in 1:20
            y = signal(mT, (S0 = 1000.0, T2 = 0.05 + 0.1 * rand(rng))) .+ 10 .* randn(rng, length(te))
            rT, rR = fit(mT, y), fit(mR, y)
            @test isconverged(rT) && isconverged(rR)
            @test rR.params.R2 ≈ 1 / rT.params.T2 rtol = 1e-5
            @test rR.params.S0 ≈ rT.params.S0 rtol = 1e-5
        end
        # an offset: recovered by the offset versions, biases the plain one, or can be fixed
        θ = (S0 = 1000.0, T2 = 0.08, C = 50.0)
        y = signal(T2DecayOffset(TE = te), θ)
        r = fit(T2DecayOffset(TE = te), y)
        @test r.params.T2 ≈ 0.08 rtol = 1e-6
        @test r.params.C ≈ 50 rtol = 1e-6
        @test abs(fit(mT, y).params.T2 - 0.08) > 0.01
        @test fit(T2DecayOffset(TE = te), y; C = 50.0).params.T2 ≈ 0.08 rtol = 1e-6
        @test fit(R2DecayOffset(TE = te), y).params.R2 ≈ 1 / 0.08 rtol = 1e-6
        # diffusion in clinical units (b in s/mm², ADC in mm²/s), including fast decays,
        # from the default start
        b = [0.0, 500, 1000, 1500, 2000, 2500, 3000]
        m = ADCDecay(b = b)
        @test signal(m, (S0 = 1.0, ADC = 1e-3)) ≈ exp.(-b .* 1e-3)
        @test signal(m, (S0 = 1.0, ADC = 1e-3)) ≈ signal(R2Decay(TE = b), (S0 = 1.0, R2 = 1e-3))
        for adc in (0.1e-3, 0.7e-3, 3e-3, 5e-3)
            r = fit(m, 800 .* exp.(-b .* adc))
            @test r.params.ADC ≈ adc rtol = 1e-6
            @test r.params.S0 ≈ 800 rtol = 1e-6
        end
    end

    @testset "T1IRTD / R1IRTD: Li 2010 Eq. 1 and its limits" begin
        m = T1IRTD(TI = TI6, TD = TD6, magnitude = false)
        rng = MersenneTwister(8)
        for _ in 1:50
            M0, Sf, T1 = 1000 * rand(rng), -0.5 - 0.6 * rand(rng), 0.2 + 3 * rand(rng)
            # Li et al. 2010, Eq. 1, written as in the paper
            li = @. M0 * (Sf * (1 - exp(-TD6 / T1)) * exp(-TI6 / T1) + 1 - exp(-TI6 / T1))
            @test signal(m, (M0 = M0, Sf = Sf, T1 = T1, Mz_td0 = 0.0)) ≈ li
            @test signal(R1IRTD(TI = TI6, TD = TD6, magnitude = false), (M0 = M0, Sf = Sf, R1 = 1 / T1, Mz_td0 = 0.0)) ≈ li
        end
        θ = (M0 = 1000.0, T1 = 0.9, Mz_td0 = 0.0)
        # a perfect inversion: the textbook form with TR = TI + TD
        @test signal(m, merge(θ, (Sf = -1.0,))) ≈ @. 1000 * (1 - 2 * exp(-TI6 / 0.9) + exp(-(TI6 + TD6) / 0.9))
        # a long TD, or Mz at equilibrium at the start of TD: T1IR, with the same parameters
        ir = signal(T1IR(TI = TI6, magnitude = false), (M0 = 1000.0, Sf = -0.9, T1 = 0.9))
        @test ir ≈ @. 1000 * (1 - 1.9 * exp(-TI6 / 0.9))                   # B = 1 - Sf
        @test signal(T1IRTD(TI = TI6, TD = 100.0, magnitude = false), merge(θ, (Sf = -0.9,))) ≈ ir
        @test signal(m, merge(θ, (Sf = -0.9, Mz_td0 = 1.0))) ≈ ir
        # one TD for every TI
        @test T1IRTD(TI = TI6, TD = 1.0).TD == fill(1.0, 6)
        @test R1IRTD(TI = TI6, TD = 1.0).TD == fill(1.0, 6)
    end

    @testset "solver: steps into a dead zone are rejected; bounds are still reached" begin
        # A step that pushes a time constant to ~0 leaves it with no influence on the signal
        # (every sample has fully recovered). Such steps are rejected, so short T1 is found from
        # the default start (T1 = 1 s); before the guard these fits ended at T1 = 1e-6.
        sr = T1SR(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0])
        for T1 in (0.05, 0.1, 0.2)
            @test fit(sr, signal(sr, (M0 = 1000.0, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
        end
        mfa = T1MFA(FA = [2.0, 5, 10, 15, 20, 30], TR = 0.015)
        for T1 in (0.05, 0.2)
            @test fit(mfa, signal(mfa, (M0 = 1000.0, T1 = T1, B1 = 1.0))).params.T1 ≈ T1 rtol = 1e-6
        end
        # fits whose answer lies on a bound still end exactly there (there the parameter still
        # affects the signal, so the guard does not apply)
        y = 1000 .+ 0.05 .* (1:32) .+ [isodd(j) ? 1.0 : -1.0 for j in 1:32]     # slightly rising
        r = fit(R2Decay(TE = collect(0.01:0.01:0.32)), y)
        @test r.params.R2 == 0
        @test isconverged(r)
        m = QMTSIR(TI = TI5, TD = TD5)
        r = fit(m, signal(m, merge(defaults(m), (PSR = 0.0, R1f = 1.0, Sf = -0.95, M0f = 1000.0))))   # no MT
        @test r.params.PSR == 0
        @test r.params.R1f ≈ 1.0 rtol = 1e-6
        # the guard itself: parameter 2 loses (almost) all influence; frozen parameters don't count
        A = [4.0 0.0; 0.0 9.0]
        @test EQUIP._lost_influence(A, [4.0 0.0; 0.0 1e-20], [1.0, 1.0])
        @test !EQUIP._lost_influence(A, [4.0 0.0; 0.0 1e-3], [1.0, 1.0])
        @test !EQUIP._lost_influence(A, [4.0 0.0; 0.0 0.0], [1.0, 0.0])
    end

    @testset "bi-exponential T2 and diffusion kurtosis" begin
        te = collect(0.01:0.01:0.32)
        mT, mR = T2BiexpDecay(TE = te), R2BiexpDecay(TE = te)
        θT = (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08)
        θR = (S0 = 1000.0, fshort = 0.15, R2short = 50.0, R2long = 12.5)
        # the two forms are the same signal; the components can swap; fshort = 0 is mono-exponential
        @test signal(mT, θT) ≈ signal(mR, θR)
        @test signal(mT, θT) ≈ signal(mT, (S0 = 1000.0, fshort = 0.85, T2short = 0.08, T2long = 0.02))
        @test signal(mT, merge(θT, (fshort = 0.0,))) ≈ signal(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08))
        # relabel puts the shorter T2 (the larger rate) first, only when all three are free
        θs = EQUIP.relabel(mT, (S0 = 1000.0, fshort = 0.85, T2short = 0.08, T2long = 0.02))
        @test θs.T2short == 0.02
        @test θs.T2long == 0.08
        @test θs.fshort ≈ 0.15
        @test EQUIP.relabel(mR, (S0 = 1000.0, fshort = 0.85, R2short = 12.5, R2long = 50.0)).R2short == 50.0
        @test EQUIP.relabel(mT, (S0 = 1000.0, fshort = 0.85, T2short = 0.08)) ==
              (S0 = 1000.0, fshort = 0.85, T2short = 0.08)                       # T2long fixed
        @test EQUIP.relabel(T2Decay(TE = te), (S0 = 1.0, T2 = 0.1)) == (S0 = 1.0, T2 = 0.1)  # default
        # noisy data (SNR 100): both forms report the same, ordered components
        Y = reduce(vcat, [permutedims(signal(mT, θT) .+ 10 .* randn(MersenneTwister(i), 32)) for i in 1:100])
        rT, rR = fit(mT, Y), fit(mR, Y)
        @test all(rT.params.T2short .<= rT.params.T2long)
        @test all(rR.params.R2short .>= rR.params.R2long)
        # where both reach the same minimum (equal SSE) they agree; local minima are real in this
        # model (with these seeds, 1 voxel of 100 converged to different minima in the two forms)
        same = isconverged.(rT.status) .& isconverged.(rR.status) .& (abs.(rT.sse .- rR.sse) .<= 1e-6 .* rR.sse)
        @test count(same) >= 90
        @test rT.params.T2short[same] ≈ 1 ./ rR.params.R2short[same] rtol = 1e-4
        @test rT.params.fshort[same] ≈ rR.params.fshort[same] rtol = 1e-4
        # mono-exponential data: a degenerate but exact description (SSE ≈ 0), whichever it is
        y = signal(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08))
        for m in (mT, mR)
            r = fit(m, y)
            @test isconverged(r)
            @test r.sse < 1e-10
        end

        # kurtosis: K = 0 is ADCDecay; the equation; the signal decays while b·ADC·K < 3
        b = [0.0, 250, 500, 1000, 1500, 2000, 2500]
        mk = ADCKurtosis(b = b)
        @test signal(mk, (S0 = 1000.0, ADC = 1e-3, K = 0.0)) ≈ signal(ADCDecay(b = b), (S0 = 1000.0, ADC = 1e-3))
        @test signal(mk, (S0 = 1000.0, ADC = 1e-3, K = 1.2)) ≈
              1000 .* exp.(-b .* 1e-3 .+ (b .* 1e-3) .^ 2 .* 1.2 ./ 6)
        @test all(diff(signal(mk, (S0 = 1000.0, ADC = 0.8e-3, K = 1.0))) .< 0)    # 2500·0.8e-3·1 = 2
        @test fit(mk, signal(mk, (S0 = 1000.0, ADC = 2e-3, K = 0.3))).params.K ≈ 0.3 rtol = 1e-6
    end

    @testset "data start: log-linear T2 for T2Decay" begin
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        y = signal(m, (S0 = 1000.0, T2 = 0.04))
        # Noise-free data: the log-linear fit is exact, so the start is already the answer.
        # `maxiter = 0` returns the starting point; S0 comes from least squares given that T2.
        @test datastart(m, y, (;)).T2 ≈ 0.04
        s = fit(m, y; solver = LM(maxiter = 0)).params
        @test s.T2 ≈ 0.04
        @test s.S0 ≈ 1000
        @test fit(m, Float32.(y); precision = Float32, solver = LM(maxiter = 0)).params.T2 ≈ 0.04f0
        # points ≤ 0 are skipped
        @test datastart(m, [y[1:5]; zeros(27)], (;)).T2 ≈ 0.04
        # no usable estimate (rising data): the fallback, mean(TE)
        @test datastart(m, collect(1.0:32.0), (;)).T2 ≈ sum(te) / 32
        @test fit(m, collect(1.0:32.0); solver = LM(maxiter = 0)).params.T2 ≈ sum(te) / 32
        # T2 starts at :data by default; another start replaces it; a fixed T2 is left alone
        @test setup(m).fromdata == (false, true)
        @test fit(m, y; T2 = Free(start = 0.2), solver = LM(maxiter = 0)).params.T2 == 0.2
        @test setup(m; T2 = Free(start = 0.2)).fromdata == (false, false)
        @test fit(m, y; T2 = 0.04).params.S0 ≈ 1000
        # noisy data: the same answer as from the old start, without allocating
        yn = y .+ 10 .* randn(MersenneTwister(3), 32)
        r, r0 = fit(m, yn), fit(m, yn; T2 = Free(start = sum(te) / 32))
        @test r.params.T2 ≈ r0.params.T2 rtol = 1e-6
        @test r.iterations < r0.iterations
        c = init(m); fit!(c, yn)
        @test @allocated(fit!(c, yn)) == 0
        @test datastart(R2Decay(TE = te), y, (;)) === nothing     # no data start by default
        @test_throws "no data start" setup(R2Decay(TE = te); R2 = Free(start = :data))
        # datastart gets the voxel's fixed values (here c, a number or a map); a NaN fails
        mt = DataStartTest([0.01, 0.02, 0.04, 0.08])
        yt = signal(mt, (S0 = 100.0, T2 = 0.05, c = 1.0))
        @test fit(mt, yt; solver = LM(maxiter = 0)).params.T2 ≈ 0.05
        @test fit(mt, permutedims([yt yt]); c = [1.0, 2.0], solver = LM(maxiter = 0)).params.T2 ≈ [0.05, 0.1]
        @test fit(mt, yt; c = -1.0).status == EQUIP.Failed
    end

    @testset "BestOf start: magnitude inversion recovery's local minima" begin
        # From a single start (T1 = 1 s), magnitude IR used to end in a wrong local minimum (the
        # signal null in the wrong place) at T1 = 0.05, 0.1 and 4 s with these TIs; starting from
        # the best of 30 values (BestOf) finds the right one, in both forms.
        ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        for T1 in (0.05, 0.1, 0.5, 4.0)
            m, mr = T1IR(TI = ti), R1IR(TI = ti)
            @test fit(m, signal(m, (M0 = 1000.0, Sf = -0.95, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
            @test fit(mr, signal(mr, (M0 = 1000.0, Sf = -0.95, R1 = 1 / T1))).params.R1 ≈ 1 / T1 rtol = 1e-6
        end
        for T1 in (0.05, 0.1, 0.2, 0.3, 4.0)
            m, mr = T1IRTD(TI = TI6, TD = TD6), R1IRTD(TI = TI6, TD = TD6)
            @test fit(m, signal(m, (M0 = 1000.0, Sf = -0.95, T1 = T1, Mz_td0 = 0.0))).params.T1 ≈ T1 rtol = 1e-6
            @test fit(mr, signal(mr, (M0 = 1000.0, Sf = -0.95, R1 = 1 / T1, Mz_td0 = 0.0))).params.R1 ≈ 1 / T1 rtol = 1e-6
        end
        # 30 values spanning the TIs; replaced by any other start, absent when T1 is fixed
        s = setup(T1IR(TI = ti))
        @test s.candindex == 3 && length(s.candidates) == 30
        @test s.candidates[1] ≈ 0.01 && s.candidates[end] ≈ 16
        @test setup(T1IR(TI = ti); T1 = Free(start = 2.0)).candindex == 0
        @test setup(T1IR(TI = ti); T1 = 2.0).candindex == 0
        @test setup(T2Decay(TE = [0.01, 0.02])).candindex == 0           # none by default
        # values from the call; those outside the bounds are dropped, an error if none is left
        s = setup(T1IR(TI = ti); T1 = Free(start = BestOf([0.1, 0.5, 5.0]), upper = 3))
        @test s.candidates == [0.1, 0.5] && s.start.T1 == 0.1           # middle value: placeholder
        @test_throws "no BestOf value within" setup(T1IR(TI = ti); T1 = Free(lower = 20))
        @test_throws "only one parameter" setup(T1IR(TI = ti); Sf = Free(start = BestOf([-1, -0.9])))
        m2 = T2Decay(TE = 0.01:0.01:0.32)
        r = fit(m2, signal(m2, (S0 = 1000.0, T2 = 0.04)); T2 = Free(start = BestOf([0.01, 0.04, 1.0])),
                solver = LM(maxiter = 0))
        @test r.params.T2 == 0.04 && r.params.S0 ≈ 1000                 # the best of the three
        # the values: any list of finite numbers; printed in short
        @test BestOf(0.1:0.1:0.3).values ≈ [0.1, 0.2, 0.3]
        @test_throws ArgumentError BestOf(Float64[])
        @test_throws ArgumentError BestOf([0.1, NaN])
        @test sprint(show, BestOf(1:10)) == "BestOf(10 values from 1.0 to 10.0)"
    end

    @testset "T1 and R1 forms: the same answer, R1 = 1/T1, voxel by voxel" begin
        rng = MersenneTwister(6)
        ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        td = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
        fa = [2.0, 5, 10, 15, 20, 30]
        pairs = ((T1IR(TI = ti), R1IR(TI = ti), (M0 = 1000.0, Sf = -0.9)),
                 (T1IRTD(TI = TI6, TD = TD6), R1IRTD(TI = TI6, TD = TD6), (M0 = 1000.0, Sf = -0.95, Mz_td0 = 0.0)),
                 (T1SR(TD = td), R1SR(TD = td), (M0 = 1000.0,)),
                 (T1MFA(FA = fa, TR = 0.015), R1MFA(FA = fa, TR = 0.015), (M0 = 1000.0, B1 = 1.0)))
        for (mT, mR, θ) in pairs, _ in 1:10
            T1 = 0.6 + 1.2 * rand(rng)
            @test signal(mR, merge(θ, (R1 = 1 / T1,))) ≈ signal(mT, merge(θ, (T1 = T1,)))
            y = signal(mT, merge(θ, (T1 = T1,))) .* (1 .+ 0.01 .* randn(rng, nobs(mT)))
            rT, rR = fit(mT, y), fit(mR, y)
            @test isconverged(rT) && isconverged(rR)
            @test rR.params.R1 ≈ 1 / rT.params.T1 rtol = 1e-5
        end
    end

    @testset "T1MFA with a per-voxel B1 map" begin
        rng = MersenneTwister(3)
        m = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
        B1 = 0.8 .+ 0.4 .* rand(rng, 16, 16)
        T1 = 0.6 .+ 1.0 .* rand(rng, 16, 16)
        data = zeros(16, 16, 6)
        for I in CartesianIndices(B1)
            data[I, :] = signal(m, (M0 = 1000.0, T1 = T1[I], B1 = B1[I]))
        end
        with = fit(m, data; B1 = B1)
        without = fit(m, data)                     # B1 = 1 everywhere: biased
        @test with.params.T1 ≈ T1 rtol = 1e-6
        @test maximum(abs.(without.params.T1 .- T1) ./ T1) > 0.1
        @test with.fixed.B1 === B1                 # a dataset result keeps the map (not copied)
        # one voxel fitted with the map: the result keeps that voxel's B1, so it is self-contained
        r7 = fit!(init(m; B1 = B1), data[7, 1, :]; voxel = 7)
        @test r7.fixed.B1 == B1[7]
        @test fitted(r7) ≈ data[7, 1, :] rtol = 1e-6
        @test resolve(r7) == (M0 = r7.params.M0, T1 = r7.params.T1, B1 = B1[7])
    end

    @testset "fit: one voxel (vector) or a dataset with any number of spatial dimensions" begin
        rng = MersenneTwister(5)
        m = T2Decay(TE = collect(0.01:0.01:0.32))
        curve(T2) = signal(m, (S0 = 1000.0, T2 = T2))
        # a vector is ONE voxel: plain numbers, same as init + fit!
        r = fit(m, curve(0.08))
        @test r isa FitResult && isconverged(r)
        @test r.params.T2 ≈ 0.08 rtol = 1e-6
        @test r.params == fit!(init(m), curve(0.08)).params
        @test fit(m, curve(0.08); S0 = 1000.0).params.T2 ≈ 0.08 rtol = 1e-6
        # matrices and N-d arrays: observations last, maps with the spatial size
        for sz in ((40,), (8, 5), (6, 5, 3), (4, 3, 3, 2))
            T2 = 0.04 .+ 0.08 .* rand(rng, sz...)
            data = zeros(sz..., nobs(m))
            for I in CartesianIndices(T2)
                data[I, :] = curve(T2[I])
            end
            res = fit(m, data)
            @test size(res.params.T2) == sz && size(res.status) == sz
            @test res.params.T2 ≈ T2 rtol = 1e-6
        end
    end

    @testset "FitResult: one type for one voxel and a dataset" begin
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        y = signal(m, (S0 = 1000.0, T2 = 0.08)) .+ 5 .* randn(MersenneTwister(7), 32)
        # one voxel: numbers, plus the model and settings it was fitted with
        r = fit(m, y; solver = LM(maxiter = 50))
        @test r.params.T2 isa Float64 && r.status isa FitStatus && r.iterations isa Int32
        @test r.model isa T2Decay && r.model.TE == te
        @test r.settings.solver.maxiter == 50 && r.settings.precision == Float64
        @test isconverged(r) isa Bool
        @test fitted(r) ≈ signal(m, r.params)
        @test residuals(r, y) ≈ y .- fitted(r)
        @test sum(abs2, residuals(r, y)) ≈ r.sse rtol = 1e-8
        # fixed values (numbers and ties) are used by `fitted`
        mq = QMTSIR(TI = [0.01, 0.03, 0.1, 0.3, 1.0], TD = 1.0)
        yq = signal(mq, merge(defaults(mq), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0)))
        rq = fit(mq, yq; kmf = 15.0)
        @test fitted(rq) ≈ yq rtol = 1e-6
        @test rq.fixed == (kmf = 15.0, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        rr = resolve(rq)                                   # every value, fitted and fixed, ties evaluated
        @test keys(rr) == keys(parameters(mq)) && rr.kmf == 15.0 && rr.R1m == rr.R1f == rq.params.R1f
        @test fit(mq, yq; kmf = Fixed(15.0)).params == rq.params                # role or number
        # a dataset: maps of the same fields; status is a FitStatus map (1 byte per voxel)
        T2 = 0.05 .+ 0.05 .* rand(MersenneTwister(8), 5, 4)
        data = zeros(5, 4, 32)
        for I in CartesianIndices(T2)
            data[I, :] = signal(m, (S0 = 1000.0, T2 = T2[I])) .+ 5 .* randn(MersenneTwister(I[1] + 10I[2]), 32)
        end
        mask = trues(5, 4); mask[1, 1] = false
        res = fit(m, data; mask)
        @test eltype(res.status) == FitStatus && sizeof(res.status) == 20
        @test eltype(res.iterations) == Int32
        @test UInt8.(res.status)[1, 1] == 0x00                                  # codes for files
        @test isconverged(res) isa BitMatrix && !isconverged(res)[1, 1]
        @test res.settings.mask === mask                                        # a reference
        @test fitted(res, 3, 2) ≈ signal(m, (S0 = res.params.S0[3, 2], T2 = res.params.T2[3, 2]))
        @test fitted(res, 8) == fitted(res, 3, 2)                               # linear index
        @test resolve(res, 3, 2) == (S0 = res.params.S0[3, 2], T2 = res.params.T2[3, 2])
        @test_throws ArgumentError resolve(res)
        @test_throws ArgumentError resolve(r, 1)
        @test sum(abs2, residuals(res, data, 3, 2)) ≈ res.sse[3, 2] rtol = 1e-8
        @test all(isnan, fitted(res, 1, 1))                                     # outside the mask
        # refit with the same settings, or from the previous maps
        @test isequal(fit(res.model, data; res.settings...).params, res.params)
        @test refit(res, data).params.T2[mask] ≈ res.params.T2[mask] rtol = 1e-6
        @test all(refit(res, data).iterations[mask] .<= res.iterations[mask])   # starts at the answer
        r2 = refit(res, data; S0 = 1000.0)                                      # one change
        @test keys(r2.params) == (:T2,) && r2.settings.S0 == 1000.0
        # the wrong kind of call is caught
        @test_throws ArgumentError fitted(res)
        @test_throws ArgumentError fitted(r, 1)
        # display: one line, and the summary at the REPL
        @test occursin("one voxel", sprint(show, r)) && occursin("converged", sprint(show, res))
        @test occursin("status   Converged", sprint(show, MIME("text/plain"), r))
        s = sprint(show, MIME("text/plain"), rq)
        @test occursin("kmf = 15.0", s) && occursin("R1m = R1f (tied)", s)
        s = sprint(show, MIME("text/plain"), res)
        @test occursin("19 of 20 voxels (mask)", s) && occursin("medians", s)
    end

    @testset "fit statistics: stderror, vcov, cor, residualsd, AIC/BIC, onbound" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        θ = (S0 = 1000.0, T2 = 0.08)
        y = simulate(m, θ; σ = 10.0, noise = :gaussian, rng = MersenneTwister(1))
        r = fit(m, y)
        # the counts: N, the free parameters and σ, N − P, the SSE
        @test nobs(r) == 32 && dof(r) == 3 && dof_residual(r) == 30 && rss(r) == r.sse
        @test residualsd(r) ≈ sqrt(r.sse / 30) rtol = 1e-14
        # the covariance σ̂²(JᵀJ)⁻¹ at the fit's answer; with σ given, crlb's at the fit's values
        J = jacobian(r)
        C = residualsd(r)^2 * inv(J' * J)
        @test vcov(r) ≈ C rtol = 1e-10
        @test vcov(r) == vcov(r)'
        @test collect(values(stderror(r))) ≈ sqrt.(diag(C)) rtol = 1e-10
        @test keys(stderror(r)) == keys(r.params)
        s10, c10 = stderror(r; σ = 10.0), crlb(m, merge(θ, r.params); σ = 10.0).sd
        @test all(p -> s10[p] ≈ c10[p], keys(s10))
        R = cor(r)
        @test R ≈ C ./ sqrt.(diag(C) * diag(C)') rtol = 1e-10
        @test R[1, 1] == R[2, 2] == 1
        # calibrated: over noisy copies, the standard errors match the spread of the estimates
        Y = simulate(m, θ; σ = 10.0, noise = :gaussian, n = 2000, rng = MersenneTwister(2))
        res = fit(m, Y)
        se = stderror(res)
        @test se.T2 isa Vector{Float64} && length(se.T2) == 2000
        @test median(se.T2) ≈ std(res.params.T2) rtol = 0.05
        @test median(se.S0) ≈ std(res.params.S0) rtol = 0.05
        @test se.T2[7] == stderror(fit(m, Y[7, :])).T2                      # each voxel its own fit
        @test residualsd(res) ≈ sqrt.(res.sse ./ 30)
        @test median(residualsd(res)) ≈ 10 rtol = 0.05
        @test size(vcov(res)) == (2000,)
        @test vcov(res)[7] ≈ vcov(fit(m, Y[7, :])) rtol = 1e-12
        # a parameter on a bound: flagged, its SD NaN, the others with it held
        rb = fit(m, y; S0 = Free(lower = 0, upper = 900.0))
        @test rb.params.S0 == 900 && onbound(rb) == (S0 = true, T2 = false)
        @test isnan(stderror(rb).S0) && isfinite(stderror(rb).T2)
        Jt = jacobian(rb)[:, 2:2]
        @test stderror(rb).T2 ≈ residualsd(rb) * sqrt(inv(Jt' * Jt)[1]) rtol = 1e-10
        @test cor(rb)[1, 1] == 1 && isnan(cor(rb)[1, 2])
        @test onbound(fit(m, Y; S0 = Free(lower = 0, upper = 990.0))).S0 isa BitVector
        # N = P (4-point SIR, four free parameters): no residual estimate of σ; with σ given,
        # crlb's at the fit's values and the fit's assumption
        m4 = QMTSIR(; SIR_4POINT...)
        θ4 = nominal(m4, :SIR_WM_I)
        r4 = fit(m4, simulate(m4, θ4; noise = :rician, snr = 200, rng = MersenneTwister(3)))
        @test dof_residual(r4) == 0 && isnan(residualsd(r4)) && all(isnan, values(stderror(r4)))
        s4 = stderror(r4; σ = 0.005)
        c4 = crlb(m4, merge(θ4, r4.params, (kmf = 12.5,)); σ = 0.005)
        @test all(p -> s4[p] ≈ c4.sd[p], keys(s4))
        # the information criteria where the residuals say nothing: all four NaN at N = P; AICc
        # NaN at N ≤ k + 1 (k = P + 1: its correction undefined, or negative below), finite
        # from N = k + 2 (T2: P = 2, k = 3)
        @test all(isnan, (loglikelihood(r4), aic(r4), aicc(r4), bic(r4)))
        for (n, defined) in ((3, false), (4, false), (5, true))
            rn = fit(T2Decay(TE = te[1:n]), y[1:n])
            @test isfinite(aic(rn)) && isfinite(bic(rn)) && isfinite(loglikelihood(rn))
            @test isfinite(aicc(rn)) == defined
        end
        r5 = fit(T2Decay(TE = te[1:5]), y[1:5])
        @test aicc(r5) ≈ aic(r5) + 2 * 3 * 4 / (5 - 3 - 1)
        @test all(isnan, aicc(fit(T2Decay(TE = te[1:4]), Y[:, 1:4])))
        # maps: fixed values per voxel (B1), a mask (unfitted voxels NaN), σ as a map
        mf = T1MFA(FA = [3.0, 6, 10, 15, 20], TR = 0.015)
        b1 = [0.8 + 0.4 * (i + j) / 20 for i in 1:6, j in 1:5]
        data = simulate(mf, (M0 = fill(1000.0, 6, 5), T1 = fill(1.2, 6, 5), B1 = b1); σ = 5.0,
                        noise = :gaussian, rng = MersenneTwister(4))
        mask = trues(6, 5); mask[1, 1] = false
        rm = fit(mf, data; B1 = b1, mask)
        sm = stderror(rm)
        one = fit!(init(mf; B1 = b1), data[3, 4, :]; voxel = LinearIndices((6, 5))[3, 4])
        @test size(sm.T1) == (6, 5) && isnan(sm.T1[1, 1]) && sm.T1[3, 4] == stderror(one).T1
        @test stderror(rm; σ = fill(5.0, 6, 5)).T1[3, 4] == stderror(rm; σ = 5.0).T1[3, 4]
        @test occursin("`σ` is a map of size", msg(() -> stderror(rm; σ = fill(5.0, 2, 2))))
        # a Float32 fit: the statistics in Float64, as the Float64 fit's to its rounding
        @test collect(values(stderror(fit(m, y; precision = Float32)))) ≈ collect(values(stderror(r))) rtol = 1e-4
        # AIC/BIC: the information criteria pick the model that made the data
        mb = T2BiexpDecay(TE = te)
        yb = simulate(mb, (S0 = 1000.0, fshort = 0.3, T2short = 0.015, T2long = 0.09); σ = 5.0,
                      noise = :gaussian, rng = MersenneTwister(5))
        ym = simulate(m, θ; σ = 5.0, noise = :gaussian, rng = MersenneTwister(6))
        @test bic(fit(mb, yb)) < bic(fit(m, yb)) && aic(fit(mb, yb)) < aic(fit(m, yb))
        @test bic(fit(m, ym)) < bic(fit(mb, ym))
        @test loglikelihood(r) ≈ -16 * (log(2π * r.sse / 32) + 1) rtol = 1e-14
        @test aic(r) ≈ -2loglikelihood(r) + 6 && bic(r) ≈ -2loglikelihood(r) + 3log(32)
        @test aicc(r) ≈ aic(r) + 2 * 3 * 4 / (32 - 3 - 1)
        @test bic(res) isa Vector{Float64} && bic(res)[7] == bic(fit(m, Y[7, :]))
    end

    @testset "simulation: simulate, tissue presets, maps, compare" begin
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        θ = (S0 = 1000.0, T2 = 0.08)
        # shapes: one voxel, n copies; the same seed gives the same data
        @test size(simulate(m, θ; noise = :rician, snr = 50)) == (32,)
        @test size(simulate(m, θ; noise = :rician, snr = 50, n = 7)) == (7, 32)
        @test simulate(m, θ; noise = :rician, snr = 50, rng = MersenneTwister(1)) == simulate(m, θ; noise = :rician, snr = 50, rng = MersenneTwister(1))
        # snr is relative to the scale parameter (here σ = 1000/50 = 20); Gaussian SD; Rician floor
        # Float32 data: the same draws, computed in Float64 and rounded; for maps and Σ too
        @test simulate(m, θ; snr = 50, n = 7, noise = :rician, precision = Float32, rng = MersenneTwister(1)) ==
              Float32.(simulate(m, θ; snr = 50, n = 7, noise = :rician, rng = MersenneTwister(1)))
        @test eltype(simulate(m, (S0 = fill(1000.0, 3, 2), T2 = 0.08); σ = 5.0, noise = :gaussian, precision = Float32)) == Float32
        Σ32 = [25.0 * 0.5^abs(i - j) for i in 1:32, j in 1:32]
        @test simulate(m, θ; Σ = Σ32, n = 5, noise = :rician, precision = Float32, rng = MersenneTwister(4)) ==
              Float32.(simulate(m, θ; Σ = Σ32, n = 5, noise = :rician, rng = MersenneTwister(4)))
        Y = simulate(m, θ; snr = 50, n = 20_000, noise = :gaussian, rng = MersenneTwister(2))
        @test std(Y[:, 1]) ≈ 20 rtol = 0.03
        @test mean(Y[:, 1]) ≈ signal(m, θ)[1] rtol = 0.003
        Yr = simulate(m, (S0 = 1000.0, T2 = 1e-4); noise = :rician, σ = 20, n = 20_000, rng = MersenneTwister(3))
        @test mean(Yr[:, end]) ≈ 20 * sqrt(π / 2) rtol = 0.02              # Rayleigh: no signal left
        # the noise's distribution is the user's choice: no default, whatever the model's options
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        @test occursin("give the noise's distribution", msg(() -> simulate(m, θ; snr = 50)))
        @test occursin("give the noise's distribution",
                       msg(() -> simulate(T1IR(TI = [0.1], magnitude = false), (M0 = 1.0, Sf = -1.0, T1 = 1.0); snr = 50)))
        @test occursin("must be :rician or :gaussian", msg(() -> simulate(m, θ; snr = 50, noise = :poisson)))
        @test_throws ArgumentError simulate(m, θ)                            # no noise level
        @test_throws ArgumentError simulate(m, θ; noise = :rician, snr = 50, σ = 1.0)         # both
        # tissue presets and truth
        @test tissue(:WM) == (T1 = 1.084, T2 = 0.069)
        @test tissue(:CSF) == (T1 = 4.124, T2 = 1.75)
        @test nominal(R2Decay(TE = te), merge(tissue(:CSF), (T2 = 2.0,))).R2 == 0.5    # own value
        @test nominal(T2Decay(TE = te), (T2 = 0.06,); S0 = 500.0) == (S0 = 500.0, T2 = 0.06)
        @test nominal(R2Decay(TE = te), :WM) == (S0 = 1.0, R2 = 1 / 0.069)
        @test nominal(T1IR(TI = [0.1]), :GM; Sf = -0.95, M0 = 1000.0) == (M0 = 1000.0, Sf = -0.95, T1 = 1.82)
        @test_throws ArgumentError nominal(T1IR(TI = [0.1]), :GM)              # Sf is not a tissue value
        @test_throws ArgumentError nominal(m, :brain)                          # unknown tissue
        @test nominal(QMTSIR(; SIR_4POINT...), :SIR_lesion) ==
              (PSR = 0.05, R1f = 0.5, Sf = -1.0, M0f = 1.0, kmf = 20.0, R1m = :R1f, Sm = 0.83,
               Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        @test SIR_5POINT.TI == TI5 && SIR_5POINT.TD == TD5
        @test SIR_16POINT.TI == [0.010, 0.015, 0.021, 0.031, 0.045, 0.066, 0.097, 0.141, 0.206, 0.301,
                                 0.440, 0.643, 0.938, 1.370, 2.000, 10.0]
        @test SIR_16POINT.TD == fill(2.5, 16)
        m16 = QMTSIR(; SIR_16POINT...)                                      # kmf estimated
        t16 = nominal(m16, :SIR_WM_II)
        r16 = fit(m16, signal(m16, t16); kmf = Free())
        @test r16.params.PSR ≈ 0.12 rtol = 1e-6
        @test r16.params.kmf ≈ 10.0 rtol = 1e-6
        # maps, end to end: simulate, fit, compare by label (a label image made by hand: 0 the
        # background, then three tissues; each parameter a map, NaN in the background)
        labels = [j <= 2 ? 0 : 1 + (i - 1) * 3 ÷ 24 for i in 1:24, j in 1:20]
        @test sort(unique(labels)) == [0, 1, 2, 3]
        ts = [nominal(m, :GM; S0 = 1000.0), nominal(m, :WM; S0 = 1000.0), nominal(m, :blood; S0 = 1000.0)]
        tm = NamedTuple{keys(ts[1])}(Tuple([l == 0 ? NaN : Float64(ts[l][n]) for l in labels] for n in keys(ts[1])))
        @test isnan(tm.T2[1, 1]) && all(==(0.069), tm.T2[labels .== 2])
        data = simulate(m, tm; noise = :rician, snr = 100, rng = MersenneTwister(4))
        @test all(isnan, data[1, 1, :])                                      # background
        res = fit(m, data; mask = labels .> 0)
        c = compare(res, tm; labels)
        @test [r.group for r in c.rows if r.parameter == :T2] == [:all, 1, 2, 3]
        @test all(r -> r.converged == 1, c.rows)
        @test all(r -> abs(r.relbias) < 0.01 && r.relrmse < 0.03, c.rows)
        @test occursin("rel bias", sprint(show, MIME("text/plain"), c))
        # Dortch et al. 2018, Fig. 1 (RRMS lesion, Rician noise, SNR 200 at thermal equilibrium):
        # fixing kmf at 12.5 (4-point scheme) biases PSR below its true 5%; the 5-point does not
        for (mm, kw, lo, hi) in ((QMTSIR(; SIR_5POINT...), (kmf = Free(),), 4.95, 5.2),
                                 (QMTSIR(; SIR_4POINT...), (;), 4.65, 4.85))
            t = nominal(mm, :SIR_lesion)
            r = compare(fit(mm, simulate(mm, t; noise = :rician, snr = 200, n = 2000, rng = MersenneTwister(5)); kw...), t)
            psr = only(filter(x -> x.parameter == :PSR, r.rows))
            @test lo < 100 * psr.mean < hi
            @test psr.converged == 1
        end
    end

    @testset "crlb: bias, SD and RMSE predicted from derivatives" begin
        FD = EQUIP.ForwardDiff
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        tissues = (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)       # Dortch 2018, Table 1
        # Dortch et al. 2018, Eq. 5 with the MATLAB code's settings (signed signal, t_FSE =
        # 0.3 s, SNR 200, the bounds of its biased fits): the optima of Table 2's schemes,
        # 4-point (kmf fixed at 12.5: biased) and 5-point (kmf fitted: unbiased). The fit holds
        # the other fixed parameters at the truth's values, which are the model's here.
        bounds4 = (PSR = Free(lower = 0, upper = 1), R1f = Free(lower = 0, upper = 3),
                   Sf = Free(lower = -1.05, upper = 0), M0f = Free(lower = 0, upper = 10))
        m4 = QMTSIR(; SIR_4POINT..., magnitude = false)
        m5 = QMTSIR(; SIR_5POINT..., magnitude = false)
        eq5(c, m) = sum(abs2, values(c.relrmse)) * sum(m.TI .+ m.TD .+ 0.3)
        for cov in (:sandwich, :biased)
            @test maximum(eq5(crlb(m4, nominal(m4, t); snr = 200, kmf = 12.5, bounds4..., cov), m4)
                          for t in tissues) ≈ 0.243395 rtol = 2e-6
            @test maximum(eq5(crlb(m5, nominal(m5, t); snr = 200, kmf = Free(), cov), m5)
                          for t in tissues) ≈ 1.602062 rtol = 2e-6
        end
        # The 4-point fit is exact (4 observations, 4 free parameters): the sandwich and Eq. 3
        # are equal, and both are σ²(J₀ᵀJ₀)⁻¹ at the noise-free fit θ̂
        θ = nominal(m4, :SIR_WM_I)                                    # kmf = 5
        c = crlb(m4, θ; snr = 200, kmf = 12.5, bounds4...)
        e3 = crlb(m4, θ; snr = 200, kmf = 12.5, bounds4..., cov = :biased)
        @test collect(values(c.sd)) ≈ collect(values(e3.sd)) rtol = 1e-10
        F = keys(c.fit)
        g(x) = signal(m4, merge(θ, NamedTuple{F}(Tuple(x)), (kmf = 12.5,)))
        J0 = FD.jacobian(g, collect(values(c.fit)))
        @test collect(values(c.sd)) ≈ sqrt.(diag(inv(J0' * J0))) .* (1 / 200) rtol = 1e-8
        @test c.relbias.PSR ≈ -0.1409 atol = 1e-4                     # PSR 14% low in WM I
        @test c.fit.PSR - c.bias.PSR ≈ 0.2 && c.rmse.PSR ≈ hypot(c.bias.PSR, c.sd.PSR)
        @test isconverged(c.status) && isempty(c.onbound) && c.method === :sandwich
        # Nothing fixed wrong: no bias, and the classical CRLB σ²(JᵀJ)⁻¹ at the truth
        te = collect(0.01:0.01:0.32)
        u = crlb(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); σ = 10.0)
        J = FD.jacobian(x -> signal(T2Decay(TE = te), (S0 = x[1], T2 = x[2])), [1000.0, 0.08])
        @test u.bias == (S0 = 0.0, T2 = 0.0) && u.fit == (S0 = 1000.0, T2 = 0.08)
        @test collect(values(u.sd)) ≈ 10 .* sqrt.(diag(inv(J' * J))) rtol = 1e-10
        @test crlb(m4, θ; snr = 200, kmf = 5.0).bias.PSR == 0      # kmf at the truth
        @test crlb(m4, θ; snr = 200).sd == crlb(m4, θ; snr = 200, kmf = 5.0).sd   # the default
        # More observations than free parameters (16-point, kmf fixed): Eq. 3 ≤ sandwich
        ti16 = [round.(exp10.(range(log10(0.01), log10(2), 15)) .* 1e3) ./ 1e3; 10]
        m16 = QMTSIR(TI = ti16, TD = 2.5, magnitude = false)
        θ16 = nominal(m16, :SIR_WM_I)
        sw, b3 = crlb(m16, θ16; snr = 200, kmf = 12.5), crlb(m16, θ16; snr = 200, kmf = 12.5, cov = :biased)
        @test all(values(b3.sd) .< values(sw.sd))
        @test 100 * sw.relsd.PSR ≈ 2.879 rtol = 1e-3
        @test 100 * b3.relsd.PSR ≈ 2.719 rtol = 1e-3
        # ∇b by the implicit function theorem = central differences of noise-free fits (what
        # the MATLAB code does), through Eq. 3: (I + ∇b) F⁻¹ (I + ∇b)ᵀ. The differenced fits
        # converge tightly (crlb's own stop at xtol 1e-10, θ̂ to ~1e-9: too coarse for h = 1e-5)
        F16 = keys(sw.fit)
        x0 = [θ16[n] for n in F16]
        tight = LM(maxiter = 1000, xtol = 1e-13, ftol = 0.0, gtol = 0.0)
        function fitof(x)
            θx = merge(θ16, NamedTuple{F16}(Tuple(x)))
            starts = NamedTuple{F16}(map(n -> Free(start = θx[n]), F16))
            return collect(values(fit(m16, signal(m16, θx); solver = tight, starts...).params))
        end
        h = 1e-5 .* abs.(x0)
        D = reduce(hcat, (fitof(x0 .+ h[k] .* (1:4 .== k)) .- fitof(x0 .- h[k] .* (1:4 .== k))) ./ (2h[k])
                   for k in 1:4)
        Jt = FD.jacobian(x -> signal(m16, merge(θ16, NamedTuple{F16}(Tuple(x)))), x0)
        @test sqrt.(diag(D * inv(Jt' * Jt) * D')) ./ 200 ≈ collect(values(b3.sd)) rtol = 1e-5
        # Monte Carlo: least squares follows the sandwich (Eq. 3 is 5 standard errors lower for
        # PSR here, 12 with 20 000 fits), and centers on θ̂, not on the truth
        Y = simulate(m16, θ16; noise = :gaussian, snr = 200, n = 4000, rng = MersenneTwister(7))
        res = fit(m16, Y)
        x = res.params.PSR[isconverged(res)]
        @test length(x) == 4000
        @test abs(std(x) - sw.sd.PSR) < 4 * std(x) / sqrt(2 * length(x))
        @test abs(mean(x) - sw.fit.PSR) < 4 * std(x) / sqrt(length(x))
        # σ per observation: as one σ when all equal; with unequal σ, the unweighted fit's
        # sandwich is above the bound (JᵀΣ⁻¹J)⁻¹ that a weighted fit would reach
        @test crlb(m4, θ; σ = fill(0.005, 4), kmf = 12.5, bounds4...).sd == c.sd
        σs = range(5.0, 20.0, length = 32)
        w1 = crlb(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); σ = σs)
        w2 = crlb(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); σ = σs, cov = :biased)
        @test all(values(w2.sd) .< values(w1.sd))
        @test collect(values(w2.sd)) ≈ sqrt.(diag(inv(J' * Diagonal(1 ./ σs .^ 2) * J))) rtol = 1e-10
        # A parameter on a bound is held there: SD 0, the others without it
        cb = crlb(m4, θ; snr = 200, kmf = 12.5, bounds4..., Sf = Free(lower = -1.0, upper = 0))
        @test cb.onbound == [:Sf] && cb.fit.Sf == -1.0 && cb.sd.Sf == 0 && all(isfinite, values(cb.sd))
        # Undetermined parameters get Inf: 5 free parameters, 4 observations
        @test all(==(Inf), values(crlb(m4, θ; snr = 200, kmf = Free()).sd))
        @test cond(crlb(m4, θ; snr = 200, kmf = Free())) == Inf
        # cond: computed when called, from what the result keeps (the kernel's J₀ᵀJ₀); the
        # condition number of J₀ with its columns scaled by the true values, without the
        # parameters held on a bound
        @test cond(c) isa Float64 && !hasproperty(c, :cond)
        @test cond(c) == EQUIP._condition(EQUIP._crlb!(CRLBCache(m4, θ; kmf = 12.5, bounds4...), m4, EQUIP._crlb_truth(m4, θ), 0.005^2).G,
                                          [θ[n] for n in keys(c.fit)], falses(4), 4)
        θf = [Float64(θ[p]) for p in F]
        @test cond(c) ≈ cond(J0 * Diagonal(θf)) rtol = 1e-10
        @test cond(u) ≈ cond(J * Diagonal([1000.0, 0.08])) rtol = 1e-10
        jb = findfirst(==(:Sf), keys(cb.fit))
        kb = setdiff(1:4, jb)
        @test cond(cb) ≈ cond(FD.jacobian(g, collect(values(cb.fit)))[:, kb] * Diagonal(abs.(θf[kb]))) rtol = 1e-10
        # cor: the covariance scaled by the SDs, 1 on the diagonal (as Statistics' cor); NaN off
        # the diagonal for a parameter on a bound (SD 0) or undetermined (SD Inf)
        R = cor(c)
        @test R ≈ c.cov ./ (stderror(c) * stderror(c)') rtol = 1e-12
        @test R == R' && all(==(1), diag(R)) && all(x -> -1 <= x <= 1, R)
        Rb = cor(cb)
        @test all(isnan, Rb[jb, kb]) && all(isnan, Rb[kb, jb]) && Rb[jb, jb] == 1 && all(isfinite, Rb[kb, kb])
        Ru = cor(crlb(m4, θ; snr = 200, kmf = Free()))
        @test all(==(1), diag(Ru)) && all(isnan(Ru[i, j]) for i in 1:5, j in 1:5 if i != j)
        # The allocation-free kernel the protocol optimizer will call: crlb's own numbers; one
        # plan for another protocol with as many observations (static vectors too); no
        # allocation, for the sandwich and Eq. 3
        SV = EQUIP.SVector
        plan = CRLBCache(m4, θ; kmf = 12.5, bounds4...)
        tt = EQUIP._crlb_truth(m4, θ)
        k = EQUIP._crlb!(plan, m4, tt, 0.005^2)
        @test collect(k.fit) == collect(values(c.fit)) && collect(k.bias) == collect(values(c.bias))
        @test sqrt.(diag(k.cov)) ≈ collect(values(c.sd)) rtol = 1e-12
        ti, td = [0.02, 0.1, 0.3, 1.0], [1.0, 2.0, 3.0, 0.01]
        mb = QMTSIR(TI = SV{4}(ti), TD = SV{4}(td), magnitude = false)
        kb = EQUIP._crlb!(plan, mb, tt, 0.005^2)
        cb2 = crlb(QMTSIR(TI = ti, TD = td, magnitude = false), θ; snr = 200, kmf = 12.5, bounds4...)
        @test collect(kb.fit) ≈ collect(values(cb2.fit)) rtol = 1e-12
        @test sqrt.(diag(kb.cov)) ≈ collect(values(cb2.sd)) rtol = 1e-12
        allocs(p, mm, t, s2) = @allocated EQUIP._crlb!(p, mm, t, s2)
        allocs(plan, mb, tt, 0.005^2)
        @test allocs(plan, mb, tt, 0.005^2) == 0 && allocs(plan, m4, tt, 0.005^2) == 0
        plan16 = CRLBCache(m16, θ16; cov = :biased, kmf = 12.5)
        t16 = EQUIP._crlb_truth(m16, θ16)
        allocs(plan16, m16, t16, fill(0.005^2, 16))
        @test allocs(plan16, m16, t16, fill(0.005^2, 16)) == 0
        @test sqrt.(diag(EQUIP._crlb!(plan16, m16, t16, 0.005^2).cov)) ≈ collect(values(b3.sd)) rtol = 1e-12
        # Several tissues by broadcasting; the display
        cs = crlb.(Ref(m4), [nominal(m4, t) for t in tissues]; snr = 200, kmf = 12.5, bounds4...)
        @test length(cs) == 4 && cs[1].sd == c.sd
        s = sprint(show, MIME"text/plain"(), c)
        @test startswith(s, "crlb for QMTSIR, 4 observations, σ = 0.005, covariance: sandwich")
        @test occursin("fixed in the fit: kmf = 12.5 (true 5.0), R1m = :R1f, Sm = 0.83", s)
        @test occursin(r"\n  parameter +true +noise-free fit +bias +rel bias +SD +rel SD +RMSE +rel RMSE\n", s)
        @test occursin(r"\n  PSR +0.2 +0.1718 +-0.02818 +-14.1% ", s)
        @test occursin("on a bound in the noise-free fit (held there, SD 0): Sf",
                       sprint(show, MIME"text/plain"(), cb))
        @test repr(c) == "CRLBResult(QMTSIR: PSR, R1f, Sf, M0f)"
        # Mistakes
        @test occursin("`snr`, `σ` or `Σ`", msg(() -> crlb(m4, θ)))
        # `noise` is for where data are made: crlb (Gaussian, first order), its pieces, the fit
        # (least squares) and a scenario refuse it, saying why
        @test occursin("crlb takes no noise distribution", msg(() -> crlb(m4, θ; noise = :rician, snr = 200)))
        @test occursin("assume Gaussian noise", msg(() -> crlb(m4, Scenario(m4, θ); noise = :gaussian, snr = 200)))
        @test occursin("jacobian takes no noise", msg(() -> jacobian(m4, θ; noise = :rician)))
        @test occursin("Scenario takes no noise", msg(() -> Scenario(m4, θ; noise = :rician)))
        @test occursin("fit takes no noise", msg(() -> fit(m4, signal(m4, θ); noise = :rician)))
        @test occursin("`snr`, `σ` or `Σ`", msg(() -> crlb(m4, θ; snr = 200, σ = 0.01)))
        @test occursin("`σ` has 3 values", msg(() -> crlb(m4, θ; σ = [1, 2, 3])))
        @test occursin("`cov` must be", msg(() -> crlb(m4, θ; snr = 200, cov = :eq3)))
        @test occursin("sets its own options", msg(() -> crlb(m4, θ; snr = 200, solver = LM())))
        @test occursin("one voxel", msg(() -> crlb(m4, (; θ..., PSR = fill(0.1, 2)); snr = 200)))
        @test occursin("missing: Sf", msg(() -> crlb(m4, Base.structdiff(θ, NamedTuple{(:Sf,)}); snr = 200)))
    end

    @testset "montecarlo: measured bias, SD and RMSE, with crlb alongside" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        mt = T2Decay(TE = collect(0.01:0.01:0.32))
        θt = (S0 = 1000.0, T2 = 0.08)
        mc = montecarlo(mt, θt; σ = 10.0, n = 4000, noise = :gaussian, rng = MersenneTwister(3))
        @test mc.n == 4000 && mc.converged == 1 && mc.noise === :gaussian && mc.σ == fill(10.0, 32)
        @test length(mc.result.params.T2) == 4000
        # the statistics, by hand from the estimates
        x = mc.result.params.T2
        @test mc.mean.T2 ≈ mean(x) && mc.sd.T2 ≈ std(x) && mc.bias.T2 ≈ mean(x) - 0.08
        @test mc.rmse.T2 ≈ sqrt(mean(abs2, x .- 0.08)) && mc.relsd.T2 ≈ std(x) / 0.08
        @test mc.se_mean.T2 ≈ std(x) / sqrt(4000) && mc.se_sd.T2 ≈ std(x) / sqrt(2 * 3999)
        # nothing fixed wrong: the mean at the truth, the SD at the CRLB (within 4 standard errors)
        for p in (:S0, :T2)
            @test abs(mc.bias[p]) < 4 * mc.se_mean[p]
            @test abs(mc.sd[p] - mc.crlb.sd[p]) < 4 * mc.se_sd[p]
        end
        # the same generator and seed, the same result; the crlb attached is crlb's
        @test montecarlo(mt, θt; σ = 10.0, n = 4000, noise = :gaussian, rng = MersenneTwister(3)).sd == mc.sd
        @test mc.crlb.sd == crlb(mt, θt; σ = 10.0).sd
        # a fixed value wrong (kmf 12.5; WM I has 5): the fits center on crlb's noise-free fit,
        # not on the truth, with crlb's SD
        m4 = QMTSIR(; SIR_4POINT...)
        θ = nominal(m4, :SIR_WM_I)
        mq = montecarlo(m4, θ; snr = 200, n = 4000, noise = :gaussian, kmf = 12.5, rng = MersenneTwister(4))
        @test mq.relbias.PSR < -0.13
        for p in keys(mq.sd)
            @test abs(mq.mean[p] - mq.crlb.fit[p]) < 4 * mq.se_mean[p]
            @test abs(mq.sd[p] - mq.crlb.sd[p]) < 4 * mq.se_sd[p]
        end
        # compare on the same estimates gives montecarlo's statistics (one true value), the
        # relative bias with the bias's sign, also for Sf's negative truth
        cq = compare(mq.result, θ)
        for r in cq.rows
            p = r.parameter
            @test r.bias ≈ mq.bias[p] rtol = 1e-10
            @test r.relbias ≈ mq.relbias[p] rtol = 1e-10
            @test r.sd ≈ mq.sd[p] rtol = 1e-10
            @test r.rmse ≈ mq.rmse[p] rtol = 1e-10
        end
        @test θ.Sf < 0 && sign(only(r.relbias for r in cq.rows if r.parameter == :Sf)) == sign(mq.bias.Sf) != 0
        # the correlations of the estimates: measured (Statistics' cor, as vcov's scaled), and
        # near crlb's within 4 standard errors, (1 − r²)/√n
        Rm, Rp = cor(mq), cor(mq.crlb)
        @test Rm ≈ vcov(mq) ./ (sqrt.(diag(vcov(mq))) * sqrt.(diag(vcov(mq)))') rtol = 1e-12
        @test all(abs.(Rm - Rp) .< 4 .* (1 .- Rp .^ 2) ./ sqrt(4000) .+ 1e-12)
        # the noise, the user's choice (no default, checked before anything runs); the fit's
        # changes and options (precision) as for `fit`
        @test montecarlo(m4, θ; noise = :rician, snr = 200, n = 50).noise === :rician
        @test montecarlo(m4, θ; noise = :gaussian, snr = 200, n = 50).noise === :gaussian
        @test occursin("give the noise's distribution", msg(() -> montecarlo(m4, θ; snr = 200, n = 50)))
        @test occursin("must be :rician or :gaussian", msg(() -> montecarlo(m4, θ; noise = :magnitude, snr = 200, n = 50)))
        m32 = montecarlo(m4, θ; noise = :rician, snr = 200, n = 200, kmf = 5.0, precision = Float32)
        @test eltype(m32.result.params.PSR) == Float32 && m32.crlb.bias.PSR == 0
        # display
        s = sprint(show, MIME"text/plain"(), mq)
        @test occursin(r"^montecarlo for QMTSIR, 4 observations, σ = 0.005, gaussian noise: 4000 noisy copies, [0-9.]+% converged", s)
        @test occursin("fixed in the fit: kmf = 12.5 (true 5.0)", s)
        @test occursin(r"\n  parameter +true +mean +rel bias +crlb +SD +crlb +SD/crlb +rel RMSE +crlb\n", s)
        @test repr(mq) == "MonteCarloResult(QMTSIR: PSR, R1f, Sf, M0f; n = 4000)"
        # mistakes
        @test occursin("at least 2", msg(() -> montecarlo(mt, θt; noise = :rician, σ = 10.0, n = 1)))
        @test occursin("one voxel", msg(() -> montecarlo(mt, (S0 = fill(1000.0, 3), T2 = 0.08); noise = :rician, σ = 10.0)))
        @test montecarlo(mt, θt; noise = :rician, σ = fill(10.0, 32), n = 50).σ == fill(10.0, 32)   # σ per observation
        @test occursin("`snr`, `σ` or `Σ`", msg(() -> montecarlo(mt, θt; noise = :rician)))
        @test occursin("mask", msg(() -> montecarlo(mt, θt; noise = :rician, σ = 10.0, mask = trues(2))))
        @test occursin("missing: T2", msg(() -> montecarlo(mt, (S0 = 1000.0,); noise = :rician, σ = 10.0)))
    end

    @testset "sweeps: crlb and montecarlo over noise levels and true values" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        m4 = QMTSIR(; SIR_4POINT...)
        θ = nominal(m4, :SIR_WM_II)                                   # kmf = 10
        # crlb against the true kmf, the fit keeping 12.5 (Dortch 2018 Fig. 2B): one crlb each
        s = crlb(m4, θ; snr = 200, kmf = 12.5, vary = :kmf => 5.0:2.5:20.0)
        @test s isa Sweep && length(s) == 7 && eltype(s) <: CRLBResult
        @test s.settings[1] == (kmf = 5.0,) && s.settings[end] == (kmf = 20.0,)
        @test s[3].bias == crlb(m4, (; θ..., kmf = 10.0); snr = 200, kmf = 12.5).bias
        @test s[4].bias.PSR == 0                                      # true kmf = the assumed 12.5
        # each setting as crlb! gives it, without the truth checked again (numbers varied) or
        # with it (a tie varied); cond as crlb's
        @test s[2].sd == crlb!(CRLBCache(m4, θ; kmf = 12.5), (; θ..., kmf = 7.5); snr = 200).sd
        @test cond(s[2]) == cond(crlb(m4, (; θ..., kmf = 7.5); snr = 200, kmf = 12.5))
        @test cond.(s.results)[2] == cond(s[2])
        st = crlb(m4, (; θ..., R1m = 1.0); snr = 200, vary = :R1m => [:R1f, 1.2])
        @test st[1].bias == crlb(m4, (; θ..., R1m = 1.0); snr = 200, R1m = 1.0, vary = :R1m => [θ.R1f]).results[1].bias
        @test st[2].sd == crlb!(CRLBCache(m4, (; θ..., R1m = 1.0)), (; θ..., R1m = 1.2); snr = 200).sd
        @test [r.fixed.kmf for r in s] == fill(12.5, 7)               # the fit never changes
        @test length(s.rows) == 7 * 4 && s.rows[1].kmf == 5.0 && s.rows[1].parameter === :PSR
        @test s.rows[1].truth == 0.12 && s.rows[1].relbias == s[1].relbias.PSR
        # over the SNR: the SD scales as 1/SNR (first order in σ), the bias not at all
        t = crlb(m4, θ; snr = [100, 200, 400], kmf = 12.5)
        @test t.settings == [(snr = 100,), (snr = 200,), (snr = 400,)]
        @test t[1].sd.PSR ≈ 2 * t[2].sd.PSR rtol = 1e-12
        @test t[1].bias == t[3].bias && t[1].bias.PSR < 0
        # every combination, the first keyword varying slowest
        u = crlb(m4, θ; snr = [100, 200], vary = (kmf = [5.0, 10.0], Sm = [0.8, 0.83]))
        @test length(u) == 8 && u.settings[1] == (snr = 100, kmf = 5.0, Sm = 0.8)
        @test u.settings[2] == (snr = 100, kmf = 5.0, Sm = 0.83) && u.settings[end] == (snr = 200, kmf = 10.0, Sm = 0.83)
        # several parameters as a vector of pairs: the same settings, in the same order
        u2 = crlb(m4, θ; snr = [100, 200], vary = [:kmf => [5.0, 10.0], :Sm => [0.8, 0.83]])
        @test u2.settings == u.settings && [r.sd for r in u2] == [r.sd for r in u]
        @test occursin("vary = :kmf => 5:2.5:20", msg(() -> crlb(m4, θ; snr = 200, vary = 5:10)))
        # the fit's settings are keywords, as for `fit`; the old form and unknown names explained
        @test occursin("as keywords, as for `fit`", msg(() -> crlb(m4, θ; snr = 200, fit = (kmf = 5.0,))))
        @test occursin("side by side as scenarios", msg(() -> crlb(m4, θ; snr = 200, fits = (a = (;),))))
        @test occursin("neither a parameter of QMTSIR", msg(() -> crlb(m4, θ; snr = 200, kfm = 5.0)))
        @test occursin("nor an option of montecarlo", msg(() -> montecarlo(m4, θ; noise = :rician, snr = 200, kfm = 5.0)))
        @test occursin("nor an option of jacobian", msg(() -> jacobian(m4, θ; snr = 200)))
        # montecarlo sweeps: each setting is the one-case run from the same seed, and every
        # setting sees the same noise, so adding a setting changes no other result
        a = montecarlo(m4, θ; noise = :rician, snr = [100, 200], n = 500, rng = MersenneTwister(1))
        b = montecarlo(m4, θ; noise = :rician, snr = [100, 200, 400], n = 500, rng = MersenneTwister(1))
        @test a[1].sd == b[1].sd && a[2].sd == b[2].sd
        @test a[2].sd == montecarlo(m4, θ; noise = :rician, snr = 200, n = 500, rng = MersenneTwister(1)).sd
        @test a[1].crlb.sd == crlb(m4, θ; snr = 100).sd
        # the bias curve measured matches crlb's (Gaussian noise; within 4 standard errors)
        v = montecarlo(m4, θ; snr = 200, n = 2000, noise = :gaussian, kmf = 12.5,
                       vary = :kmf => [5.0, 20.0], rng = MersenneTwister(2))
        for r in v, p in keys(r.mean)
            @test abs(r.mean[p] - r.crlb.fit[p]) < 4 * r.se_mean[p]
        end
        @test v.rows[1].crlb_relbias == v[1].crlb.relbias.PSR && v.rows[1].se_sd == v[1].se_sd.PSR
        # display
        sc = sprint(show, MIME"text/plain"(), s)
        @test startswith(sc, "crlb sweep for QMTSIR, 4 observations: 7 settings of kmf, σ = 0.005, covariance: sandwich")
        @test occursin("fixed in the fit: kmf = 12.5, R1m = :R1f", sc)
        @test occursin(r"\n  kmf +parameter +noise-free fit +rel bias +rel SD +rel RMSE\n  5.0 +PSR ", sc)
        sm = sprint(show, MIME"text/plain"(), v)
        @test startswith(sm, "montecarlo sweep for QMTSIR, 4 observations: 2 settings of kmf, " *
                             "2000 noisy copies each, gaussian noise")
        @test occursin(r"\n  kmf +parameter +rel bias +crlb +SD/crlb +rel RMSE +crlb +converged\n", sm)
        @test repr(s) == "Sweep(7 crlb results: kmf)" && repr(a) == "Sweep(2 montecarlo results: snr)"
        # mistakes
        @test occursin("not a parameter", msg(() -> crlb(m4, θ; snr = 200, vary = :kfm => [1.0])))
        @test occursin("vector or a range", msg(() -> crlb(m4, θ; snr = 200, vary = :kmf => 5.0)))
        @test occursin("vector or a range", msg(() -> montecarlo(m4, θ; noise = :rician, snr = 200, vary = :kmf => Float64[])))
    end

    @testset "scenarios side by side, and data from another model" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        m5 = QMTSIR(; SIR_5POINT...)
        θ = nominal(m5, :SIR_WM_I)                                    # kmf = 5
        fs = (fixed = Scenario(m5, θ; kmf = 12.5), free = Scenario(m5, θ; kmf = Free()))
        # crlb: one result per scenario, each as alone; the scenarios vary fastest
        s = crlb(m5, fs; snr = 200)
        @test s.settings == [(fit = :fixed,), (fit = :free,)]
        @test s[1].bias == crlb(m5, θ; snr = 200, kmf = 12.5).bias && s[2].sd == crlb(m5, fs.free; snr = 200).sd
        @test keys(s[2].fit) == (:PSR, :R1f, :Sf, :M0f, :kmf) && s.rows[end].fit === :free
        t = crlb(m5, fs; snr = [100, 200])
        @test t.settings[1:2] == [(snr = 100, fit = :fixed), (snr = 100, fit = :free)]
        # montecarlo: the same noisy copies fitted each way (a paired comparison): each equals
        # the run with that one scenario and the same generator
        v = montecarlo(m5, fs; noise = :rician, snr = 200, n = 300, rng = MersenneTwister(1))
        @test v[2].sd == montecarlo(m5, θ; noise = :rician, snr = 200, n = 300, kmf = Free(), rng = MersenneTwister(1)).sd
        @test v[1].sd == montecarlo(m5, θ; noise = :rician, snr = 200, n = 300, kmf = 12.5, rng = MersenneTwister(1)).sd
        sv = sprint(show, MIME"text/plain"(), v)
        @test occursin("fixed in fit `fixed`: kmf = 12.5 (true 5.0)", sv) && occursin("fixed in fit `free`: R1m = :R1f", sv)
        @test occursin(r"\n  fixed +PSR ", sv) && occursin(r"\n  free +PSR ", sv)
        @test occursin("the scenario is for QMTSIR", msg(() -> crlb(T2Decay(TE = TD5), fs; σ = 1.0)))
        # data from another model: a mono-exponential fit to bi-exponential data. The bias is
        # against the reference values given (here the long T2); crlb's noise-free fit is the
        # fit of the noise-free bi-exponential signal
        te = collect(0.01:0.01:0.32)
        mt, mb = T2Decay(TE = te), T2BiexpDecay(TE = te)
        θb = (S0 = 1000.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)
        ref = (S0 = 1000.0, T2 = 0.08)
        c = crlb(mt, ref; snr = 100, simulate_with = mb => θb)
        @test c.fit.T2 ≈ fit(mt, signal(mb, θb)).params.T2 rtol = 1e-8
        @test c.bias.T2 == c.fit.T2 - 0.08 && c.fit.T2 < 0.08 && c.σ[1] == 10   # snr: the data's S0
        @test occursin("data from T2BiexpDecay: S0 = 1000.0, fshort = 0.2", sprint(show, MIME"text/plain"(), c))
        # Monte Carlo (Gaussian noise: the Rician floor at the late echoes would lengthen T2)
        mc = montecarlo(mt, ref; snr = 100, n = 3000, noise = :gaussian, simulate_with = mb => θb,
                        rng = MersenneTwister(2))
        @test abs(mc.mean.T2 - c.fit.T2) < 4 * mc.se_mean.T2
        @test abs(mc.sd.T2 - c.sd.T2) < 4 * mc.se_sd.T2
        @test montecarlo(mt, ref; noise = :rician, snr = 100, n = 50, simulate_with = mb => θb).noise === :rician
        # vary changes the data's truth; the reference stays
        w = crlb(mt, ref; snr = 100, simulate_with = mb => θb, vary = :fshort => [0.1, 0.3])
        @test w[1].fit.T2 > w[2].fit.T2 && w[1].truth == ref && w[2].data.second.fshort == 0.3
        @test occursin("not a parameter of T2BiexpDecay",
                       msg(() -> crlb(mt, ref; snr = 100, simulate_with = mb => θb, vary = :T2 => [0.07])))
        # mistakes
        @test occursin("the sandwich only", msg(() -> crlb(mt, ref; snr = 100, simulate_with = mb => θb, cov = :biased)))
        @test occursin("observations", msg(() -> crlb(mt, ref; snr = 100, simulate_with = T2BiexpDecay(TE = te[1:10]) => θb)))
        @test occursin("simulate_with = model => truth", msg(() -> crlb(mt, ref; snr = 100, simulate_with = mb)))
        # the kernel with another data model allocates nothing either
        plan = CRLBCache(mt, ref)
        tr, tb = EQUIP._crlb_truth(mt, ref), EQUIP._crlb_truth(mb, θb)
        allocs(p, a, b) = @allocated EQUIP._crlb!(p, mt, a, 100.0, mb, b)
        allocs(plan, tr, tb)
        @test allocs(plan, tr, tb) == 0
    end

    @testset "noise: one σ, one per observation, a full Σ; informationmatrix, vcov, stderror" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        FD = EQUIP.ForwardDiff
        te = collect(0.01:0.01:0.32)
        mt = T2Decay(TE = te)
        θ = (S0 = 1000.0, T2 = 0.08)
        J = FD.jacobian(x -> signal(mt, (S0 = x[1], T2 = x[2])), [1000.0, 0.08])
        # the same noise in three forms: the same answer
        σv = range(5.0, 20.0, length = 32)
        for cov in (:sandwich, :biased)
            @test stderror(crlb(mt, θ; Σ = 100.0 * I(32), cov)) ≈ stderror(crlb(mt, θ; σ = 10.0, cov)) rtol = 1e-12
            @test stderror(crlb(mt, θ; Σ = Diagonal(σv .^ 2), cov)) ≈ stderror(crlb(mt, θ; σ = σv, cov)) rtol = 1e-12
        end
        # correlated noise: the bound (JᵀΣ⁻¹J)⁻¹, and the unweighted fit's sandwich
        # (JᵀJ)⁻¹JᵀΣJ(JᵀJ)⁻¹, above it
        Σ = [100.0 * 0.5^abs(i - j) for i in 1:32, j in 1:32]
        s, b = crlb(mt, θ; Σ), crlb(mt, θ; Σ, cov = :biased)
        @test vcov(b) ≈ inv(J' * (Σ \ J)) rtol = 1e-10
        A = inv(J' * J)
        @test vcov(s) ≈ A * (J' * Σ * J) * A rtol = 1e-10
        @test all(stderror(s) .> stderror(b)) && s.Σ == Σ && crlb(mt, θ; σ = 10.0).Σ === nothing
        @test occursin("σ = 10.0, correlated (Σ given)", sprint(show, MIME"text/plain"(), s))
        # Monte Carlo with that correlated noise follows the sandwich
        mc = montecarlo(mt, θ; Σ, n = 4000, noise = :gaussian, rng = MersenneTwister(1))
        @test all(abs.(stderror(mc) .- stderror(s)) .< 4 .* collect(values(mc.se_sd)))
        @test vcov(mc) ≈ cov(hcat(mc.result.params.S0, mc.result.params.T2)) && stderror(mc) == collect(values(mc.sd))
        # simulate: noise with the covariance asked for, and one SD per observation
        Y = simulate(mt, (S0 = 0.0, T2 = 0.08); Σ, n = 40_000, noise = :gaussian, rng = MersenneTwister(2))
        @test cov(Y)[1, 1] ≈ 100 rtol = 0.03
        @test cov(Y)[1, 2] ≈ 50 rtol = 0.06
        @test abs(cov(Y)[1, 8]) < 2.5                                   # 0.5⁷ × 100 ≈ 0.8
        Y = simulate(mt, (S0 = 0.0, T2 = 0.08); σ = σv, n = 40_000, noise = :gaussian, rng = MersenneTwister(3))
        @test std(Y[:, 1]) ≈ 5 rtol = 0.03
        @test std(Y[:, end]) ≈ 20 rtol = 0.03
        @test simulate(mt, θ; noise = :rician, σ = fill(10.0, 32), n = 3, rng = MersenneTwister(4)) ==
              simulate(mt, θ; noise = :rician, σ = 10.0, n = 3, rng = MersenneTwister(4))   # draw for draw
        # informationmatrix: the Fisher (expected) and observed information. Nothing fixed wrong:
        # its inverse is the CRLB; an exact fit: the two agree; more points and kmf wrong: they
        # differ, and the sandwich is inv(I_obs) I inv(I_obs)
        c1 = crlb(mt, θ; σ = 10.0)
        @test informationmatrix(c1) ≈ J' * J / 100 rtol = 1e-10
        @test inv(informationmatrix(c1)) ≈ vcov(c1) rtol = 1e-10
        m4 = QMTSIR(; SIR_4POINT...)
        c4 = crlb(m4, nominal(m4, :SIR_WM_I); snr = 200, kmf = 12.5)
        @test informationmatrix(c4) ≈ informationmatrix(c4; expected = false) rtol = 1e-6
        ti16 = [round.(exp10.(range(log10(0.01), log10(2), 15)) .* 1e3) ./ 1e3; 10]
        m16 = QMTSIR(TI = ti16, TD = 2.5)
        θ16 = nominal(m16, :SIR_WM_I)
        c16 = crlb(m16, θ16; snr = 200, kmf = 12.5)
        Ie, Io = informationmatrix(c16), informationmatrix(c16; expected = false)
        @test !isapprox(Ie, Io; rtol = 1e-3)
        @test vcov(c16) ≈ inv(Io) * Ie * inv(Io) rtol = 1e-8
        # the same at the values θ̂ (every value: the fitted ones and the fit's fixed ones)
        θ̂16 = merge(c16.fixed, c16.fit)
        @test informationmatrix(m16, θ̂16; σ = c16.σ[1]) ≈ Ie rtol = 1e-12
        @test informationmatrix(m16, θ̂16; σ = c16.σ[1], expected = false, y = signal(m16, θ16)) ≈ Io rtol = 1e-12
        # at the truth, where the data are fitted exactly, the two are equal
        @test informationmatrix(m16, θ16; snr = 200, expected = false) ≈ informationmatrix(m16, θ16; snr = 200) rtol = 1e-12
        # the kernel with a full Σ allocates nothing
        plan = CRLBCache(mt, θ; cov = :biased)
        tr = EQUIP._crlb_truth(mt, θ)
        full = EQUIP._kernelnoise(noisecovariance(mt, θ; Σ), 2)
        allocs(p, t, n) = @allocated EQUIP._crlb!(p, mt, t, n)
        allocs(plan, tr, full)
        @test allocs(plan, tr, full) == 0
        # mistakes
        @test occursin("`snr`, `σ` or `Σ`", msg(() -> crlb(mt, θ; σ = 10.0, Σ)))
        @test occursin("`Σ` is 3×3", msg(() -> crlb(mt, θ; Σ = Matrix(1.0I, 3, 3))))
        @test occursin("positive definite", msg(() -> crlb(mt, θ; Σ = -Matrix(1.0I, 32, 32))))
        @test occursin("symmetric", msg(() -> crlb(mt, θ; Σ = Σ + triu(ones(32, 32), 1))))
        @test occursin("give `Σ`", msg(() -> crlb(mt, θ; σ = Σ)))
        @test occursin("`snr`, `σ` or `Σ`", msg(() -> simulate(mt, θ; noise = :rician, snr = 10, Σ)))
        @test occursin("`σ` has 3 values", msg(() -> simulate(mt, θ; noise = :rician, σ = [1.0, 2.0, 3.0])))
    end

    @testset "averages: σ/√NEX, complex or magnitude averaging; crlb, montecarlo, design" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        # Diffusion with more averages at high b: SNR 20 per average at b = 0, 1.8σ at b = 3000
        b = [0.0, 500, 1000, 1500, 2000, 2500, 3000]
        nex = [1, 1, 2, 2, 4, 4, 8]
        md = ADCDecay(; b)
        θ = (S0 = 1.0, ADC = 0.8e-3)
        σ1 = 0.05
        # Averaged in complex (or real-valued data): σ/√NEX, the same data draw for draw as σ
        # given per observation; with `snr` (one average's σ = S0/snr) too; one average changes
        # nothing
        @test simulate(md, θ; σ = σ1, averages = nex, noise = :rician, averaging = :complex, n = 50, rng = MersenneTwister(1)) ==
              simulate(md, θ; σ = σ1 ./ sqrt.(nex), noise = :rician, n = 50, rng = MersenneTwister(1))
        @test simulate(md, θ; snr = 20, averages = nex, noise = :gaussian, n = 50, rng = MersenneTwister(1)) ==
              simulate(md, θ; σ = (1.0 / 20) ./ sqrt.(nex), noise = :gaussian, n = 50, rng = MersenneTwister(1))
        @test simulate(md, θ; σ = σ1, averages = 1, noise = :gaussian, rng = MersenneTwister(1)) ==
              simulate(md, θ; σ = σ1, noise = :gaussian, rng = MersenneTwister(1))
        # Magnitude images averaged: the SD falls with √NEX, the floor does not (the mean of one
        # average's Rician values); averaged in complex, the floor falls too
        Ym = simulate(md, θ; σ = σ1, averages = nex, noise = :rician, averaging = :magnitude, n = 40_000, rng = MersenneTwister(2))
        Yc = simulate(md, θ; σ = σ1, averages = nex, noise = :rician, averaging = :complex, n = 40_000, rng = MersenneTwister(2))
        Y1 = simulate(md, θ; σ = σ1, noise = :rician, n = 40_000, rng = MersenneTwister(3))
        @test mean(Ym[:, end]) ≈ mean(Y1[:, end]) rtol = 0.01
        @test std(Ym[:, end]) ≈ std(Y1[:, end]) / sqrt(8) rtol = 0.03
        @test mean(Yc[:, end]) < mean(Ym[:, end]) - 0.005
        # crlb: σ/√NEX (first order, however the averages were combined), as σ per observation;
        # the averages recorded and shown
        c = crlb(md, θ; σ = σ1, averages = nex)
        @test c.sd.ADC ≈ crlb(md, θ; σ = σ1 ./ sqrt.(nex)).sd.ADC rtol = 1e-12
        @test c.σ ≈ σ1 ./ sqrt.(nex) rtol = 1e-12
        @test c.averages == nex && crlb(md, θ; σ = σ1).averages === nothing
        @test crlb(md, θ; σ = σ1, averages = 2.0).averages == fill(2, 7)
        @test noisecovariance(md, θ; snr = 20, averages = nex) ≈ Diagonal((1 / 20)^2 ./ nex) rtol = 1e-12
        @test informationmatrix(md, θ; σ = σ1, averages = nex) ≈ informationmatrix(md, θ; σ = σ1 ./ sqrt.(nex)) rtol = 1e-12
        @test occursin("σ = 0.05 per average, 1 to 8 averages, covariance", sprint(show, MIME"text/plain"(), c))
        cs = crlb(md, θ; snr = [10, 20], averages = 4)
        @test cs[2].sd.ADC ≈ crlb(md, θ; snr = 40).sd.ADC rtol = 1e-12       # 4 averages: SNR × 2
        @test occursin("2 settings of snr, 4 averages, covariance", sprint(show, MIME"text/plain"(), cs))
        # montecarlo: the averages to the data and to crlb alongside; the floor magnitude
        # averaging keeps is a bias crlb does not predict (−1.96% ± 0.04% in ADC with 20 000
        # copies; −1.7% ± 0.14% here), the SD as predicted
        mc = montecarlo(md, θ; σ = σ1, averages = nex, noise = :rician, averaging = :magnitude, n = 2000, rng = MersenneTwister(4))
        @test mc.averages == nex && mc.averaging === :magnitude && mc.crlb.averages == nex
        @test mc.crlb.sd.ADC ≈ c.sd.ADC rtol = 1e-12
        @test mc.σ ≈ σ1 ./ sqrt.(nex) rtol = 1e-12
        @test mc.sd.ADC / c.sd.ADC ≈ 1 atol = 0.05
        @test mc.relbias.ADC < -0.01
        @test occursin("σ = 0.05 per average, 1 to 8 averages, rician noise, magnitude averaging: 2000 noisy copies",
                       sprint(show, MIME"text/plain"(), mc))
        @test montecarlo(md, θ; σ = σ1, averages = nex, noise = :rician, averaging = :complex, n = 300, rng = MersenneTwister(5)).sd ==
              montecarlo(md, θ; σ = σ1 ./ sqrt.(nex), noise = :rician, n = 300, rng = MersenneTwister(5)).sd
        @test montecarlo(md, θ; σ = σ1, averages = nex, noise = :gaussian, n = 50).averaging === nothing
        sw = montecarlo(md, θ; snr = [10, 20], averages = nex, noise = :rician, averaging = :magnitude, n = 100, rng = MersenneTwister(6))
        @test sw[2].averaging === :magnitude && sw[2].averages == nex
        @test occursin("rician noise, magnitude averaging, 1 to 8 averages", sprint(show, MIME"text/plain"(), sw))
        # Sets of scenarios, and the design (an unbiased scenario: Eq. 5 is all variance, so 4
        # averages divide it by 4; the scan time is the user's, here left unchanged)
        m4 = QMTSIR(; SIR_4POINT...)
        set = scenarios(QMTSIR, [:SIR_WM_I, :SIR_lesion]; kmf = 12.5)
        @test collect(values(crlb(m4, set; snr = 200, averages = 4)[2].sd)) ≈ collect(values(crlb(m4, set[2]; snr = 400).sd)) rtol = 1e-12
        a4 = [1, 2, 2, 4]
        ms = montecarlo(m4, set; snr = 200, averages = a4, noise = :rician, averaging = :complex, n = 50, rng = MersenneTwister(7))
        @test ms[2].averages == a4
        @test collect(values(ms[2].crlb.sd)) ≈ collect(values(crlb(m4, set[2]; snr = 200, averages = a4).sd)) rtol = 1e-12
        sU = Scenario(m4, nominal(m4, :SIR_WM_I))
        space = (TI = (0.01, 10.0), TD = (0.01, 10.0))
        st(m) = sum(m.TI .+ m.TD .+ 0.3)
        d1 = ProtocolDesign(m4, [sU]; snr = 200, acquisition = space, scantime = st)
        d4 = ProtocolDesign(m4, [sU]; snr = 200, averages = 4, acquisition = space, scantime = st)
        @test d4(m4) ≈ d1(m4) / 4 rtol = 1e-9
        # averages that differ per point: Eq. 5 from crlb with them (unbiased: Σ rel RMSE² × T),
        # as the search evaluates it, without allocation
        dv = ProtocolDesign(m4, [sU]; snr = 200, averages = a4, acquisition = space, scantime = st)
        cv = crlb(m4, sU; snr = 200, averages = a4)
        @test dv(m4) ≈ sum(abs2, collect(values(cv.relrmse))) * st(m4) rtol = 1e-9
        xv = [m4.TI; m4.TD]
        allocs(d, x) = @allocated d(x)
        allocs(dv, xv)
        @test allocs(dv, xv) == 0
        # Mistakes
        @test occursin("`averages` with `Σ`", msg(() -> simulate(md, θ; Σ = Matrix(0.0025I, 7, 7), averages = nex, noise = :gaussian)))
        @test occursin("`averages` with `Σ`", msg(() -> crlb(md, θ; Σ = Matrix(0.0025I, 7, 7), averages = nex)))
        @test occursin("give how they were combined", msg(() -> simulate(md, θ; σ = σ1, averages = nex, noise = :rician)))
        @test occursin("give how they were combined", msg(() -> montecarlo(md, θ; σ = σ1, averages = nex, noise = :rician, n = 10)))
        @test occursin("leave it out", msg(() -> simulate(md, θ; σ = σ1, averages = nex, noise = :gaussian, averaging = :complex)))
        @test occursin("give `averages` too", msg(() -> simulate(md, θ; σ = σ1, noise = :rician, averaging = :magnitude)))
        @test occursin("must be :complex or :magnitude", msg(() -> simulate(md, θ; σ = σ1, averages = nex, noise = :rician, averaging = :sum)))
        @test occursin("`averages` has 2 values", msg(() -> crlb(md, θ; σ = σ1, averages = [1, 2])))
        @test occursin("whole numbers, at least 1", msg(() -> crlb(md, θ; σ = σ1, averages = 1.5)))
        @test occursin("whole numbers, at least 1", msg(() -> crlb(md, θ; σ = σ1, averages = [0, 1, 1, 1, 1, 1, 1])))
        @test occursin("crlb takes no averaging", msg(() -> crlb(md, θ; σ = σ1, averages = nex, averaging = :magnitude)))
        @test occursin("crlb takes no averaging", msg(() -> crlb(m4, set; snr = 200, averages = 4, averaging = :complex)))
        @test occursin("ProtocolDesign takes no averaging", msg(() -> ProtocolDesign(m4, [sU]; snr = 200, averaging = :magnitude, acquisition = space, scantime = st)))
        @test occursin("fit takes no averages", msg(() -> fit(md, signal(md, θ); averages = nex)))
        @test occursin("give `weights = nex`", msg(() -> fit(md, signal(md, θ); averages = nex)))
        @test occursin("Scenario takes no averages", msg(() -> Scenario(md, θ; averages = nex)))
    end

    @testset "weighted fits: fit, statistics, crlb, leakage, montecarlo, design" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        # T2 decay with σ rising over the echoes, 5 to 40: W = Σ⁻¹ (wᵢ = 1/σᵢ²) is the efficient fit
        te = collect(0.01:0.01:0.32)
        mt = T2Decay(TE = te)
        θ = (S0 = 1000.0, T2 = 0.08)
        σv = collect(range(5.0, 40.0; length = 32))
        W = 1 ./ σv .^ 2
        Y = simulate(mt, θ; σ = σv, noise = :gaussian, n = 2000, rng = MersenneTwister(1))
        y = Y[1, :]
        # Weights all 1 are the unweighted fit, bit for bit; scaled weights give the same estimates
        r0, r1 = fit(mt, y), fit(mt, y; weights = ones(32))
        @test r1.params == r0.params && r1.sse == r0.sse
        @test fit(mt, Y[1:50, :]; weights = ones(32)).params == fit(mt, Y[1:50, :]).params
        rw, rs = fit(mt, y; weights = W), fit(mt, y; weights = 7 .* W)
        @test collect(values(rw.params)) ≈ collect(values(rs.params)) rtol = 1e-12
        @test rw.settings.weights == W && !haskey(r0.settings, :weights)
        # At the answer the weighted gradient JᵀW(g − y) is 0; a weight 0 leaves an observation out
        J = jacobian(mt, merge(θ, rw.params))
        rr = signal(mt, rw.params) .- y
        @test norm(J' * (W .* rr)) <= 1e-6 * norm(J' * (W .* abs.(rr)))
        w0 = ones(32); w0[[3, 10, 30]] .= 0
        keep = setdiff(1:32, [3, 10, 30])
        rz, rd = fit(mt, y; weights = w0), fit(T2Decay(TE = te[keep]), y[keep])
        @test collect(values(rz.params)) ≈ collect(values(rd.params)) rtol = 1e-8
        @test nobs(rz) == 29 && dof_residual(rz) == 27
        @test residualsd(rz) ≈ residualsd(rd) rtol = 1e-8
        @test loglikelihood(rz) ≈ loglikelihood(rd) rtol = 1e-8
        # Statistics: σ̂² = Σwr²/(N − P) (about 1 with wᵢ = 1/σᵢ²), Cov = σ̂²(JᵀWJ)⁻¹, ℓ the Gaussian
        # log-likelihood for variances s²/wᵢ (the same for any scaling of the weights)
        @test residualsd(rw) ≈ sqrt(sum(W .* rr .^ 2) / 30) rtol = 1e-10
        @test vcov(rw) ≈ residualsd(rw)^2 * inv(J' * (W .* J)) rtol = 1e-8
        @test collect(values(stderror(rw; σ = 1.0))) ≈ sqrt.(diag(inv(J' * (W .* J)))) rtol = 1e-8
        s2 = sum(W .* rr .^ 2) / 32
        @test loglikelihood(rw) ≈ sum(@. -0.5 * log(2π * s2 / W) - W * rr^2 / (2s2)) rtol = 1e-10
        @test loglikelihood(rw) ≈ loglikelihood(rs) rtol = 1e-10
        @test aic(rw) ≈ -2 * loglikelihood(rw) + 6 rtol = 1e-12
        # The weighted fit reaches the bound (JᵀΣ⁻¹J)⁻¹ (here SD 1.12e-3 for T2), the unweighted
        # fit the sandwich, 1.5 times larger; stderror of each copy about the bound
        Rw, Ru = fit(mt, Y; weights = W), fit(mt, Y)
        Jt = jacobian(mt, θ)
        bound = sqrt(inv(Jt' * (W .* Jt))[2, 2])
        @test std(Rw.params.T2) ≈ bound rtol = 0.07
        @test std(Ru.params.T2) / std(Rw.params.T2) > 1.4
        @test median(stderror(Rw).T2) ≈ bound rtol = 0.02
        # Float32, refit keeps the weights, a cache with them allocates nothing per voxel
        @test fit(mt, Float32.(y); weights = W, precision = Float32).params.T2 ≈ rw.params.T2 rtol = 1e-4
        @test refit(rw, y).settings.weights == W
        @test refit(rw, y).params.T2 ≈ rw.params.T2 rtol = 1e-10
        cw = init(mt; weights = W)
        fit!(cw, y)
        @test @allocated(fit!(cw, y)) == 0
        # crlb of the weighted fit: the sandwich H⁻¹J₀ᵀWΣWJ₀H⁻¹ against the closed form, for any W;
        # with W = Σ⁻¹ (unbiased) the bound (JᵀΣ⁻¹J)⁻¹; with a full Σ; weights all 1 unweighted
        @test vcov(crlb(mt, θ; σ = σv, weights = W)) ≈ inv(Jt' * (W .* Jt)) rtol = 1e-9
        w2 = collect(1.0:32.0)
        A = inv(Jt' * (w2 .* Jt))
        @test vcov(crlb(mt, θ; σ = σv, weights = w2)) ≈ A * Jt' * Diagonal(w2 .^ 2 .* σv .^ 2) * Jt * A rtol = 1e-9
        Σ = [100.0 * 0.5^abs(i - j) for i in 1:32, j in 1:32]
        @test vcov(crlb(mt, θ; Σ, weights = w2)) ≈ A * Jt' * Diagonal(w2) * Σ * Diagonal(w2) * Jt * A rtol = 1e-9
        @test vcov(crlb(mt, θ; σ = σv, weights = ones(32))) == vcov(crlb(mt, θ; σ = σv))
        @test crlb(mt, θ; σ = σv, weights = W).weights == W && crlb(mt, θ; σ = σv).weights === nothing
        # A biased fit (5-point SIR, kmf assumed 12.5 for a true 5), weighted: its noise-free fit
        # moves; ∇b by the implicit function theorem against Gauss–Newton steps unrolled (weighted);
        # leakage against central differences of the noise-free fit in kmf
        m5 = QMTSIR(; SIR_5POINT...)
        θs = nominal(m5, :SIR_WM_I)
        wv = [1.0, 2, 4, 8, 16]
        cb = crlb(m5, θs; snr = 200, kmf = 12.5, weights = wv)
        @test abs(cb.bias.PSR - crlb(m5, θs; snr = 200, kmf = 12.5).bias.PSR) > 0.005
        @test biasgradient(m5, θs; kmf = 12.5, weights = wv) ≈
              biasgradient(m5, θs; kmf = 12.5, weights = wv, method = :unrolled, steps = 8) atol = 1e-8
        lk = leakage(m5, θs; kmf = 12.5, weights = wv)
        h = 1e-4
        fd = (collect(values(crlb(m5, θs; snr = 200, kmf = 12.5 + h, weights = wv).fit)) .-
              collect(values(crlb(m5, θs; snr = 200, kmf = 12.5 - h, weights = wv).fit))) ./ 2h
        @test collect(values(lk.absolute.kmf)) ≈ fd rtol = 1e-6
        # The weighted kernel allocates nothing
        c5 = CRLBCache(m5, θs; kmf = 12.5, weights = wv, cov = :biased)
        t5 = EQUIP._crlb_truth(m5, θs)
        k(c, t) = EQUIP._crlb!(c, c.mT, t, 2.5e-5)
        k(c5, t5)
        @test @allocated(k(c5, t5)) == 0
        # montecarlo fits with the scenario's weights; its SD as crlb predicts (within 4 SE)
        mc = montecarlo(mt, θ; σ = σv, weights = W, noise = :gaussian, n = 4000, rng = MersenneTwister(3))
        @test mc.result.settings.weights == W && mc.crlb.weights == W
        @test all(abs.(collect(values(mc.sd)) .- collect(values(mc.crlb.sd))) .< 4 .* collect(values(mc.se_sd)))
        @test occursin("weighted fit: weights 0.000625 to 0.04 (32 observations)", sprint(show, MIME"text/plain"(), mc))
        @test occursin("weighted fit: weights 1.0 to 16.0 (5 observations)", sprint(show, MIME"text/plain"(), cb))
        # Scenarios hold the weights; side by side, weighted against not; sets; the design
        s = Scenario(m5, θs; kmf = 12.5, weights = wv)
        @test s.weights == wv && Scenario(m5, θs).weights === nothing
        @test occursin("weighted fit: weights 1.0 to 16.0", sprint(show, MIME"text/plain"(), s))
        two = crlb(m5, (plain = Scenario(m5, θs; kmf = 12.5), weighted = s); snr = 200)
        @test two[2].sd == cb.sd && two[1].weights === nothing
        set = scenarios(QMTSIR, [:SIR_WM_I, :SIR_lesion]; kmf = 12.5, weights = wv)
        @test all(x -> x.weights == wv, set.results)
        sp = (TI = (0.01, 10.0), TD = (0.01, 10.0))
        st(m) = sum(m.TI .+ m.TD .+ 0.3)
        d = ProtocolDesign(m5, set.results; snr = 200, acquisition = sp, scantime = st)
        @test d(m5) ≈ maximum(sum(abs2, collect(values(crlb(m5, x; snr = 200).relrmse))) for x in set.results) * st(m5) rtol = 1e-10
        mixed = [set[1], Scenario(QMTSIR, nominal(QMTSIR, :SIR_lesion); kmf = 12.5)]
        @test length(ProtocolDesign(m5, mixed; snr = 200, acquisition = sp, scantime = st).groups) == 2
        x5 = [m5.TI; m5.TD]
        d(x5); d(x5)
        @test @allocated(d(x5)) == 0
        @test copy(d)(x5) == d(x5)                 # the copy each search task gets
        # Mistakes
        @test occursin("`weights` has 3 values", msg(() -> fit(mt, y; weights = ones(3))))
        @test occursin("finite and ≥ 0", msg(() -> fit(mt, y; weights = -ones(32))))
        @test occursin("finite and ≥ 0", msg(() -> fit(mt, y; weights = fill(NaN, 32))))
        @test occursin("at least one must be positive", msg(() -> fit(mt, y; weights = zeros(32))))
        @test occursin("one weight per observation", msg(() -> fit(mt, y; weights = 2.0)))
        @test occursin("finite and ≥ 0", msg(() -> Scenario(m5, θs; weights = [-1.0, 1, 1, 1, 1])))
        @test occursin("32 weights; the protocol has 5 observations", msg(() -> crlb(m5, θs; snr = 200, weights = W)))
        @test occursin("rename the parameter", msg(() -> fit(WeightsNamed([0.1, 0.2, 0.3]), [1.0, 0.8, 0.6])))
    end

    @testset "crlb's pieces: jacobian, hessian, informationmatrix, biasgradient, sandwich, biasedcrlb" begin
        FD = EQUIP.ForwardDiff
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        m4 = QMTSIR(; SIR_4POINT...)
        θ = nominal(m4, :SIR_WM_I)                 # kmf = 5; the fit fixes it at 12.5 (wrong)
        F4 = (:PSR, :R1f, :Sf, :M0f)
        g(vals, x) = signal(m4, merge(vals, NamedTuple{F4}(Tuple(x))))
        x0 = [θ[n] for n in F4]
        # jacobian: ForwardDiff on signal, the fit's free parameters as columns, at θ's values
        J = jacobian(m4, θ)
        @test J == FD.jacobian(x -> g(θ, x), x0)
        J5 = jacobian(m4, θ; kmf = Free())
        @test size(J5) == (4, 5) && J5[:, 1:4] == J
        # a tie is followed: with R1m = :R1f, the R1f column is R1f's and R1m's together
        Jr = jacobian(m4, merge(θ, (R1m = θ.R1f,)); R1m = Free(start = 1.0))   # ..., M0f, R1m
        @test J[:, 2] ≈ Jr[:, 2] + Jr[:, 5] rtol = 1e-12
        # the noise-free fit, its bias, and the derivatives at its answer (the fit's kmf = 12.5)
        r = noisefree(m4, θ; kmf = 12.5)
        c = crlb(m4, θ; snr = 200, kmf = 12.5)
        @test r.params == c.fit && bias(m4, θ; kmf = 12.5) == c.bias == bias(r, θ)
        @test noisefree(m4, θ).params == NamedTuple{F4}(map(n -> θ[n], F4))     # nothing wrong
        θ̂ = merge(r.fixed, r.params)
        @test θ̂.kmf == 12.5 && θ̂.R1m === :R1f
        J₀ = jacobian(r)
        @test J₀ == jacobian(m4, θ̂)
        # hessian: JᵀJ when the data are fitted exactly; with a residual, + Σᵢ rᵢ∇²gᵢ, and with a
        # noise covariance, JᵀΣ⁻¹J + Σᵢ (Σ⁻¹r)ᵢ∇²gᵢ
        y = signal(m4, θ)
        @test hessian(m4, θ, y) ≈ J' * J rtol = 1e-12
        x̂ = collect(values(r.params))
        res = g(θ̂, x̂) - y
        ∇²g = [FD.hessian(x -> g(θ̂, x)[i], x̂) for i in 1:4]
        @test hessian(r, y) ≈ J₀' * J₀ + sum(res .* ∇²g) rtol = 1e-10
        Σ = [1e-4 * 0.5^abs(i - j) for i in 1:4, j in 1:4]
        @test hessian(r, y; Σ) ≈ J₀' * (Σ \ J₀) + sum((Σ \ res) .* ∇²g) rtol = 1e-10
        # noisecovariance: σ = M0/snr, a Diagonal; σ per observation; Σ as given
        @test noisecovariance(m4, θ; snr = 200) == Diagonal(fill((θ.M0f / 200)^2, 4))
        @test noisecovariance(m4, θ; σ = [1.0, 2.0, 3.0, 4.0]) == Diagonal([1.0, 4.0, 9.0, 16.0])
        @test noisecovariance(m4, θ; Σ) === Σ
        @test noisecovariance(m4, Scenario(QMTSIR, θ; kmf = 12.5); snr = 200) == noisecovariance(m4, θ; snr = 200)
        @test occursin("the scenario is for QMTSIR", msg(() -> noisecovariance(T2Decay(TE = TD5), Scenario(QMTSIR, θ); σ = 1.0)))
        # crlb is their composition: the sandwich at θ̂ ...
        Σn = noisecovariance(m4, θ; snr = 200)
        # (crlb leaves out the curvature of the exact 4-point fit, whose residual is ~1e-13)
        @test vcov(c) ≈ sandwich(hessian(r, y), J₀, Σn) rtol = 1e-10
        # ... and Eq. 3, with F the Fisher information of the true signal at the truth
        Ft = informationmatrix(m4, θ; snr = 200)
        @test Ft ≈ J' * J / (θ.M0f / 200)^2 rtol = 1e-12
        ∇b = biasgradient(m4, θ; kmf = 12.5)
        @test vcov(crlb(m4, θ; snr = 200, kmf = 12.5, cov = :biased)) ≈ biasedcrlb(∇b, Ft) rtol = 1e-12
        @test biasedcrlb(zeros(4, 4), Ft) ≈ inv(Ft) rtol = 1e-10
        # ∇b: the implicit function theorem, ForwardDiff through Gauss–Newton steps and central
        # differences of noise-free fits agree; zero when nothing fixed is wrong
        @test biasgradient(m4, θ; kmf = 12.5, method = :unrolled) ≈ ∇b atol = 1e-8
        tight = LM(maxiter = 1000, xtol = 1e-13, ftol = 0.0, gtol = 0.0)
        function tightbias(θx)                     # a tightly converged noise-free fit's bias
            r = fit(m4, signal(m4, θx); solver = tight, NamedTuple{F4}(map(n -> Free(start = θx[n]), F4))...)
            return [r.params[n] - θx[n] for n in F4]
        end
        for (j, n) in enumerate(F4)
            h = 1e-5 * abs(θ[n])
            bp = tightbias(merge(θ, NamedTuple{(n,)}((θ[n] + h,))))
            bm = tightbias(merge(θ, NamedTuple{(n,)}((θ[n] - h,))))
            @test (bp - bm) / 2h ≈ ∇b[:, j] atol = 1e-6
        end
        @test maximum(abs, biasgradient(m4, θ; kmf = 5.0)) < 1e-8 && maximum(abs, biasgradient(m4, θ)) < 1e-8
        # a parameter held on a bound: its row of I + ∇b is 0; undetermined parameters: NaN
        bounds4 = (PSR = Free(lower = 0, upper = 1), R1f = Free(lower = 0, upper = 3),
                   Sf = Free(lower = -1.0, upper = 0), M0f = Free(lower = 0, upper = 10))
        ∇h = biasgradient(m4, θ; kmf = 12.5, bounds4...)
        @test ∇h[3, :] == [0.0, 0.0, -1.0, 0.0] && all(isfinite, ∇h)
        @test all(isnan, biasgradient(m4, θ; kmf = Free()))
        # the FitResult forms are for one voxel; mistakes
        Y = simulate(m4, θ; noise = :rician, snr = 200, n = 3, rng = MersenneTwister(1))
        @test occursin("one-voxel", msg(() -> jacobian(fit(m4, Y))))
        @test occursin(":implicit or :unrolled", msg(() -> biasgradient(m4, θ; method = :fd)))
        @test occursin("`y` has 3 values", msg(() -> hessian(m4, θ, [1.0, 2.0, 3.0])))
        # the allocation-free kernel, with the noise from noisecovariance: crlb's numbers
        mt = T2Decay(TE = collect(0.01:0.01:0.32))
        θt = (S0 = 1000.0, T2 = 0.08)
        Σt = [100.0 * 0.5^abs(i - j) for i in 1:32, j in 1:32]
        for cov in (:sandwich, :biased)
            k = EQUIP._crlb!(CRLBCache(mt, θt; cov), mt, EQUIP._crlb_truth(mt, θt),
                             EQUIP._kernelnoise(noisecovariance(mt, θt; Σ = Σt), 2))
            @test k.cov ≈ vcov(crlb(mt, θt; Σ = Σt, cov)) rtol = 1e-10
        end
        @test EQUIP._kernelnoise(noisecovariance(m4, θ; snr = 200), 4) == (θ.M0f / 200)^2
        @test EQUIP._kernelnoise(noisecovariance(m4, θ; σ = [1.0, 2.0, 3.0, 4.0]), 4) == [1.0, 4.0, 9.0, 16.0]
    end

    @testset "crlb against the textbook formulas; CRLBCache and crlb!" begin
        FD = EQUIP.ForwardDiff
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        # The reference: ForwardDiff on the allocating `signal`, plain matrices and `\`, the
        # noise-free fit converged tightly (crlb's stops at xtol 1e-10). crlb sums the same
        # matrices row by row from dual numbers, without storing J.
        tight = LM(maxiter = 1000, xtol = 1e-13, ftol = 0.0, gtol = 0.0)
        function reference(m, θ, fitset, Σ, cov)
            F = EQUIP.freenames(EQUIP.setup(m, Float64; fitset...))
            θt = resolve(m, θ)
            y = signal(m, θ)
            start(n) = (x = get(fitset, n, nothing);
                        x isa Free ? Free(start = θt[n], lower = x.lower, upper = x.upper) : Free(start = θt[n]))
            r = fit(m, y; solver = tight, fitset..., NamedTuple{F}(map(start, F))...)
            θ̂ = merge(r.fixed, r.params)               # the fit's fixed values and ties
            g(vals, x) = signal(m, merge(vals, NamedTuple{F}(Tuple(x))))
            x̂, xt = [r.params[n] for n in F], [θt[n] for n in F]
            J₀ = FD.jacobian(x -> g(θ̂, x), x̂)
            H = FD.hessian(x -> sum(abs2, g(θ̂, x) - y) / 2, x̂)      # JᵀJ + Σᵢ rᵢ∇²gᵢ
            if cov === :sandwich
                A = H \ J₀'
                C = A * Σ * A'                                         # H⁻¹J₀ᵀΣJ₀H⁻¹
            else
                Jt = FD.jacobian(x -> g(θ, x), xt)
                D = H \ (J₀' * Jt)                                     # I + ∇b
                C = D * ((Jt' * (Σ \ Jt)) \ D')                        # (I + ∇b)F⁻¹(I + ∇b)ᵀ
            end
            return x̂ .- xt, sqrt.(diag(C))
        end
        te = collect(0.01:0.01:0.32)
        cases = Any[]
        for m in (QMTSIR(; SIR_4POINT...), QMTSIR(; SIR_5POINT...), QMTSIR(TI = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0], TD = 2.5)),
            t in (:SIR_WM_I, :SIR_GM), f in ((;), (Sm = 0.9,), (kmf = Free(),))
            nobs(m) == 4 && haskey(f, :kmf) && continue                  # 5 parameters, 4 points
            θ = nominal(m, t)
            push!(cases, (m, θ, f, noisecovariance(m, θ; snr = 200)))
        end
        Σt = [100.0 * 0.5^abs(i - j) for i in 1:32, j in 1:32]
        θb = (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08)
        push!(cases, (T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08), (;), Σt),
                     (T2BiexpDecay(TE = te), θb, (;), Σt),
                     (T2BiexpDecay(TE = te), θb, (;), Diagonal(collect(range(5.0, 20.0, length = 32)) .^ 2)),
                     (T1IR(TI = [0.05, 0.1, 0.3, 0.6, 1.2, 2.4]), (M0 = 1000.0, Sf = -0.95, T1 = 1.0), (;), Diagonal(fill(100.0, 6))))
        for (m, θ, f0, Σ) in cases, cov in (:sandwich, :biased)
            f = merge(defaults(m), f0)            # the model's fixed values: kmf = 12.5 (wrong)
            b, sd = reference(m, θ, f, Matrix(Σ), cov)
            c = crlb(m, θ; Σ = Matrix(Σ), f..., cov)
            @test isempty(c.onbound)
            @test collect(values(c.bias)) ≈ b atol = 1e-8 * maximum(abs, values(c.fit))
            @test collect(values(c.sd)) ≈ sd rtol = 1e-6
        end
        # the cache: made once, reused for several truths and noise levels; crlb is cache + crlb!
        # (the tissues differ only in kmf, which this fit estimates)
        m5 = QMTSIR(; SIR_5POINT...)
        c = CRLBCache(m5, nominal(m5, :SIR_WM_I); kmf = Free())
        @test sprint(show, c) == "CRLBCache(QMTSIR: PSR, R1f, Sf, M0f, kmf; sandwich)"
        for t in (:SIR_WM_I, :SIR_GM), snr in (100, 200)
            θ = nominal(m5, t)
            @test crlb!(c, θ; snr).sd == crlb(m5, θ; snr, kmf = Free()).sd
        end
        # the second derivatives are left out when the noise-free fit is exact: nothing fixed
        # wrong, or as many observations as free parameters (4-point); not otherwise
        exact(m, θ; kw...) = EQUIP._crlb!(CRLBCache(m, θ; kw...), EQUIP._with_precision(m, Float64), EQUIP._crlb_truth(m, θ), 1.0).exact
        m4 = QMTSIR(; SIR_4POINT...)
        @test exact(m4, nominal(m4, :SIR_WM_I)) && exact(m5, nominal(m5, :SIR_WM_I); kmf = Free())
        @test exact(m5, nominal(m5, :SIR_WM_I)) && !exact(m5, nominal(m5, :SIR_WM_I); kmf = 12.5)
        # mistakes
        @test occursin("`cov` must be :sandwich or :biased", msg(() -> CRLBCache(m5, nominal(m5, :SIR_WM_I); cov = :eq3)))
        @test occursin("the sandwich only", msg(() -> crlb!(CRLBCache(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); cov = :biased),
            (S0 = 1000.0, T2 = 0.08); snr = 100, simulate_with = T2BiexpDecay(TE = te) => θb)))
    end

    @testset "Scenario: a truth and the fit judged against it" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        m4, m5 = QMTSIR(; SIR_4POINT...), QMTSIR(; SIR_5POINT...)
        θ = nominal(QMTSIR, :SIR_WM_I)                                # kmf = 5
        own = parameters(QMTSIR)
        # from the truth: estimated as the model estimates them (its bounds), starting at the
        # true values; fixed at the true values, so nothing is wrong
        u = Scenario(QMTSIR, θ)
        @test u.model === QMTSIR && u.truth == θ && u.starts === :truth && isempty(u.wrong)
        @test u.roles.PSR == Free(start = 0.2, lower = own.PSR.lower, upper = own.PSR.upper)
        @test u.roles.M0f == Free(start = 1.0, lower = own.M0f.lower)          # not :scale
        @test u.roles.kmf == Fixed(5.0) && u.roles.R1m == Fixed(:R1f) && u.roles.Sm == Fixed(0.83)
        @test u.attruth == (:PSR, :R1f, :Sf, :M0f)
        @test Scenario(m4, θ).roles == u.roles && Scenario(m4, θ).model === QMTSIR   # a model: its type
        # the changes, by fit's rules: a value fixes, and is wrong when it differs from the truth
        s = Scenario(QMTSIR, θ; kmf = 12.5)
        @test s.roles.kmf == Fixed(12.5) && s.wrong == (kmf = (fit = 12.5, truth = 5.0),)
        @test isempty(Scenario(QMTSIR, θ; kmf = 5.0).wrong)
        # a tie is judged at the truth: R1m = :R1f is 1.5 here, wrong for a tissue with R1m = 1.0
        θ1 = merge(θ, (R1m = 1.0,))
        @test Scenario(QMTSIR, θ1).roles.R1m == Fixed(1.0) && isempty(Scenario(QMTSIR, θ1).wrong)
        @test Scenario(QMTSIR, θ1; R1m = :R1f).wrong == (R1m = (fit = 1.5, truth = 1.0),)
        @test isempty(Scenario(QMTSIR, merge(θ, (R1m = 1.5,)); R1m = :R1f).wrong)
        # freed: from its true value, or (starts = :model) where `fit` would start it
        @test Scenario(QMTSIR, θ; kmf = Free()).roles.kmf == Free(start = 5.0)
        @test Scenario(QMTSIR, θ; kmf = Free(), starts = :model).roles.kmf == Free(start = 12.5)
        @test Scenario(QMTSIR, θ; R1m = Free()).roles.R1m == Free(start = 1.5)
        @test occursin("give it a start", msg(() -> Scenario(QMTSIR, θ; R1m = Free(), starts = :model)))
        # starts = :model: the model's rules; a start or a bound given wins, the rest is kept
        sm = Scenario(QMTSIR, θ; starts = :model, R1f = Free(start = 0.8), PSR = Free(lower = 0.05))
        @test sm.roles.M0f == own.M0f && sm.roles.Sf == own.Sf && sm.attruth == ()
        @test sm.roles.R1f == Free(start = 0.8, lower = own.R1f.lower, upper = own.R1f.upper)
        @test sm.roles.PSR == Free(start = own.PSR.start, lower = 0.05, upper = own.PSR.upper)
        st = Scenario(QMTSIR, θ; R1f = Free(start = 0.8))
        @test st.roles.R1f.start == 0.8 && st.attruth == (:PSR, :Sf, :M0f)
        # crlb(m, s) is the keyword form; its noise-free fit starts at the truth either way; the
        # default is unbiased; one scenario, judged under several protocols
        c = crlb(m4, s; snr = 200)
        k = crlb(m4, θ; snr = 200, kmf = 12.5)
        @test c.sd == k.sd && c.bias == k.bias && c.fit == k.fit
        @test crlb(m4, Scenario(QMTSIR, θ; kmf = 12.5, starts = :model); snr = 200).fit == c.fit
        @test crlb!(CRLBCache(m4, s), θ; snr = 200).sd == c.sd
        @test crlb(m5, s; snr = 200).relbias.PSR < -0.01 && crlb(m5, u; snr = 200).bias.PSR == 0
        # the pieces take a scenario too
        @test noisefree(m4, s).params == c.fit && bias(m4, s) == c.bias
        @test jacobian(m4, s) == jacobian(m4, θ) && biasgradient(m4, s) == biasgradient(m4, θ; kmf = 12.5)
        @test informationmatrix(m4, s; snr = 200) == informationmatrix(m4, θ; snr = 200)
        @test hessian(m4, s, signal(m4, θ)) == hessian(m4, θ, signal(m4, θ))
        # montecarlo: the fits start where the scenario says; the keyword form is the same
        mc = montecarlo(m4, s; noise = :rician, snr = 200, n = 200, rng = MersenneTwister(1))
        @test mc.sd == montecarlo(m4, θ; noise = :rician, snr = 200, n = 200, kmf = 12.5, rng = MersenneTwister(1)).sd
        @test mc.result.settings.PSR == s.roles.PSR && mc.crlb.sd == c.sd
        mm = montecarlo(m4, θ; noise = :rician, snr = 200, n = 200, kmf = 12.5, starts = :model, rng = MersenneTwister(1))
        @test mm.result.settings.M0f.start === :scale && mm.result.settings.PSR.start == own.PSR.start
        # in a sweep over true values the starts from the truth follow it; the fixed values stay
        v = montecarlo(m4, s; noise = :rician, snr = 200, n = 50, vary = :PSR => [0.1, 0.15], rng = MersenneTwister(2))
        @test [r.result.settings.PSR.start for r in v] == [0.1, 0.15]
        @test [r.result.settings.kmf for r in v] == [Fixed(12.5), Fixed(12.5)]
        # a start that depends on the protocol: kept as a function, computed for each protocol
        ti = [0.05, 0.1, 0.3, 0.6, 1.2, 2.4]
        sir = Scenario(T1IR, (M0 = 1000.0, Sf = -0.95, T1 = 1.0); starts = :model)
        @test sir.roles.T1.start isa Function
        r1 = montecarlo(T1IR(TI = ti), sir; noise = :rician, σ = 10.0, n = 50, rng = MersenneTwister(3))
        @test r1.result.settings.T1.start.values == parameters(T1IR(TI = ti)).T1.start.values
        # a model with its own `parameters(m)` method: from a model, not its type
        ds = Scenario(DataStartTest([0.01, 0.02, 0.04]), (S0 = 1.0, T2 = 0.05, c = 1.0))
        @test ds.model === DataStartTest && ds.roles.c == Fixed(1.0) && ds.roles.T2.start == 0.05
        @test crlb(DataStartTest([0.01, 0.02, 0.04]), ds; σ = 0.01).bias == (S0 = 0.0, T2 = 0.0)
        @test occursin("give a model", msg(() -> Scenario(DataStartTest, (S0 = 1.0, T2 = 0.05, c = 1.0))))
        # display: the truth, the fit, and what is wrong
        str = sprint(show, MIME"text/plain"(), s)
        @test startswith(str, "Scenario for QMTSIR: a truth and the fit judged against it\nstarts: the truth\n")
        @test occursin(r"\n  name +true +role +start/value +bounds\n", str)
        @test occursin(r"\n  PSR +0.2 +free +0.2 +\[0.0, 1.0\]\n", str)
        @test occursin(r"\n  kmf +5.0 +fixed +12.5 +← wrong\n", str)
        @test occursin(r"\n  R1m +:R1f +fixed +:R1f \(tie\)\n", str)
        @test endswith(str, "\nwrong: kmf = 12.5 (true 5.0): the fit is biased")
        @test endswith(sprint(show, MIME"text/plain"(), u), "\nnothing fixed is wrong: the fit is unbiased")
        @test occursin("the model's rules", sprint(show, MIME"text/plain"(), sm))
        @test occursin(r"\n  T1 +1.0 +free +from the protocol ", sprint(show, MIME"text/plain"(), sir))
        @test repr(s) == "Scenario(QMTSIR; wrong: kmf)" && repr(u) == "Scenario(QMTSIR; unbiased)"
        # the results mark a wrong tie too
        ct = crlb(m4, θ1; snr = 200, R1m = :R1f)
        @test occursin("fixed in the fit: kmf = 5.0, R1m = :R1f = 1.5 (true 1.0), Sm = 0.83", sprint(show, MIME"text/plain"(), ct))
        # mistakes
        @test occursin("`starts` must be :truth", msg(() -> Scenario(QMTSIR, θ; starts = :data)))
        @test occursin("`kfm` is not a parameter of QMTSIR", msg(() -> Scenario(QMTSIR, θ; kfm = 1.0)))
        @test occursin("option of the fit", msg(() -> Scenario(QMTSIR, θ; solver = LM())))
        @test occursin("missing: Sf", msg(() -> Scenario(QMTSIR, Base.structdiff(θ, NamedTuple{(:Sf,)}))))
        @test occursin("is a role", msg(() -> Scenario(QMTSIR, parameters(QMTSIR))))
        @test occursin("one voxel", msg(() -> Scenario(QMTSIR, θ; Sm = fill(0.8, 2))))
        @test occursin("no free parameters", msg(() -> Scenario(T2Decay, (S0 = 1.0, T2 = 0.1); S0 = 1.0, T2 = 0.1)))
        @test occursin(":scale is for", msg(() -> Scenario(QMTSIR, θ; PSR = Free(start = :scale))))
        @test occursin("lower bound 2.0 exceeds", msg(() -> Scenario(QMTSIR, θ; PSR = Free(lower = 2.0))))
        @test occursin("which is itself a tie", msg(() -> Scenario(QMTSIR, θ; R1f = :R1m)))
        @test occursin("the scenario is for QMTSIR", msg(() -> crlb(T2Decay(TE = TD5), s; σ = 1.0)))
        @test occursin("starts at the truth", msg(() -> crlb(m4, θ; snr = 200, starts = :model)))
        @test occursin("sets its own options", msg(() -> jacobian(m4, θ; precision = Float32)))
    end

    @testset "leakage: how an error in a fixed parameter moves the fitted ones" begin
        FD = EQUIP.ForwardDiff
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        m4, m5 = QMTSIR(; SIR_4POINT...), QMTSIR(; SIR_5POINT...)
        ti16 = [round.(exp10.(range(log10(0.01), log10(2), 15)) .* 1e3) ./ 1e3; 10]
        m16 = QMTSIR(TI = ti16, TD = 2.5)
        θ = nominal(m5, :SIR_WM_I)
        # PSR's leakage from kmf, % of the true PSR per s⁻¹ (docs/design.md, "Diagnostics"): at
        # the truth, WM I and WM II, the three schemes; and where the fit assumes kmf = 12.5
        for (m, wm1, wm2) in ((m4, -7.85, -0.62), (m5, -11.90, -2.30), (m16, -15.42, -2.98))
            @test 100 * leakage(m, θ).relative.kmf.PSR ≈ wm1 atol = 0.005
            @test 100 * leakage(m, nominal(m, :SIR_WM_II)).relative.kmf.PSR ≈ wm2 atol = 0.005
        end
        @test 100 * leakage(m5, θ; kmf = 12.5).relative.kmf.PSR ≈ -3.25 atol = 0.005
        @test 100 * leakage(m16, θ; kmf = 12.5).relative.kmf.PSR ≈ -2.09 atol = 0.005
        # At the truth: the linear leakage −(JᵀJ)⁻¹Jᵀ∂g/∂c
        l = leakage(m5, θ)
        F, C = keys(l.fit), keys(l.absolute)
        @test C == (:kmf, :Sm, :Mzf_td0, :Mzm_td0)                  # the tie R1m = :R1f is not one
        Jall = FD.jacobian(x -> signal(m5, merge(θ, NamedTuple{(F..., C...)}(Tuple(x)))),
                           [[θ[p] for p in F]; [θ[c] for c in C]])
        J, Jc = Jall[:, 1:4], Jall[:, 5:end]
        @test [l.absolute[c][p] for p in F, c in C] ≈ -(J' * J) \ (J' * Jc) rtol = 1e-12
        # Away from the truth (kmf assumed 12.5): central differences of noise-free fits, each
        # converged to rounding (Newton on the cost's exact Hessian), within 1e-6
        lb = leakage(m5, θ; kmf = 12.5)
        y = signal(m5, θ)
        function nf(assumed)
            cost(x) = sum(abs2, signal(m5, merge(θ, assumed, NamedTuple{F}(Tuple(x)))) .- y) / 2
            x = collect(values(lb.fit))
            for _ in 1:20
                x -= FD.hessian(cost, x) \ FD.gradient(cost, x)
            end
            return x
        end
        for c in C
            x = c === :kmf ? 12.5 : θ[c]
            h = max(abs(x), 1.0) * 1e-4
            fd = (nf(merge((kmf = 12.5,), NamedTuple{(c,)}((x + h,)))) -
                  nf(merge((kmf = 12.5,), NamedTuple{(c,)}((x - h,))))) / 2h
            @test [lb.absolute[c][p] for p in F] ≈ fd rtol = 1e-6
        end
        # Times an error, the first-order bias: crlb's noise-free bias within the second-order
        # term (a quarter when the error is halved)
        e(Δ) = crlb(m5, θ; snr = 200, kmf = 5.0 + Δ).bias.PSR - l.absolute.kmf.PSR * Δ
        @test abs(e(0.25)) < 0.05 * abs(l.absolute.kmf.PSR * 0.25)
        @test 3.5 < e(0.25) / e(0.125) < 4.5
        # relative: per |true value|; elasticity: per % of the assumed value
        @test lb.relative.kmf.PSR == lb.absolute.kmf.PSR / θ.PSR
        @test lb.elasticity.kmf.PSR == lb.absolute.kmf.PSR * 12.5 / θ.PSR
        @test lb.relative.Sm.Sf == lb.absolute.Sm.Sf / abs(θ.Sf) && lb.elasticity.Mzf_td0.PSR == 0
        # A parameter held on a bound: slope 0, the others as with it fixed there
        lo = leakage(m4, θ; kmf = 12.5, Sf = Free(lower = -1.0, upper = 0))
        lf = leakage(m4, θ; kmf = 12.5, Sf = -1.0)
        @test lo.onbound == [:Sf] && all(c -> lo.absolute[c].Sf == 0, keys(lo.absolute))
        @test all(lo.absolute[c][p] ≈ lf.absolute[c][p] for c in keys(lo.absolute), p in (:PSR, :R1f, :M0f))
        # Undetermined (5 free parameters, 4 observations): NaN
        @test all(isnan, values(leakage(m4, θ; kmf = Free()).absolute.Sm))
        # One free and one fixed parameter: T2 with S0 assumed
        te = collect(0.01:0.01:0.32)
        lt = leakage(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); S0 = 1000.0)
        Jt = FD.jacobian(x -> signal(T2Decay(TE = te), (S0 = 1000.0, T2 = x[1])), [0.08])
        Js = FD.jacobian(x -> signal(T2Decay(TE = te), (S0 = x[1], T2 = 0.08)), [1000.0])
        @test lt.absolute.S0.T2 ≈ (-(Jt' * Jt) \ (Jt' * Js))[1] rtol = 1e-12
        # An untied R1m is a fixed number like the others; the same from a scenario
        @test keys(leakage(m5, θ; R1m = 1.0).absolute) == (:kmf, :R1m, :Sm, :Mzf_td0, :Mzm_td0)
        @test leakage(m5, Scenario(m5, θ; kmf = 12.5)).absolute == lb.absolute
        # The kernel allocates nothing, exact fit or not
        for s in (Scenario(m5, θ), Scenario(m5, θ; kmf = 12.5))
            lc = EQUIP._LeakageCache(m5, s)
            t = EQUIP._crlb_truth(m5, s.truth)
            k(lc, t) = EQUIP._leakage!(lc, lc.crlb.mT, t)
            k(lc, t)
            @test @allocated(k(lc, t)) == 0
        end
        out = sprint(show, MIME"text/plain"(), lb)
        @test occursin("kmf = 12.5 (true 5.0)", out) && occursin("relative:", out) && occursin("elasticity:", out)
        @test occursin("fixes no parameter at a number", msg(() -> leakage(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08))))
        @test occursin("the scenario is for QMTSIR", msg(() -> leakage(T2Decay(TE = te), Scenario(m5, θ))))
    end

    @testset "scenarios: tissues, boxes of true values, and both (calcVmax's modes)" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        tissues = [:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion]
        m4 = QMTSIR(; SIR_4POINT..., magnitude = false)
        kw = (snr = 200, acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)), scantime = m -> sum(m.TI .+ m.TD .+ 0.3))
        # 'multiple': the tissues, as the hand-built list (the same design value, to the bit)
        s4 = scenarios(QMTSIR, tissues; kmf = 12.5)
        @test s4 isa Sweep && length(s4) == 4 && s4.settings == [(tissue = t,) for t in tissues]
        hand = [Scenario(QMTSIR, nominal(QMTSIR, t); kmf = 12.5) for t in tissues]
        @test all(i -> s4[i].truth == hand[i].truth && s4[i].wrong == hand[i].wrong, 1:4)
        v4 = ProtocolDesign(m4, s4; kw...)(m4)
        @test v4 == ProtocolDesign(m4, hand; kw...)(m4)
        @test v4 ≈ 0.243395 rtol = 2e-6                                  # Table 2's 4-point
        # 'ranges': every combination of two values each (the first slowest) and, asked for, the
        # midpoint first: 2⁵ + 1 = 33 (Li 2010's sets)
        box = scenarios(QMTSIR, nominal(QMTSIR); kmf = 12.5, midpoint = true,
                        vary = (PSR = [0.05, 0.2], kmf = [5.0, 20.0], R1f = [0.5, 1.5], Sf = [-1.0, -0.8], M0f = [0.8, 1.2]))
        @test length(box) == 33 && !haskey(box.settings[1], :tissue)
        @test box.settings[1] == (PSR = 0.125, kmf = 12.5, R1f = 1.0, Sf = -0.9, M0f = 1.0)
        @test box.settings[2] == (PSR = 0.05, kmf = 5.0, R1f = 0.5, Sf = -1.0, M0f = 0.8)
        @test box.settings[3] == (PSR = 0.05, kmf = 5.0, R1f = 0.5, Sf = -1.0, M0f = 1.2)
        @test box.settings[end] == (PSR = 0.2, kmf = 20.0, R1f = 1.5, Sf = -0.8, M0f = 1.2)
        @test box[2].truth == merge(nominal(QMTSIR), box.settings[2]) && box[2].wrong.kmf.truth == 5.0
        @test box.rows[2].PSR == 0.05 && box.rows[2].Sm == 0.83                       # every true value
        # both: each tissue × each condition, the fit's assumption kept (a robust design's set)
        rob = scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.83, 0.88], kmf = 12.5, Sm = 0.83)
        @test length(rob) == 12 && rob.settings[1:3] == [(tissue = :SIR_WM_I, Sm = s) for s in (0.78, 0.83, 0.88)]
        @test rob[1].wrong.Sm.truth == 0.78 && !haskey(rob[2].wrong, :Sm)     # 0.83: not wrong
        hr = [Scenario(QMTSIR, merge(nominal(QMTSIR, t), (Sm = s,)); kmf = 12.5, Sm = 0.83) for t in tissues for s in (0.78, 0.83, 0.88)]
        d = ProtocolDesign(m4, rob; kw...)
        @test d(m4) == ProtocolDesign(m4, hr; kw...)(m4) && length(d.groups) == 1
        @test rob.settings[argmax(d.values)] == (tissue = :SIR_WM_I, Sm = 0.78)
        # the midpoint only when asked for (vary's combinations as for crlb), rounded (not
        # 0.8300000000000001) and not repeated; a fixed value not given follows each truth
        @test length(scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.88])) == 8
        mid = scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.88], midpoint = true, kmf = 12.5, Sm = 0.83)
        @test length(mid) == 12 && mid.settings[1] == (tissue = :SIR_WM_I, Sm = 0.83) && !haskey(mid[1].wrong, :Sm)
        @test length(scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.83, 0.88], midpoint = true)) == 12
        own = scenarios(QMTSIR, [:SIR_WM_I]; vary = :Sm => [0.78, 0.83, 0.88])
        @test all(x -> isempty(x.wrong), own) && length(ProtocolDesign(m4, own; kw...).groups) == 3
        # truths: one name, a named tuple, named truths, a vector of named tuples (by place)
        @test scenarios(QMTSIR, :SIR_GM).settings == [(tissue = :SIR_GM,)]
        @test scenarios(QMTSIR, nominal(QMTSIR, :SIR_GM)).settings == [(;)]
        @test [st.tissue for st in scenarios(QMTSIR, (wm = nominal(QMTSIR, :SIR_WM_I), gm = (PSR = 0.1, R1f = 1.1, Sf = -0.9, M0f = 1.0))).settings] == [:wm, :gm]
        @test [st.tissue for st in scenarios(QMTSIR, [nominal(QMTSIR, :SIR_WM_I), nominal(QMTSIR, :SIR_GM)]).settings] == [1, 2]
        out = sprint(show, MIME"text/plain"(), rob)
        @test occursin("12 scenarios for QMTSIR: tissue × Sm", out) && occursin("the fit: PSR, R1f, Sf, M0f free", out)
        @test occursin("SIR_WM_I    0.78  kmf (true 5.0), Sm (true 0.78)", out)
        @test occursin("the fits differ", sprint(show, MIME"text/plain"(), own))
        @test occursin("a tie has no midpoint", msg(() -> scenarios(QMTSIR, tissues; vary = :R1m => [:R1f, 1.0], midpoint = true)))
        @test length(scenarios(QMTSIR, tissues; vary = :R1m => [:R1f, 1.0])) == 8
        @test occursin("names the truths", msg(() -> scenarios(QMTSIR, tissues; vary = :tissue => [1, 2])))
        @test occursin("not a parameter of QMTSIR", msg(() -> scenarios(QMTSIR, tissues; vary = :T1 => [1.0, 2.0])))
        @test occursin("a tissue name", msg(() -> scenarios(QMTSIR, [1.0, 2.0])))
        @test occursin("at least one truth", msg(() -> scenarios(QMTSIR, Symbol[])))
        # the scale parameter need not be varied: with snr relative to it, Eq. 5's terms do not
        # depend on it
        sm0 = scenarios(QMTSIR, nominal(QMTSIR, :SIR_WM_I); vary = :M0f => [0.8, 1.0, 1.2], kmf = 12.5)
        dm0 = ProtocolDesign(m4, sm0; kw...)
        dm0(m4)
        @test maximum(dm0.values) - minimum(dm0.values) < 1e-12 * maximum(dm0.values)
        # crlb and montecarlo take a set: one result per scenario, the set's settings (after the
        # noise levels), each as the single-scenario call
        mm = QMTSIR(; SIR_4POINT...)
        two = scenarios(QMTSIR, [:SIR_WM_I, :SIR_lesion]; vary = :Sm => [0.78, 0.83, 0.88], kmf = 12.5, Sm = 0.83)
        c = crlb(mm, two; snr = 200)
        @test c isa Sweep && length(c) == 6 && c.settings == two.settings
        @test all(i -> c[i].sd == crlb(mm, two[i]; snr = 200).sd && c[i].bias == crlb(mm, two[i]; snr = 200).bias, 1:6)
        @test length(c.rows) == 6 * 4 && c.rows[5].tissue === :SIR_WM_I && c.rows[5].Sm == 0.83
        c2 = crlb(mm, two; snr = [100, 200], cov = :biased)
        @test length(c2) == 12 && c2.settings[1] == (snr = 100, tissue = :SIR_WM_I, Sm = 0.78)
        @test c2[8].sd == crlb(mm, two[2]; snr = 200, cov = :biased).sd
        mc = montecarlo(mm, two; noise = :rician, snr = 200, n = 200, rng = MersenneTwister(1))
        @test length(mc) == 6 && mc[3].sd == montecarlo(mm, two[3]; noise = :rician, snr = 200, n = 200, rng = MersenneTwister(1)).sd
        @test occursin("SIR_lesion", sprint(show, MIME"text/plain"(), c)) && !occursin(":SIR_lesion", sprint(show, MIME"text/plain"(), c))
        @test occursin("vary them in `scenarios`", msg(() -> crlb(mm, two; snr = 200, vary = :kmf => [5.0, 10.0])))
        @test occursin("crlb takes no noise", msg(() -> crlb(mm, two; snr = 200, noise = :rician)))
        @test occursin("Scenario takes no noise", msg(() -> scenarios(QMTSIR, [:SIR_WM_I]; noise = :rician)))
        @test occursin("not a set", msg(() -> montecarlo(mm, two; noise = :rician, snr = 200, n = 10, simulate_with = mm => nominal(mm))))
    end

    @testset "ProtocolDesign and optimize_protocol: Dortch 2018 Table 2 by search" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        tissues = (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)
        # Eq. 5 with the MATLAB code's settings, as in the crlb tests: signed signal, t_FSE 0.3 s,
        # SNR 200, the bounds of its biased fits; TR ≥ 0.8 s (TI + TD ≥ 0.5 s), times 10 ms–10 s
        bounds4 = (PSR = Free(lower = 0, upper = 1), R1f = Free(lower = 0, upper = 3),
                   Sf = Free(lower = -1.05, upper = 0), M0f = Free(lower = 0, upper = 10))
        st(m) = sum(m.TI .+ m.TD .+ 0.3)
        g(m) = m.TI .+ m.TD .- 0.5
        space = (TI = (0.01, 10.0), TD = (0.01, 10.0))
        m4 = QMTSIR(; SIR_4POINT..., magnitude = false)
        m5 = QMTSIR(; SIR_5POINT..., magnitude = false)
        s4 = [Scenario(QMTSIR, nominal(QMTSIR, t); kmf = 12.5, bounds4...) for t in tissues]
        s5 = [Scenario(QMTSIR, nominal(QMTSIR, t); kmf = Free()) for t in tissues]
        d4 = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, constraints = g)
        d5 = ProtocolDesign(m5, s5; snr = 200, acquisition = space, scantime = st, constraints = g)
        # the objective: Eq. 5 at the Table 2 schemes, as crlb gives it, both covariances
        eq5(m, ss; cov = :sandwich) = maximum(sum(abs2, values(crlb(m, s; snr = 200, cov).relrmse)) for s in ss) * st(m)
        @test d4(m4) ≈ 0.243395 rtol = 2e-6
        @test d4(m4) ≈ eq5(m4, s4) rtol = 1e-12
        @test d5(m5) ≈ 1.602062 rtol = 2e-6
        @test d5(m5) ≈ eq5(m5, s5) rtol = 1e-12
        d4b = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, cov = :biased)
        @test d4b(m4) ≈ eq5(m4, s4; cov = :biased) rtol = 1e-12
        # the same as x (the TIs, then the TDs), without allocation; one cache for the 4 tissues
        x4 = [m4.TI; m4.TD]
        @test d4(x4) == d4(m4) && length(d4.groups) == 1
        allocs(d, x) = @allocated d(x)
        allocs(d4, x4)
        @test allocs(d4, x4) == 0
        # a broken constraint: Inf; the options: weights, a budget, another aggregate, a shared TD
        @test d4([0.01, 0.01, 0.278, 0.1, 4.171, 0.684, 2.73, 0.01]) == Inf     # TI + TD < 0.5
        cs = [crlb(m4, s; snr = 200) for s in s4]
        dw = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, weights = (R1f = 0, Sf = 0, M0f = 0))
        @test dw(m4) ≈ maximum(c.relrmse.PSR^2 for c in cs) * st(m4) rtol = 1e-12
        db = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, budget = st(m4))
        @test db(m4) ≈ d4(m4) / st(m4) rtol = 1e-12             # B = T: Eq. 5 without × T
        dm = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, aggregate = v -> sum(v) / length(v))
        @test dm(m4) ≈ sum(sum(abs2, values(c.relrmse)) for c in cs) / 4 * st(m4) rtol = 1e-12
        m4s = QMTSIR(TI = m4.TI, TD = 2.5, magnitude = false)
        dsh = ProtocolDesign(m4s, s4; snr = 200, acquisition = space, shared = (:TD,), scantime = st)
        @test length(dsh.lower) == 5 && dsh([m4.TI; 2.5]) == dsh(m4s)
        @test dsh(m4s) ≈ eq5(m4s, s4) rtol = 1e-12
        # the search reaches the paper's optima (Fopt4 0.2433, Fopt5 1.6020, the GA's before
        # rounding), and the rounded protocols are no worse than Table 2's own
        r4 = optimize_protocol(d4; starts = 2, generations = 400, rng = MersenneTwister(1))
        @test r4.value <= 0.2433 && r4.rounded_value <= d4(m4)
        @test all(g(r4.rounded) .>= 0) && issorted(r4.protocol.TI)
        @test r4.rounded_value ≈ d4(r4.rounded) rtol = 1e-12   # the order of the points aside
        @test all(x -> isinteger(round(1000x; digits = 9)), [r4.rounded.TI; r4.rounded.TD])  # the ms
        r5 = optimize_protocol(d5; starts = 4, generations = 400, rng = MersenneTwister(2))
        @test r5.value <= 1.6021 && r5.rounded_value <= 1.6021
        @test optimize_protocol(d4; starts = 2, generations = 50, rng = MersenneTwister(3)).x ==
              optimize_protocol(d4; starts = 2, generations = 50, rng = MersenneTwister(3)).x   # repeatable
        s = sprint(show, MIME"text/plain"(), r4)
        @test startswith(s, "DesignResult for QMTSIR, 4 points: value 0.2432") && occursin("\n  rounded:  TI = [0.01, 0.01,", s)
        @test sprint(show, d4) == "ProtocolDesign(QMTSIR, 4 points: TI ∈ [0.01, 10.0], TD ∈ [0.01, 10.0]; 4 scenarios)"
        # mistakes
        @test occursin("give `scantime", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = space)))
        @test occursin("the acquisition to optimize is `acquisition", msg(() -> ProtocolDesign(m4, s4; snr = 200, vary = space, scantime = st)))
        @test occursin("ProtocolDesign takes no noise", msg(() -> ProtocolDesign(m4, s4; snr = 200, noise = :rician, acquisition = space, scantime = st)))
        @test occursin("give `acquisition`", msg(() -> ProtocolDesign(m4, s4; snr = 200, scantime = st)))
        @test occursin("has no field `TR`", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = (TR = (0.1, 1.0),), scantime = st)))
        @test occursin("not a per-observation field", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = (magnitude = (0, 1),), scantime = st)))
        @test occursin("(lower, upper)", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = (TI = (1.0, 0.1),), scantime = st)))
        @test occursin("`snr` or `σ`", msg(() -> ProtocolDesign(m4, s4; acquisition = space, scantime = st)))
        @test occursin("not a free parameter", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, weights = (kmf = 1,))))
        @test occursin("the scenario is for QMTSIR", msg(() -> ProtocolDesign(T2Decay(TE = TD5), s4; snr = 200, acquisition = (TE = (0.01, 1.0),), scantime = sum)))
        @test occursin("is 0 in a scenario's truth", msg(() -> ProtocolDesign(m4, [Scenario(QMTSIR, nominal(QMTSIR, :SIR_WM_I; PSR = 0.0))];
                                                                              snr = 200, acquisition = space, scantime = st)))
    end

    @testset "parameters(m; changes...): roles made by a call, kept and passed" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        m = QMTSIR(; SIR_4POINT...)
        θ = nominal(m, :SIR_WM_I)
        roles = parameters(m; kmf = 10.0, Sm = 0.9)
        # every parameter's role, the two changed by the same rules as fit's keywords
        @test keys(roles) == keys(parameters(m))
        @test roles.kmf isa Fixed && roles.kmf.value == 10.0 && roles.Sm.value == 0.9
        @test roles.PSR == parameters(m).PSR && roles.R1m == parameters(m).R1m
        @test parameters(m; kmf = Free()).kmf.start == 12.5             # freed: starts at its value
        @test parameters(m; R1f = Free(lower = 0.5)).R1f == Free(start = 1.0, lower = 0.5, upper = 10.0)
        @test parameters(T1IR(TI = [0.1, 0.5, 1.0, 2.0]); T1 = Free(start = 1.0)).T1.start == 1.0
        # a model with its own `parameters(m)` method: its roles, changed
        dm = DataStartTest([0.01, 0.02])
        @test parameters(dm; c = 2.0).c.value == 2.0 && parameters(dm; c = 2.0).T2 == parameters(dm).T2
        # passed whole, after the data or the truth: the same as the keywords
        y = simulate(m, θ; noise = :rician, snr = 200, rng = MersenneTwister(1))
        Y = simulate(m, θ; noise = :rician, snr = 200, n = 20, rng = MersenneTwister(2))
        kw = (kmf = 10.0, Sm = 0.9)
        @test fit(m, y, roles).params == fit(m, y; kw...).params
        @test isequal(fit(m, Y, roles).params, fit(m, Y; kw...).params)
        @test fit!(init(m, roles), y).params == fit(m, y; kw...).params
        # a scenario's fit, fitted to data
        s = Scenario(m, θ; kw...)
        @test fit(m, y, s.roles).params == fit(m, y; kw..., NamedTuple{(:PSR, :R1f, :Sf, :M0f)}(
            map(n -> Free(start = θ[n]), (:PSR, :R1f, :Sf, :M0f)))...).params
        # mistakes
        @test occursin("`kfm` is not a parameter of QMTSIR", msg(() -> parameters(m; kfm = 1.0)))
        @test occursin("role", msg(() -> crlb(m, roles; snr = 200)))   # roles where the truth goes
    end

    @testset "io: loaddata, loadmap, savemaps, writetable (NIfTI and MATLAB files)" begin
        dir = mktempdir()
        path(name) = joinpath(dir, name)
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        writejson(p, d) = open(io -> JSON.json(io, d), p, "w")
        # A small T2 dataset: 6×5×2 voxels, 8 echoes, written as a scanner pipeline would, with a
        # geometry that is not the default (2 × 2 × 3 mm voxels, an offset)
        te = collect(0.01:0.02:0.15)
        T2map = collect(reshape(range(0.04, 0.12; length = 60), 6, 5, 2))
        m = T2Decay(TE = te)
        Y = simulate(m, (S0 = fill(1000.0, 6, 5, 2), T2 = T2map); snr = 200, noise = :gaussian,
                     rng = MersenneTwister(1))
        geometry = (voxel_size = (2.0f0, 2.0f0, 3.0f0),
                    orientation = Float32[2 0 0 -10; 0 2 0 -20; 0 0 3 5])
        niwrite(p, a) = NIfTI.niwrite(p, NIfTI.NIVolume(a; geometry...))

        # One 4-D file, its sidecar giving one EchoTime per volume
        niwrite(path("t2.nii.gz"), Float32.(Y))
        writejson(path("t2.json"), Dict("EchoTime" => te, "RepetitionTime" => 2.0))
        data, info = loaddata(path("t2.nii.gz"))
        @test size(data) == (6, 5, 2, 8) && eltype(data) == Float64 && data == Float32.(Y)
        @test info.spatial == (6, 5, 2) && info.shape == [6, 5, 2]
        @test info.found[:TE] ≈ te && info.found[:TR] == fill(2.0, 8) && !haskey(info.found, :TI)
        @test info.header !== nothing && info.files == [path("t2.nii.gz")]
        # One 3-D file per echo (as dcm2niix writes a multi-echo series), each with its
        # sidecar, stacked in the order given; the intensity scaling applied
        files = [path("echo$k.nii") for k in 1:8]
        for k in 1:8
            raw = round.(Int16, Y[:, :, :, k] ./ 0.5)
            NIfTI.niwrite(files[k], NIfTI.NIVolume(raw; geometry..., scl_slope = 0.5f0))
            writejson(path("echo$k.json"), Dict("EchoTime" => te[k]))
        end
        d2, i2 = loaddata(files)
        @test size(d2) == (6, 5, 2, 8) && maximum(abs.(d2 .- Y)) <= 0.25 && i2.found[:TE] ≈ te
        # Float32: half the memory, the same values rounded (scaled in Float64 first)
        d32, _ = loaddata(files; precision = Float32)
        @test eltype(d32) == Float32 && d32 == Float32.(d2)
        # Diffusion: an FSL .bval file beside the data
        niwrite(path("dwi.nii"), Float32.(Y[:, :, :, 1:4]))
        write(path("dwi.bval"), "0 500 1000 2000\n")
        @test loaddata(path("dwi.nii"))[2].found[:b] == [0.0, 500, 1000, 2000]
        # Mistakes
        NIfTI.niwrite(path("other.nii"), NIfTI.NIVolume(Float32.(Y[:, :, :, 1])))
        niwrite(path("small.nii"), zeros(Float32, 4, 4, 2))
        @test occursin("different geometry", msg(() -> loaddata([files[1], path("other.nii")])))
        @test occursin("same size", msg(() -> loaddata([files[1], path("small.nii")])))
        @test occursin("one image is one observation", msg(() -> loaddata(files[1])))
        @test occursin("give NIfTI files", msg(() -> loaddata(path("t2.json"))))
        @test occursin("no such file", msg(() -> loaddata(path("missing.nii.gz"))))
        writejson(path("echo1.json"), Dict("EchoTime" => [0.01, 0.02]))
        @test occursin("EchoTime has 2 values for 1 images", msg(() -> loaddata(files)))

        # loadmap: a mask in the data's geometry; mistakes named
        mask = T2map .> 0.06
        niwrite(path("mask.nii.gz"), UInt8.(mask))
        @test loadmap(path("mask.nii.gz"), info) == mask
        @test occursin("different geometry", msg(() -> loadmap(path("other.nii"), info)))
        @test occursin("the data are 6×5×2", msg(() -> loadmap(path("small.nii"), info)))
        @test readvalues === EQUIP.MRImages.readvalues        # re-exported

        # savemaps: the maps in the data's geometry, with sidecars recording how they were made
        res = fit(T2Decay(TE = info.found[:TE]), data; mask, T2 = Free(start = 0.07))
        written = savemaps(path("out/sub-01_"), res, info; stderror = true,
                           sidecar = Dict("Data" => "t2.nii.gz"))
        @test basename.(written) == ["sub-01_S0map.nii.gz", "sub-01_T2map.nii.gz",
                                     "sub-01_status.nii.gz", "sub-01_SSE.nii.gz",
                                     "sub-01_S0map_SE.nii.gz", "sub-01_T2map_SE.nii.gz"]
        t2 = readimage(written[2])
        @test size(t2) == (6, 5, 2) && eltype(NIfTI.niread(written[2]).raw) == Float32
        @test isequal(t2.data, Float64.(Float32.(res.params.T2)))
        @test all(isnan, t2.data[.!mask]) && maximum(abs.(t2.data[mask] .- T2map[mask])) < 0.01
        @test samegeometry(t2, readimage(path("t2.nii.gz")))
        @test t2.header.scl_slope == 1 && t2.header.scl_inter == 0
        @test readimage(written[3]).data == Int.(res.status)
        @test t2.meta["Map"] == "T2map" && t2.meta["Model"] == "T2Decay" &&
              t2.meta["Acquisition"]["TE"] ≈ te && t2.meta["Data"] == "t2.nii.gz"
        @test t2.meta["Fit"]["T2"] == "Free(start = 0.07)" && t2.meta["Fit"]["precision"] == "Float64"
        @test startswith(t2.meta["Software"], "EQUIP ")
        @test occursin("one voxel's", msg(() -> savemaps(path("x_"), fit(m, Y[1, 1, 1, :]), info)))
        @test occursin("the data were 6×5×2", msg(() -> savemaps(path("x_"), fit(m, Y[1:2, :, :, :]), info)))

        # MATLAB files: the data as an array, observations last; maps back in its own shape
        S = reshape(Y, 60, 8)                                  # voxels already masked: nvoxels×N
        writemat(path("data.mat"), Dict("S" => S, "TE" => collect(te'), "mask" => Float64.(vec(mask)),
                                        "TEcolumn" => reshape(te, 8, 1), "five" => rand(2, 2, 2, 2, 2)))
        dm, im = loaddata(path("data.mat:S"))
        @test size(dm) == (60, 1, 1, 8) && vec(dm[:, 1, 1, 3]) == S[:, 3]
        @test im.header === nothing && im.spatial == (60, 1, 1) && im.shape == [60]
        @test isempty(im.found) && readvalues(path("data.mat:TE")) == te
        mm = loadmap(path("data.mat:mask"), im) .> 0
        @test size(mm) == (60, 1, 1) && vec(mm) == vec(mask)
        rm_ = fit(m, dm; mask = mm)
        @test isequal(vec(rm_.params.T2), vec(fit(m, S; mask = vec(mask)).params.T2))
        out = savemaps(path("mat/sub-01_"), rm_, im)               # :mat, the default for .mat data
        @test out == [path("mat/sub-01_maps.mat")] && isfile(path("mat/sub-01_maps.json"))
        back = readmat(out[1])
        @test Set(keys(back)) == Set(["S0map", "T2map", "status", "SSE"])
        @test size(back["T2map"]) == (60,) && isequal(back["T2map"], Float32.(vec(rm_.params.T2)))
        @test JSON.parsefile(path("mat/sub-01_maps.json"))["Maps"] == ["S0map", "T2map", "status", "SSE"]
        # 4-D, as NIfTI; maps to NIfTI in another image's geometry
        writemat(path("data4.mat"), Dict("S" => Y))
        d4, i4 = loaddata(path("data4.mat"))                     # one variable: no name needed
        @test d4 == Y && i4.shape == [6, 5, 2]
        r4 = fit(m, d4)
        f4 = savemaps(path("nii/x_"), r4, i4; like = readimage(path("t2.nii.gz")))
        @test samegeometry(readimage(f4[2]), readimage(path("t2.nii.gz")))
        @test occursin("observations along the last", msg(() -> loaddata(path("data.mat:five"))))
        @test occursin("has one observation", msg(() -> loaddata(path("data.mat:TEcolumn"))))
        @test occursin("give one", msg(() -> loaddata([path("data.mat:S"), path("data.mat:S")])))
        @test occursin("the data's images are 60", msg(() -> loadmap(path("data4.mat:S"), im)))

        # writetable: rows as TSV
        rows = [(name = "WM", PSR = 0.1, n = 3), (name = "GM", PSR = 0.05, n = 4)]
        f = writetable(path("tables/t.tsv"), rows)
        @test readlines(f) == ["name\tPSR\tn", "WM\t0.1\t3", "GM\t0.05\t4"]
        @test_throws ArgumentError writetable(path("e.tsv"), [])
    end

    @testset "SMT: the spherical mean technique (Kaden 2016; Devan et al., MRI 2020)" begin
        rng = MersenneTwister(21)
        # The stick's spherical mean, √π·erf(x)/(2x): its series near 0 and erf beyond, the same
        # (values and derivatives) on both sides of the switch; and against the mean over
        # directions it stands for, ∫₀¹ exp(−x²u²) du (u = cos ψ: uniform over the sphere)
        f(x2) = sqrt(π) * SpecialFunctions.erf(sqrt(x2)) / (2 * sqrt(x2))
        for x2 in (1e-3, 0.0499999, 0.0500001, 0.7, 12.0)
            @test EQUIP._sphmean(x2) ≈ f(x2) rtol = 1e-11
            @test ForwardDiff.derivative(EQUIP._sphmean, x2) ≈ ForwardDiff.derivative(f, x2) rtol = 1e-8
        end
        @test EQUIP._sphmean(0.0) == 1 && ForwardDiff.derivative(EQUIP._sphmean, 0.0) ≈ -1 / 3
        u = range(0, 1; length = 200_001)                         # the trapezoid rule over cos ψ
        trap(g) = (sum(g.(u)) - (g(0) + g(1)) / 2) * step(u)
        b, D, Dperp = 2000.0, 2.0e-3, 0.7e-3
        @test EQUIP._sphmean(b * D) ≈ trap(c -> exp(-b * D * c^2)) rtol = 1e-9                     # stick
        @test exp(-b * Dperp) * EQUIP._sphmean(b * (D - Dperp)) ≈
              trap(c -> exp(-b * (Dperp + (D - Dperp) * c^2))) rtol = 1e-9                      # zeppelin
        # The model: S0 at b = 0; free diffusion when v_ax = 0 (D⊥ = D_ax); sticks alone when 1
        m = SMT(b = [0.0, 1000.0, 3000.0])
        @test signal(m, (S0 = 2.0, vax = 0.4, Dax = 2e-3))[1] == 2.0
        @test signal(m, (S0 = 1.0, vax = 0.0, Dax = 2e-3)) ≈ exp.(-m.b .* 2e-3)
        @test signal(m, (S0 = 1.0, vax = 1.0, Dax = 2e-3)) ≈ [1.0; f.(m.b[2:3] .* 2e-3)]
        @test nominal(m, :SMT_cBH).vax == 0.241 && nominal(m, :SMT_NWM).Dax == 1.9e-3

        # Devan 2020, Fig. 1: the vax precision efficiency 1/√(N·CRLB/vax²) of the five
        # protocols, for an isotropic fibre arrangement (their worst case, where each direction's
        # signal is the spherical mean). At SNR 20 per direction (their simulations' SNR; Fig. 1
        # does not say), with the shells' means weighted by their directions, it is their figure:
        # within 5% for normal white matter, 15% for the others. Theirs adds the Rician factor,
        # which lowers the precision most where the signal is lowest (protocols A and B's
        # b = 3790–4230 shells, in black holes, the lowest vax): ours is higher there.
        fig1 = (SMT_NWM = (A = 0.90, B = 1.00, C = 1.10, D = 0.865, E = 0.84),     # read off the plot
                SMT_NAWM = (A = 1.10, B = 1.06, C = 1.05, D = 0.83, E = 0.82),
                SMT_cBH = (A = 0.90, B = 0.905, C = 0.825, D = 0.685, E = 0.68),
                SMT_T2lesion = (A = 1.11, B = 1.06, C = 0.99, D = 0.79, E = 0.79))
        for name in keys(fig1), P in keys(SMT_PROTOCOLS)
            pr = SMT_PROTOCOLS[P]
            ms = SMT(b = pr.b)
            θ = nominal(ms, name)
            c = crlb(ms, θ; snr = 20, averages = pr.directions, weights = pr.directions)
            eff = θ.vax / (c.sd.vax * sqrt(sum(pr.directions)))
            @test eff ≈ fig1[name][P] rtol = (name === :SMT_NWM ? 0.06 : 0.15)
        end

        # Fits from the default starts: unbiased, and as precise as crlb says (Gaussian, SNR 20,
        # chronic black hole, protocol A), in both precisions
        pr = SMT_PROTOCOLS.A
        ms = SMT(b = pr.b)
        θ = nominal(ms, :SMT_cBH)
        Y = simulate(ms, θ; snr = 20, averages = pr.directions, noise = :gaussian, n = 2000, rng)
        c = crlb(ms, θ; snr = 20, averages = pr.directions, weights = pr.directions)
        for T in (Float64, Float32)
            r = fit(ms, T.(Y); weights = pr.directions, precision = T)
            @test all(isconverged, r.status) && eltype(r.params.vax) == T
            @test abs(mean(r.params.vax) - θ.vax) < 4 * c.sd.vax / sqrt(2000)
            @test std(r.params.vax) ≈ c.sd.vax rtol = 0.1
            @test std(r.params.Dax) ≈ c.sd.Dax rtol = 0.1
        end

        # sphericalmean: the volumes of each shell averaged (b-values within 100 s/mm² merged),
        # observations last, any shape; the shells' b and their numbers of volumes
        bvals = [0, 1000, 5, 995, 3000, 1010, 2990, 0]
        data = reshape(Float64.(1:2*3*8), 2, 3, 8)
        means, shells, dirs = sphericalmean(data, bvals)
        @test size(means) == (2, 3, 3) && dirs == [3, 3, 2]
        @test shells ≈ [5 / 3, (995 + 1000 + 1010) / 3, 2995]
        @test means[1, 1, :] ≈ [mean(data[1, 1, [1, 3, 8]]), mean(data[1, 1, [2, 4, 6]]), mean(data[1, 1, [5, 7]])]
        @test sphericalmean(data, bvals; tol = 10)[3] == [3, 2, 1, 2]          # 1010 more than 10 above 995
        @test sphericalmean(collect(1.0:4.0), [0, 0, 1000, 1000])[1] == [1.5, 3.5]
        @test_throws DimensionMismatch sphericalmean(data, bvals[1:7])
    end

    @testset "@mrimodel: a model from a function of one observation" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        mmerr(ex) = try macroexpand(@__MODULE__, ex); "" catch e; sprint(showerror, e isa LoadError ? e.error : e) end
        rng = MersenneTwister(11)
        # The pieces: the type, its acquisition, the roles (a start from the acquisition, a value
        # fixed, a tie by name, a tie by formula), the docstring
        te = collect(0.01:0.01:0.32)
        m = MacroT2Offset(TE = te)
        @test m isa MRIModel && m.TE === te && nobs(m) == 32
        @test keys(parameters(MacroT2Offset)) == (:S0, :T2, :C)
        @test parameters(m).T2.start == mean(te)                  # Free(start = mean(TE))
        @test parameters(MacroSIR) == parameters(QMTSIR)          # kmf = 12.5, R1m = R1f, ...
        @test parameters(TieFormula).k2 isa Fixed{<:Function}
        @test occursin("made with @mrimodel", string(@doc MacroT2Offset))
        @test MacroSIR(TI = TI5, TD = TD5).magnitude && !MacroSIR(TI = TI5, TD = TD5, magnitude = false).magnitude
        @test resolve(TieFormula(t = [1.0]), (A = 1.0, k1 = 2.0, k2 = 0.5 * 2.0)) == (A = 1.0, k1 = 2.0, k2 = 1.0)

        # The same fits as the built-in models, bit for bit (the same equation, per observation),
        # in both precisions
        mb = T2DecayOffset(TE = te)
        θ = (S0 = fill(1000.0, 4, 3), T2 = collect(reshape(range(0.04, 0.12; length = 12), 4, 3)), C = fill(20.0, 4, 3))
        Y = simulate(mb, θ; snr = 100, noise = :rician, rng)
        @test signal(m, (S0 = 1000.0, T2 = 0.08, C = 20.0)) == signal(mb, (S0 = 1000.0, T2 = 0.08, C = 20.0))
        for T in (Float64, Float32)
            a, b = fit(m, T.(Y); precision = T), fit(mb, T.(Y); precision = T)
            @test isequal(a.params, b.params) && a.iterations == b.iterations && eltype(a.params.T2) == T
        end
        fa = [2.0, 5, 10, 15, 20, 30]
        mm, mt = MacroMFA(FA = fa, TR = 0.015), T1MFA(FA = fa, TR = 0.015)    # one TR for all
        Ym = simulate(mt, (M0 = fill(1000.0, 5), T1 = collect(range(0.6, 1.8; length = 5)), B1 = 1.0); snr = 200,
                      noise = :gaussian, rng)
        for T in (Float64, Float32)
            @test isequal(fit(mm, T.(Ym); precision = T).params, fit(mt, T.(Ym); precision = T).params)
        end
        @test isequal(fit(mm, Ym; B1 = 0.9).params, fit(mt, Ym; B1 = 0.9).params)
        ms, mq = MacroSIR(TI = TI5, TD = TD5), QMTSIR(TI = TI5, TD = TD5)
        θs = map(x -> x isa Real ? fill(Float64(x), 3, 2) : x, nominal(mq, :SIR_lesion))
        Ys = simulate(mq, θs; snr = 200, noise = :rician, rng)
        for T in (Float64, Float32)
            a, b = fit(ms, T.(Ys); precision = T), fit(mq, T.(Ys); precision = T)
            @test isequal(a.params, b.params) && a.iterations == b.iterations
        end
        @test isequal(fit(ms, Ys; kmf = Free(start = 10.0)).params, fit(mq, Ys; kmf = Free(start = 10.0)).params)
        # crlb and montecarlo take it as any model
        c1, c2 = crlb(ms, nominal(mq, :SIR_WM_I); snr = 200), crlb(mq, nominal(mq, :SIR_WM_I); snr = 200)
        @test all(k -> isapprox(c1.sd[k], c2.sd[k]; rtol = 1e-10), keys(c2.sd))

        # No allocation per fit
        c = init(ms)
        y = Ys[1, 1, :]
        fit!(c, y)
        @test (@allocated fit!(c, y)) == 0

        # A tie by formula; a free parameter with no start: given in the call
        mt2 = TieFormula(t = collect(0.1:0.1:2.0))
        yt = signal(mt2, (A = 2.0, k1 = 1.5, k2 = 0.75))
        @test fit(mt2, yt).params.k1 ≈ 1.5
        mn = NoStart(t = collect(0.1:0.1:2.0))
        yn = signal(mn, (A = 2.0, k = 1.5))
        @test occursin("`k` has no starting value: give one in the call, `k = Free(start = ...)`",
                       msg(() -> fit(mn, yn)))
        @test fit(mn, yn; k = Free(start = 1.0)).params.k ≈ 1.5

        # Lengths checked once against the output
        @test occursin("MacroSIR: `TD` has 4 values for 5 observations",
                       msg(() -> signal!(zeros(5), MacroSIR(TI = TI5, TD = TD5[1:4]), resolve(mq, nominal(mq)))))

        # Mistakes, said while the macro expands
        @test occursin("no acquisition", mmerr(:(@mrimodel A1(; S0) = S0)))
        @test occursin("no parameters", mmerr(:(@mrimodel A2(t) = t)))
        @test occursin("`t` is used twice", mmerr(:(@mrimodel A3(t; t) = t)))
        @test occursin("no types", mmerr(:(@mrimodel A4(t::Vector; S0) = S0 * t)))
        @test occursin("no types", mmerr(:(@mrimodel A5(t; S0::Float64) = S0 * t)))
        @test occursin("acquisition comes before the options", mmerr(:(@mrimodel A6(t, mag = true, u; S0) = S0)))
        @test occursin("cannot depend on another parameter", mmerr(:(@mrimodel A7(t; a = 1.0, b = Free(start = 2a)) = a * b)))
        @test occursin("only the start may use the acquisition", mmerr(:(@mrimodel A8(t; a = Free(lower = minimum(t))) = a)))
        @test occursin("cannot depend on the acquisition", mmerr(:(@mrimodel A9(t; a = maximum(t)) = a)))
        @test occursin("tied to itself", mmerr(:(@mrimodel A10(t; a = a) = a)))
        @test occursin("write a function definition", mmerr(:(@mrimodel A11)))
        @test occursin("where", mmerr(:(@mrimodel A12(t; a) where {T} = a)))
    end

    @testset "newmodel: a model file, blank or from an @mrimodel definition" begin
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        dir = mktempdir()
        rng = MersenneTwister(12)
        # A file loaded into a module of its own (as a user's `include`), its model found by name
        # (through `invokelatest`: the names are newer than this code).
        function load(file)
            mod = Module()
            Core.eval(mod, :(using EQUIP, Statistics))
            Base.include(mod, file)
            return mod
        end
        # From @mrimodel: the code written out, loaded, gives the same fits as the macro's model
        # and the built-in one
        f = newmodel(MacroSIR; file = joinpath(dir, "sir.jl"))
        @test f == joinpath(dir, "sir.jl") && occursin("@inline function _macrosir(TI, TD, magnitude, PSR", read(f, String))
        @test occursin("R1m     = Fixed(:R1f)", read(f, String)) && occursin("return magnitude ? abs(s) : s", read(f, String))
        mod = load(f)
        mw = Base.invokelatest(Base.invokelatest(getproperty, mod, :MacroSIR); TI = TI5, TD = TD5)
        mq = QMTSIR(TI = TI5, TD = TD5)
        Ys = simulate(mq, map(x -> x isa Real ? fill(Float64(x), 3, 2) : x, nominal(mq, :SIR_lesion));
                      snr = 200, noise = :rician, rng)
        @test isequal(Base.invokelatest(fit, mw, Ys).params, fit(mq, Ys).params)
        @test Base.invokelatest(parameters, mw) == parameters(mq)
        f = newmodel(MacroT2Offset; file = joinpath(dir, "t2.jl"))
        txt = read(f, String)
        @test occursin("T2 = Free(start = m -> mean(m.TE), lower = 1.0e-6)", txt)
        @test occursin("made with @mrimodel", txt)                  # its docstring kept
        mw = Base.invokelatest(Base.invokelatest(getproperty, load(f), :MacroT2Offset); TE = collect(0.01:0.01:0.32))
        Yt = simulate(T2DecayOffset(TE = mw.TE), (S0 = fill(1000.0, 4), T2 = [0.05, 0.07, 0.09, 0.11],
                      C = fill(10.0, 4)); snr = 100, noise = :gaussian, rng)
        @test isequal(Base.invokelatest(fit, mw, Yt).params, fit(T2DecayOffset(TE = mw.TE), Yt).params)
        @test occursin("MacroMFA(FA, TR::Real)", read(newmodel(MacroMFA; file = joinpath(dir, "mfa.jl")), String))

        # A blank model: it loads, its roles are there, its signal is a TODO (zeros)
        f = newmodel(:BlankIR; acquisition = [:TI, :TD], parameters = [:M0, :T1],
                     options = (magnitude = true,), file = joinpath(dir, "sub", "blank.jl"))
        mod = load(f)
        mb = Base.invokelatest(Base.invokelatest(getproperty, mod, :BlankIR); TI = [0.1, 1.0], TD = [2.0, 2.0])
        @test mb.magnitude === true && Base.invokelatest(nobs, mb) == 2
        @test keys(Base.invokelatest(parameters, mb)) == (:M0, :T1)
        @test Base.invokelatest(signal, mb, (M0 = 1.0, T1 = 1.0)) == [0.0, 0.0]
        f = newmodel(:Roles; acquisition = [:TE], file = joinpath(dir, "roles.jl"),
                     parameters = (S0 = Free(start = :scale, lower = 0), T2 = Free(start = m -> mean(m.TE)), C = Fixed(0.0)))
        txt = read(f, String)
        @test occursin("S0 = Free(start = :scale, lower = 0.0),", txt) && occursin("T2 = Free(start = m -> TODO)", txt)
        @test keys(Base.invokelatest(parameters, Base.invokelatest(getproperty, load(f), :Roles))) == (:S0, :T2, :C)   # (the type: the TODO start is not called)

        # Mistakes
        @test occursin("exists", msg(() -> newmodel(:Roles; acquisition = [:TE], parameters = [:S0], file = f)))
        @test newmodel(:Roles; acquisition = [:TE], parameters = [:S0], file = f, force = true) == f
        @test occursin("was not made with @mrimodel", msg(() -> newmodel(T2Decay)))
        @test occursin("`t` is used twice", msg(() -> newmodel(:X; acquisition = [:t], parameters = [:t], file = joinpath(dir, "x.jl"))))
        @test occursin("give the acquisition", msg(() -> newmodel(:X; acquisition = Symbol[], parameters = [:a], file = joinpath(dir, "x.jl"))))
        @test occursin("not a Julia name", msg(() -> newmodel("my model"; acquisition = [:t], parameters = [:a])))
    end

    @testset "fit, fit! and nobs are StatsAPI's functions (no clash with StatsBase etc.)" begin
        @test nameof(parentmodule(fit)) == :StatsAPI
        @test nameof(parentmodule(fit!)) == :StatsAPI
        @test nameof(parentmodule(nobs)) == :StatsAPI
        @test nameof(parentmodule(init)) == :CommonSolve
    end
end
