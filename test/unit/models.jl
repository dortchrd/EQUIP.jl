# =================================================================================================
# test/unit/models.jl: The models' signals, against their equations and their limits: noise-free recovery for every
# built-in model, lengths, construction, QMTSIR against the matrix form (Dortch et al., MRM 2018,
# Eq. 1), T1IRTD's limits (Li et al., MRM 2010, Eq. 1), the time and rate forms, the compartments
# against Gauss–Legendre quadrature and closed forms.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl models
# =================================================================================================

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
    @test_noalloc @allocated(nobs(mq)) == 0
    @test occursin("TI has 2 values and TD has 3", sprint(showerror, try nobs(QMTSIR(TI = [0.01, 0.1], TD = [1.0, 2.0, 3.0])) catch e; e end))
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

@testset "QMTSIREqualR1: the one line against the matrix form, its limits and derivatives" begin
    # The matrix form (Dortch et al. 2018, Eq. 1, test/setup.jl) with R1m = R1f and both pools
    # saturated at the start of TD: random parameters, signed and magnitude
    rng = MersenneTwister(9)
    for _ in 1:200
        θ = (PSR = 0.3 * rand(rng), R1f = 0.3 + 2 * rand(rng), Sf = -1.1 * rand(rng), M0f = 1000 * rand(rng),
             kmf = 40 * rand(rng), Sm = rand(rng))
        for mag in (false, true)
            @test signal(QMTSIREqualR1(TI = TI5, TD = TD5, magnitude = mag), θ) ≈
                  sir_matrix(TI5, TD5; θ..., R1m = θ.R1f, magnitude = mag) rtol = 1e-10
        end
    end
    m = QMTSIREqualR1(TI = TI5, TD = TD5, magnitude = false)
    θ = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, Sm = 0.83)
    # no exchange (kmf = 0, where QMTSIR's form divides by zero), no macromolecular pool, or both
    # pools inverted alike: inversion recovery, Li et al. 2010 Eq. 1 written out
    li = @. 1000 * (-0.95 * (1 - exp(-TD5)) * exp(-TI5) + 1 - exp(-TI5))
    @test signal(m, merge(θ, (kmf = 0.0,))) ≈ li
    @test signal(m, merge(θ, (PSR = 0.0,))) ≈ li
    @test signal(m, merge(θ, (Sm = -0.95,))) ≈ li
    # the fit's derivatives (dual numbers): QMTSIR's with its defaults, R1m tied to R1f in the
    # values (`jacobian` follows θ's ties: with R1m = 1.0 the R1f column would hold R1m still)
    q = QMTSIR(TI = TI5, TD = TD5, magnitude = false)
    θq = merge(θ, (R1m = :R1f, Mzf_td0 = 0.0, Mzm_td0 = 0.0))
    @test jacobian(m, θ) ≈ jacobian(q, θq) rtol = 1e-10
    @test jacobian(m, θ; kmf = Free()) ≈ jacobian(q, θq; kmf = Free()) rtol = 1e-10
    # one TD for every TI; the tissue presets and protocols are QMTSIR's
    @test QMTSIREqualR1(TI = [0.01, 0.1], TD = 2.5).TD == [2.5, 2.5]
    @test nominal(QMTSIREqualR1(; SIR_4POINT...), :SIR_WM_I) == (PSR = 0.2, R1f = 1.5, Sf = -1.0, M0f = 1.0, kmf = 5.0, Sm = 0.83)
end

@testset "qMT lineshapes and pulses: normalized, the super-Lorentzian's table, SIR's Sf and Sm" begin
    # each lineshape integrates to 1 over the angular offset: ∫ g d(2πΔ) = 4π ∫₀^∞ g dΔ, by the
    # trapezoid rule in ln Δ (the super-Lorentzian's ln(1/Δ) divergence integrable; below 1e-6 Hz
    # about 1e-11 of the whole), T2 = 12 µs; the Lorentzian's to 1e7 Hz against its arctangent
    v = range(log(1e-6), log(1e7); length = 4001)
    area(g) = 4π * sum(i -> (i in (1, length(v)) ? 0.5 : 1.0) * g(12e-6, exp(v[i])) * exp(v[i]), eachindex(v)) * step(v)
    @test area(gaussian) ≈ 1 rtol = 1e-8
    @test area(superlorentzian) ≈ 1 rtol = 1e-8
    @test area(lorentzian) ≈ (2 / π) * atan(2π * 1e7 * 12e-6) rtol = 1e-8
    @test gaussian(15e-6, 0.0) ≈ 15e-6 / sqrt(2π) && superlorentzian(12e-6, -3e3) == superlorentzian(12e-6, 3e3)
    # The super-Lorentzian's G(x) = g/T2 at x = 2π|Δ|T2: by quadrature and from the table, against
    # 25-digit values (QuadGK in BigFloat, rtol 1e-25, the u integral split at the magic angle;
    # computed 2026-11-26), from near resonance to far past the MT offsets used
    ref = ((1e-6, 6.40691009505922401299), (1e-2, 2.16392593577816564919), (1.0, 0.0658473493416759753783),
           (5.0, 1.86586916378355878189e-8), (20.0, 4.58179051200014677726e-91))
    @test all(isapprox(EQUIP._slGexact(x), G; rtol = 1e-13) for (x, G) in ref)
    @test all(isapprox(EQUIP._slG(x), G; rtol = 1e-12) for (x, G) in ref)
    @test superlorentzian(12e-6, 1e3) ≈ 12e-6 * EQUIP._slGexact(2π * 1e3 * 12e-6) rtol = 1e-12
    # the table (quintic Hermite in t = ln x, spacing 0.01) against the quadrature over its range,
    # on its points, 1e-9 to either side, mid-cell and near its ends: G within 1e-12, its first
    # derivative in x (a dual number through the interpolant) within 1e-10, its second (nested
    # duals, as crlb's) within 1e-8 (1e-13, 6e-12 and 9e-10 measured, relative), the second
    # continuous across a point; outside it (x < 8e-7, x ≥ 33, 0, NaN), G by quadrature
    G, Gq = EQUIP._slG, EQUIP._slGexact
    D1(f, x) = ForwardDiff.derivative(f, x)
    D2(f, x) = ForwardDiff.derivative(y -> ForwardDiff.derivative(f, y), x)
    ts = [range(-13.999, 3.499; length = 301); [-14 + 0.01i + s for i in (1, 700, 1500, 1749) for s in (-1e-9, 0.0, 1e-9)];
          [-14 + 0.01(i + 0.5) for i in (0, 900, 1749)]]
    rel(a, b) = abs(a / b - 1)
    @test maximum(t -> rel(G(exp(t)), Gq(exp(t))), ts) < 1e-12
    @test maximum(t -> rel(D1(G, exp(t)), D1(Gq, exp(t))), ts) < 1e-10
    @test maximum(t -> rel(D2(G, exp(t)), D2(Gq, exp(t))), ts) < 1e-8
    xp = exp(-14 + 0.01 * 900)
    @test rel(D2(G, xp * (1 - 1e-12)), D2(G, xp * (1 + 1e-12))) < 1e-8
    @test all(G(x) == Gq(x) for x in (5e-7, 34.0, 40.0)) && G(0.0) == Inf && isnan(G(NaN))
    @test !hasmethod(setindex!, Tuple{typeof(EQUIP._SL_TABLE.v),Any,Int})     # read-only
    # SIR's Sm, the macromolecular pool's saturation, Gaussian on resonance: the 1-ms block
    # inversion at T2m = 15 µs gives Dortch et al. 2018's 0.83; −ln Sm is the pulse's ∫ω1²dt
    # times π·g(0), so the composite 90x-180y-90x (twice the block's energy at its amplitude)
    # gives Sm², and B1 = 1.1, 1.21 times −ln Sm
    blk = blockpulse(180, 1e-3)
    cmp = RFPulse([90, 180, 90], [0, 90, 0], [0.5e-3, 1e-3, 0.5e-3])
    Sm = boundsaturation(blk, 15e-6)
    @test Sm ≈ 0.83 atol = 0.005
    @test boundsaturation(cmp, 15e-6) ≈ Sm^2 rtol = 1e-12
    @test log(boundsaturation(blk, 15e-6; B1 = 1.1)) ≈ 1.21 * log(Sm) rtol = 1e-12
    @test boundsaturation(blk, 12e-6; Δ = 2e3, lineshape = superlorentzian) ≈ exp(-π * superlorentzian(12e-6, 2e3) * π^2 / 1e-3)
    # SIR's Sf, the free pool's Mz after the pulse: a block on resonance, cos(B1·180°); off
    # resonance, the Rabi formula, (Δω² + ω1²·cos(ωτ))/ω²; the composite against a product of
    # matrix exponentials of the rotations' generators (Ω×), and its tolerance of B1 errors
    for B1 in (0.8, 1.0, 1.15)
        @test inversionefficiency(blk; B1) ≈ cosd(180 * B1) atol = 1e-14
    end
    ω1, Δω = π / 1e-3, 2π * 150.0
    ω = hypot(ω1, Δω)
    @test inversionefficiency(blk; Δ = 150.0) ≈ (Δω^2 + ω1^2 * cos(ω * 1e-3)) / ω^2 rtol = 1e-12
    function viaexp(p; B1, Δ)
        M = [0.0, 0.0, 1.0]
        for k in eachindex(p.flip)
            w1, φ = B1 * deg2rad(p.flip[k]) / p.duration[k], deg2rad(p.phase[k])
            Ωx, Ωy, Ωz = w1 * cos(φ), w1 * sin(φ), 2π * Δ
            M = exp([0 -Ωz Ωy; Ωz 0 -Ωx; -Ωy Ωx 0] * p.duration[k]) * M
        end
        return M[3]
    end
    @test maximum(abs(inversionefficiency(cmp; B1, Δ) - viaexp(cmp; B1, Δ)) for B1 in 0.7:0.1:1.3 for Δ in (-200.0, 0.0, 90.0)) < 1e-13
    @test inversionefficiency(cmp) ≈ -1 && inversionefficiency(cmp; B1 = 0.9) < inversionefficiency(blk; B1 = 0.9) - 0.04
    # dual numbers (a fitted B1 or T2m), against central differences; nothing allocated
    @test ForwardDiff.derivative(b -> inversionefficiency(cmp; B1 = b), 0.9) ≈
          (inversionefficiency(cmp; B1 = 0.9 + 1e-6) - inversionefficiency(cmp; B1 = 0.9 - 1e-6)) / 2e-6 rtol = 1e-6
    @test ForwardDiff.derivative(t -> boundsaturation(blk, t; Δ = 2e3, lineshape = superlorentzian), 12e-6) ≈
          (boundsaturation(blk, 12e-6 + 1e-11; Δ = 2e3, lineshape = superlorentzian) -
           boundsaturation(blk, 12e-6 - 1e-11; Δ = 2e3, lineshape = superlorentzian)) / 2e-11 rtol = 1e-5
    inversionefficiency(cmp; B1 = 0.9); boundsaturation(cmp, 15e-6)
    @test_noalloc @allocated(inversionefficiency(cmp; B1 = 0.9)) == 0
    @test_noalloc @allocated(superlorentzian(12e-6, 2e3)) == 0
    # a pulse's segments: one flip, phase and duration each; durations > 0; a number split evenly
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("phase (1)", msg(() -> RFPulse([90, 90], [0], [1e-3, 1e-3])))
    @test occursin("-0.001", msg(() -> blockpulse(180, -1e-3)))
    @test RFPulse([90, 180, 90], [0, 90, 0], 3e-3).duration == fill(1e-3, 3)
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
    # T2 ≤ 2.5 s and R2 ≥ 1/2.5 s in every T2 model, both forms (builtin.jl; the owner, 2026-11-08
    # and 09), so the two forms agree where the data do not decay: both end on the bound, flagged
    # in both precisions. Float32's bound 0.4f0 is 6e-9 from 0.4: `onbound` missed it until its
    # tolerance followed the fit's precision (2026-11-09), the regression the Float32 case guards.
    @test parameters(T2DecayOffset).T2.upper == 2.5 && parameters(R2DecayOffset).R2.lower == 0.4
    @test parameters(T2BiexpDecay).T2short.upper == parameters(T2BiexpDecay).T2long.upper == 2.5
    @test parameters(R2BiexpDecay).R2short.lower == parameters(R2BiexpDecay).R2long.lower == 0.4
    rising = collect(range(1000.0, 1010.0; length = length(te)))
    for T in (Float64, Float32)
        rT, rR = fit(mT, T.(rising); precision = T), fit(mR, T.(rising); precision = T)
        @test rT.params.T2 == T(2.5) && onbound(rT).T2
        @test rR.params.R2 ≈ T(0.4) && onbound(rR).R2
        @test rR.params.R2 ≈ 1 / rT.params.T2
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
    for adc in (0.1e-3, 0.7e-3, 3e-3)
        r = fit(m, 800 .* exp.(-b .* adc))
        @test r.params.ADC ≈ adc rtol = 1e-6
        @test r.params.S0 ≈ 800 rtol = 1e-6
    end
    # The diffusivity cap, 4e-3 mm²/s, above free water (builtin.jl's _DMAX; the owner,
    # 2026-11-09: "cap everywhere"): a faster decay ends on it, flagged; widened, the fit recovers
    # it from the default start (5e-3, as this test did before the cap). DTI's and DKI's diagonal
    # elements have it, their off-diagonal elements no bound.
    y5 = 800 .* exp.(-b .* 5e-3)
    r = fit(m, y5)
    @test r.params.ADC == 4e-3 && onbound(r).ADC
    r = fit(m, y5; ADC = Free(upper = Inf))
    @test r.params.ADC ≈ 5e-3 rtol = 1e-6
    @test r.params.S0 ≈ 800 rtol = 1e-6
    @test parameters(ADCKurtosis).ADC.upper == 4e-3
    @test all(parameters(M)[k].upper == 4e-3 for M in (DTI, DKI), k in (:Dxx, :Dyy, :Dzz))
    # ... and from 0: a diffusivity along an axis is not negative (D positive semi-definite)
    @test all(parameters(M)[k].lower == 0 for M in (DTI, DKI), k in (:Dxx, :Dyy, :Dzz))
    @test all(parameters(M)[k].upper === nothing for M in (DTI, DKI), k in (:Dxy, :Dxz, :Dyz))
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
    # T1 ≤ 5 s and R1 ≥ 1/5 s in every T1 model, both forms (builtin.jl; the owner, 2026-11-09:
    # "make T1 cap 5 sec"), so where the data barely recover (a T1 of 50 s) both forms end on the
    # bound, flagged, and agree.
    for (mT, mR, _) in pairs
        @test parameters(mT).T1.upper == 5.0 && parameters(mR).R1.lower == 0.2
    end
    mT, mR = T1SR(TD = td), R1SR(TD = td)
    slow = signal(mT, (M0 = 1000.0, T1 = 50.0))
    rT, rR = fit(mT, slow), fit(mR, slow)
    @test rT.params.T1 == 5.0 && onbound(rT).T1
    @test rR.params.R1 ≈ 0.2 && onbound(rR).R1
end

@testset "Compartments: iso, stick, zeppelin and the powder view, exported by EQUIP" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    # plain functions of EQUIP (no module of their own), diffusion only (no exponential unit)
    @test Set(names(EQUIP)) ⊇ Set([:iso, :stick, :zeppelin, :powder, :fiberdirection, :inner])
    @test !isdefined(EQUIP, :ball)
    @test !isdefined(EQUIP, :Blocks) && !isdefined(EQUIP, :decay) && !isdefined(EQUIP, :_decay)
    @test iso(1000.0, 1e-3) == exp(-1.0) && iso(0.0, 3e-3) == 1 && iso(4000.0, 0.0) == 1    # D = 0: a dot
    @test stick(1000f0, 2f-3, powder) isa Float32 && stick(1000f0, 2f-3, 0.5f0) isa Float32
    @test stick(0.0, 2e-3, powder) == 1 && stick(1000.0, 2e-3, 0.0) == 1 && stick(1000.0, 2e-3, 1.0) ≈ exp(-2)
    @test stick(1000.0, 2e-3, powder) ≈ 0.5981440066613041 rtol = 1e-14                # √π·erf(√2)/(2√2)
    x, w = GL01                                            # Gauss–Legendre on [0, 1]: the mean over ξ
    @test stick(2500.0, 2e-3, powder) ≈ sum(w .* [stick(2500.0, 2e-3, ξ) for ξ in x]) rtol = 1e-12
    @test zeppelin(2500.0, 2e-3, 0.5e-3, powder) ≈ sum(w .* [zeppelin(2500.0, 2e-3, 0.5e-3, ξ) for ξ in x]) rtol = 1e-12
    @test zeppelin(1000.0, 1e-3, 1e-3, powder) ≈ iso(1000.0, 1e-3)
    # D⊥ > D∥ (oblate): defined in both views, the spherical mean the mean of the directional one
    @test zeppelin(1000.0, 0.5e-3, 1e-3, powder) ≈ sum(w .* [zeppelin(1000.0, 0.5e-3, 1e-3, ξ) for ξ in x]) rtol = 1e-12
    @test zeppelin(1000f0, 0.5f-3, 1f-3, powder) isa Float32
    # no view: an error that says which to write
    @test occursin("stick(b, D, powder)", msg(() -> stick(1000.0, 2e-3)))
    @test occursin("zeppelin(b, D∥, D⊥, powder)", msg(() -> zeppelin(1000.0, 2e-3, 0.5e-3)))
    # the SMT models are written with them: each one's signal is Kaden's equation written out
    bs = [0.0, 1000, 2500, 4000]
    sph(x2) = x2 == 0 ? 1.0 : sqrt(π) * SpecialFunctions.erf(sqrt(x2)) / (2 * sqrt(x2))
    θs = (S0 = 1.0, vax = 0.6, Dax = 2.0e-3)
    Dperp = 0.4 * 2.0e-3
    @test signal(SMT(b = bs), θs) ≈ [0.6 * sph(b * 2e-3) + 0.4 * exp(-b * Dperp) * sph(b * 1.2e-3) for b in bs] rtol = 1e-10
    bt, tt = [bs; bs], [fill(0.07, 4); fill(0.12, 4)]
    θt = (; θs..., T2ax = 0.09, T2ex = 0.05)
    smtt2(b, TE, p) = p.S0 * (p.vax * exp(-TE / p.T2ax) * stick(b, p.Dax, powder) +
                              (1 - p.vax) * exp(-TE / p.T2ex) * zeppelin(b, p.Dax, (1 - p.vax) * p.Dax, powder))
    @test smtt2.(bt, tt, Ref(θt)) ≈ signal(SMTT2(b = bt, TE = tt), θt) rtol = 1e-12
    # relaxation is written as its equation: exp(-TE / T2) gives T2Decay's fit, from the same
    # start, to rounding (T2Decay multiplies by the rate 1/T2, this divides by T2)
    te = collect(0.01:0.01:0.32)
    m = T2Decay(TE = te)
    yb2 = simulate(m, (S0 = 1000.0, T2 = 0.08); σ = 5.0, noise = :gaussian, rng = MersenneTwister(21))
    rb, rm = fit(BlockT2(TE = te), yb2), fit(m, yb2; T2 = Free(start = 0.05))
    @test rb.params.T2 ≈ rm.params.T2 rtol = 1e-10
    @test rb.params.S0 ≈ rm.params.S0 rtol = 1e-10
end

@testset "Dispersed views: watson and bingham, between one direction and powder" begin
    @test Set(names(EQUIP)) ⊇ Set([:watson, :bingham, :watsontensor, :binghamtensor])
    # the view averages the signal over the distribution: at κ = 0 every direction equally, the
    # spherical mean (powder), whatever the cosine ξ; the stick, and the zeppelin with D⊥ < D∥
    # and D⊥ > D∥ (a negative excess: the series of exp(+|β|v²)), against powder's closed forms
    # (erf, erfi)
    w0 = watson(0.0)
    for (b, ξ) in ((1000.0, 0.3), (2500.0, 0.9), (6000.0, 0.5))
        @test stick(b, 1.7e-3, w0(ξ)) ≈ stick(b, 1.7e-3, powder) rtol = 1e-8
        @test zeppelin(b, 2e-3, 0.5e-3, w0(ξ)) ≈ zeppelin(b, 2e-3, 0.5e-3, powder) rtol = 1e-8
        @test zeppelin(b, 0.5e-3, 1e-3, w0(ξ)) ≈ zeppelin(b, 0.5e-3, 1e-3, powder) rtol = 1e-8
    end
    # ... and at κ → ∞ one direction: the gap to the cosine view halves as κ doubles (its leading
    # term is 1/κ); above b·D = 12.75, through the quadrature, it closes faster still at these κ
    # (b·D/κ not yet small: ratios 2.8 and 2.6 measured), so there only at least halving
    gap(b, κ) = abs(stick(b, 1.7e-3, watson(κ)(0.6)) - stick(b, 1.7e-3, 0.6))
    @test 1.8 < gap(2500.0, 16.0) / gap(2500.0, 32.0) < 2.2 && 1.8 < gap(2500.0, 32.0) / gap(2500.0, 64.0) < 2.2
    @test gap(9000.0, 16.0) / gap(9000.0, 32.0) > 1.8 && gap(9000.0, 32.0) / gap(9000.0, 64.0) > 1.8
    # the zeppelin's view is its signal averaged, against a quadrature over the distribution (u =
    # μ·n Gauss–Legendre, the azimuth by the trapezoid rule)
    function averaged(f, κ, ξ)
        x, wts = GL01
        num = den = 0.0
        for (u0, wu) in zip(x, wts), u in (u0, -u0), k in 0:199
            ψ = 2π * k / 200
            c = ξ * u + sqrt(1 - ξ^2) * sqrt(1 - u^2) * cos(ψ)      # the gradient's cosine with n
            W = wu * exp(κ * (u^2 - 1))
            num += W * f(c); den += W
        end
        return num / den
    end
    @test zeppelin(2500.0, 2e-3, 0.5e-3, watson(8.0)(0.4)) ≈ averaged(c -> zeppelin(2500.0, 2e-3, 0.5e-3, c), 8.0, 0.4) rtol = 1e-7
    # the stick's kₗ from the table in β (quintic Hermite between the points, h = 0.05), against
    # kₗ made exactly (their largest: 2, 2/3, 0.4), over the series' range, on the points, 1e-9
    # to either side, mid-cell, near 0 and near 12.75: kₗ within 1e-12, its first derivative in
    # β (a dual number through the interpolant) within 1e-10, its second (nested duals, as
    # crlb's) within 1e-8 (3e-14, 5e-12 and 1e-9 relative measured), and the second continuous
    # across a point; outside the table (β < 0, past 12.85, NaN), kₗ made exactly
    kx(β) = EQUIP._stickk(β)
    ki(β) = EQUIP._stickkinterp(β)
    d1(β) = ForwardDiff.derivative(ki, β)
    d2(β) = ForwardDiff.derivative(x -> ForwardDiff.derivative(ki, x), β)
    βs = [1e-8; 1e-4; 0.0123; range(0.031, 12.74; length = 97); [i * 0.05 + s for i in (1, 34, 85, 254) for s in (-1e-9, 0.0, 1e-9)];
          [(i + 0.5) * 0.05 for i in (0, 85, 254)]; 12.75]
    @test maximum(maximum(abs.(ki(β) .- kx(β)[1])) for β in βs) < 1e-12
    @test maximum(maximum(abs.(d1(β) .- kx(β)[2])) for β in βs) < 1e-10
    @test maximum(maximum(abs.(d2(β) .- kx(β)[3])) for β in βs) < 1e-8
    @test maximum(abs.(d2(4.25 - 1e-12) .- d2(4.25 + 1e-12))) < 1e-8
    wk = EQUIP._watsoncoeffs(12.0)
    @test all(EQUIP._shellcoeffs(β, wk) == EQUIP._shellcoeffsexact(β, wk) for β in (-0.3, 12.86, 20.0))
    @test all(isnan, EQUIP._shellcoeffs(NaN, wk))
    @test !hasmethod(setindex!, Tuple{typeof(EQUIP._STICK_TABLE.v),Any,Int})     # read-only
    # watsontensor: the averaged tensor keeps the trace, D∥ₑ + 2D⊥ₑ = D∥ + 2D⊥, isotropic at κ = 0
    Dp, Dq = watsontensor(2e-3, 0.5e-3, watson(6.0))
    @test Dp + 2Dq ≈ 2e-3 + 2 * 0.5e-3 && all(watsontensor(2e-3, 0.5e-3, w0) .≈ (3e-3 / 3, 3e-3 / 3))
    # bingham: at β = 0 the Watson view (μβ any perpendicular), at κ = β = 0 powder; its tensor
    # keeps the trace and at β = 0 is the Watson tensor's zeppelin
    μ, μβ = SVector(0.0, 0.6, 0.8), SVector(1.0, 0.0, 0.0)
    g = normalize(SVector(0.3, -0.5, 0.8))
    Bw = bingham(6.0, 0.0, μ, μβ)
    @test stick(2500.0, 1.7e-3, Bw(g)) ≈ stick(2500.0, 1.7e-3, watson(6.0)(g ⋅ μ)) rtol = 1e-8
    # Watson with its axis takes the gradient direction as Bingham does: the view from the cosine
    # g ⋅ μ, exactly; without an axis, an error naming `watson(κ, μ)`
    @test stick(2500.0, 1.7e-3, watson(6.0, μ)(g)) == stick(2500.0, 1.7e-3, watson(6.0)(g ⋅ μ))
    @test stick(2500.0, 1.7e-3, watson(6.0, μ)(Tuple(g))) == stick(2500.0, 1.7e-3, watson(6.0, μ)(g))
    @test occursin("watson(κ, μ)", sprint(showerror, try watson(6.0)(g) catch e; e end))
    @test zeppelin(2500.0, 2e-3, 0.5e-3, Bw(g)) ≈ zeppelin(2500.0, 2e-3, 0.5e-3, watson(6.0)(g ⋅ μ)) rtol = 1e-8
    @test stick(2500.0, 1.7e-3, bingham(0.0, 0.0, μ, μβ)(g)) ≈ stick(2500.0, 1.7e-3, powder) rtol = 1e-8
    D3 = binghamtensor(2e-3, 0.5e-3, bingham(6.0, 3.0, μ, μβ))
    @test tr(D3) ≈ 2e-3 + 2 * 0.5e-3
    Dp6, Dq6 = watsontensor(2e-3, 0.5e-3, watson(6.0))
    @test binghamtensor(2e-3, 0.5e-3, Bw) ≈ Dq6 * I + (Dp6 - Dq6) * μ * μ' rtol = 1e-8
    # Float32 stays Float32 where the view's own numbers are
    @test stick(1000f0, 1.7f-3, watson(5f0)(0.5f0)) isa Real
end
