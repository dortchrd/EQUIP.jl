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
    Sm = saturationefficiency(blk, 15e-6)
    @test Sm ≈ 0.83 atol = 0.005
    @test saturationefficiency(cmp, 15e-6) ≈ Sm^2 rtol = 1e-12
    @test log(saturationefficiency(blk, 15e-6; B1 = 1.1)) ≈ 1.21 * log(Sm) rtol = 1e-12
    @test saturationefficiency(blk, 12e-6; Δ = 2e3, lineshape = superlorentzian) ≈ exp(-π * superlorentzian(12e-6, 2e3) * π^2 / 1e-3)
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
    @test ForwardDiff.derivative(t -> saturationefficiency(blk, t; Δ = 2e3, lineshape = superlorentzian), 12e-6) ≈
          (saturationefficiency(blk, 12e-6 + 1e-11; Δ = 2e3, lineshape = superlorentzian) -
           saturationefficiency(blk, 12e-6 - 1e-11; Δ = 2e3, lineshape = superlorentzian)) / 2e-11 rtol = 1e-5
    inversionefficiency(cmp; B1 = 0.9); saturationefficiency(cmp, 15e-6)
    @test_noalloc @allocated(inversionefficiency(cmp; B1 = 0.9)) == 0
    @test_noalloc @allocated(superlorentzian(12e-6, 2e3)) == 0
    # a pulse's segments: one flip, phase and duration each; durations > 0; a number split evenly
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("phase (1)", msg(() -> RFPulse([90, 90], [0], [1e-3, 1e-3])))
    @test occursin("-0.001", msg(() -> blockpulse(180, -1e-3)))
    @test RFPulse([90, 180, 90], [0, 90, 0], 3e-3).duration == fill(1e-3, 3)
    # SIR with its inversion pulse: Sm not a parameter but made from it, T2m, B1 and B0 in its
    # place; the same signal as Sm given that value, in both SIR models; without the pulse, the
    # models (and the type's parameters) as before; tissue presets still apply (their Sm unused)
    for M in (QMTSIR, QMTSIREqualR1)
        mp, m0 = M(TI = TI5, TD = TD5, pulse = blk), M(TI = TI5, TD = TD5)
        @test keys(parameters(mp)) == Tuple(Iterators.flatten(n === :Sm ? (:T2m, :B1, :B0) : (n,) for n in keys(parameters(m0))))
        θp = merge(nominal(mp, :SIR_WM_I), (B1 = 0.9, B0 = 60.0, T2m = 12e-6))
        θ0 = merge(nominal(m0, :SIR_WM_I), (Sm = saturationefficiency(blk, 12e-6; B1 = 0.9, Δ = 60.0),))
        @test signal(mp, θp) == signal(m0, θ0)
        @test :Sm in keys(parameters(M)) && !(:B1 in keys(parameters(M)))
    end
    # its errors by name: PSR's slope on B1 is its slope on Sm times dSm/dB1 (the chain rule; the
    # slope on Sm from the model without the pulse, dSm/dB1 by central differences)
    mp, m0 = QMTSIR(; SIR_5POINT..., pulse = blk), QMTSIR(; SIR_5POINT...)
    θp = nominal(mp, :SIR_WM_I)
    θ0 = merge(nominal(m0, :SIR_WM_I), (Sm = saturationefficiency(blk, 15e-6),))
    dSm = (saturationefficiency(blk, 15e-6; B1 = 1 + 1e-6) - saturationefficiency(blk, 15e-6; B1 = 1 - 1e-6)) / 2e-6
    @test leakage(mp, θp).absolute.B1.PSR ≈ leakage(m0, θ0).absolute.Sm.PSR * dSm rtol = 1e-6
    @test crlb(mp, θp; snr = 200, uncertain = (B1 = 0.05,)).sd.PSR > crlb(mp, θp; snr = 200).sd.PSR
    @test occursin("RFPulse(1 segment, 1.0 ms)", sprint(show, MIME"text/plain"(), mp))
    @test EQUIP._jsonvalue(blk) == Dict("flip" => [180.0], "phase" => [0.0], "duration" => [1e-3])
end

@testset "QMTSPGRYarnykh: the lab's MATLAB, brute force, limits, the two pulse forms, B1 and B0" begin
    m = QMTSPGRYarnykh(; SPGR_ACQ...)
    θ = (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1.0, R1f = 1.2, R1m = 1.0, R1obs = 1.0, B1 = 1.0, B0 = 0.0)
    # the lab's yarnykh_pulseMT.m (A. Smith, 2013), run in GNU Octave 9 (2026-11-26) with the same
    # values (kab = kfm = 3, the excitation 10° in radians, as its code takes it; B1e as a column:
    # given as a row, its `ga.*B1e.^2` expands to a matrix and every offset takes B1e(1)), the
    # Gaussian lineshape for the macromolecular pool (the MATLAB's super-Lorentzian is a rectangle
    # rule, 1e-4 off); its γ is 42.58 MHz/T, so B1e scaled to give the same ω1
    matlab = [0.46211402574358929, 0.50211831922905914, 0.51687069041057643, 0.53534459786136468, 0.59401812191124237,
              0.70943232855283989, 0.27361348865082347, 0.34007194275486763, 0.36441653179065386, 0.38202376858950554,
              0.43851954163236384, 0.65700597281971629]
    mg = QMTSPGRYarnykh(; SPGR_ACQ..., B1e = SPGR_ACQ.B1e .* (42.58 / 42.577478518), lineshape = gaussian)
    @test signal(mg, θ)[1:12] ./ sind(10) ≈ matlab rtol = 1e-12
    # against the sequence played TR after TR to its steady state (test/setup.jl), the
    # super-Lorentzian, every image, the MT-off one too
    y = signal(m, θ)
    bf = [spgr_bruteforce(; offset = m.offset[j], B1e = m.B1e[j], tm = 0.012, ts = 0.002, TR = 0.030, FA = 10.0,
                          lineshape = superlorentzian, θ...) for j in 1:13]
    @test y ≈ sind(10) .* bf rtol = 1e-12
    # ... and where TR and ts change between images (the free recoveries, made once and reused
    # while the timings are the same, made again where they change)
    mv = QMTSPGRYarnykh(; offset = [1e3, 4e3, 4e3, 16e3, 0.0], B1e = [4.0, 4.0, 8.0, 8.0, 0.0], tm = 0.012,
                        ts = [0.002, 0.002, 0.004, 0.004, 0.002], TR = [0.030, 0.050, 0.050, 0.030, 0.030], FA = 10)
    @test signal(mv, θ) ≈ sind(10) .* [spgr_bruteforce(; offset = mv.offset[j], B1e = mv.B1e[j], tm = 0.012, ts = mv.ts[j],
                                       TR = mv.TR[j], FA = 10.0, lineshape = superlorentzian, θ...) for j in 1:5] rtol = 1e-12
    # no macromolecular pool and no MT pulse: the Ernst equation, sin α·(1 − E1)/(1 − cos α·E1)
    E1 = exp(-1.2 * 0.030)
    @test signal(m, merge(θ, (PSR = 0.0,)))[13] ≈ sind(10) * (1 - E1) / (1 - cosd(10) * E1) rtol = 1e-13
    # the MT pulse as its shape and flip angle: a block of flip angle θ over tm has B1e = (θ/tm)/γ
    θmt = rad2deg(4.0 * 2π * 42.577478518 * 0.012)                    # the flip angle that gives B1e = 4 µT
    mp = QMTSPGRYarnykh(; offset = SPGR_ACQ.offset[1:6], mtpulse = blockpulse(θmt, 0.012), mtflip = θmt, ts = 0.002, TR = 0.030, FA = 10)
    @test mp.B1e ≈ fill(4.0, 6) && mp.tm == fill(0.012, 6) && signal(mp, θ) ≈ y[1:6]
    # ... and a shape: its B1e the root mean square of its amplitude, scaled by mtflip/its own net flip
    ps = RFPulse([10, 20, 30, 20, 10], [0, 0, 0, 0, 0], 0.010)
    ms = QMTSPGRYarnykh(; offset = [2e3], mtpulse = ps, mtflip = 450, ts = 0.002, TR = 0.030, FA = 10)
    @test ms.B1e[1] ≈ (450 / 90) * sqrt(sum(abs2, deg2rad.([10, 20, 30, 20, 10]) ./ 0.002) / 5) / (2π * 42.577478518)
    # B1 scales the excitation and the MT pulse; B0 moves every offset
    m11 = QMTSPGRYarnykh(; SPGR_ACQ..., B1e = 1.1 .* SPGR_ACQ.B1e, FA = 11)
    @test signal(m, merge(θ, (B1 = 1.1,))) ≈ signal(m11, θ) rtol = 1e-13
    mb0 = QMTSPGRYarnykh(; SPGR_ACQ..., offset = SPGR_ACQ.offset .- 150)
    @test signal(m, merge(θ, (B0 = 150.0,))) ≈ signal(mb0, θ) rtol = 1e-13
    # R1f tied to R1obs by default: the two pools' slower recovery rate is R1obs
    q = resolve(m, merge(nominal(m), (PSR = 0.15, kmf = 20.0, R1obs = 1.1)))
    kfm = q.kmf * q.PSR
    @test minimum(eigvals([q.R1f + kfm -q.kmf; -kfm q.R1m + q.kmf])) ≈ 1.1 rtol = 1e-13
    # the closed-form 2×2 exponential against LinearAlgebra's: real eigenvalues, complex ones,
    # nearly equal ones (the series), and its derivative (a dual number) against central differences
    for X in ([-3.0 2.0; 0.5 -40.0], [-1.0 -2.0; 3.0 -0.5], [-2.0 1e-5; 1e-5 -2.0 + 1e-6], [-25.0 0.0; 0.0 -25.0])
        @test EQUIP._expm2(SMatrix{2,2}(X)) ≈ exp(X) rtol = 1e-13
    end
    f(t) = EQUIP._expm2(SMatrix{2,2}([-3.0 2.0; 0.5 -40.0] * t))[1, 2]
    @test ForwardDiff.derivative(f, 0.03) ≈ (f(0.03 + 1e-7) - f(0.03 - 1e-7)) / 2e-7 rtol = 1e-7
    # the fit's derivatives against central differences (the super-Lorentzian's table among them)
    J = jacobian(m, θ)
    for (k, name) in enumerate((:PSR, :kmf, :T2f, :T2m, :M0f))
        h = 1e-6 * θ[name]
        up, dn = signal(m, merge(θ, NamedTuple{(name,)}((θ[name] + h,)))), signal(m, merge(θ, NamedTuple{(name,)}((θ[name] - h,))))
        @test J[:, k] ≈ (up .- dn) ./ 2h rtol = 1e-6
    end
    out = zeros(13); qθ = resolve(m, θ); signal!(out, m, qθ)
    @test_noalloc @allocated(signal!(out, m, qθ)) == 0
    # the acquisition's checks: one MT pulse form; one value or one per image; TR ≥ tm + ts
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    base = (offset = [1e3, 2e3], ts = 0.002, TR = 0.030, FA = 10)
    @test occursin("B1e", msg(() -> QMTSPGRYarnykh(; base...)))
    @test occursin("not both", msg(() -> QMTSPGRYarnykh(; base..., B1e = 4.0, tm = 0.012, mtpulse = blockpulse(500, 0.012), mtflip = 500)))
    @test occursin("mtflip", msg(() -> QMTSPGRYarnykh(; base..., mtpulse = blockpulse(500, 0.012))))
    @test occursin("B1e has 3", msg(() -> QMTSPGRYarnykh(; base..., B1e = [4.0, 4.0, 4.0], tm = 0.012)))
    @test occursin("TR = 0.01", msg(() -> QMTSPGRYarnykh(; base..., TR = 0.010, B1e = 4.0, tm = 0.012)))
end

@testset "QMTCWHenkelman and QMTSPGRRamani: Henkelman's closed form, the Bloch steady state, the CWPE" begin
    θ = (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1.0, R1f = 1.2, R1m = 1.0, R1obs = 1.0, B1 = 0.95, B0 = 30.0)
    off, b1 = [0.5e3, 1e3, 2e3, 4e3, 8e3, 16e3, 0.0], [fill(2.0, 6); 0.0]
    mc = QMTCWHenkelman(offset = off, B1cw = b1)
    # The model is Henkelman et al. 1993's closed form; against the two-pool steady state as the
    # linear system it solves, (W − A)·Mss = R1·M0, solved by LinearAlgebra (kfm = kmf·PSR; the
    # irradiation scaled by B1, the offsets moved by B0)
    function steadystate(o, b)
        b == 0 && return 1.0
        ω1² = (2π * 42.577478518 * 0.95 * b)^2
        Δ = o - 30.0
        Wf, Wm = π * ω1² * lorentzian(0.04, Δ), π * ω1² * superlorentzian(12e-6, Δ)
        A = [-(1.2 + 20.0 * 0.15) 20.0; 20.0 * 0.15 -(1.0 + 20.0)]
        return ((Diagonal([Wf, Wm]) - A) \ [1.2, 1.0 * 0.15])[1]
    end
    @test signal(mc, θ) ≈ steadystate.(off, b1) rtol = 1e-12
    # MTR predicted, against the image without irradiation (Mzf/M0f = 1): 1 − Mzf, from the same
    # independent steady state
    @test mtr(mc, θ).mtr ≈ 1 .- steadystate.(off[1:6], b1[1:6]) rtol = 1e-12
    # the free pool alone (PSR = 0): the Bloch equations' steady state under CW irradiation,
    # Mz/M0 = (1 + (ΔωT2)²)/(1 + (ΔωT2)² + ω1²·T1·T2)
    ω1², Δω = (2π * 42.577478518 * 0.95 * 2.0)^2, 2π * (1e3 - 30.0)
    @test signal(mc, merge(θ, (PSR = 0.0,)))[2] ≈ (1 + (Δω * 0.04)^2) / (1 + (Δω * 0.04)^2 + ω1² * 0.04 / 1.2) rtol = 1e-12
    # Ramani's: Henkelman's steady state at the continuous-wave power equivalent, ω1² = (tm/TR)·ω1e²
    # (the MT pulse's power over TR), its acquisition QMTSPGRYarnykh's; ts and FA not used
    mr = QMTSPGRRamani(; SPGR_ACQ...)
    s = sqrt(SPGR_ACQ.tm / SPGR_ACQ.TR)
    @test signal(mr, θ) ≈ signal(QMTCWHenkelman(offset = SPGR_ACQ.offset, B1cw = s .* SPGR_ACQ.B1e), θ) rtol = 1e-14
    @test signal(QMTSPGRRamani(; SPGR_ACQ..., FA = 30, ts = 0.004), θ) == signal(mr, θ)
    # the family's parameters: Henkelman's, which the pulsed models take
    @test parameters(QMTSPGRYarnykh) == parameters(QMTSPGRRamani) == parameters(QMTCWHenkelman)
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("QMTSPGRRamani: TR", msg(() -> QMTSPGRRamani(; SPGR_ACQ..., TR = 0.010)))
    # an MT pulse's shape with no net flip angle (a 1-1̄ binomial; in floating point its sum is
    # 1e-14°, not 0, which had passed and given a B1e 10¹⁶ too large) is refused, the message
    # naming the model it was given to
    @test occursin("QMTSPGRRamani: the MT pulse", msg(() -> QMTSPGRRamani(offset = [1e3, 0], ts = 0.002, TR = 0.030,
        FA = 10, mtpulse = RFPulse([90, 90], [0, 180], 1e-3), mtflip = [600, 0])))
    @test occursin("B1cw has 2", msg(() -> QMTCWHenkelman(offset = [1e3, 2e3, 0.0], B1cw = [1.0, 2.0])))
end

@testset "mtr: MTR from an MT image and its reference" begin
    # MTR = (S_ref − S)/S_ref, the definition written the other way round
    @test mtr(700.0, 1000.0) ≈ (1000.0 - 700.0) / 1000.0
    @test mtr(1000.0, 1000.0) == 0 && mtr(0.0, 1000.0) == 1
    @test mtr(1100.0, 1000.0) ≈ -0.1                  # noise can make S > S_ref: kept, not clamped
    # no answer, NaN: the reference 0 (outside the field of view), negative or not finite, or S not finite
    @test all(isnan, (mtr(500.0, 0.0), mtr(500.0, -3.0), mtr(500.0, Inf), mtr(500.0, NaN), mtr(Inf, 1000.0), mtr(NaN, 1.0)))
    # a scanner's integers give floating point; Float32 stays Float32
    @test mtr(UInt16(700), UInt16(1000)) isa Float64 && mtr(UInt16(700), UInt16(1000)) ≈ 0.3
    @test mtr(0.7f0, 1.0f0) isa Float32 && eltype(mtr(Float32[0.7, 0.5], Float32[1, 1])) == Float32
    # a stack of MT images against one reference: one map per image, each the two-image MTR
    ref = [1000.0 800.0; 0.0 900.0]                    # one voxel outside the field of view
    stack = cat([700.0 600.0; 5.0 450.0], [900.0 760.0; 5.0 855.0]; dims = 3)
    M = mtr(stack, ref)
    @test size(M) == (2, 2, 2)
    @test isequal(M[:, :, 2], mtr(stack[:, :, 2], ref)) && M[1, 2, 1] ≈ 0.25 && isnan(M[2, 1, 1])   # isequal: NaN equals NaN
    # the mask: zero or false outside, as fit's, the size of one image of the stack
    mask = [1 1; 1 0]
    Mm = mtr(stack, ref; mask)
    @test all(isnan, Mm[2, 2, :]) && Mm[1, :, :] == M[1, :, :]
    @test_throws DimensionMismatch mtr(stack, ref; mask = trues(3, 2))
end

@testset "mtr from a qMT acquisition: which images are references, and the pairing" begin
    # two MT powers, each with a reference at 100 kHz (at 4 µT also one at −100 kHz), an image with
    # the MT pulse off, and a third power with no reference of its own
    off = [2e3, 8e3, 100e3, 2e3, 8e3, 100e3, 0.0, 4e3, -100e3]
    m = QMTSPGRYarnykh(offset = off, B1e = [4, 4, 4, 8, 8, 8, 0, 6, 4], tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
    θ = (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1.0, R1f = 1.2, R1m = 1.0, R1obs = 1.0, B1 = 1.0, B0 = 0.0)
    S = signal(m, θ)
    # by default only the MT-off image (7) is a reference: every other image against it
    p = mtr(m, θ)
    @test p.image == [1, 2, 3, 4, 5, 6, 8, 9] && all(==([7]), p.reference)
    @test p.mtr ≈ 1 .- S[p.image] ./ S[7] && p.offset == off[p.image] && p.B1e == m.B1e[p.image]
    # :maxoffset: at each power its largest |offset| (±100 kHz at 4 µT, averaged); the 6 µT image,
    # its power's only one, is not made its own reference (it was, before the rule required images
    # nearer resonance too) and falls back to the MT-off image
    p = mtr(m, θ; reference = :maxoffset)
    @test p.image == [1, 2, 4, 5, 8] && p.reference == [[3, 9], [3, 9], [6], [6], [7]]
    @test p.mtr[1] ≈ 1 - S[1] / ((S[3] + S[9]) / 2) && p.mtr[3] ≈ 1 - S[4] / S[6]
    @test mtr(m, θ; reference = [3, 6, 9]) == p == mtr(m, θ; reference = abs.(off) .== 100e3)
    # from data: one map per MT image, the same as predicted voxel by voxel (each voxel a tissue)
    psr = [0.05 0.1 0.15; 0.2 0.25 0.3]
    data = zeros(2, 3, 9)
    for i in 1:2, j in 1:3
        data[i, j, :] = signal(m, merge(θ, (PSR = psr[i, j],)))
    end
    M = mtr(m, data; reference = :maxoffset)
    @test size(M.mtr) == (2, 3, 5) && M.reference == p.reference
    @test M.mtr[2, 3, :] ≈ mtr(m, merge(θ, (PSR = 0.3,)); reference = :maxoffset).mtr
    @test eltype(mtr(m, Float32.(data)).mtr) == Float32
    @test all(isnan, mtr(m, data; mask = [1 1 1; 1 1 0]).mtr[2, 3, :])
    @test mtr(m, S).mtr ≈ mtr(m, θ).mtr                      # one voxel: a vector
    # the error paths, each naming what it is about
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    nooff = QMTSPGRYarnykh(offset = [2e3, 100e3], B1e = 4, tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
    @test occursin("reference = :maxoffset", msg(() -> mtr(nooff, θ)))        # no reference: how to name one
    @test occursin(":max`", msg(() -> mtr(m, θ; reference = :max)))
    @test occursin("reference image 12", msg(() -> mtr(m, θ; reference = [12])))
    other = QMTSPGRYarnykh(offset = [2e3, 0.0], B1e = [4, 0], tm = 0.012, ts = 0.002, TR = 0.030, FA = [10, 5])
    @test occursin("image 1 ", msg(() -> mtr(other, θ)))                     # the MT-off image at another FA
    @test_throws DimensionMismatch mtr(m, data[:, :, 1:8])
    @test_throws DimensionMismatch mtr(m, θ; reference = trues(4))
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
