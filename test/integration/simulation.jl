# =================================================================================================
# test/integration/simulation.jl: Simulation: simulate, tissue presets, maps, compare; Rician noise (mean, variance,
# information, RicianMean, crlb's noise = :rician, ricianbound).
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl simulation
# =================================================================================================

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

@testset "Rician noise: mean, variance, information; RicianMean, crlb's noise = :rician, ricianbound" begin
    rng = MersenneTwister(31)
    # The mean and variance of a magnitude against simulated ones; the limits
    for a in (0.0, 0.7, 2.0, 6.0)
        y = abs.(a .+ randn(rng, 400_000) .+ im .* randn(rng, 400_000))
        @test ricianmean(a, 1.0) ≈ mean(y) atol = 4 * std(y) / sqrt(400_000)
        @test ricianvar(a, 1.0) ≈ var(y) rtol = 0.01
    end
    @test ricianmean(0.0, 2.0) ≈ 2 * sqrt(π / 2)
    @test ricianvar(0.0, 1.0) ≈ 2 - π / 2
    @test ricianmean(300.0, 1.0) ≈ 300 + 1 / 600 rtol = 1e-10
    @test ricianmean(999.9, 1.0) ≈ 999.9 + 1 / (2 * 999.9) + 1 / (8 * 999.9^3) rtol = 1e-12  # (its switch)
    @test ricianmean(1.0, 1e-6) ≈ 1.0                                        # (Bessel's range)
    @test ricianvar(999.0, 1.0) ≈ ricianvar(1001.0, 1.0) rtol = 1e-6          # across its switch
    @test ForwardDiff.derivative(a -> ricianmean(a, 1.0), 1.3) ≈
          (ricianmean(1.3 + 1e-6, 1.0) - ricianmean(1.3 - 1e-6, 1.0)) / 2e-6 rtol = 1e-6
    # The fast pieces (tier 2 step 1, src/models/rician.jl) against independent references.
    # The scaled Bessel functions (Chebyshev series fitted at precompile time) against
    # SpecialFunctions' besselix (AMOS), x from 0 to 10⁶; the mean against its formula written
    # with besselix; on dual numbers, the closed forms of its derivatives with besselix:
    # dE/dA = √(π/2)·(A/2σ)·(I₀e + I₁e)(x), d²E/dA² = √(π/2)/(2σ)·(I₀e − I₁e)(x), x = A²/4σ², and
    # dE/dσ = (E − A·dE/dA)/σ (E is homogeneous of degree 1 in A and σ: Euler's theorem).
    bix = SpecialFunctions.besselix
    xs = [0.0; 10 .^ range(-8, 6; length = 2001); range(0, 20; length = 2001)]
    @test maximum(x -> abs(EQUIP._i01e(x)[1] / bix(0, x) - 1), xs) < 1e-13
    @test maximum(x -> x == 0 ? 0.0 : abs(EQUIP._i01e(x)[2] / bix(1, x) - 1), xs) < 1e-13
    Eref(A, σ) = (x = A^2 / (4σ^2); σ * sqrt(π / 2) * ((1 + 2x) * bix(0, x) + 2x * bix(1, x)))
    d1(A, σ) = (x = A^2 / (4σ^2); sqrt(π / 2) * (A / (2σ)) * (bix(0, x) + bix(1, x)))
    d2(A, σ) = (x = A^2 / (4σ^2); sqrt(π / 2) / (2σ) * (bix(0, x) - bix(1, x)))
    for A in (0.0, 1e-3, 0.5, 2.0, 20.0, 300.0), σ in (1.0, 2.5)
        @test ricianmean(A, σ) ≈ Eref(A, σ) rtol = 1e-13
        @test ForwardDiff.derivative(a -> ricianmean(a, σ), A) ≈ d1(A, σ) atol = 1e-13
        @test ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> ricianmean(b, σ), a), A) ≈
              d2(A, σ) atol = 1e-13
        @test ForwardDiff.derivative(s -> ricianmean(A, s), σ) ≈ (Eref(A, σ) - A * d1(A, σ)) / σ atol = 1e-12
    end
    @test ricianmean(3.0f0, 1.0f0) isa Float32                     # a Float32 fit stays Float32
    # R(s) from its table, against the quadrature it is made from, between the table's points
    @test maximum(s -> abs(EQUIP._ricianRtable(s) - EQUIP._ricianR(s)), range(0, 40; length = 4999)) < 2e-8
    # The information R(s)/σ²: the variance of the score against Monte Carlo; 0 at A = 0, and
    # its quadrature and its expansion agree at the switch (s = 40)
    for a in (1.0, 3.0)
        y = abs.(a .+ randn(rng, 1_000_000) .+ im .* randn(rng, 1_000_000))
        score = y .* (SpecialFunctions.besselix.(1, y .* a) ./ SpecialFunctions.besselix.(0, y .* a)) .- a
        @test ricianinformation(a, 1.0) ≈ var(score) rtol = 0.01
    end
    @test ricianinformation(0.0, 1.0) == 0 && ricianinformation(2.0, 2.0) ≈ ricianinformation(1.0, 1.0) / 4
    @test EQUIP._ricianR(39.999) ≈ EQUIP._ricianR(40.001) rtol = 1e-7

    # RicianMean: the model's signal through the Rician mean; complex averaging divides σ by
    # √NEX, magnitude averaging leaves it
    pr = SMT_PROTOCOLS.A
    m = SMT(b = pr.b)
    θ = nominal(m, :SMT_cBH)
    σ = θ.S0 / 20
    @test signal(RicianMean(m; σ), θ) ≈ ricianmean.(signal(m, θ), σ)
    @test RicianMean(m; σ, averages = 4).σ == fill(σ / 2, 3)
    @test RicianMean(m; σ, averages = 4, averaging = :magnitude).σ == fill(σ, 3)
    @test parameters(RicianMean(m; σ)) == parameters(m) && nobs(RicianMean(m; σ)) == 3
    # ... and the model's derived quantities (DTI's MD and FA from a fit of its Rician mean)
    mdti = DTI(; diffusionencoding([0.0; fill(1000.0, 6)],
                                   reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), 1); directionset(6)]))...)
    θd = nominal(DTI; S0 = 1000.0, MD = 0.8e-3, FA = 0.6, direction = (1, 0, 0))
    @test derived(RicianMean(mdti; σ = 20.0), θd) == derived(mdti, θd)

    # crlb of the least-squares fit of magnitude data (`noise = :rician`) against montecarlo:
    # black holes, protocol A, SNR 20 per direction, the spherical mean a magnitude average.
    # The floor biases v_ax by about 0.07 (29%, three times its SD); predicted within 10%.
    c = crlb(m, θ; snr = 20, averages = pr.directions, noise = :rician, averaging = :magnitude,
             weights = pr.directions)
    # (the same as the data's mean and the Rician variance given by hand: what `riciannoise`
    # wrote until 2026-11-12, with magnitude averaging's σ of one average and Var/NEX)
    A = signal(m, θ)
    ch = crlb(m, θ; simulate_with = RicianMean(m; σ) => θ,
              Σ = Diagonal(ricianvar.(A, σ) ./ pr.directions), weights = pr.directions)
    @test isapprox(c.bias.vax, ch.bias.vax; rtol = 1e-12) && isapprox(c.sd.vax, ch.sd.vax; rtol = 1e-12)
    mc = montecarlo(m, θ; snr = 20, averages = pr.directions, noise = :rician,
                    averaging = :magnitude, weights = pr.directions, n = 5000, rng)
    @test c.bias.vax ≈ mc.bias.vax rtol = 0.1
    @test c.bias.vax > 2 * c.sd.vax
    @test c.sd.vax ≈ mc.sd.vax rtol = 0.1
    @test c.bias.Dax ≈ mc.bias.Dax rtol = 0.15
    # Fitting the Rician mean itself (σ known) removes the floor's bias
    Y = simulate(m, θ; snr = 20, averages = pr.directions, noise = :rician, averaging = :magnitude,
                 n = 2000, rng)
    rr = fit(RicianMean(m; σ), Y; weights = pr.directions)
    @test abs(mean(rr.params.vax) - θ.vax) < 0.25 * c.sd.vax
    @test mean(fit(m, Y; weights = pr.directions).params.vax) - θ.vax > 2 * c.sd.vax

    # The Rician bound against Devan et al. 2020, Fig. 1: with the Rician information, the
    # v_ax precision efficiency of the five protocols at SNR 20 per direction is their figure
    # within 9% in every tissue (within 3% but protocol D, 5-8%), where the Gaussian one is up
    # to 13% off (black holes, A and B)
    fig1 = (SMT_NWM = (A = 0.90, B = 1.00, C = 1.10, D = 0.865, E = 0.84),
            SMT_NAWM = (A = 1.10, B = 1.06, C = 1.05, D = 0.83, E = 0.82),
            SMT_cBH = (A = 0.90, B = 0.905, C = 0.825, D = 0.685, E = 0.68),
            SMT_T2lesion = (A = 1.11, B = 1.06, C = 0.99, D = 0.79, E = 0.79))
    for name in keys(fig1), P in keys(SMT_PROTOCOLS)
        p = SMT_PROTOCOLS[P]
        ms = SMT(b = p.b)
        t = nominal(ms, name)
        sd = ricianbound(ms, t; snr = 20, averages = p.directions, averaging = :magnitude).sd.vax
        eff = t.vax / (sd * sqrt(sum(p.directions)))
        @test eff ≈ fig1[name][P] rtol = (P === :D ? 0.09 : 0.05)
    end
    # A bound: never below the Gaussian information's SD at the same σ
    @test ricianbound(m, θ; snr = 20).sd.vax >= crlb(m, θ; snr = 20, weights = ones(3)).sd.vax
end
