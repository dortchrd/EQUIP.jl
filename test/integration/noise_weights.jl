# =================================================================================================
# test/integration/noise_weights.jl: The noise model (one σ, one per observation, a full Σ), averages (σ/√NEX, complex or
# magnitude averaging), and weighted fits through fit, statistics, crlb, leakage, montecarlo, design.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl noise_weights
# =================================================================================================

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
    @test_noalloc allocs(plan, tr, full) == 0
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
    @test_noalloc allocs(dv, xv) == 0
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
    # `averaging` is for Rician noise (crlb's `noise = :rician`), where it changes the floor: with
    # Gaussian noise refused, as simulate refuses it; for a set of scenarios too
    @test occursin("`averaging` is for Rician noise", msg(() -> crlb(md, θ; σ = σ1, averages = nex, averaging = :magnitude)))
    @test occursin("`averaging` is for Rician noise", msg(() -> crlb(m4, set; snr = 200, averages = 4, averaging = :complex)))
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
    @test_noalloc @allocated(fit!(cw, y)) == 0
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
    @test_noalloc @allocated(k(c5, t5)) == 0
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
    @test_noalloc @allocated(d(x5)) == 0
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
