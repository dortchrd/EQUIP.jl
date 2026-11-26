# =================================================================================================
# test/integration/montecarlo_sweeps.jl: montecarlo, with crlb alongside; sweeps over noise and true values; scenarios side by
# side and data from another model.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl montecarlo_sweeps
# =================================================================================================

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
    @test repr(mq) == "MonteCarloResult(QMTSIR: PSR, R1f, Sf, M0f, MPF, kfm; n = 4000)"
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
    s = crlb(m4, θ; snr = 200, kmf = 12.5, vary = (kmf = 5.0:2.5:20.0,))
    @test s isa Sweep && length(s) == 7 && eltype(s) <: CRLBResult
    @test s.settings[1] == (kmf = 5.0,) && s.settings[end] == (kmf = 20.0,)
    @test s[3].bias == crlb(m4, (; θ..., kmf = 10.0); snr = 200, kmf = 12.5).bias
    @test s[4].bias.PSR == 0                                      # true kmf = the assumed 12.5
    # each setting as crlb! gives it, without the truth checked again (numbers varied) or
    # with it (a tie varied); cond as crlb's
    @test s[2].sd == crlb!(CRLBCache(m4, θ; kmf = 12.5), (; θ..., kmf = 7.5); snr = 200).sd
    @test cond(s[2]) == cond(crlb(m4, (; θ..., kmf = 7.5); snr = 200, kmf = 12.5))
    @test cond.(s.results)[2] == cond(s[2])
    st = crlb(m4, (; θ..., R1m = 1.0); snr = 200, vary = (R1m = [:R1f, 1.2],))
    @test st[1].bias == crlb(m4, (; θ..., R1m = 1.0); snr = 200, R1m = 1.0, vary = (R1m = [θ.R1f],)).results[1].bias
    @test st[2].sd == crlb!(CRLBCache(m4, (; θ..., R1m = 1.0)), (; θ..., R1m = 1.2); snr = 200).sd
    @test [r.fixed.kmf for r in s] == fill(12.5, 7)               # the fit never changes
    @test length(s.rows) == 7 * 6 && s.rows[1].kmf == 5.0 && s.rows[1].parameter === :PSR    # 4 parameters, MPF, kfm
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
    u2 = crlb(m4, θ; snr = [100, 200], vary = (kmf = [5.0, 10.0], Sm = [0.8, 0.83]))
    @test u2.settings == u.settings && [r.sd for r in u2] == [r.sd for r in u]
    @test occursin("vary = (kmf = 5:2.5:20,)", msg(() -> crlb(m4, θ; snr = 200, vary = 5:10)))
    @test occursin("Got a pair", msg(() -> crlb(m4, θ; snr = 200, vary = :kmf => 5:2.5:20)))   # pairs: not since 2026-11-13
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
                   vary = (kmf = [5.0, 20.0],), rng = MersenneTwister(2))
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
    @test occursin("not a parameter", msg(() -> crlb(m4, θ; snr = 200, vary = (kab = [1.0],))))
    @test occursin("vector or a range", msg(() -> crlb(m4, θ; snr = 200, vary = (kmf = 5.0,))))
    @test occursin("vector or a range", msg(() -> montecarlo(m4, θ; noise = :rician, snr = 200, vary = (kmf = Float64[],))))
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
    @test keys(s[2].fit) == (:PSR, :R1f, :Sf, :M0f, :kmf, :MPF, :kfm) && s.rows[end].fit === :free
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
    w = crlb(mt, ref; snr = 100, simulate_with = mb => θb, vary = (fshort = [0.1, 0.3],))
    @test w[1].fit.T2 > w[2].fit.T2 && w[1].truth == ref && w[2].data.second.fshort == 0.3
    @test occursin("not a parameter of T2BiexpDecay",
                   msg(() -> crlb(mt, ref; snr = 100, simulate_with = mb => θb, vary = (T2 = [0.07],))))
    # mistakes
    @test occursin("the sandwich only", msg(() -> crlb(mt, ref; snr = 100, simulate_with = mb => θb, cov = :biased)))
    @test occursin("observations", msg(() -> crlb(mt, ref; snr = 100, simulate_with = T2BiexpDecay(TE = te[1:10]) => θb)))
    @test occursin("simulate_with = model => truth", msg(() -> crlb(mt, ref; snr = 100, simulate_with = mb)))
    # the kernel with another data model allocates nothing either
    plan = CRLBCache(mt, ref)
    tr, tb = EQUIP._crlb_truth(mt, ref), EQUIP._crlb_truth(mb, θb)
    allocs(p, a, b) = @allocated EQUIP._crlb!(p, mt, a, 100.0, mb, b)
    allocs(plan, tr, tb)
    @test_noalloc allocs(plan, tr, tb) == 0
end
