# =================================================================================================
# test/integration/rician.jl: fits with the Rician noise model, `likelihood = :rician` with
# `method = :ml` (the full Rician model), `:mean` (least squares on the Rician mean) or `:wmean`
# (the same weighted by 1/Var); src/fit/rician.jl. Checked against what is independent of that
# code: the high-SNR limit (the least-squares fit), the fit of `RicianMean` written by hand, a
# Rician likelihood written here with SpecialFunctions' besselix, crlb and ricianbound at the
# fit's values, and invariances (a σ map, refit, a dataset against one voxel).
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl rician
# =================================================================================================

@testset "likelihood = :rician: keywords, settings and errors" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    m = T2Decay(TE = collect(0.01:0.01:0.32))
    y = signal(m, (S0 = 1000.0, T2 = 0.07))
    c = init(m; likelihood = :rician, σ = 10.0)
    @test c isa EQUIP.RicianFitCache && c.method === :ml            # the default: the full model
    r = fit!(c, y)
    @test r.settings.likelihood === :rician && r.settings.method === :ml && r.settings.σ == 10.0
    # the display names the noise model (least squares' display has no such line)
    @test occursin("noise    Rician, σ = 10.0, method ml", sprint(show, MIME"text/plain"(), r))
    @test !occursin("noise", sprint(show, MIME"text/plain"(), fit(m, y)))
    @test occursin("`σ`", msg(() -> fit(m, y; likelihood = :rician)))
    @test occursin(":wmean", msg(() -> fit(m, y; likelihood = :rician, σ = 10.0, method = :lsq)))
    @test occursin("likelihood = :rician", msg(() -> fit(m, y; method = :mean)))
    @test occursin("stderror(res; σ)", msg(() -> fit(m, y; σ = 10.0)))
    @test occursin(":gaussian", msg(() -> fit(m, y; likelihood = :poisson)))
    @test occursin("-1.0", msg(() -> fit(m, y; likelihood = :rician, σ = -1.0)))
    @test occursin("`weights`", msg(() -> fit(m, y; likelihood = :rician, σ = 10.0, weights = ones(32))))
    @test occursin("res.settings.σ", msg(() -> loglikelihood(r)))
    @test occursin("res.settings.σ", msg(() -> stderror(r; σ = 5.0)))
end

@testset "likelihood = :rician: limits, the fit by hand, the likelihood's maximum" begin
    m = T2Decay(TE = collect(0.01:0.01:0.32))
    θ = (S0 = 1000.0, T2 = 0.069)
    # 1. At high SNR (σ = 0.01, the smallest signal 1e3 σ) every method is the least-squares fit:
    #    the floor's correction is σ²/2A, 5e-7 relative here
    y = signal(m, θ)
    for method in (:ml, :mean, :wmean)
        r = fit(m, y; likelihood = :rician, σ = 0.01, method)
        @test isconverged(r) && isapprox(r.params.T2, 0.069; rtol = 1e-5) &&
              isapprox(r.params.S0, 1000; rtol = 1e-5)
    end

    # 2. :mean is the fit of RicianMean written by hand, from the least-squares fit's values;
    #    its iterations both fits', its sse the data against their mean
    σ = 100.0                                                    # SNR 10
    Y = simulate(m, θ; noise = :rician, σ, n = 200, rng = MersenneTwister(2))
    g = fit(m, Y)
    byhand = fit(RicianMean(m; σ), Y; NamedTuple(k => Free(start = g.params[k]) for k in keys(g.params))...)
    rm = fit(m, Y; likelihood = :rician, σ, method = :mean)
    @test rm.params.T2 == byhand.params.T2 && rm.params.S0 == byhand.params.S0
    @test rm.status == byhand.status && rm.iterations == g.iterations .+ byhand.iterations
    @test rm.sse ≈ byhand.sse rtol = 1e-12
    # ... and a voxel alone gives what the dataset gives (fit! with the cache)
    @test fit!(init(m; likelihood = :rician, σ, method = :mean), Y[7, :]).params.T2 == rm.params.T2[7]

    # 3. :ml maximizes the Rician likelihood, written here independently (besselix): its gradient
    #    is 0 at the fit (central differences), and no other method's values do better
    nll(p, yv) = (A = signal(m, p); sum(@. (yv - A)^2 / (2σ^2) - log(SpecialFunctions.besselix(0, yv * A / σ^2))))
    rml = fit(m, Y; likelihood = :rician, σ, method = :ml)
    rw = fit(m, Y; likelihood = :rician, σ, method = :wmean)
    vox = findall(isconverged, rml.status)[1:40]
    pv(r, v) = (S0 = r.params.S0[v], T2 = r.params.T2[v])
    function relgrad(p, yv)                    # d nll / d log θⱼ: the change for a relative step
        map(keys(p)) do k
            h = 1e-6 * p[k]
            (nll(merge(p, (; k => p[k] + h)), yv) - nll(merge(p, (; k => p[k] - h)), yv)) / 2h * p[k]
        end
    end
    @test maximum(v -> maximum(abs, relgrad(pv(rml, v), Y[v, :])), vox) < 1e-3
    @test all(v -> nll(pv(rml, v), Y[v, :]) <= min(nll(pv(rm, v), Y[v, :]), nll(pv(rw, v), Y[v, :]),
                                                   nll(pv(g, v), Y[v, :])) + 1e-9, vox)
    # 4. The floor's bias: least squares +41% at SNR 10 (step 0's baseline), each Rician method
    #    within a few % (2000 voxels: the Monte Carlo SE of the mean is 0.3%)
    Y2 = simulate(m, θ; noise = :rician, σ, n = 2000, rng = MersenneTwister(3))
    bias(r) = mean(r.params.T2) / 0.069 - 1
    @test bias(fit(m, Y2)) > 0.3
    for method in (:ml, :mean, :wmean)
        r = fit(m, Y2; likelihood = :rician, σ, method)
        @test abs(bias(r)) < 0.03 && all(isconverged, r.status)
    end
    # 5. No information (free water at b up to 3000, SNR 10: the signal under the floor from
    #    b = 1500): the fits converge, a quarter of them on the cap (ADC ≤ 4e-3) rather than
    #    running away, and the mean is nearer the truth than least squares' (validation's
    #    rician_fits.jl at 5000 copies: +2.1% against −20.2%, 1225 on the cap)
    ma = ADCDecay(b = collect(0.0:500.0:3000.0))
    Ya = simulate(ma, (S0 = 1000.0, ADC = 3.0e-3); noise = :rician, snr = 10, n = 1000,
                  rng = MersenneTwister(8))
    ra = fit(ma, Ya; likelihood = :rician, σ = 100.0)
    @test all(isconverged, ra.status) && count(onbound(ra).ADC) > 100
    @test abs(mean(ra.params.ADC) / 3.0e-3 - 1) < abs(mean(fit(ma, Ya).params.ADC) / 3.0e-3 - 1)
end

@testset "likelihood = :rician: standard errors, fitted values, maps, refit, precision" begin
    m = T2Decay(TE = collect(0.01:0.01:0.32))
    θ = (S0 = 1000.0, T2 = 0.069)
    σ = 50.0
    y = simulate(m, θ; noise = :rician, σ, rng = MersenneTwister(4))
    # 1. The standard errors, at the fit's values, as crlb predicts that fit (other code: crlb's
    #    sums, R by quadrature): :ml the Rician bound; :mean the sandwich with the Rician
    #    variance; :wmean the same weighted by 1/Var, (JᵀV⁻¹J)⁻¹
    for (method, pred) in (
            :ml => p -> ricianbound(m, p; σ),
            :mean => p -> crlb(RicianMean(m; σ), p; Σ = Diagonal(ricianvar.(signal(m, p), σ))),
            :wmean => p -> (v = ricianvar.(signal(m, p), σ);
                            crlb(RicianMean(m; σ), p; Σ = Diagonal(v), weights = 1 ./ v)))
        r = fit(m, y; likelihood = :rician, σ, method)
        c = pred(r.params)
        se = stderror(r)
        @test isapprox(se.T2, c.sd.T2; rtol = 1e-5) && isapprox(se.S0, c.sd.S0; rtol = 1e-5)
    end
    # 2. The fitted values are the data's mean under the fit, E|y|
    r = fit(m, y; likelihood = :rician, σ)
    @test fitted(r) ≈ ricianmean.(signal(m, r.params), σ) && residuals(r, y) ≈ y .- fitted(r)

    # 3. A σ map: each voxel fitted with its own σ, as alone with that number; a voxel with no σ
    #    (NaN, 0: a map's background) or negative data (not magnitudes) not fitted
    Y = vcat(simulate(m, θ; noise = :rician, σ = 50.0, n = 30, rng = MersenneTwister(5)),
             simulate(m, θ; noise = :rician, σ = 100.0, n = 30, rng = MersenneTwister(6)))
    σmap = [fill(50.0, 30); fill(100.0, 30)]
    rmap = fit(m, Y; likelihood = :rician, σ = σmap)
    @test rmap.params.T2[1:30] == fit(m, Y[1:30, :]; likelihood = :rician, σ = 50.0).params.T2
    @test rmap.params.T2[31:60] == fit(m, Y[31:60, :]; likelihood = :rician, σ = 100.0).params.T2
    bad = copy(σmap); bad[2] = NaN; bad[3] = 0.0
    Yneg = copy(Y); Yneg[4, 1] = -1.0
    rb = fit(m, Yneg; likelihood = :rician, σ = bad)
    @test rb.status[2:4] == fill(EQUIP.InvalidData, 3) && all(isconverged, rb.status[5:end])
    @test derived(fit(m, Y; likelihood = :rician, σ = σmap)) === nothing   # (T2Decay derives none)

    # 4. refit keeps the noise model and lands where the fit did
    r2 = refit(rmap, Y)
    @test r2.settings.likelihood === :rician && r2.settings.σ === σmap
    @test r2.params.T2 ≈ rmap.params.T2 rtol = 1e-6

    # 5. Float32: the same fits, to Float32's tolerances
    r32 = fit(m, Y; likelihood = :rician, σ = σmap, precision = Float32)
    @test eltype(r32.params.T2) == Float32 && median(abs.(r32.params.T2 ./ rmap.params.T2 .- 1)) < 1e-3

    # 6. Allocation-free, every method
    for method in (:ml, :mean, :wmean)
        c = init(m; likelihood = :rician, σ, method)
        fit!(c, y)
        @test_noalloc fitallocs(c, y) == 0
    end
end

@testset "likelihood = :rician: a directional model and its derived quantities (DTI)" begin
    # The tensor's MD from the full Rician fit in white matter at SNR 10, where least squares
    # reads the floor: step 0 measured −8.6% for it, +0.7% for least squares on the Rician mean
    acq = diffusionencoding([fill(0.0, 4); fill(1000.0, 32); fill(2500.0, 32)],
                            reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), 4); directionset(32); directionset(32)]))
    m = DTI(; acq...)
    θ = nominal(DTI; S0 = 1000.0, MD = 0.66e-3, FA = 0.51, direction = fiberdirection(1.0, 0.5))
    Y = simulate(m, θ; noise = :rician, snr = 10, n = 400, rng = MersenneTwister(7))
    rg = fit(m, Y)
    rml = fit(m, Y; likelihood = :rician, σ = 100.0)
    @test mean(derived(rg).MD) / 0.66e-3 - 1 < -0.05
    @test abs(mean(derived(rml).MD) / 0.66e-3 - 1) < 0.02 && all(isconverged, rml.status)
end

@testset "likelihood = :rician in crlb, montecarlo and sweeps" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    m = T2Decay(TE = collect(0.01:0.01:0.32))
    θ = (S0 = 1000.0, T2 = 0.069)
    σ = 100.0                                                    # S0/σ = 10
    # 1. crlb predicts each fit as other code computes it: :ml the Rician bound (`ricianbound`,
    #    crlb with the information as weights); :mean crlb of RicianMean with the Rician
    #    variance; :wmean the same weighted by 1/Var. Nothing fixed wrong: no bias.
    v = ricianvar.(signal(m, θ), σ)
    refs = (ml = ricianbound(m, θ; σ), mean = crlb(RicianMean(m; σ), θ; Σ = Diagonal(v)),
            wmean = crlb(RicianMean(m; σ), θ; Σ = Diagonal(v), weights = 1 ./ v))
    for method in (:ml, :mean, :wmean)
        c = crlb(m, θ; σ, likelihood = :rician, method)
        @test c.rician === method && c.bias.T2 == 0
        @test isapprox(c.sd.T2, refs[method].sd.T2; rtol = 1e-10) &&
              isapprox(c.sd.S0, refs[method].sd.S0; rtol = 1e-10)
    end
    @test crlb(m, Scenario(m, θ; likelihood = :rician); σ).sd == crlb(m, θ; σ, likelihood = :rician).sd
    # The least-squares fit of the same magnitude data, `noise = :rician`: the floor's bias (the
    # noise-free fit of the data's mean, the Rician mean) and the sandwich with the Rician
    # variance, as given by hand; +40% for T2 at SNR 10 (step 0 measured +41%)
    cg = crlb(m, θ; σ, noise = :rician)
    ch = crlb(m, θ; simulate_with = RicianMean(m; σ) => θ, Σ = Diagonal(v))
    @test isapprox(cg.bias.T2, ch.bias.T2; rtol = 1e-12) && isapprox(cg.sd.T2, ch.sd.T2; rtol = 1e-12)
    @test 0.35 < cg.relbias.T2 < 0.45 && cg.noise === :rician
    @test crlb(m, θ; σ).relbias.T2 == 0                          # Gaussian noise: no floor
    # 2. A sweep over noise levels: each setting as the single call
    sw = crlb(m, θ; snr = [10, 20], likelihood = :rician)
    @test sw.results[2].sd.T2 ≈ crlb(m, θ; snr = 20, likelihood = :rician).sd.T2
    # 3. montecarlo fits with the scenario's Rician fit, and its crlb is that fit's, within 3
    #    Monte Carlo standard errors: T2Decay at SNR 20 (least squares +10% biased there, step 0);
    #    and QMTSIR with kmf fixed wrong (12.5 in white matter) at SNR 200, Dortch et al. 2018's
    #    case: the wrong assumption's bias, every method
    z(mc, p) = (abs(mc.bias[p] - mc.crlb.bias[p]) / mc.se_mean[p], abs(mc.sd[p] - mc.crlb.sd[p]) / mc.se_sd[p])
    mc = montecarlo(m, θ; snr = 20, noise = :rician, likelihood = :rician, n = 4000, rng = MersenneTwister(6))
    @test mc.result.settings.likelihood === :rician && mc.crlb.rician === :ml
    @test all(<(3), z(mc, :T2))
    mq = QMTSIR(; SIR_5POINT...)
    θq = nominal(mq, :SIR_WM_I)
    for method in (:ml, :mean, :wmean)
        mcq = montecarlo(mq, θq; snr = 200, noise = :rician, likelihood = :rician, method, kmf = 12.5,
                         n = 4000, rng = MersenneTwister(3))
        @test mcq.crlb.bias.PSR < -0.05 && all(<(3), z(mcq, :PSR))
    end
    # 4. The second-order bias (`secondorder = true`). :ml is Cox and Snell's (J R Stat Soc B 1968),
    #    written here from its closed form with the model right, every piece from the signal or
    #    the moments: b = I⁻¹ Σᵢ Jᵢ [(Cov(vᵢ,uᵢ) + ½E[wᵢ])·JᵢᵀI⁻¹Jᵢ − ½(Rᵢ/σ²)·tr(I⁻¹∇²Aᵢ)]
    FD = EQUIP.ForwardDiff
    c2 = crlb(m, θ; σ, likelihood = :rician, secondorder = true)
    A(x) = signal(m, (S0 = x[1], T2 = x[2]))
    x0 = [θ.S0, θ.T2]
    J = FD.jacobian(A, x0)
    Hs = reshape(FD.jacobian(x -> vec(FD.jacobian(A, x)), x0), 32, 2, 2)
    mo = [EQUIP._ricianmoments(a / σ, a / σ) for a in A(x0)]
    R = [EQUIP._ricianR(a / σ) for a in A(x0)]
    Ii = inv(sum(R[i] / σ^2 * J[i, :] * J[i, :]' for i in 1:32))
    bcs = Ii * sum(J[i, :] * ((mo[i].vu + mo[i].w / 2) / σ^3 * dot(J[i, :], Ii * J[i, :]) -
                              R[i] / σ^2 / 2 * tr(Ii * Hs[i, :, :])) for i in 1:32)
    @test collect(c2.noisebias) ≈ bcs rtol = 1e-7
    @test c2.bias == c2.noisebias && c2.noisebias.T2 > 0          # nothing wrong: T2 +0.78% at SNR 10
    #    ... and it is the maximum-likelihood fit's bias: montecarlo at 20,000 copies (under a
    #    second) within 3 standard errors of it, and more than 5 from the first order's 0
    #    (measured +0.83 ± 0.10% against +0.78%: z 0.5, and 8.3 from 0)
    mc2 = montecarlo(m, θ; snr = 10, noise = :rician, likelihood = :rician, secondorder = true,
                     n = 20_000, rng = MersenneTwister(11))
    @test abs(mc2.bias.T2 - mc2.crlb.bias.T2) < 3 * mc2.se_mean.T2 && mc2.bias.T2 > 5 * mc2.se_mean.T2
    #    Its general form (with a fixed value wrong too) with Gaussian moments is least squares'
    #    second-order bias, written independently (Box's, `_noisebias`): T1MFA, B1 right and 5%
    #    wrong (the residual's third-derivative term), equal to rounding
    gauss(a, as) = (u = as - a, v = -1.0, w = 0.0, uu = 1.0, vu = 0.0)
    mv = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
    θv = (M0 = 1000.0, T1 = 1.0, B1 = 1.0)
    for b1 in (1.0, 1.05)
        cc = CRLBCache(mv, Scenario(mv, θv; B1 = b1))
        k = EQUIP._crlb!(cc, cc.mT, EQUIP._crlb_truth(mv, θv), 100.0)
        @test k.exact == (b1 == 1.0)
        @test EQUIP._mlnoisebias(cc, k, fill(10.0, 6), gauss) ≈ EQUIP._noisebias(cc, k, Diagonal(fill(100.0, 6))) rtol = 1e-12
    end
    #    At high SNR the Rician likelihood is Gaussian, and the bias least squares' (T1MFA at
    #    SNR 1000, its smallest signal 33σ); :mean is least squares on the Rician mean with the
    #    Rician variance, so Box's on RicianMean, as given by hand
    @test crlb(mv, θv; snr = 1000, likelihood = :rician, secondorder = true).noisebias.T1 ≈
          crlb(mv, θv; snr = 1000, secondorder = true).noisebias.T1 rtol = 0.01
    @test crlb(m, θ; σ, likelihood = :rician, method = :mean, secondorder = true).noisebias.T2 ≈
          crlb(RicianMean(m; σ), θ; Σ = Diagonal(v), secondorder = true).noisebias.T2 rtol = 1e-10
    # 5. Not yet, and why
    @test occursin(":wmean", msg(() -> crlb(m, θ; σ, likelihood = :rician, method = :wmean, secondorder = true)))
    @test occursin("noise = :rician", msg(() -> montecarlo(m, θ; snr = 20, noise = :gaussian,
                                                            likelihood = :rician, n = 10)))
    @test occursin("sandwich", msg(() -> crlb(m, θ; snr = 20, likelihood = :rician, cov = :biased)))
    @test occursin("leakage", msg(() -> leakage(mq, θq; kmf = 12.5, likelihood = :rician)))
    md = QMTSIR(; SIR_4POINT..., magnitude = false)
    @test occursin("protocol design", msg(() -> ProtocolDesign(md,
        [Scenario(QMTSIR, θq; kmf = 12.5, likelihood = :rician)]; snr = 200,
        acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)), scantime = p -> sum(p.TI .+ p.TD))))
    @test occursin("`method`", msg(() -> Scenario(m, θ; method = :mean)))
    @test occursin(":rician", msg(() -> crlb(m, θ; σ, likelihood = :rician, noise = :gaussian)))
    @test occursin("`averages`", msg(() -> crlb(m, θ; σ, noise = :rician, averaging = :complex)))
end
