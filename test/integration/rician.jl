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
    # 4. Not yet, and why
    @test occursin("noise = :rician", msg(() -> montecarlo(m, θ; snr = 20, noise = :gaussian,
                                                            likelihood = :rician, n = 10)))
    @test occursin("sandwich", msg(() -> crlb(m, θ; snr = 20, likelihood = :rician, cov = :biased)))
    @test occursin("leakage", msg(() -> leakage(mq, θq; kmf = 12.5, likelihood = :rician)))
    md = QMTSIR(; SIR_4POINT..., magnitude = false)
    @test occursin("protocol design", msg(() -> ProtocolDesign(md,
        [Scenario(QMTSIR, θq; kmf = 12.5, likelihood = :rician)]; snr = 200,
        acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)), scantime = p -> sum(p.TI .+ p.TD))))
    @test occursin("`method`", msg(() -> Scenario(m, θ; method = :mean)))
end
