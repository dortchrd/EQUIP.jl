# =================================================================================================
# test/integration/crlb.jl: crlb: bias, SD and RMSE from derivatives (Dortch et al., MRM 2018), its pieces, and the
# textbook formulas.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl crlb
# =================================================================================================

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
    @test_noalloc allocs(plan, mb, tt, 0.005^2) == 0 && allocs(plan, m4, tt, 0.005^2) == 0
    plan16 = CRLBCache(m16, θ16; cov = :biased, kmf = 12.5)
    t16 = EQUIP._crlb_truth(m16, θ16)
    allocs(plan16, m16, t16, fill(0.005^2, 16))
    @test_noalloc allocs(plan16, m16, t16, fill(0.005^2, 16)) == 0
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
    # `noise` is the data's: crlb takes it (:gaussian, the default; :rician, magnitude data:
    # test/integration/rician.jl), as simulate and montecarlo; its pieces, the fit (whose noise
    # model is `likelihood`) and a scenario refuse it, saying why
    @test crlb(m4, Scenario(m4, θ); noise = :gaussian, snr = 200).sd == crlb(m4, θ; snr = 200).sd
    @test occursin(":poisson", msg(() -> crlb(m4, θ; noise = :poisson, snr = 200)))
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
