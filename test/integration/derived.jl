# =================================================================================================
# test/integration/derived.jl: Derived quantities: in crlb and montecarlo (the delta method), truths given by them
# (nominal, fromderived) with the eigen derivatives, presets, Scenario, simulate and sweeps.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl derived
# =================================================================================================

@testset "Derived quantities in crlb and montecarlo: after the parameters, by the delta method" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    te = collect(0.01:0.01:0.16)
    # R2 = 1/T2 after the parameters in every named tuple: its SD SD(T2)/T2² (the delta
    # method), its bias the change in 1/T2 when the fit assumes something wrong; the
    # parameters' own numbers, covariance, cond and cor as without it
    m = DerivedT2(TE = te)
    θ = (S0 = 1000.0, T2 = 0.05)
    c, c0 = crlb(m, θ; snr = 100), crlb(BlockT2(TE = te), θ; snr = 100)    # (BlockT2: none)
    @test keys(c.sd) == (:S0, :T2, :R2) && keys(c0.sd) == (:S0, :T2)
    @test c.sd.S0 == c0.sd.S0 && c.sd.T2 == c0.sd.T2 && vcov(c) == vcov(c0)
    @test c.sd.R2 ≈ c.sd.T2 / θ.T2^2 rtol = 1e-12
    @test c.fit.R2 ≈ 20 && c.bias.R2 == 0 && c.relsd.R2 ≈ c.sd.R2 / 20
    @test stderror(c) == stderror(c0) && cond(c) == cond(c0) && cor(c) == cor(c0)
    cb = crlb(m, θ; snr = 100, S0 = 950.0)          # S0 assumed 5% low: T2 biased, R2 with it
    @test keys(cb.sd) == (:T2, :R2)
    @test cb.fit.R2 ≈ 1 / cb.fit.T2 rtol = 1e-12
    @test cb.bias.R2 ≈ 1 / cb.fit.T2 - 20 rtol = 1e-10
    @test cb.rmse.R2 ≈ hypot(cb.bias.R2, cb.sd.R2)
    @test occursin("derived from the parameters", sprint(show, MIME"text/plain"(), c))
    @test !occursin("derived", sprint(show, MIME"text/plain"(), c0))
    # in a sweep's table, a row per derived quantity, with its true value
    s = crlb(m, θ; snr = [50, 100])
    @test length(s.rows) == 6 && s.rows[3].parameter == :R2 && s.rows[3].truth ≈ 20
    @test s.rows[6].sd ≈ c.sd.R2 rtol = 1e-12
    # montecarlo: the same statistics of each copy's 1/T2, against crlb's
    mc = montecarlo(m, θ; snr = 100, noise = :gaussian, n = 2000, rng = MersenneTwister(41))
    r2 = 1 ./ mc.result.params.T2
    @test keys(mc.sd) == keys(c.sd) && keys(mc.crlb.sd) == keys(c.sd)
    @test mc.mean.R2 ≈ mean(r2) && mc.sd.R2 ≈ std(r2) && mc.bias.R2 ≈ mean(r2) - 20
    @test mc.se_sd.R2 ≈ mc.sd.R2 / sqrt(2 * 1999)
    @test abs(mc.sd.R2 / c.sd.R2 - 1) < 4 * mc.se_sd.R2 / c.sd.R2
    @test length(stderror(mc)) == 2 && size(vcov(mc)) == (2, 2)
    @test occursin("derived from each copy's fit", sprint(show, MIME"text/plain"(), mc))
    sm = montecarlo(m, θ; snr = [50, 100], noise = :gaussian, n = 200, rng = MersenneTwister(43))
    @test length(sm.rows) == 6 && sm.rows[3].parameter == :R2 && sm.rows[3].truth ≈ 20
    # a derived quantity named as a parameter: refused
    @test occursin("the name of a parameter", msg(() -> crlb(ClashT2(TE = te), θ; snr = 100)))
    # DTI (a rotated tensor, eigenvalues 1.7, 0.4, 0.3e-3 mm²/s): the seven quantities by the
    # delta method, ∇g·C·∇gᵀ with ∇g through `derived` (written out here); MD, FA, AD and RD
    # within Monte Carlo's standard errors of the fits' SDs. (V1 is not compared: each fit's is
    # folded into the upper hemisphere, which crlb's derivative does not see.)
    bv = [fill(0.0, 6); fill(1000.0, 30)]
    md = DTI(; diffusionencoding(bv, reduce(hcat, [fill([0.0, 0.0, 0.0], 6); collect.(directionset(30))]))...)
    Q = Matrix(qr(randn(MersenneTwister(8), 3, 3)).Q)
    D = Q * Diagonal([1.7e-3, 0.4e-3, 0.3e-3]) * Q'
    θd = (S0 = 1000.0, Dxx = D[1, 1], Dyy = D[2, 2], Dzz = D[3, 3], Dxy = D[1, 2], Dxz = D[1, 3], Dyz = D[2, 3])
    cd = crlb(md, θd; snr = 30)
    names = (:MD, :FA, :AD, :RD, :V1x, :V1y, :V1z)
    @test keys(cd.sd) == (keys(θd)..., names...) && length(stderror(cd)) == 7
    ∇ = ForwardDiff.jacobian(x -> collect(values(derived(md, NamedTuple{keys(θd)}(Tuple(x))))), collect(values(θd)))
    sdd = sqrt.(diag(∇ * vcov(cd) * ∇'))
    @test all(j -> isapprox(cd.sd[names[j]], sdd[j]; rtol = 1e-10), 1:7)
    @test all(k -> cd.bias[k] == 0 && cd.fit[k] ≈ derived(md, θd)[k], names)
    mcd = montecarlo(md, θd; snr = 30, noise = :gaussian, n = 3000, rng = MersenneTwister(42))
    for k in (:MD, :FA, :AD, :RD)
        @test abs(mcd.sd[k] / cd.sd[k] - 1) < 4 * mcd.se_sd[k] / cd.sd[k]
    end
end

@testset "Truths from derived quantities (nominal), and derivatives where eigenvalues are equal" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    FD = EQUIP.ForwardDiff
    n = fiberdirection(0.3, 1.2)
    # DTI: an axially symmetric tensor from (MD, FA) or (AD, RD) and its axis; `derived` gives
    # them back, V1 the axis (in the upper hemisphere)
    bv = [fill(0.0, 6); fill(1000.0, 30); fill(2000.0, 30)]
    G = reduce(hcat, [fill([0.0, 0.0, 0.0], 6); collect.(directionset(30)); collect.(directionset(30))])
    mt = DTI(; diffusionencoding(bv[1:36], G[:, 1:36])...)
    θ = nominal(DTI; S0 = 1000.0, MD = 0.8e-3, FA = 0.7, direction = n)
    d = derived(mt, θ)
    @test keys(θ) == keys(parameters(DTI)) && θ.S0 == 1000
    @test d.MD ≈ 0.8e-3 rtol = 1e-12
    @test d.FA ≈ 0.7 rtol = 1e-12
    @test d.RD ≈ d.MD - (d.AD - d.MD) / 2 rtol = 1e-10
    @test abs(dot([d.V1x, d.V1y, d.V1z], collect(n))) ≈ 1 rtol = 1e-12
    θx = nominal(mt; AD = 1.7e-3, RD = 0.35e-3, direction = (2, 0, 0))     # any length
    @test θx == (S0 = 1.0, Dxx = 1.7e-3, Dyy = 0.35e-3, Dzz = 0.35e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0)
    # what it refuses, and what it says
    @test occursin("give the direction", msg(() -> nominal(DTI; MD = 0.8e-3, FA = 0.7)))
    @test occursin("MD and FA together", msg(() -> nominal(DTI; MD = 0.8e-3, direction = n)))
    @test occursin("not both", msg(() -> nominal(DTI; MD = 0.8e-3, FA = 0.7, AD = 1e-3, RD = 1e-4, direction = n)))
    @test occursin("unknown `Fa`", msg(() -> nominal(DTI; Fa = 0.7)))
    @test occursin("oblate", msg(() -> nominal(DTI; AD = 0.3e-3, RD = 0.5e-3, direction = n)))
    @test occursin("0 ≤ FA < 1", msg(() -> nominal(DTI; MD = 0.8e-3, FA = 1.0, direction = n)))
    @test occursin("three finite numbers", msg(() -> nominal(DTI; MD = 0.8e-3, FA = 0.7, direction = (0, 0, 0))))
    @test occursin("give one or the other", msg(() -> nominal(DTI; MD = 0.8e-3, FA = 0.7, direction = n, Dxx = 1e-3)))
    @test occursin("unknown parameter `FA`", msg(() -> nominal(T2Decay; T2 = 0.08, FA = 0.7)))  # no hook
    # DKI: D as DTI's; W isotropic from MKT alone, the same whatever the axis (V1111 = MD²·MKT,
    # V1122 = MD²·MKT/3), or axially symmetric about it from MK, AK and RK, given back
    mk = DKI(; diffusionencoding(bv, G)...)
    θi = nominal(DKI; S0 = 1000.0, AD = 1.7e-3, RD = 0.35e-3, direction = (1, 0, 0), MKT = 1.0)
    MD2 = ((1.7e-3 + 0.7e-3) / 3)^2
    @test θi.V1111 ≈ MD2 && θi.V2222 ≈ MD2 && θi.V1122 ≈ MD2 / 3 && abs(θi.V1123) < 1e-12 * MD2
    @test derived(mk, θi).MKT ≈ 1 rtol = 1e-12
    θi2 = nominal(DKI; S0 = 1000.0, AD = 1.7e-3, RD = 0.35e-3, direction = n, MKT = 1.0)
    V(p) = [p[k] for k in keys(p)[8:22]]
    @test V(θi2) ≈ V(θi) rtol = 1e-10
    θk = nominal(DKI; S0 = 1000.0, AD = 1.7e-3, RD = 0.35e-3, direction = n, MK = 1.0, AK = 0.8, RK = 1.3)
    dk = derived(mk, θk)
    @test all(k -> isapprox(dk[k], (MK = 1.0, AK = 0.8, RK = 1.3)[k]; rtol = 1e-10), (:MK, :AK, :RK))
    @test dk.AD ≈ 1.7e-3 rtol = 1e-12
    @test dk.RD ≈ 0.35e-3 rtol = 1e-12
    W = V(θk) ./ dk.MD^2                     # symmetric about n: the same K at each angle to it
    W4(x) = dot(EQUIP._kurtosisweights(x...), W)
    u = normalize(cross(collect(n), [1.0, 0, 0])); w = cross(collect(n), u)
    ring = [W4(cos(0.4) .* collect(n) .+ sin(0.4) .* (cos(φ) .* u .+ sin(φ) .* w)) for φ in 0:0.5:3]
    @test maximum(ring) - minimum(ring) < 1e-10 * maximum(abs, ring)
    @test occursin("the kurtosis: give MKT", msg(() -> nominal(DKI; AD = 1.7e-3, RD = 0.35e-3, direction = n, MK = 1.0)))
    @test occursin("outside [−3/7, 10]", msg(() -> nominal(DKI; AD = 1.7e-3, RD = 0.35e-3, direction = n, MK = 11.0, AK = 1.0, RK = 1.0)))
    @test occursin("no value for `V1111`", msg(() -> nominal(DKI; AD = 1.7e-3, RD = 0.35e-3, direction = n)))
    # the eigensolver's derivatives (perturbation theory, `_symeigen`): DTI's quantities against
    # central differences for a rotated tensor with distinct eigenvalues, and for an exactly
    # prolate one (λ2 = λ3), where differentiating the closed form gave NaN; second
    # derivatives through nested duals; an oblate one (λ1 = λ2): AD, RD and V1 NaN (e1 is not
    # defined), MD and FA not
    Q = Matrix(qr(randn(MersenneTwister(8), 3, 3)).Q)
    el(D) = [D[1, 1], D[2, 2], D[3, 3], D[1, 2], D[1, 3], D[2, 3]]
    mat(x) = SMatrix{3,3}(x[1], x[4], x[5], x[4], x[2], x[6], x[5], x[6], x[3])
    q(x) = collect(values(first(EQUIP._tensorindices(mat(x)))))
    central(f, x, h) = reduce(hcat, [(f(x .+ h .* (1:6 .== k)) .- f(x .- h .* (1:6 .== k))) ./ 2h for k in 1:6])
    for λ in ([1.7e-3, 0.4e-3, 0.3e-3], [1.7e-3, 0.4e-3, 0.4e-3])
        x = el(Q * Diagonal(λ) * Q')
        J = FD.jacobian(q, x)
        @test all(isfinite, J)
        @test J ≈ central(q, x, 1e-10) rtol = 1e-6
    end
    x3 = el(Q * Diagonal([1.7e-3, 0.4e-3, 0.3e-3]) * Q')
    gad(x) = FD.gradient(y -> q(y)[3], x)
    @test FD.jacobian(gad, x3) ≈ central(gad, x3, 1e-10) rtol = 1e-6
    Jo = FD.jacobian(q, el(Q * Diagonal([1.7e-3, 1.7e-3, 0.4e-3]) * Q'))
    @test all(isfinite, Jo[1:2, :]) && all(isnan, Jo[3:7, :])
    # DKI at its axially symmetric truth (D prolate, W symmetric about the axis): MK, AK and
    # RK differentiable, against central differences in random directions; with W made
    # asymmetric, MK's and RK's derivatives NaN (the basis in D's equal pair matters), AK's not
    names = keys(θk)[2:22]
    x0 = [θk[k] for k in names]
    ind(x) = (r = derived(mk, merge(θk, NamedTuple{names}(Tuple(x)))); [r.MK, r.AK, r.RK, r.AD, r.RD])
    J = FD.jacobian(ind, x0)
    rng = MersenneTwister(5)
    for _ in 1:3
        v = randn(rng, 21) .* abs.(x0 .+ 1e-4 * maximum(abs, x0))
        @test J * v ≈ (ind(x0 .+ 1e-6 .* v) .- ind(x0 .- 1e-6 .* v)) ./ 2e-6 rtol = 1e-7
    end
    θa = merge(θk, (V2223 = 0.1 * θk.V1111,))
    Ja = FD.jacobian(ind, [θa[k] for k in names])
    @test all(isnan, Ja[1, :]) && all(isnan, Ja[3, :]) && all(isfinite, Ja[2, :])
    # G1 and G2 without their cancellation: DIPY's forms where those are accurate, and their
    # limits at b = c; RK where λ2 and λ3 are 1e-7 apart, its value at equality (DIPY's form: 3% off)
    G1d(a, b, c) = (a + b + c)^2 / (18b * (b - c)^2) * (2b + (c^2 - 3b * c) / sqrt(b * c))
    G2d(a, b, c) = (a + b + c)^2 / (3 * (b - c)^2) * ((b + c) / sqrt(b * c) - 2)
    @test EQUIP._G1(1.7, 0.4, 0.3) ≈ G1d(1.7, 0.4, 0.3) rtol = 1e-12
    @test EQUIP._G2(1.7, 0.4, 0.3) ≈ G2d(1.7, 0.4, 0.3) rtol = 1e-12
    @test EQUIP._G1(1.7, 0.4, 0.4) ≈ (1.7 + 0.8)^2 / (24 * 0.16) rtol = 1e-14
    @test EQUIP._G2(1.7, 0.4, 0.4) ≈ (1.7 + 0.8)^2 / (12 * 0.16) rtol = 1e-14
    θn = merge(θi, (Dzz = 0.35e-3 * (1 - 1e-7),))
    @test derived(mk, θn).RK ≈ derived(mk, θi).RK rtol = 1e-6
    # crlb at the axially symmetric truth, along x (V's off-diagonal elements and the pairs
    # fixed at the truth's: P = 10, as the DKI test set's): MK, AK and RK finite (NaN before),
    # against 400 fits
    θxk = nominal(DKI; S0 = 1000.0, AD = 1.7e-3, RD = 0.35e-3, direction = (1, 0, 0), MK = 1.0, AK = 0.8, RK = 1.3)
    fixk = NamedTuple{(:V1112, :V1113, :V1222, :V2223, :V1333, :V2333, :V1123, :V1223, :V1233, :V1122, :V1133, :V2233)}(
        map(k -> θxk[k], (:V1112, :V1113, :V1222, :V2223, :V1333, :V2333, :V1123, :V1223, :V1233, :V1122, :V1133, :V2233)))
    ck = crlb(mk, θxk; snr = 200, fixk...)
    @test all(k -> isfinite(ck.sd[k]) && ck.sd[k] > 0, (:MK, :AK, :RK, :AD, :RD))
    Yk = simulate(mk, θxk; snr = 200, n = 400, noise = :gaussian, rng = MersenneTwister(33))
    dkf = derived(fit(mk, Yk; fixk...))
    for k in (:MK, :AK, :RK)
        @test std(dkf[k]) ≈ ck.sd[k] rtol = 0.15
    end
end

@testset "The truth by derived quantities: tissue presets, Scenario, simulate, sweeps" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    bv = [fill(0.0, 6); fill(1000.0, 30); fill(2000.0, 30)]
    G = reduce(hcat, [fill([0.0, 0.0, 0.0], 6); collect.(directionset(30)); collect.(directionset(30))])
    mt = DTI(; diffusionencoding(bv[1:36], G[:, 1:36])...)
    mk = DKI(; diffusionencoding(bv, G)...)
    # the presets (Lätt et al. 2013, Table 1): MD and FA for DTI, the axially symmetric
    # tensor's RD within 4% of the table's; MK and RK for DKI, with AK given in the call
    rd = (DKI_CC_splenium = 0.31e-3, DKI_CC_genu = 0.36e-3, DKI_CC_body = 0.38e-3,
          DKI_PLIC = 0.45e-3, DKI_centrum_semiovale = 0.47e-3, DKI_corona_radiata = 0.56e-3,
          DKI_frontal_WM = 0.66e-3, DKI_thalamus = 0.73e-3, DKI_putamen = 0.73e-3, DKI_caudate = 0.80e-3)
    for (name, r) in pairs(rd)
        d = derived(mt, nominal(DTI, name; S0 = 1000.0, direction = (1, 0, 0)))
        @test d.MD ≈ tissue(name).MD rtol = 1e-12
        @test d.FA ≈ tissue(name).FA rtol = 1e-12
        @test d.RD ≈ r rtol = 0.04
    end
    dk = derived(mk, nominal(DKI, :DKI_CC_splenium; S0 = 1000.0, direction = (1, 0, 0), AK = 0.8))
    @test all(k -> isapprox(dk[k], (MK = 1.32, AK = 0.8, RK = 2.72)[k]; rtol = 1e-10), (:MK, :AK, :RK))
    @test occursin("give AK too", msg(() -> nominal(DKI, :DKI_CC_splenium; direction = (1, 0, 0))))
    # whole GM and WM with every index (Maiter et al., Sci Rep 2021;11:8857), no AK in the call:
    # the tensors made from them give the published values back; free water (DKI_CSF): isotropic,
    # MD 3.0e-3 mm²/s, no kurtosis
    for name in (:DKI_GM, :DKI_WM)
        d = derived(mk, nominal(DKI, name; S0 = 1000.0, direction = (1, 0, 0)))
        @test all(k -> isapprox(d[k], tissue(name)[k]; rtol = 1e-10), (:MD, :FA, :MK, :AK, :RK))
    end
    dc = derived(mk, nominal(DKI, :DKI_CSF; S0 = 1000.0, direction = (1, 0, 0)))
    @test dc.MD ≈ 3.0e-3 && dc.AD ≈ dc.RD && abs(dc.FA) < 1e-12
    @test all(k -> abs(dc[k]) < 1e-12, (:MK, :AK, :RK, :MKT))
    @test derived(mt, nominal(DTI, :DKI_CC_splenium; FA = 0.5, direction = (1, 0, 0))).FA ≈ 0.5   # the keyword wins
    @test occursin("no value for `T2`", msg(() -> nominal(T2Decay(TE = [0.01, 0.02]), :DKI_thalamus)))
    # a truth by derived quantities: a Scenario keeps it as given (`given`), with its
    # parameters; simulate and crlb take it, the same as by its parameters
    spec = (S0 = 1000.0, MD = 0.8e-3, FA = 0.7, direction = fiberdirection(0.3, 1.2))
    θ = nominal(DTI; spec...)
    s = Scenario(DTI, spec)
    @test s.truth == Scenario(DTI, θ).truth && s.given == spec
    @test simulate(mt, spec; snr = 50, noise = :rician, n = 3, rng = MersenneTwister(2)) ==
          simulate(mt, θ; snr = 50, noise = :rician, n = 3, rng = MersenneTwister(2))
    @test crlb(mt, spec; snr = 50).sd == crlb(mt, θ; snr = 50).sd
    @test scenarios(DTI, spec).results[1].truth == s.truth          # (a named tuple: a tissue)
    # a sweep over FA: MD and the direction held as given, each setting the crlb of its own
    # truth; montecarlo's too; a parameter varied as before
    sw = crlb(mt, spec; snr = 50, vary = (FA = [0.3, 0.6],))
    @test [r.fit.FA for r in sw.results] ≈ [0.3, 0.6] && all(r -> r.fit.MD ≈ 0.8e-3, sw.results)
    @test sw.results[2].sd == crlb(mt, merge(spec, (FA = 0.6,)); snr = 50).sd
    @test count(r -> r.parameter == :FA, sw.rows) == 2 && sw.settings[1] == (FA = 0.3,)
    msw = montecarlo(mt, spec; snr = 50, noise = :gaussian, n = 50, vary = (FA = [0.3, 0.6],), rng = MersenneTwister(3))
    @test [r.crlb.fit.FA for r in msw.results] ≈ [0.3, 0.6]
    @test crlb(mt, spec; snr = 50, vary = (S0 = [500.0, 1000.0],)).results[2].sd == crlb(mt, θ; snr = 50).sd
    # refused: a derived quantity varied with a truth given by parameters (what would stay
    # fixed?); a name the model does not take; `scenarios`' vary, which takes parameters
    @test occursin("given by its parameters", msg(() -> crlb(mt, θ; snr = 50, vary = (FA = [0.3],))))
    @test occursin("the derived quantities it takes", msg(() -> crlb(mt, spec; snr = 50, vary = (Fa = [0.3],))))
    @test occursin("takes parameters", msg(() -> scenarios(DTI, spec; vary = (FA = [0.3],))))
    # data from another model, by derived quantities: DTI fitted to DKI's signal with K = 1
    # (b to 2000 s/mm²): the kurtosis the tensor leaves out biases MD down and FA up
    m66 = DTI(; diffusionencoding(bv, G)...)
    θk = merge(spec, (MKT = 1.0,))
    cw = crlb(m66, θ; snr = 50, simulate_with = mk => θk)
    @test cw.bias.MD < 0 && cw.bias.FA > 0
    @test cw.bias == crlb(m66, θ; snr = 50, simulate_with = mk => nominal(DKI; θk...)).bias
end
