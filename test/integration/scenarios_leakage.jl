# =================================================================================================
# test/integration/scenarios_leakage.jl: Scenario (a truth and the fit judged against it), leakage (how an error in a fixed
# parameter moves the fitted ones), uncertain inputs (a measured fixed value's SD, through
# leakage's slope), and sets of scenarios (calcVmax's modes).
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl scenarios_leakage
# =================================================================================================

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
    @test Scenario(QMTSIR, θ; kmf = Free()).roles.kmf == Free(start = 5.0, lower = 0)
    @test Scenario(QMTSIR, θ; kmf = Free(), starts = :model).roles.kmf == Free(start = 12.5, lower = 0)
    @test Scenario(QMTSIR, θ; kmf = Free(start = 8.0)).roles.kmf == Free(start = 8.0, lower = 0)
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
    @test noisefree(m4, s).params == params_only(c, c.fit) && bias(m4, s) == params_only(c, c.bias)
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
    v = montecarlo(m4, s; noise = :rician, snr = 200, n = 50, vary = (PSR = [0.1, 0.15],), rng = MersenneTwister(2))
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
        @test_noalloc @allocated(k(lc, t)) == 0
    end
    out = sprint(show, MIME"text/plain"(), lb)
    @test occursin("kmf = 12.5 (true 5.0)", out) && occursin("relative:", out) && occursin("elasticity:", out)
    @test occursin("fixes no parameter at a number", msg(() -> leakage(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08))))
    @test occursin("the scenario is for QMTSIR", msg(() -> leakage(T2Decay(TE = te), Scenario(m5, θ))))
end

@testset "uncertain inputs: a measured fixed value's SD in crlb and montecarlo" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    # VFA T1 with a B1 map 5% high whose values scatter with an SD of 0.03 (docs/design.md,
    # roadmap item 16: the case measured when the method was worked out)
    m = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
    θ = (M0 = 1000.0, T1 = 1.0, B1 = 1.0)
    c0 = crlb(m, θ; snr = 1000, B1 = 1.05)
    cu = crlb(m, θ; snr = 1000, B1 = 1.05, uncertain = (B1 = 0.03,))
    # The covariance adds d·s²·dᵀ, d = dθ̂/dB1 where the fit runs (B1 = 1.05), here by central
    # differences of crlb's noise-free fits (independent of leakage's derivatives); the bias is
    # the map's systematic error, as without `uncertain`
    h = 1e-5
    d = [(crlb(m, θ; snr = 1000, B1 = 1.05 + h).fit[p] - crlb(m, θ; snr = 1000, B1 = 1.05 - h).fit[p]) / 2h
         for p in (:M0, :T1)]
    @test cu.cov ≈ c0.cov + d * d' * 0.03^2 rtol = 1e-6
    @test cu.bias == c0.bias && cu.rmse.T1 ≈ hypot(cu.bias.T1, cu.sd.T1)
    # An SD of 0 is no uncertainty; the result keeps the SDs given and shows them
    @test crlb(m, θ; snr = 1000, B1 = 1.05, uncertain = (B1 = 0.0,)).sd == c0.sd
    @test cu.uncertain == (B1 = 0.03,) && occursin("B1 ± 0.03", sprint(show, MIME"text/plain"(), cu))
    # A sweep and fits side by side (code paths of their own): each setting as the single call
    @test crlb(m, θ; snr = [500, 1000], B1 = 1.05, uncertain = (B1 = 0.03,))[2].sd == cu.sd
    @test crlb(m, (a = Scenario(m, θ; B1 = 1.05),); snr = 1000, uncertain = (B1 = 0.03,))[1].sd == cu.sd
    # montecarlo: each copy fitted with its own B1, drawn from N(1.05, 0.03²); the SDs as crlb
    # predicts within 3 standard errors (T1: 0.0572 s; the image noise alone, 0.0223). The bias
    # is not compared: the measured one adds the second-order terms (−0.0926 against −0.0945)
    mc = montecarlo(m, θ; snr = 1000, noise = :gaussian, B1 = 1.05, uncertain = (B1 = 0.03,),
                    n = 20_000, rng = MersenneTwister(1))
    @test mc.crlb.sd == cu.sd
    @test all(p -> abs(mc.sd[p] - cu.sd[p]) < 3 * mc.se_sd[p], (:M0, :T1))
    # Mistakes, and what is not yet
    θq = nominal(QMTSIR, :SIR_WM_I)
    @test occursin("`M0` is estimated", msg(() -> crlb(m, θ; snr = 1000, uncertain = (M0 = 10.0,))))
    @test occursin("`R1m` is tied", msg(() -> crlb(QMTSIR(; SIR_4POINT...), θq; snr = 200, uncertain = (R1m = 0.1,))))
    @test occursin("`T2` is not a parameter of T1MFA", msg(() -> crlb(m, θ; snr = 1000, uncertain = (T2 = 0.01,))))
    @test occursin("SD of `B1`", msg(() -> crlb(m, θ; snr = 1000, uncertain = (B1 = -0.03,))))
    @test occursin("trailing comma", msg(() -> crlb(m, θ; snr = 1000, uncertain = 0.03)))   # `(B1 = 0.03)`: the number
    @test occursin("likelihood = :rician: not yet", msg(() -> crlb(m, θ; snr = 1000, likelihood = :rician, uncertain = (B1 = 0.03,))))
    @test occursin("noise = :rician: not yet", msg(() -> crlb(m, θ; snr = 1000, noise = :rician, uncertain = (B1 = 0.03,))))
    @test occursin("`simulate_with`: not yet", msg(() -> crlb(m, θ; snr = 1000, simulate_with = m => θ, uncertain = (B1 = 0.03,))))
    @test occursin("give it to `crlb(m, θ", msg(() -> crlb!(CRLBCache(m, θ), θ; snr = 1000, uncertain = (B1 = 0.03,))))
    @test occursin("protocol design: not yet", msg(() -> ProtocolDesign(m, [Scenario(m, θ)]; snr = 1000, uncertain = (B1 = 0.03,))))
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
    rob = scenarios(QMTSIR, tissues; vary = (Sm = [0.78, 0.83, 0.88],), kmf = 12.5, Sm = 0.83)
    @test length(rob) == 12 && rob.settings[1:3] == [(tissue = :SIR_WM_I, Sm = s) for s in (0.78, 0.83, 0.88)]
    @test rob[1].wrong.Sm.truth == 0.78 && !haskey(rob[2].wrong, :Sm)     # 0.83: not wrong
    hr = [Scenario(QMTSIR, merge(nominal(QMTSIR, t), (Sm = s,)); kmf = 12.5, Sm = 0.83) for t in tissues for s in (0.78, 0.83, 0.88)]
    d = ProtocolDesign(m4, rob; kw...)
    @test d(m4) == ProtocolDesign(m4, hr; kw...)(m4) && length(d.groups) == 1
    @test rob.settings[argmax(d.values)] == (tissue = :SIR_WM_I, Sm = 0.78)
    # the midpoint only when asked for (vary's combinations as for crlb), rounded (not
    # 0.8300000000000001) and not repeated; a fixed value not given follows each truth
    @test length(scenarios(QMTSIR, tissues; vary = (Sm = [0.78, 0.88],))) == 8
    mid = scenarios(QMTSIR, tissues; vary = (Sm = [0.78, 0.88],), midpoint = true, kmf = 12.5, Sm = 0.83)
    @test length(mid) == 12 && mid.settings[1] == (tissue = :SIR_WM_I, Sm = 0.83) && !haskey(mid[1].wrong, :Sm)
    @test length(scenarios(QMTSIR, tissues; vary = (Sm = [0.78, 0.83, 0.88],), midpoint = true)) == 12
    own = scenarios(QMTSIR, [:SIR_WM_I]; vary = (Sm = [0.78, 0.83, 0.88],))
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
    @test occursin("a tie has no midpoint", msg(() -> scenarios(QMTSIR, tissues; vary = (R1m = [:R1f, 1.0],), midpoint = true)))
    @test length(scenarios(QMTSIR, tissues; vary = (R1m = [:R1f, 1.0],))) == 8
    @test occursin("names the truths", msg(() -> scenarios(QMTSIR, tissues; vary = (tissue = [1, 2],))))
    @test occursin("not a parameter of QMTSIR", msg(() -> scenarios(QMTSIR, tissues; vary = (T1 = [1.0, 2.0],))))
    @test occursin("a tissue name", msg(() -> scenarios(QMTSIR, [1.0, 2.0])))
    @test occursin("at least one truth", msg(() -> scenarios(QMTSIR, Symbol[])))
    # the scale parameter need not be varied: with snr relative to it, Eq. 5's terms do not
    # depend on it
    sm0 = scenarios(QMTSIR, nominal(QMTSIR, :SIR_WM_I); vary = (M0f = [0.8, 1.0, 1.2],), kmf = 12.5)
    dm0 = ProtocolDesign(m4, sm0; kw...)
    dm0(m4)
    @test maximum(dm0.values) - minimum(dm0.values) < 1e-12 * maximum(dm0.values)
    # crlb and montecarlo take a set: one result per scenario, the set's settings (after the
    # noise levels), each as the single-scenario call
    mm = QMTSIR(; SIR_4POINT...)
    two = scenarios(QMTSIR, [:SIR_WM_I, :SIR_lesion]; vary = (Sm = [0.78, 0.83, 0.88],), kmf = 12.5, Sm = 0.83)
    c = crlb(mm, two; snr = 200)
    @test c isa Sweep && length(c) == 6 && c.settings == two.settings
    @test all(i -> c[i].sd == crlb(mm, two[i]; snr = 200).sd && c[i].bias == crlb(mm, two[i]; snr = 200).bias, 1:6)
    @test length(c.rows) == 6 * 6 && c.rows[7].tissue === :SIR_WM_I && c.rows[7].Sm == 0.83   # 4 parameters, MPF, kfm
    c2 = crlb(mm, two; snr = [100, 200], cov = :biased)
    @test length(c2) == 12 && c2.settings[1] == (snr = 100, tissue = :SIR_WM_I, Sm = 0.78)
    @test c2[8].sd == crlb(mm, two[2]; snr = 200, cov = :biased).sd
    mc = montecarlo(mm, two; noise = :rician, snr = 200, n = 200, rng = MersenneTwister(1))
    @test length(mc) == 6 && mc[3].sd == montecarlo(mm, two[3]; noise = :rician, snr = 200, n = 200, rng = MersenneTwister(1)).sd
    @test occursin("SIR_lesion", sprint(show, MIME"text/plain"(), c)) && !occursin(":SIR_lesion", sprint(show, MIME"text/plain"(), c))
    @test occursin("vary them in `scenarios`", msg(() -> crlb(mm, two; snr = 200, vary = (kmf = [5.0, 10.0],))))
    cr = crlb(mm, two; snr = 200, noise = :rician)      # magnitude data: each as the single call
    @test cr[4].sd == crlb(mm, two[4]; snr = 200, noise = :rician).sd && cr[4].sd != c[4].sd
    @test occursin("Scenario takes no noise", msg(() -> scenarios(QMTSIR, [:SIR_WM_I]; noise = :rician)))
    @test occursin("not a set", msg(() -> montecarlo(mm, two; noise = :rician, snr = 200, n = 10, simulate_with = mm => nominal(mm))))
end
