# =================================================================================================
# test/unit/starts_solver.jl: Starting values and the solver: fit! allocates nothing, the dead-zone rule, T2Decay's
# start on 1/T2, BestOf for magnitude inversion recovery, the generic :loglinear start.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl starts_solver
# =================================================================================================

@testset "fit! does not allocate" begin
    for (m, truth) in CASES, T in (Float64, Float32)
        y = signal(m, truth) .* (1 .+ 0.01 .* randn(MersenneTwister(0), nobs(m)))
        c = init(m; precision = T)
        fit!(c, y)                              # first call compiles
        fitallocs(c, y)
        @test_noalloc fitallocs(c, y) == 0
        @inferred fit!(c, y)                    # the result's type is known when compiling
    end
end

@testset "solver: steps into a dead zone are rejected; bounds are still reached" begin
    # A step that pushes a time constant to ~0 leaves it with no influence on the signal
    # (every sample has fully recovered). Such steps are rejected, so short T1 is found from
    # the default start (T1 = 1 s); before the guard these fits ended at T1 = 1e-6.
    sr = T1SR(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0])
    for T1 in (0.05, 0.1, 0.2)
        @test fit(sr, signal(sr, (M0 = 1000.0, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
    end
    mfa = T1MFA(FA = [2.0, 5, 10, 15, 20, 30], TR = 0.015)
    for T1 in (0.05, 0.2)
        @test fit(mfa, signal(mfa, (M0 = 1000.0, T1 = T1, B1 = 1.0))).params.T1 ≈ T1 rtol = 1e-6
    end
    # fits whose answer lies on a bound still end exactly there (there the parameter still
    # affects the signal, so the guard does not apply)
    y = 1000 .+ 0.05 .* (1:32) .+ [isodd(j) ? 1.0 : -1.0 for j in 1:32]     # slightly rising
    r = fit(R2Decay(TE = collect(0.01:0.01:0.32)), y)
    @test r.params.R2 == 0.4                         # R2's bound, 1/2.5 s (builtin.jl, 2026-11-09)
    @test isconverged(r)
    @test fit(R2Decay(TE = collect(0.01:0.01:0.32)), y; R2 = Free(lower = 0)).params.R2 == 0   # widened
    m = QMTSIR(TI = TI5, TD = TD5)
    r = fit(m, signal(m, merge(defaults(m), (PSR = 0.0, R1f = 1.0, Sf = -0.95, M0f = 1000.0))))   # no MT
    @test r.params.PSR == 0
    @test r.params.R1f ≈ 1.0 rtol = 1e-6
    # the guard itself: parameter 2 loses (almost) all influence; frozen parameters don't count
    A = [4.0 0.0; 0.0 9.0]
    @test EQUIP._lost_influence(A, [4.0 0.0; 0.0 1e-20], [1.0, 1.0])
    @test !EQUIP._lost_influence(A, [4.0 0.0; 0.0 1e-3], [1.0, 1.0])
    @test !EQUIP._lost_influence(A, [4.0 0.0; 0.0 0.0], [1.0, 0.0])
end

@testset "T2Decay's start: the log-linear fit, on 1/T2" begin
    te = collect(0.01:0.01:0.32)
    m = T2Decay(TE = te)
    y = signal(m, (S0 = 1000.0, T2 = 0.04))
    start(m, y; kw...) = fit(m, y; solver = LM(maxiter = 0), kw...).params
    # Noise-free data: the log-linear fit is exact, so the start is already the answer.
    # `maxiter = 0` returns the starting point; S0 comes from least squares given that T2.
    s = start(m, y)
    @test s.T2 ≈ 0.04 rtol = 1e-12
    @test s.S0 ≈ 1000 rtol = 1e-12
    @test start(m, Float32.(y); precision = Float32).T2 ≈ 0.04f0
    # points ≤ 0 are skipped
    @test start(m, [y[1:5]; zeros(27)]).T2 ≈ 0.04 rtol = 1e-10
    # no decay (rising data): T2 stays where it started, very long, clamped to its upper bound
    # (2.5 s, `_T2MAX`); the fit ends there and `onbound` flags it, in both precisions, where
    # before the bound it reported convergence at about 1e10 s (2026-11-08)
    @test start(m, collect(1.0:32.0)).T2 == 2.5
    for T in (Float64, Float32)
        rb = fit(m, T.(collect(1.0:32.0)); precision = T)
        @test rb.params.T2 == T(2.5) && onbound(rb).T2
    end
    @test setup(m).upper[2] == 2.5
    @test setup(m; T2 = Free(upper = 10)).upper[2] == 10 && setup(m; T2 = Free(upper = 10)).fromlog[2]
    @test fit(m, collect(1.0:32.0); T2 = Free(upper = Inf)).params.T2 > 1e6   # widened: as before
    @test !onbound(fit(m, y)).T2                                               # tissue: not on it
    # T2 is a time constant (no signal at 0): its log fit runs on 1/T2, the same estimator as
    # R2Decay's at :loglinear, in the other coordinate
    @test setup(m).fromlog == (false, true) && setup(m).reciprocal == (false, true)
    mr = R2Decay(TE = te)
    @test setup(mr; R2 = Free(start = :loglinear)).reciprocal == (false, false)
    yn = y .+ 10 .* randn(MersenneTwister(3), 32)
    @test start(m, yn).T2 ≈ 1 / start(mr, yn; R2 = Free(start = :loglinear, lower = 0)).R2 rtol = 1e-10
    # T2 starts at :loglinear by default; another start replaces it; a fixed T2 is left alone
    @test setup(m).fromdata == (false, false)
    @test fit(m, y; T2 = Free(start = 0.2), solver = LM(maxiter = 0)).params.T2 == 0.2
    @test setup(m; T2 = Free(start = 0.2)).fromlog == (false, false)
    @test fit(m, y; T2 = 0.04).params.S0 ≈ 1000
    # noisy data: the same answer as from a constant start, mean(TE), in fewer iterations,
    # without allocating
    r, r0 = fit(m, yn), fit(m, yn; T2 = Free(start = sum(te) / 32))
    @test r.params.T2 ≈ r0.params.T2 rtol = 1e-6
    @test r.iterations < r0.iterations
    c = init(m); fit!(c, yn)
    @test_noalloc @allocated(fit!(c, yn)) == 0
    @test datastart(R2Decay(TE = te), y, (;)) === nothing     # no data start by default
    @test_throws "no data start" setup(R2Decay(TE = te); R2 = Free(start = :data))
    # datastart gets the voxel's fixed values (here c, a number or a map); a NaN fails
    mt = DataStartTest([0.01, 0.02, 0.04, 0.08])
    yt = signal(mt, (S0 = 100.0, T2 = 0.05, c = 1.0))
    @test fit(mt, yt; solver = LM(maxiter = 0)).params.T2 ≈ 0.05
    @test fit(mt, permutedims([yt yt]); c = [1.0, 2.0], solver = LM(maxiter = 0)).params.T2 ≈ [0.05, 0.1]
    @test fit(mt, yt; c = -1.0).status == EQUIP.Failed
end

@testset "BestOf start: magnitude inversion recovery's local minima" begin
    # From a single start (T1 = 1 s), magnitude IR used to end in a wrong local minimum (the
    # signal null in the wrong place) at T1 = 0.05, 0.1 and 4 s with these TIs; starting from
    # the best of 30 values (BestOf) finds the right one, in both forms.
    ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    for T1 in (0.05, 0.1, 0.5, 4.0)
        m, mr = T1IR(TI = ti), R1IR(TI = ti)
        @test fit(m, signal(m, (M0 = 1000.0, Sf = -0.95, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
        @test fit(mr, signal(mr, (M0 = 1000.0, Sf = -0.95, R1 = 1 / T1))).params.R1 ≈ 1 / T1 rtol = 1e-6
    end
    for T1 in (0.05, 0.1, 0.2, 0.3, 4.0)
        m, mr = T1IRTD(TI = TI6, TD = TD6), R1IRTD(TI = TI6, TD = TD6)
        @test fit(m, signal(m, (M0 = 1000.0, Sf = -0.95, T1 = T1, Mz_td0 = 0.0))).params.T1 ≈ T1 rtol = 1e-6
        @test fit(mr, signal(mr, (M0 = 1000.0, Sf = -0.95, R1 = 1 / T1, Mz_td0 = 0.0))).params.R1 ≈ 1 / T1 rtol = 1e-6
    end
    # 30 values spanning the TIs (with T1's bound removed); by default those above the 5 s bound
    # dropped (since 2026-11-09: the 5 of 30 above it here); replaced by any other start, absent
    # when T1 is fixed
    s = setup(T1IR(TI = ti); T1 = Free(upper = Inf))
    @test s.candindex == 3 && length(s.candidates) == 30
    @test s.candidates[1] ≈ 0.01 && s.candidates[end] ≈ 16
    @test setup(T1IR(TI = ti)).candidates == filter(<=(5), s.candidates)
    @test length(setup(T1IR(TI = ti)).candidates) == 25
    @test setup(T1IR(TI = ti); T1 = Free(start = 2.0)).candindex == 0
    @test setup(T1IR(TI = ti); T1 = 2.0).candindex == 0
    @test setup(T2Decay(TE = [0.01, 0.02])).candindex == 0           # none by default
    # values from the call; those outside the bounds are dropped, an error if none is left
    s = setup(T1IR(TI = ti); T1 = Free(start = BestOf([0.1, 0.5, 5.0]), upper = 3))
    @test s.candidates == [0.1, 0.5] && s.start.T1 == 0.1           # middle value: placeholder
    @test_throws "no BestOf value within" setup(T1IR(TI = ti); T1 = Free(lower = 20, upper = Inf))
    @test_throws "only one parameter" setup(T1IR(TI = ti); Sf = Free(start = BestOf([-1, -0.9])))
    m2 = T2Decay(TE = 0.01:0.01:0.32)
    r = fit(m2, signal(m2, (S0 = 1000.0, T2 = 0.04)); T2 = Free(start = BestOf([0.01, 0.04, 1.0])),
            solver = LM(maxiter = 0))
    @test r.params.T2 == 0.04 && r.params.S0 ≈ 1000                 # the best of the three
    # the values: any list of finite numbers; printed in short
    @test BestOf(0.1:0.1:0.3).values ≈ [0.1, 0.2, 0.3]
    @test_throws ArgumentError BestOf(Float64[])
    @test_throws ArgumentError BestOf([0.1, NaN])
    @test sprint(show, BestOf(1:10)) == "BestOf(10 values from 1.0 to 10.0)"
end

@testset "The log-linear start, :loglinear, from the model's own signal" begin
    b = [0.0, 250, 500, 750, 1000, 1500, 2000, 2500]
    start(m, y; kw...) = fit(m, y; solver = LM(maxiter = 0), kw...).params
    # ADCDecay: ln S is linear in ADC. The start is the two passes, OLS then WLS weighted by
    # OLS's predicted signal (written out here): on noise-free data the truth; the fit's own
    # weights all 2, the same start
    m = ADCDecay(b = b)
    ll = (ADC = Free(start = :loglinear, lower = 0),)
    @test start(m, signal(m, (S0 = 1000.0, ADC = 0.8e-3)); ll...).ADC ≈ 0.8e-3 rtol = 1e-12
    twopass(X, y) = (u = X \ log.(y); w = exp.(X * u); (w .* X) \ (w .* log.(y)))
    y = simulate(m, (S0 = 1000.0, ADC = 0.8e-3); snr = 40, n = 1, noise = :rician, rng = MersenneTwister(3))[1, :]
    @test start(m, y; ll...).ADC ≈ twopass([ones(8) -b], y)[2] rtol = 1e-12
    @test start(m, y; weights = fill(2.0, 8), ll...).ADC ≈ start(m, y; ll...).ADC rtol = 1e-12
    @test_logs min_level = Base.CoreLogging.Warn init(m; ll...)        # the log fit itself: no warning
    # bounds hold in the start: an upper bound below the truth is where the start stops (and
    # the fit, which reports convergence there); a signal rising with b starts at the lower 0
    y0 = signal(m, (S0 = 1000.0, ADC = 0.8e-3))
    llu = (ADC = Free(start = :loglinear, lower = 0, upper = 0.5e-3),)
    @test start(m, y0; llu...).ADC == 0.5e-3
    ru = fit(m, y0; llu...)
    @test ru.params.ADC == 0.5e-3 && isconverged(ru)
    @test start(m, reverse(y0); ll...).ADC == 0
    # ADCKurtosis: ln S = ln S0 − b·ADC + b²·ADC²·K/6 is not linear in (ADC, K), so the two
    # passes are two Gauss–Newton steps (K, whose column is zero at ADC = 0, moves in the
    # second only): an approximate start, from which the fit reaches the same minimum as from
    # the model's own start. The fit says so, naming K (∂ln S/∂K = b²·ADC²/6 is 0 at the start
    # and not at the second point), once per model and parameters (`maxlog`: two fits, one
    # warning); K alone at :loglinear is linear (ADC held at its start), and says nothing
    mk = ADCKurtosis(b = b)
    llk = (ADC = Free(start = :loglinear, lower = 0), K = Free(start = :loglinear, lower = 0, upper = 3))
    θk = (S0 = 1000.0, ADC = 1e-3, K = 0.9)
    yk = simulate(mk, θk; snr = 80, n = 1, noise = :gaussian, rng = MersenneTwister(4))[1, :]
    sk, rk = @test_logs (:warn, r"`ADC`, `K` start at :loglinear.*∂ln S/∂K changed") (start(mk, yk; llk...), fit(mk, yk; llk...))
    @test_logs min_level = Base.CoreLogging.Warn init(mk; K = Free(start = :loglinear, lower = 0, upper = 3))
    @test isfinite(sk.ADC) && 0 <= sk.K <= 3
    rd = fit(mk, yk)
    @test isconverged(rk) && isconverged(rd)
    @test rk.params.ADC ≈ rd.params.ADC rtol = 1e-6
    @test rk.params.K ≈ rd.params.K rtol = 1e-5
    # where it cannot work, an error at once rather than a fit stuck at a bound: a time
    # constant (T2 = 0 is no signal) and a signal that changes sign (real-valued IR)
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    te = collect(0.01:0.01:0.32)
    # a time constant (T2 = 0 is no signal) starts very long and is fitted through 1/T2
    # (T2Decay's own start; until 2026-11-08 refused here, the start then stuck at T2's
    # lower bound); a signal that changes sign is refused
    @test start(T2Decay(TE = te), signal(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08)); T2 = Free(start = :loglinear, lower = 1e-6)).T2 ≈ 0.08 rtol = 1e-12
    @test_logs min_level = Base.CoreLogging.Warn init(T2Decay(TE = te))   # linear in 1/T2: no warning
    mir = R1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2], magnitude = false)
    @test occursin("not positive and finite", msg(() -> fit(mir, signal(mir, (M0 = 1000.0, Sf = -0.95, R1 = 1.1)); R1 = Free(start = :loglinear))))
    # and where it suits, a rate: R2Decay's start is the log-linear fit's
    mr = R2Decay(TE = te)
    @test start(mr, signal(mr, (S0 = 1000.0, R2 = 12.5)); R2 = Free(start = :loglinear, lower = 0)).R2 ≈ 12.5 rtol = 1e-12
    # a start Free accepts, and names among the others
    @test Free(start = :loglinear).start === :loglinear
    @test occursin(":loglinear", sprint(showerror, try Free(start = :other) catch e; e end))
    # the first pass, made once per fit when the start is the same in every voxel: the same
    # starts and fits, bit for bit, as making it in each voxel; not made when the start varies
    # by voxel (a map, a `:data` start) or nothing starts at :loglinear
    mt2 = T2Decay(TE = te)
    Yt = simulate(mt2, (S0 = 1000.0, T2 = 0.06); snr = 40, n = 20, noise = :rician, rng = MersenneTwister(5))
    for T in (Float64, Float32)
        c = init(mt2; precision = T)
        c0 = EQUIP.FitCache(c.model, c.spec, c.opts, similar(c.buf), similar(c.y), c.settings, nothing)
        @test c.logpass !== nothing
        @test all(i -> EQUIP._fit_voxel(c, c.spec, Yt[i, :], i) === EQUIP._fit_voxel(c0, c0.spec, Yt[i, :], i), 1:20)
    end
    @test init(mt2; S0 = Free(start = fill(1000.0, 20))).logpass === nothing     # a map start
    @test init(R2Decay(TE = te)).logpass === nothing                             # no :loglinear
    @test init(DataStartTest([0.01, 0.02, 0.04, 0.08])).logpass === nothing        # :data
end
