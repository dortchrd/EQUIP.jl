# =================================================================================================
# Precompile workload: compile the common cases while the package is installed, not on first use
#
# Julia compiles a function the first time it is called with new argument types. For a fit that
# means every combination of model, free parameters, precision and data shape compiles once per
# session (a few seconds for the first fit, more than the fit itself for small datasets). The
# workload below runs small fits of every built-in model while EQUIP precompiles, and since Julia
# 1.9 the compiled code is saved with the package (PrecompileTools.jl), so a new session starts
# with it. Combinations not run here (other free parameters, Float32, ...) still compile on first
# use. Since 2026-11-08 also DTI (fits and derived quantities; not DKI, see below), and crlb,
# montecarlo and a protocol design as the README calls them (the SIR schemes, T2 decay). Measurements:
# docs/design.md (Performance notes; "The prerelease: the precompile workload").
# =================================================================================================

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Code here runs during precompilation but is not itself cached: the inputs.
    te = collect(0.01:0.01:0.32)
    ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    td = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
    fa = [2.0, 5, 10, 15, 20, 30]
    b = [0.0, 250, 500, 1000, 1500, 2000, 2500]
    ti6 = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4]; td6 = [0.5, 3.0, 1.0, 0.2, 2.0, 0.05]
    cases = ((T1IR(TI = ti), (M0 = 1000.0, Sf = -0.95, T1 = 0.9)),
             (R1IR(TI = ti), (M0 = 1000.0, Sf = -0.95, R1 = 1.1)),
             (T1IRTD(TI = ti6, TD = td6), (M0 = 1000.0, Sf = -0.95, T1 = 0.9)),
             (R1IRTD(TI = ti6, TD = td6), (M0 = 1000.0, Sf = -0.95, R1 = 1.1)),
             (T1SR(TD = td), (M0 = 1000.0, T1 = 1.2)),
             (R1SR(TD = td), (M0 = 1000.0, R1 = 0.8)),
             (T1MFA(FA = fa, TR = 0.015), (M0 = 1000.0, T1 = 1.2)),
             (R1MFA(FA = fa, TR = 0.015), (M0 = 1000.0, R1 = 0.8)),
             (T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08)),
             (R2Decay(TE = te), (S0 = 1000.0, R2 = 12.5)),
             (T2DecayOffset(TE = te), (S0 = 1000.0, T2 = 0.08, C = 30.0)),
             (R2DecayOffset(TE = te), (S0 = 1000.0, R2 = 12.5, C = 30.0)),
             (T2BiexpDecay(TE = te), (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08)),
             (R2BiexpDecay(TE = te), (S0 = 1000.0, fshort = 0.15, R2short = 50.0, R2long = 12.5)),
             (ADCDecay(b = b), (S0 = 1000.0, ADC = 1e-3)),
             (ADCKurtosis(b = b), (S0 = 1000.0, ADC = 1e-3, K = 1.0)),
             (IVIM(b = [0.0, 10, 20, 50, 100, 200, 400, 600, 800]), (S0 = 1000.0, f = 0.1, Dstar = 0.02, D = 1e-3)),
             (QMTSIR(; SIR_5POINT...), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)),
             (QMTSIREqualR1(; SIR_5POINT...), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)))
    dirs = directionset(15)
    dwib = [0.0; fill(1000.0, 15); fill(2000.0, 15)]
    dwig = reduce(hcat, [[[0.0, 0.0, 0.0]]; collect.(dirs); collect.(dirs)])   # 3 × 31, a .bvec
    dti = (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0)
    @compile_workload begin
        # Code here runs during precompilation, and what it compiles is saved with the package.
        for (m, θ) in cases
            y = signal(m, merge(defaults(m), θ))
            fit(m, y)                                        # one voxel
            Y = permutedims(hcat(y, 0.9 .* y))               # a dataset: 2 voxels × observations
            res = fit(m, Y)
            fit(m, Y; mask = [true, false])                  # with a mask
            fit(m, reshape(Y, 2, 1, 1, :))                   # 4-D data (X×Y×Z×observations)
            simulate(m, merge(defaults(m), θ); noise = :rician, snr = 50, n = 2)
            show(IOBuffer(), MIME"text/plain"(), res)        # the summary the REPL prints
        end
        # Common variations: QMTSIR with kmf estimated, T1MFA with a B1 map.
        m = QMTSIR(; SIR_5POINT...)
        y = signal(m, merge(defaults(m), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)))
        fit(m, permutedims(hcat(y, y)); kmf = Free())
        m = T1MFA(FA = fa, TR = 0.015)
        y = signal(m, merge(defaults(m), (M0 = 1000.0, T1 = 1.2)))
        fit(m, permutedims(hcat(y, y)); B1 = [1.0, 0.9])

        # DTI (added 2026-11-08): a voxel, a dataset, its derived quantities. The acquisition: a
        # b = 0 image and two shells of 15 directions; D's eigenvalues 1.7, 0.4, 0.3 × 10⁻³ mm²/s.
        # Not DKI: its fit (22 parameters) ran 20–70% slower from the package image than
        # compiled at first use (measured 2026-11-08, cause not found; DTI, T2Decay, T1IR,
        # QMTSIR, crlb, montecarlo and the design unchanged), so it compiles once per session
        # (about 6 s) and runs at full speed. The command line's workload does include it: each
        # `equip` call is a new process (docs/design.md, "The prerelease").
        m = DTI(; diffusionencoding(dwib, dwig)...)
        y = signal(m, dti)
        fit(m, y)
        res = fit(m, permutedims(hcat(y, 0.9 .* y)))
        derived(res)
        show(IOBuffer(), MIME"text/plain"(), res)
        # the multi-compartment models (added 2026-11-23), on the same acquisition, the fiber along x
        for (M, θ) in ((StickZeppelin, (S0 = 1000.0, vax = 0.6, Dax = 1.9e-3, theta = π / 2, phi = 0.0)),
                       (BallStick, (S0 = 1000.0, f = 0.5, d = 1.2e-3, theta = π / 2, phi = 0.0)),
                       (NODDI, (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, theta = π / 2, phi = 0.0)),
                       (BinghamNODDI, (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, betafraction = 0.5, theta = π / 2, phi = 0.0, psi = 0.5)),
                       (NODDISignal, (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, theta = π / 2, phi = 0.0)),
                       (BinghamNODDISignal, (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, betafraction = 0.5, theta = π / 2, phi = 0.0, psi = 0.5)))
            m = M(; diffusionencoding(dwib, dwig)...)
            y = signal(m, merge(defaults(m), θ))
            fit(m, y)
            res = fit(m, permutedims(hcat(y, 0.9 .* y)))
            derived(res)
        end

        # crlb, montecarlo and a protocol design (added 2026-11-08), as the README calls them:
        # the SIR schemes with the fit assuming kmf (Dortch 2018), T2 decay at an SNR. Each
        # model's crlb compiles its own second derivatives, so other models compile at first use.
        m4 = QMTSIR(; SIR_4POINT...)
        θ = nominal(QMTSIR, :SIR_WM_I)
        s = Scenario(QMTSIR, θ; kmf = 12.5)
        c = crlb(m4, s; snr = 200)
        show(IOBuffer(), MIME"text/plain"(), c)
        show(IOBuffer(), MIME"text/plain"(), s)
        crlb(m4, θ; snr = 200)
        crlb(m4, θ; snr = [100, 200], kmf = 12.5)                       # a sweep
        mc = montecarlo(m4, s; noise = :rician, snr = 200, n = 4)
        show(IOBuffer(), MIME"text/plain"(), mc)
        m2 = T2Decay(TE = te)
        show(IOBuffer(), MIME"text/plain"(), crlb(m2, (S0 = 1000.0, T2 = 0.08); snr = 100))
        montecarlo(m2, (S0 = 1000.0, T2 = 0.08); noise = :gaussian, snr = 100, n = 4)
        ss = scenarios(QMTSIR, [:SIR_WM_I, :SIR_lesion]; kmf = 12.5)
        mr = QMTSIR(; SIR_4POINT..., magnitude = false)
        d = ProtocolDesign(mr, ss; snr = 200, acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),
                           constraints = m -> m.TI .+ m.TD .- 0.5, scantime = m -> sum(m.TI .+ m.TD .+ 0.3))
        d(mr)
        design(d; starts = 1, generations = 2, polish = false, rng = Xoshiro(1))
    end
end
