# =================================================================================================
# Precompile workload: compile the common cases while the package is installed, not on first use
#
# Julia compiles a function the first time it is called with new argument types. For a fit that
# means every combination of model, free parameters, precision and data shape compiles once per
# session (a few seconds for the first fit, more than the fit itself for small datasets). The
# workload below runs small fits of every built-in model while EQUIP precompiles, and since Julia
# 1.9 the compiled code is saved with the package (PrecompileTools.jl), so a new session starts
# with it. Combinations not run here (other free parameters, Float32, ...) still compile on first
# use. Measurements: docs/design.md (Performance notes).
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
             (QMTSIR(; SIR_5POINT...), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)))
    @compile_workload begin
        # Code here runs during precompilation, and what it compiles is saved with the package.
        for (m, θ) in cases
            y = signal(m, merge(defaults(m), θ))
            fit(m, y)                                        # one voxel
            Y = permutedims(hcat(y, 0.9 .* y))               # a dataset: 2 voxels × observations
            res = fit(m, Y)
            fit(m, Y; mask = [true, false])                  # with a mask
            fit(m, reshape(Y, 2, 1, 1, :))                   # 4-D data (X×Y×Z×observations)
            simulate(m, merge(defaults(m), θ); snr = 50, n = 2)
            show(IOBuffer(), MIME"text/plain"(), res)        # the summary the REPL prints
        end
        # Common variations: QMTSIR with kmf estimated, T1MFA with a B1 map.
        m = QMTSIR(; SIR_5POINT...)
        y = signal(m, merge(defaults(m), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)))
        fit(m, permutedims(hcat(y, y)); kmf = Free())
        m = T1MFA(FA = fa, TR = 0.015)
        y = signal(m, merge(defaults(m), (M0 = 1000.0, T1 = 1.2)))
        fit(m, permutedims(hcat(y, y)); B1 = [1.0, 0.9])
    end
end
