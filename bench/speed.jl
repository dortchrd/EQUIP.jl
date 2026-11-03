# =================================================================================================
# Speed: how fast EQUIP's main paths are, on this machine
#
#   julia -t auto --project=. bench/speed.jl
#
# Runs in the package's own environment (EQUIP and the standard library only; bench/solvers.jl,
# the comparison with other packages, has its own). Each time is the best of several runs after
# a warm-up, so compilation is not counted; "bytes" is what one call allocates (0 for the paths
# meant to allocate nothing). About 1–2 minutes with 4 threads.
#
#   1. fit!: one voxel, every built-in model (µs, bytes)
#   2. fit: a dataset of 100,000 voxels on threads (voxels per second)
#   3. crlb!: one prediction (µs, bytes), and a sweep (µs a setting)
#   4. simulate (one thread, µs a copy and bytes beyond the data) and montecarlo: 10,000 noisy
#      copies, fitted on threads
#   5. a protocol design: one evaluation of Eq. 5 (µs, bytes), and a search
#   6. fit statistics on a dataset, scenarios, leakage (where the commit has them)
#
# Written to run on older commits too (a section is skipped when its function is missing; the
# design's keyword was `vary` before `acquisition`), to check that nothing has become slower.
# =================================================================================================

using EQUIP, Printf, Random, Statistics

# The best of `reps` runs of `f()`, each run `n` calls (so that very short calls are timed well).
function best(f, n, reps = 5)
    f()
    return minimum(@elapsed(for _ in 1:n; f(); end) for _ in 1:reps) / n
end
row(label, value) = println(rpad("  " * label, 52), value)

println("EQUIP speed, Julia ", VERSION, ", ", Threads.nthreads(), " threads, ",
        Sys.cpu_info()[1].model)

# -------------------------------------------------------------------------------------------------
println("\n1. fit!: one voxel (noisy data, SNR 50; the model's default starts)")
te = collect(0.01:0.01:0.32)
ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
td = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
fa = [2.0, 5, 10, 15, 20, 30]
b = [0.0, 250, 500, 1000, 1500, 2000, 2500]
ti6 = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4]; td6 = [0.5, 3.0, 1.0, 0.2, 2.0, 0.05]
cases = (("T1IR, 7 TIs", T1IR(TI = ti), (M0 = 1000.0, Sf = -0.95, T1 = 0.9)),
         ("T1IRTD, 6 points", T1IRTD(TI = ti6, TD = td6), (M0 = 1000.0, Sf = -0.95, T1 = 0.9)),
         ("T1SR, 6 TDs", T1SR(TD = td), (M0 = 1000.0, T1 = 1.2)),
         ("T1MFA, 6 flip angles", T1MFA(FA = fa, TR = 0.015), (M0 = 1000.0, T1 = 1.2)),
         ("T2Decay, 32 echoes", T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08)),
         ("T2DecayOffset, 32 echoes", T2DecayOffset(TE = te), (S0 = 1000.0, T2 = 0.08, C = 30.0)),
         ("T2BiexpDecay, 32 echoes", T2BiexpDecay(TE = te), (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2 = 0.08)),
         ("ADCDecay, 7 b-values", ADCDecay(b = b), (S0 = 1000.0, ADC = 1e-3)),
         ("ADCKurtosis, 7 b-values", ADCKurtosis(b = b), (S0 = 1000.0, ADC = 1e-3, K = 1.0)),
         ("QMTSIR 5-point (kmf fixed)", QMTSIR(; SIR_5POINT...), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)),
         ("QMTSIR 16-point (kmf free)", QMTSIR(; SIR_16POINT...), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)))
for (label, m, θ) in cases
    kw = label == "QMTSIR 16-point (kmf free)" ? (kmf = Free(),) : (;)
    y = simulate(m, merge(defaults(m), θ); noise = :rician, snr = 50, rng = MersenneTwister(1))
    c = init(m; kw...)
    local t = best(() -> fit!(c, y), 2000)
    a = (fit!(c, y); @allocated fit!(c, y))
    row(label, @sprintf("%7.2f µs, %d bytes", 1e6t, a))
end

# -------------------------------------------------------------------------------------------------
println("\n2. fit: a dataset of 100,000 voxels on ", Threads.nthreads(), " threads (SNR 50)")
for (label, m, θ) in (cases[5], cases[1], cases[10])
    Y = simulate(m, merge(defaults(m), θ); noise = :rician, snr = 50, n = 100_000, rng = MersenneTwister(2))
    local t = best(() -> fit(m, Y), 1, 3)
    row(label, @sprintf("%6.0f ms, %5.2f M voxels/s", 1e3t, 100_000 / t / 1e6))
end

# -------------------------------------------------------------------------------------------------
println("\n3. crlb: one prediction (crlb!, a cache made once), and a sweep")
ti16 = [round.(exp10.(range(log10(0.01), log10(2), 15)) .* 1e3) ./ 1e3; 10]
θwm = nominal(QMTSIR, :SIR_WM_II)
for (label, m, kw) in (("4-point, kmf assumed 12.5", QMTSIR(; SIR_4POINT...), (kmf = 12.5,)),
                       ("5-point, kmf free", QMTSIR(; SIR_5POINT...), (kmf = Free(),)),
                       ("16-point, kmf assumed 12.5", QMTSIR(TI = ti16, TD = 2.5), (kmf = 12.5,)))
    c = CRLBCache(m, θwm; kw...)
    local t = best(() -> crlb!(c, θwm; snr = 200), 3000)
    a = (crlb!(c, θwm; snr = 200); @allocated crlb!(c, θwm; snr = 200))
    row("crlb! " * label, @sprintf("%6.2f µs, %d bytes", 1e6t, a))
end
m4 = QMTSIR(; SIR_4POINT...)
t = best(() -> crlb(m4, θwm; snr = 200, kmf = 12.5, vary = :kmf => 5.0:0.5:20.0), 20) / 31
row("crlb sweep over 31 true kmf, 4-point", @sprintf("%6.2f µs a setting", 1e6t))

# -------------------------------------------------------------------------------------------------
println("\n4. simulate (one thread) and montecarlo: 10,000 noisy copies (Rician, SNR 200), fitted on threads")
for (label, m, θ) in (("simulate, 4-point SIR", m4, nominal(m4, :SIR_WM_I)),
                      ("simulate, T2 decay, 32 echoes", T2Decay(TE = 0.01:0.01:0.32), (S0 = 1000.0, T2 = 0.08)))
    sim(n) = simulate(m, θ; noise = :rician, snr = 200, n, rng = MersenneTwister(1))
    local t = best(() -> sim(10_000), 1, 5) / 10_000
    local a = (@allocated(sim(20_000)) - @allocated(sim(10_000))) / 10_000 - 8 * length(signal(m, θ))
    row(label, @sprintf("%6.2f µs a copy, %d bytes beyond the data", 1e6t, round(Int, a)))
end
for (label, m, kw) in (("4-point, kmf assumed 12.5", m4, (kmf = 12.5,)),
                       ("5-point, kmf free", QMTSIR(; SIR_5POINT...), (kmf = Free(),)))
    local t = best(() -> montecarlo(m, nominal(m, :SIR_WM_I); noise = :rician, snr = 200, n = 10_000, kw...), 1, 3)
    row(label, @sprintf("%6.0f ms", 1e3t))
end

# -------------------------------------------------------------------------------------------------
println("\n5. protocol design: Eq. 5 over the four tissues of Dortch et al. 2018")
tissues = (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)
md = QMTSIR(; SIR_4POINT..., magnitude = false)
ss = [Scenario(QMTSIR, nominal(QMTSIR, t); kmf = 12.5) for t in tissues]
space = (TI = (0.01, 10.0), TD = (0.01, 10.0))
st(m) = sum(m.TI .+ m.TD .+ 0.3)
# The keyword for the acquisition to optimize was `vary` before it was `acquisition`.
d = try
    ProtocolDesign(md, ss; snr = 200, acquisition = space, scantime = st)
catch
    ProtocolDesign(md, ss; snr = 200, vary = space, scantime = st)
end
x = EQUIP._x(d, md)
function evaluate(d, x, n)          # inside a function, so that nothing global is timed
    v = 0.0
    for _ in 1:n
        v += d(x)
    end
    return v
end
evaluate(d, x, 1)
t = minimum(@elapsed(evaluate(d, x, 1000)) for _ in 1:5) / 1000
row("one evaluation d(x), 4-point", @sprintf("%6.2f µs, %d bytes", 1e6t, @allocated(d(x))))
t = @elapsed optimize_protocol(d; starts = Threads.nthreads(), rng = MersenneTwister(1))
row("optimize_protocol, $(Threads.nthreads()) starts on threads", @sprintf("%6.1f s", t))

# -------------------------------------------------------------------------------------------------
println("\n6. newer functions (skipped where this commit does not have them)")
if isdefined(EQUIP, :residualsd)
    m = T2Decay(TE = te)
    res = fit(m, simulate(m, (S0 = 1000.0, T2 = 0.08); noise = :rician, snr = 50, n = 100_000, rng = MersenneTwister(3)))
    t = best(() -> stderror(res), 1, 3)
    row("stderror of a 100,000-voxel T2 fit", @sprintf("%6.0f ms", 1e3t))
end
if isdefined(EQUIP, :scenarios)
    grid = (PSR = [0.05, 0.125, 0.2], kmf = [5.0, 12.5, 20.0], R1f = [0.5, 1.0, 1.5], Sf = [-1.0, -0.9, -0.8])
    t = best(() -> scenarios(QMTSIR, nominal(QMTSIR); vary = grid, kmf = 12.5), 3)
    row("scenarios, a 3⁴ grid (81)", @sprintf("%6.2f ms", 1e3t))
end
if isdefined(EQUIP, :leakage)
    m5 = QMTSIR(; SIR_5POINT...)
    t = best(() -> leakage(m5, nominal(m5, :SIR_WM_I); kmf = 12.5), 200)
    row("leakage, 5-point, kmf assumed 12.5", @sprintf("%6.1f µs", 1e6t))
end
