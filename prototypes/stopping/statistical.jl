# The statistical stop (branch stopping-proto, LM(ctol = c)) against the defaults and uniformly
# looser tolerances, on fits with more observations than parameters. As tolerances.jl: The same data fitted with a very tight
# reference, the defaults, and looser settings; for each, how far the estimates move from the
# reference in units of their statistical SD (the spread of the reference's errors over the
# voxels), and what it saves. 10,000 voxels with varied truths, one thread.
using EQUIP, Random, Statistics, Printf
N = 10_000
rng = MersenneTwister(3)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
TE = collect(0.01:0.01:0.32)
cases = [
 ("T2, 32 echoes", T2Decay(TE = TE), (S0 = fill(1000.0, N), T2 = lu(0.02, 0.3)), (;), (:T2,)),
 ("bi-exponential", T2BiexpDecay(TE = TE), (S0 = fill(1000.0, N), fshort = uu(0.05, 0.4), T2short = lu(0.008, 0.03), T2long = lu(0.06, 0.15)), (;), (:fshort, :T2long)),
 ("kurtosis", ADCKurtosis(b = collect(0.0:250.0:3000.0)), (S0 = fill(1000.0, N), ADC = lu(0.3e-3, 2.5e-3), K = uu(0.3, 2.0)), (;), (:ADC, :K)),
 ("T1IR", T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]), (M0 = fill(1000.0, N), Sf = uu(-1.0, -0.85), T1 = lu(0.3, 4.0)), (;), (:T1,)),
 ("QMTSIR 16-point, kmf free", QMTSIREqualR1(; SIR_16POINT...), (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = lu(5, 20), Sm = fill(0.83, N)), (kmf = Free(),), (:PSR, :kmf)),
 ("QMTSIR 5-point, kmf fixed", QMTSIREqualR1(; SIR_5POINT...), (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = fill(12.5, N), Sm = fill(0.83, N)), (;), (:PSR, :R1f)),
]
settings = [("reference", LM(maxiter = 500, xtol = 1e-14, ftol = 1e-16, gtol = 1e-15)),
            ("default", LM()),
            ("statistical, c = 0.01", LM(ctol = 0.01)),
            ("statistical, c = 0.03", LM(ctol = 0.03)),
            ("statistical, c = 0.1", LM(ctol = 0.1)),
            ("uniform 1e-5, 1e-8, 1e-6", LM(xtol = 1e-5, ftol = 1e-8, gtol = 1e-6))]
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("| model | SNR | stopping | µs/voxel | iterations | converged | stopped by x / f / g or statistical | largest median and 99.9th pct |Δ|/SD |")
println("|---|---:|---|---:|---:|---:|---|---|")
for (name, m, θ, kw, qs) in cases, snr in (50, 200)
    Y = simulate(m, θ; snr, noise = :gaussian, rng = MersenneTwister(hash((name, snr))))
    ref = fit(m, Y; kw..., solver = settings[1][2])
    ok0 = isconverged.(ref.status)
    sd = Dict(q => std((ref.params[q] .- θ[q])[ok0]) for q in qs)            # each one's statistical SD
    for (label, lm) in settings
        r = label == "reference" ? ref : fit(m, Y; kw..., solver = lm)
        t = best(() -> fit(m, Y; kw..., solver = lm))
        ok = ok0 .& isconverged.(r.status)
        dev = [abs.((r.params[q] .- ref.params[q])[ok]) ./ sd[q] for q in qs]
        med = maximum(median.(dev)); p999 = maximum(quantile.(dev, 0.999))
        st = r.status
        @printf("| %s | %d | %s | %.2f | %.1f | %.1f%% | %.0f / %.0f / %.0f %% | %.1e, %.1e |\n", name, snr, label, 1e6t / N,
                mean(r.iterations), 100 * count(isconverged, st) / N, 100 * count(==(EQUIP.ConvergedX), st) / N,
                100 * count(==(EQUIP.ConvergedF), st) / N, 100 * count(==(EQUIP.ConvergedG), st) / N, med, p999)
        flush(stdout)
    end
end
