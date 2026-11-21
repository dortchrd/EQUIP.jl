# Geodesic acceleration (branch geodesic-proto) against plain LM: 10,000 voxels with varied
# truths a model, SNR 50 and 200, one thread, best of 3; voxels not best: not converged, or a sum
# of squares above the best of the settings (by more than 1e-8·Σy²)
using EQUIP, Random, Statistics, Printf
N = 10_000
rng = MersenneTwister(3)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
TE = collect(0.01:0.01:0.32)
cases = [
 ("bi-exponential", T2BiexpDecay(TE = TE), (S0 = fill(1000.0, N), fshort = uu(0.05, 0.4), T2short = lu(0.008, 0.03), T2long = lu(0.06, 0.15)), (;), :T2long),
 ("kurtosis", ADCKurtosis(b = collect(0.0:250.0:3000.0)), (S0 = fill(1000.0, N), ADC = lu(0.3e-3, 2.5e-3), K = uu(0.3, 2.0)), (;), :K),
 ("QMTSIR 5-point, kmf free", QMTSIREqualR1(; SIR_5POINT...), (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = lu(5, 20), Sm = fill(0.83, N)), (kmf = Free(),), :PSR),
 ("QMTSIR 4-point", QMTSIREqualR1(; SIR_4POINT...), (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = fill(12.5, N), Sm = fill(0.83, N)), (;), :PSR),
 ("T2, 32 echoes", T2Decay(TE = TE), (S0 = fill(1000.0, N), T2 = lu(0.02, 0.3)), (;), :T2),
 ("T1IR", T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]), (M0 = fill(1000.0, N), Sf = uu(-1.0, -0.85), T1 = lu(0.3, 4.0)), (;), :T1),
]
settings = [("plain LM", LM()), ("geodesic, α 0.5", LM(geodesic = true, alpha = 0.5)),
            ("geodesic, α 0.75", LM(geodesic = true)), ("geodesic, α 1.0", LM(geodesic = true, alpha = 1.0))]
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("| model | SNR | solver | µs/voxel | iterations | converged | not best | median error |")
println("|---|---:|---|---:|---:|---:|---:|---:|")
for (name, m, θ, kw, q) in cases, snr in (50, 200)
    Y = simulate(m, θ; snr, noise = :gaussian, rng = MersenneTwister(hash((name, snr))))
    rs = [fit(m, Y; kw..., solver = lm) for (_, lm) in settings]
    sy2 = vec(sum(abs2, Y; dims = 2))
    bestsse = vec(minimum(reduce(hcat, [ifelse.(isconverged.(r.status), r.sse, Inf) for r in rs]); dims = 2))
    for ((label, lm), r) in zip(settings, rs)
        t = best(() -> fit(m, Y; kw..., solver = lm))
        nb = count(i -> !isconverged(r.status[i]) || r.sse[i] > bestsse[i] + 1e-8 * sy2[i], 1:N)
        @printf("| %s | %d | %s | %.2f | %.1f | %.1f%% | %d | %.2f%% |\n", name, snr, label, 1e6t / N, mean(r.iterations),
                100 * count(isconverged, r.status) / N, nb, 100 * median(abs.(r.params[q] ./ θ[q] .- 1)))
        flush(stdout)
    end
end
