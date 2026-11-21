# QMTSIREqualR1 (docs/design.md, item 6): julia -t 1 --project=. bench/qmtsir_equalr1/fits.jl (about 1 minute)
# Whole fits, QMTSIR (defaults: R1m tied, Mz = 0 at the start of TD) against QMTSIREqualR1: 10,000
# voxels with varied truths, Gaussian noise, one thread, best of 3; the maps compared
using EQUIP, Random, Statistics, Printf
N = 10_000; rng = MersenneTwister(5)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
θ = (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = lu(5, 20), Sm = fill(0.83, N))
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("| scheme | kmf | SNR | QMTSIR µs/voxel | QMTSIREqualR1 µs/voxel | ratio | iterations | largest PSR difference (converged in both) |")
println("|---|---|---:|---:|---:|---:|---|---:|")
for (scheme, acq, kw) in (("4-point", SIR_4POINT, (;)), ("5-point", SIR_5POINT, (kmf = Free(),)))
    q, s = QMTSIR(; acq...), QMTSIREqualR1(; acq...)
    for snr in (50, 200)
        Y = simulate(s, θ; snr, noise = :gaussian, rng = MersenneTwister(snr))
        tq = best(() -> fit(q, Y; kw...)); ts = best(() -> fit(s, Y; kw...))
        rq, rs = fit(q, Y; kw...), fit(s, Y; kw...)
        ok = isconverged.(rq.status) .& isconverged.(rs.status)
        d = maximum(abs.(rq.params.PSR[ok] .- rs.params.PSR[ok]))
        @printf("| %s | %s | %d | %.2f | %.2f | %.2f | %.1f / %.1f | %.1e |\n", scheme, isempty(kw) ? "12.5" : "free", snr,
                1e6tq / N, 1e6ts / N, ts / tq, mean(rq.iterations), mean(rs.iterations), d)
    end
end
