# IVIM's start (docs/design.md, roadmap item 17): the segmented fit (`datastart`, the default)
# against constant starts and BestOf over D*; 10,000 voxels, truths varied (f 0.05–0.3, D*
# 0.005–0.05, D 0.5–2e-3 mm²/s), 13 b-values 0–800 s/mm², SNR 50 and 200 (Gaussian), one thread.
# Columns: µs a voxel, iterations, % converged, voxels not at the best minimum found by any start
# (SSE above the best by more than 1e-8 of Σy², or not converged), median |error| of f and D.
#   julia -t 1 --project=. bench/models/ivim_starts.jl
using EQUIP, Random, Statistics, Printf
N = 10_000; rng = MersenneTwister(5)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
m = IVIM(b = [0.0, 10, 20, 30, 50, 80, 100, 150, 200, 300, 400, 600, 800])
θ = (S0 = fill(1000.0, N), f = uu(0.05, 0.3), Dstar = lu(0.005, 0.05), D = lu(0.5e-3, 2e-3))
starts = [("segmented (default)", (;)),
          ("constant: f 0.1, D* 0.02, D 1e-3", (f = Free(start = 0.1), Dstar = Free(start = 0.02), D = Free(start = 1e-3))),
          ("BestOf D* (8, 0.003–0.1), f and D constant", (f = Free(start = 0.1), Dstar = Free(start = BestOf(exp.(range(log(0.003), log(0.1); length = 8)))), D = Free(start = 1e-3)))]
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("| SNR | start | µs/voxel | iterations | converged | not best | median error f | median error D |")
println("|---:|---|---:|---:|---:|---:|---:|---:|")
for snr in (50, 200)
    Y = simulate(m, θ; snr, noise = :gaussian, rng = MersenneTwister(snr))
    rs = [fit(m, Y; kw...) for (_, kw) in starts]
    sse = reduce(hcat, [ifelse.(isconverged.(r.status), r.sse, Inf) for r in rs]); bs = vec(minimum(sse; dims = 2))
    sy = vec(sum(abs2, Y; dims = 2))
    for ((label, kw), r) in zip(starts, rs)
        t = best(() -> fit(m, Y; kw...))
        nb = count(i -> !isconverged(r.status[i]) || r.sse[i] > bs[i] + 1e-8 * sy[i], 1:N)
        @printf("| %d | %s | %.2f | %.1f | %.1f%% | %d | %.1f%% | %.1f%% |\n", snr, label, 1e6t / N, mean(r.iterations),
                100 * count(isconverged, r.status) / N, nb, 100median(abs.(r.params.f ./ θ.f .- 1)), 100median(abs.(r.params.D ./ θ.D .- 1)))
        flush(stdout)
    end
end
