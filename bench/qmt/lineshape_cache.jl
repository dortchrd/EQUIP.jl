# QMTSPGRYarnykh's lineshapes made once per distinct offset (a cache in signal!, docs/design.md
# item 6): one signal (plain numbers, and the dual numbers of a fit's Jacobian, 5 partials), one
# voxel's fit (one thread), a whole brain (294,912 voxels, threads as started), on protocols
# with each offset at two MT powers, and with every offset distinct (where the cache only costs).
#   julia -t 4 --project=test bench/qmt/lineshape_cache.jl
using EQUIP, ForwardDiff, Random, Printf
off6 = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
off12 = [0.5e3, 1e3, 1.5e3, 2e3, 3e3, 4e3, 6e3, 8e3, 12e3, 16e3, 32e3, 64e3]
off20 = collect(range(0.5e3, 60e3; length = 20))
protos = (("13 images, 6 offsets × 2 powers", [off6; off6; 0], [fill(4.0, 6); fill(8.0, 6); 0]),
          ("25 images, 12 offsets × 2 powers", [off12; off12; 0], [fill(4.0, 12); fill(8.0, 12); 0]),
          ("41 images, 20 offsets × 2 powers", [off20; off20; 0], [fill(4.0, 20); fill(8.0, 20); 0]),
          ("13 images, 12 offsets distinct", [collect(range(1e3, 40e3; length = 12)); 0], [fill(6.0, 12); 0]))
const D = ForwardDiff.Dual{Nothing}
dual(x, i) = D(x, ntuple(k -> Float64(k == i), 5))
N = 294912
println("| protocol | one signal | one signal, dual | one voxel's fit | whole brain ($(Threads.nthreads()) threads) |")
println("|---|---:|---:|---:|---:|")
for (lab, off, b1e) in protos
    m = QMTSPGRYarnykh(offset = off, B1e = b1e, tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
    θ = resolve(m, merge(nominal(m), (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1000.0, R1obs = 1.1)))
    θd = merge(θ, (PSR = dual(0.15, 1), kmf = dual(20.0, 2), T2f = dual(0.04, 3), T2m = dual(12e-6, 4), M0f = dual(1000.0, 5)))
    t(p) = (out = Vector{typeof(p.M0f * 1.0)}(undef, nobs(m)); signal!(out, m, p);
            minimum(@elapsed(for _ in 1:20000; signal!(out, m, p); end) for _ in 1:7) / 20000)
    ts, td = t(θ), t(θd)
    rng = MersenneTwister(1); u(a, b) = a .+ (b - a) .* rand(rng, N)
    tt = merge(map(x -> x isa Real ? fill(x, N) : x, nominal(m)), (PSR = u(0.05, 0.25), kmf = u(10, 40), T2f = u(0.02, 0.08),
               T2m = 1e-6 .* u(8, 16), M0f = fill(1000.0, N), R1obs = u(0.6, 1.4)))
    Y = simulate(m, tt; snr = 800, noise = :rician, rng = MersenneTwister(2))
    c = init(m; R1obs = 1.1); fit!(c, view(Y, 1, :))
    t1 = minimum(@elapsed(for i in 1:2000; fit!(c, view(Y, i, :)); end) for _ in 1:3) / 2000
    fit(m, Y[1:100, :]; R1obs = tt.R1obs[1:100])
    tb = minimum(@elapsed(fit(m, Y; R1obs = tt.R1obs)) for _ in 1:2)
    @printf("| %s | %.2f µs | %.2f µs | %.1f µs | %.1f s |\n", lab, 1e6ts, 1e6td, 1e6t1, tb)
    flush(stdout)
end
