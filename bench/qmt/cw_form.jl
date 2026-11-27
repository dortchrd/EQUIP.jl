# Henkelman's closed form against the 2×2 solve for the CW models' free pool (docs/design.md,
# item 6): one signal (plain numbers; the dual numbers of a fit's Jacobian, 5 partials), one
# voxel's fit (one thread), a whole brain (294,912 voxels, threads as started).
#   julia -t 4 --project=test bench/qmt/cw_form.jl
using EQUIP, ForwardDiff, Random, Printf
const D = ForwardDiff.Dual{Nothing}
dual(x, i) = D(x, ntuple(k -> Float64(k == i), 5))
off = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
models = (("QMTCWHenkelman, 13 images", QMTCWHenkelman(offset = [off; off; 0.0], B1cw = [fill(1.5, 6); fill(3.0, 6); 0.0]), 100),
          ("QMTSPGRRamani, 13 images", QMTSPGRRamani(offset = [off; off; 0.0], B1e = [fill(4.0, 6); fill(8.0, 6); 0.0], tm = 0.012,
                                                      ts = 0.002, TR = 0.030, FA = 10), 100))
N = 294912
for (lab, m, snr) in models
    θ = resolve(m, merge(nominal(m), (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1000.0, R1obs = 1.1)))
    θd = merge(θ, (PSR = dual(0.15, 1), kmf = dual(20.0, 2), T2f = dual(0.04, 3), T2m = dual(12e-6, 4), M0f = dual(1000.0, 5)))
    t(p) = (out = Vector{typeof(p.M0f * 1.0)}(undef, nobs(m)); signal!(out, m, p);
            minimum(@elapsed(for _ in 1:50000; signal!(out, m, p); end) for _ in 1:7) / 50000)
    ts, td = t(θ), t(θd)
    rng = MersenneTwister(1); u(a, b) = a .+ (b - a) .* rand(rng, N)
    tt = merge(map(x -> x isa Real ? fill(x, N) : x, nominal(m)), (PSR = u(0.05, 0.25), kmf = u(10, 40), T2f = u(0.02, 0.08),
               T2m = 1e-6 .* u(8, 16), M0f = fill(1000.0, N), R1obs = u(0.6, 1.4)))
    Y = simulate(m, tt; snr, noise = :rician, rng = MersenneTwister(2))
    c = init(m; R1obs = 1.1); fit!(c, view(Y, 1, :))
    t1 = minimum(@elapsed(for i in 1:5000; fit!(c, view(Y, i, :)); end) for _ in 1:3) / 5000
    fit(m, Y[1:100, :]; R1obs = tt.R1obs[1:100])
    tb = minimum(@elapsed(fit(m, Y; R1obs = tt.R1obs)) for _ in 1:3)
    @printf("| %s | %.0f ns | %.0f ns | %.2f µs | %.2f s |\n", lab, 1e9ts, 1e9td, 1e6t1, tb)
    flush(stdout)
end
