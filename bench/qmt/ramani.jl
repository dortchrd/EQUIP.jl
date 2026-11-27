# Ramani's continuous-wave power equivalent (QMTSPGRRamani) against Yarnykh's pulsed model
# (QMTSPGRYarnykh), on the same acquisition (docs/design.md, item 6): the bias of fitting
# Yarnykh's noise-free signals with Ramani's (white matter, kmf 10, 20, 40, excitation 10° and
# 5°), their precision (crlb, MT-off SNR 100), and their speed (one signal, one voxel's fit, a
# whole brain of 294,912 voxels on the threads started).
#   julia -t 4 --project=test bench/qmt/ramani.jl
using EQUIP, Printf, Random
off = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
acq(FA) = (offset = [off; off; 0.0], B1e = [fill(4.0, 6); fill(8.0, 6); 0.0], tm = 0.012, ts = 0.002, TR = 0.030, FA = FA)
println("| excitation | kmf | PSR bias | kmf bias | T2m bias | T2f bias |")
println("|---|---:|---:|---:|---:|---:|")
for FA in (10, 5), kmf in (10.0, 20.0, 40.0)
    my, mr = QMTSPGRYarnykh(; acq(FA)...), QMTSPGRRamani(; acq(FA)...)
    θ = merge(nominal(my), (PSR = 0.15, kmf = kmf, T2f = 0.04, T2m = 12e-6, M0f = 1000.0, R1obs = 1.1))
    r = fit(mr, signal(my, θ); R1obs = 1.1)
    @printf("| %d° | %.0f | %+.1f%% | %+.1f%% | %+.1f%% | %+.1f%% |\n", FA, kmf, 100(r.params.PSR / 0.15 - 1),
            100(r.params.kmf / kmf - 1), 100(r.params.T2m / 12e-6 - 1), 100(r.params.T2f / 0.04 - 1))
end
my, mr = QMTSPGRYarnykh(; acq(10)...), QMTSPGRRamani(; acq(10)...)
θ = merge(nominal(my), (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1000.0, R1obs = 1.1))
# the same noise in the image without the MT pulse (SNR 100): Yarnykh's images are M0f·sin α·Mzf,
# Ramani's M0f·Mzf (the excitation left out), so snr (relative to M0f) is 800 for one, 100 for the other
cy, cr = crlb(my, θ; snr = 800), crlb(mr, θ; snr = 100)
@printf("crlb, MT-off SNR 100: PSR SD %.1f%% (Yarnykh) and %.1f%% (Ramani, its own data); kmf %.1f%% and %.1f%%\n",
        100cy.relsd.PSR, 100cr.relsd.PSR, 100cy.relsd.kmf, 100cr.relsd.kmf)
N = 294912
rng = MersenneTwister(1); u(a, b) = a .+ (b - a) .* rand(rng, N)
tt = merge(map(x -> x isa Real ? fill(x, N) : x, nominal(my)), (PSR = u(0.05, 0.25), kmf = u(10, 40), T2f = u(0.02, 0.08),
           T2m = 1e-6 .* u(8, 16), M0f = fill(1000.0, N), R1obs = u(0.6, 1.4)))
for (lab, m, snr) in (("QMTSPGRYarnykh", my, 800), ("QMTSPGRRamani", mr, 100))
    θr = resolve(m, θ); out = zeros(nobs(m)); signal!(out, m, θr)
    ts = minimum(@elapsed(for _ in 1:20000; signal!(out, m, θr); end) for _ in 1:5) / 20000
    Y = simulate(m, tt; snr, noise = :rician, rng = MersenneTwister(2))
    c = init(m; R1obs = 1.1); fit!(c, view(Y, 1, :))
    t1 = minimum(@elapsed(for i in 1:2000; fit!(c, view(Y, i, :)); end) for _ in 1:3) / 2000
    fit(m, Y[1:100, :]; R1obs = tt.R1obs[1:100])
    tb = minimum(@elapsed(r = fit(m, Y; R1obs = tt.R1obs)) for _ in 1:2)
    r = fit(m, Y; R1obs = tt.R1obs)
    @printf("%s: one signal %.2f µs, one voxel's fit %.1f µs, a whole brain %.1f s (%d threads), converged %.1f%%, %.1f iterations\n",
            lab, 1e6ts, 1e6t1, tb, Threads.nthreads(), 100count(isconverged, r.status) / N, sum(r.iterations) / N)
end
