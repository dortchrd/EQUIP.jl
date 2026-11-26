# A whole brain on 4 threads: the pulsed-MT model (QMTSPGRYarnykh) and SIR beside the diffusion
# models, timed in one session on one machine so the numbers compare (docs/design.md, item 6).
# 128×128×60 at 30%: 294,912 voxels; tissue varied in every voxel; Rician noise. Each model fits
# data it made, from its default starts (the best of two timings, after a warm-up on 100 voxels,
# so compilation is not counted; BinghamNODDI timed once).
#   diffusion: 63 images (3 at b = 0, 30 directions at b 1000 and 2500; DKI: 1 + 30 at b 1000 and
#              2000), b = 0 SNR 30
#   QMTSPGRYarnykh: 13 images (offsets 1–32 kHz at B1e 4 and 8 µT, one without the MT pulse;
#              tm 12 ms, ts 2 ms, TR 30 ms, 10°), and 25 (12 offsets, 0.5–64 kHz, at the two
#              powers, and one without); SNR 100 in the image without the MT pulse (snr = 800
#              relative to M0f); R1obs a map
#   QMTSIR, QMTSIREqualR1: Dortch 2018's 5-point scheme, SNR 200 (relative to M0f)
#   julia -t 4 --project=test bench/qmt/brain.jl
using EQUIP, Random, Statistics, Printf, LinearAlgebra, StaticArrays
N = round(Int, 128 * 128 * 60 * 0.3)
u(rng, a, b) = a .+ (b - a) .* rand(rng, N)
rng = MersenneTwister(4)
dirs = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:N]
theta, phi = acos.(getindex.(dirs, 3)), atan.(getindex.(dirs, 2), getindex.(dirs, 1))
viso = u(rng, 0, 0.3); csf = rand(rng, N) .< 0.1; viso[csf] .= 0.9
g30 = directionset(30)
dwi = diffusionencoding([fill(0.0, 3); fill(1000.0, 30); fill(2500.0, 30)], [[SVector(0.0, 0, 0) for _ in 1:3]; g30; g30])
dki = diffusionencoding([0.0; fill(1000.0, 30); fill(2000.0, 30)], [[SVector(0.0, 0, 0)]; g30; g30])
v = (0.8e-3)^2
cases = [
    ("DTI", DTI(; dwi...), (S0 = fill(1000.0, N), Dxx = 1.7e-3 .* u(rng, 0.7, 1.3), Dyy = fill(0.4e-3, N), Dzz = fill(0.3e-3, N),
                            Dxy = fill(0.1e-3, N), Dxz = fill(-0.05e-3, N), Dyz = fill(0.02e-3, N)), 30),
    ("DKI", DKI(; dki...), merge((S0 = fill(1000.0, N), Dxx = 1.7e-3 .* u(rng, 0.7, 1.3), Dyy = fill(0.4e-3, N), Dzz = fill(0.3e-3, N),
                                  Dxy = fill(0.1e-3, N), Dxz = fill(-0.05e-3, N), Dyz = fill(0.02e-3, N)),
                                 map(x -> fill(x, N), (V1111 = v, V2222 = v, V3333 = v, V1112 = 0.0, V1113 = 0.0, V1222 = 0.0,
                                  V2223 = 0.0, V1333 = 0.0, V2333 = 0.0, V1122 = v / 3, V1133 = v / 3, V2233 = v / 3,
                                  V1123 = 0.0, V1223 = 0.0, V1233 = 0.0))), 30),
    ("StickZeppelinBall", StickZeppelinBall(; dwi...), (S0 = fill(1000.0, N), vax = u(rng, 0.4, 0.8), viso = viso,
                            Dax = 1e-3 .* u(rng, 1.5, 2.2), theta = theta, phi = phi, Diso = fill(3e-3, N)), 30),
    ("NODDI", NODDI(; dwi...), (S0 = fill(1000.0, N), vic = u(rng, 0.3, 0.8), viso = viso, ODI = u(rng, 0.05, 0.7),
                            theta = theta, phi = phi, Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N)), 30),
    ("NODDIConvolved", NODDIConvolved(; dwi...), (S0 = fill(1000.0, N), vic = u(rng, 0.3, 0.8), viso = viso, ODI = u(rng, 0.05, 0.7),
                            theta = theta, phi = phi, Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N)), 30),
    ("BinghamNODDI", BinghamNODDI(; dwi...), (S0 = fill(1000.0, N), vic = u(rng, 0.3, 0.8), viso = viso, ODI = u(rng, 0.05, 0.7),
                            betafraction = u(rng, 0, 0.8), theta = theta, phi = phi, psi = u(rng, 0, π),
                            Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N)), 30),
]
off13 = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
off25 = [0.5e3, 1e3, 1.5e3, 2e3, 3e3, 4e3, 6e3, 8e3, 12e3, 16e3, 32e3, 64e3]
mt(off) = QMTSPGRYarnykh(offset = [off; off; 0], B1e = [fill(4.0, length(off)); fill(8.0, length(off)); 0], tm = 0.012,
                         ts = 0.002, TR = 0.030, FA = 10)
R1obs = u(rng, 0.6, 1.4)
spgr = (PSR = u(rng, 0.05, 0.25), kmf = u(rng, 10, 40), T2f = u(rng, 0.02, 0.08), T2m = 1e-6 .* u(rng, 8, 16), M0f = fill(1000.0, N))
for (lab, m) in (("QMTSPGRYarnykh", mt(off13)), ("QMTSPGRYarnykh", mt(off25)))
    push!(cases, (lab, m, merge(map(x -> x isa Real ? fill(x, N) : x, nominal(m)), spgr, (R1obs = R1obs,)), 800))
end
sir = (PSR = u(rng, 0.05, 0.25), R1f = u(rng, 0.6, 1.4), Sf = u(rng, -1.0, -0.9), M0f = fill(1000.0, N))
for (lab, m) in (("QMTSIR", QMTSIR(; SIR_5POINT...)), ("QMTSIREqualR1", QMTSIREqualR1(; SIR_5POINT...)))
    push!(cases, (lab, m, merge(map(x -> x isa Real ? fill(x, N) : x, nominal(m)), sir), 200))
end
@printf("%d voxels, %d threads\n", N, Threads.nthreads())
println("| model | images | free parameters | whole brain | µs a voxel (one thread) | iterations | converged |")
println("|---|---:|---:|---:|---:|---:|---:|")
for (lab, m, θ, snr) in cases
    Y = simulate(m, θ; snr, noise = :rician, rng = MersenneTwister(9))
    kw = m isa QMTSPGRYarnykh ? (R1obs = θ.R1obs,) : (;)
    fit(m, Y[1:100, :]; map(x -> x isa AbstractVector ? x[1:100] : x, kw)...)      # warm-up: compile
    reps = lab == "BinghamNODDI" ? 1 : 2
    t = minimum(@elapsed(fit(m, Y; kw...)) for _ in 1:reps)
    r = fit(m, Y; kw...)
    nf = count(x -> x isa Free, values(parameters(m)))
    @printf("| %s | %d | %d | %.1f s | %.0f | %.1f | %.1f%% |\n", lab, nobs(m), nf, t, 1e6t * Threads.nthreads() / N,
            mean(r.iterations), 100mean(isconverged.(r.status)))
    flush(stdout)
end
