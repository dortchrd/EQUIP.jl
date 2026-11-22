# Voxels a poor mask lets in (docs/design.md, Performance notes): julia -t 1 --project=. bench/mask/classes.jl
# What a poor mask lets in: QMTSIR fits (the owner's 4-point protocol, kmf = 12.5) of tissue,
# CSF, weak-signal and background (noise only) voxels, magnitude data (Rician noise, σ = 10),
# 10,000 voxels a class, one thread: iterations, how often the 100-iteration cap is hit, time
using EQUIP, Random, Statistics, Printf
TI = [1.007, 0.010, 0.278, 0.010]; TD = [0.010, 4.174, 2.730, 0.684]
m = QMTSIR(TI = TI, TD = TD)
N = 10_000; σ = 10.0
rng = MersenneTwister(1)
uu(a, b) = a .+ (b - a) .* rand(rng, N)
base(M0f; PSR, R1f, Sf = fill(-0.95, N)) = (PSR = PSR, R1f = R1f, Sf = Sf, M0f = fill(M0f, N), kmf = fill(12.5, N),
    R1m = R1f, Sm = fill(0.83, N), Mzf_td0 = zeros(N), Mzm_td0 = zeros(N))
classes = [
 ("white matter (M0f 1000, SNR 100)", base(1000.0; PSR = uu(0.12, 0.2), R1f = uu(1.0, 1.5))),
 ("grey matter", base(1000.0; PSR = uu(0.05, 0.1), R1f = uu(0.6, 0.8))),
 ("CSF-like (PSR 0, T1 4 s)", base(1000.0; PSR = fill(1e-4, N), R1f = fill(0.25, N), Sf = fill(-1.0, N))),
 ("weak signal (M0f 3σ)", base(3σ; PSR = uu(0.05, 0.2), R1f = uu(0.6, 1.5))),
 ("background (noise only)", base(0.0; PSR = uu(0.05, 0.2), R1f = uu(0.6, 1.5))),
]
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("| voxels | µs/voxel | iterations, mean | hit the cap (100) | converged | on a bound (any parameter) |")
println("|---|---:|---:|---:|---:|---:|")
for (label, θ) in classes
    Y = simulate(m, θ; σ, noise = :rician, rng = MersenneTwister(hash(label)))
    r = fit(m, Y)
    t = best(() -> fit(m, Y))
    ob = onbound(r)
    nb = count(i -> any(k -> ob[k][i], keys(ob)), 1:N)
    @printf("| %s | %.2f | %.1f | %.1f%% | %.1f%% | %.1f%% |\n", label, 1e6t / N, mean(r.iterations),
            100 * count(==(EQUIP.MaxIterations), r.status) / N, 100 * count(isconverged, r.status) / N, 100nb / N)
end
