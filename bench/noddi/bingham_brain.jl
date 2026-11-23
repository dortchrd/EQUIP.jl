# Bingham–NODDI on a whole brain's mask (docs/design.md, item 7): bench/noddi/brain.jl's 294,912
# voxels (tissue varied, 10% CSF-like), with fanning varied too (betafraction 0–0.8, the fan at
# random), Rician SNR 30, 63 images (b 1000, 2500); NODDI (Watson) on the same data for comparison.
#   julia -t 4 --project=. bench/noddi/bingham_brain.jl
using EQUIP, Random, Statistics, Printf, LinearAlgebra, StaticArrays
N = round(Int, 128 * 128 * 60 * 0.3)
rng = MersenneTwister(4)
g = directionset(30)
acq = diffusionencoding([fill(0.0, 3); fill(1000.0, 30); fill(2500.0, 30)], reduce(hcat, [fill([0.0, 0.0, 0.0], 3); collect.(g); collect.(g)]))
dirs = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:N]
viso = 0.3 .* rand(rng, N); csf = rand(rng, N) .< 0.1; viso[csf] .= 0.9
θ = (S0 = fill(1000.0, N), vic = 0.3 .+ 0.5 .* rand(rng, N), viso = viso, ODI = 0.05 .+ 0.65 .* rand(rng, N),
     betafraction = 0.8 .* rand(rng, N), theta = acos.(getindex.(dirs, 3)), phi = atan.(getindex.(dirs, 2), getindex.(dirs, 1)),
     psi = π .* rand(rng, N), Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N))
Y = simulate(BinghamNODDI(; acq...), θ; snr = 30, noise = :rician, rng)
for M in (BinghamNODDI, NODDI)
    m = M(; acq...)
    fit(m, Y[1:100, :])
    t = @elapsed (r = fit(m, Y))
    @printf("%s: %d voxels in %.1f s (%d threads), %.0f µs a voxel; iterations %.1f, converged %.1f%%\n", nameof(M), N, t,
            Threads.nthreads(), 1e6t / N, mean(r.iterations), 100mean(isconverged.(r.status)))
end
