# NODDI and NODDIConvolved on a whole brain's mask, before and after the stick's series by a table
# in β (docs/design.md, roadmap item 27, "Shells"): 128×128×60 at 30%, 294,912 voxels, tissue
# varied (vic 0.3–0.8, ODI 0.05–0.7, viso 0–0.3, 10% CSF-like), Rician SNR 30, 63 images (3 at
# b = 0, 30 directions at b 1000 and 2500): the b-values exact, and as scanners write them (each
# direction's b within ±5 of 1000, ±10 of 2500). Each model fits data it made; 4 threads.
#   julia -t 4 --project=. bench/noddi/brain_shells.jl
using EQUIP, Random, Statistics, Printf, LinearAlgebra, StaticArrays
N = round(Int, 128 * 128 * 60 * 0.3)
rng = MersenneTwister(4)
g = directionset(30)
G = [[SVector(0.0, 0, 0) for _ in 1:3]; g; g]
bexact = [fill(0.0, 3); fill(1000.0, 30); fill(2500.0, 30)]
bjit = [fill(0.0, 3); 1000 .+ rand(MersenneTwister(1), -5:5, 30); 2500 .+ rand(MersenneTwister(2), -10:10, 30)]
dirs = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:N]
viso = 0.3 .* rand(rng, N); csf = rand(rng, N) .< 0.1; viso[csf] .= 0.9
θ = (S0 = fill(1000.0, N), vic = 0.3 .+ 0.5 .* rand(rng, N), viso = viso, ODI = 0.05 .+ 0.65 .* rand(rng, N),
     theta = acos.(getindex.(dirs, 3)), phi = atan.(getindex.(dirs, 2), getindex.(dirs, 1)),
     Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N))
println("| model | b-values | whole brain | µs a voxel | iterations | converged |")
println("|---|---|---:|---:|---:|---:|")
for M in (NODDI, NODDIConvolved), (lab, b) in (("exact", bexact), ("as scanners write them", bjit))
    m = M(; diffusionencoding(b, G)...)
    Y = simulate(m, θ; snr = 30, noise = :rician, rng = MersenneTwister(9))
    fit(m, Y[1:100, :])
    t = minimum(@elapsed(r = fit(m, Y)) for _ in 1:2)
    r = fit(m, Y)
    @printf("| %s | %s | %.1f s | %.0f | %.1f | %.1f%% |\n", nameof(M), lab, t, 1e6t / N, mean(r.iterations),
            100mean(isconverged.(r.status)))
    flush(stdout)
end
