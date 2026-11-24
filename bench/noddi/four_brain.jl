# The four NODDI models on a whole brain's mask (docs/design.md, item 7): 128×128×60 at 30%,
# 294,912 voxels, tissue varied (vic 0.3–0.8, ODI 0.05–0.7, viso 0–0.3, 10% CSF-like, fanning
# betafraction 0–0.8 at random), Rician SNR 30, 63 images (b 1000, 2500 × 30 directions), the
# same data for each (made by BinghamNODDIConvolved); the fit only, 4 threads.
#   julia -t 4 --project=. bench/noddi/four_brain.jl
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
Y = simulate(BinghamNODDIConvolved(; acq...), θ; snr = 30, noise = :rician, rng)
println("| model | the zeppelin | one signal | whole brain | µs a voxel | iterations | converged |")
println("|---|---|---:|---:|---:|---:|---:|")
for (M, zep) in ((NODDI, "tensor averaged (Zhang)"), (NODDIConvolved, "signal averaged"),
                 (BinghamNODDI, "tensor averaged"), (BinghamNODDIConvolved, "signal averaged"))
    m = M(; acq...)
    p = merge(map(first, θ), (theta = 0.7, phi = 0.3))
    out = zeros(length(acq.b)); signal!(out, m, p)
    ts = minimum((t0 = time_ns(); for _ in 1:200; signal!(out, m, p); end; (time_ns() - t0) / 200) for _ in 1:3)
    fit(m, Y[1:100, :])
    t = @elapsed (r = fit(m, Y))
    @printf("| %s | %s | %.1f µs | %.1f s | %.0f | %.1f | %.1f%% |\n", nameof(M), zep, 1e-3ts, t, 1e6t / N, mean(r.iterations),
            100mean(isconverged.(r.status)))
    flush(stdout)
end
