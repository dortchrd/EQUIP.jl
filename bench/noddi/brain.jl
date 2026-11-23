# NODDI on a whole brain's mask, every version of bench/noddi/options.jl (docs/design.md, item 7):
# 128×128×60 at 30%, 294,912 voxels, tissue varied (vic 0.3–0.8, ODI 0.05–0.7, viso 0–0.3; 10%
# CSF-like, viso 0.9), orientations random, Rician SNR 30, two protocols; the fit only (no files).
#   LMAX=30 julia -t 4 --project=. bench/noddi/brain.jl
include(joinpath(@__DIR__, "options.jl"))           # the versions
N = round(Int, 128 * 128 * 60 * 0.3)
println("\n| version | protocol | fit, s | µs a voxel | iterations | converged |")
println("|---|---|---:|---:|---:|---:|")
for (plabel, bv, G) in (let g = directionset(30)
                            ("63 images: b 1000, 2500 × 30", [fill(0.0, 3); fill(1000.0, 30); fill(2500.0, 30)], [fill([0.0, 0.0, 0.0], 3); collect.(g); collect.(g)])
                        end,
                        let g30 = directionset(30), g60 = directionset(60)
                            ("99 images: b 700 × 30, 2000 × 60 (Zhang 2012)", [fill(0.0, 9); fill(700.0, 30); fill(2000.0, 60)], [fill([0.0, 0.0, 0.0], 9); collect.(g30); collect.(g60)])
                        end)
    acqp = diffusionencoding(bv, reduce(hcat, G))
    rngb = MersenneTwister(4)
    dirsb = [normalize(SVector{3}(randn(rngb, 3))) for _ in 1:N]
    viso = 0.3 .* rand(rngb, N); csf = rand(rngb, N) .< 0.1; viso[csf] .= 0.9
    θ = (S0 = fill(1000.0, N), vic = 0.3 .+ 0.5 .* rand(rngb, N), viso = viso, ODI = 0.05 .+ 0.65 .* rand(rngb, N),
         theta = acos.(getindex.(dirsb, 3)), phi = atan.(getindex.(dirsb, 2), getindex.(dirsb, 1)), Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N))
    Yb = simulate(NODDI(; acqp...), θ; snr = 30, noise = :rician, rng = rngb)
    for (label, M) in models
        mb = M(; acqp...)
        fit(mb, Yb[1:100, :])
        t = @elapsed (r = fit(mb, Yb))
        @printf("| %s | %s | %.1f | %.0f | %.1f | %.1f%% |\n", label, plabel, t, 1e6t / N, mean(r.iterations), 100mean(isconverged.(r.status)))
        flush(stdout)
    end
end
