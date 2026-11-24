# julia -t 4 --project=. bench/noddi/averaging.jl  (docs/design.md, item 7, "Which averaging")
# NODDI's extra-neurite space averaged two ways: the tensor (Zhang 2012, the built-in) or the
# signal (dmipy; the Standard Model's convolution). How far the signals differ, and what fitting
# one to data made by the other does to vic, ODI and viso (SNR 100, 2000 voxels, 4 threads).
using EQUIP, Random, Statistics, Printf, LinearAlgebra, StaticArrays
# the signal-averaged one is built in: NODDIConvolved (first written here, as NODDISignal)
g = directionset(30)
acq = diffusionencoding([fill(0.0, 3); fill(1000.0, 30); fill(2500.0, 30)], reduce(hcat, [fill([0.0, 0.0, 0.0], 3); collect.(g); collect.(g)]))
mt, ms = NODDI(; acq...), NODDIConvolved(; acq...)
println("the extra-neurite signal alone (vic 0.5, viso 0), largest difference over the 63 images, relative to the signal:")
for odi in (0.05, 0.15, 0.3, 0.5, 0.8)
    p = (S0 = 1.0, vic = 0.5, viso = 0.0, ODI = odi, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3e-3)
    a, c = signal(mt, p), signal(ms, p)
    @printf("   ODI %.2f: whole signal %.2f%% (b 1000 shell %.2f%%, b 2500 %.2f%%)\n", odi, 100maximum(abs.(c ./ a .- 1)),
            100maximum(abs.(c[4:33] ./ a[4:33] .- 1)), 100maximum(abs.(c[34:63] ./ a[34:63] .- 1)))
end
@printf("   (b = 0: equal; the signal average is the larger, as Jensen's inequality says: %s)\n",
        all(signal(ms, (S0 = 1.0, vic = 0.5, viso = 0.0, ODI = 0.3, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3e-3)) .>=
            signal(mt, (S0 = 1.0, vic = 0.5, viso = 0.0, ODI = 0.3, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3e-3)) .- 1e-12))
rng = MersenneTwister(5); nv = 2000
dirs = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:nv]
println("\nfitted with the wrong averaging (SNR 100): mean estimate (truth), and the bias in SDs of the matched fit")
for (label, vic, viso, odi) in (("white matter", 0.6, 0.05, 0.15), ("grey matter", 0.35, 0.1, 0.45))
    tr = (S0 = fill(1000.0, nv), vic = fill(vic, nv), viso = fill(viso, nv), ODI = fill(odi, nv),
          theta = acos.(getindex.(dirs, 3)), phi = atan.(getindex.(dirs, 2), getindex.(dirs, 1)), Dpar = fill(1.7e-3, nv), Diso = fill(3e-3, nv))
    for (gen, fitm, glab, flab) in ((ms, mt, "signal-averaged", "tensor-averaged (NODDI)"), (mt, ms, "tensor-averaged", "signal-averaged"))
        Y = simulate(gen, tr; snr = 100, noise = :rician, rng)
        rw, rr = fit(fitm, Y), fit(gen, Y)
        @printf("%s, data %s, fitted %s:", label, glab, flab)
        for k in (:vic, :ODI, :viso)
            @printf("  %s %.3f (%.2f; %+.1f SD)", k, mean(rw.params[k]), tr[k][1], (mean(rw.params[k]) - mean(rr.params[k])) / std(rr.params[k]))
        end
        @printf("; SSE %.2f× the matched fit's\n", mean(rw.sse) / mean(rr.sse))
    end
end
