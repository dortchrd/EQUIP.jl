# GPU fitting's precision (docs/design.md, item 12): julia -t 1 --project=. bench/simd/float32.jl
# Float32 against Float64, QMTSIREqualR1 (CPU, one thread): convergence, the PSR difference
# relative to its statistical SD (the spread of the Float64 estimates around the truth), speed
using EQUIP, Random, Statistics, Printf
N = 10_000; rng = MersenneTwister(5)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
θ = (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = lu(5, 20), Sm = fill(0.83, N))
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("| scheme | SNR | converged F64 / F32 | PSR: median and largest |F32 − F64| / SD | µs/voxel F64 / F32 |")
println("|---|---:|---|---|---|")
for (scheme, acq, kw) in (("4-point, kmf fixed", SIR_4POINT, (;)), ("5-point, kmf free", SIR_5POINT, (kmf = Free(lower = 0),)))
    m = QMTSIREqualR1(; acq...)
    for snr in (50, 200)
        Y = simulate(m, θ; snr, noise = :gaussian, rng = MersenneTwister(snr))
        a = fit(m, Y; kw...); b = fit(m, Y; precision = Float32, kw...)
        ok = isconverged.(a.status) .& isconverged.(b.status)
        sd = std(a.params.PSR[ok] .- θ.PSR[ok])
        d = abs.(Float64.(b.params.PSR[ok]) .- a.params.PSR[ok]) ./ sd
        ta = best(() -> fit(m, Y; kw...)); tb = best(() -> fit(m, Y; precision = Float32, kw...))
        @printf("| %s | %d | %d / %d | %.1e, %.2f | %.2f / %.2f |\n", scheme, snr, count(isconverged, a.status),
                count(isconverged, b.status), median(d), maximum(d), 1e6ta / N, 1e6tb / N)
    end
end
