# The deviation from the reference in each voxel's OWN standard error (the reference's
# `stderror`, σ from its residuals), which is what the statistical stop promises; and for the
# worst voxels, how large that own SE is against the population SD. Kurtosis and the
# bi-exponential, SNR 200, the data of statistical.jl
using EQUIP, Random, Statistics, Printf
N = 10_000
rng = MersenneTwister(3)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
TE = collect(0.01:0.01:0.32)
θT2 = (S0 = fill(1000.0, N), T2 = lu(0.02, 0.3))                         # consumed, as there
cases = [("bi-exponential", T2BiexpDecay(TE = TE), (S0 = fill(1000.0, N), fshort = uu(0.05, 0.4), T2short = lu(0.008, 0.03), T2long = lu(0.06, 0.15)), (:fshort, :T2long)),
         ("kurtosis", ADCKurtosis(b = collect(0.0:250.0:3000.0)), (S0 = fill(1000.0, N), ADC = lu(0.3e-3, 2.5e-3), K = uu(0.3, 2.0)), (:ADC, :K))]
for (name, m, θ, qs) in cases
    snr = 200
    Y = simulate(m, θ; snr, noise = :gaussian, rng = MersenneTwister(hash((name, snr))))
    ref = fit(m, Y; solver = LM(maxiter = 500, xtol = 1e-14, ftol = 1e-16, gtol = 1e-15))
    se = stderror(ref)
    for (label, lm) in (("default", LM()), ("statistical, c = 0.01", LM(ctol = 0.01)), ("statistical, c = 0.1", LM(ctol = 0.1)),
                        ("uniform 1e-5, 1e-8, 1e-6", LM(xtol = 1e-5, ftol = 1e-8, gtol = 1e-6)))
        r = fit(m, Y; solver = lm)
        ok = isconverged.(ref.status) .& isconverged.(r.status) .& reduce(.&, [isfinite.(se[q]) .& (se[q] .> 0) for q in qs])
        own = [abs.(r.params[q] .- ref.params[q])[ok] ./ se[q][ok] for q in qs]
        pop = [abs.(r.params[q] .- ref.params[q])[ok] ./ std((ref.params[q] .- θ[q])[ok]) for q in qs]
        worst = argmax(pop[end])
        ratio = se[qs[end]][ok][worst] / std((ref.params[qs[end]] .- θ[qs[end]])[ok])
        @printf("%-15s %-26s own SE: median %.1e, 99.9th pct %.1e, largest %.1e;  the worst voxel in population SDs: its own SE is %.0f× the population SD\n",
                name, label, maximum(median.(own)), maximum(quantile.(own, 0.999)), maximum(maximum.(own)), ratio)
    end
end
# The worst kurtosis voxels under the statistical stop: truth, reference, stop, SSEs, iterations
name, m, θ, qs = cases[2]
Y = simulate(m, θ; snr = 200, noise = :gaussian, rng = MersenneTwister(hash((name, 200))))
ref = fit(m, Y; solver = LM(maxiter = 500, xtol = 1e-14, ftol = 1e-16, gtol = 1e-15)); se = stderror(ref)
r = fit(m, Y; solver = LM(ctol = 0.01)); d = fit(m, Y)
own = abs.(r.params.K .- ref.params.K) ./ se.K
for i in sortperm(replace(own, NaN => -1.0); rev = true)[1:6]
    @printf("voxel %5d: truth ADC %.2e K %.2f | ref ADC %.3e K %.4f SSE %.4g it %d %s | stop K %.4f SSE %.4g it %d | default K %.4f it %d %s | SE(K) %.2e\n",
            i, θ.ADC[i], θ.K[i], ref.params.ADC[i], ref.params.K[i], ref.sse[i], ref.iterations[i], ref.status[i],
            r.params.K[i], r.sse[i], r.iterations[i], d.params.K[i], d.iterations[i], d.status[i], se.K[i])
end
println("voxels with the reference's K at a bound (0 or 3): ", count((ref.params.K .<= 1e-9) .| (ref.params.K .>= 3 - 1e-9)))
