# Per-parameter tolerances (branch stopping-proto, LM(atol = ...)): calibrated from a sample
# (stoptolerances, 1000 voxels, c = 0.01), the same with PSR set to 5e-4 ("3 decimals"), PSR's
# alone, and the statistical stop, against the defaults and a tight reference. 10,000 voxels with
# varied truths a model, SNR 50 and 200, one thread. Deviations from the reference in units of
# each parameter's statistical SD (the spread of the reference's errors): the largest over the
# tested parameters of the median and the 99.9th percentile, and for SIR the largest |ΔPSR|.
using EQUIP, Random, Statistics, Printf
N = 10_000
rng = MersenneTwister(3)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
TE = collect(0.01:0.01:0.32)
sir(acq, kmf) = (QMTSIREqualR1(; acq...), (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N),
                 kmf = kmf ? lu(5, 20) : fill(12.5, N), Sm = fill(0.83, N)), kmf ? (kmf = Free(),) : (;))
cases = [("QMTSIR 4-point", sir(SIR_4POINT, false)..., (:PSR, :R1f)),
         ("QMTSIR 5-point, kmf fixed", sir(SIR_5POINT, false)..., (:PSR, :R1f)),
         ("QMTSIR 5-point, kmf free", sir(SIR_5POINT, true)..., (:PSR, :kmf)),
         ("QMTSIR 16-point, kmf free", sir(SIR_16POINT, true)..., (:PSR, :kmf)),
         ("T2, 32 echoes", T2Decay(TE = TE), (S0 = fill(1000.0, N), T2 = lu(0.02, 0.3)), (;), (:T2,)),
         ("bi-exponential", T2BiexpDecay(TE = TE), (S0 = fill(1000.0, N), fshort = uu(0.05, 0.4), T2short = lu(0.008, 0.03), T2long = lu(0.06, 0.15)), (;), (:fshort, :T2long)),
         ("kurtosis", ADCKurtosis(b = collect(0.0:250.0:3000.0)), (S0 = fill(1000.0, N), ADC = lu(0.3e-3, 2.5e-3), K = uu(0.3, 2.0)), (;), (:ADC, :K)),
         ("T1IR", T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]), (M0 = fill(1000.0, N), Sf = uu(-1.0, -0.85), T1 = lu(0.3, 4.0)), (;), (:T1,))]
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("| model | SNR | stopping | µs/voxel | iterations | converged | median, 99.9th pct |Δ|/SD | largest |ΔPSR| |")
println("|---|---:|---|---:|---:|---:|---|---:|")
for (name, m, θ, kw, qs) in cases, snr in (50, 200)
    Y = simulate(m, θ; snr, noise = :gaussian, rng = MersenneTwister(hash((name, snr))))
    ref = fit(m, Y; kw..., solver = LM(maxiter = 500, xtol = 1e-14, ftol = 1e-16, gtol = 1e-15))
    ok0 = isconverged.(ref.status)
    sd = Dict(q => std((ref.params[q] .- θ[q])[ok0]) for q in qs)
    tcal = @elapsed (tols = stoptolerances(m, Y; kw...))
    sirs = haskey(ref.params, :PSR)
    settings = [("default", LM()), ("calibrated, every parameter", LM(atol = tols))]
    sirs && push!(settings, ("calibrated, PSR at 5e-4", LM(atol = merge(tols, (PSR = 5e-4,)))),
                            ("PSR alone at 5e-4", LM(atol = (PSR = 5e-4,))))
    nobs(m) > length(tols) && push!(settings, ("statistical, c = 0.01", LM(ctol = 0.01)))
    @printf("| %s | %d | (calibration: %.0f ms, %s) | | | | | |\n", name, snr, 1e3tcal,
            join(("$k $(round(v; sigdigits = 2))" for (k, v) in pairs(tols)), ", "))
    for (label, lm) in settings
        r = fit(m, Y; kw..., solver = lm)
        t = best(() -> fit(m, Y; kw..., solver = lm))
        ok = ok0 .& isconverged.(r.status)
        dev = [abs.((r.params[q] .- ref.params[q])[ok]) ./ sd[q] for q in qs]
        dpsr = sirs ? @sprintf("%.1e", maximum(abs.(r.params.PSR .- ref.params.PSR)[ok])) : ""
        @printf("| %s | %d | %s | %.2f | %.1f | %.1f%% | %.1e, %.1e | %s |\n", name, snr, label, 1e6t / N, mean(r.iterations),
                100 * count(isconverged, r.status) / N, maximum(median.(dev)), maximum(quantile.(dev, 0.999)), dpsr)
        flush(stdout)
    end
end
