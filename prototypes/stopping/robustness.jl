# Robustness of the combination (calibrated absolute tolerances + the stall rule at 1%) across
# models: nine models, varied truths, SNR 50 and 200, 10,000 voxels, one thread, against a tight
# reference fit; then the voxels a poor mask lets in (Rician, background and weak signal) for
# four of them. Columns: µs/voxel, iterations, % stalled, % stalled where the reference has no
# parameter on a bound (the rule must not fire there), the 99.9th percentile of |Δ|/SD over the
# tested parameters (SD: the spread of the reference's errors), the largest |Δ|/SD among the
# stalled voxels.
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
         ("T2 + offset", T2DecayOffset(TE = TE), (S0 = fill(1000.0, N), T2 = lu(0.02, 0.3), C = uu(0, 200)), (;), (:T2, :C)),
         ("T1IR", T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]), (M0 = fill(1000.0, N), Sf = uu(-1.0, -0.85), T1 = lu(0.3, 4.0)), (;), (:T1,))]
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
anybound(r) = (ob = onbound(r); [any(k -> ob[k][i], keys(ob)) for i in 1:N])
function row(name, tag, m, Y, kw, qs, ref, sd, tols)
    nb0 = .!anybound(ref)
    for (label, lm) in (("default", LM()), ("atol", LM(atol = tols)), ("atol + stall 1%", LM(atol = tols, stall = true, stalltol = 0.01)))
        r = fit(m, Y; kw..., solver = lm)
        t = best(() -> fit(m, Y; kw..., solver = lm))
        st = r.status .== EQUIP.StalledOnBound
        ok = (isconverged.(ref.status) .| (ref.status .== EQUIP.StalledOnBound)) .& (isconverged.(r.status) .| st)
        dev = reduce(hcat, [abs.(r.params[q] .- ref.params[q]) ./ sd[q] for q in qs])
        dmax = vec(maximum(dev; dims = 2))
        p999 = quantile(dmax[ok], 0.999)
        dst = any(st) ? @sprintf("%.1e", maximum(dmax[st])) : "–"
        @printf("| %s | %s | %s | %.2f | %.1f | %.1f%% | %.2f%% | %.1e | %s |\n", name, tag, label, 1e6t / N, mean(r.iterations),
                100count(st) / N, 100count(st .& nb0) / N, p999, dst)
        flush(stdout)
    end
end
tight = LM(maxiter = 500, xtol = 1e-14, ftol = 1e-16, gtol = 1e-15)
println("| model | data | stopping | µs/voxel | iterations | stalled | stalled, reference off bounds | 99.9th pct |Δ|/SD | largest |Δ|/SD, stalled |")
println("|---|---|---|---:|---:|---:|---:|---:|---:|")
for (name, m, θ, kw, qs) in cases
    sdref = nothing
    for snr in (50, 200)
        Y = simulate(m, θ; snr, noise = :gaussian, rng = MersenneTwister(hash((name, snr))))
        ref = fit(m, Y; kw..., solver = tight)
        ok0 = isconverged.(ref.status)
        sd = Dict(q => std((ref.params[q] .- θ[q])[ok0]) for q in qs)
        snr == 50 && (sdref = sd)
        row(name, "SNR $snr", m, Y, kw, qs, ref, sd, stoptolerances(m, Y; kw...))
    end
    # A poor mask: half tissue at SNR 50, a quarter weak signal (S0 or M0 at 3σ), a quarter noise
    # only, Rician; tolerances calibrated on the whole, deviations in the SNR-50 SDs
    name in ("QMTSIR 4-point", "T2, 32 echoes", "kurtosis", "T1IR") || continue
    s0 = haskey(θ, :M0f) ? :M0f : haskey(θ, :S0) ? :S0 : :M0
    a = fill(1000.0, N); a[N÷2+1:3N÷4] .= 3 * 1000 / 50; a[3N÷4+1:end] .= 0
    θp = merge(θ, NamedTuple{(s0,)}((a,)))
    Y = simulate(m, θp; σ = 1000 / 50, noise = :rician, rng = MersenneTwister(hash((name, :mask))))
    ref = fit(m, Y; kw..., solver = tight)
    row(name, "poor mask (Rician)", m, Y, kw, qs, ref, sdref, stoptolerances(m, Y; kw...))
end
