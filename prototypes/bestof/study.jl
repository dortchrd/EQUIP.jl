# Smart BestOf (branch bestof-proto): the dictionary, linear parameters solved in the score,
# several parameters at once, grids from the acquisition; against each model's default start.
# 10,000 voxels a model, truths varied over a range, SNR 50 and 200, one thread
# (prototypes/bestof/README.md).
# Columns: µs per voxel for the whole fit and for the start alone (LM(maxiter = 0)), mean
# iterations, % converged, voxels not best (worse than the best of all strategies, or not
# converged), median |error| of the main parameter.
using EQUIP, Random, Statistics, Printf
const N = parse(Int, get(ENV, "NVOX", "10000"))
const ONLY = isempty(ARGS) ? nothing : ARGS        # model names to run, or all
loggrid(a, b, k) = exp.(range(log(a), log(b); length = k))
# k values from a to b about `r`× apart
ratiogrid(a, b, r = 1.4) = loggrid(a, b, round(Int, log(b / a) / log(r)) + 1)
lu(rng, a, b, n) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, n))      # log-uniform
uu(rng, a, b, n) = a .+ (b - a) .* rand(rng, n)
# Grids from the acquisition (proposal point 2)
t2grid(TE) = ratiogrid(minimum(TE) / 2, 2 * maximum(TE))
t1grid(t; hi = 5.0) = ratiogrid(minimum(filter(>(0), t)) / 5, min(5 * maximum(t), hi))
adcgrid(b) = ratiogrid(0.1 / maximum(b), min(10 / maximum(b), 4e-3))
function biexpgrids(TE)                     # disjoint: T2short below the split, T2long above
    s = sqrt(minimum(TE) * maximum(TE))
    return ratiogrid(minimum(TE) / 2, s / 1.2), ratiogrid(s, 2 * maximum(TE))
end

TE = collect(0.01:0.01:0.32); BA = collect(0.0:200.0:1000.0); BK = collect(0.0:250.0:3000.0)
TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]; TD = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
FA = [2.0, 5, 10, 15, 20, 30]
SIR_TI = [1.007, 0.010, 0.278, 0.010]; SIR_TD = [0.010, 4.174, 2.730, 0.684]
rng = MersenneTwister(7)
B(v) = Free(start = BestOf(v))
# Each strategy: label, keywords, dictionary on, linear parameters solved
S(label, kw = (;); dict = true, linear = true) = (; label, kw, dict, linear)
short, long = biexpgrids(TE)
cases = [
 ("T2", T2Decay(TE = TE), (S0 = fill(1000.0, N), T2 = lu(rng, 0.02, 0.3, N)), :T2,
  [S("default (:loglinear)"), S("BestOf $(length(t2grid(TE))), per voxel", (T2 = B(t2grid(TE)),); dict = false),
   S("BestOf $(length(t2grid(TE))), dictionary", (T2 = B(t2grid(TE)),))]),
 ("T2offset", T2DecayOffset(TE = TE), (S0 = fill(1000.0, N), T2 = lu(rng, 0.02, 0.3, N), C = uu(rng, 0, 200, N)), :T2,
  [S("default (fixed)"), S("default start, C solved (BestOf 1 value)", (T2 = B([mean(TE)]),)),
   S("BestOf T2, per voxel (C at 0)", (T2 = B(t2grid(TE)),); dict = false),
   S("BestOf T2, dictionary (C at 0)", (T2 = B(t2grid(TE)),); linear = false),
   S("BestOf T2, dictionary, C solved", (T2 = B(t2grid(TE)),))]),
 ("biexp", T2BiexpDecay(TE = TE), (S0 = fill(1000.0, N), fshort = uu(rng, 0.05, 0.4, N),
                                    T2short = lu(rng, 0.008, 0.03, N), T2long = lu(rng, 0.06, 0.15, N)), :T2long,
  [S("default (fixed)"), S("default starts, fshort solved (BestOf 1 value)", (T2long = B([2 * mean(TE)]),)),
   S("BestOf T2long $(length(long)), per voxel", (T2long = B(long),); dict = false),
   S("BestOf T2long $(length(long)), dictionary, fshort solved", (T2long = B(long),)),
   S("T2short × T2long $(length(short))×$(length(long)), per voxel", (T2short = B(short), T2long = B(long)); dict = false),
   S("T2short × T2long, dictionary (fshort at 0.2)", (T2short = B(short), T2long = B(long)); linear = false),
   S("T2short × T2long, dictionary, fshort solved", (T2short = B(short), T2long = B(long)))]),
 ("ADC", ADCDecay(b = BA), (S0 = fill(1000.0, N), ADC = lu(rng, 0.2e-3, 3e-3, N)), :ADC,
  [S("default (fixed)"), S("BestOf $(length(adcgrid(BA))), per voxel", (ADC = B(adcgrid(BA)),); dict = false),
   S("BestOf $(length(adcgrid(BA))), dictionary", (ADC = B(adcgrid(BA)),))]),
 ("kurtosis", ADCKurtosis(b = BK), (S0 = fill(1000.0, N), ADC = lu(rng, 0.3e-3, 2.5e-3, N), K = uu(rng, 0.3, 2.0, N)), :ADC,
  [S("default (fixed)"), S("BestOf ADC $(length(adcgrid(BK))), dictionary", (ADC = B(adcgrid(BK)),)),
   S("ADC × K $(length(adcgrid(BK)))×7, per voxel", (ADC = B(adcgrid(BK)), K = B(0:0.5:3)); dict = false),
   S("ADC × K, dictionary", (ADC = B(adcgrid(BK)), K = B(0:0.5:3)))]),
 ("IR", T1IR(TI = TI), (M0 = fill(1000.0, N), Sf = uu(rng, -1.0, -0.85, N), T1 = lu(rng, 0.3, 4.0, N)), :T1,
  [S("default: BestOf 30, per voxel (main)"; dict = false), S("BestOf 30, dictionary"),
   S("BestOf $(length(t1grid(TI))), dictionary", (T1 = B(t1grid(TI)),))]),
 ("IR signed", T1IR(TI = TI, magnitude = false), (M0 = fill(1000.0, N), Sf = uu(rng, -1.0, -0.85, N), T1 = lu(rng, 0.3, 4.0, N)), :T1,
  [S("default: BestOf 30, per voxel (main)"; dict = false), S("BestOf 30, dictionary (Sf at -0.95)"; linear = false),
   S("BestOf 30, dictionary, Sf solved"), S("BestOf $(length(t1grid(TI))), dictionary, Sf solved", (T1 = B(t1grid(TI)),))]),
 ("SR", T1SR(TD = TD), (M0 = fill(1000.0, N), T1 = lu(rng, 0.3, 4.0, N)), :T1,
  [S("default (fixed)"), S("BestOf $(length(t1grid(TD))), per voxel", (T1 = B(t1grid(TD)),); dict = false),
   S("BestOf $(length(t1grid(TD))), dictionary", (T1 = B(t1grid(TD)),))]),
 ("SPGR", T1MFA(FA = FA, TR = 0.015), (M0 = fill(1000.0, N), T1 = lu(rng, 0.3, 4.0, N), B1 = ones(N)), :T1,
  [S("default (fixed)"), S("BestOf 15, per voxel", (T1 = B(ratiogrid(0.05, 5.0)),); dict = false),
   S("BestOf 15, dictionary", (T1 = B(ratiogrid(0.05, 5.0)),))]),
 ("QMTSIR 4-point", QMTSIR(TI = SIR_TI, TD = SIR_TD), (PSR = uu(rng, 0.05, 0.25, N), R1f = lu(rng, 0.6, 1.8, N),
     Sf = uu(rng, -1.05, -0.9, N), M0f = fill(1000.0, N), kmf = fill(12.5, N)), :PSR,
  [S("default (fixed)"), S("BestOf R1f 8, per voxel", (R1f = B(loggrid(0.3, 3, 8)),); dict = false),
   S("BestOf R1f 8, dictionary", (R1f = B(loggrid(0.3, 3, 8)),)),
   S("PSR × R1f 6×8, per voxel", (PSR = B([0.02, 0.05, 0.1, 0.15, 0.2, 0.3]), R1f = B(loggrid(0.3, 3, 8))); dict = false),
   S("PSR × R1f 6×8, dictionary", (PSR = B([0.02, 0.05, 0.1, 0.15, 0.2, 0.3]), R1f = B(loggrid(0.3, 3, 8))))]),
]
function withswitches(f, s)
    EQUIP._PROTO_DICTIONARY[] = s.dict
    EQUIP._PROTO_LINEAR[] = s.linear
    try
        return f()
    finally
        EQUIP._PROTO_DICTIONARY[] = true
        EQUIP._PROTO_LINEAR[] = true
    end
end
best(f, k = 3) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:k))
println("| model | SNR | start | µs/voxel | start µs | iterations | converged | not best | median error |")
println("|---|---:|---|---:|---:|---:|---:|---:|---:|")
for (name, m, θ, q, strategies) in cases, snr in (50, 200)
    ONLY === nothing || name in ONLY || continue
    sim = name == "QMTSIR 4-point" ? merge(θ, (Sm = fill(0.83, N), R1m = θ.R1f, Mzf_td0 = zeros(N), Mzm_td0 = zeros(N))) : θ
    Y = simulate(m, sim; snr, noise = :gaussian, rng = MersenneTwister(hash((name, snr))))
    rs = [withswitches(() -> fit(m, Y; s.kw...), s) for s in strategies]
    sumy2 = vec(sum(abs2, Y; dims = 2))
    sse = reduce(hcat, [ifelse.(isconverged.(r.status), r.sse, Inf) for r in rs])
    bestsse = vec(minimum(sse; dims = 2))
    for (s, r) in zip(strategies, rs)
        t = withswitches(() -> best(() -> fit(m, Y; s.kw...)), s)
        t0 = withswitches(() -> best(() -> fit(m, Y; s.kw..., solver = LM(maxiter = 0))), s)
        notbest = count(i -> !isconverged(r.status[i]) || r.sse[i] > bestsse[i] + 1e-8 * sumy2[i], 1:N)
        err = median(abs.(r.params[q] ./ θ[q] .- 1))
        @printf("| %s | %d | %s | %.2f | %.2f | %.1f | %.1f%% | %d | %.2f%% |\n", name, snr, s.label,
                1e6t / N, 1e6t0 / N, mean(r.iterations), 100 * count(isconverged, r.status) / N, notbest, 100err)
        flush(stdout)
    end
end
