# The owner's scan at full size (its 5 slices repeated to 80: 512×512×80, 4 volumes, Float64, 6.5
# million voxels in the mask), read, fitted and written in one process, warm, 4 threads: the
# fastest combination of combination.jl (QMTSIREqualR1, calibrated absolute tolerances, the stall
# rule at 1%, uncompressed maps) with the smart starts too (branch combined-proto, 2026-11-22:
# BestOf from a dictionary, the linear parameters solved in each candidate's score).
# Accuracy against the baseline: PSR in the voxels where it fits cleanly (no parameter on a bound,
# SSE below 10σ², σ from the background's median), and the voxels whose SSE is more than σ² above
# or below the baseline's (another minimum; not relative: 4 points, 4 parameters, a clean fit's SSE is 0).
using EQUIP, MRImages, Statistics, Printf
dir = ARGS[1]                    # a folder with the scan, SIR.nii.gz, and its mask, MASK.nii.gz
img = readimage(joinpath(dir, "SIR.nii.gz")); mk = readimage(joinpath(dir, "MASK.nii.gz"))
full = joinpath(dir, "full"); mkpath(full)
writeimage(joinpath(full, "SIR.nii"), MRImage(repeat(img.data, 1, 1, 16, 1)))
writeimage(joinpath(full, "MASK.nii"), MRImage(repeat(mk.data, 1, 1, 16)))
TI = [1.007, 0.010, 0.278, 0.010]; TD = [0.010, 4.174, 2.730, 0.684]
data, info = readdata(joinpath(full, "SIR.nii")); mask = readmap(joinpath(full, "MASK.nii"), info)
Yall = reshape(data, :, 4); inm = vec(mask .!= 0)
bg = Yall[.!inm, :]; bg = bg[vec(all(bg .> 0; dims = 2)), :]; σ = median(bg) / 1.1774
qs, qe = QMTSIR(TI = TI, TD = TD), QMTSIREqualR1(TI = TI, TD = TD)
tols = stoptolerances(qe, data; mask, kmf = 12.5)
loggrid(a, b, k) = exp.(range(log(a), log(b); length = k))
B(v) = Free(start = BestOf(v))
fast = LM(atol = tols, stall = true, stalltol = 0.01)
psrs = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
configs = [
 ("baseline: QMTSIR, default stopping, gzipped maps", qs, LM(), (;), true),
 ("QMTSIREqualR1, atol, stall 1%, uncompressed", qe, fast, (;), false),
 ("+ BestOf R1f 8 (0.3–3), dictionary", qe, fast, (R1f = B(loggrid(0.3, 3, 8)),), false),
 ("+ PSR × R1f 6×8, dictionary", qe, fast, (PSR = B(psrs), R1f = B(loggrid(0.3, 3, 8))), false),
 ("+ PSR × R1f 6×10 (R1f 0.1–3), dictionary", qe, fast, (PSR = B(psrs), R1f = B(loggrid(0.1, 3, 10))), false),
 ("  PSR × R1f 6×8, dictionary, default stopping", qe, LM(), (PSR = B(psrs), R1f = B(loggrid(0.3, 3, 8))), false),
]
base = nothing; clean = nothing; sse0 = nothing
println("| configuration | read s | fit s | write s | total s | iterations | stalled | clean voxels: largest |ΔPSR| | SSE > σ² above the baseline | > σ² below |")
println("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (label, m, lm, kw, gz) in configs
    tr = @elapsed (d2, i2) = readdata(joinpath(full, "SIR.nii")); readmap(joinpath(full, "MASK.nii"), i2)
    fit(m, data[1:2, 1:2, 1:1, :]; kw..., kmf = 12.5, solver = lm)            # compile
    tf = @elapsed (r = fit(m, data; mask, kw..., kmf = 12.5, solver = lm))
    writemaps(joinpath(full, "w_"), r, info; compress = gz)                     # compile
    tw = @elapsed writemaps(joinpath(full, "w_"), r, info; compress = gz)
    it = vec(r.iterations)[inm]; st = vec(r.status)[inm]; psr = vec(r.params.PSR)[inm]; sse = vec(r.sse)[inm]
    if base === nothing
        global base, sse0 = psr, sse
        ob = onbound(r)
        global clean = .!reduce(.|, [vec(ob[k])[inm] for k in keys(ob)]) .& (sse .< 10σ^2)
    end
    @printf("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f%% | %.1e | %.2f%% | %.2f%% |\n", label, tr, tf, tw, tr + tf + tw,
            mean(it), 100 * count(==(EQUIP.StalledOnBound), st) / length(st), maximum(abs.(psr .- base)[clean]),
            100 * count(sse .> sse0 .+ σ^2) / length(sse), 100 * count(sse .< sse0 .- σ^2) / length(sse))
    flush(stdout)
end
@printf("\n(σ = %.1f; clean voxels: %.0f%% of the mask)\n", σ, 100 * count(clean) / length(clean))
rm(full; recursive = true)
