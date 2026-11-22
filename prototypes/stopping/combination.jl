# The owner's scan at full size (its 5 slices repeated to 80: 512×512×80, 4 volumes, Float64, 6.5
# million voxels in the mask), read, fitted and written in one process, warm, 4 threads, for each
# combination of: QMTSIREqualR1 for QMTSIR, calibrated absolute or relative tolerances, the stall
# rule (0.1% or 1%), R1f ≥ 0.2 s⁻¹ (T1 ≤ 5 s, the T1 models' cap) and uncompressed maps.
# Accuracy: PSR against the baseline in the voxels where the baseline fits cleanly (no parameter on
# a bound, SSE below 10σ²), σ from the background's median.
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
rels = stoptolerances(qe, data; mask, kmf = 12.5, relative = true)
r1 = (R1f = Free(lower = 0.2),)
configs = [
 ("baseline: QMTSIR, default stopping, gzipped maps", qs, LM(), (;), true),
 ("QMTSIREqualR1", qe, LM(), (;), true),
 ("+ calibrated absolute tolerances", qe, LM(atol = tols), (;), true),
 ("+ stall rule 0.1%", qe, LM(atol = tols, stall = true), (;), true),
 ("+ stall rule 1%", qe, LM(atol = tols, stall = true, stalltol = 0.01), (;), true),
 ("+ uncompressed maps", qe, LM(atol = tols, stall = true, stalltol = 0.01), (;), false),
 ("  the same with relative tolerances", qe, LM(rtol = rels, stall = true, stalltol = 0.01), (;), false),
 ("  the same with absolute, and R1f ≥ 0.2 s⁻¹", qe, LM(atol = tols, stall = true, stalltol = 0.01), r1, false),
]
base = nothing; clean = nothing
println("| configuration | read s | fit s | write s | total s | iterations | stalled | clean voxels: largest |ΔPSR| |")
println("|---|---:|---:|---:|---:|---:|---:|---:|")
for (label, m, lm, kw, gz) in configs
    tr = @elapsed (d2, i2) = readdata(joinpath(full, "SIR.nii")); readmap(joinpath(full, "MASK.nii"), i2)
    fit(m, data[1:2, 1:2, 1:1, :]; kw..., kmf = 12.5, solver = lm)            # compile
    tf = @elapsed (r = fit(m, data; mask, kw..., kmf = 12.5, solver = lm))
    writemaps(joinpath(full, "w_"), r, info; compress = gz)                     # compile
    tw = @elapsed writemaps(joinpath(full, "w_"), r, info; compress = gz)
    it = vec(r.iterations)[inm]; st = vec(r.status)[inm]; psr = vec(r.params.PSR)[inm]
    if base === nothing
        global base = psr
        ob = onbound(r)
        global clean = .!reduce(.|, [vec(ob[k])[inm] for k in keys(ob)]) .& (vec(r.sse)[inm] .< 10σ^2)
    end
    @printf("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f%% | %.1e |\n", label, tr, tf, tw, tr + tf + tw, mean(it),
            100 * count(==(EQUIP.StalledOnBound), st) / length(st), maximum(abs.(psr .- base)[clean]))
    flush(stdout)
end
@printf("\n(σ = %.1f; clean voxels: %.0f%% of the mask; tolerances %s)\n", σ, 100 * count(clean) / length(clean),
        join(("$k $(round(v; sigdigits = 2))" for (k, v) in pairs(tols)), ", "))
rm(full; recursive = true)
