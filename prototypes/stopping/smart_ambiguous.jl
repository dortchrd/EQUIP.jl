using EQUIP, MRImages, Statistics, Printf
dir = ARGS[1]
TI = [1.007, 0.010, 0.278, 0.010]; TD = [0.010, 4.174, 2.730, 0.684]
data, info = readdata(joinpath(dir, "SIR.nii.gz")); mask = readmap(joinpath(dir, "MASK.nii.gz"), info)
Yall = reshape(data, :, 4); inm = vec(mask .!= 0)
bg = Yall[.!inm, :]; bg = bg[vec(all(bg .> 0; dims = 2)), :]; σ = median(bg) / 1.1774
qe = QMTSIREqualR1(TI = TI, TD = TD)
loggrid(a, b, k) = exp.(range(log(a), log(b); length = k)); B(v) = Free(start = BestOf(v))
tight = LM(maxiter = 500, xtol = 1e-14, ftol = 1e-16, gtol = 1e-15)
r0 = fit(qe, data; mask, kmf = 12.5, solver = tight)
ob = onbound(r0); sse0 = vec(r0.sse)[inm]; p0 = vec(r0.params.PSR)[inm]
clean = .!reduce(.|, [vec(ob[k])[inm] for k in keys(ob)]) .& (sse0 .< 10σ^2)
println("clean ", count(clean), " of ", count(inm))
for (label, kw) in (("default start", (;)), ("BestOf R1f 8", (R1f = B(loggrid(0.3, 3, 8)),)),
                    ("PSR × R1f 6×8", (PSR = B([0.02, 0.05, 0.1, 0.15, 0.2, 0.3]), R1f = B(loggrid(0.3, 3, 8)))))
    r = fit(qe, data; mask, kmf = 12.5, solver = tight, kw...)
    sse = vec(r.sse)[inm]; p = vec(r.params.PSR)[inm]; d = abs.(p .- p0)
    @printf("%s (tight): clean voxels |ΔPSR| > 0.01: %d, > 0.1: %d; mask SSE higher by > σ²: %.2f%%, lower by > σ²: %.2f%%\n", label,
            count(d[clean] .> 0.01), count(d[clean] .> 0.1), 100count(sse .> sse0 .+ σ^2) / length(sse), 100count(sse .< sse0 .- σ^2) / length(sse))
    sf = vec(r0.params.Sf)[inm]; inv = clean .& (sf .< -0.8)
    @printf("   well-inverted clean voxels (Sf < -0.8): %d, of them |ΔPSR| > 0.01: %d; ambiguous clean voxels Sf median %.2f\n", count(inv), count(d[inv] .> 0.01), (any(clean .& (d .> 0.01)) ? median(sf[clean .& (d .> 0.01)]) : NaN))
    w = findall(clean .& (d .> 0.1))
    for i in w[1:0]
        j = findall(inm)[i]
        @printf("   voxel %d: y %s; default PSR %.3f R1f %.3f Sf %.3f SSE/σ² %.2e; this PSR %.3f R1f %.3f Sf %.3f SSE/σ² %.2e\n", j,
                string(round.(Yall[j, :]; digits = 0)), r0.params.PSR[j], r0.params.R1f[j], r0.params.Sf[j], r0.sse[j] / σ^2,
                r.params.PSR[j], r.params.R1f[j], r.params.Sf[j], r.sse[j] / σ^2)
    end
end
