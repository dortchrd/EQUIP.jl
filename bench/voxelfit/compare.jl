# Accuracy and agreement of VoxelFit and EQUIP on the same data (from vf.jl and eq.jl "acc")
using Statistics, Printf
include(joinpath(@__DIR__, "cases.jl"))
conv(s) = (s .>= 1) .& (s .<= 3)
# the quantities compared, from a VoxelFit-ordered matrix, and their truths
function quantities(c, θ)
    n = c.name
    n == "T2" && return (T2 = θ[:, 2],)
    n == "T2offset" && return (T2 = θ[:, 2], C = θ[:, 3])
    n == "biexp" && return (fshort = θ[:, 1] ./ (θ[:, 1] .+ θ[:, 3]), T2short = θ[:, 2], T2long = θ[:, 4])
    n == "ADC" && return (D = θ[:, 2],)
    n == "kurtosis" && return (D = θ[:, 2], K = θ[:, 3])
    n == "IR" && return (T1 = θ[:, 3],)
    return (T1 = θ[:, 2],)
end
truthq(c) = quantities(c, reshape(collect(tovf(c, c.truth)), 1, :))
pct(x) = @sprintf("%+.2f", 100x)
println("| case | SNR | quantity | converged VF / EQ | bias % VF / EQ | SD % VF / EQ | agreement, own starts: median, 99th pct, voxels > 1e-4 | same starts: median, 99th pct, voxels > 1e-4 |")
println("|---|---:|---|---|---|---|---|---|")
for c in CASES, snr in SNRS
    P = length(vfnames(c))
    vf = readbin(outfile("vf", c, snr, "default"), Float64, N, P)
    eq = readbin(outfile("eq", c, snr, "default"), Float64, N, P)
    eg = readbin(outfile("eq", c, snr, "given"), Float64, N, P)
    sv = readbin(outfile("vf", c, snr, "default_status"), UInt8, N)
    se = readbin(outfile("eq", c, snr, "default_status"), UInt8, N)
    sg = readbin(outfile("eq", c, snr, "given_status"), UInt8, N)
    qv, qe, qg, qt = quantities(c, vf), quantities(c, eq), quantities(c, eg), truthq(c)
    cv, ce, cg = conv(sv), conv(se), conv(sg)
    for k in keys(qt)
        t = qt[k][1]
        b(q, cc) = median(q[cc]) / t - 1
        s(q, cc) = std(q[cc]) / abs(t)
        both, bothg = cv .& ce, cv .& cg
        d(a, b, cc) = (r = abs.(a[cc] .- b[cc]) ./ abs(t); (median(r), quantile(r, 0.99), count(>(1e-4), r)))
        a1, a2 = d(qv[k], qe[k], both), d(qv[k], qg[k], bothg)
        @printf("| %s | %d | %s | %.1f%% / %.1f%% | %s / %s | %.2f / %.2f | %.1e, %.1e, %d | %.1e, %.1e, %d |\n",
                c.name, snr, k, 100mean(cv), 100mean(ce), pct(b(qv[k], cv)), pct(b(qe[k], ce)),
                100s(qv[k], cv), 100s(qe[k], ce), a1..., a2...)
    end
end
