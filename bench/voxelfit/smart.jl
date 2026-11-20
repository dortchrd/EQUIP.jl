# EQUIP's smart BestOf starts (branch bestof-proto) on the comparison's data, against VoxelFit's
# recorded fits and timings (out/, from vf.jl and eq.jl) and EQUIP's default starts, rerun here
# so that a change of machine speed shows. One thread, Float64, SNR 200 (and 50 for the sums of
# squares), best of 3. For each case: µs per voxel, mean iterations, converged %, and the voxels
# where one fit's sum of squares is above the other's (by more than 1e-8·Σy²): EQUIP's smart
# start against VoxelFit's fit.
#     julia -t 1 --project=../.. smart.jl          (after run.sh has made data/ and out/)
using EQUIP, Printf, Statistics
include(joinpath(@__DIR__, "cases.jl"))
eqmodel(c) = c.eq == "T1MFA" ? T1MFA(FA = c.x, TR = TR) :
             getfield(EQUIP, Symbol(c.eq))(; (c.eq in ("ADCDecay", "ADCKurtosis") ? :b :
                 c.eq == "T1IR" ? :TI : c.eq == "T1SR" ? :TD : :TE) => c.x)
best(f, k = 3) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:k))
loggrid(a, b, k) = exp.(range(log(a), log(b); length = k))
ratiogrid(a, b, r = 1.4) = loggrid(a, b, round(Int, log(b / a) / log(r)) + 1)   # about r× apart
B(v) = Free(start = BestOf(v))
# The grids from the acquisition (docs/design.md, item 4, "smart BestOf", point 2)
function smart(c)
    x = c.x
    c.name in ("T2", "T2offset") && return (T2 = B(ratiogrid(minimum(x) / 2, 2 * maximum(x))),)
    if c.name == "biexp"
        s = sqrt(minimum(x) * maximum(x))       # disjoint: T2short below, T2long above
        return (T2short = B(ratiogrid(minimum(x) / 2, s / 1.2)), T2long = B(ratiogrid(s, 2 * maximum(x))))
    end
    adc = B(ratiogrid(0.1 / maximum(x), min(10 / maximum(x), 4e-3)))
    c.name == "ADC" && return (ADC = adc,)
    c.name == "kurtosis" && return (ADC = adc, K = B(0:0.5:3))
    c.name in ("IR", "SR") && return (T1 = B(ratiogrid(minimum(filter(>(0), x)) / 5, 5.0)),)
    return (T1 = B(ratiogrid(0.05, 5.0)),)       # SPGR
end
vft(c) = parse(Float64, only(r[6] for r in (split(l, '\t') for l in readlines(joinpath(@__DIR__, "out", "timings.tsv")))
                             if r[1] == "VoxelFit" && r[2] == c.name && r[3] == "1" && r[4] == "Float64" && r[5] == "default"))
vfiters(c, snr) = (f = outfile("vf", c, snr, "default_iter"); isfile(f) ? mean(readbin(f, Int32, N)) : NaN)
conv(s) = 1 <= s <= 3
function sse(m, θ, Y)                            # each voxel's sum of squares at θ (a named tuple of vectors)
    return [sum(abs2, signal(m, map(v -> v[i], θ)) .- Y[i, :]) for i in 1:size(Y, 1)]
end
println("| model | start | µs/voxel (VoxelFit: its own) | iterations | converged | SSE above VoxelFit's / below (SNR 200; 50) |")
println("|---|---|---:|---:|---:|---|")
for c in CASES
    m = eqmodel(c)
    fixedB1 = c.name == "SPGR" ? (B1 = ones(N),) : (;)
    @printf("| %s | VoxelFit | %.2f | %.1f | | |\n", c.name, vft(c), vfiters(c, 200))
    for (label, kw) in (("EQUIP default", (;)), ("EQUIP smart BestOf", smart(c)))
        Y = readbin(datafile(c, 200), Float64, N, length(c.x))
        t = best(() -> fit(m, Y; kw...))
        r = fit(m, Y; kw...)
        cmp = String[]
        for snr in (200, 50)
            Ys = readbin(datafile(c, snr), Float64, N, length(c.x))
            rs = fit(m, Ys; kw...)
            vf = merge(fromvf(c, readbin(outfile("vf", c, snr, "default"), Float64, N, length(vfnames(c)))), fixedB1)
            svf = sse(m, vf, Ys)
            sy2 = vec(sum(abs2, Ys; dims = 2))
            sv = readbin(outfile("vf", c, snr, "default_status"), UInt8, N)
            ok = conv.(sv) .& isconverged.(rs.status)
            push!(cmp, "$(count(ok .& (rs.sse .> svf .+ 1e-8 .* sy2))) / $(count(ok .& (rs.sse .< svf .- 1e-8 .* sy2)))")
        end
        @printf("| %s | %s | %.2f | %.1f | %.1f%% | %s |\n", c.name, label, 1e6t / N, mean(r.iterations),
                100 * count(isconverged, r.status) / N, join(cmp, "; "))
        flush(stdout)
    end
end
