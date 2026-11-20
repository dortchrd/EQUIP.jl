# EQUIP (main): the same fits and timings on the same data
using EQUIP
include(joinpath(@__DIR__, "cases.jl"))
eqmodel(c) = c.eq == "T1MFA" ? T1MFA(FA = c.x, TR = TR) :
             getfield(EQUIP, Symbol(c.eq))(; (c.eq in ("ADCDecay", "ADCKurtosis") ? :b :
                 c.eq == "T1IR" ? :TI : c.eq == "T1SR" ? :TD : :TE) => c.x)
function best(f, k = 3)
    f()
    return minimum((GC.gc(); @elapsed f()) for _ in 1:k)
end
# VoxelFit's starts as EQUIP start maps, inside EQUIP's bounds
function startroles(m, c, G)
    s = fromvf(c, G)
    d = parameters(m)
    return NamedTuple{keys(s)}(Tuple(Free(start = clamp.(s[k], something(d[k].lower, -Inf), something(d[k].upper, Inf)))
                                     for k in keys(s)))
end
asvf(c, r) = reduce(hcat, [Float64.(v) for v in tovf(c, r.params)])
if ARGS[1] == "acc"
    for c in CASES, snr in SNRS
        m = eqmodel(c)
        Y = readbin(datafile(c, snr), Float64, N, length(c.x))
        r = fit(m, Y)
        writebin(outfile("eq", c, snr, "default"), asvf(c, r))
        writebin(outfile("eq", c, snr, "default_status"), UInt8.(r.status))
        writebin(outfile("eq", c, snr, "default_iter"), Int32.(r.iterations))
        G = readbin(outfile("vf", c, snr, "starts"), Float64, N, length(vfnames(c)))
        r2 = fit(m, Y; startroles(m, c, G)...)
        writebin(outfile("eq", c, snr, "given"), asvf(c, r2))
        writebin(outfile("eq", c, snr, "given_status"), UInt8.(r2.status))
    end
else
    nt = Threads.nthreads()
    tile = nt > 1 ? 10 : 1
    open(joinpath(@__DIR__, "out", "timings.tsv"), "a") do io
        for c in CASES
            m = eqmodel(c)
            Y = repeat(readbin(datafile(c, 200), Float64, N, length(c.x)), tile)
            G = repeat(readbin(outfile("vf", c, 200, "starts"), Float64, N, length(vfnames(c))), tile)
            roles = startroles(m, c, G)
            for prec in (Float64, Float32), mode in ("default", "given")
                kw = mode == "given" ? roles : (;)
                t = best(() -> fit(m, Y; precision = prec, kw...))
                println(io, join(("EQUIP", c.name, nt, prec, mode, 1e6 * t / size(Y, 1)), '\t'))
                println(rpad("EQUIP $(c.name) $nt thr $prec $mode", 44), round(1e6 * t / size(Y, 1); digits = 3), " µs/voxel")
            end
        end
    end
end
