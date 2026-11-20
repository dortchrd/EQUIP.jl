# VoxelFit (the one-shot branch): accuracy fits ("acc") and timings ("time"), on gen.jl's data
using VoxelFit, StaticArrays
include(joinpath(@__DIR__, "cases.jl"))
vfmodel(c) = c.vf == "SPGR" ? SPGR(TR = TR) : getfield(VoxelFit, Symbol(c.vf))()
function best(f, k = 3)
    f()
    return minimum((GC.gc(); @elapsed f()) for _ in 1:k)
end
if ARGS[1] == "acc"
    for c in CASES, snr in SNRS
        m, P = vfmodel(c), length(vfnames(c))
        Y = readbin(datafile(c, snr), Float64, N, length(c.x))
        r = fit_maps(m, c.x, Y)
        writebin(outfile("vf", c, snr, "default"), Float64.(r.params))
        writebin(outfile("vf", c, snr, "default_status"), r.status)
        writebin(outfile("vf", c, snr, "default_iter"), r.iterations)
        G = zeros(N, P)                                    # its own per-voxel starts, for EQUIP
        for v in 1:N
            G[v, :] = initial_guess(m, c.x, Y[v, :])
        end
        writebin(outfile("vf", c, snr, "starts"), G)
    end
else
    nt = Threads.nthreads()
    tile = nt > 1 ? 10 : 1
    open(joinpath(@__DIR__, "out", "timings.tsv"), "a") do io
        for c in CASES
            m = vfmodel(c)
            Y = repeat(readbin(datafile(c, 200), Float64, N, length(c.x)), tile)
            G = repeat(readbin(outfile("vf", c, 200, "starts"), Float64, N, length(vfnames(c))), tile)
            for prec in (Float64, Float32), mode in ("default", "given")
                kw = mode == "given" ? (init = G,) : (;)
                t = best(() -> fit_maps(m, c.x, Y; precision = prec, kw...))
                println(io, join(("VoxelFit", c.name, nt, prec, mode, 1e6 * t / size(Y, 1)), '\t'))
                println(rpad("VoxelFit $(c.name) $nt thr $prec $mode", 44), round(1e6 * t / size(Y, 1); digits = 3), " µs/voxel")
            end
        end
    end
end
