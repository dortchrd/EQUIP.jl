# VoxelFit: load time, the first fit (compiling) and the second, in a fresh process (ARGS[1]: a case's index)
t0 = time(); using VoxelFit; t1 = time()
include(joinpath(@__DIR__, "cases.jl"))
c = CASES[parse(Int, ARGS[1])]
m = c.vf == "SPGR" ? SPGR(TR = TR) : getfield(VoxelFit, Symbol(c.vf))()
Y = readbin(datafile(c, 200), Float64, N, length(c.x))[1:2000, :]
t2 = time(); fit_maps(m, c.x, Y); t3 = time(); fit_maps(m, c.x, Y); t4 = time()
println(join(("VoxelFit", c.name, round(t1 - t0; digits = 2), round(t3 - t2; digits = 2), round(1e3 * (t4 - t3); digits = 1)), '\t'))
