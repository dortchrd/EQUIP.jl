# Mean iterations per voxel (SNR 200), VoxelFit against EQUIP, beside EQUIP's default starts
using Statistics, EQUIP
include(joinpath(@__DIR__, "cases.jl"))
eqm(c) = c.eq == "T1MFA" ? T1MFA(FA = c.x, TR = TR) :
         getfield(EQUIP, Symbol(c.eq))(; (c.eq in ("ADCDecay", "ADCKurtosis") ? :b : c.eq == "T1IR" ? :TI : c.eq == "T1SR" ? :TD : :TE) => c.x)
println(rpad("case", 10), rpad("mean iterations, SNR 200: VoxelFit / EQUIP", 46), "EQUIP's starts")
for c in CASES
    iv = readbin(outfile("vf", c, 200, "default_iter"), Int32, N); ie = readbin(outfile("eq", c, 200, "default_iter"), Int32, N)
    st = join(("$k: $(r isa Free ? r.start : "fixed")" for (k, r) in pairs(parameters(eqm(c))) if r isa Free), ", ")
    println(rpad(c.name, 10), rpad("$(round(mean(iv); digits = 1)) / $(round(mean(ie); digits = 1))", 46), st)
end
