# =================================================================================================
# EQUIPMATExt: MATLAB files for EQUIP's fits, compiled ahead. Loaded by Julia when EQUIP and MAT
# are both loaded (`using EQUIP, MAT`), a package extension holding only a precompile workload:
# readdata, readmap, readvalues and writemaps on a small .mat file.
#
# Why a workload of its own: MRImages' MAT extension compiles MAT.jl's reading and writing when
# it is built, but StaticArrays (which EQUIP loads, directly and through ForwardDiff) invalidates
# that code when it is loaded first, as it is in every EQUIP session: the first .mat file of a
# session then compiled for 2.3 s (measured; 7 ms without StaticArrays loaded). Compiled here,
# with EQUIP and StaticArrays already loaded, the code stays valid.
#
# Its triggers are MAT and MRImages (Project.toml): while an extension is built, Julia loads only
# the extensions whose triggers are a strict subset of its own, and the workload reads through
# MRImages' MAT extension (triggers MRImages and MAT).
# =================================================================================================

module EQUIPMATExt

using EQUIP
using EQUIP: MRImages
using MAT: MAT
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    dir = mktempdir()
    f = joinpath(dir, "data.mat")
    te = collect(0.01:0.02:0.15)
    m = T2Decay(TE = te)
    S = Float64[1000 * exp(-t / T2) for T2 in (0.05, 0.07, 0.09, 0.11), t in te]   # 4 voxels × 8
    @compile_workload begin
        MRImages.writemat(f, Dict{String,Any}("S" => S, "S4" => reshape(S, 2, 2, 1, 8),
                                              "TE" => collect(te'), "mask" => [1.0, 1.0, 1.0, 0.0]))
        for name in ("S", "S4"), precision in (Float64, Float32)
            data, info = readdata(f * ":" * name; precision)
            name == "S" && readmap(f * ":mask", info)
            res = fit(m, data)
            writemaps(joinpath(dir, "out_"), res, info; stderror = true)
        end
        readvalues(f * ":TE")
    end
    rm(dir; recursive = true, force = true)
end

end
