# =================================================================================================
# MRImagesMATExt: MATLAB files for MRImages, through MAT.jl. Loaded by Julia when both MRImages
# and MAT are loaded (`using MRImages, MAT`), a package extension; it adds the methods that
# src/mat.jl's functions call.
#
# With a precompile workload: MAT.jl compiles its reading code on first use, 6.9 s in each new
# session (measured: the first `matread` of a v7.3 file), so a `.mat` command took 11 s against
# 1.2 s for NIfTI. Writing and reading a small file here, while the extension is built, saves
# that compiled code with it.
# =================================================================================================

module MRImagesMATExt

using MRImages: MRImages, readimage, readvalues, readmat, writemat
using MAT: matopen, matwrite
using PrecompileTools: @setup_workload, @compile_workload

MRImages._matnames(file::AbstractString) = matopen(f -> String.(collect(keys(f))), file)
MRImages._matread(file::AbstractString, name::String) = matopen(f -> read(f, name), file)
MRImages._matwrite(file::String, vars::Dict{String,Any}) = (matwrite(file, vars); file)

@setup_workload begin
    dir = mktempdir()
    f = joinpath(dir, "w.mat")
    # The kinds of variables MATLAB users keep data in: 2-D (voxels × observations) to 4-D,
    # double and single, integer images, logical masks, a row of values, a number.
    vars = Dict{String,Any}("S4" => rand(2, 2, 2, 3), "S3" => rand(2, 2, 3), "S2" => rand(4, 3),
                            "s" => rand(Float32, 4, 3), "u" => UInt16[1 2; 3 4], "b" => [true false],
                            "row" => [0.01 0.02 0.03], "col" => [1.0, 2.0], "n" => 2.5)
    @compile_workload begin
        writemat(f, vars)
        readmat(f)
        for name in ("S4", "S3", "S2", "s", "u", "b")
            readimage(f * ":" * name)
            readimage(f * ":" * name; precision = Float32)
        end
        readvalues(f * ":row"), readvalues(f * ":col"), readvalues(f * ":n")
        maps = Dict{String,Any}("T2map" => rand(Float32, 4), "status" => rand(Float32, 2, 2, 1))
        writemat(joinpath(dir, "maps.mat"), maps)
    end
    rm(dir; recursive = true, force = true)
end

end
