# =================================================================================================
# Precompile workload: write and read a small image (NIfTI, compressed and not, with a sidecar),
# and a values file, so that the first readimage or writeimage of a session does not compile
# them (PrecompileTools). Runs when the package is installed or changed, not when it is loaded.
# =================================================================================================

@setup_workload begin
    dir = mktempdir()
    data = reshape(collect(1.0:24.0), 2, 3, 2, 2)
    @compile_workload begin
        for (ext, T) in ((".nii.gz", Float64), (".nii", Float32))
            p = joinpath(dir, "img" * ext)
            writeimage(p, MRImage(data; meta = Dict{String,Any}("EchoTime" => [0.01, 0.02])))
            img = readimage(p; precision = T)
            m = MRImage(img.data[:, :, :, 1], img; descrip = "map")
            writeimage(joinpath(dir, "map" * ext), m)
            writeimage(joinpath(dir, "mask" * ext), img.data[:, :, :, 1] .> 2; like = img)
            samegeometry(img, readimage(joinpath(dir, "map" * ext)))
            affine(img), voxelsize(img)
            sprint(show, MIME"text/plain"(), img)
            findmeta(img, "echo"), findmeta(img, r"^Echo"), headerinfo(img), headerinfo(p)
        end
        v = joinpath(dir, "values.txt")
        write(v, "0.01 0.02\n0.03\n")
        readvalues(v)
    end
    rm(dir; recursive = true, force = true)
end
