# Tests of EQUIPCLI: julia --project=cli -e 'using Pkg; Pkg.test()'
using EQUIPCLI, EQUIP, Test, Random
import JSON
using NIfTI: NIVolume, niwrite, niread

msg(f) = try f(); "" catch e; sprint(showerror, e) end
tmp = mktempdir()
path(name) = joinpath(tmp, name)

# A small T2 dataset: 6×5×2 voxels, 8 echoes, T2 from 0.04 to 0.12 s, with a geometry that is not
# the default (2 × 2 × 3 mm voxels, an offset), written as NIfTI the way a scanner pipeline would.
te = collect(0.01:0.02:0.15)
T2map = reshape(range(0.04, 0.12; length = 60), 6, 5, 2) |> collect
S0map = fill(1000.0, 6, 5, 2)
m = T2Decay(TE = te)
Y = simulate(m, (S0 = S0map, T2 = T2map); snr = 200, noise = :gaussian, rng = MersenneTwister(1))
geometry = (voxel_size = (2.0f0, 2.0f0, 3.0f0),
            orientation = Float32[2 0 0 -10; 0 2 0 -20; 0 0 3 5])
writejson(p, d) = open(io -> JSON.json(io, d), p, "w")

@testset "EQUIPCLI" begin
    @testset "loaddata: NIfTI files and their BIDS sidecars" begin
        # One 4-D file, its sidecar giving one EchoTime per volume
        niwrite(path("t2.nii.gz"), NIVolume(Float32.(Y); geometry...))
        writejson(path("t2.json"), Dict("EchoTime" => te, "RepetitionTime" => 2.0))
        data, info = loaddata(path("t2.nii.gz"))
        @test size(data) == (6, 5, 2, 8) && eltype(data) == Float64
        @test data ≈ Float32.(Y)
        @test info.spatial == (6, 5, 2) && info.found.TE ≈ te && info.found.TR == fill(2.0, 8)
        @test !haskey(info.found, :TI)
        # One 3-D file per echo (as dcm2niix writes a multi-echo series), each with its sidecar,
        # stacked in the order given; the intensity scaling applied
        files = [path("echo$k.nii") for k in 1:8]
        for k in 1:8
            raw = round.(Int16, Y[:, :, :, k] ./ 0.5)
            niwrite(files[k], NIVolume(raw; geometry..., scl_slope = 0.5f0, scl_inter = 0.0f0))
            writejson(replace(files[k], ".nii" => ".json"), Dict("EchoTime" => te[k]))
        end
        d2, i2 = loaddata(files)
        @test size(d2) == (6, 5, 2, 8) && maximum(abs.(d2 .- Y)) <= 0.25
        @test i2.found.TE ≈ te
        # Diffusion: an FSL .bval file beside the data
        niwrite(path("dwi.nii"), NIVolume(Float32.(Y[:, :, :, 1:4]); geometry...))
        write(path("dwi.bval"), "0 500 1000 2000\n")
        @test loaddata(path("dwi.nii"))[2].found.b == [0.0, 500, 1000, 2000]
        # Mistakes
        niwrite(path("other.nii"), NIVolume(Float32.(Y[:, :, :, 1]); voxel_size = (1.0f0, 1.0f0, 1.0f0)))
        niwrite(path("small.nii"), NIVolume(zeros(Float32, 4, 4, 2); geometry...))
        @test occursin("different geometry", msg(() -> loaddata([files[1], path("other.nii")])))
        @test occursin("same size", msg(() -> loaddata([files[1], path("small.nii")])))
        @test occursin("one image is one observation", msg(() -> loaddata(files[1])))
        @test occursin("only NIfTI", msg(() -> loaddata(path("t2.json"))))
        @test occursin("no such file", msg(() -> loaddata(path("missing.nii.gz"))))
        writejson(path("echo1.json"), Dict("EchoTime" => [0.01, 0.02]))
        @test occursin("EchoTime has 2 values for 1 images", msg(() -> loaddata(files)))
    end

    @testset "loadmap, readvalues" begin
        _, info = loaddata(path("t2.nii.gz"))
        mask = T2map .> 0.06
        niwrite(path("mask.nii.gz"), NIVolume(UInt8.(mask); geometry...))
        @test loadmap(path("mask.nii.gz"), info) == mask
        @test occursin("different geometry", msg(() -> loadmap(path("other.nii"), info)))
        @test occursin("the data are 6×5×2", msg(() -> loadmap(path("small.nii"), info)))
        write(path("ti.txt"), "# TIs (s)\n0.01\n0.05, 0.056\n\n0.277\t0.843\n")
        @test readvalues(path("ti.txt")) == [0.01, 0.05, 0.056, 0.277, 0.843]
        write(path("bad.txt"), "0.01\nten\n")
        @test occursin("line 2: `ten` is not a number", msg(() -> readvalues(path("bad.txt"))))
    end

    @testset "savemaps: the maps in the data's geometry, with sidecars" begin
        data, info = loaddata(path("t2.nii.gz"))
        mask = loadmap(path("mask.nii.gz"), info) .> 0
        res = fit(T2Decay(TE = info.found.TE), data; mask)
        files = savemaps(joinpath(tmp, "out", "sub-01_"), res, info; stderror = true,
                         sidecar = Dict("Model" => "T2Decay"))
        @test basename.(files) == ["sub-01_S0map.nii.gz", "sub-01_T2map.nii.gz", "sub-01_status.nii.gz",
                                   "sub-01_SSE.nii.gz", "sub-01_S0map_SE.nii.gz", "sub-01_T2map_SE.nii.gz"]
        t2 = niread(files[2])
        @test size(t2) == (6, 5, 2)
        @test t2.raw ≈ Float32.(res.params.T2) nans = true
        @test all(isnan, t2.raw[.!mask]) && maximum(abs.(t2.raw[mask] .- T2map[mask])) < 0.01
        # the geometry is the data's
        h, hd = t2.header, info.header
        @test (h.pixdim[2:4], h.srow_x, h.srow_y, h.srow_z, h.sform_code) ==
              (hd.pixdim[2:4], hd.srow_x, hd.srow_y, hd.srow_z, hd.sform_code)
        @test h.scl_slope == 1 && h.scl_inter == 0
        @test niread(files[3]).raw == Float32.(Int.(res.status))
        car = JSON.parsefile(replace(files[2], ".nii.gz" => ".json"))
        @test car["Map"] == "T2map" && car["Model"] == "T2Decay"
    end
end
