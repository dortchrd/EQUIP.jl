# MRImages' tests: NIfTI round trips (values, scaling, geometry, sidecars), MRImage in another
# image's geometry, the geometry functions, values files, and MATLAB files through the extension.

using Test
using MRImages
import NIfTI

const DIR = mktempdir()
tmp(name) = joinpath(DIR, name)

# A NIfTI file written with NIfTI.jl directly: a chosen geometry, data type and scaling.
function niftifile(name, raw; voxel_size = (2.0f0, 2.0f0, 3.0f0), slope = 1.0f0, inter = 0.0f0,
                   orientation = Float32[2 0 0 -10; 0 2 0 -20; 0 0 3 5], time_step = 0.0f0,
                   xyzt_units = UInt8(18))                 # NIfTI.jl's default: mm and ms
    p = tmp(name)
    NIfTI.niwrite(p, NIfTI.NIVolume(raw; voxel_size, scl_slope = slope, scl_inter = inter,
                                    orientation, time_step, xyzt_units))
    return p
end

@testset "MRImages" begin

@testset "paths" begin
    @test filestem("a/sub-01_T2w.nii.gz") == "a/sub-01_T2w"
    @test filestem("x.nii") == "x"
    @test filestem("d/data.mat:S") == "d/data"
    @test filestem("other.txt") == "other.txt"
    @test sidecarpath("t2.nii.gz") == "t2.json"
    @test MRImages._matpath("data.mat:S") == ("data.mat", "S")
    @test MRImages._matpath("data.mat") == ("data.mat", nothing)
    @test MRImages._matpath(raw"C:\data\x.mat:S_1") == (raw"C:\data\x.mat", "S_1")
    @test MRImages._matpath("t2.nii.gz") === nothing
    @test MRImages._format("A.NII.GZ") === :nifti
    @test_throws ArgumentError readimage(tmp("x.png"))
end

@testset "NIfTI: values, geometry, sidecar" begin
    raw = reshape(Int16.(1:2*3*4*5), 2, 3, 4, 5)
    p = niftifile("scaled.nii.gz", raw; slope = 0.5f0, inter = 10.0f0)
    write(sidecarpath(p), """{"EchoTime": [0.01, 0.02, 0.03, 0.04, 0.05], "FlipAngle": 90,
                             "Nested": {"a": "b"}}""")
    img = readimage(p)
    @test img isa MRImage{Float64,4}
    @test img.data == raw .* 0.5 .+ 10                  # the scaling applied
    @test size(img) == (2, 3, 4, 5) && ndims(img) == 4
    @test hasgeometry(img)
    @test voxelsize(img) == (2.0, 2.0, 3.0)
    @test affine(img) == [2 0 0 -10; 0 2 0 -20; 0 0 3 5; 0 0 0 1]
    @test img.meta["EchoTime"] == [0.01, 0.02, 0.03, 0.04, 0.05]
    @test img.meta["FlipAngle"] == 90
    @test img.meta["Nested"] == Dict("a" => "b")
    @test img.path == p
    # precision: the Float64 values rounded
    img32 = readimage(p; precision = Float32)
    @test img32 isa MRImage{Float32,4} && img32.data == Float32.(img.data)
    # no sidecar: empty metadata
    q = niftifile("plain.nii", Float32.(raw))
    @test isempty(readimage(q).meta)
    # a slope of 0 means no scaling
    z = niftifile("zero.nii", raw; slope = 0.0f0, inter = 5.0f0)
    @test readimage(z).data == raw
    # the summary
    s = sprint(show, MIME"text/plain"(), img)
    @test occursin("2×3×4×5 Float64", s) && occursin("2×2×3 mm voxels", s) &&
          occursin("EchoTime", s)
    @test sprint(show, img) == "MRImage(2×3×4×5 Float64, $(repr(p)))"
end

@testset "writing, in another image's geometry" begin
    raw = reshape(Int16.(1:2*3*4*5), 2, 3, 4, 5)
    img = readimage(niftifile("data.nii.gz", raw; slope = 2.0f0))
    map3 = rand(2, 3, 4)
    m = MRImage(map3, img; meta = Dict("Units" => "s"), descrip = "T2 (s)")
    @test m.header.scl_slope == 1 && m.header.scl_inter == 0
    @test img.header.scl_slope == 2                     # the source's header is not changed
    f = writeimage(tmp("out/T2map.nii.gz"), m)          # the directory is made
    @test f == tmp("out/T2map.nii.gz") && isfile(tmp("out/T2map.json"))
    back = readimage(f)
    @test back.data == map3                             # Float64 written as Float64
    @test samegeometry(back, img)
    @test affine(back) == affine(img)
    @test back.meta == Dict("Units" => "s")
    @test String(collect(back.header.descrip[1:6])) == "T2 (s)"
    # Float32 values written as Float32
    writeimage(tmp("f32.nii"), Float32.(map3); like = img)
    r = NIfTI.niread(tmp("f32.nii"))
    @test eltype(r.raw) == Float32 && r.raw == Float32.(map3)
    # a mask: true/false as UInt8
    writeimage(tmp("mask.nii.gz"), map3 .> 0.5; like = img)
    r = NIfTI.niread(tmp("mask.nii.gz"))
    @test eltype(r.raw) == UInt8 && r.raw == (map3 .> 0.5)
    @test readimage(tmp("mask.nii.gz")).data == (map3 .> 0.5)
    # written back as it was read: the values (scaling applied, so none written), the geometry
    writeimage(tmp("copy.nii"), img)
    @test readimage(tmp("copy.nii")).data == img.data
    @test samegeometry(readimage(tmp("copy.nii")), img)
    # without geometry: NIfTI.jl's default (1 mm, at the origin)
    writeimage(tmp("nogeom.nii"), MRImage(map3))
    @test voxelsize(readimage(tmp("nogeom.nii"))) == (1.0, 1.0, 1.0)
    # the wrong size for the geometry
    @test_throws DimensionMismatch MRImage(rand(3, 3, 4), img)
    @test_throws ArgumentError writeimage(tmp("x.mat"), MRImage(map3))
    @test_throws ArgumentError readimage(tmp("missing.nii.gz"))
end

@testset "samegeometry" begin
    a = readimage(niftifile("g1.nii", rand(Float32, 2, 3, 4)))
    b = readimage(niftifile("g2.nii", rand(Float32, 2, 3, 4)))
    c = readimage(niftifile("g3.nii", rand(Float32, 2, 3, 4); voxel_size = (1.0f0, 2.0f0, 3.0f0)))
    d = readimage(niftifile("g4.nii", rand(Float32, 2, 3, 4);
                            orientation = Float32[2 0 0 -11; 0 2 0 -20; 0 0 3 5]))
    e = readimage(niftifile("g5.nii", rand(Float32, 2, 3, 5)))
    @test samegeometry(a, b)
    @test !samegeometry(a, c)                   # voxel size
    @test !samegeometry(a, d)                   # position
    @test !samegeometry(a, e)                   # size
    @test !samegeometry(a, MRImage(rand(2, 3, 4)))
    @test samegeometry(MRImage(rand(2, 3, 4)), MRImage(rand(2, 3, 4, 2)))
    @test_throws ArgumentError affine(MRImage(rand(2, 2)))
    @test_throws ArgumentError voxelsize(MRImage(rand(2, 2)))
end

@testset "sidecars" begin
    p = tmp("s.nii.gz")
    writesidecar(p, Dict("EchoTime" => 0.03, "List" => [1, 2]))
    @test readsidecar(p) == Dict("EchoTime" => 0.03, "List" => [1, 2])
    @test readsidecar(tmp("s.json")) == readsidecar(p)
    @test readsidecar(tmp("none.nii")) == Dict{String,Any}()
    write(tmp("bad.json"), "{not json")
    @test_throws ArgumentError readsidecar(tmp("bad.json"))
    write(tmp("list.json"), "[1, 2]")
    @test_throws ArgumentError readsidecar(tmp("list.json"))
end

@testset "values, .bval, .bvec" begin
    write(tmp("ti.txt"), "# TIs (s)\n0.01\n0.05, 0.1\n\n0.2 0.5\n")
    @test readvalues(tmp("ti.txt")) == [0.01, 0.05, 0.1, 0.2, 0.5]
    write(tmp("dwi.bval"), "0 1000 1000 2000\n")
    @test readvalues(tmp("dwi.bval")) == [0, 1000, 1000, 2000]
    write(tmp("bad.txt"), "0.1\nabc\n")
    @test_throws ArgumentError readvalues(tmp("bad.txt"))
    write(tmp("empty.txt"), "# nothing\n")
    @test_throws ArgumentError readvalues(tmp("empty.txt"))
    @test_throws ArgumentError readvalues(tmp("none.txt"))
    write(tmp("dwi.bvec"), "1 0 0 0.6\n0 1 0 0.8\n0 0 1 0\n")
    @test readbvec(tmp("dwi.bvec")) == [1 0 0 0.6; 0 1 0 0.8; 0 0 1 0]
    write(tmp("rows.bvec"), "1 0 0\n0 1 0\n0 0 1\n0.6 0.8 0\n")
    @test readbvec(tmp("rows.bvec")) == [1 0 0 0.6; 0 1 0 0.8; 0 0 1 0]
    write(tmp("odd.bvec"), "1 0\n0 1\n")
    @test_throws ArgumentError readbvec(tmp("odd.bvec"))
end

@testset "findmeta, headerinfo" begin
    raw = reshape(Int16.(1:2*3*4*5), 2, 3, 4, 5)
    p = niftifile("info.nii.gz", raw; slope = 0.5f0, inter = 10.0f0)
    write(sidecarpath(p), """{"EchoTime": 0.012, "EchoTrainLength": 8, "RepetitionTime": 3.2,
                             "FlipAngle": 90, "Coil": {"EchoSpacing": 0.006, "Name": "head"}}""")
    img = readimage(p)
    # by a string anywhere in the name, any case; nested entries by their path; sorted
    @test findmeta(img, "echo") == ["Coil.EchoSpacing" => 0.006, "EchoTime" => 0.012,
                                    "EchoTrainLength" => 8]
    @test first.(findmeta(img, "TIME")) == ["EchoTime", "RepetitionTime"]
    @test first.(findmeta(img, r"^Echo")) == ["Coil.EchoSpacing", "EchoTime", "EchoTrainLength"]
    @test first.(findmeta(img, r"^Echo(?!Spacing)")) == ["EchoTime", "EchoTrainLength"]
    @test findmeta(img, "coil") == ["Coil" => Dict("EchoSpacing" => 0.006, "Name" => "head")]
    @test isempty(findmeta(img, "nothing"))
    @test findmeta(img.meta, "flip") == ["FlipAngle" => 90]
    # the header, readable; the same from the path alone
    h = Dict(headerinfo(img))
    @test h["size"] == "2×3×4×5" && h["voxel size"] == "2×2×3 mm"
    @test startswith(h["axes"], "R, A, S")
    @test h["affine"] == "2 0 0 -10; 0 2 0 -20; 0 0 3 5"
    @test h["stored as"] == "Int16, value = 0.5 × stored + 10"
    @test occursin("scanner", h["sform, qform"]) && h["format"] == "NIfTI-1"
    @test headerinfo(p) == headerinfo(img)
    # flipped axes: L, P, I
    q = niftifile("flipped.nii", rand(Float32, 2, 2, 2); orientation = Float32[-2 0 0 0; 0 -2 0 0; 0 0 -3 0])
    @test startswith(Dict(headerinfo(q))["axes"], "L, P, I")
    @test Dict(headerinfo(q))["stored as"] == "Float32, no scaling"
    @test Dict(headerinfo(MRImage(rand(3, 4))))["geometry"] == "none (an array from a MATLAB file)"
    @test_throws ArgumentError headerinfo(tmp("none.nii"))
    # Units the header leaves unknown (code 0, as MATLAB's niftiwrite and other tools leave them):
    # `headerinfo` used to fail on an unknown time unit (NIfTI.jl's `time_step` indexes its table
    # with it: an owner's SIR data, 2026-11-22, made `equip info` fail), and to show a 1-mm voxel
    # as 0.001 mm (NIfTI.jl's `voxel_size` takes unknown for microns); unknown is taken as mm
    u = niftifile("units0.nii", rand(Float32, 2, 2, 2, 3); voxel_size = (1.0f0, 1.0f0, 1.0f0),
                  time_step = 2.5f0, xyzt_units = 0x00)
    hu = Dict(headerinfo(u))
    @test startswith(hu["voxel size"], "1×1×1 mm") && occursin("unknown", hu["voxel size"])
    @test startswith(hu["time step"], "2.5 ") && occursin("unknown", hu["time step"])
    @test voxelsize(readimage(u)) == (1.0, 1.0, 1.0)
    # known units: the time step in seconds from s and from ms, a voxel size in µm in mm
    s = niftifile("units_s.nii", rand(Float32, 2, 2, 2, 3); time_step = 2.5f0, xyzt_units = 0x0a)
    @test Dict(headerinfo(s))["time step"] == "2.5 s (pixdim[5])"
    ms = niftifile("units_ms.nii", rand(Float32, 2, 2, 2, 3); time_step = 2500.0f0)
    @test Dict(headerinfo(ms))["time step"] == "2.5 s (pixdim[5])"
    um = niftifile("units_um.nii", rand(Float32, 2, 2, 2); voxel_size = (500.0f0, 500.0f0, 1000.0f0), xyzt_units = 0x03)
    @test voxelsize(readimage(um)) == (0.5, 0.5, 1.0)
end

@testset "MATLAB files need MAT" begin
    # before `using MAT`: a message that says how
    @test !MRImages._matloaded()
    err = try readimage(tmp("d.mat:S")); nothing catch e; e end
    @test err isa ArgumentError && occursin("using MAT", err.msg)
    @test_throws ArgumentError readvalues(tmp("d.mat:TE"))
    @test_throws ArgumentError writemat(tmp("d.mat"), Dict("S" => rand(2)))
end

# The extension, loaded here (as a user's `using MAT` would): a new world, so the tests that use
# it run at top level, after the load.
@eval using MAT
@test MRImages._matloaded()

@testset "MATLAB files" begin
    S = rand(4, 5, 3)
    p = writemat(tmp("data.mat"), Dict("S" => S, "TE" => [0.01 0.02 0.03], "n" => 7.0,
                                       "u" => UInt16[1 2; 3 4], "s" => "text"))
    @test p == tmp("data.mat")
    img = readimage(tmp("data.mat:S"))
    @test img isa MRImage{Float64,3} && img.data == S
    @test !hasgeometry(img) && isempty(img.meta)
    @test occursin("no geometry", sprint(show, MIME"text/plain"(), img))
    @test readimage(tmp("data.mat:S"); precision = Float32).data == Float32.(S)
    @test readimage(tmp("data.mat:u")).data == [1 2; 3 4]           # uint16 images
    @test readvalues(tmp("data.mat:TE")) == [0.01, 0.02, 0.03]      # a row vector, as a vector
    @test readvalues(tmp("data.mat:n")) == [7.0]
    @test readmat(tmp("data.mat"), "S") == S
    @test Set(keys(readmat(tmp("data.mat")))) == Set(["S", "TE", "n", "u", "s"])
    # mistakes
    err = try readimage(tmp("data.mat")); nothing catch e; e end    # several variables
    @test err isa ArgumentError && occursin("data.mat:NAME", err.msg) && occursin("TE", err.msg)
    err = try readimage(tmp("data.mat:X")); nothing catch e; e end
    @test err isa ArgumentError && occursin("no variable `X`", err.msg)
    @test_throws ArgumentError readimage(tmp("data.mat:s"))        # a string
    @test_throws ArgumentError readvalues(tmp("data.mat"))          # no variable named
    @test_throws ArgumentError readimage(tmp("nofile.mat:S"))
    @test_throws ArgumentError writemat(tmp("bad.mat"), Dict("1x" => 1.0))
    # one variable: no name needed
    writemat(tmp("one.mat"), Dict("data" => S))
    @test readimage(tmp("one.mat")).data == S
end

end
