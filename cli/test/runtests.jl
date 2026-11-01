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
        # Float32: half the memory, the same values rounded (scaled in Float64 first)
        d32, _ = loaddata(files; precision = Float32)
        @test eltype(d32) == Float32 && d32 == Float32.(d2)
        @test loaddata(path("t2.nii.gz"); precision = Float32)[1] == Float32.(data)
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

    @testset "equip fit: the same maps as the Julia API's fit" begin
        run(args...) = (o = IOBuffer(); e = IOBuffer(); c = EQUIPCLI.main(collect(String, args); io = o, err = e);
                        (code = c, out = String(take!(o)), err = String(take!(e))))
        readmap(f) = niread(f).raw
        out(name) = joinpath(tmp, "cli", name)
        data, info = loaddata(path("t2.nii.gz"))
        mask = loadmap(path("mask.nii.gz"), info) .> 0
        # T2 decay: TE from the sidecar, a mask
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--mask", path("mask.nii.gz"), "--out", out("a_"))
        @test r.code == 0 && occursin("wrote 4 maps", r.out) && isempty(r.err)
        res = fit(T2Decay(TE = te), data; mask)
        @test readmap(out("a_T2map.nii.gz")) ≈ Float32.(res.params.T2) nans = true
        car = JSON.parsefile(out("a_T2map.json"))
        @test car["Model"] == "T2Decay" && car["Acquisition"]["TE"] ≈ te && car["Mask"] == path("mask.nii.gz")
        # TE given (from a text file, a range): the same; differing from the sidecar, a warning
        write(path("te.txt"), join(te, "\n"))
        @test run("fit", "T2Decay", path("t2.nii.gz"), "--TE", "@" * path("te.txt"), "--out", out("b_")).code == 0
        @test readmap(out("b_T2map.nii.gz")) ≈ Float32.(fit(T2Decay(TE = te), data).params.T2)
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--TE", "0.02:0.02:0.16", "--out", out("c_"))
        @test r.code == 0 && occursin("--TE differs from the value in the data's sidecar", r.err)
        # parameters' flags: fixed, freed with a start, weights, Float32, standard errors
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--S0", "1000", "--T2", "free,start=0.07,lower=0.01",
                "--weights", "1,1,2,2,3,3,4,4", "--precision", "Float32", "--stderror", "--out", out("d_"))
        @test r.code == 0 && occursin("wrote 4 maps", r.out) && isfile(out("d_T2map_SE.nii.gz"))
        rd = fit(T2Decay(TE = te), Float32.(data); S0 = 1000.0, T2 = Free(start = 0.07, lower = 0.01),
                 weights = [1, 1, 2, 2, 3, 3, 4, 4], precision = Float32)
        @test readmap(out("d_T2map.nii.gz")) == rd.params.T2
        # --save, then --config (and a flag after it wins): the same fit
        cnf = path("run.cnf")
        @test run("fit", "T2Decay", path("t2.nii.gz"), "--mask", path("mask.nii.gz"), "--save", cnf, "--out", out("e_")).code == 0
        @test occursin("--TE 0.01,0.03", read(cnf, String)) && occursin("--model T2Decay", read(cnf, String))
        @test run("fit", "--config", cnf, "--out", out("f_")).code == 0
        @test isequal(readmap(out("f_T2map.nii.gz")), readmap(out("e_T2map.nii.gz")))
        # qMT: a named protocol, kmf estimated with a start, the 5-point scheme
        m5 = QMTSIR(; SIR_5POINT...)
        θ = nominal(m5, :SIR_lesion)
        Ys = simulate(m5, map(x -> x isa Real ? fill(Float64(x), 3, 2, 1) : x, θ); snr = 200, noise = :rician, rng = MersenneTwister(5))
        niwrite(path("sir.nii.gz"), NIVolume(Float32.(Ys); geometry...))
        @test run("fit", "QMTSIR", path("sir.nii.gz"), "--protocol", "SIR_5POINT", "--kmf", "free,start=10", "--out", out("g_")).code == 0
        rs = fit(m5, loaddata(path("sir.nii.gz"))[1]; kmf = Free(start = 10.0))
        @test readmap(out("g_kmf.nii.gz")) ≈ Float32.(rs.params.kmf)
        @test readmap(out("g_PSR.nii.gz")) ≈ Float32.(rs.params.PSR)
        # the same acquisition as flags
        @test run("fit", "QMTSIR", path("sir.nii.gz"), "--TI", join(SIR_5POINT.TI, ','), "--TD", join(SIR_5POINT.TD, ','),
                  "--kmf", "free,start=10", "--out", out("h_")).code == 0
        @test readmap(out("h_PSR.nii.gz")) == readmap(out("g_PSR.nii.gz"))
        # help and the model list
        r = run("fit", "QMTSIR", "--help")
        @test r.code == 0 && occursin("--kmf", r.out) && occursin("fixed at 12.5", r.out) && occursin("SIR_5POINT", r.out)
        @test occursin("T2Decay", run("models").out) && occursin("equip fit MODEL DATA", run().out)
        # mistakes: a message naming what to do, exit code 1, nothing written
        bad(args...) = (r = run(args...); r.code == 1 ? r.err : "exit $(r.code)")
        @test occursin("`--kmf` is not a flag of `equip fit T2Decay`", bad("fit", "T2Decay", path("t2.nii.gz"), "--kmf", "3"))
        @test occursin("--S0, --T2", bad("fit", "T2Decay", path("t2.nii.gz"), "--kmf", "3"))
        @test occursin("QMTSIR needs `TI`", bad("fit", "QMTSIR", path("t2.nii.gz")))
        @test occursin("has 5 observations", bad("fit", "QMTSIR", path("t2.nii.gz"), "--protocol", "SIR_5POINT"))
        @test occursin("unknown model `T2decay`", bad("fit", "T2decay", path("t2.nii.gz")))
        @test occursin("`--TE`: `abc` is not a number", bad("fit", "T2Decay", path("t2.nii.gz"), "--TE", "abc"))
        @test occursin("a range is start:step:stop", bad("fit", "T2Decay", path("t2.nii.gz"), "--TE", "0.01:0.3"))
        @test occursin("`--T2 fre`: give a number", bad("fit", "T2Decay", path("t2.nii.gz"), "--T2", "fre"))
        @test occursin("`--out` needs a value", bad("fit", "T2Decay", path("t2.nii.gz"), "--out"))
        @test occursin("give the data", bad("fit", "T2Decay"))
        @test occursin("unknown command `fitt`", bad("fitt"))
        @test occursin("not built yet", bad("crlb", "QMTSIR"))
    end
end
