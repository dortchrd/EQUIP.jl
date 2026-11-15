# Tests of EQUIPCLI: julia --project=cli -e 'using Pkg; Pkg.test()'
using EQUIPCLI, EQUIP, Test, Random
using Random: Xoshiro
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

# Run a command in this process: its exit code and what it printed.
run(args...) = (o = IOBuffer(); e = IOBuffer(); c = EQUIPCLI.equip(collect(String, args); io = o, err = e);
                (code = c, out = String(take!(o)), err = String(take!(e))))
bad(args...) = (r = run(args...); r.code == 1 ? r.err : "exit $(r.code): $(r.err)")
# A written map's values as stored (Float32), read directly (EQUIP's `readmap` checks them against a dataset's `info`).
rawmap(f) = niread(f).raw
# A TSV table: its header and rows, as strings.
function table(f)
    lines = split.(readlines(f), '\t')
    return [Dict(zip(lines[1], l)) for l in lines[2:end]]
end

@testset "EQUIPCLI" begin
    @testset "readdata: NIfTI files and their BIDS sidecars" begin
        # One 4-D file, its sidecar giving one EchoTime per volume
        niwrite(path("t2.nii.gz"), NIVolume(Float32.(Y); geometry...))
        writejson(path("t2.json"), Dict("EchoTime" => te, "RepetitionTime" => 2.0))
        data, info = readdata(path("t2.nii.gz"))
        @test size(data) == (6, 5, 2, 8) && eltype(data) == Float64
        @test data ≈ Float32.(Y)
        @test info.spatial == (6, 5, 2) && info.found[:TE] ≈ te && info.found[:TR] == fill(2.0, 8)
        @test !haskey(info.found, :TI)
        # One 3-D file per echo (as dcm2niix writes a multi-echo series), each with its sidecar,
        # stacked in the order given; the intensity scaling applied
        files = [path("echo$k.nii") for k in 1:8]
        for k in 1:8
            raw = round.(Int16, Y[:, :, :, k] ./ 0.5)
            niwrite(files[k], NIVolume(raw; geometry..., scl_slope = 0.5f0, scl_inter = 0.0f0))
            writejson(replace(files[k], ".nii" => ".json"), Dict("EchoTime" => te[k]))
        end
        d2, i2 = readdata(files)
        @test size(d2) == (6, 5, 2, 8) && maximum(abs.(d2 .- Y)) <= 0.25
        @test i2.found[:TE] ≈ te
        # Float32: half the memory, the same values rounded (scaled in Float64 first)
        d32, _ = readdata(files; precision = Float32)
        @test eltype(d32) == Float32 && d32 == Float32.(d2)
        @test readdata(path("t2.nii.gz"); precision = Float32)[1] == Float32.(data)
        # Diffusion: an FSL .bval file beside the data
        niwrite(path("dwi.nii"), NIVolume(Float32.(Y[:, :, :, 1:4]); geometry...))
        write(path("dwi.bval"), "0 500 1000 2000\n")
        @test readdata(path("dwi.nii"))[2].found[:b] == [0.0, 500, 1000, 2000]
        # Mistakes
        niwrite(path("other.nii"), NIVolume(Float32.(Y[:, :, :, 1]); voxel_size = (1.0f0, 1.0f0, 1.0f0)))
        niwrite(path("small.nii"), NIVolume(zeros(Float32, 4, 4, 2); geometry...))
        @test occursin("different geometry", msg(() -> readdata([files[1], path("other.nii")])))
        @test occursin("same size", msg(() -> readdata([files[1], path("small.nii")])))
        @test occursin("one image is one observation", msg(() -> readdata(files[1])))
        @test occursin("give NIfTI files", msg(() -> readdata(path("t2.json"))))
        @test occursin("no such file", msg(() -> readdata(path("missing.nii.gz"))))
        writejson(path("echo1.json"), Dict("EchoTime" => [0.01, 0.02]))
        @test occursin("EchoTime has 2 values for 1 images", msg(() -> readdata(files)))
    end

    @testset "readmap, readvalues" begin
        _, info = readdata(path("t2.nii.gz"))
        mask = T2map .> 0.06
        niwrite(path("mask.nii.gz"), NIVolume(UInt8.(mask); geometry...))
        @test readmap(path("mask.nii.gz"), info) == mask
        @test occursin("different geometry", msg(() -> readmap(path("other.nii"), info)))
        @test occursin("the data are 6×5×2", msg(() -> readmap(path("small.nii"), info)))
        write(path("ti.txt"), "# TIs (s)\n0.01\n0.05, 0.056\n\n0.277\t0.843\n")
        @test readvalues(path("ti.txt")) == [0.01, 0.05, 0.056, 0.277, 0.843]
        write(path("bad.txt"), "0.01\nten\n")
        @test occursin("line 2: `ten` is not a number", msg(() -> readvalues(path("bad.txt"))))
    end

    @testset "writemaps: the maps in the data's geometry, with sidecars" begin
        data, info = readdata(path("t2.nii.gz"))
        mask = readmap(path("mask.nii.gz"), info) .> 0
        res = fit(T2Decay(TE = info.found[:TE]), data; mask)
        files = writemaps(joinpath(tmp, "out", "sub-01_"), res, info; stderror = true,
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
        out(name) = joinpath(tmp, "cli", name)
        data, info = readdata(path("t2.nii.gz"))
        mask = readmap(path("mask.nii.gz"), info) .> 0
        # T2 decay: TE from the sidecar, a mask
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--mask", path("mask.nii.gz"), "--out", out("a_"))
        @test r.code == 0 && occursin("wrote 4 maps", r.out) && isempty(r.err)
        res = fit(T2Decay(TE = te), data; mask)
        @test rawmap(out("a_T2map.nii.gz")) ≈ Float32.(res.params.T2) nans = true
        car = JSON.parsefile(out("a_T2map.json"))
        @test car["Model"] == "T2Decay" && car["Acquisition"]["TE"] ≈ te && car["Mask"] == path("mask.nii.gz")
        # TE given (from a text file, a range): the same; differing from the sidecar, a warning
        write(path("te.txt"), join(te, "\n"))
        @test run("fit", "T2Decay", path("t2.nii.gz"), "--TE", "@" * path("te.txt"), "--out", out("b_")).code == 0
        @test rawmap(out("b_T2map.nii.gz")) ≈ Float32.(fit(T2Decay(TE = te), data).params.T2)
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--TE", "0.02:0.02:0.16", "--out", out("c_"))
        @test r.code == 0 && occursin("--TE differs from the value in the data's sidecar", r.err)
        # parameters' flags: fixed, freed with a start, weights, Float32, standard errors
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--S0", "1000", "--T2", "free,start=0.07,lower=0.01",
                "--weights", "1,1,2,2,3,3,4,4", "--precision", "Float32", "--stderror", "--out", out("d_"))
        @test r.code == 0 && occursin("wrote 4 maps", r.out) && isfile(out("d_T2map_SE.nii.gz"))
        rd = fit(T2Decay(TE = te), Float32.(data); S0 = 1000.0, T2 = Free(start = 0.07, lower = 0.01),
                 weights = [1, 1, 2, 2, 3, 3, 4, 4], precision = Float32)
        @test rawmap(out("d_T2map.nii.gz")) == rd.params.T2
        # --save, then --config (and a flag after it wins): the same fit
        cnf = path("run.cnf")
        @test run("fit", "T2Decay", path("t2.nii.gz"), "--mask", path("mask.nii.gz"), "--save", cnf, "--out", out("e_")).code == 0
        @test occursin("--TE 0.01,0.03", read(cnf, String)) && occursin("--model T2Decay", read(cnf, String))
        @test run("fit", "--config", cnf, "--out", out("f_")).code == 0
        @test isequal(rawmap(out("f_T2map.nii.gz")), rawmap(out("e_T2map.nii.gz")))
        # qMT: a named protocol, kmf estimated with a start, the 5-point scheme
        m5 = QMTSIR(; SIR_5POINT...)
        θ = nominal(m5, :SIR_lesion)
        Ys = simulate(m5, map(x -> x isa Real ? fill(Float64(x), 3, 2, 1) : x, θ); snr = 200, noise = :rician, rng = MersenneTwister(5))
        niwrite(path("sir.nii.gz"), NIVolume(Float32.(Ys); geometry...))
        @test run("fit", "QMTSIR", path("sir.nii.gz"), "--protocol", "SIR_5POINT", "--kmf", "free,start=10", "--out", out("g_")).code == 0
        rs = fit(m5, readdata(path("sir.nii.gz"))[1]; kmf = Free(start = 10.0))
        @test rawmap(out("g_kmf.nii.gz")) ≈ Float32.(rs.params.kmf)
        @test rawmap(out("g_PSR.nii.gz")) ≈ Float32.(rs.params.PSR)
        # the same acquisition as flags
        @test run("fit", "QMTSIR", path("sir.nii.gz"), "--TI", join(SIR_5POINT.TI, ','), "--TD", join(SIR_5POINT.TD, ','),
                  "--kmf", "free,start=10", "--out", out("h_")).code == 0
        @test rawmap(out("h_PSR.nii.gz")) == rawmap(out("g_PSR.nii.gz"))
        # magnitude data at low SNR: the Rician fit, σ a number or a map (each channel's noise SD:
        # S0/SNR = 1000/20), method ml by default or mean; recorded in the sidecar as typed
        Yr = simulate(m, (S0 = S0map, T2 = T2map); snr = 20, noise = :rician, rng = MersenneTwister(7))
        niwrite(path("t2r.nii.gz"), NIVolume(Float32.(Yr); geometry...))
        niwrite(path("sigma.nii.gz"), NIVolume(fill(50.0f0, 6, 5, 2); geometry...))
        dr = readdata(path("t2r.nii.gz"))[1]
        rician(args...) = run("fit", "T2Decay", path("t2r.nii.gz"), "--TE", join(te, ','), args...)
        @test rician("--likelihood", "rician", "--sigma", "50", "--out", out("r_")).code == 0
        rr = fit(T2Decay(TE = te), dr; likelihood = :rician, σ = 50.0)
        @test rawmap(out("r_T2map.nii.gz")) ≈ Float32.(rr.params.T2) nans = true
        car = JSON.parsefile(out("r_T2map.json"))
        @test car["Likelihood"] == "rician" && car["Sigma"] == "50" && car["Method"] == "ml"
        @test rician("--likelihood", "rician", "--sigma", path("sigma.nii.gz"), "--method", "mean", "--out", out("s_")).code == 0
        rm_ = fit(T2Decay(TE = te), dr; likelihood = :rician, σ = fill(50.0, 6, 5, 2), method = :mean)
        @test rawmap(out("s_T2map.nii.gz")) ≈ Float32.(rm_.params.T2) nans = true
        @test JSON.parsefile(out("s_T2map.json"))["Sigma"] == path("sigma.nii.gz")
        @test JSON.parsefile(out("a_T2map.json"))["Likelihood"] == "gaussian"
        @test occursin("needs `--sigma`", bad("fit", "T2Decay", path("t2.nii.gz"), "--likelihood", "rician"))
        @test occursin("`--sigma` is for `--likelihood rician`", bad("fit", "T2Decay", path("t2.nii.gz"), "--sigma", "50"))
        @test occursin("`--method`", bad("fit", "T2Decay", path("t2.nii.gz"), "--likelihood", "rician", "--sigma", "50", "--method", "mle"))
        @test occursin("`--likelihood`", bad("fit", "T2Decay", path("t2.nii.gz"), "--likelihood", "poisson"))
        # help and the model list
        # a command's help without a model: the models, and how to see one's flags
        r = run("fit", "--help")
        @test r.code == 0 && occursin("QMTSIR", r.out) && occursin("equip fit MODEL --help", r.out)
        @test run("crlb", "--help").code == 0 && run("design", "--help").code == 0
        @test occursin("give the model", bad("fit"))
        r = run("fit", "QMTSIR", "--help")
        @test r.code == 0 && occursin("--kmf", r.out) && occursin("fixed at 12.5", r.out) && occursin("SIR_5POINT", r.out)
        @test occursin("--likelihood rician --sigma", r.out)
        @test occursin("T2Decay", run("models").out) && occursin("equip fit MODEL DATA", run().out)
        # mistakes: a message naming what to do, exit code 1, nothing written
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
    end

    @testset "equip fit DTI and DKI: the gradients from the .bval and .bvec beside the data" begin
        out(name) = joinpath(tmp, "dwi", name)
        # a b = 0 image and two shells of 15 directions, written as dcm2niix would: dti.nii.gz,
        # dti.bval (b in s/mm²), dti.bvec (three rows, the gradient directions)
        g = directionset(15)
        bv = [0.0; fill(1000.0, 15); fill(2000.0, 15)]
        G = reduce(hcat, [[[0.0, 0.0, 0.0]]; collect.(g); collect.(g)])
        acq = diffusionencoding(bv, G)
        θ = (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.1e-3, Dxz = 0.0, Dyz = 0.05e-3)
        Yd = simulate(DTI(; acq...), map(x -> fill(x, 3, 2, 1), θ); snr = 100, noise = :gaussian,
                      rng = MersenneTwister(9))
        niwrite(path("dti.nii.gz"), NIVolume(Float32.(Yd); geometry...))
        write(path("dti.bval"), join(bv, ' ') * "\n")
        write(path("dti.bvec"), join((join(G[i, :], ' ') for i in 1:3), "\n") * "\n")
        data, info = readdata(path("dti.nii.gz"))
        # DTI: the same maps as the Julia API's fit, the derived ones (MD, FA, AD, RD, V1) too
        r = run("fit", "DTI", path("dti.nii.gz"), "--out", out("a_"))
        @test r.code == 0 && isempty(r.err)
        res = fit(DTI(; acq...), data)
        @test rawmap(out("a_Dxx.nii.gz")) ≈ Float32.(res.params.Dxx)
        @test rawmap(out("a_Dxy.nii.gz")) ≈ Float32.(res.params.Dxy)
        d = derived(res)
        for k in (:MD, :FA, :AD, :RD)
            @test rawmap(out("a_$k.nii.gz")) ≈ Float32.(d[k])
        end
        @test all(isfile(out("a_V1$c.nii.gz")) for c in "xyz")
        @test JSON.parsefile(out("a_FA.json"))["Model"] == "DTI"
        # DKI: the same, with its kurtosis indices
        r = run("fit", "DKI", path("dti.nii.gz"), "--out", out("b_"))
        @test r.code == 0 && isempty(r.err)
        rk = fit(DKI(; acq...), data)
        @test rawmap(out("b_MK.nii.gz")) ≈ Float32.(derived(rk).MK)
        @test rawmap(out("b_V1111.nii.gz")) ≈ Float32.(rk.params.V1111)
        # help: the gradients from the files; the start described
        r = run("fit", "DTI", "--help")
        @test r.code == 0 && occursin("--gx", r.out) && occursin(".bval and .bvec", r.out) &&
              occursin("the log-linear fit to the data", r.out)
        @test occursin("DTI", run("models").out) && occursin("DKI", run("models").out)
        # no .bvec beside the data: a message saying where the directions come from
        cp(path("dti.nii.gz"), path("nobvec.nii.gz"); force = true)
        cp(path("dti.bval"), path("nobvec.bval"); force = true)
        e = bad("fit", "DTI", path("nobvec.nii.gz"))
        @test occursin("DTI needs `gx`", e) && occursin(".bval and .bvec files beside the data", e)
    end

    @testset "equip fit with MATLAB files: MAT loaded when named, maps back as .mat" begin
        out(name) = joinpath(tmp, "mat", name)
        MRI = EQUIP.MRImages
        # MAT is loaded by the first command that names a .mat file, not before
        @test !MRI._matloaded()
        @test run("fit", "T2Decay", path("t2.nii.gz"), "--out", out("n_")).code == 0
        @test !MRI._matloaded()
        @test occursin("no such file", bad("fit", "T2Decay", path("none.mat:S"), "--TE", "0.01,0.02"))
        @test MRI._matloaded()
        # voxels already masked, nvoxels×N, as MATLAB users often keep them; TE and the mask in
        # the same file
        mask = T2map .> 0.06
        MRI.writemat(path("t2.mat"), Dict("S" => reshape(Y, 60, 8), "TE" => collect(te'),
                                          "mask" => Float64.(vec(mask)), "B" => fill(1000.0, 60)))
        r = run("fit", "T2Decay", path("t2.mat:S"), "--TE", "@" * path("t2.mat:TE"),
                "--mask", path("t2.mat:mask"), "--out", out("a_"))
        @test r.code == 0 && occursin("wrote the maps to a_maps.mat", r.out)
        res = fit(T2Decay(TE = te), reshape(Y, 60, 8); mask = vec(mask))
        maps = MRI.readmat(out("a_maps.mat"))
        @test size(maps["T2map"]) == (60,) && isequal(maps["T2map"], Float32.(res.params.T2))
        car = JSON.parsefile(out("a_maps.json"))
        @test car["Maps"] == ["S0map", "T2map", "status", "SSE"] && car["Model"] == "T2Decay"
        @test car["Data"] == [path("t2.mat:S")] && car["Acquisition"]["TE"] ≈ te
        # a fixed parameter's map from the file; NIfTI maps in another image's geometry (--like)
        r = run("fit", "T2Decay", path("t2.mat:S"), "--TE", "@" * path("t2.mat:TE"),
                "--S0", path("t2.mat:B"), "--out", out("b_"))
        @test r.code == 0 && size(MRI.readmat(out("b_maps.mat"))["T2map"]) == (60,)
        MRI.writemat(path("t2_4d.mat"), Dict("S" => Y))
        r = run("fit", "T2Decay", path("t2_4d.mat"), "--TE", join(te, ','),
                "--like", path("t2.nii.gz"), "--out", out("c_"))
        @test r.code == 0 && occursin("wrote 4 maps", r.out)
        @test rawmap(out("c_T2map.nii.gz")) == Float32.(fit(T2Decay(TE = te), Y).params.T2)
        @test MRI.samegeometry(readimage(out("c_T2map.nii.gz")), readimage(path("t2.nii.gz")))
        # --format: NIfTI data's maps as one .mat too (what MATLAB's `load` reads without a
        # toolbox: matlab/equip_fit.m), or only; the same numbers as the NIfTI maps
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--format", "both", "--out", out("d_"))
        @test r.code == 0 && occursin("wrote 4 maps", r.out) && occursin("wrote the maps to d_maps.mat", r.out)
        @test isequal(MRI.readmat(out("d_maps.mat"))["T2map"], rawmap(out("d_T2map.nii.gz")))
        r = run("fit", "T2Decay", path("t2.nii.gz"), "--format", "mat", "--out", out("e_"))
        @test r.code == 0 && isfile(out("e_maps.mat")) && !isfile(out("e_T2map.nii.gz"))
        @test occursin("--like", bad("fit", "T2Decay", path("t2_4d.mat"), "--TE", join(te, ','), "--format", "nifti"))
        @test occursin("`--format`", bad("fit", "T2Decay", path("t2.nii.gz"), "--format", "csv"))
        # mistakes
        @test occursin("name the one to read", bad("fit", "T2Decay", path("t2.mat"), "--TE", "0.01"))
        @test occursin("the data's images are 60", bad("fit", "T2Decay", path("t2.mat:S"),
                       "--TE", "@" * path("t2.mat:TE"), "--mask", path("t2_4d.mat:S")))
    end

    @testset "equip info: the header, the sidecar, what EQUIP reads; --find; .mat" begin
        writejson(path("info.json"), Dict("EchoTime" => te, "RepetitionTime" => 2.0,
                                          "Manufacturer" => "Siemens", "SeriesDescription" => "ME-SE",
                                          "Coil" => Dict("EchoSpacing" => 0.02)))
        niwrite(path("info.nii.gz"), NIVolume(Float32.(Y); geometry...))
        r = run("info", path("info.nii.gz"))
        @test r.code == 0 && isempty(r.err)
        @test occursin("6×5×2×8", r.out) && occursin("2×2×3 mm", r.out) && occursin("R, A, S", r.out)
        @test occursin("sidecar info.json (5 entries)", r.out) && occursin("Siemens", r.out)
        @test occursin(r"TE \(from EchoTime\)\s+\[0.01, 0.03, 0.05, 0.07, 0.09, …\]  \(8 values\)", r.out)
        @test occursin(r"TR \(from RepetitionTime\)\s+\[2.0, 2.0", r.out)
        @test occursin("{…}  (1 entries: --find to list them)", r.out)
        # --find: only the matching entries, nested too
        r = run("info", path("info.nii.gz"), "--find", "echo")
        @test r.code == 0 && occursin("Coil.EchoSpacing", r.out) && occursin("EchoTime", r.out)
        @test !occursin("Manufacturer", r.out) && !occursin("voxel size", r.out)
        @test occursin("no entry of info.json matches `flip`", run("info", path("info.nii.gz"), "--find", "flip").out)
        # no sidecar; a .bval beside the data; several files
        r = run("info", path("dwi.nii"), path("mask.nii.gz"))
        @test r.code == 0 && occursin("beside it: dwi.bval", r.out) && occursin("b (from dwi.bval)", r.out)
        @test occursin("no sidecar (mask.json)", r.out)
        # a sidecar that does not fit the data: said, not an error
        writejson(path("bad.json"), Dict("EchoTime" => [0.01, 0.02]))
        niwrite(path("bad.nii"), NIVolume(Float32.(Y); geometry...))
        @test occursin("EQUIP cannot use these values", run("info", path("bad.nii")).out)
        # MATLAB files: the variables, their sizes and types
        r = run("info", path("t2.mat"))
        @test r.code == 0 && occursin(r"S\s+60×8 Float64", r.out) && occursin("TE", r.out)
        @test occursin(r"TE\s+1×8 Float64", run("info", path("t2.mat:TE")).out)
        # mistakes
        @test occursin("give a file", bad("info"))
        @test occursin("no such file", bad("info", path("none.nii.gz")))
        @test occursin("convert it with dcm2niix", bad("info", path("t2.json")))
        @test occursin("`--TE` is not a flag of `equip info`", bad("info", path("info.nii.gz"), "--TE", "1"))
        @test occursin("dcm2niix", run("info", "--help").out)
    end

    @testset "equip crlb, montecarlo, design: the Julia API's results" begin
        m4, m5 = QMTSIR(; SIR_4POINT...), QMTSIR(; SIR_5POINT...)
        tsv = path("t.tsv")
        sir = ("QMTSIR", "--protocol", "SIR_4POINT", "--snr", "200", "--kmf", "12.5")
        # crlb, one tissue: the table's numbers are crlb's, in full precision
        r = run("crlb", sir..., "--tissue", "SIR_WM_I", "--out", tsv)
        c = crlb(m4, nominal(m4, :SIR_WM_I); snr = 200, kmf = 12.5)
        @test r.code == 0 && occursin("crlb for QMTSIR", r.out) && occursin("wrote", r.out)
        t = table(tsv)
        @test length(t) == 4 && [x["parameter"] for x in t] == ["PSR", "R1f", "Sf", "M0f"]
        @test parse(Float64, t[1]["sd"]) == c.sd.PSR && parse(Float64, t[1]["bias"]) == c.bias.PSR
        # several tissues and an SNR sweep: one row per setting and parameter, crlb's numbers
        r = run("crlb", "QMTSIR", "--protocol", "SIR_4POINT", "--tissue", "SIR_WM_I,SIR_GM,SIR_lesion",
                "--snr", "100,200", "--kmf", "12.5", "--out", tsv)
        t = table(tsv)
        @test r.code == 0 && length(t) == 3 * 2 * 4
        cs = crlb(m4, scenarios(QMTSIR, [:SIR_WM_I, :SIR_GM, :SIR_lesion]; kmf = 12.5); snr = [100, 200])
        @test [parse(Float64, x["sd"]) for x in t] == [x.sd for x in cs.rows]
        @test t[1]["snr"] == "100" && t[1]["tissue"] == "SIR_WM_I"
        # a true value changed in every tissue: a column of its own
        r = run("crlb", sir..., "--tissue", "SIR_WM_I,SIR_lesion", "--truth", "PSR=0.15", "--out", tsv)
        t = table(tsv)
        @test r.code == 0 && all(x -> x["PSR"] == "0.15", t) && t[1]["truth"] == "0.15"
        # T2: --truth, --sigma, --vary; the 5-point scheme with kmf estimated, Eq. 3
        @test run("crlb", "T2Decay", "--TE", "0.01:0.01:0.32", "--truth", "S0=1000,T2=0.08", "--sigma", "10",
                  "--vary", "T2=0.04:0.04:0.12", "--out", tsv).code == 0
        @test length(table(tsv)) == 3 * 2 && table(tsv)[2]["sd"] ==
              repr(crlb(T2Decay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, T2 = 0.04); σ = 10.0).sd.T2)
        r = run("crlb", "QMTSIR", "--protocol", "SIR_5POINT", "--tissue", "SIR_WM_I", "--snr", "200",
                "--kmf", "free", "--cov", "biased", "--out", tsv)
        @test r.code == 0 && parse(Float64, table(tsv)[5]["sd"]) ==
              crlb(m5, nominal(m5, :SIR_WM_I); snr = 200, kmf = Free(), cov = :biased).sd.kmf
        # montecarlo: the same seed gives the same numbers as the Julia API
        r = run("montecarlo", sir..., "--tissue", "SIR_WM_I", "--noise", "rician", "--n", "500", "--seed", "7", "--out", tsv)
        mc = montecarlo(m4, nominal(m4, :SIR_WM_I); snr = 200, kmf = 12.5, noise = :rician, n = 500, rng = Xoshiro(7))
        @test r.code == 0 && occursin("montecarlo for QMTSIR", r.out)
        @test parse(Float64, table(tsv)[1]["sd"]) == mc.sd.PSR && parse(Float64, table(tsv)[1]["crlb_sd"]) == mc.crlb.sd.PSR
        @test run("montecarlo", sir..., "--tissue", "SIR_WM_I,SIR_lesion", "--noise", "rician", "--n", "100",
                  "--seed", "1", "--starts", "model", "--out", tsv).code == 0
        @test length(table(tsv)) == 2 * 4
        # uncertain inputs: a B1 map 5% high whose values have an SD of 0.03, crlb's and
        # montecarlo's numbers (the same seed); the display names it
        mfa = ("T1MFA", "--FA", "2,5,10,15,20,30", "--TR", "0.015", "--truth", "M0=1000,T1=1", "--B1", "1.05",
               "--snr", "1000", "--uncertain", "B1=0.03")
        mm, θm = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015), (M0 = 1000.0, T1 = 1.0, B1 = 1.0)
        r = run("crlb", mfa..., "--out", tsv)
        cu = crlb(mm, θm; snr = 1000, B1 = 1.05, uncertain = (B1 = 0.03,))
        @test r.code == 0 && occursin("B1 ± 0.03", r.out) && parse(Float64, table(tsv)[2]["sd"]) == cu.sd.T1
        r = run("montecarlo", mfa..., "--noise", "gaussian", "--n", "200", "--seed", "4", "--out", tsv)
        mu = montecarlo(mm, θm; snr = 1000, B1 = 1.05, uncertain = (B1 = 0.03,), noise = :gaussian, n = 200, rng = Xoshiro(4))
        @test r.code == 0 && parse(Float64, table(tsv)[2]["sd"]) == mu.sd.T1
        @test occursin("`--uncertain B1`: `abc` is not a number", bad("crlb", mfa[1:end-1]..., "B1=abc"))
        @test occursin("`--uncertain`: `T2` is not a parameter", bad("crlb", mfa[1:end-1]..., "T2=0.03"))
        # design: the search with its seed, as the Julia API's; the protocol written as flags
        cnf = path("design.cnf")
        r = run("design", sir..., "--tissue", "SIR_WM_I,SIR_lesion", "--acquisition", "TI=0.01..10,TD=0.01..10",
                "--readout", "0.3", "--starts", "1", "--generations", "20", "--seed", "3", "--out", cnf)
        @test r.code == 0 && occursin("DesignResult for QMTSIR", r.out) && occursin("below the start's", r.out)
        d = ProtocolDesign(m4, scenarios(QMTSIR, [:SIR_WM_I, :SIR_lesion]; kmf = 12.5); snr = 200,
                           acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)), scantime = m -> sum(m.TI .+ m.TD .+ 0.3))
        dr = design(d; starts = 1, generations = 20, rng = Xoshiro(3))
        words = EQUIPCLI.configwords(cnf)
        @test words[findfirst(==("--TI"), words) + 1] == join(repr.(Float64.(dr.rounded.TI)), ',')
        @test words[findfirst(==("--TD"), words) + 1] == join(repr.(Float64.(dr.rounded.TD)), ',')
        # the design's config file fits data acquired with it (the data given on the command line);
        # data from another protocol are refused, with the counts
        md = QMTSIR(; TI = dr.rounded.TI, TD = dr.rounded.TD)
        Yd = simulate(md, map(x -> x isa Real ? fill(Float64(x), 3, 2, 1) : x, nominal(md, :SIR_WM_I));
                      snr = 200, noise = :rician, rng = MersenneTwister(6))
        niwrite(path("sir4.nii.gz"), NIVolume(Float32.(Yd); geometry...))
        @test run("fit", "--config", cnf, path("sir4.nii.gz"), "--out", path("fromdesign_")).code == 0
        @test rawmap(path("fromdesign_PSR.nii.gz")) ≈ Float32.(fit(md, readdata(path("sir4.nii.gz"))[1]).params.PSR)
        @test occursin("has 4 observations", bad("fit", "--config", cnf, path("sir.nii.gz")))
        # --scantime and --constraint as expressions of the model's fields
        @test run("design", sir..., "--tissue", "SIR_WM_I", "--acquisition", "TI=0.01..10,TD=0.01..10",
                  "--scantime", "sum(TI .+ TD .+ 0.3)", "--constraint", "TI .+ TD .- 0.5",
                  "--starts", "1", "--generations", "5", "--seed", "1").code == 0
        # mistakes
        @test occursin("give the noise level", bad("crlb", "QMTSIR", "--protocol", "SIR_4POINT", "--tissue", "SIR_WM_I"))
        @test occursin("--noise rician", bad("montecarlo", sir..., "--tissue", "SIR_WM_I"))
        @test occursin("give what to optimize", bad("design", sir..., "--tissue", "SIR_WM_I"))
        @test occursin("NAME=lower..upper", bad("design", sir..., "--tissue", "SIR_WM_I", "--acquisition", "TI=0.01:10"))
        @test occursin("give the scan time", bad("design", sir..., "--tissue", "SIR_WM_I", "--acquisition", "TI=0.01..10"))
        @test occursin("not a Julia expression", bad("design", sir..., "--tissue", "SIR_WM_I",
                                                    "--acquisition", "TI=0.01..10", "--scantime", "sum(TI"))
        @test occursin("NAME=values", bad("crlb", sir..., "--tissue", "SIR_WM_I", "--vary", "kmf"))
        @test occursin("`PSX` is not a parameter", bad("crlb", sir..., "--truth", "PSX=0.1"))
        @test occursin("takes no files", bad("crlb", sir..., "data.nii.gz"))
        @test occursin("T1IR: no value for `T1`", bad("crlb", "T1IR", "--TI", "0.1,0.5,1.0", "--snr", "100"))
        @test occursin("--truth NAME=value", bad("crlb", "T1IR", "--TI", "0.1,0.5,1.0", "--snr", "100"))
    end
end
