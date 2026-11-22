# =================================================================================================
# test/integration/io.jl: Data files: readdata, readmap, writemaps, writetable (NIfTI with BIDS sidecars, MATLAB files).
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl io
# =================================================================================================

@testset "io: readdata, readmap, writemaps, writetable (NIfTI and MATLAB files)" begin
    dir = mktempdir()
    path(name) = joinpath(dir, name)
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    writejson(p, d) = open(io -> JSON.json(io, d), p, "w")
    # A small T2 dataset: 6×5×2 voxels, 8 echoes, written as a scanner pipeline would, with a
    # geometry that is not the default (2 × 2 × 3 mm voxels, an offset)
    te = collect(0.01:0.02:0.15)
    T2map = collect(reshape(range(0.04, 0.12; length = 60), 6, 5, 2))
    m = T2Decay(TE = te)
    Y = simulate(m, (S0 = fill(1000.0, 6, 5, 2), T2 = T2map); snr = 200, noise = :gaussian,
                 rng = MersenneTwister(1))
    geometry = (voxel_size = (2.0f0, 2.0f0, 3.0f0),
                orientation = Float32[2 0 0 -10; 0 2 0 -20; 0 0 3 5])
    niwrite(p, a) = NIfTI.niwrite(p, NIfTI.NIVolume(a; geometry...))

    # One 4-D file, its sidecar giving one EchoTime per volume
    niwrite(path("t2.nii.gz"), Float32.(Y))
    writejson(path("t2.json"), Dict("EchoTime" => te, "RepetitionTime" => 2.0))
    data, info = readdata(path("t2.nii.gz"))
    @test size(data) == (6, 5, 2, 8) && eltype(data) == Float64 && data == Float32.(Y)
    @test info.spatial == (6, 5, 2) && info.shape == [6, 5, 2]
    @test info.found[:TE] ≈ te && info.found[:TR] == fill(2.0, 8) && !haskey(info.found, :TI)
    @test info.header !== nothing && info.files == [path("t2.nii.gz")]
    # One 3-D file per echo (as dcm2niix writes a multi-echo series), each with its
    # sidecar, stacked in the order given; the intensity scaling applied
    files = [path("echo$k.nii") for k in 1:8]
    for k in 1:8
        raw = round.(Int16, Y[:, :, :, k] ./ 0.5)
        NIfTI.niwrite(files[k], NIfTI.NIVolume(raw; geometry..., scl_slope = 0.5f0))
        writejson(path("echo$k.json"), Dict("EchoTime" => te[k]))
    end
    d2, i2 = readdata(files)
    @test size(d2) == (6, 5, 2, 8) && maximum(abs.(d2 .- Y)) <= 0.25 && i2.found[:TE] ≈ te
    # Float32: half the memory, the same values rounded (scaled in Float64 first)
    d32, _ = readdata(files; precision = Float32)
    @test eltype(d32) == Float32 && d32 == Float32.(d2)
    # Diffusion: an FSL .bval file beside the data
    niwrite(path("dwi.nii"), Float32.(Y[:, :, :, 1:4]))
    write(path("dwi.bval"), "0 500 1000 2000\n")
    @test readdata(path("dwi.nii"))[2].found[:b] == [0.0, 500, 1000, 2000]
    # and an FSL .bvec beside both: gx, gy, gz (0 0 0 at b = 0), checked by diffusionencoding
    write(path("dwi.bvec"), "0 1 0 0.6\n0 0 1 0.8\n0 0 0 0\n")
    fd = readdata(path("dwi.nii"))[2].found
    @test fd[:gx] == [0, 1, 0, 0.6] && fd[:gy] == [0, 0, 1, 0.8] && fd[:gz] == zeros(4)
    @test (; (k => fd[k] for k in (:b, :gx, :gy, :gz))...) ==
          diffusionencoding(path("dwi.bval"), path("dwi.bvec"))
    write(path("dwi.bvec"), "0 1 0 0.6\n0 0 1 0.8\n0 0 0 0.5\n")          # image 4 not unit
    @test occursin("dwi.bvec (with", msg(() -> readdata(path("dwi.nii")))) &&
          occursin("unit vectors", msg(() -> readdata(path("dwi.nii"))))
    write(path("dwi.bvec"), "0 1 0\n0 0 1\n0 0 0\n")                      # three images, not four
    @test occursin("one direction per image", msg(() -> readdata(path("dwi.nii"))))
    rm(path("dwi.bvec"))
    # Mistakes
    NIfTI.niwrite(path("other.nii"), NIfTI.NIVolume(Float32.(Y[:, :, :, 1])))
    niwrite(path("small.nii"), zeros(Float32, 4, 4, 2))
    @test occursin("different geometry", msg(() -> readdata([files[1], path("other.nii")])))
    @test occursin("same size", msg(() -> readdata([files[1], path("small.nii")])))
    @test occursin("one image is one observation", msg(() -> readdata(files[1])))
    @test occursin("give NIfTI files", msg(() -> readdata(path("t2.json"))))
    @test occursin("no such file", msg(() -> readdata(path("missing.nii.gz"))))
    writejson(path("echo1.json"), Dict("EchoTime" => [0.01, 0.02]))
    @test occursin("EchoTime has 2 values for 1 images", msg(() -> readdata(files)))

    # readmap: a mask in the data's geometry; mistakes named
    mask = T2map .> 0.06
    niwrite(path("mask.nii.gz"), UInt8.(mask))
    @test readmap(path("mask.nii.gz"), info) == mask
    @test occursin("different geometry", msg(() -> readmap(path("other.nii"), info)))
    @test occursin("the data are 6×5×2", msg(() -> readmap(path("small.nii"), info)))
    @test readvalues === EQUIP.MRImages.readvalues        # re-exported

    # writemaps: the maps in the data's geometry, with sidecars recording how they were made
    res = fit(T2Decay(TE = info.found[:TE]), data; mask, T2 = Free(start = 0.07))
    written = writemaps(path("out/sub-01_"), res, info; stderror = true,
                        sidecar = Dict("Data" => "t2.nii.gz"))
    @test basename.(written) == ["sub-01_S0map.nii.gz", "sub-01_T2map.nii.gz",
                                 "sub-01_status.nii.gz", "sub-01_SSE.nii.gz",
                                 "sub-01_S0map_SE.nii.gz", "sub-01_T2map_SE.nii.gz"]
    t2 = readimage(written[2])
    @test size(t2) == (6, 5, 2) && eltype(NIfTI.niread(written[2]).raw) == Float32
    @test isequal(t2.data, Float64.(Float32.(res.params.T2)))
    @test all(isnan, t2.data[.!mask]) && maximum(abs.(t2.data[mask] .- T2map[mask])) < 0.01
    @test samegeometry(t2, readimage(path("t2.nii.gz")))
    @test t2.header.scl_slope == 1 && t2.header.scl_inter == 0
    # compress = false: .nii, the same maps (gzip takes about 1 s per 21M-voxel map, docs/design.md)
    plain = writemaps(path("plain/sub-01_"), res, info; compress = false)
    @test basename(plain[2]) == "sub-01_T2map.nii" && isequal(readimage(plain[2]).data, t2.data)
    @test readimage(written[3]).data == Int.(res.status)
    @test t2.meta["Map"] == "T2map" && t2.meta["Model"] == "T2Decay" &&
          t2.meta["Acquisition"]["TE"] ≈ te && t2.meta["Data"] == "t2.nii.gz"
    @test t2.meta["Fit"]["T2"] == "Free(start = 0.07)" && t2.meta["Fit"]["precision"] == "Float64"
    # every parameter as the fit used it: the call's start for T2 with the model's bounds, and
    # S0 as the model declares it (`parameters(T2Decay)`), given nothing
    pr = t2.meta["Parameters"]
    d = parameters(T2Decay)
    @test pr["T2"] == Dict("Role" => "free", "Start" => 0.07, "Lower" => d.T2.lower, "Upper" => d.T2.upper)
    @test pr["S0"]["Role"] == "free" && pr["S0"]["Start"] == String(d.S0.start)
    @test t2.meta["ParameterNames"] == ["S0", "T2"]                 # the model's order
    # ... and the defaults the call did not name, for QMTSIR: kmf fixed (here given), R1m tied
    # to R1f, Sm at its default; an unbounded upper bound as null
    q = EQUIP._parameterroles(fit(QMTSIR(; SIR_4POINT...), ones(4); kmf = 12.5))
    dq = parameters(QMTSIR)
    @test q["kmf"] == Dict("Role" => "fixed", "Value" => 12.5) && q["R1m"] == Dict("Role" => "tied", "To" => "R1f")
    @test q["Sm"] == Dict("Role" => "fixed", "Value" => dq.Sm.value) && q["M0f"]["Upper"] === nothing
    @test startswith(t2.meta["Software"], "EQUIP ")
    @test occursin("one voxel's", msg(() -> writemaps(path("x_"), fit(m, Y[1, 1, 1, :]), info)))
    @test occursin("the data were 6×5×2", msg(() -> writemaps(path("x_"), fit(m, Y[1:2, :, :, :]), info)))
    # a model's derived quantities, written beside its parameters (DTI: MD, FA, AD, RD)
    gd = directionset(12)
    bvd = [0.0; fill(1000.0, 12)]
    Gd = reduce(hcat, [[[0.0, 0.0, 0.0]]; collect.(gd)])
    mdt = DTI(; diffusionencoding(bvd, Gd)...)
    Yt = simulate(mdt, (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0);
                  snr = 50, n = 12, noise = :rician, rng = MersenneTwister(5))
    niwrite(path("dti.nii.gz"), Float32.(reshape(Yt, 3, 2, 2, 13)))
    write(path("dti.bval"), join(bvd, " ") * "\n")
    write(path("dti.bvec"), join((join(Gd[k, :], " ") for k in 1:3), "\n") * "\n")
    dtd, dti_info = readdata(path("dti.nii.gz"))
    rdt = fit(DTI(; (k => dti_info.found[k] for k in (:b, :gx, :gy, :gz))...), dtd)
    wd = writemaps(path("dti/sub-01_"), rdt, dti_info)
    @test basename.(wd) == ["sub-01_S0map.nii.gz", "sub-01_Dxx.nii.gz", "sub-01_Dyy.nii.gz",
                            "sub-01_Dzz.nii.gz", "sub-01_Dxy.nii.gz", "sub-01_Dxz.nii.gz", "sub-01_Dyz.nii.gz",
                            "sub-01_MD.nii.gz", "sub-01_FA.nii.gz", "sub-01_AD.nii.gz", "sub-01_RD.nii.gz",
                            "sub-01_V1x.nii.gz", "sub-01_V1y.nii.gz", "sub-01_V1z.nii.gz",
                            "sub-01_status.nii.gz", "sub-01_SSE.nii.gz"]
    fa = readimage(wd[9])
    @test isequal(fa.data, Float64.(Float32.(derived(rdt).FA))) && fa.meta["Map"] == "FA"

    # MATLAB files: the data as an array, observations last; maps back in its own shape
    S = reshape(Y, 60, 8)                                  # voxels already masked: nvoxels×N
    writemat(path("data.mat"), Dict("S" => S, "TE" => collect(te'), "mask" => Float64.(vec(mask)),
                                    "TEcolumn" => reshape(te, 8, 1), "five" => rand(2, 2, 2, 2, 2)))
    dm, im = readdata(path("data.mat:S"))
    @test size(dm) == (60, 1, 1, 8) && vec(dm[:, 1, 1, 3]) == S[:, 3]
    @test im.header === nothing && im.spatial == (60, 1, 1) && im.shape == [60]
    @test isempty(im.found) && readvalues(path("data.mat:TE")) == te
    mm = readmap(path("data.mat:mask"), im) .> 0
    @test size(mm) == (60, 1, 1) && vec(mm) == vec(mask)
    rm_ = fit(m, dm; mask = mm)
    @test isequal(vec(rm_.params.T2), vec(fit(m, S; mask = vec(mask)).params.T2))
    out = writemaps(path("mat/sub-01_"), rm_, im)               # :mat, the default for .mat data
    @test out == [path("mat/sub-01_maps.mat")] && isfile(path("mat/sub-01_maps.json"))
    back = readmat(out[1])
    @test Set(keys(back)) == Set(["S0map", "T2map", "status", "SSE"])
    @test size(back["T2map"]) == (60,) && isequal(back["T2map"], Float32.(vec(rm_.params.T2)))
    @test JSON.parsefile(path("mat/sub-01_maps.json"))["Maps"] == ["S0map", "T2map", "status", "SSE"]
    # 4-D, as NIfTI; maps to NIfTI in another image's geometry
    writemat(path("data4.mat"), Dict("S" => Y))
    d4, i4 = readdata(path("data4.mat"))                     # one variable: no name needed
    @test d4 == Y && i4.shape == [6, 5, 2]
    r4 = fit(m, d4)
    f4 = writemaps(path("nii/x_"), r4, i4; like = readimage(path("t2.nii.gz")))
    @test samegeometry(readimage(f4[2]), readimage(path("t2.nii.gz")))
    @test occursin("observations along the last", msg(() -> readdata(path("data.mat:five"))))
    @test occursin("has one observation", msg(() -> readdata(path("data.mat:TEcolumn"))))
    @test occursin("give one", msg(() -> readdata([path("data.mat:S"), path("data.mat:S")])))
    @test occursin("the data's images are 60", msg(() -> readmap(path("data4.mat:S"), im)))

    # writetable: rows as TSV
    rows = [(name = "WM", PSR = 0.1, n = 3), (name = "GM", PSR = 0.05, n = 4)]
    f = writetable(path("tables/t.tsv"), rows)
    @test readlines(f) == ["name\tPSR\tn", "WM\t0.1\t3", "GM\t0.05\t4"]
    @test_throws ArgumentError writetable(path("e.tsv"), [])
end
