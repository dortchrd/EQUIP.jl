using VoxelFit
using VoxelFit: setup, freenames, read_npy, write_npy, create_npy, FitStatus
using StaticArrays
using LinearAlgebra
using Random
using Statistics
using Test
using NIfTI

# Model, protocol, true parameters (every parameter, fixed ones included)
const CASES = [
    (MonoExp(), (TE = collect(10.0:10:320),), (S0 = 1000.0, T2 = 80.0)),
    (MonoExpOffset(), (TE = collect(10.0:10:320),), (S0 = 1000.0, T2 = 80.0, C = 50.0)),
    (BiExp(), (TE = collect(5.0:5:320),), (A1 = 300.0, T2a = 15.0, A2 = 700.0, T2b = 90.0)),
    (ADC(), (b = [0, 100, 300, 500, 800, 1000.0],), (S0 = 1000.0, D = 1.5e-3)),
    (Kurtosis(), (b = collect(0:250:2500.0),), (S0 = 1000.0, D = 1.5e-3, K = 0.9)),
    (IVIM(), (b = [0, 10, 20, 40, 80, 110, 140, 170, 200, 300, 400, 500, 600, 700, 800, 1000.0],),
     (S0 = 1000.0, f = 0.1, Dp = 0.02, D = 1.2e-3)),
    (InversionRecovery(), (TI = [50, 100, 200, 400, 800, 1600, 3200.0],), (S0 = 1000.0, B = 1.9, T1 = 900.0)),
    (SaturationRecovery(), (t = [100, 300, 600, 1000, 2000, 4000.0],), (S0 = 1000.0, T1 = 1200.0)),
    (SPGR(), (FA = [2, 5, 10, 15, 20, 30.0], TR = 15.0), (M0 = 5000.0, T1 = 1000.0, B1 = 1.0)),
    (SIRFSE(), (TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010]),
     (pmf = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, R1m = 1.0, Sm = 0.83)),
]

estimated(m, θ) = NamedTuple{freenames(setup(m))}(θ)

function synth_monoexp(sz; σ=5.0, seed=1)
    rng = MersenneTwister(seed)
    te = collect(10.0:10.0:320.0)
    T2 = 40 .+ 80 .* rand(rng, sz...)
    S0 = 800 .+ 400 .* rand(rng, sz...)
    data = S0 .* exp.(-reshape(te, ntuple(_ -> 1, length(sz))..., :) ./ T2) .+
           σ .* randn(rng, sz..., length(te))
    return te, data, S0, T2
end

# SIR-FSE signal from Eq. 1 of Dortch et al., MRM 2018 (matrix exponential), free-pool component.
function sir_eq1(TI, TD; pmf, R1f, Sf, M0f, kmf, R1m, Sm, magnitude=true)
    kfm = kmf * pmf
    A = [-(R1f + kfm) kmf; kfm -(R1m + kmf)]
    M0 = [M0f, pmf * M0f]
    S = Diagonal([Sf, Sm])
    map(TI, TD) do ti, td
        Mz = (exp(A * ti) * S * (I - exp(A * td)) + (I - exp(A * ti))) * M0
        magnitude ? abs(Mz[1]) : Mz[1]
    end
end

@testset "VoxelFit" begin
    @testset "noise-free recovery: $(typeof(m))" for (m, acq, θ) in CASES
        y = signal(m, acq, θ)
        for T in (Float64, Float32)
            r = fit!(init(m, acq; precision=T), y)
            @test isconverged(r)
            for (k, v) in pairs(r.params)
                @test v ≈ θ[k] rtol = (T == Float64 ? 1e-6 : 2e-3)
            end
        end
    end

    @testset "fit! does not allocate" begin
        for (m, acq, θ) in CASES, T in (Float64, Float32)
            y = signal(m, acq, θ) .* (1 .+ 0.01 .* randn(MersenneTwister(0), length(acq[1])))
            c = init(m, acq; precision=T)
            fit!(c, y)
            @test (@allocated fit!(c, y)) == 0
        end
    end

    @testset "SIRFSE matches Eq. 1 of the 2018 paper" begin
        TI = [0.010, 0.015, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0, 2.0, 5.0]
        for TD in (0.01, 1.0, 2.5, 5.0), mag in (true, false)
            θ = (pmf = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 14.5, R1m = 1.2, Sm = 0.83)
            @test signal(SIRFSE(magnitude=mag), (TI = TI, TD = TD), θ) ≈ sir_eq1(TI, fill(TD, 10); θ..., magnitude=mag)
        end
    end

    @testset "parameter roles" begin
        s = setup(SIRFSE())
        @test freenames(s) == (:pmf, :R1f, :Sf, :M0f)
        @test keys(s.fixed) == (:kmf, :R1m, :Sm) && s.fixed.kmf == 12.5
        @test freenames(setup(SIRFSE(); fixed = (kmf = 10.0,))) == (:pmf, :R1f, :Sf, :M0f)
        @test setup(SIRFSE(); fixed = (kmf = 10.0,)).fixed.kmf == 10.0
        s = setup(SIRFSE(); free = (:kmf,))
        @test freenames(s) == (:pmf, :R1f, :Sf, :M0f, :kmf) && s.start.kmf == 12.5
        s = setup(SIRFSE(); free = (:kmf => Estimate(20.0; lower=5, upper=50),))
        @test s.lower[end] == 5 && s.upper[end] == 50 && s.start.kmf == 20
        @test freenames(setup(SIRFSE(); fixed = (pmf = 0.1,))) == (:R1f, :Sf, :M0f)
        @test_throws ArgumentError setup(SIRFSE(); fixed = (kmff = 1.0,))
        @test_throws ArgumentError setup(SIRFSE(); free = (:nope,))
        @test_throws ArgumentError setup(SIRFSE(); fixed = (kmf = Estimate(1.0),))
        @test_throws ArgumentError setup(SIRFSE(); start = (kmf = 1.0,))    # kmf is not estimated
        @test_throws ArgumentError init(SPGR(), [2.0, 5.0])                # needs FA and TR
        @test_throws ArgumentError init(SIRFSE(), (TI = [0.1, 1.0],))      # missing TD
    end

    @testset "SIRFSE: fixed, free, tied, per-voxel" begin
        acq = (TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])
        truth = (pmf = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0, R1m = 1.0, Sm = 0.83)
        y = signal(SIRFSE(), acq, truth)
        # kmf estimated: exact recovery
        r = fit!(init(SIRFSE(), acq; free = (:kmf,)), y)
        @test r.params.kmf ≈ 15.0 rtol = 1e-5
        @test r.params.pmf ≈ 0.12 rtol = 1e-5
        # kmf assumed correctly: exact; assumed wrongly: biased pmf
        @test fit!(init(SIRFSE(), acq; fixed = (kmf = 15.0,)), y).params.pmf ≈ 0.12 rtol = 1e-6
        biased = fit!(init(SIRFSE(), acq; fixed = (kmf = 12.5,)), y).params.pmf
        @test abs(biased - 0.12) > 1e-4
        # tie R1m = R1f (default) vs untied constant
        y2 = signal(SIRFSE(), (TI = acq.TI, TD = acq.TD), merge(truth, (R1m = 2.0, kmf = 12.5)))
        @test fit!(init(SIRFSE(), acq; fixed = (R1m = 2.0,)), y2).params.R1f ≈ 1.0 rtol = 1e-6
        # per-voxel Sm map through fit_maps
        n = 50
        sm = 0.75 .+ 0.15 .* rand(MersenneTwister(4), n)
        Y = reduce(vcat, [signal(SIRFSE(), acq, merge(truth, (kmf = 12.5, Sm = sm[v])))' for v in 1:n])
        res = fit_maps(SIRFSE(), acq, Y; fixed = (Sm = sm,))
        @test res[:pmf] ≈ fill(0.12, n) rtol = 1e-5
        # a bad map value marks that voxel invalid
        sm[7] = NaN
        res = fit_maps(SIRFSE(), acq, Y; fixed = (Sm = sm,))
        @test res.status[7] == UInt8(VoxelFit.InvalidData) && all(isnan, res.params[7, :])
        @test all(isconverged, res.status[setdiff(1:n, 7)])
    end

    @testset "fit_maps: layouts, threads, mask, invalid data" begin
        te, data, S0, T2 = synth_monoexp((24, 20, 5))
        mask = rand(MersenneTwister(2), 24, 20, 5) .> 0.3
        data[1, 1, 1, 7] = NaN
        mask[1, 1, 1] = true
        a = fit_maps(MonoExp(), te, data; mask)
        b = fit_maps(MonoExp(), te, permutedims(data, (4, 1, 2, 3)); obsdim=1, mask)
        c = fit_maps(MonoExp(), te, permutedims(data, (1, 4, 2, 3)); obsdim=2, mask)
        d = fit_maps(MonoExp(), te, data; mask, ntasks=1, chunksize=7)
        e = fit_maps(MonoExp(), (TE = te,), data; mask)
        @test isequal(a.params, b.params) && isequal(a.params, c.params) && isequal(a.params, d.params) &&
              isequal(a.params, e.params)
        @test a.status[1, 1, 1] == UInt8(VoxelFit.InvalidData)
        @test all(isnan, a[:T2][.!mask]) && all(==(0x00), a.status[.!mask])
        sel = mask .& (a.status .!= UInt8(VoxelFit.InvalidData))
        @test all(isconverged, a.status[sel])
        @test median(abs.(a[:T2][sel] .- T2[sel]) ./ T2[sel]) < 0.01
        di = round.(Int16, clamp.(replace(data, NaN => 0), 0, 30000))   # integer input (raw NIfTI)
        f = fit_maps(MonoExp(), te, di; mask, precision=Float32)
        @test eltype(f.params) == Float32
        @test median(abs.(f[:T2][sel] .- T2[sel]) ./ T2[sel]) < 0.01
    end

    @testset "start, bounds, weights, stderr" begin
        te, data, S0, T2 = synth_monoexp((30, 30))
        r = fit_maps(MonoExp(), te, data; stderr=true)
        r2 = fit_maps(MonoExp(), te, data; start=r)                       # warm start from a fit
        @test maximum(r2.iterations) <= 2
        @test r2.params ≈ r.params rtol = 1e-6
        r3 = fit_maps(MonoExp(), te, data; start=(S0 = 1000.0, T2 = 60.0))   # constant start
        @test r3.params ≈ r.params rtol = 1e-5
        r4 = fit_maps(MonoExp(), te, data; start=(T2 = r[:T2],))           # map for one parameter
        @test r4.params ≈ r.params rtol = 1e-6
        z = (r[:T2] .- T2) ./ selectdim(r.stderr, 3, 2)                     # calibrated errors
        @test 0.85 < std(vec(z)) < 1.15
        rb = fit_maps(MonoExp(), te, data; free = (:T2 => Estimate(:auto; lower=1e-6, upper=60.0),))
        @test all(rb[:T2] .<= 60.0) && all(isconverged, rb.status) && count(==(60.0), rb[:T2]) > 0
        bad = copy(data)
        bad[:, :, 3] .+= 300
        w = ones(length(te))
        w[3] = 0
        rw = fit_maps(MonoExp(), te, bad; weights=w)
        @test median(abs.(rw[:T2] .- T2) ./ T2) < 0.01
    end

    @testset "per-voxel B1 map (SPGR)" begin
        rng = MersenneTwister(3)
        acq = (FA = [2, 5, 10, 15, 20, 30.0], TR = 15.0)
        n = 400
        T1, M0, B1 = 600 .+ 1000 .* rand(rng, n), 4000 .+ 2000 .* rand(rng, n), 0.8 .+ 0.4 .* rand(rng, n)
        Y = reduce(vcat, [signal(SPGR(), acq, (M0 = M0[v], T1 = T1[v], B1 = B1[v]))' for v in 1:n])
        r = fit_maps(SPGR(), acq, Y; fixed = (B1 = B1,))
        @test r[:T1] ≈ T1 rtol = 1e-6
        @test median(abs.(fit_maps(SPGR(), acq, Y)[:T1] .- T1) ./ T1) > 0.05   # ignoring B1 is biased
    end

    @testset "custom models" begin
        te = collect(10.0:10:320)
        θ = (S0 = 1000.0, T2 = 70.0, C = 30.0)
        y = signal(MonoExpOffset(), te, θ)
        fm = FunctionModel((out, acq, p) -> (out .= p.S0 .* exp.(-acq.TE ./ p.T2) .+ p.C),
                           (S0 = Estimate(900.0; lower=0), T2 = Estimate(50.0; lower=0), C = Estimate(0.0));
                           protocol = (:TE,))
        @test collect(fit!(init(fm, te), y).params) ≈ collect(θ) rtol = 1e-6
        em = expr_model("S0 * exp(-TE / T2) + C"; params = (:S0, :T2, :C), protocol = (:TE,), p0 = (900, 50, 0))
        @test collect(Base.invokelatest(() -> fit!(init(em, te), y)).params) ≈ collect(θ) rtol = 1e-6
        # two settings per observation, and a fixed parameter in an expression model
        acq = (TE = repeat([10.0, 30.0, 60.0, 90.0], inner=4), TR = repeat([300.0, 800.0, 2000.0, 5000.0], 4))
        em2 = expr_model("S0 * exp(-TE / T2) * (1 - exp(-TR / T1))"; protocol = (:TE, :TR),
                         params = (S0 = Estimate(900.0), T2 = Estimate(50.0), T1 = Fixed(1200.0)))
        y2 = Base.invokelatest(signal, em2, acq, (S0 = 1000.0, T2 = 70.0, T1 = 1200.0))
        r = Base.invokelatest(() -> fit!(init(em2, acq), y2))
        @test r.params.S0 ≈ 1000 rtol = 1e-6
        @test r.params.T2 ≈ 70 rtol = 1e-6
        @test parse_model("SIRFSE(magnitude=false)") == SIRFSE(magnitude=false)
        @test parse_model("IVIM(bthresh=250)").bthresh == 250
        @test_throws ArgumentError parse_model("Foo")
        @test_throws ArgumentError parse_model("SIRFSE(magnitude=rm(\"x\"))")
    end

    @testset "npy round trip" begin
        mktempdir() do dir
            A = rand(Float32, 3, 4, 5)
            write_npy(joinpath(dir, "f.npy"), A)
            B, rev = read_npy(joinpath(dir, "f.npy"))
            @test B == A && !rev
            write_npy(joinpath(dir, "c.npy"), A; reversed=true)
            C, rev = read_npy(joinpath(dir, "c.npy"))
            @test C == A && rev
            M = create_npy(joinpath(dir, "m.npy"), Int32, (6, 7))
            M .= reshape(1:42, 6, 7)
            VoxelFit.Mmap.sync!(M)
            @test first(read_npy(joinpath(dir, "m.npy"); mmap=false)) == reshape(1:42, 6, 7)
            raw = UInt8[0x93; collect(b"NUMPY"); 0x01; 0x00; 0x76; 0x00;
                        collect(rpad("{'descr': '|u1', 'fortran_order': False, 'shape': (2, 3), }", 117) * "\n");
                        0x01:0x06]
            write(joinpath(dir, "np.npy"), raw)
            D, rev = read_npy(joinpath(dir, "np.npy"))
            @test rev && permutedims(D) == UInt8[1 2 3; 4 5 6]
        end
    end

    @testset "command line" begin
        quiet(f) = redirect_stdout(f, devnull)
        mktempdir() do dir
            te, data, _, T2 = synth_monoexp((12, 10, 3))
            data32 = Float32.(data)
            niwrite(joinpath(dir, "d.nii.gz"), NIVolume(data32; voxel_size=(2.0, 2.0, 3.0)))
            niwrite(joinpath(dir, "mask.nii"), NIVolume(UInt8.(T2 .> 60)))
            open(joinpath(dir, "te.txt"), "w") do io
                foreach(t -> println(io, t), te)
            end
            @test quiet(() -> VoxelFit.main(["fit", "--model", "MonoExp", "--x-file", joinpath(dir, "te.txt"),
                                             "--data", joinpath(dir, "d.nii.gz"), "--mask", joinpath(dir, "mask.nii"),
                                             "--out", joinpath(dir, "out", "s"), "--stderr"])) == 0
            t2 = niread(joinpath(dir, "out", "s_T2.nii.gz"))
            @test size(t2) == size(T2) && t2.header.pixdim[2:4] == (2.0f0, 2.0f0, 3.0f0)
            sel = T2 .> 60
            @test median(abs.(t2.raw[sel] .- T2[sel]) ./ T2[sel]) < 0.01
            @test all(isnan, t2.raw[.!sel])
            @test isfile(joinpath(dir, "out", "s_stderr_S0.nii.gz"))
            # NumPy C order in, .npy out
            write_npy(joinpath(dir, "d.npy"), permutedims(data32, (4, 3, 2, 1)); reversed=true)
            @test quiet(() -> VoxelFit.main(["fit", "--model", "MonoExp", "--x", join(te, ","), "--data",
                                             joinpath(dir, "d.npy"), "--out", joinpath(dir, "np"),
                                             "--precision", "double"])) == 0
            t2n, rev = read_npy(joinpath(dir, "np_T2.npy"))
            @test rev && eltype(t2n) == Float64
            @test permutedims(t2n, (3, 2, 1)) ≈ fit_maps(MonoExp(), te, data32)[:T2] rtol = 1e-5
            @test readlines(joinpath(dir, "np_params.txt")) == ["S0", "T2"]
            # expression model
            @test quiet(() -> VoxelFit.main(["fit", "--expr", "S0*exp(-TE/T2)", "--params", "S0,T2",
                                             "--protocol", "TE", "--p0", "1000,50", "--x", join(te, ","),
                                             "--data", joinpath(dir, "d.npy"), "--out", joinpath(dir, "ex"),
                                             "--quiet"])) == 0
            @test read_npy(joinpath(dir, "ex_T2.npy"))[1] ≈ t2n rtol = 1e-4
            # SIRFSE: protocol with --acq, kmf freed, Sm from a map file, bounds
            acq = (TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])
            truth = (pmf = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0, R1m = 1.0, Sm = 0.83)
            ysir = signal(SIRFSE(), acq, truth)
            Ysir = Float64[ysir[k] for _ in 1:4, _ in 1:3, k in 1:5]
            write_npy(joinpath(dir, "sir.npy"), Ysir)                                    # Fortran order
            write_npy(joinpath(dir, "sm.npy"), fill(0.83, 4, 3))
            @test quiet(() -> VoxelFit.main(["fit", "--model", "SIRFSE", "--data", joinpath(dir, "sir.npy"),
                                             "--acq", "TI=" * join(acq.TI, ","), "--acq", "TD=" * join(acq.TD, ","),
                                             "--free", "kmf", "--fixed", "Sm=" * joinpath(dir, "sm.npy"),
                                             "--bounds", "kmf=1,50", "--precision", "double",
                                             "--out", joinpath(dir, "sir")])) == 0
            @test readlines(joinpath(dir, "sir_params.txt")) == ["pmf", "R1f", "Sf", "M0f", "kmf"]
            @test first(read_npy(joinpath(dir, "sir_kmf.npy"))) ≈ fill(15.0, 4, 3) rtol = 1e-5
            @test first(read_npy(joinpath(dir, "sir_pmf.npy"))) ≈ fill(0.12, 4, 3) rtol = 1e-5
            # --acq-file with a header
            open(joinpath(dir, "acq.txt"), "w") do io
                println(io, "TI TD")
                foreach(k -> println(io, acq.TI[k], " ", acq.TD[k]), 1:5)
            end
            @test quiet(() -> VoxelFit.main(["fit", "--model", "SIRFSE", "--data", joinpath(dir, "sir.npy"),
                                             "--acq-file", joinpath(dir, "acq.txt"), "--fixed", "kmf=15",
                                             "--precision", "double", "--out", joinpath(dir, "sir2"), "--quiet"])) == 0
            @test first(read_npy(joinpath(dir, "sir2_pmf.npy"))) ≈ fill(0.12, 4, 3) rtol = 1e-5
            # errors give exit code 1
            err(args) = redirect_stdout(() -> redirect_stderr(() -> VoxelFit.main(args), devnull), devnull)
            @test err(["fit", "--model", "MonoExp", "--x", "1,2", "--data", joinpath(dir, "d.npy"),
                       "--out", joinpath(dir, "bad")]) == 1
            @test err(["fit", "--model", "SIRFSE", "--data", joinpath(dir, "sir.npy"), "--acq", "TI=1,2,3,4,5",
                       "--out", joinpath(dir, "bad")]) == 1          # missing TD
            @test err(["fit", "--model", "SIRFSE", "--data", joinpath(dir, "sir.npy"), "--acq-file",
                       joinpath(dir, "acq.txt"), "--fixed", "kmff=1", "--out", joinpath(dir, "bad")]) == 1
        end
    end
end
