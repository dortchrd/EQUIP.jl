using VoxelFit
using VoxelFit: lm_fit, standard_errors, value_gradient, _svec, read_npy, write_npy, create_npy
using StaticArrays
using Random
using Statistics
using Test
using NIfTI
using KernelAbstractions

const CASES = [
    (MonoExp(), collect(10.0:10:320), [1000.0, 80.0]),
    (MonoExpOffset(), collect(10.0:10:320), [1000.0, 80.0, 50.0]),
    (BiExp(), collect(5.0:5:320), [300.0, 15.0, 700.0, 90.0]),
    (ADC(), [0, 100, 300, 500, 800, 1000.0], [1000.0, 1.5e-3]),
    (Kurtosis(), collect(0:250:2500.0), [1000.0, 1.5e-3, 0.9]),
    (IVIM(), [0, 10, 20, 40, 80, 110, 140, 170, 200, 300, 400, 500, 600, 700, 800, 1000.0],
     [1000.0, 0.1, 0.02, 1.2e-3]),
    (InversionRecovery(), [50, 100, 200, 400, 800, 1600, 3200.0], [1000.0, 1.9, 900.0]),
    (SaturationRecovery(), [100, 300, 600, 1000, 2000, 4000.0], [1000.0, 1200.0]),
    (SPGR(TR=15.0), [2, 5, 10, 15, 20, 30.0], [5000.0, 1000.0]),
]

# Whole-curve version of MonoExp, for testing the curve-model path.
struct CurveMonoExp <: AbstractCurveModel{2} end
VoxelFit.curve!(out, ::CurveMonoExp, x, θ) = (out .= θ[1] .* exp.(-x ./ θ[2]); out)
VoxelFit.initial_guess(::CurveMonoExp, x, y) = initial_guess(MonoExp(), x, y)
VoxelFit.lower_bounds(::CurveMonoExp) = lower_bounds(MonoExp())
VoxelFit.paramnames(::CurveMonoExp) = (:S0, :T2)

signal(m, x, θ) = [m(xi, SVector{length(θ)}(θ)) for xi in x]

function synth_monoexp(sz; σ=5.0, seed=1)
    rng = MersenneTwister(seed)
    te = collect(10.0:10.0:320.0)
    T2 = 40 .+ 80 .* rand(rng, sz...)
    S0 = 800 .+ 400 .* rand(rng, sz...)
    data = S0 .* exp.(-reshape(te, ntuple(_ -> 1, length(sz))..., :) ./ T2) .+
           σ .* randn(rng, sz..., length(te))
    return te, data, S0, T2
end

@testset "VoxelFit" begin
    @testset "noise-free recovery: $(typeof(m))" for (m, x, θ) in CASES
        y = signal(m, x, θ)
        for T in (Float64, Float32)
            r = fit_voxel(m, x, y; precision=T)
            @test isconverged(r)
            @test r.θ ≈ θ rtol = (T == Float64 ? 1e-6 : 2e-3)
        end
    end

    @testset "allocation-free solver" begin
        for (m, x, θ) in CASES, T in (Float64, Float32)
            P = nparams(m)
            y = T.(signal(m, x, θ) .* (1 .+ 0.01 .* randn(MersenneTwister(0), length(x))))
            xs = T.(x)
            lo, hi = _svec(lower_bounds(m), Val(P), T), _svec(upper_bounds(m), Val(P), T)
            θ0 = SVector{P,T}(initial_guess(m, xs, y))
            opts = LMOptions{T}()
            lm_fit(m, xs, y, θ0, lo, hi, opts)
            @test (@allocated lm_fit(m, xs, y, θ0, lo, hi, opts)) == 0
            @test (@allocated initial_guess(m, xs, y)) == 0
        end
    end

    @testset "analytic gradients match AD" begin
        for (m, x, θ) in CASES[[1, 4]]
            P = length(θ)
            # A PointModel wrapping the same function always uses automatic differentiation.
            ad = PointModel((xi, t) -> m(xi, t), paramnames(m); p0=θ)
            for xi in x
                f1, g1 = value_gradient(m, xi, SVector{P}(θ))
                f2, g2 = value_gradient(ad, xi, SVector{P}(θ))
                @test f1 ≈ f2 && g1 ≈ g2
            end
        end
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
        @test isequal(a.params, b.params) && isequal(a.params, c.params) && isequal(a.params, d.params)
        @test a.status[1, 1, 1] == UInt8(VoxelFit.InvalidData)
        @test all(isnan, a[:T2][.!mask]) && all(==(0x00), a.status[.!mask])
        sel = mask .& (a.status .!= UInt8(VoxelFit.InvalidData))
        @test all(isconverged, a.status[sel])
        @test median(abs.(a[:T2][sel] .- T2[sel]) ./ T2[sel]) < 0.01
        # integer input data (e.g. raw NIfTI Int16)
        di = round.(Int16, clamp.(replace(data, NaN => 0), 0, 30000))
        f = fit_maps(MonoExp(), te, di; mask, precision=Float32)
        @test eltype(f.params) == Float32
        @test median(abs.(f[:T2][sel] .- T2[sel]) ./ T2[sel]) < 0.01
    end

    @testset "init, bounds, weights, stderr" begin
        te, data, S0, T2 = synth_monoexp((30, 30))
        r = fit_maps(MonoExp(), te, data; stderr=true)
        # warm start from a previous fit converges at once
        r2 = fit_maps(MonoExp(), te, data; init=r.params)
        @test maximum(r2.iterations) <= 2
        @test r2.params ≈ r.params rtol = 1e-6
        # constant init gives the same answer
        r3 = fit_maps(MonoExp(), te, data; init=[1000, 60])
        @test r3.params ≈ r.params rtol = 1e-5
        # standard errors are calibrated: z-scores have unit spread
        z = (r[:T2] .- T2) ./ r.stderr[:, :, 2]
        @test 0.85 < std(vec(z)) < 1.15
        # an active upper bound
        r4 = fit_maps(MonoExp(), te, data; upper=[Inf, 60.0])
        @test all(r4[:T2] .<= 60.0)
        @test all(isconverged, r4.status)
        @test count(==(60.0), r4[:T2]) > 0
        # zero weight on a corrupted echo removes its influence
        bad = copy(data)
        bad[:, :, 3] .+= 300
        w = ones(length(te))
        w[3] = 0
        rw = fit_maps(MonoExp(), te, bad; weights=w)
        @test median(abs.(rw[:T2] .- T2) ./ T2) < 0.01
    end

    @testset "covariates (SPGR with B1)" begin
        rng = MersenneTwister(3)
        m = SPGR(TR=15.0)
        fa = [2, 5, 10, 15, 20, 30.0]
        n = 400
        T1, M0, B1 = 600 .+ 1000 .* rand(rng, n), 4000 .+ 2000 .* rand(rng, n), 0.8 .+ 0.4 .* rand(rng, n)
        Y = [m(fa[i], SVector(M0[v], T1[v]), SVector(B1[v])) for v in 1:n, i in eachindex(fa)]
        r = fit_maps(m, fa, Y; covariates=B1)
        @test r[:T1] ≈ T1 rtol = 1e-6
        rk = fit_maps(m, fa, Y; covariates=reshape(B1, n, 1))
        @test rk.params == r.params
    end

    @testset "custom models" begin
        te = collect(10.0:10:320)
        θ = [1000.0, 70.0, 30.0]
        pm = PointModel((t, p) -> p[1] * exp(-t / p[2]) + p[3], (:S0, :T2, :C); p0=(900, 50, 0),
                        lower=(0, 0, 0))
        em = expr_model("S0 * exp(-TE / T2) + C"; params=(:S0, :T2, :C), x=:TE, p0=(900, 50, 0))
        y = signal(MonoExpOffset(), te, θ)
        @test fit_voxel(pm, te, y).θ ≈ θ rtol = 1e-6
        @test Base.invokelatest(fit_voxel, em, te, y).θ ≈ θ rtol = 1e-6
        @test paramnames(em) == (:S0, :T2, :C)
        # several settings per observation (x as an nobs × 2 matrix)
        x = permutedims(reduce(hcat, [[a, b] for a in (10.0, 30.0, 60.0, 90.0) for b in (300.0, 800.0, 2000.0, 5000.0)]))
        em2 = expr_model("S0 * exp(-TE / T2) * (1 - exp(-TR / T1))"; params=(:S0, :T2, :T1),
                         x=(:TE, :TR), p0=(1000, 60, 1000))
        θ2 = [1000.0, 70.0, 1200.0]
        y2 = [θ2[1] * exp(-x[i, 1] / θ2[2]) * (1 - exp(-x[i, 2] / θ2[3])) for i in axes(x, 1)]
        @test Base.invokelatest(fit_voxel, em2, x, y2).θ ≈ θ2 rtol = 1e-6
        # model specs
        @test parse_model("SPGR(TR=15)") == SPGR(TR=15.0)
        @test parse_model("IVIM(bthresh=250)").bthresh == 250
        @test parse_model("InversionRecovery(magnitude=false)").magnitude == false
        @test_throws ArgumentError parse_model("Foo")
        @test_throws ArgumentError parse_model("SPGR(TR=rm(\"x\"))")
    end

    @testset "curve models" begin
        # Mono-exponential written as a whole-curve model must match the pointwise version.
        te, data, _, _ = synth_monoexp((10, 10))
        a = fit_maps(MonoExp(), te, data)
        b = fit_maps(CurveMonoExp(), te, data; ntasks=2)
        @test a.params ≈ b.params rtol = 1e-8
    end

    # KernelAbstractions ≥ 0.10 runs CPU() kernels through POCL (OpenCL/SPIR-V), so this
    # exercises device compilation and argument conversion as a GPU would. Floating-point
    # contraction differs from the threaded path, hence approximate comparisons.
    @testset "KernelAbstractions backend matches CPU threads" begin
        te, data, _, _ = synth_monoexp((16, 12, 3))
        data[2, 2, 2, 5] = Inf
        mask = rand(MersenneTwister(5), 16, 12, 3) .> 0.2
        for T in (Float32, Float64)
            a = fit_maps(MonoExp(), te, data; mask, precision=T, stderr=true)
            b = fit_maps(MonoExp(), te, data; mask, precision=T, stderr=true, backend=CPU(), batchsize=100)
            tol = T == Float32 ? 1e-3 : 1e-7
            @test isequal(isnan.(a.params), isnan.(b.params))
            @test isapprox(filter(isfinite, a.params), filter(isfinite, b.params); rtol=tol)
            @test isapprox(filter(isfinite, a.stderr), filter(isfinite, b.stderr); rtol=10tol)
            @test isconverged.(a.status) == isconverged.(b.status)
            @test (a.status .== 0x06) == (b.status .== 0x06)
        end
        m, fa = SPGR(TR=15.0), [2, 5, 10, 15, 20, 30.0]
        B1 = 0.8 .+ 0.4 .* rand(MersenneTwister(6), 8, 8)
        Y = [m(fa[i], SVector(5000.0, 1000.0), SVector(B1[I])) for i in 1:6, I in CartesianIndices(B1)]
        a = fit_maps(m, fa, Y; obsdim=1, covariates=B1)
        b = fit_maps(m, fa, Y; obsdim=1, covariates=B1, backend=CPU(), init=fill(1.0, 8, 8, 2) .* reshape([4000, 900], 1, 1, 2))
        @test a[:T1] ≈ b[:T1] rtol = 1e-6
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
            # header written by NumPy for a C-ordered (2, 3) uint8 array
            raw = UInt8[0x93; collect(b"NUMPY"); 0x01; 0x00; 0x76; 0x00;
                        collect(rpad("{'descr': '|u1', 'fortran_order': False, 'shape': (2, 3), }", 117) * "\n");
                        0x01:0x06]
            write(joinpath(dir, "np.npy"), raw)
            D, rev = read_npy(joinpath(dir, "np.npy"))
            @test rev && permutedims(D) == UInt8[1 2 3; 4 5 6]
        end
    end

    @testset "command line" begin
        mktempdir() do dir
            te, data, _, T2 = synth_monoexp((12, 10, 3))
            data32 = Float32.(data)
            # NIfTI in, NIfTI out
            niwrite(joinpath(dir, "d.nii.gz"), NIVolume(data32; voxel_size=(2.0, 2.0, 3.0)))
            niwrite(joinpath(dir, "mask.nii"), NIVolume(UInt8.(T2 .> 60)))
            open(joinpath(dir, "te.txt"), "w") do io
                foreach(t -> println(io, t), te)
            end
            code = redirect_stdout(devnull) do
                VoxelFit.main(["fit", "--model", "MonoExp", "--x-file", joinpath(dir, "te.txt"),
                               "--data", joinpath(dir, "d.nii.gz"), "--mask", joinpath(dir, "mask.nii"),
                               "--out", joinpath(dir, "out", "s"), "--stderr"])
            end
            @test code == 0
            t2 = niread(joinpath(dir, "out", "s_T2.nii.gz"))
            @test size(t2) == size(T2) && t2.header.pixdim[2:4] == (2.0f0, 2.0f0, 3.0f0)
            sel = T2 .> 60
            @test median(abs.(t2.raw[sel] .- T2[sel]) ./ T2[sel]) < 0.01
            @test all(isnan, t2.raw[.!sel])
            @test isfile(joinpath(dir, "out", "s_stderr_S0.nii.gz"))
            # NumPy (C order, observations last in NumPy's axis order) in, .npy out
            write_npy(joinpath(dir, "d.npy"), permutedims(data32, (4, 3, 2, 1)); reversed=true)
            code = redirect_stdout(devnull) do
                VoxelFit.main(["fit", "--model", "MonoExp", "--x", join(te, ","), "--data",
                               joinpath(dir, "d.npy"), "--out", joinpath(dir, "np"), "--precision", "double"])
            end
            @test code == 0
            t2n, rev = read_npy(joinpath(dir, "np_T2.npy"))
            @test rev && eltype(t2n) == Float64
            @test permutedims(t2n, (3, 2, 1)) ≈ fit_maps(MonoExp(), te, data32)[:T2] rtol = 1e-5
            @test readlines(joinpath(dir, "np_params.txt")) == ["S0", "T2"]
            # custom expression and errors
            code = redirect_stdout(devnull) do
                VoxelFit.main(["fit", "--expr", "S0*exp(-TE/T2)", "--params", "S0,T2", "--xname", "TE",
                               "--p0", "1000,50", "--x", join(te, ","), "--data", joinpath(dir, "d.npy"),
                               "--out", joinpath(dir, "ex"), "--quiet"])
            end
            @test code == 0
            @test read_npy(joinpath(dir, "ex_T2.npy"))[1] ≈ t2n rtol = 1e-4
            @test redirect_stderr(devnull) do
                VoxelFit.main(["fit", "--model", "MonoExp", "--x", "1,2", "--data", joinpath(dir, "d.npy"),
                               "--out", joinpath(dir, "bad")])
            end == 1
        end
    end
end
