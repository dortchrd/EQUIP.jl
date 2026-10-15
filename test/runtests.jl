# Tests for VoxelFit. Run with threads so the parallel path is exercised:
#     julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
using VoxelFit
using VoxelFit: setup, freenames      # internal functions that the tests inspect
using LinearAlgebra
using Random
using Test

# A SIR-FSE protocol used throughout (s)
const TI5 = [0.010, 0.050, 0.056, 0.277, 0.843]
const TD5 = [3.270, 4.489, 1.652, 2.922, 0.010]

# Each case: a model (with its acquisition), and the true values of EVERY parameter.
const CASES = [
    (ExpDecay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, T2 = 0.08)),
    (InversionRecovery(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]), (M0 = 1000.0, B = 1.9, T1 = 0.9)),
    (SaturationRecovery(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]), (M0 = 1000.0, T1 = 1.2)),
    (MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015), (M0 = 1000.0, T1 = 1.4, B1 = 1.0)),
    (SIRFSE(TI = TI5, TD = TD5),
     (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, R1m = 1.0, Sm = 0.83)),
]

# SIR-FSE signal straight from Eq. 1 of Dortch et al., MRM 2018, with matrix exponentials:
#   Mz(TI, TD) = {exp(A·TI)·S·[I − exp(A·TD)] + [I − exp(A·TI)]}·M0.
# An independent check of the closed-form solution in `signal!(out, ::SIRFSE, p)`.
function sir_eq1(TI, TD; PSR, R1f, Sf, M0f, kmf, R1m, Sm, magnitude = true)
    kfm = kmf * PSR
    A = [-(R1f + kfm) kmf; kfm -(R1m + kmf)]
    M0 = [M0f, PSR * M0f]
    S = Diagonal([Sf, Sm])
    return map(TI, TD) do ti, td
        Mz = (exp(A * ti) * S * (I - exp(A * td)) + (I - exp(A * ti))) * M0
        magnitude ? abs(Mz[1]) : Mz[1]
    end
end

@testset "VoxelFit" begin

    @testset "noise-free recovery from default starts: $(nameof(typeof(m)))" for (m, truth) in CASES
        y = signal(m, truth)
        for T in (Float64, Float32)
            r = fit!(init(m; precision = T), y)
            @test isconverged(r)
            for (name, v) in pairs(r.params)        # only the free parameters are returned
                @test v ≈ truth[name] rtol = (T == Float64 ? 1e-6 : 1e-3)
            end
        end
    end

    @testset "fit! does not allocate" begin
        for (m, truth) in CASES, T in (Float64, Float32)
            y = signal(m, truth) .* (1 .+ 0.01 .* randn(MersenneTwister(0), nobs(m)))
            c = init(m; precision = T)
            fit!(c, y)                              # first call compiles
            @test (@allocated fit!(c, y)) == 0
        end
    end

    @testset "models: keyword construction, single TD/TR, lengths, precision" begin
        # one number stands for every observation
        m = MFA(FA = [2, 5, 10], TR = 0.015)
        @test m.TR == [0.015, 0.015, 0.015] && nobs(m) == 3
        @test SIRFSE(TI = [0.01, 0.1], TD = 2.5).TD == [2.5, 2.5]
        @test SIRFSE(TI = [0.01, 0.1], TD = 2.5).magnitude                      # default option
        @test !SIRFSE(TI = [0.01, 0.1], TD = 2.5, magnitude = false).magnitude
        # ranges and integer vectors are accepted ...
        @test nobs(ExpDecay(TE = 0.01:0.01:0.1)) == 10
        # ... and init converts the acquisition to plain vectors of the working precision
        c = init(SIRFSE(TI = 0.01:0.01:0.05, TD = 2); precision = Float32)
        @test c.model.TI isa Vector{Float32} && c.model.TD == fill(2.0f0, 5) && c.model.magnitude
        # mistakes are caught
        @test_throws UndefKeywordError SIRFSE(TD = 2.5)                         # TI missing
        @test_throws DimensionMismatch init(SIRFSE(TI = [0.1, 1.0], TD = [1.0, 2.0, 3.0]))
        @test_throws DimensionMismatch fit!(init(CASES[1][1]), ones(5))         # wrong data length
    end

    @testset "SIRFSE matches Eq. 1 of the 2018 paper" begin
        TI = [0.010, 0.015, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0, 2.0, 5.0]
        θ = (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 14.5, R1m = 1.2, Sm = 0.83)
        for TD in (0.01, 1.0, 2.5, 5.0), mag in (true, false)
            @test signal(SIRFSE(TI = TI, TD = TD, magnitude = mag), θ) ≈
                  sir_eq1(TI, fill(TD, length(TI)); θ..., magnitude = mag)
        end
    end

    @testset "signal fills fixed defaults and resolves ties" begin
        m = SIRFSE(TI = [0.01, 0.1, 1.0], TD = 2.5)
        free = (PSR = 0.12, R1f = 1.1, Sf = -0.95, M0f = 1.0)
        everything = merge(free, (kmf = 12.5, R1m = 1.1, Sm = 0.83))
        @test signal(m, free) == signal(m, everything)
        @test signal(m, merge(free, (kmf = 20.0,))) != signal(m, free)
        @test_throws ArgumentError signal(m, (PSR = 0.1,))                     # Free value missing
        @test_throws ArgumentError signal(m, merge(free, (nope = 1.0,)))       # unknown name
        @test_throws DimensionMismatch signal(SIRFSE(TI = [0.1, 1.0], TD = [1.0, 2.0, 3.0]), free)
    end

    @testset "parameter roles (setup)" begin
        m = SIRFSE(TI = TI5, TD = TD5)
        s = setup(m)
        @test freenames(s) == (:PSR, :R1f, :Sf, :M0f)
        @test keys(s.fixed) == (:kmf, :R1m, :Sm) && s.fixed.kmf == 12.5
        @test s.lower == [0, 0.05, -1.2, 0] && s.start.PSR == 0.1
        # fix a parameter that is free by default
        @test freenames(setup(m; fixed = (PSR = 0.1,))) == (:R1f, :Sf, :M0f)
        # change a fixed value
        @test setup(m; fixed = (kmf = 15.0,)).fixed.kmf == 15.0
        # free a fixed parameter by name: its value becomes the start
        s = setup(m; free = (:kmf,))
        @test freenames(s) == (:PSR, :R1f, :Sf, :M0f, :kmf) && s.start.kmf == 12.5
        @test freenames(setup(m; free = :kmf)) == freenames(s)
        # ... or with a full role
        s = setup(m; free = (kmf = Free(20.0; lower = 5, upper = 50),))
        @test s.lower[end] == 5 && s.upper[end] == 50 && s.start.kmf == 20
        # start values override the defaults; precision is applied
        mfa = MFA(FA = [2, 10], TR = 0.015)
        s = setup(mfa, Float32; start = (T1 = 2,))
        @test s.start == (M0 = 1.0f0, T1 = 2.0f0)          # M0 = 1: placeholder for :scale
        @test s.scale == 1                                  # M0 is the :scale parameter
        @test setup(mfa; start = (M0 = 500.0,)).scale == 0  # an explicit start replaces :scale
        @test setup(m).scale == 4                           # M0f
        @test_throws ArgumentError Free(:auto)              # only numbers or :scale
        # mistakes are caught
        @test_throws ArgumentError setup(m; fixed = (kmff = 1.0,))          # misspelled
        @test_throws ArgumentError setup(m; free = (:nope,))
        @test_throws ArgumentError setup(m; fixed = (kmf = Free(1.0),))     # wrong keyword
        @test_throws ArgumentError setup(m; start = (kmf = 1.0,))           # kmf is fixed
        @test_throws ArgumentError setup(m; free = (:R1m,))                 # a tie needs a start
        @test_throws ArgumentError setup(m; fixed = (kmf = 1.0,), free = (:kmf,))
        @test_throws ArgumentError setup(ExpDecay(TE = [0.01]); fixed = (S0 = 1.0, T2 = 0.1))  # nothing free
    end

    @testset "SIRFSE: kmf fixed vs free, tie, per-voxel Sm map" begin
        m = SIRFSE(TI = TI5, TD = TD5)
        truth = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0, R1m = 1.0, Sm = 0.83)
        y = signal(m, truth)
        # kmf estimated: exact
        r = fit!(init(m; free = (kmf = Free(12.5; lower = 1, upper = 50),)), y)
        @test r.params.kmf ≈ 15 rtol = 1e-5
        @test r.params.PSR ≈ 0.12 rtol = 1e-5
        # kmf assumed correctly: exact; assumed wrongly (default 12.5): PSR is biased
        @test fit!(init(m; fixed = (kmf = 15.0,)), y).params.PSR ≈ 0.12 rtol = 1e-6
        @test abs(fit!(init(m), y).params.PSR - 0.12) > 1e-4
        # untie R1m by fixing it to a number
        y2 = signal(m, merge(truth, (kmf = 12.5, R1m = 2.0)))
        @test fit!(init(m; fixed = (R1m = 2.0,)), y2).params.R1f ≈ 1.0 rtol = 1e-6
        # per-voxel Sm map through fit_maps (data: voxels × observations)
        n = 50
        sm = 0.75 .+ 0.15 .* rand(MersenneTwister(4), n)
        Y = reduce(vcat, [signal(m, merge(truth, (kmf = 12.5, Sm = sm[v])))' for v in 1:n])
        res = fit_maps(m, Y; fixed = (Sm = sm,))
        @test res.params.PSR ≈ fill(0.12, n) rtol = 1e-5
        # a NaN in a map marks only that voxel invalid
        sm[7] = NaN
        res = fit_maps(m, Y; fixed = (Sm = sm,))
        @test res.status[7] == UInt8(VoxelFit.InvalidData) && isnan(res.params.PSR[7])
        @test all(isconverged, res.status[setdiff(1:n, 7)])
    end

    @testset "fit_maps: 3-D data, mask, threads, invalid data, start maps" begin
        rng = MersenneTwister(1)
        nx, ny, nz = 20, 16, 4
        te = collect(0.01:0.01:0.32)
        m = ExpDecay(TE = te)
        T2 = 0.04 .+ 0.08 .* rand(rng, nx, ny, nz)
        S0 = 800 .+ 400 .* rand(rng, nx, ny, nz)
        data = S0 .* exp.(-reshape(te, 1, 1, 1, :) ./ T2) .+ 2 .* randn(rng, nx, ny, nz, length(te))
        mask = rand(rng, nx, ny, nz) .> 0.3
        data[1, 1, 1, 5] = NaN; mask[1, 1, 1] = true            # one bad voxel inside the mask

        res = fit_maps(m, data; mask)
        @test keys(res) == (:params, :sse, :status) && keys(res.params) == (:S0, :T2)
        @test size(res.params.T2) == (nx, ny, nz)
        @test res.status[1, 1, 1] == UInt8(VoxelFit.InvalidData)
        @test all(isnan, res.params.T2[.!mask]) && all(==(0), res.status[.!mask])
        good = mask .& (res.status .!= UInt8(VoxelFit.InvalidData))
        @test all(isconverged, res.status[good])
        @test maximum(abs.(res.params.T2[good] .- T2[good]) ./ T2[good]) < 0.05

        # the threaded result equals fitting voxel by voxel with one cache
        c = init(m)
        for I in CartesianIndices(mask)[good][1:20]
            @test fit!(c, data[I, :]).params.T2 == res.params.T2[I]
        end

        # Float32 data and outputs; starting from the previous result's maps
        res32 = fit_maps(m, Float32.(data); mask, precision = Float32, start = res.params)
        @test eltype(res32.params.T2) == Float32
        @test res32.params.T2[good] ≈ res.params.T2[good] rtol = 1e-3

        # size mistakes are caught
        @test_throws DimensionMismatch fit_maps(ExpDecay(TE = te[1:5]), data)
        @test_throws DimensionMismatch fit_maps(m, data; mask = mask[:, :, 1])
        @test_throws DimensionMismatch fit_maps(m, data; start = (T2 = T2[:, :, 1],))
    end

    @testset "MFA with a per-voxel B1 map" begin
        rng = MersenneTwister(3)
        m = MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
        B1 = 0.8 .+ 0.4 .* rand(rng, 16, 16)
        T1 = 0.6 .+ 1.0 .* rand(rng, 16, 16)
        data = zeros(16, 16, 6)
        for I in CartesianIndices(B1)
            data[I, :] = signal(m, (M0 = 1000.0, T1 = T1[I], B1 = B1[I]))
        end
        with = fit_maps(m, data; fixed = (B1 = B1,))
        without = fit_maps(m, data)                          # B1 = 1 everywhere: biased
        @test with.params.T1 ≈ T1 rtol = 1e-6
        @test maximum(abs.(without.params.T1 .- T1) ./ T1) > 0.1
    end
end
