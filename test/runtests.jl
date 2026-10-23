# Tests for EQUIP. Run with threads so the parallel path is exercised:
#     julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
using EQUIP
using EQUIP: setup, freenames      # internal functions that the tests inspect
using LinearAlgebra
using Random
using Statistics: mean, std
using Test

# A model for testing `datastart`: T2 starts at 0.05·c, from the fixed parameter c (as DESPOT1
# needs the voxel's B1), and at NaN when c < 0, which must make that voxel's fit fail.
struct DataStartTest <: MRIModel
    t::Vector{Float64}
end
EQUIP.parameters(::DataStartTest) = (S0 = Free(:scale; lower = 0), T2 = Free(:data; lower = 1e-6), c = Fixed(1.0))
function EQUIP.signal!(out, m::DataStartTest, p)
    for j in eachindex(out, m.t)
        out[j] = p.S0 * abs(p.c) * exp(-m.t[j] / p.T2)
    end
    return out
end
EQUIP.datastart(::DataStartTest, y, p) = (T2 = p.c < 0 ? NaN : 0.05 * p.c,)

# A SIR protocol used throughout (s): the optimized 5-point scheme of Dortch et al., MRM 2018
const TI5 = [0.010, 0.050, 0.056, 0.277, 0.843]
const TD5 = [3.270, 4.489, 1.652, 2.922, 0.010]
# An inversion recovery protocol with a varying delay TD (s)
const TI6 = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4]
const TD6 = [0.5, 3.0, 1.0, 0.2, 2.0, 0.05]

# Each case: a model (with its acquisition), and the true values of EVERY parameter.
const CASES = [
    (T2Decay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, T2 = 0.08)),
    (R2Decay(TE = collect(0.003:0.003:0.036)), (S0 = 1000.0, R2 = 33.0)),
    (T2DecayOffset(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, T2 = 0.08, C = 50.0)),
    (R2DecayOffset(TE = collect(0.003:0.003:0.036)), (S0 = 1000.0, R2 = 33.0, C = 50.0)),
    (ADCDecay(b = [0.0, 250, 500, 750, 1000, 1500, 2000]), (S0 = 1000.0, ADC = 0.8e-3)),
    (ADCKurtosis(b = [0.0, 250, 500, 1000, 1500, 2000, 2500]), (S0 = 1000.0, ADC = 0.8e-3, K = 1.0)),
    (T2BiexpDecay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08)),
    (R2BiexpDecay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, fshort = 0.15, R2short = 50.0, R2long = 12.5)),
    (T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]), (M0 = 1000.0, Sf = -0.9, T1 = 0.9)),
    (T1SR(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]), (M0 = 1000.0, T1 = 1.2)),
    (T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015), (M0 = 1000.0, T1 = 1.4, B1 = 1.0)),
    (R1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]), (M0 = 1000.0, Sf = -0.9, R1 = 1 / 0.9)),
    (T1IRTD(TI = TI6, TD = TD6), (M0 = 1000.0, Sf = -0.9, T1 = 1.3, Mz_td0 = 0.0)),
    (R1IRTD(TI = TI6, TD = TD6), (M0 = 1000.0, Sf = -0.9, R1 = 1 / 1.3, Mz_td0 = 0.0)),
    (R1SR(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]), (M0 = 1000.0, R1 = 1 / 1.2)),
    (R1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015), (M0 = 1000.0, R1 = 1 / 1.4, B1 = 1.0)),
    (QMTSIR(TI = TI5, TD = TD5),
     (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, R1m = 1.0, Sm = 0.83,
      Mzf_td0 = 0.0, Mzm_td0 = 0.0)),
]

# SIR signal straight from the matrix form (Dortch et al., MRM 2018, Eq. 1), with matrix
# exponentials, extended to start TD at Mz = Z·M0 instead of 0, with Z = diag(Mzf_td0, Mzm_td0):
#   Mz(TI, TD) = {exp(A·TI)·S·[I − exp(A·TD)·(I − Z)] + [I − exp(A·TI)]}·M0.
# An independent check of the closed-form solution in `signal!(out, ::QMTSIR, p)`.
function sir_matrix(TI, TD; PSR, R1f, Sf, M0f, kmf, R1m, Sm, Mzf_td0 = 0.0, Mzm_td0 = 0.0,
                    magnitude = true)
    kfm = kmf * PSR
    A = [-(R1f + kfm) kmf; kfm -(R1m + kmf)]
    M0 = [M0f, PSR * M0f]
    S = Diagonal([Sf, Sm])
    Z = Diagonal([Mzf_td0, Mzm_td0])
    return map(TI, TD) do ti, td
        Mz = (exp(A * ti) * S * (I - exp(A * td) * (I - Z)) + (I - exp(A * ti))) * M0
        magnitude ? abs(Mz[1]) : Mz[1]
    end
end

@testset "EQUIP" begin

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

    @testset "signal! checks lengths when called directly: $(nameof(typeof(m)))" for (m, truth) in CASES
        n = nobs(m)
        @test signal!(zeros(n), m, truth) ≈ signal(m, truth)
        @test_throws DimensionMismatch signal!(zeros(n + 1), m, truth)   # would read past the end
        @test_throws DimensionMismatch signal!(zeros(n - 1), m, truth)
    end

    @testset "fit! does not allocate" begin
        for (m, truth) in CASES, T in (Float64, Float32)
            y = signal(m, truth) .* (1 .+ 0.01 .* randn(MersenneTwister(0), nobs(m)))
            c = init(m; precision = T)
            fit!(c, y)                              # first call compiles
            @test (@allocated fit!(c, y)) == 0
            @inferred fit!(c, y)                    # the result's type is known when compiling
        end
    end

    @testset "models: keyword construction, single TD/TR, lengths, precision" begin
        # one number stands for every observation
        m = T1MFA(FA = [2, 5, 10], TR = 0.015)
        @test m.TR == [0.015, 0.015, 0.015] && nobs(m) == 3
        @test QMTSIR(TI = [0.01, 0.1], TD = 2.5).TD == [2.5, 2.5]
        @test QMTSIR(TI = [0.01, 0.1], TD = 2.5).magnitude                      # default option
        @test !QMTSIR(TI = [0.01, 0.1], TD = 2.5, magnitude = false).magnitude
        # ranges and integer vectors are accepted ...
        @test nobs(T2Decay(TE = 0.01:0.01:0.1)) == 10
        # ... and init converts the acquisition to plain vectors of the working precision
        c = init(QMTSIR(TI = 0.01:0.01:0.05, TD = 2); precision = Float32)
        @test c.model.TI isa Vector{Float32} && c.model.TD == fill(2.0f0, 5) && c.model.magnitude
        # mistakes are caught
        @test_throws UndefKeywordError QMTSIR(TD = 2.5)                         # TI missing
        @test_throws DimensionMismatch init(QMTSIR(TI = [0.1, 1.0], TD = [1.0, 2.0, 3.0]))
        @test_throws DimensionMismatch fit!(init(CASES[1][1]), ones(5))         # wrong data length
    end

    @testset "QMTSIR matches the matrix form, also with nonzero start values" begin
        TI = [0.010, 0.015, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0, 2.0, 5.0]
        θ = (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 14.5, R1m = 1.2, Sm = 0.83)
        for TD in (0.01, 1.0, 2.5, 5.0), mag in (true, false)
            @test signal(QMTSIR(TI = TI, TD = TD, magnitude = mag), θ) ≈
                  sir_matrix(TI, fill(TD, length(TI)); θ..., magnitude = mag)
        end
        # random parameters, including Mz left over from the readout (start values ≠ 0)
        rng = MersenneTwister(7)
        for _ in 1:200
            θr = (PSR = 0.3 * rand(rng), R1f = 0.3 + 2 * rand(rng), Sf = -1.1 * rand(rng), M0f = 1000 * rand(rng),
                  kmf = 1 + 40 * rand(rng), R1m = 0.3 + 3 * rand(rng), Sm = rand(rng),
                  Mzf_td0 = 0.4 * rand(rng) - 0.2, Mzm_td0 = 0.4 * rand(rng) - 0.2)
            @test signal(QMTSIR(TI = TI5, TD = TD5, magnitude = false), θr) ≈
                  sir_matrix(TI5, TD5; θr..., magnitude = false) rtol = 1e-10
        end
        # special cases with known answers
        m = QMTSIR(TI = TI5, TD = TD5, magnitude = false)
        θ0 = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)
        # no macromolecular pool (PSR = 0): single-pool inversion recovery (Li et al. 2010, Eq. 1)
        @test signal(m, merge(θ0, (PSR = 0.0, Mzf_td0 = 0.1))) ≈
              @. 1000 * (-0.95 * (1 - 0.9 * exp(-TD5)) * exp(-TI5) + 1 - exp(-TI5))
        # exchange off: a tiny kmf gives single-pool inversion recovery; kmf = 0 gives NaN
        @test signal(m, merge(θ0, (kmf = 1e-12,))) ≈
              @. 1000 * (-0.95 * (1 - exp(-TD5)) * exp(-TI5) + 1 - exp(-TI5))
        @test all(isnan, signal(m, merge(θ0, (kmf = 0.0,))))
        # start at equilibrium and no inversion: nothing changes
        @test signal(m, merge(θ0, (Sf = 1.0, Sm = 1.0, Mzf_td0 = 1.0, Mzm_td0 = 1.0))) ≈ fill(1000.0, 5)
    end

    @testset "signal fills fixed defaults and resolves ties" begin
        m = QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)
        θfree = (PSR = 0.12, R1f = 1.1, Sf = -0.95, M0f = 1.0)
        everything = merge(θfree, (kmf = 12.5, R1m = 1.1, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0))
        @test signal(m, θfree) == signal(m, everything)
        @test signal(m, merge(θfree, (kmf = 20.0,))) != signal(m, θfree)
        @test_throws ArgumentError signal(m, (PSR = 0.1,))                     # Free value missing
        @test_throws ArgumentError signal(m, merge(θfree, (nope = 1.0,)))       # unknown name
        @test_throws DimensionMismatch signal(QMTSIR(TI = [0.1, 1.0], TD = [1.0, 2.0, 3.0]), θfree)
        # roles are not values: passing parameters(m), or a Fixed(...), is a clear error
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        @test occursin("is a role", msg(() -> signal(m, parameters(m))))
        @test occursin("is a role", msg(() -> simulate(m, parameters(m); snr = 100)))
        @test occursin("write `kmf = 10.0`", msg(() -> signal(m, merge(θfree, (kmf = Fixed(10.0),)))))
        @test occursin("must give a value for `R1f`", msg(() -> signal(m, (PSR = 0.1,))))
    end

    @testset "roles print readably" begin
        @test repr(Free()) == "Free()"
        @test repr(Free(0.1; lower = 0, upper = 1)) == "Free(start = 0.1, lower = 0.0, upper = 1.0)"
        @test repr(Free(:scale; lower = 0)) == "Free(start = :scale, lower = 0.0)"
        @test repr(Free(start = [0.1, 0.2])) == "Free(start = 2-element Vector{Float64})"
        @test repr(Fixed(12.5)) == "Fixed(12.5)"
        @test repr(Fixed(:R1f)) == "Fixed(:R1f)"
        @test repr(Fixed(p -> p.R1f)) == "Fixed(<function>)"
        @test repr(Free(BestOf([0.3, 0.8]))) == "Free(start = BestOf([0.3, 0.8]))"
    end

    @testset "ties: by name (Fixed(:R1f)) and by function" begin
        m = QMTSIR(TI = TI5, TD = TD5)
        @test parameters(m).R1m == Fixed(:R1f)                       # the default for QMTSIR
        @test setup(m).fixed.R1m isa EQUIP._Tie{:R1f}       # name known to the compiler
        θfree = (PSR = 0.12, R1f = 1.1, Sf = -0.95, M0f = 1000.0)
        # simulation: tie by name == tie by function == the value written out
        @test signal(m, θfree) == signal(m, merge(θfree, (R1m = p -> p.R1f,))) ==
              signal(m, merge(θfree, (R1m = 1.1,)))
        # fitting: the override forms give the same fit as the default
        y = signal(m, θfree)
        r = fit(m, y)
        @test r.params == fit(m, y; R1m = :R1f).params
        @test r.params.PSR ≈ fit(m, y; R1m = p -> p.R1f).params.PSR rtol = 1e-10
        # a tie can follow a per-voxel map: an R1 map fixes R1f, and R1m follows it
        r1 = [0.7, 1.0, 1.4]
        Y = reduce(vcat, [permutedims(signal(m, merge(θfree, (R1f = v,)))) for v in r1])
        res = fit(m, Y; R1f = r1)
        @test keys(res.params) == (:PSR, :Sf, :M0f)
        @test res.params.PSR ≈ fill(0.12, 3) rtol = 1e-6
        # mistakes are caught before fitting
        @test_throws ArgumentError setup(m; R1m = :R1F)   # not a parameter
        @test_throws ArgumentError setup(m; R1m = :R1m)   # tied to itself
        @test_throws ArgumentError setup(m; Sm = :R1m)    # tied to a tie
        @test_throws ArgumentError signal(m, merge(θfree, (R1m = :nope,)))
    end

    @testset "parameter roles (setup): keywords named after the parameters" begin
        m = QMTSIR(TI = TI5, TD = TD5)
        s = setup(m)
        @test freenames(s) == (:PSR, :R1f, :Sf, :M0f)
        @test keys(s.fixed) == (:kmf, :R1m, :Sm, :Mzf_td0, :Mzm_td0) && s.fixed.kmf == 12.5
        @test s.lower == [0, 0.05, -1.2, 0] && s.start.PSR == 0.1
        # a value fixes a parameter that is free by default, or changes a fixed value
        @test freenames(setup(m; PSR = 0.1)) == (:R1f, :Sf, :M0f)
        @test setup(m; kmf = 15.0).fixed.kmf == 15.0
        @test setup(m; kmf = Fixed(15.0)).fixed.kmf == 15.0
        # Free() estimates a fixed parameter: its fixed value becomes the start, no bounds
        s = setup(m; kmf = Free())
        @test freenames(s) == (:PSR, :R1f, :Sf, :M0f, :kmf) && s.start.kmf == 12.5
        @test s.lower[end] == -Inf && s.upper[end] == Inf
        s = setup(m; kmf = Free(20.0; lower = 5, upper = 50))           # with start and bounds
        @test s.lower[end] == 5 && s.upper[end] == 50 && s.start.kmf == 20
        # Free(...) over a free parameter keeps whatever it leaves out
        s = setup(m; Sf = Free(lower = -1.5))
        @test s.lower[3] == -1.5 && s.upper[3] == 0 && s.start.Sf == -0.95
        s = setup(m; R1f = Free(start = 0.8))
        @test s.start.R1f == 0.8 && s.lower[2] == 0.05 && s.fromdata == (false, false, false, false)
        @test Free(0.8) == Free(start = 0.8)
        @test Free(:data).start === :data && Free(start = BestOf([0.3, 0.8])).start isa BestOf
        @test_throws ArgumentError Free(0.8; start = 0.9)               # the start given twice
        @test_throws ArgumentError Free(:auto)                         # not a start option
        # starts and precision; an explicit start replaces :scale
        mfa = T1MFA(FA = [2, 10], TR = 0.015)
        s = setup(mfa, Float32; T1 = Free(start = 2))
        @test s.start == (M0 = 1.0f0, T1 = 2.0f0)          # M0 = 1: placeholder for :scale
        @test s.scale == 1                                  # M0 is the :scale parameter
        @test setup(mfa; M0 = Free(start = 500.0)).scale == 0
        @test setup(m).scale == 4                           # M0f
        # mistakes are caught, with the parameters and options listed
        @test_throws ArgumentError setup(m; kmff = 1.0)                  # misspelled
        @test_throws ArgumentError setup(m; R1m = Free())                # a tie freed needs a start
        @test_throws ArgumentError setup(m; mask = trues(3))             # mask is for datasets
        @test_throws "solver = LM(maxiter" setup(m; maxiter = 50)        # solver settings go in LM
        ir = T1IR(TI = [0.1, 0.5, 1.0])
        @test_throws "M0 in T1IR" setup(ir; T1 = Free(start = :scale))                # only the model's scale
        @test_throws "M0 in T1IR" setup(ir; T1 = Free(start = :scale), M0 = Free(start = 1000.0))
        @test setup(ir; M0 = Free(start = 1000.0)).scale == 0                        # turned off: allowed
        @test setup(ir; M0 = Free(start = :scale)).scale == 1                        # the model's own: allowed
        @test_throws ArgumentError setup(T2Decay(TE = [0.01]); S0 = 1.0, T2 = 0.1)   # nothing free
        e = try setup(m; kmff = 1.0) catch err err end
        @test occursin("PSR, R1f", e.msg) && occursin("mask, precision, solver", e.msg)
    end

    @testset "QMTSIR: kmf fixed vs free, tie, per-voxel Sm map" begin
        m = QMTSIR(TI = TI5, TD = TD5)
        truth = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0, R1m = 1.0, Sm = 0.83,
                 Mzf_td0 = 0.0, Mzm_td0 = 0.0)
        y = signal(m, truth)
        # kmf estimated: exact
        r = fit!(init(m; kmf = Free(12.5; lower = 1, upper = 50)), y)
        @test r.params.kmf ≈ 15 rtol = 1e-5
        @test r.params.PSR ≈ 0.12 rtol = 1e-5
        # kmf assumed correctly: exact; assumed wrongly (default 12.5): PSR is biased
        @test fit!(init(m; kmf = 15.0), y).params.PSR ≈ 0.12 rtol = 1e-6
        @test abs(fit!(init(m), y).params.PSR - 0.12) > 1e-4
        # untie R1m by fixing it to a number
        y2 = signal(m, merge(truth, (kmf = 12.5, R1m = 2.0)))
        @test fit!(init(m; R1m = 2.0), y2).params.R1f ≈ 1.0 rtol = 1e-6
        # Mz left over from the readout biases the fit, unless the start values are given
        y3 = signal(m, merge(truth, (Mzf_td0 = 0.05, Mzm_td0 = 0.05)))
        @test fit!(init(m; kmf = 15.0, Mzf_td0 = 0.05, Mzm_td0 = 0.05), y3).params.PSR ≈ 0.12 rtol = 1e-6
        @test abs(fit!(init(m; kmf = 15.0), y3).params.PSR - 0.12) > 1e-4
        # per-voxel Sm map through fit on a dataset (data: voxels × observations)
        n = 50
        sm = 0.75 .+ 0.15 .* rand(MersenneTwister(4), n)
        Y = reduce(vcat, [signal(m, merge(truth, (kmf = 12.5, Sm = sm[v])))' for v in 1:n])
        res = fit(m, Y; Sm = sm)
        @test res.params.PSR ≈ fill(0.12, n) rtol = 1e-5
        # a NaN in a map marks only that voxel invalid
        sm[7] = NaN
        res = fit(m, Y; Sm = sm)
        @test res.status[7] == EQUIP.InvalidData && isnan(res.params.PSR[7])
        @test all(isconverged, res.status[setdiff(1:n, 7)])
    end

    @testset "fit on a dataset: 3-D data, mask, threads, invalid data, start maps" begin
        rng = MersenneTwister(1)
        nx, ny, nz = 20, 16, 4
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        T2 = 0.04 .+ 0.08 .* rand(rng, nx, ny, nz)
        S0 = 800 .+ 400 .* rand(rng, nx, ny, nz)
        data = S0 .* exp.(-reshape(te, 1, 1, 1, :) ./ T2) .+ 2 .* randn(rng, nx, ny, nz, length(te))
        mask = rand(rng, nx, ny, nz) .> 0.3
        data[1, 1, 1, 5] = NaN; mask[1, 1, 1] = true            # one bad voxel inside the mask

        res = fit(m, data; mask)
        @test res isa FitResult && keys(res.params) == (:S0, :T2)
        @test size(res.params.T2) == size(res.iterations) == size(res.status) == (nx, ny, nz)
        @test res.status[1, 1, 1] == EQUIP.InvalidData
        @test all(isnan, res.params.T2[.!mask]) && all(==(EQUIP.NotFitted), res.status[.!mask])
        @test all(==(0), res.iterations[.!mask]) && all(>(0), res.iterations[mask .& isconverged(res)])
        good = mask .& (res.status .!= EQUIP.InvalidData)
        @test all(isconverged, res.status[good])
        @test maximum(abs.(res.params.T2[good] .- T2[good]) ./ T2[good]) < 0.05

        # the threaded result equals fitting voxel by voxel with one cache
        c = init(m)
        for I in CartesianIndices(mask)[good][1:20]
            r = fit!(c, data[I, :])
            @test r.params.T2 == res.params.T2[I] && r.iterations == res.iterations[I]
        end

        # Float32 data and outputs; starting from the previous result's maps
        res32 = refit(res, Float32.(data); precision = Float32)
        @test eltype(res32.params.T2) == Float32
        @test res32.params.T2[good] ≈ res.params.T2[good] rtol = 1e-3

        # size mistakes are caught
        @test_throws DimensionMismatch fit(T2Decay(TE = te[1:5]), data)
        @test_throws DimensionMismatch fit(m, data; mask = mask[:, :, 1])
        @test_throws DimensionMismatch fit(m, data; T2 = Free(start = T2[:, :, 1]))
    end

    @testset "mono-exponential: time vs rate form, offset versions, diffusion" begin
        te = collect(0.01:0.01:0.32)
        rng = MersenneTwister(4)
        # the same data in both forms: the same least-squares answer, R2 = 1/T2, voxel by voxel
        mT, mR = T2Decay(TE = te), R2Decay(TE = te)
        for _ in 1:20
            y = signal(mT, (S0 = 1000.0, T2 = 0.05 + 0.1 * rand(rng))) .+ 10 .* randn(rng, length(te))
            rT, rR = fit(mT, y), fit(mR, y)
            @test isconverged(rT) && isconverged(rR)
            @test rR.params.R2 ≈ 1 / rT.params.T2 rtol = 1e-5
            @test rR.params.S0 ≈ rT.params.S0 rtol = 1e-5
        end
        # an offset: recovered by the offset versions, biases the plain one, or can be fixed
        θ = (S0 = 1000.0, T2 = 0.08, C = 50.0)
        y = signal(T2DecayOffset(TE = te), θ)
        r = fit(T2DecayOffset(TE = te), y)
        @test r.params.T2 ≈ 0.08 rtol = 1e-6
        @test r.params.C ≈ 50 rtol = 1e-6
        @test abs(fit(mT, y).params.T2 - 0.08) > 0.01
        @test fit(T2DecayOffset(TE = te), y; C = 50.0).params.T2 ≈ 0.08 rtol = 1e-6
        @test fit(R2DecayOffset(TE = te), y).params.R2 ≈ 1 / 0.08 rtol = 1e-6
        # diffusion in clinical units (b in s/mm², ADC in mm²/s), including fast decays,
        # from the default start
        b = [0.0, 500, 1000, 1500, 2000, 2500, 3000]
        m = ADCDecay(b = b)
        @test signal(m, (S0 = 1.0, ADC = 1e-3)) ≈ exp.(-b .* 1e-3)
        @test signal(m, (S0 = 1.0, ADC = 1e-3)) ≈ signal(R2Decay(TE = b), (S0 = 1.0, R2 = 1e-3))
        for adc in (0.1e-3, 0.7e-3, 3e-3, 5e-3)
            r = fit(m, 800 .* exp.(-b .* adc))
            @test r.params.ADC ≈ adc rtol = 1e-6
            @test r.params.S0 ≈ 800 rtol = 1e-6
        end
    end

    @testset "T1IRTD / R1IRTD: Li 2010 Eq. 1 and its limits" begin
        m = T1IRTD(TI = TI6, TD = TD6, magnitude = false)
        rng = MersenneTwister(8)
        for _ in 1:50
            M0, Sf, T1 = 1000 * rand(rng), -0.5 - 0.6 * rand(rng), 0.2 + 3 * rand(rng)
            # Li et al. 2010, Eq. 1, written as in the paper
            li = @. M0 * (Sf * (1 - exp(-TD6 / T1)) * exp(-TI6 / T1) + 1 - exp(-TI6 / T1))
            @test signal(m, (M0 = M0, Sf = Sf, T1 = T1)) ≈ li
            @test signal(R1IRTD(TI = TI6, TD = TD6, magnitude = false), (M0 = M0, Sf = Sf, R1 = 1 / T1)) ≈ li
        end
        θ = (M0 = 1000.0, T1 = 0.9)
        # a perfect inversion: the textbook form with TR = TI + TD
        @test signal(m, merge(θ, (Sf = -1.0,))) ≈ @. 1000 * (1 - 2 * exp(-TI6 / 0.9) + exp(-(TI6 + TD6) / 0.9))
        # a long TD, or Mz at equilibrium at the start of TD: T1IR, with the same parameters
        ir = signal(T1IR(TI = TI6, magnitude = false), (M0 = 1000.0, Sf = -0.9, T1 = 0.9))
        @test ir ≈ @. 1000 * (1 - 1.9 * exp(-TI6 / 0.9))                   # B = 1 - Sf
        @test signal(T1IRTD(TI = TI6, TD = 100.0, magnitude = false), merge(θ, (Sf = -0.9,))) ≈ ir
        @test signal(m, merge(θ, (Sf = -0.9, Mz_td0 = 1.0))) ≈ ir
        # one TD for every TI
        @test T1IRTD(TI = TI6, TD = 1.0).TD == fill(1.0, 6)
        @test R1IRTD(TI = TI6, TD = 1.0).TD == fill(1.0, 6)
    end

    @testset "solver: steps into a dead zone are rejected; bounds are still reached" begin
        # A step that pushes a time constant to ~0 leaves it with no influence on the signal
        # (every sample has fully recovered). Such steps are rejected, so short T1 is found from
        # the default start (T1 = 1 s); before the guard these fits ended at T1 = 1e-6.
        sr = T1SR(TD = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0])
        for T1 in (0.05, 0.1, 0.2)
            @test fit(sr, signal(sr, (M0 = 1000.0, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
        end
        mfa = T1MFA(FA = [2.0, 5, 10, 15, 20, 30], TR = 0.015)
        for T1 in (0.05, 0.2)
            @test fit(mfa, signal(mfa, (M0 = 1000.0, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
        end
        # fits whose answer lies on a bound still end exactly there (there the parameter still
        # affects the signal, so the guard does not apply)
        y = 1000 .+ 0.05 .* (1:32) .+ [isodd(j) ? 1.0 : -1.0 for j in 1:32]     # slightly rising
        r = fit(R2Decay(TE = collect(0.01:0.01:0.32)), y)
        @test r.params.R2 == 0
        @test isconverged(r)
        m = QMTSIR(TI = TI5, TD = TD5)
        r = fit(m, signal(m, (PSR = 0.0, R1f = 1.0, Sf = -0.95, M0f = 1000.0)))   # no MT
        @test r.params.PSR == 0
        @test r.params.R1f ≈ 1.0 rtol = 1e-6
        # the guard itself: parameter 2 loses (almost) all influence; frozen parameters don't count
        A = [4.0 0.0; 0.0 9.0]
        @test EQUIP._lost_influence(A, [4.0 0.0; 0.0 1e-20], [1.0, 1.0])
        @test !EQUIP._lost_influence(A, [4.0 0.0; 0.0 1e-3], [1.0, 1.0])
        @test !EQUIP._lost_influence(A, [4.0 0.0; 0.0 0.0], [1.0, 0.0])
    end

    @testset "bi-exponential T2 and diffusion kurtosis" begin
        te = collect(0.01:0.01:0.32)
        mT, mR = T2BiexpDecay(TE = te), R2BiexpDecay(TE = te)
        θT = (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08)
        θR = (S0 = 1000.0, fshort = 0.15, R2short = 50.0, R2long = 12.5)
        # the two forms are the same signal; the components can swap; fshort = 0 is mono-exponential
        @test signal(mT, θT) ≈ signal(mR, θR)
        @test signal(mT, θT) ≈ signal(mT, (S0 = 1000.0, fshort = 0.85, T2short = 0.08, T2long = 0.02))
        @test signal(mT, merge(θT, (fshort = 0.0,))) ≈ signal(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08))
        # relabel puts the shorter T2 (the larger rate) first, only when all three are free
        θs = EQUIP.relabel(mT, (S0 = 1000.0, fshort = 0.85, T2short = 0.08, T2long = 0.02))
        @test θs.T2short == 0.02
        @test θs.T2long == 0.08
        @test θs.fshort ≈ 0.15
        @test EQUIP.relabel(mR, (S0 = 1000.0, fshort = 0.85, R2short = 12.5, R2long = 50.0)).R2short == 50.0
        @test EQUIP.relabel(mT, (S0 = 1000.0, fshort = 0.85, T2short = 0.08)) ==
              (S0 = 1000.0, fshort = 0.85, T2short = 0.08)                       # T2long fixed
        @test EQUIP.relabel(T2Decay(TE = te), (S0 = 1.0, T2 = 0.1)) == (S0 = 1.0, T2 = 0.1)  # default
        # noisy data (SNR 100): both forms report the same, ordered components
        Y = reduce(vcat, [permutedims(signal(mT, θT) .+ 10 .* randn(MersenneTwister(i), 32)) for i in 1:100])
        rT, rR = fit(mT, Y), fit(mR, Y)
        @test all(rT.params.T2short .<= rT.params.T2long)
        @test all(rR.params.R2short .>= rR.params.R2long)
        # where both reach the same minimum (equal SSE) they agree; local minima are real in this
        # model (with these seeds, 1 voxel of 100 converged to different minima in the two forms)
        same = isconverged.(rT.status) .& isconverged.(rR.status) .& (abs.(rT.sse .- rR.sse) .<= 1e-6 .* rR.sse)
        @test count(same) >= 90
        @test rT.params.T2short[same] ≈ 1 ./ rR.params.R2short[same] rtol = 1e-4
        @test rT.params.fshort[same] ≈ rR.params.fshort[same] rtol = 1e-4
        # mono-exponential data: a degenerate but exact description (SSE ≈ 0), whichever it is
        y = signal(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08))
        for m in (mT, mR)
            r = fit(m, y)
            @test isconverged(r)
            @test r.sse < 1e-10
        end

        # kurtosis: K = 0 is ADCDecay; the equation; the signal decays while b·ADC·K < 3
        b = [0.0, 250, 500, 1000, 1500, 2000, 2500]
        mk = ADCKurtosis(b = b)
        @test signal(mk, (S0 = 1000.0, ADC = 1e-3, K = 0.0)) ≈ signal(ADCDecay(b = b), (S0 = 1000.0, ADC = 1e-3))
        @test signal(mk, (S0 = 1000.0, ADC = 1e-3, K = 1.2)) ≈
              1000 .* exp.(-b .* 1e-3 .+ (b .* 1e-3) .^ 2 .* 1.2 ./ 6)
        @test all(diff(signal(mk, (S0 = 1000.0, ADC = 0.8e-3, K = 1.0))) .< 0)    # 2500·0.8e-3·1 = 2
        @test fit(mk, signal(mk, (S0 = 1000.0, ADC = 2e-3, K = 0.3))).params.K ≈ 0.3 rtol = 1e-6
    end

    @testset "data start: log-linear T2 for T2Decay" begin
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        y = signal(m, (S0 = 1000.0, T2 = 0.04))
        # Noise-free data: the log-linear fit is exact, so the start is already the answer.
        # `maxiter = 0` returns the starting point; S0 comes from least squares given that T2.
        @test datastart(m, y, (;)).T2 ≈ 0.04
        s = fit(m, y; solver = LM(maxiter = 0)).params
        @test s.T2 ≈ 0.04
        @test s.S0 ≈ 1000
        @test fit(m, Float32.(y); precision = Float32, solver = LM(maxiter = 0)).params.T2 ≈ 0.04f0
        # points ≤ 0 are skipped
        @test datastart(m, [y[1:5]; zeros(27)], (;)).T2 ≈ 0.04
        # no usable estimate (rising data): the fallback, mean(TE)
        @test datastart(m, collect(1.0:32.0), (;)).T2 ≈ sum(te) / 32
        @test fit(m, collect(1.0:32.0); solver = LM(maxiter = 0)).params.T2 ≈ sum(te) / 32
        # T2 starts at :data by default; another start replaces it; a fixed T2 is left alone
        @test setup(m).fromdata == (false, true)
        @test fit(m, y; T2 = Free(start = 0.2), solver = LM(maxiter = 0)).params.T2 == 0.2
        @test setup(m; T2 = Free(start = 0.2)).fromdata == (false, false)
        @test fit(m, y; T2 = 0.04).params.S0 ≈ 1000
        # noisy data: the same answer as from the old start, without allocating
        yn = y .+ 10 .* randn(MersenneTwister(3), 32)
        r, r0 = fit(m, yn), fit(m, yn; T2 = Free(start = sum(te) / 32))
        @test r.params.T2 ≈ r0.params.T2 rtol = 1e-6
        @test r.iterations < r0.iterations
        c = init(m); fit!(c, yn)
        @test @allocated(fit!(c, yn)) == 0
        @test datastart(R2Decay(TE = te), y, (;)) === nothing     # no data start by default
        @test_throws "no data start" setup(R2Decay(TE = te); R2 = Free(start = :data))
        # datastart gets the voxel's fixed values (here c, a number or a map); a NaN fails
        mt = DataStartTest([0.01, 0.02, 0.04, 0.08])
        yt = signal(mt, (S0 = 100.0, T2 = 0.05))
        @test fit(mt, yt; solver = LM(maxiter = 0)).params.T2 ≈ 0.05
        @test fit(mt, permutedims([yt yt]); c = [1.0, 2.0], solver = LM(maxiter = 0)).params.T2 ≈ [0.05, 0.1]
        @test fit(mt, yt; c = -1.0).status == EQUIP.Failed
    end

    @testset "BestOf start: magnitude inversion recovery's local minima" begin
        # From a single start (T1 = 1 s), magnitude IR used to end in a wrong local minimum (the
        # signal null in the wrong place) at T1 = 0.05, 0.1 and 4 s with these TIs; starting from
        # the best of 30 values (BestOf) finds the right one, in both forms.
        ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        for T1 in (0.05, 0.1, 0.5, 4.0)
            m, mr = T1IR(TI = ti), R1IR(TI = ti)
            @test fit(m, signal(m, (M0 = 1000.0, Sf = -0.95, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
            @test fit(mr, signal(mr, (M0 = 1000.0, Sf = -0.95, R1 = 1 / T1))).params.R1 ≈ 1 / T1 rtol = 1e-6
        end
        for T1 in (0.05, 0.1, 0.2, 0.3, 4.0)
            m, mr = T1IRTD(TI = TI6, TD = TD6), R1IRTD(TI = TI6, TD = TD6)
            @test fit(m, signal(m, (M0 = 1000.0, Sf = -0.95, T1 = T1))).params.T1 ≈ T1 rtol = 1e-6
            @test fit(mr, signal(mr, (M0 = 1000.0, Sf = -0.95, R1 = 1 / T1))).params.R1 ≈ 1 / T1 rtol = 1e-6
        end
        # 30 values spanning the TIs; replaced by any other start, absent when T1 is fixed
        s = setup(T1IR(TI = ti))
        @test s.candindex == 3 && length(s.candidates) == 30
        @test s.candidates[1] ≈ 0.01 && s.candidates[end] ≈ 16
        @test setup(T1IR(TI = ti); T1 = Free(start = 2.0)).candindex == 0
        @test setup(T1IR(TI = ti); T1 = 2.0).candindex == 0
        @test setup(T2Decay(TE = [0.01, 0.02])).candindex == 0           # none by default
        # values from the call; those outside the bounds are dropped, an error if none is left
        s = setup(T1IR(TI = ti); T1 = Free(start = BestOf([0.1, 0.5, 5.0]), upper = 3))
        @test s.candidates == [0.1, 0.5] && s.start.T1 == 0.1           # middle value: placeholder
        @test_throws "no BestOf value within" setup(T1IR(TI = ti); T1 = Free(lower = 20))
        @test_throws "only one parameter" setup(T1IR(TI = ti); Sf = Free(start = BestOf([-1, -0.9])))
        m2 = T2Decay(TE = 0.01:0.01:0.32)
        r = fit(m2, signal(m2, (S0 = 1000.0, T2 = 0.04)); T2 = Free(start = BestOf([0.01, 0.04, 1.0])),
                solver = LM(maxiter = 0))
        @test r.params.T2 == 0.04 && r.params.S0 ≈ 1000                 # the best of the three
        # the values: any list of finite numbers; printed in short
        @test BestOf(0.1:0.1:0.3).values ≈ [0.1, 0.2, 0.3]
        @test_throws ArgumentError BestOf(Float64[])
        @test_throws ArgumentError BestOf([0.1, NaN])
        @test sprint(show, BestOf(1:10)) == "BestOf(10 values from 1.0 to 10.0)"
    end

    @testset "T1 and R1 forms: the same answer, R1 = 1/T1, voxel by voxel" begin
        rng = MersenneTwister(6)
        ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        td = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
        fa = [2.0, 5, 10, 15, 20, 30]
        pairs = ((T1IR(TI = ti), R1IR(TI = ti), (M0 = 1000.0, Sf = -0.9)),
                 (T1IRTD(TI = TI6, TD = TD6), R1IRTD(TI = TI6, TD = TD6), (M0 = 1000.0, Sf = -0.95)),
                 (T1SR(TD = td), R1SR(TD = td), (M0 = 1000.0,)),
                 (T1MFA(FA = fa, TR = 0.015), R1MFA(FA = fa, TR = 0.015), (M0 = 1000.0,)))
        for (mT, mR, θ) in pairs, _ in 1:10
            T1 = 0.6 + 1.2 * rand(rng)
            @test signal(mR, merge(θ, (R1 = 1 / T1,))) ≈ signal(mT, merge(θ, (T1 = T1,)))
            y = signal(mT, merge(θ, (T1 = T1,))) .* (1 .+ 0.01 .* randn(rng, nobs(mT)))
            rT, rR = fit(mT, y), fit(mR, y)
            @test isconverged(rT) && isconverged(rR)
            @test rR.params.R1 ≈ 1 / rT.params.T1 rtol = 1e-5
        end
    end

    @testset "T1MFA with a per-voxel B1 map" begin
        rng = MersenneTwister(3)
        m = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
        B1 = 0.8 .+ 0.4 .* rand(rng, 16, 16)
        T1 = 0.6 .+ 1.0 .* rand(rng, 16, 16)
        data = zeros(16, 16, 6)
        for I in CartesianIndices(B1)
            data[I, :] = signal(m, (M0 = 1000.0, T1 = T1[I], B1 = B1[I]))
        end
        with = fit(m, data; B1 = B1)
        without = fit(m, data)                     # B1 = 1 everywhere: biased
        @test with.params.T1 ≈ T1 rtol = 1e-6
        @test maximum(abs.(without.params.T1 .- T1) ./ T1) > 0.1
    end

    @testset "fit: one voxel (vector) or a dataset with any number of spatial dimensions" begin
        rng = MersenneTwister(5)
        m = T2Decay(TE = collect(0.01:0.01:0.32))
        curve(T2) = signal(m, (S0 = 1000.0, T2 = T2))
        # a vector is ONE voxel: plain numbers, same as init + fit!
        r = fit(m, curve(0.08))
        @test r isa FitResult && isconverged(r)
        @test r.params.T2 ≈ 0.08 rtol = 1e-6
        @test r.params == fit!(init(m), curve(0.08)).params
        @test fit(m, curve(0.08); S0 = 1000.0).params.T2 ≈ 0.08 rtol = 1e-6
        # matrices and N-d arrays: observations last, maps with the spatial size
        for sz in ((40,), (8, 5), (6, 5, 3), (4, 3, 3, 2))
            T2 = 0.04 .+ 0.08 .* rand(rng, sz...)
            data = zeros(sz..., nobs(m))
            for I in CartesianIndices(T2)
                data[I, :] = curve(T2[I])
            end
            res = fit(m, data)
            @test size(res.params.T2) == sz && size(res.status) == sz
            @test res.params.T2 ≈ T2 rtol = 1e-6
        end
    end

    @testset "FitResult: one type for one voxel and a dataset" begin
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        y = signal(m, (S0 = 1000.0, T2 = 0.08)) .+ 5 .* randn(MersenneTwister(7), 32)
        # one voxel: numbers, plus the model and settings it was fitted with
        r = fit(m, y; solver = LM(maxiter = 50))
        @test r.params.T2 isa Float64 && r.status isa FitStatus && r.iterations isa Int32
        @test r.model isa T2Decay && r.model.TE == te
        @test r.settings.solver.maxiter == 50 && r.settings.precision == Float64
        @test isconverged(r) isa Bool
        @test fitted(r) ≈ signal(m, r.params)
        @test residuals(r, y) ≈ y .- fitted(r)
        @test sum(abs2, residuals(r, y)) ≈ r.sse rtol = 1e-8
        # fixed values (numbers and ties) are used by `fitted`
        mq = QMTSIR(TI = [0.01, 0.03, 0.1, 0.3, 1.0], TD = 1.0)
        yq = signal(mq, (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0))
        rq = fit(mq, yq; kmf = 15.0)
        @test fitted(rq) ≈ yq rtol = 1e-6
        @test fit(mq, yq; kmf = Fixed(15.0)).params == rq.params                # role or number
        # a dataset: maps of the same fields; status is a FitStatus map (1 byte per voxel)
        T2 = 0.05 .+ 0.05 .* rand(MersenneTwister(8), 5, 4)
        data = zeros(5, 4, 32)
        for I in CartesianIndices(T2)
            data[I, :] = signal(m, (S0 = 1000.0, T2 = T2[I])) .+ 5 .* randn(MersenneTwister(I[1] + 10I[2]), 32)
        end
        mask = trues(5, 4); mask[1, 1] = false
        res = fit(m, data; mask)
        @test eltype(res.status) == FitStatus && sizeof(res.status) == 20
        @test eltype(res.iterations) == Int32
        @test UInt8.(res.status)[1, 1] == 0x00                                  # codes for files
        @test isconverged(res) isa BitMatrix && !isconverged(res)[1, 1]
        @test res.settings.mask === mask                                        # a reference
        @test fitted(res, 3, 2) ≈ signal(m, (S0 = res.params.S0[3, 2], T2 = res.params.T2[3, 2]))
        @test fitted(res, 8) == fitted(res, 3, 2)                               # linear index
        @test sum(abs2, residuals(res, data, 3, 2)) ≈ res.sse[3, 2] rtol = 1e-8
        @test all(isnan, fitted(res, 1, 1))                                     # outside the mask
        # refit with the same settings, or from the previous maps
        @test isequal(fit(res.model, data; res.settings...).params, res.params)
        @test refit(res, data).params.T2[mask] ≈ res.params.T2[mask] rtol = 1e-6
        @test all(refit(res, data).iterations[mask] .<= res.iterations[mask])   # starts at the answer
        r2 = refit(res, data; S0 = 1000.0)                                      # one change
        @test keys(r2.params) == (:T2,) && r2.settings.S0 == 1000.0
        # the wrong kind of call is caught
        @test_throws ArgumentError fitted(res)
        @test_throws ArgumentError fitted(r, 1)
        # display: one line, and the summary at the REPL
        @test occursin("one voxel", sprint(show, r)) && occursin("converged", sprint(show, res))
        @test occursin("status   Converged", sprint(show, MIME("text/plain"), r))
        s = sprint(show, MIME("text/plain"), rq)
        @test occursin("kmf = 15.0", s) && occursin("R1m = R1f (tied)", s)
        s = sprint(show, MIME("text/plain"), res)
        @test occursin("19 of 20 voxels (mask)", s) && occursin("medians", s)
    end

    @testset "simulation: simulate, tissue presets, phantoms, compare" begin
        te = collect(0.01:0.01:0.32)
        m = T2Decay(TE = te)
        θ = (S0 = 1000.0, T2 = 0.08)
        # shapes: one voxel, n copies; the same seed gives the same data
        @test size(simulate(m, θ; snr = 50)) == (32,)
        @test size(simulate(m, θ; snr = 50, n = 7)) == (7, 32)
        @test simulate(m, θ; snr = 50, rng = MersenneTwister(1)) == simulate(m, θ; snr = 50, rng = MersenneTwister(1))
        # snr is relative to the scale parameter (here σ = 1000/50 = 20); Gaussian SD; Rician floor
        Y = simulate(m, θ; snr = 50, n = 20_000, noise = :gaussian, rng = MersenneTwister(2))
        @test std(Y[:, 1]) ≈ 20 rtol = 0.03
        @test mean(Y[:, 1]) ≈ signal(m, θ)[1] rtol = 0.003
        Yr = simulate(m, (S0 = 1000.0, T2 = 1e-4); σ = 20, n = 20_000, rng = MersenneTwister(3))
        @test mean(Yr[:, end]) ≈ 20 * sqrt(π / 2) rtol = 0.02              # Rayleigh: no signal left
        @test EQUIP._default_noise(T1IR(TI = [0.1], magnitude = false)) == :gaussian
        @test EQUIP._default_noise(T1IR(TI = [0.1])) == :rician
        @test_throws ArgumentError simulate(m, θ)                            # no noise level
        @test_throws ArgumentError simulate(m, θ; snr = 50, σ = 1.0)         # both
        # tissue presets and truth
        @test tissue(:WM) == (T1 = 1.084, T2 = 0.069)
        @test tissue(:CSF) == (T1 = 4.124, T2 = 1.75)
        @test truth(R2Decay(TE = te), merge(tissue(:CSF), (T2 = 2.0,))).R2 == 0.5    # own value
        @test truth(T2Decay(TE = te), (T2 = 0.06,); S0 = 500.0) == (S0 = 500.0, T2 = 0.06)
        @test truth(R2Decay(TE = te), :WM) == (S0 = 1.0, R2 = 1 / 0.069)
        @test truth(T1IR(TI = [0.1]), :GM; Sf = -0.95, M0 = 1000.0) == (M0 = 1000.0, Sf = -0.95, T1 = 1.82)
        @test_throws ArgumentError truth(T1IR(TI = [0.1]), :GM)              # Sf is not a tissue value
        @test_throws ArgumentError truth(m, :brain)                          # unknown tissue
        @test truth(QMTSIR(; SIR_4POINT...), :SIR_lesion) ==
              (PSR = 0.05, R1f = 0.5, Sf = -1.0, M0f = 1.0, kmf = 20.0, Sm = 0.83)
        @test SIR_5POINT.TI == TI5 && SIR_5POINT.TD == TD5
        @test SIR_16POINT.TI == [0.010, 0.015, 0.021, 0.031, 0.045, 0.066, 0.097, 0.141, 0.206, 0.301,
                                 0.440, 0.643, 0.938, 1.370, 2.000, 10.0]
        @test SIR_16POINT.TD == fill(2.5, 16)
        m16 = QMTSIR(; SIR_16POINT...)                                      # kmf estimated
        t16 = truth(m16, :SIR_WM_II)
        r16 = fit(m16, signal(m16, t16); kmf = Free())
        @test r16.params.PSR ≈ 0.12 rtol = 1e-6
        @test r16.params.kmf ≈ 10.0 rtol = 1e-6
        # a phantom, end to end: simulate, fit, compare
        labels = labelshapes((24, 20))
        @test sort(unique(labels)) == [0, 1, 2, 3]
        tm = phantom(labels, [1 => truth(m, :GM; S0 = 1000.0), 2 => truth(m, :WM; S0 = 1000.0),
                              3 => truth(m, :blood; S0 = 1000.0)])
        @test isnan(tm.T2[1, 1]) && all(==(0.069), tm.T2[labels .== 2])
        data = simulate(m, tm; snr = 100, rng = MersenneTwister(4))
        @test all(isnan, data[1, 1, :])                                      # background
        res = fit(m, data; mask = labels .> 0)
        c = compare(res, tm; labels)
        @test [r.group for r in c.rows if r.parameter == :T2] == [:all, 1, 2, 3]
        @test all(r -> r.converged == 1, c.rows)
        @test all(r -> abs(r.relbias) < 0.01 && r.relrmse < 0.03, c.rows)
        @test occursin("rel bias", sprint(show, MIME("text/plain"), c))
        # Dortch et al. 2018, Fig. 1 (RRMS lesion, Rician noise, SNR 200 at thermal equilibrium):
        # fixing kmf at 12.5 (4-point scheme) biases PSR below its true 5%; the 5-point does not
        for (mm, kw, lo, hi) in ((QMTSIR(; SIR_5POINT...), (kmf = Free(),), 4.95, 5.2),
                                 (QMTSIR(; SIR_4POINT...), (;), 4.65, 4.85))
            t = truth(mm, :SIR_lesion)
            r = compare(fit(mm, simulate(mm, t; snr = 200, n = 2000, rng = MersenneTwister(5)); kw...), t)
            psr = only(filter(x -> x.parameter == :PSR, r.rows))
            @test lo < 100 * psr.mean < hi
            @test psr.converged == 1
        end
    end

    @testset "fit, fit! and nobs are StatsAPI's functions (no clash with StatsBase etc.)" begin
        @test nameof(parentmodule(fit)) == :StatsAPI
        @test nameof(parentmodule(fit!)) == :StatsAPI
        @test nameof(parentmodule(nobs)) == :StatsAPI
        @test nameof(parentmodule(init)) == :CommonSolve
    end
end
