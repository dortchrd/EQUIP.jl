# =================================================================================================
# test/integration/fitting.jl: Fitting datasets: Float32 flushing subnormals, QMTSIR's roles and maps, 3-D data, masks,
# threads, refit, B1 maps, any number of spatial dimensions, FitResult, the fit's own statistics.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl fitting
# =================================================================================================

@testset "Float32 dataset fits flush subnormal numbers to zero, and restore the mode" begin
    mp = FlushProbe(collect(0.01:0.01:0.08))
    Y = simulate(mp, (S0 = 1000.0, T2 = 0.05); σ = 5.0, noise = :gaussian, n = 40, rng = MersenneTwister(1))
    was = get_zero_subnormals()
    countreset!(); fit(mp, Float32.(Y); precision = Float32)
    @test NFLUSHED[] > 0 && NKEPT[] == 0 && get_zero_subnormals() == was
    countreset!(); fit(mp, Y)
    @test NKEPT[] > 0 && NFLUSHED[] == 0                     # Float64: left as it is
    countreset!(); fit(mp, Float32.(Y[1, :]); precision = Float32)
    @test NFLUSHED[] == 0                                    # one voxel: not switched per call
    # in the 5-point SIR's lesion the derivatives reach 4e-40, subnormal in Float32 (1.2e-38):
    # flushed, the fit agrees with Float64's to Float32's precision
    m5 = QMTSIR(; SIR_5POINT...)
    Yl = simulate(m5, nominal(m5, :SIR_lesion); snr = 200, noise = :rician, n = 50, rng = MersenneTwister(2))
    r64, r32 = fit(m5, Yl; kmf = Free()), fit(m5, Float32.(Yl); kmf = Free(), precision = Float32)
    ok = isconverged.(r64.status) .& isconverged.(r32.status)
    @test count(ok) >= 45
    @test maximum(abs.(r32.params.PSR[ok] .- r64.params.PSR[ok])) < 1e-3
end

@testset "QMTSIR: kmf fixed vs free, tie, per-voxel Sm map" begin
    m = QMTSIR(TI = TI5, TD = TD5)
    truth = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0, R1m = 1.0, Sm = 0.83,
             Mzf_td0 = 0.0, Mzm_td0 = 0.0)
    y = signal(m, truth)
    # kmf estimated: exact
    r = fit!(init(m; kmf = Free(start = 12.5, lower = 1, upper = 50)), y)
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
    # a map must have the data's spatial size, not just its number of voxels: a reshaped map
    # (or mask) would put values in the wrong voxels; trailing length-1 dimensions are ignored
    @test_throws DimensionMismatch fit(m, data; T2 = Free(start = reshape(T2, nx * ny, nz)))
    @test_throws DimensionMismatch fit(m, data; mask = reshape(mask, ny, nx, nz))
    d1 = data[:, :, 1:1, :]                                    # one slice: nx × ny × 1 × M
    @test size(fit(m, d1; T2 = Free(start = T2[:, :, 1])).params.T2) == (nx, ny, 1)
    @test size(fit(m, reshape(data, :, length(te)); mask = vec(mask)).params.T2) == (nx * ny * nz,)
end

@testset "refit: starts and bounds" begin
    rng = MersenneTwister(9)
    td = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
    m = T1SR(TD = td)                                          # T1's default start: 1.0
    T1 = 0.2 .+ 0.3 .* rand(rng, 10)
    Y = reduce(vcat, [permutedims(signal(m, (M0 = 1000.0, T1 = t)) .+ randn(rng, 6)) for t in T1])
    res = fit(m, Y; T1 = Free(lower = 0.05))
    start(r; kw...) = refit(r, Y; solver = LM(maxiter = 0), kw...).params.T1   # the start used
    @test start(res) == res.params.T1                          # previous values
    @test start(res; T1 = Free(upper = 5.0)) == res.params.T1  # a bounds-only change keeps them
    @test start(res; T1 = Free(start = 0.7)) == fill(0.7, 10)  # an explicit start wins
    @test start(res; T1 = Free(upper = 0.25)) == min.(res.params.T1, 0.25)   # moved onto a new bound
    r2 = refit(res, Y; T1 = Free(upper = 5.0))
    @test r2.settings.T1.lower == 0.05 && r2.settings.T1.upper == 5.0   # stored bound kept
    @test r2.params.T1 ≈ res.params.T1 rtol = 1e-6
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
    @test with.fixed.B1 === B1                 # a dataset result keeps the map (not copied)
    # one voxel fitted with the map: the result keeps that voxel's B1, so it is self-contained
    r7 = fit!(init(m; B1 = B1), data[7, 1, :]; voxel = 7)
    @test r7.fixed.B1 == B1[7]
    @test fitted(r7) ≈ data[7, 1, :] rtol = 1e-6
    @test resolve(r7) == (M0 = r7.params.M0, T1 = r7.params.T1, B1 = B1[7])
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
    yq = signal(mq, merge(defaults(mq), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0)))
    rq = fit(mq, yq; kmf = 15.0)
    @test fitted(rq) ≈ yq rtol = 1e-6
    @test rq.fixed == (kmf = 15.0, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)
    rr = resolve(rq)                                   # every value, fitted and fixed, ties evaluated
    @test keys(rr) == keys(parameters(mq)) && rr.kmf == 15.0 && rr.R1m == rr.R1f == rq.params.R1f
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
    @test resolve(res, 3, 2) == (S0 = res.params.S0[3, 2], T2 = res.params.T2[3, 2])
    @test_throws ArgumentError resolve(res)
    @test_throws ArgumentError resolve(r, 1)
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

@testset "fit statistics: stderror, vcov, cor, residualsd, AIC/BIC, onbound" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    te = collect(0.01:0.01:0.32)
    m = T2Decay(TE = te)
    θ = (S0 = 1000.0, T2 = 0.08)
    y = simulate(m, θ; σ = 10.0, noise = :gaussian, rng = MersenneTwister(1))
    r = fit(m, y)
    # the counts: N, the free parameters and σ, N − P, the SSE
    @test nobs(r) == 32 && dof(r) == 3 && dof_residual(r) == 30 && rss(r) == r.sse
    @test residualsd(r) ≈ sqrt(r.sse / 30) rtol = 1e-14
    # the covariance σ̂²(JᵀJ)⁻¹ at the fit's answer; with σ given, crlb's at the fit's values
    J = jacobian(r)
    C = residualsd(r)^2 * inv(J' * J)
    @test vcov(r) ≈ C rtol = 1e-10
    @test vcov(r) == vcov(r)'
    @test collect(values(stderror(r))) ≈ sqrt.(diag(C)) rtol = 1e-10
    @test keys(stderror(r)) == keys(r.params)
    s10, c10 = stderror(r; σ = 10.0), crlb(m, merge(θ, r.params); σ = 10.0).sd
    @test all(p -> s10[p] ≈ c10[p], keys(s10))
    R = cor(r)
    @test R ≈ C ./ sqrt.(diag(C) * diag(C)') rtol = 1e-10
    @test R[1, 1] == R[2, 2] == 1
    # calibrated: over noisy copies, the standard errors match the spread of the estimates
    Y = simulate(m, θ; σ = 10.0, noise = :gaussian, n = 2000, rng = MersenneTwister(2))
    res = fit(m, Y)
    se = stderror(res)
    @test se.T2 isa Vector{Float64} && length(se.T2) == 2000
    @test median(se.T2) ≈ std(res.params.T2) rtol = 0.05
    @test median(se.S0) ≈ std(res.params.S0) rtol = 0.05
    @test se.T2[7] == stderror(fit(m, Y[7, :])).T2                      # each voxel its own fit
    @test residualsd(res) ≈ sqrt.(res.sse ./ 30)
    @test median(residualsd(res)) ≈ 10 rtol = 0.05
    @test size(vcov(res)) == (2000,)
    @test vcov(res)[7] ≈ vcov(fit(m, Y[7, :])) rtol = 1e-12
    # a parameter on a bound: flagged, its SD NaN, the others with it held
    rb = fit(m, y; S0 = Free(lower = 0, upper = 900.0))
    @test rb.params.S0 == 900 && onbound(rb) == (S0 = true, T2 = false)
    @test isnan(stderror(rb).S0) && isfinite(stderror(rb).T2)
    Jt = jacobian(rb)[:, 2:2]
    @test stderror(rb).T2 ≈ residualsd(rb) * sqrt(inv(Jt' * Jt)[1]) rtol = 1e-10
    @test cor(rb)[1, 1] == 1 && isnan(cor(rb)[1, 2])
    @test onbound(fit(m, Y; S0 = Free(lower = 0, upper = 990.0))).S0 isa BitVector
    # N = P (4-point SIR, four free parameters): no residual estimate of σ; with σ given,
    # crlb's at the fit's values and the fit's assumption
    m4 = QMTSIR(; SIR_4POINT...)
    θ4 = nominal(m4, :SIR_WM_I)
    r4 = fit(m4, simulate(m4, θ4; noise = :rician, snr = 200, rng = MersenneTwister(3)))
    @test dof_residual(r4) == 0 && isnan(residualsd(r4)) && all(isnan, values(stderror(r4)))
    s4 = stderror(r4; σ = 0.005)
    c4 = crlb(m4, merge(θ4, r4.params, (kmf = 12.5,)); σ = 0.005)
    @test all(p -> s4[p] ≈ c4.sd[p], keys(s4))
    # the information criteria where the residuals say nothing: all four NaN at N = P; AICc
    # NaN at N ≤ k + 1 (k = P + 1: its correction undefined, or negative below), finite
    # from N = k + 2 (T2: P = 2, k = 3)
    @test all(isnan, (loglikelihood(r4), aic(r4), aicc(r4), bic(r4)))
    for (n, defined) in ((3, false), (4, false), (5, true))
        rn = fit(T2Decay(TE = te[1:n]), y[1:n])
        @test isfinite(aic(rn)) && isfinite(bic(rn)) && isfinite(loglikelihood(rn))
        @test isfinite(aicc(rn)) == defined
    end
    r5 = fit(T2Decay(TE = te[1:5]), y[1:5])
    @test aicc(r5) ≈ aic(r5) + 2 * 3 * 4 / (5 - 3 - 1)
    @test all(isnan, aicc(fit(T2Decay(TE = te[1:4]), Y[:, 1:4])))
    # maps: fixed values per voxel (B1), a mask (unfitted voxels NaN), σ as a map
    mf = T1MFA(FA = [3.0, 6, 10, 15, 20], TR = 0.015)
    b1 = [0.8 + 0.4 * (i + j) / 20 for i in 1:6, j in 1:5]
    data = simulate(mf, (M0 = fill(1000.0, 6, 5), T1 = fill(1.2, 6, 5), B1 = b1); σ = 5.0,
                    noise = :gaussian, rng = MersenneTwister(4))
    mask = trues(6, 5); mask[1, 1] = false
    rm = fit(mf, data; B1 = b1, mask)
    sm = stderror(rm)
    one = fit!(init(mf; B1 = b1), data[3, 4, :]; voxel = LinearIndices((6, 5))[3, 4])
    @test size(sm.T1) == (6, 5) && isnan(sm.T1[1, 1]) && sm.T1[3, 4] == stderror(one).T1
    @test stderror(rm; σ = fill(5.0, 6, 5)).T1[3, 4] == stderror(rm; σ = 5.0).T1[3, 4]
    @test occursin("`σ` is a map of size", msg(() -> stderror(rm; σ = fill(5.0, 2, 2))))
    # a Float32 fit: the statistics in Float64, as the Float64 fit's to its rounding
    @test collect(values(stderror(fit(m, y; precision = Float32)))) ≈ collect(values(stderror(r))) rtol = 1e-4
    # AIC/BIC: the information criteria pick the model that made the data
    mb = T2BiexpDecay(TE = te)
    yb = simulate(mb, (S0 = 1000.0, fshort = 0.3, T2short = 0.015, T2long = 0.09); σ = 5.0,
                  noise = :gaussian, rng = MersenneTwister(5))
    ym = simulate(m, θ; σ = 5.0, noise = :gaussian, rng = MersenneTwister(6))
    @test bic(fit(mb, yb)) < bic(fit(m, yb)) && aic(fit(mb, yb)) < aic(fit(m, yb))
    @test bic(fit(m, ym)) < bic(fit(mb, ym))
    @test loglikelihood(r) ≈ -16 * (log(2π * r.sse / 32) + 1) rtol = 1e-14
    @test aic(r) ≈ -2loglikelihood(r) + 6 && bic(r) ≈ -2loglikelihood(r) + 3log(32)
    @test aicc(r) ≈ aic(r) + 2 * 3 * 4 / (32 - 3 - 1)
    @test bic(res) isa Vector{Float64} && bic(res)[7] == bic(fit(m, Y[7, :]))
end
