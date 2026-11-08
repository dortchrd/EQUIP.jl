# =================================================================================================
# test/integration/diffusion.jl: The diffusion models: SMT and its variants (Kaden 2016; Devan et al., MRI 2020), directions
# per image (diffusionencoding, fiberdirection), DTI, DKI (against DIPY's forms).
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl diffusion
# =================================================================================================

@testset "SMT and its variants: the spherical mean technique (Kaden 2016; Devan et al., MRI 2020)" begin
    rng = MersenneTwister(21)
    # The mean over directions of exp(−a·ξ²), ∫₀¹ exp(−a·ξ²) dξ (ξ = cos ψ: uniform over the
    # sphere), what the spherical means are made of: its series near 0, erf for a > 0 and
    # erfi for a < 0 (a zeppelin with D⊥ > D∥), the same (values and derivatives) on both
    # sides of each switch
    f(a) = sqrt(π) * SpecialFunctions.erf(sqrt(a)) / (2 * sqrt(a))
    fneg(a) = sqrt(π) * SpecialFunctions.erfi(sqrt(-a)) / (2 * sqrt(-a))
    for a in (1e-3, 0.0499999, 0.0500001, 0.7, 12.0)
        @test EQUIP._meanexp(a) ≈ f(a) rtol = 1e-11
        @test ForwardDiff.derivative(EQUIP._meanexp, a) ≈ ForwardDiff.derivative(f, a) rtol = 1e-8
        @test EQUIP._meanexp(-a) ≈ fneg(-a) rtol = 1e-11
        @test ForwardDiff.derivative(EQUIP._meanexp, -a) ≈ ForwardDiff.derivative(fneg, -a) rtol = 1e-8
    end
    @test EQUIP._meanexp(0.0) == 1 && ForwardDiff.derivative(EQUIP._meanexp, 0.0) ≈ -1 / 3
    @test EQUIP._meanexp(-3f0) isa Float32 && EQUIP._meanexp(3f0) isa Float32
    u = range(0, 1; length = 200_001)                         # the trapezoid rule over cos ψ
    trap(g) = (sum(g.(u)) - (g(0) + g(1)) / 2) * step(u)
    b, D, Dperp = 2000.0, 2.0e-3, 0.7e-3
    @test stick(b, D, powder) ≈ trap(c -> exp(-b * D * c^2)) rtol = 1e-9                         # stick
    @test zeppelin(b, D, Dperp, powder) ≈ trap(c -> exp(-b * (Dperp + (D - Dperp) * c^2))) rtol = 1e-9  # zeppelin
    @test zeppelin(b, Dperp, D, powder) ≈ trap(c -> exp(-b * (D + (Dperp - D) * c^2))) rtol = 1e-9  # D⊥ > D∥
    # The model: S0 at b = 0; free diffusion when v_ax = 0 (D⊥ = D_ax); sticks alone when 1
    m = SMT(b = [0.0, 1000.0, 3000.0])
    @test signal(m, (S0 = 2.0, vax = 0.4, Dax = 2e-3))[1] == 2.0
    @test signal(m, (S0 = 1.0, vax = 0.0, Dax = 2e-3)) ≈ exp.(-m.b .* 2e-3)
    @test signal(m, (S0 = 1.0, vax = 1.0, Dax = 2e-3)) ≈ [1.0; f.(m.b[2:3] .* 2e-3)]
    @test nominal(m, :SMT_cBH).vax == 0.241 && nominal(m, :SMT_NWM).Dax == 1.9e-3

    # Devan 2020, Fig. 1: the vax precision efficiency 1/√(N·CRLB/vax²) of the five
    # protocols, for an isotropic fiber arrangement (their worst case, where each direction's
    # signal is the spherical mean). At SNR 20 per direction (their simulations' SNR; Fig. 1
    # does not say), with the shells' means weighted by their directions, it is their figure:
    # within 5% for normal white matter, 15% for the others. Theirs adds the Rician factor,
    # which lowers the precision most where the signal is lowest (protocols A and B's
    # b = 3790–4230 shells, in black holes, the lowest vax): ours is higher there.
    fig1 = (SMT_NWM = (A = 0.90, B = 1.00, C = 1.10, D = 0.865, E = 0.84),     # read off the plot
            SMT_NAWM = (A = 1.10, B = 1.06, C = 1.05, D = 0.83, E = 0.82),
            SMT_cBH = (A = 0.90, B = 0.905, C = 0.825, D = 0.685, E = 0.68),
            SMT_T2lesion = (A = 1.11, B = 1.06, C = 0.99, D = 0.79, E = 0.79))
    for name in keys(fig1), P in keys(SMT_PROTOCOLS)
        pr = SMT_PROTOCOLS[P]
        ms = SMT(b = pr.b)
        θ = nominal(ms, name)
        c = crlb(ms, θ; snr = 20, averages = pr.directions, weights = pr.directions)
        eff = θ.vax / (c.sd.vax * sqrt(sum(pr.directions)))
        @test eff ≈ fig1[name][P] rtol = (name === :SMT_NWM ? 0.06 : 0.15)
    end

    # Fits from the default starts: unbiased, and as precise as crlb says (Gaussian, SNR 20,
    # chronic black hole, protocol A), in both precisions
    pr = SMT_PROTOCOLS.A
    ms = SMT(b = pr.b)
    θ = nominal(ms, :SMT_cBH)
    Y = simulate(ms, θ; snr = 20, averages = pr.directions, noise = :gaussian, n = 2000, rng)
    c = crlb(ms, θ; snr = 20, averages = pr.directions, weights = pr.directions)
    for T in (Float64, Float32)
        r = fit(ms, T.(Y); weights = pr.directions, precision = T)
        @test all(isconverged, r.status) && eltype(r.params.vax) == T
        @test abs(mean(r.params.vax) - θ.vax) < 4 * c.sd.vax / sqrt(2000)
        @test std(r.params.vax) ≈ c.sd.vax rtol = 0.1
        @test std(r.params.Dax) ≈ c.sd.Dax rtol = 0.1
    end

    # sphericalmean: the volumes of each shell averaged (b-values within 100 s/mm² merged),
    # observations last, any shape; the shells' b and their numbers of volumes
    bvals = [0, 1000, 5, 995, 3000, 1010, 2990, 0]
    data = reshape(Float64.(1:2*3*8), 2, 3, 8)
    means, shells, dirs = sphericalmean(data, bvals)
    @test size(means) == (2, 3, 3) && dirs == [3, 3, 2]
    @test shells ≈ [5 / 3, (995 + 1000 + 1010) / 3, 2995]
    @test means[1, 1, :] ≈ [mean(data[1, 1, [1, 3, 8]]), mean(data[1, 1, [2, 4, 6]]), mean(data[1, 1, [5, 7]])]
    @test sphericalmean(data, bvals; tol = 10)[3] == [3, 2, 1, 2]          # 1010 more than 10 above 995
    @test sphericalmean(collect(1.0:4.0), [0, 0, 1000, 1000])[1] == [1.5, 3.5]
    @test_throws DimensionMismatch sphericalmean(data, bvals[1:7])

    # SMTFreeWater: SMT when fiso = 0, free diffusion when 1. SMTT2: SMT times exp(−TE/T2)
    # when the T2s are equal; at one TE with different T2s, SMT's fraction is T2-weighted
    # (0.664 here, the T2-weighted signal fraction 0.625, the volume fraction 0.45)
    bs = [0.0, 500, 1000, 2000, 3000, 4500]
    t3 = (S0 = 1.0, vax = 0.45, Dax = 1.8e-3)
    @test signal(SMTFreeWater(b = bs), (; t3..., fiso = 0.0, Diso = 3e-3)) ≈ signal(SMT(b = bs), t3)
    @test signal(SMTFreeWater(b = bs), (; t3..., fiso = 1.0, Diso = 3e-3)) ≈ exp.(-bs .* 3e-3)
    bt, tt = repeat([0.0, 1000, 2500, 4500], 4), repeat([0.06, 0.1, 0.15, 0.2]; inner = 4)
    mt = SMTT2(b = bt, TE = tt)
    @test signal(mt, (; t3..., T2ax = 0.07, T2ex = 0.07)) ≈ signal(SMT(b = bt), t3) .* exp.(-tt ./ 0.07)
    m1 = SMTT2(b = [0.0, 1000, 2500, 4000], TE = fill(0.08, 4))
    y1 = signal(m1, (; t3..., T2ax = 0.09, T2ex = 0.05))
    @test fit(SMT(b = m1.b), y1).params.vax ≈ 0.6642 atol = 1e-3
    # Fits at SNR 500 per shell mean: as precise as crlb says, and their bias small next to
    # it (both variants; the bias of a nonlinear fit from the noise itself, measured at up to
    # 10% of the SD with 20,000 copies: crlb's `bias` is that of wrong fixed values, 0 here)
    for (mv, tv) in ((SMTFreeWater(b = bs), (; t3..., fiso = 0.2, Diso = 3e-3)),
                     (mt, (; t3..., T2ax = 0.09, T2ex = 0.05)))
        Yv = simulate(mv, tv; snr = 500, noise = :gaussian, n = 2000, rng)
        rv, cv = fit(mv, Yv), crlb(mv, tv; snr = 500)
        @test all(isconverged, rv.status)
        for k in keys(rv.params)
            k === :S0 && continue
            @test abs(mean(rv.params[k]) - tv[k]) < 0.2 * cv.sd[k]
            @test std(rv.params[k]) ≈ cv.sd[k] rtol = 0.1
        end
    end
    # sphericalmean with echo times: one shell per (TE, b), by TE then b
    tes = [0.08, 0.08, 0.08, 0.08, 0.12, 0.12, 0.1205, 0.12]
    bv = [0, 1000, 1005, 0, 0, 1000, 990, 2000]
    sm = sphericalmean(collect(1.0:8.0), bv; TE = tes)
    @test isapprox(sm.TE, [0.08, 0.08, 0.12, 0.12, 0.12]; atol = 1e-3)   # (a shell's mean TE)
    @test sm.b ≈ [0, 1002.5, 0, 995, 2000]
    @test sm.directions == [2, 2, 1, 2, 1] && sm.means ≈ [2.5, 2.5, 5, 6.5, 8]
    @test sphericalmean(collect(1.0:8.0), bv).TE === nothing
    @test_throws DimensionMismatch sphericalmean(collect(1.0:8.0), bv; TE = tes[1:7])
end

@testset "Directions per image: diffusionencoding, fiberdirection, g ⋅ n" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    rng = MersenneTwister(5)
    # direction sets: unit vectors with z ≥ 0, the same each time, evenly spread (second
    # moment I/3, which averages order 2 exactly)
    dirs = directionset(30)
    @test length(dirs) == 30 && all(g -> g[3] >= 0 && norm(g) ≈ 1, dirs) && dirs == directionset(30)
    @test norm(sum(g * g' for g in dirs) / 30 - I / 3) < 1e-6
    @test length(directionset(1)) == 1
    G = reduce(hcat, [fill([0.0, 0.0, 0.0], 4); collect.(dirs); collect.(dirs)])   # 3×64, FSL: 0 0 0 at b = 0
    bv = [fill(0.0, 2); fill(5.0, 2); fill(1000.0, 30); fill(2500.0, 30)]
    acq = diffusionencoding(bv, G)
    @test keys(acq) == (:b, :gx, :gy, :gz) && all(v -> v isa Vector{Float64}, values(acq))
    @test acq.b == bv && (acq.gx, acq.gy, acq.gz) == (G[1, :], G[2, :], G[3, :])   # b kept as given
    @test diffusionencoding(bv, permutedims(G)) == acq                           # N×3 too
    @test diffusionencoding(1:3, [1 0 0; 0 1 0; 0 0 1]).gy == [0, 1, 0]          # 3×3: a .bvec
    Gc = copy(G); a2 = diffusionencoding(bv, Gc); Gc[1, 5] = 9.0
    @test a2.gx[5] == G[1, 5]                                                    # copied
    # mistakes, said with what is wrong
    @test occursin("one direction per image", msg(() -> diffusionencoding(bv[1:63], G)))
    @test occursin("3×N or N×3", msg(() -> diffusionencoding(bv, [G; G[1:1, :]])))
    @test occursin("image 5", msg(() -> diffusionencoding(bv, 2 .* G))) &&
          occursin("unit vectors", msg(() -> diffusionencoding(bv, 2 .* G)))
    @test diffusionencoding(bv, 2 .* G; b0threshold = 3000).gx == 2 .* G[1, :]   # (checked above it)
    # from the files, as dcm2niix writes them
    dir = mktempdir()
    write(joinpath(dir, "dwi.bval"), join(bv, " ") * "\n")
    write(joinpath(dir, "dwi.bvec"), join((join(G[k, :], " ") for k in 1:3), "\n") * "\n")
    @test diffusionencoding(joinpath(dir, "dwi.bval"), joinpath(dir, "dwi.bvec")) == acq
    # the directional view: a fiber's direction, and its cosine with a gradient direction
    n = fiberdirection(0.7, 0.3)
    @test n isa NTuple{3,Float64} && sum(abs2, n) ≈ 1 && fiberdirection(0.0, 1.2) == (0.0, 0.0, 1.0)
    # the cosine is Julia's dot, exported by EQUIP too (the same function: no clash with LinearAlgebra)
    @test EQUIP.:⋅ === LinearAlgebra.:⋅ && :⋅ in names(EQUIP)
    @test Tuple(dirs[3]) ⋅ n ≈ dot(dirs[3], SVector(n))                     # two tuples
    @test dirs[3] ⋅ n == Tuple(dirs[3]) ⋅ n                                  # or an SVector
    @test (0.0, 0.0, 1.0) ⋅ fiberdirection(π / 3, 0.4) ≈ 0.5
    tdot(g, n) = @allocated g ⋅ n
    @test (tdot((1.0, 2.0, 3.0), n); tdot((1.0, 2.0, 3.0), n) == 0)
    # a directional model on it: its signal is the stick and zeppelin written out, no allocation
    m = DirStickZeppelin(; acq...)
    θ = (S0 = 1000.0, vax = 0.6, Dax = 1.9e-3, θ = 0.7, φ = 0.3)
    ξ = [dot(SVector(acq.gx[j], acq.gy[j], acq.gz[j]), SVector(n)) for j in 1:64]
    Dp = 0.4 * 1.9e-3
    @test signal(m, θ) ≈ (@. 1000 * (0.6 * exp(-bv * 1.9e-3 * ξ^2) +
                                    0.4 * exp(-bv * Dp) * exp(-bv * (1.9e-3 - Dp) * ξ^2))) rtol = 1e-12
    out = zeros(64); signal!(out, m, θ)
    allocs(m, out, θ) = @allocated signal!(out, m, θ)
    @test_noalloc allocs(m, out, θ) == 0
    # fitted in both precisions, from the true direction's neighborhood
    Y = simulate(m, θ; snr = 100, n = 200, noise = :gaussian, rng)
    for T in (Float64, Float32)
        r = fit(m, Y; precision = T, θ = Free(start = 0.6), φ = Free(start = 0.2))
        @test eltype(r.params.vax) == T && count(isconverged, r.status) == 200
        @test abs(mean(r.params.vax) - 0.6) < 0.01 && abs(mean(r.params.θ) - 0.7) < 0.01
    end
    # one 3-vector per image is refused where every fit and signal starts, naming the remedy
    mv = DirVectors(b = bv, g = [SVector(c...) for c in eachcol(G)])
    for f in (() -> signal(mv, (S0 = 1.0,)), () -> fit(mv, Y), () -> nobs(mv))
        @test occursin("the field `g` holds", msg(f)) && occursin("diffusionencoding", msg(f))
    end
end

@testset "DTI: the diffusion tensor, written out, from the log-linear start" begin
    rng = MersenneTwister(8)
    dirs = directionset(30)
    bv = [fill(0.0, 6); fill(1000.0, 30)]
    G = reduce(hcat, [fill([0.0, 0.0, 0.0], 6); collect.(dirs)])       # 3×36, as a .bvec
    m = DTI(; diffusionencoding(bv, G)...)
    @test nobs(m) == 36 && keys(parameters(m)) == (:S0, :Dxx, :Dyy, :Dzz, :Dxy, :Dxz, :Dyz)
    # a rotated prolate tensor (eigenvalues 1.7, 0.4, 0.3e-3 mm²/s): S0·exp(−b·gᵀDg), no allocation
    Q = Matrix(qr(randn(rng, 3, 3)).Q)
    D = Q * Diagonal([1.7e-3, 0.4e-3, 0.3e-3]) * Q'
    θ = (S0 = 1000.0, Dxx = D[1, 1], Dyy = D[2, 2], Dzz = D[3, 3], Dxy = D[1, 2], Dxz = D[1, 3], Dyz = D[2, 3])
    @test signal(m, θ) ≈ [1000 * exp(-bv[j] * (G[:, j]' * D * G[:, j])) for j in 1:36] rtol = 1e-12
    out = zeros(36)
    allocs(m, out, θ) = @allocated signal!(out, m, θ)
    @test_noalloc (allocs(m, out, θ); allocs(m, out, θ) == 0)
    # the start, `:loglinear` (the weighted log-linear fit, from the signal): the tensor itself
    # from noise-free data, in both precisions; no allocation in a voxel's fit
    y = signal(m, θ)
    start(m, y; kw...) = fit(m, y; solver = LM(maxiter = 0), kw...).params
    for (T, tol) in ((Float64, 1e-12), (Float32, 1e-5))
        s = start(m, T.(y); precision = T)
        @test eltype(s.Dxx) == T && all(k -> abs(s[k] - θ[k]) < tol * 1.7e-3, keys(θ)[2:7])
    end
    c = init(m)
    allocfit(c, y) = @allocated fit!(c, y)
    @test_noalloc (allocfit(c, y); allocfit(c, y) == 0)
    # on noisy data: the two passes written out, OLS then WLS weighted by OLS's predicted
    # signal (DIPY's WLS); with the off-diagonal elements fixed, the same on the diagonal's
    # columns; with a tie (Dzz = Dyy), on the two columns merged; with Dxx started from a map,
    # Dxx held at the map's values and the others fitted to ln y − Xₓₓ·Dxx, the weights from
    # the whole prediction
    X = [ones(36) -bv .* G[1, :] .^ 2 -bv .* G[2, :] .^ 2 -bv .* G[3, :] .^ 2 -2bv .* G[1, :] .* G[2, :] -2bv .* G[1, :] .* G[3, :] -2bv .* G[2, :] .* G[3, :]]
    twopass(X, l, off = 0) = (u = X \ l; w = exp.(X * u .+ off); (w .* X) \ (w .* l))
    yn = simulate(m, θ; snr = 30, n = 1, noise = :rician, rng = MersenneTwister(21))[1, :]
    tol = 1e-12 * 1.7e-3
    u = twopass(X, log.(yn))
    sn = start(m, yn)
    @test all(k -> abs(sn[k] - u[k]) < tol, 2:7) && keys(sn)[2:7] == (:Dxx, :Dyy, :Dzz, :Dxy, :Dxz, :Dyz)
    sf, uf = start(m, yn; Dxy = 0.0, Dxz = 0.0, Dyz = 0.0), twopass(X[:, 1:4], log.(yn))
    @test keys(sf) == (:S0, :Dxx, :Dyy, :Dzz) && all(k -> abs(sf[k] - uf[k]) < tol, 2:4)
    st, ut = start(m, yn; Dzz = Fixed(:Dyy)), twopass([X[:, 1:2] X[:, 3] .+ X[:, 4] X[:, 5:7]], log.(yn))
    @test all(k -> abs(st[k] - ut[k]) < tol, 2:6) && keys(st)[2:6] == (:Dxx, :Dyy, :Dxy, :Dxz, :Dyz)
    Yn = simulate(m, θ; snr = 30, n = 3, noise = :rician, rng = MersenneTwister(22))
    dxx = [1.5e-3, 1.6e-3, 1.7e-3]
    sm = fit(m, Yn; solver = LM(maxiter = 0), Dxx = Free(start = dxx)).params
    @test sm.Dxx == dxx
    for i in 1:3
        ui = twopass(X[:, [1, 3, 4, 5, 6, 7]], log.(Yn[i, :]) .- X[:, 2] .* dxx[i], X[:, 2] .* dxx[i])
        @test all(k -> abs(sm[k][i] - ui[k - 1]) < tol, 3:7)
    end
    # ... where the data cannot give all six: directions in one plane give the in-plane
    # elements (the others, unmeasured, stay at 0); one shell with no b = 0 gives none
    inplane = [[cos(a), sin(a), 0.0] for a in range(0, π; length = 13)[1:12]]
    mp = DTI(; diffusionencoding([0.0; fill(1000.0, 12)], reduce(hcat, [[[0.0, 0.0, 0.0]]; inplane]))...)
    sp = start(mp, signal(mp, θ))
    @test all(k -> abs(sp[k] - θ[k]) < 1e-12 * 1.7e-3, (:Dxx, :Dyy, :Dxy)) && sp.Dzz == sp.Dxz == sp.Dyz == 0
    m1 = DTI(; diffusionencoding(fill(1000.0, 30), reduce(hcat, collect.(dirs)))...)
    s1 = start(m1, signal(m1, θ))
    @test all(k -> s1[k] == 0, keys(θ)[2:7]) && s1.S0 > 0
    @test all(k -> start(m, -y)[k] == 0, keys(θ)[2:7])                  # no positive data
    # what is read off the tensor (`derived`): MD, FA, AD and RD against the eigenvalues; no
    # allocation; differentiable with dual numbers (FA's gradient against finite differences)
    λ, Ev = eigen(Symmetric(D))
    dv = derived(m, θ)
    @test keys(dv) == (:MD, :FA, :AD, :RD, :V1x, :V1y, :V1z)
    # the principal eigenvector, in the upper hemisphere (z ≥ 0): D·e1 = λ1·e1, unit length
    e1 = [dv.V1x, dv.V1y, dv.V1z]
    @test D * e1 ≈ λ[3] * e1 rtol = 1e-10
    @test norm(e1) ≈ 1 && dv.V1z >= 0 && abs(dot(e1, Ev[:, 3])) ≈ 1
    @test EQUIP._upperhemisphere(SVector(0.6, 0.0, -0.8)) == SVector(-0.6, -0.0, 0.8)
    @test EQUIP._upperhemisphere(SVector(0.6, -0.8, 0.0)) == SVector(-0.6, 0.8, -0.0)
    @test dv.MD ≈ mean(λ) rtol = 1e-12
    @test dv.FA ≈ sqrt(1.5 * sum((λ .- mean(λ)) .^ 2) / sum(λ .^ 2)) rtol = 1e-12
    @test dv.AD ≈ λ[3] rtol = 1e-12
    @test dv.RD ≈ (λ[1] + λ[2]) / 2 rtol = 1e-12
    allocd(m, θ) = @allocated derived(m, θ)
    @test_noalloc (allocd(m, θ); allocd(m, θ) == 0)
    names6 = (:Dxx, :Dyy, :Dzz, :Dxy, :Dxz, :Dyz)
    faof(x) = derived(m, merge(θ, NamedTuple{names6}(Tuple(x)))).FA
    x0 = [θ[k] for k in names6]
    gfa = ForwardDiff.gradient(faof, x0)
    h = 1e-9
    gfd = [(faof(x0 .+ h .* (1:6 .== k)) - faof(x0 .- h .* (1:6 .== k))) / 2h for k in 1:6]
    @test gfa ≈ gfd rtol = 1e-5
    # for a fit: numbers for one voxel; for a dataset, maps (NaN outside the mask), each voxel's
    # its own parameters'; nothing for a model that derives nothing
    r1 = fit(m, y)
    @test derived(r1) == derived(m, resolve(r1))
    Yd = simulate(m, θ; snr = 50, n = 20, noise = :rician, rng = MersenneTwister(23))
    maskd = [isodd(i) for i in 1:20]
    rd = fit(m, Yd; mask = maskd)
    dd = derived(rd)
    @test keys(dd) == (:MD, :FA, :AD, :RD, :V1x, :V1y, :V1z) && size(dd.FA) == (20,) && eltype(dd.FA) == Float64
    @test all(isnan, dd.FA[.!maskd]) && all(i -> dd.FA[i] == derived(m, resolve(rd, i)).FA, findall(maskd))
    @test eltype(derived(fit(m, Yd; precision = Float32)).MD) == Float32
    @test derived(fit(T2Decay(TE = [0.01, 0.02, 0.04]), [900.0, 800.0, 650.0])) === nothing
    # fitted in both precisions: the truth on average, the SD that crlb predicts, and the same
    # fits as from a constant start
    Y = simulate(m, θ; snr = 50, n = 300, noise = :gaussian, rng)
    c = crlb(m, θ; snr = 50)
    const0 = (Dxx = Free(start = 1e-3), Dyy = Free(start = 1e-3), Dzz = Free(start = 1e-3),
              Dxy = Free(start = 0.0), Dxz = Free(start = 0.0), Dyz = Free(start = 0.0))
    for T in (Float64, Float32)
        r = fit(m, Y; precision = T)
        @test eltype(r.params.Dxx) == T && count(isconverged, r.status) == 300
        for k in keys(const0)
            @test abs(mean(r.params[k]) - θ[k]) < 4 * c.sd[k] / sqrt(300)
            @test std(r.params[k]) ≈ c.sd[k] rtol = 0.15
        end
        rc = fit(m, Y; precision = T, const0...)
        @test all(k -> maximum(abs.(r.params[k] .- rc.params[k])) < (T == Float64 ? 1e-6 : 1e-3) * 1.7e-3, keys(const0))
    end
end

@testset "DKI: the start's two passes, Float32 at 22 parameters, and MK, AK, RK as DIPY's" begin
    dirs = directionset(30)
    bv = [fill(0.0, 6); fill(1000.0, 30); fill(2000.0, 30)]
    G = reduce(hcat, [fill([0.0, 0.0, 0.0], 6); collect.(dirs); collect.(dirs)])
    m = DKI(; diffusionencoding(bv, G)...)
    VN = (:V1111, :V2222, :V3333, :V1112, :V1113, :V1222, :V2223, :V1333, :V2333,
          :V1122, :V1133, :V2233, :V1123, :V1223, :V1233)
    MD2 = (0.8e-3)^2                     # D's eigenvalues 1.7, 0.4, 0.3e-3; isotropic K = 1
    θ = merge((S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0),
              NamedTuple{VN}((MD2, MD2, MD2, 0, 0, 0, 0, 0, 0, MD2 / 3, MD2 / 3, MD2 / 3, 0, 0, 0)))
    Y = simulate(m, θ; snr = 100, n = 200, noise = :gaussian, rng = MersenneTwister(31))
    # the start: OLS then WLS on the design matrix of DIPY's dki_design_matrix, ln S0 first
    X = zeros(66, 22)
    for j in 1:66
        g1, g2, g3 = G[:, j]
        X[j, 1] = 1
        X[j, 2:7] = -bv[j] .* [g1^2, g2^2, g3^2, 2g1 * g2, 2g1 * g3, 2g2 * g3]
        X[j, 8:22] = bv[j]^2 .* EQUIP._kurtosisweights(g1, g2, g3) ./ 6
    end
    y = Y[1, :]
    u = (X \ log.(y)); w = exp.(X * u); u = (w .* X) \ (w .* log.(y))
    s = fit(m, y; solver = LM(maxiter = 0)).params
    @test all(k -> abs(s[k] - u[k]) < 1e-10 * abs(u[k]) + 1e-12 * 1.7e-3, 2:7)
    @test all(k -> abs(s[k] - u[k]) < 1e-10 * abs(u[k]) + 1e-12 * MD2, 8:22)
    # Float32 against Float64: every voxel converged in both; within a hundredth of an SD
    r64, r32 = fit(m, Y), fit(m, Float32.(Y); precision = Float32)
    # the dataset loop calls the voxel fit as a function above 8 free parameters (dataset.jl):
    # the same fits, bit for bit, as `fit!` voxel by voxel
    @test EQUIP._FIT_INLINE_MAXP < 22
    cv = init(m)
    @test all(i -> fit!(cv, Y[i, :]).params.V1111 === r64.params.V1111[i] &&
                   fit!(cv, Y[i, :]).params.Dxx === r64.params.Dxx[i], 1:20)
    @test count(isconverged, r64.status) == 200 && count(isconverged, r32.status) == 200
    @test eltype(r32.params.V1111) == Float32
    for k in (:Dxx, :Dxy, :V1111, :V1122, :V1123)
        @test maximum(abs.(r32.params[k] .- r64.params[k])) < 0.01 * std(r64.params[k])
    end
    # The indices. DIPY 1.12.1's own values, from its fit methods' defaults (analytic; K
    # clipped to [−3/7, 10]; 0 where an eigenvalue is not positive), for tensors reaching each
    # branch: D's six elements in EQUIP's order, W's fifteen in DIPY's (= V's / MD²). MK to
    # 1e-4 (DIPY's Carlson tolerances leave it up to 4e-5 off; EQUIP's are tighter), the
    # others to 1e-10; for an oblate tensor (λ1 = λ2) the principal direction is undefined,
    # so only MK and MKT are compared.
    ref = [
        ((1.030382074422773e-3, 3.6535853537401707e-4, 1.0042593902032104e-3, -1.6421296987182451e-4, -6.304008115368565e-4, 2.0922097553227018e-4),
         (1.3, 0.8, 1.1, 0.05, -0.04, 0.03, 0.02, -0.06, 0.04, 0.3833333333333333, 0.30333333333333334, 0.35333333333333333, 0.01, -0.02, 0.03),
         (MK = 1.985126560318915, AK = 0.25190150071599327, RK = 4.928360541116469, MKT = 1.056), true),
        ((1.00630090511318e-3, 4.0079204446461897e-4, 9.929070504222013e-4, -1.8257837975696443e-4, -6.495694567140111e-4, 1.807057372959184e-4),
         (1.05, 0.8, 0.95, 0.025, -0.02, 0.015, 0.01, -0.03, 0.02, 0.325, 0.285, 0.31, 0.005, -0.01, 0.015),
         (MK = 1.7884832070798407, AK = 0.21453206523688922, RK = 4.527171910218768, MKT = 0.928), true),
        ((9.887202716779727e-4, 4.1513374418994744e-4, 8.961459841320801e-4, 8.122273370343272e-5, -3.422637994503159e-5, 2.4645630324970035e-4),
         (0.85, 0.6, 0.75, 0.025, -0.02, 0.015, 0.01, -0.03, 0.02, 0.2583333333333333, 0.21833333333333332, 0.24333333333333332, 0.005, -0.01, 0.015),
         (MK = 0.96582247883616, AK = NaN, RK = NaN, MKT = 0.728), false),
        ((1.030382074422773e-3, 3.6535853537401707e-4, 1.0042593902032104e-3, -1.6421296987182451e-4, -6.304008115368565e-4, 2.0922097553227018e-4),
         (40.0, 40.0, 40.0, 0, 0, 0, 0, 0, 0, 40 / 3, 40 / 3, 40 / 3, 0, 0, 0),
         (MK = 10.0, AK = 8.858131487889272, RK = 10.0, MKT = 10.0), true),
        ((1.030382074422773e-3, 3.6535853537401707e-4, 1.0042593902032104e-3, -1.6421296987182451e-4, -6.304008115368565e-4, 2.0922097553227018e-4),
         (-2.0, -2.0, -2.0, 0, 0, 0, 0, 0, 0, -2 / 3, -2 / 3, -2 / 3, 0, 0, 0),
         (MK = -3 / 7, AK = -3 / 7, RK = -3 / 7, MKT = -3 / 7), true),
        ((1.0239365153816144e-3, 3.114924633970135e-5, 9.449142382786847e-4, -1.1779997918414874e-4, -6.499587429340172e-4, 3.500531488178132e-4),
         (1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0),
         (MK = 0.0, AK = 0.0, RK = 0.0, MKT = 1.0), true)]   # a negative eigenvalue
    withW(Dl, kt) = (md = (Dl[1] + Dl[2] + Dl[3]) / 3;
        merge((S0 = 1.0,), NamedTuple{(:Dxx, :Dyy, :Dzz, :Dxy, :Dxz, :Dyz)}(Dl), NamedTuple{VN}(Tuple(md^2 .* kt))))
    for (Dl, kt, r, frame) in ref
        d = derived(m, withW(Dl, kt))
        @test d.MK ≈ r.MK rtol = 1e-4
        @test d.MKT ≈ r.MKT rtol = 1e-10
        frame && @test d.AK ≈ r.AK rtol = 1e-10
        frame && @test d.RK ≈ r.RK rtol = 1e-10
    end
    # independent of DIPY, for the first tensor: MK is the mean of K(n) = MD²·W(n)/D(n)² over
    # the sphere (Gauss–Legendre in z, uniform in φ), RK its mean over the circle ⊥ e1, AK
    # K(e1); and isotropic D and W give K in all four
    Dl, kt = ref[1][1], ref[1][2]
    pg = withW(Dl, kt)
    D = Symmetric([Dl[1] Dl[4] Dl[5]; Dl[4] Dl[2] Dl[6]; Dl[5] Dl[6] Dl[3]])
    mdg = tr(D) / 3
    Kn(n) = mdg^2 * dot(EQUIP._kurtosisweights(n...), collect(kt)) / dot(n, D * n)^2
    Jg = SymTridiagonal(zeros(64), [k / sqrt(4k^2 - 1) for k in 1:63])
    zg = eigen(Jg); zs, wz = zg.values, 2 .* zg.vectors[1, :] .^ 2
    MKavg = sum(wz[i] * sum(Kn([sqrt(1 - zs[i]^2) * cos(φ), sqrt(1 - zs[i]^2) * sin(φ), zs[i]]) for φ in (0:255) .* (π / 128)) / 256 for i in 1:64) / 2
    Eg = eigen(D)
    e1, e2, e3 = Eg.vectors[:, 3], Eg.vectors[:, 2], Eg.vectors[:, 1]
    dg = derived(m, pg)
    @test dg.MK ≈ MKavg rtol = 1e-9
    @test dg.RK ≈ mean(Kn(cos(t) .* e2 .+ sin(t) .* e3) for t in (0:719) .* (π / 360)) rtol = 1e-9
    @test dg.AK ≈ Kn(e1) rtol = 1e-12
    di = derived(m, withW((0.8e-3, 0.8e-3, 0.8e-3, 0.0, 0.0, 0.0), (1.2, 1.2, 1.2, 0, 0, 0, 0, 0, 0, 0.4, 0.4, 0.4, 0, 0, 0)))
    @test all(k -> di[k] ≈ 1.2, (:MK, :AK, :RK, :MKT))
    # D's own, as DTI's; no allocation; differentiable (MK's gradient against finite differences)
    @test keys(dg) == (:MD, :FA, :AD, :RD, :V1x, :V1y, :V1z, :MK, :AK, :RK, :MKT)
    @test all(k -> dg[k] == derived(DTI(b = [0.0], gx = [0.0], gy = [0.0], gz = [0.0]), pg)[k], (:MD, :FA, :AD, :RD, :V1x))
    allocdk(m, p) = @allocated derived(m, p)
    @test_noalloc (allocdk(m, pg); allocdk(m, pg) == 0)
    fmk(x) = derived(m, merge(pg, (Dxx = x[1], V1111 = x[2]))).MK
    gmk = ForwardDiff.gradient(fmk, [pg.Dxx, pg.V1111])
    hs = (1e-10, 1e-12)
    gfd = [(fmk([pg.Dxx + hs[1], pg.V1111]) - fmk([pg.Dxx - hs[1], pg.V1111])) / 2hs[1],
           (fmk([pg.Dxx, pg.V1111 + hs[2]]) - fmk([pg.Dxx, pg.V1111 - hs[2]])) / 2hs[2]]
    @test gmk ≈ gfd rtol = 1e-5
    # and a fit's: the maps, with the truth's on average
    dr = derived(r64)
    @test keys(dr) == keys(dg) && abs(mean(dr.MKT) - 1) < 0.02
    # crlb above the kernel's inlining threshold (P > 8: its pieces called as functions, so
    # that it compiles; at P = 22 it did not finish in 23 minutes inlined): here P = 10 (V's
    # off-diagonal elements fixed at the truth's), its SD against 400 fits'
    fixV = (V1112 = 0.0, V1113 = 0.0, V1222 = 0.0, V2223 = 0.0, V1333 = 0.0, V2333 = 0.0,
            V1123 = 0.0, V1223 = 0.0, V1233 = 0.0, V1122 = MD2 / 3, V1133 = MD2 / 3, V2233 = MD2 / 3)
    @test EQUIP._CRLB_INLINE_MAXP < 10
    c10 = crlb(m, θ; snr = 100, fixV...)
    Y10 = simulate(m, θ; snr = 100, n = 400, noise = :gaussian, rng = MersenneTwister(32))
    r10 = fit(m, Y10; fixV...)
    @test count(isconverged, r10.status) == 400
    for k in (:Dxx, :Dyy, :V1111, :V3333)
        @test std(r10.params[k]) ≈ c10.sd[k] rtol = 0.15
    end
    # and the indices, by the delta method through the closed forms, against the fits' own
    d10 = derived(r10)
    for k in (:MD, :FA, :MK, :AK, :RK, :MKT)
        @test std(d10[k]) ≈ c10.sd[k] rtol = 0.15
    end
end
