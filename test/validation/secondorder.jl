# =================================================================================================
# test/validation/secondorder.jl: crlb's second-order noise bias (`secondorder = true`), measured:
# its coefficient exactly, with antithetic pairs as the noise shrinks (T1MFA); against montecarlo
# at 20,000 copies, DTI's tensor and its derived quantities (MD, FA, AD, RD) at SNR 20, and a
# least-squares fit of Rician magnitude data (T2 decay at SNR 15), where the first-order bias,
# the floor's, is several standard errors off and the second order closes it.
#
# Part of EQUIP's validation group, run apart (about a minute):
#     julia -t auto --project=test test/runtests.jl secondorder
# =================================================================================================

@testset "second-order noise bias: its coefficient, exactly (antithetic pairs)" begin
    # d = (θ̂(μ+ε) + θ̂(μ−ε))/2 − θ̂(μ): E[d] is the bias with the odd orders cancelled, and its
    # variance is O(σ⁴) rather than O(σ²), so bias/σ² is measured to 0.4% at 200,000 pairs. As
    # σ → 0 it must tend to the predicted coefficient; at larger σ the next order (σ⁴) adds a
    # part that falls fourfold per doubling of the SNR. Measured (2026-11-13), measured/predicted:
    # SNR 50 1.119, 100 1.027, 200 1.006, 400 1.000, 1000 0.997, each ± 0.004.
    m = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
    θ = (M0 = 1000.0, T1 = 1.0, B1 = 1.0)
    μ = signal(m, θ)
    ratio(snr) = begin
        σ = θ.M0 / snr
        E = σ .* randn(MersenneTwister(snr), 200_000, 6)
        rp, rm = fit(m, μ' .+ E), fit(m, μ' .- E)
        d = (rp.params.T1 .+ rm.params.T1) ./ 2 .- θ.T1
        b = crlb(m, θ; snr, secondorder = true).noisebias.T1
        mean(d) / b, std(d) / sqrt(length(d)) / b
    end
    r1000, se1000 = ratio(1000)
    @test abs(r1000 - 1) < 3 * se1000                       # the coefficient itself
    r100, se100 = ratio(100)
    r200, _ = ratio(200)
    @test r100 > 1 + 3 * se100                              # the next order shows at SNR 100 ...
    @test 3 < (r100 - 1) / (r200 - 1) < 6                   # ... and falls as σ², about 4×
end

@testset "second-order noise bias against montecarlo: DTI's derived quantities, Rician data" begin
    # DTI: 6 images at b = 0, 30 directions at b = 1000 s/mm², eigenvalues 1.7, 0.4, 0.3 µm²/ms
    # along x (the case of docs/design.md, roadmap item 16); measured z within ±0.8
    dirs = directionset(30)
    g = vcat(fill(SVector(1.0, 0, 0), 6), dirs)
    m = DTI(b = vcat(fill(0.0, 6), fill(1000.0, 30)), g = g)
    θ = (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0)
    mc = montecarlo(m, θ; snr = 20, noise = :gaussian, secondorder = true, n = 20_000,
                    rng = MersenneTwister(1))
    for p in (:Dxx, :Dyy, :MD, :FA, :AD, :RD)
        @test abs(mc.bias[p] - mc.crlb.bias[p]) < 3 * mc.se_mean[p]
    end
    @test mc.crlb.noisebias.FA > 5 * mc.se_mean.FA           # FA's bias is resolved, not 0
    # Rician magnitude data fitted by least squares: the floor's bias (first order) and the
    # noise's own; first order alone 3.8 standard errors off for T2 at SNR 15, measured
    mt = T2Decay(TE = collect(0.01:0.01:0.32))
    θt = (S0 = 1000.0, T2 = 0.08)
    c1 = crlb(mt, θt; snr = 15, noise = :rician)
    c2 = crlb(mt, θt; snr = 15, noise = :rician, secondorder = true)
    mr = montecarlo(mt, θt; snr = 15, noise = :rician, n = 20_000, rng = MersenneTwister(2))
    @test abs(mr.bias.T2 - c2.bias.T2) < 3 * mr.se_mean.T2
    @test abs(mr.bias.T2 - c1.bias.T2) > 3 * mr.se_mean.T2
end
