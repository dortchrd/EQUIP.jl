# =================================================================================================
# test/validation/secondorder.jl: crlb's second-order noise bias (`secondorder = true`) against
# montecarlo at 20,000 copies: DTI's tensor and its derived quantities (MD, FA, AD, RD) at SNR 20,
# and a least-squares fit of Rician magnitude data (T2 decay at SNR 15), where the first-order
# bias, the floor's, is several standard errors off and the second order closes it.
#
# Part of EQUIP's validation group, run apart (about a minute):
#     julia -t auto --project=test test/runtests.jl secondorder
# =================================================================================================

@testset "second-order noise bias against montecarlo: DTI's derived quantities, Rician data" begin
    # DTI: 6 images at b = 0, 30 directions at b = 1000 s/mm², eigenvalues 1.7, 0.4, 0.3 µm²/ms
    # along x (the case of docs/design.md, roadmap item 16); measured z within ±0.8
    dirs = directionset(30)
    g = vcat(fill(SVector(1.0, 0, 0), 6), dirs)
    m = DTI(b = vcat(fill(0.0, 6), fill(1000.0, 30)), gx = getindex.(g, 1), gy = getindex.(g, 2),
            gz = getindex.(g, 3))
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
