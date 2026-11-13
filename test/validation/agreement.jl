# =================================================================================================
# test/validation/agreement.jl: does crlb predict what Monte Carlo measures? montecarlo at 10,000
# copies (the paper's number) against its own crlb (`mc.crlb`, with `secondorder = true`, so the
# noise's own bias is predicted too), every parameter's bias and SD, as
#
#     z = (measured − predicted) / the Monte Carlo's standard error
#
# Under agreement z is about standard normal: with the 88 values asserted here, one beyond 3
# happens by chance in about one run in five (0.27% each, on a new random stream: another Julia's
# randn), beyond 4 in one run in 180; so 4 is the line (the disagreements asserted below: 7.7, 31).
# Measured 2026-11-13 (docs/design.md, "Step 6, built and measured"), with these seeds: |z| at
# most 1.9 wherever agreement is asserted.
#
#   1. Unbiased, nearly linear (T2 decay, Gaussian noise, SNR 50): crlb's SD is the bound the fit
#      reaches. (First order alone is 4 SE off for S0 at T2 = 20 ms, +0.2%: the second order is
#      what closes it.)
#   2. Dortch et al., Magn Reson Med 2018;80:1824, Fig. 1's comparison: the 4-point SIR with kmf
#      assumed 12.5 s⁻¹ (biased by the assumption: PSR −14% in WM I), Table 1's four tissues, Rician
#      magnitudes fitted by least squares, SNR 200 at thermal equilibrium.
#   3. Where first order ends: the 4-point in WM I agrees from SNR 100, not at 25 (PSR's bias
#      −18.9% against −15.9% predicted, its SD 0.95 of crlb's); the 5-point with kmf fitted (five
#      parameters from five points) agrees from SNR 800, not at 200 (PSR's SD 1.28 of crlb's, z
#      31; 1.04 at 400). Below those, montecarlo is the reference, as crlb's docstring says.
#
# Moved from bench/agreement.jl (its printout, made assertions). Part of EQUIP's validation group,
# run apart (about a minute):
#     julia -t auto --project=test test/runtests.jl agreement
# =================================================================================================

const _NMC = 10_000

# How many Monte Carlo standard errors the measured bias and SD are from crlb's, per parameter,
# and the largest of them
_zbias(mc, p) = (mc.bias[p] - mc.crlb.bias[p]) / mc.se_mean[p]
_zsd(mc, p) = (mc.sd[p] - mc.crlb.sd[p]) / mc.se_sd[p]
_zmax(mc) = maximum(p -> max(abs(_zbias(mc, p)), abs(_zsd(mc, p))), keys(mc.sd))

@testset "agreement: unbiased, nearly linear (T2 decay, Gaussian noise, SNR 50)" begin
    m = T2Decay(TE = collect(0.01:0.01:0.32))
    for T2 in (0.02, 0.08, 0.2)
        mc = montecarlo(m, (S0 = 1000.0, T2 = T2); snr = 50, noise = :gaussian, secondorder = true,
                        n = _NMC, rng = MersenneTwister(1))
        @test mc.converged == 1 && _zmax(mc) < 4
    end
end

@testset "agreement: Dortch et al. 2018, Fig. 1 (4-point SIR, kmf assumed, Rician, SNR 200)" begin
    m = QMTSIR(; SIR_4POINT...)
    s = montecarlo(m, scenarios(QMTSIR, [:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion]; kmf = 12.5);
                   noise = :rician, snr = 200, secondorder = true, n = _NMC, rng = MersenneTwister(2018))
    for (st, mc) in zip(s.settings, s.results)
        @test mc.converged == 1 && _zmax(mc) < 4
    end
    @test s.results[1].relbias.PSR < -0.1          # the assumption's bias, measured and predicted
end

@testset "agreement: where first order ends (SIR as the SNR falls)" begin
    m4, m5 = QMTSIR(; SIR_4POINT...), QMTSIR(; SIR_5POINT...)
    four = montecarlo(m4, nominal(m4, :SIR_WM_I); noise = :rician, kmf = 12.5,
                      snr = [25, 50, 100, 200, 400], secondorder = true, n = _NMC,
                      rng = MersenneTwister(2020))
    @test all(mc -> _zmax(mc) < 4, four.results[3:5])                  # SNR 100 to 400
    @test _zmax(four.results[1]) > 5                                    # SNR 25: PSR off
    five = montecarlo(m5, nominal(m5, :SIR_WM_I); noise = :rician, kmf = Free(),
                      snr = [200, 400, 800, 1600], secondorder = true, n = _NMC,
                      rng = MersenneTwister(2019))
    @test all(mc -> _zmax(mc) < 4, five.results[3:4])                   # SNR 800 and 1600
    r200 = five.results[1]
    @test r200.sd.PSR / r200.crlb.sd.PSR > 1.15                         # SNR 200: PSR's SD
end
