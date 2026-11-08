# =================================================================================================
# montecarlo: a fit's bias, SD and RMSE measured by simulating noisy copies and fitting them,
# with crlb's prediction alongside (the part `equip montecarlo` runs)
#
#   1. SIR qMT, white matter (WM I) fitted with kmf fixed at 12.5 s⁻¹ (Dortch et al., Magn Reson
#      Med 2018;80:1824): 10 000 copies, Rician noise at SNR 200. The measured bias and SD against
#      crlb's, each with its Monte Carlo standard error, so "agrees" has a number. (Measured
#      2026-11-08: SDs within 1–2 standard errors; PSR's bias −0.02846 against −0.02818, 2.5
#      standard errors apart: crlb's bias is first order, from the wrong kmf, and the noise's own
#      second-order bias, Box's, is not in it yet: the roadmap's tier 2, design.md item 16.)
#   2. T2 decay at low SNR, where the Rician floor (about 1.25σ where the signal approaches the
#      noise) biases T2 long: montecarlo with Rician noise sees it; crlb assumes Gaussian noise and
#      does not, unless told the noise is Rician (`riciannoise`).
#
# Run from the repository folder (about 30 s on 4 threads, most of it compiling):
#   julia -t auto --project=. examples/montecarlo.jl
# =================================================================================================

using EQUIP, Random, Printf

# --- 1. SIR: measured against predicted -----------------------------------------------------------
m = QMTSIR(; SIR_4POINT...)
s = Scenario(QMTSIR, nominal(QMTSIR, :SIR_WM_I); kmf = 12.5)
mc = montecarlo(m, s; noise = :rician, snr = 200, n = 10_000, rng = MersenneTwister(1))
show(stdout, MIME"text/plain"(), mc); println("\n")
z(meas, pred, se) = (meas - pred) / se                  # the difference in standard errors
for p in (:PSR, :R1f)
    @printf("%-4s bias %.5f measured, %.5f predicted (z = %+.1f);  SD %.5f measured, %.5f predicted (z = %+.1f)\n",
            p, mc.bias[p], mc.crlb.bias[p], z(mc.bias[p], mc.crlb.bias[p], mc.se_mean[p]),
            mc.sd[p], mc.crlb.sd[p], z(mc.sd[p], mc.crlb.sd[p], mc.se_sd[p]))
end

# --- 2. T2 decay at low SNR: the Rician floor --------------------------------------------------------
m2 = T2Decay(TE = collect(0.01:0.01:0.32))
θ2 = (S0 = 1000.0, T2 = 0.08)
for snr in (10, 30, 100)
    mg = montecarlo(m2, θ2; noise = :gaussian, snr, n = 5000, rng = MersenneTwister(2))
    mr = montecarlo(m2, θ2; noise = :rician, snr, n = 5000, rng = MersenneTwister(2))
    @printf("\nSNR %3d: T2 bias, Gaussian noise %+.2f ms; Rician noise %+.2f ms (± %.2f)", snr,
            1000 * mg.bias.T2, 1000 * mr.bias.T2, 1000 * mr.se_mean.T2)
end
println()
