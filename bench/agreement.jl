# =================================================================================================
# Agreement: does crlb predict what Monte Carlo measures?
#
#   julia -t auto --project=. bench/agreement.jl
#
# `montecarlo` simulates n noisy copies of one voxel, fits them, and measures each fitted
# parameter's bias and SD, with their standard errors; it puts crlb's prediction for the same
# scenario and noise alongside (`mc.crlb`). This script prints both, per parameter, and
#
#   z = (measured − predicted) / the Monte Carlo's standard error
#
# for the bias and for the SD: how many of its own standard errors the simulation is from the
# prediction. |z| ≤ 3 is agreement within what n copies can resolve; a |z| above it in a few of
# many rows is chance (about 1 in 370 rows for Gaussian errors), a trend in it is a real
# difference. Where the two should differ, and why:
#
#   - crlb is first order in the noise: its bias is the noise-free fit's (the effect of a wrong
#     assumption), its SD the linear propagation of the noise. A fit's own nonlinearity adds a
#     bias ∝ σ² and changes the SD at low SNR.
#   - crlb assumes Gaussian noise; magnitude images have Rician noise, whose floor biases
#     signals near zero (the inversion's null) at low SNR.
#
# So: section 1 (Gaussian, near linear) should agree everywhere; section 2 (Dortch et al.,
# MRM 2018, Fig. 1: Rician, SNR 200) should agree, as the paper found; section 3 shows the
# agreement ending as the SNR falls. n = 10,000 copies per case (the paper's); about a minute
# with 4 threads.
# =================================================================================================

using EQUIP, Printf, Random

const N = 10_000

# One case: a row per parameter, in % of its true value; the count of rows within 3 SE.
function report(io, label, mc)
    pct(x) = @sprintf("%7.2f", 100x)
    c = mc.crlb
    println(io, "\n", label, @sprintf("  (%.1f%% converged)", 100 * mc.converged))
    println(io, "  parameter     bias % (± SE)    crlb      z  |   SD % (± SE)      crlb      z    SD/crlb")
    good = 0
    for p in keys(mc.sd)
        t = abs(mc.mean[p] - mc.bias[p])                 # |true value|
        zb = (mc.bias[p] - c.bias[p]) / mc.se_mean[p]
        zs = (mc.sd[p] - c.sd[p]) / mc.se_sd[p]
        good += (abs(zb) <= 3) + (abs(zs) <= 3)
        @printf(io, "  %-9s %s ± %5.2f  %s  %5.1f  | %s ± %5.2f  %s  %5.1f    %.3f\n", p,
                pct(mc.relbias[p]), 100 * mc.se_mean[p] / t, pct(c.relbias[p]), zb,
                pct(mc.relsd[p]), 100 * mc.se_sd[p] / t, pct(c.relsd[p]), zs, mc.sd[p] / c.sd[p])
    end
    return good, 2 * length(mc.sd)
end
function section(io, title, cases)
    println(io, "\n", "="^100, "\n", title, "\n", "="^100)
    good = total = 0
    for (label, mc) in cases
        g, n = report(io, label, mc)
        good += g; total += n
    end
    println(io, "\n→ ", good, " of ", total, " values within 3 standard errors of crlb")
end

println("Monte Carlo against crlb: ", N, " noisy copies per case, ", Threads.nthreads(), " threads")
rng = MersenneTwister(2018)
tissues = [:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion]

# 1. Unbiased and nearly linear: T2 decay, Gaussian noise. crlb's SD is the classical bound.
te = collect(0.01:0.01:0.32)
mt = T2Decay(TE = te)
t2 = [("T2 decay, 32 echoes, T2 = $(1e3T2) ms, SNR 50, Gaussian",
       montecarlo(mt, (S0 = 1000.0, T2 = T2); snr = 50, n = N, noise = :gaussian, rng))
      for T2 in (0.02, 0.08, 0.2)]
section(stdout, "1. Unbiased, nearly linear: crlb's SD is the bound the fit reaches", t2)

# 2. Dortch et al. 2018, Fig. 1: the 5-point scheme (kmf fitted: unbiased) and the 4-point
#    (kmf assumed 12.5 s⁻¹: biased by the assumption), the four tissues of Table 1, Rician noise,
#    SNR 200 at thermal equilibrium.
m5, m4 = QMTSIR(; SIR_5POINT...), QMTSIR(; SIR_4POINT...)
five = montecarlo(m5, scenarios(QMTSIR, tissues; kmf = Free()); snr = 200, n = N, rng)
four = montecarlo(m4, scenarios(QMTSIR, tissues; kmf = 12.5); snr = 200, n = N, rng)
section(stdout, "2. Dortch et al. 2018, Fig. 1: SIR at SNR 200, Rician (the paper: they agree)",
        [[("5-point, kmf fitted, $(st.tissue)", r) for (st, r) in zip(five.settings, five.results)];
         [("4-point, kmf assumed 12.5, $(st.tissue)", r) for (st, r) in zip(four.settings, four.results)]])

# 3. Where first order ends: the 4-point scheme in WM I as the SNR falls, Rician noise.
low = montecarlo(m4, nominal(m4, :SIR_WM_I); kmf = 12.5, snr = [25, 50, 100, 200, 400], n = N, rng)
section(stdout, "3. Where first order ends: 4-point, WM I, kmf assumed 12.5, Rician, SNR 25 to 400",
        [("SNR $(st.snr)", r) for (st, r) in zip(low.settings, low.results)])
