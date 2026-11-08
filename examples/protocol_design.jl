# =================================================================================================
# Protocol design: the acquisition that minimizes the predicted error over tissues (the part
# `equip design` runs), here Dortch et al., Magn Reson Med 2018;80:1824 again
#
#   1. The objective, Eq. 5 by default: the worst tissue's sum of relative mean squared errors
#      (bias² + SD², from crlb) times the scan time, over the four SIR tissue classes of Table 1,
#      the fit assuming kmf = 12.5 s⁻¹ (so a wrong kmf counts as the bias it causes).
#   2. Table 2's 4-point scheme evaluated: 0.243395 (the paper: 0.2434).
#   3. The search (differential evolution, then Nelder–Mead, from several starts on threads):
#      the paper's optimum before rounding is 0.2433; the result rounded to the millisecond.
#
# Run from the repository folder (about 40 s on 4 threads, most of it compiling):
#   julia -t auto --project=. examples/protocol_design.jl
# =================================================================================================

using EQUIP, Random, Printf

# --- 1. the objective -------------------------------------------------------------------------------
tissues = [:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion]        # Table 1's classes
ss = scenarios(QMTSIR, tissues; kmf = 12.5)                    # the fit assumes kmf = 12.5
m4 = QMTSIR(; SIR_4POINT..., magnitude = false)
d = ProtocolDesign(m4, ss; snr = 200,
                   acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),  # to optimize, bounds (s)
                   constraints = m -> m.TI .+ m.TD .- 0.5,            # TR ≥ 0.8 s
                   scantime = m -> sum(m.TI .+ m.TD .+ 0.3))          # Σ(TI + TD + t_FSE)

# --- 2. Table 2's scheme ------------------------------------------------------------------------------
@printf("Eq. 5 at Table 2's 4-point scheme: %.6f (the paper: 0.2434)\n", d(m4))

# --- 3. the search ------------------------------------------------------------------------------------
r = optimize_protocol(d; starts = 8, rng = MersenneTwister(1))
@printf("optimum: %.6f (the paper's, before rounding: 0.2433); rounded to the ms: %.6f\n",
        r.value, r.rounded_value)
println("TI (s): ", r.rounded.TI, "\nTD (s): ", r.rounded.TD)
