# =================================================================================================
# crlb: a fit's bias, SD and RMSE predicted from derivatives, before scanning (the part
# `equip crlb` runs)
#
#   1. SIR qMT (Dortch et al., Magn Reson Med 2018;80:1824): white matter (WM I, kmf = 5 s⁻¹)
#      fitted with kmf fixed at 12.5 s⁻¹, as the 4-point scheme assumes. The wrong kmf biases
#      PSR; the bias does not shrink with SNR, the SD does (Fig. 1's argument).
#   2. The same over SNRs: a Sweep.
#   3. Fixing kmf at its true value: unbiased; freeing it: unbiased, larger SD.
#   4. DTI's derived quantities: FA's and MD's SD by the delta method, from the six elements'.
#
# Run from the repository folder (about 15 s, most of it compiling):
#   julia --project=. examples/crlb.jl
# =================================================================================================

using EQUIP, Printf

# --- 1. one tissue, the fit's assumption wrong --------------------------------------------------
m = QMTSIR(; SIR_4POINT...)                       # Dortch 2018, Table 2's 4-point scheme
θ = nominal(QMTSIR, :SIR_WM_I)                    # the truth: Table 1's WM I (kmf = 5 s⁻¹)
s = Scenario(QMTSIR, θ; kmf = 12.5)               # the fit: kmf fixed at 12.5, wrong here
show(stdout, MIME"text/plain"(), s); println("\n")
c = crlb(m, s; snr = 200)
show(stdout, MIME"text/plain"(), c); println("\n")
@printf("PSR: bias %.4f, SD %.4f, relative RMSE %.1f%%\n\n", c.bias.PSR, c.sd.PSR, 100 * c.relrmse.PSR)

# --- 2. over SNRs: the bias stays, the SD falls ------------------------------------------------
sw = crlb(m, θ; snr = [50, 100, 200, 400], kmf = 12.5)
println("SNR   PSR bias   PSR SD")
for (setting, r) in zip(sw.settings, sw.results)
    @printf("%-5d %.4f    %.4f\n", setting.snr, r.bias.PSR, r.sd.PSR)
end

# --- 3. kmf right, or free --------------------------------------------------------------------
right = crlb(m, θ; snr = 200)                     # kmf held at its true value
free = crlb(QMTSIR(; SIR_5POINT...), θ; snr = 200, kmf = Free())   # the 5-point scheme fits kmf
@printf("\nkmf right (4-point):  PSR bias %.1e, SD %.4f\n", right.bias.PSR, right.sd.PSR)
@printf("kmf free (5-point):   PSR bias %.1e, SD %.4f\n", free.bias.PSR, free.sd.PSR)

# --- 4. derived quantities: DTI's FA and MD --------------------------------------------------------
dirs = directionset(30)
bv = [fill(0.0, 4); fill(1000.0, 30)]
G = reduce(hcat, [fill([0.0, 0.0, 0.0], 4); collect.(dirs)])
mt = DTI(; diffusionencoding(bv, G)...)
θt = nominal(mt; MD = 0.8e-3, FA = 0.7, direction = (1, 0, 0))   # white matter-like, along x
ct = crlb(mt, θt; snr = 30)
@printf("\nDTI, 4 + 30 images, SNR 30: FA %.3f ± %.3f (SD), MD %.3g ± %.2g mm²/s\n",
        ct.fit.FA, ct.sd.FA, ct.fit.MD, ct.sd.MD)
