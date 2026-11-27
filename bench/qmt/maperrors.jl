# How errors in the B1, B0 and R1obs maps reach the fitted parameters (docs/design.md, item 6):
# crlb with the fit's fixed value wrong gives its bias, the noise-free fit's error through the whole
# nonlinear fit; `uncertain` adds the maps' SDs to the error budget.
#   julia --project=test bench/qmt/maperrors.jl
using EQUIP, Printf
off = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
m = QMTSPGRYarnykh(offset = [off; off; 0], B1e = [fill(4.0, 6); fill(8.0, 6); 0], tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
θ = merge(nominal(m), (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1000.0, R1obs = 1.1))
println("QMTSPGRYarnykh, 13 images, white matter (PSR 0.15, kmf 20, T2m 12 µs, R1obs 1.1): relative bias, %")
println("| map error | PSR | f | kmf | T2f | T2m |")
println("|---|---:|---:|---:|---:|---:|")
c0 = crlb(m, θ; snr = 800)
@printf("| (none: SD at MT-off SNR 100) | %.1f | %.1f | %.1f | %.1f | %.1f |\n", 100c0.relsd.PSR, 100c0.relsd.f, 100c0.relsd.kmf, 100c0.relsd.T2f, 100c0.relsd.T2m)
for (lab, kw) in (("B1 +5%", (B1 = 1.05,)), ("B1 −5%", (B1 = 0.95,)), ("B0 +50 Hz", (B0 = 50.0,)), ("B0 +200 Hz", (B0 = 200.0,)),
                  ("R1obs +5%", (R1obs = 1.155,)), ("R1obs −5%", (R1obs = 1.045,)))
    c = crlb(m, θ; snr = 800, kw...)
    @printf("| %s | %+.1f | %+.1f | %+.1f | %+.1f | %+.1f |\n", lab, 100c.relbias.PSR, 100c.relbias.f, 100c.relbias.kmf, 100c.relbias.T2f, 100c.relbias.T2m)
end
cu = crlb(m, θ; snr = 800, uncertain = (B1 = 0.03, B0 = 20.0, R1obs = 0.055))
@printf("| uncertain maps (SD: B1 3%%, B0 20 Hz, R1obs 5%%): RMSE | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
        100cu.relrmse.PSR, 100cu.relrmse.f, 100cu.relrmse.kmf, 100cu.relrmse.T2f, 100cu.relrmse.T2m)
# SIR: B1 reaches PSR through Sm (fixed), the free pool's Sf being fitted
ms = QMTSIR(; SIR_5POINT...)
θs = nominal(ms, :SIR_WM_I)
blk = blockpulse(180, 1e-3)
println("\nQMTSIR, 5-point, SIR_WM_I, 1-ms block, kmf fixed at its true value: Sm from a B1 map that is wrong (true B1 = 1)")
println("| B1 map error | Sm used | PSR | f | R1f |")
println("|---|---:|---:|---:|---:|")
Sm1 = saturationfraction(blk, 15e-6)
θs = merge(θs, (Sm = Sm1,))
for b in (1.05, 0.95, 1.10)
    Smw = saturationfraction(blk, 15e-6; B1 = b)
    c = crlb(ms, θs; snr = 200, Sm = Smw, kmf = θs.kmf)
    @printf("| %+.0f%% | %.3f | %+.1f%% | %+.1f%% | %+.1f%% |\n", 100(b - 1), Smw, 100c.relbias.PSR, 100c.relbias.f, 100c.relbias.R1f)
end

@printf("| (B1 0.9: true Sm %.3f; fit with Sm = 0.83) | | ", saturationfraction(blk, 15e-6; B1 = 0.9))
θ9 = merge(θs, (Sm = saturationfraction(blk, 15e-6; B1 = 0.9), Sf = inversionefficiency(blk; B1 = 0.9)))
c9 = crlb(ms, θ9; snr = 200, Sm = 0.83, kmf = θs.kmf)
@printf("%+.1f%% | %+.1f%% | %+.1f%% |\n", 100c9.relbias.PSR, 100c9.relbias.f, 100c9.relbias.R1f)
