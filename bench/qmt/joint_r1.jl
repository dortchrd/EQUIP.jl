# T1 and pulsed qMT fitted together, or R1 from its own map (docs/design.md, item 6): white matter,
# a B1 map with a 3% SD, the same noise σ in every image (MT-off SNR 100 at TR 30 ms, 10°).
#   separate: R1obs from a variable-flip-angle fit (R1MFA: 3, 10, 20° at TR 15 ms) made with the
#             B1 map, so its error is its own noise plus its slope on B1's error (R1MFA's leakage),
#             chained into the MT fit (`uncertain = (B1 = ..., R1obs = (sd = ..., B1 = slope))`);
#             and the single-pool fit's bias on two-pool data (R1MFA on QMTSPGRYarnykh's MT-off
#             images: R1obs is not the two pools' slower rate)
#   joint:    the same flip-angle images in the MT model (MT pulse off), R1f fitted, no R1obs
#   julia --project=test bench/qmt/joint_r1.jl
using EQUIP, Printf
off = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
mt = (offset = [off; off; 0.0], B1e = [fill(4.0, 6); fill(8.0, 6); 0.0], TR = fill(0.030, 13), FA = fill(10.0, 13))
vfa = (offset = zeros(3), B1e = zeros(3), TR = fill(0.015, 3), FA = [3.0, 10.0, 20.0])
ms = QMTSPGRYarnykh(; mt..., tm = 0.012, ts = 0.002)
mj = QMTSPGRYarnykh(; offset = [mt.offset; vfa.offset], B1e = [mt.B1e; vfa.B1e], TR = [mt.TR; vfa.TR], FA = [mt.FA; vfa.FA],
                    tm = 0.012, ts = 0.002)
mv2 = QMTSPGRYarnykh(; vfa..., tm = 0.012, ts = 0.002)          # the VFA images, two-pool
R1obs = 1.1
θs = merge(nominal(ms), (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1000.0, R1obs = R1obs))
R1f = resolve(ms, θs).R1f                                        # the two pools' slower rate is R1obs
θj = merge(θs, (R1f = R1f,))
snr = 800                                                        # σ = 1.25: MT-off SNR 100
# the R1 map: its noise, and its slope on B1 (single-pool R1MFA at R1 = R1obs, same σ)
mr = R1MFA(FA = vfa.FA, TR = 0.015)
θr = (M0 = 1000.0, R1 = R1obs, B1 = 1.0)
σR1 = crlb(mr, θr; snr).sd.R1
slope = leakage(mr, θr).absolute.B1.R1
# the single-pool fit on two-pool VFA data: the R1 it reports
R1app = fit(mr, signal(mv2, θj)).params.R1
@printf("R1 map: SD %.4f 1/s (%.1f%%), slope on B1 %.3f 1/s per unit B1 (%.2f%% per 1%%); on two-pool data it reports %.4f for the slower rate %.4f (%+.1f%%)\n",
        σR1, 100σR1 / R1obs, slope, slope / R1obs, R1app, R1obs, 100(R1app / R1obs - 1))
row(lab, c) = @printf("| %s | %.1f%% | %+.1f%% | %.1f%% | %.1f%% | %+.1f%% | %.1f%% |\n", lab, 100c.relsd.PSR, 100c.relbias.PSR,
                      100c.relrmse.PSR, 100c.relsd.kmf, 100c.relbias.kmf, 100c.relrmse.kmf)
println("| approach | PSR SD | PSR bias | PSR RMSE | kmf SD | kmf bias | kmf RMSE |")
println("|---|---:|---:|---:|---:|---:|---:|")
row("separate: the image noise only", crlb(ms, θs; snr))
row("separate: B1 3%, R1obs its noise (independent)", crlb(ms, θs; snr, uncertain = (B1 = 0.03, R1obs = σR1)))
row("separate: B1 3%, R1obs its noise + slope on B1", crlb(ms, θs; snr, uncertain = (B1 = 0.03, R1obs = (sd = σR1, B1 = slope))))
row("separate: as above, and R1obs from the single-pool fit", crlb(ms, θs; snr, R1obs = R1app, uncertain = (B1 = 0.03, R1obs = (sd = σR1, B1 = slope))))
row("joint: the image noise only", crlb(mj, θj; snr, R1f = Free(start = 1.0)))
row("joint: B1 3%", crlb(mj, θj; snr, R1f = Free(start = 1.0), uncertain = (B1 = 0.03,)))
cj = crlb(mj, θj; snr, R1f = Free(start = 1.0))
@printf("joint: R1f %.3f, SD %.1f%%\n", R1f, 100cj.relsd.R1f)
