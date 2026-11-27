# =================================================================================================
# examples/mtr_design.jl: an MTR protocol chosen by its CNR efficiency
#
#     julia -t auto --project=. examples/mtr_design.jl        (about half a minute)
#
# MTR is calculated from images, not fitted, so the protocol is judged by what MTR shows: the
# contrast between two tissues against MTR's noise, per √(scan time). One MT image and its
# reference (the MT pulse off), pulsed MT with SPGR (QMTSPGRYarnykh's model), the MT pulse 12 ms;
# searched: the MT pulse's offset and power, TR and the flip angle (one each, shared by the two
# images, so the reference matches the MT image's readout), under a SAR limit, the MT pulse's
# power over TR at most that of 8 µT for 12 ms every 30 ms (B1e²·tm/TR ≤ 25.6 µT²; the
# excitations' left out).
#
# The tissues' values are illustrative, not measured: white matter (PSR 0.15) against a lesion
# with less macromolecule and longer relaxation (PSR 0.08). snr = 800 on M0f is about SNR 100 in
# the reference at 10°. MTR's errors: propagate(m, θ, g), the delta method from the images' noise.
#
# Found (2026-11-27, 8 starts, 3 s, every start within 1% of the best): with offsets from 2 kHz,
# 2 kHz, 6.5 µT, TR 20 ms, 13°, at the SAR limit: CNR per √s 31.0, against 25.6 for 2 kHz, 6 µT,
# TR 30 ms, 10° (+21%). The offset and TR are at their lower bounds (TR must hold tm + ts and the
# readout). With offsets from 1 kHz the search goes there instead: 1 kHz, 10.9 µT, TR 56 ms,
# 23.5°, 32.1; but near resonance Yarnykh's model is least exact (the free pool's direct
# saturation as a rate, valid far from resonance), so that corner is to be judged against the
# exact two-pool equations (on the roadmap) before it is trusted.
# =================================================================================================

using EQUIP, Random, Printf

wm = (PSR = 0.15, kmf = 20.0, T2f = 0.040, T2m = 12e-6, M0f = 1.0, R1f = 1.2, R1m = 1.0, R1obs = 1.1, B1 = 1.0, B0 = 0.0)
lesion = merge(wm, (PSR = 0.08, T2f = 0.070, R1f = 0.9, R1obs = 0.85))
snr = 800

# MTR's CNR per √s for a protocol p: the two images' scan time ∝ 2·TR (the same readout each)
function cnrefficiency(p)
    g(y) = mtr(p, y).mtr[1]                          # the MT image against the MT-off reference
    a, b = propagate(p, wm, g; snr), propagate(p, lesion, g; snr)
    return abs(a.truth - b.truth) / sqrt(a.sd^2 + b.sd^2) / sqrt(2 * p.TR[1])
end

# A common protocol, for comparison: 2 kHz, 6 µT, TR 30 ms, 10°
m = QMTSPGRYarnykh(offset = [2e3, 0.0], B1e = [6.0, 0.0], tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
sar(p) = 25.6 - p.B1e[1]^2 * p.tm[1] / p.TR[1]

r = design(p -> -cnrefficiency(p), m;
           acquisition = (offset = [(2e3, 20e3), 0.0], B1e = [(1.0, 12.0), 0.0], TR = (0.02, 0.1), FA = (3.0, 30.0)),
           shared = (:TR, :FA), constraints = sar, round = (offset = 100, B1e = 0.1, TR = 0.001, FA = 0.5),
           rng = MersenneTwister(1))

show(stdout, MIME("text/plain"), r); println()
for (name, p) in (("common", m), ("found", r.rounded))
    g(y) = mtr(p, y).mtr[1]
    a, b = propagate(p, wm, g; snr), propagate(p, lesion, g; snr)
    @printf("%-7s offset %6.0f Hz, B1e %4.1f µT, TR %4.0f ms, FA %4.1f°: MTR WM %.3f ± %.4f, lesion %.3f ± %.4f; CNR %.1f, per √s %.1f; SAR margin %.1f µT²\n",
            name, p.offset[1], p.B1e[1], 1e3 * p.TR[1], p.FA[1], a.truth, a.sd, b.truth, b.sd,
            abs(a.truth - b.truth) / sqrt(a.sd^2 + b.sd^2), cnrefficiency(p), sar(p))
end
