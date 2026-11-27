# =================================================================================================
# bench/qmt/mamt_rk4.jl: QMTSPGRPortnoy (MAMT) against an independent RK4 integration of the
# two-pool Bloch equations, and Yarnykh's model against it, offset by offset
#
#     julia --project=test bench/qmt/mamt_rk4.jl        (about a minute)
#
# The reference: Eqs. 1a-1d of Portnoy & Stanisz (MRM 2007) integrated by RK4 (2.5e-7 s steps
# during the MT pulse), no matrix exponential; the TR's map on (Mzf, Mzm) from three starting
# states, then its fixed point. Measured 2026-11-27: the model within 3.6e-10 (a block pulse) and
# 6.3e-13 (a 64-segment Gaussian) of it; Yarnykh's signal above MAMT's near resonance (white
# matter, 12-ms block, TR 30 ms, 10°): 1 kHz +20% (4 µT), +79% (8 µT); 2 kHz +9%, +5%; 5 kHz
# +2%, +4%; 10 kHz +0.6%, +1.6%; 50 kHz +0.03%, +0.1%; MT-off images equal (4e-15).
# =================================================================================================

using EQUIP, StaticArrays, LinearAlgebra, Printf

# Eqs. 1a-1d of Portnoy & Stanisz 2007 integrated by RK4, with no matrix exponential: the TR's
# map on (Mzf, Mzm) from three starting states, then its fixed point
function rk4_signal(m, θ; dt = 2.5e-7)
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = θ
    kfm = kmf * PSR
    out = zeros(length(m.offset))
    for j in eachindex(out)
        α, Δ = B1 * deg2rad(m.FA[j]), m.offset[j] - B0
        Δω, gm = 2π * Δ, m.lineshape(T2m, Δ)
        f(x, ω1) = (Wm = m.B1e[j] == 0 ? 0.0 : π * ω1^2 * gm;
                    SVector(-Δω * x[2] - x[1] / T2f, Δω * x[1] - ω1 * x[3] - x[2] / T2f,
                            ω1 * x[2] + R1f * (1 - x[3]) - kfm * x[3] + kmf * x[4],
                            R1m * (PSR - x[4]) - (Wm + kmf) * x[4] + kfm * x[3]))
        function run(x, ω1, T, h)
            n = max(1, round(Int, T / h)); h = T / n
            for _ in 1:n
                k1 = f(x, ω1); k2 = f(x + h / 2 * k1, ω1); k3 = f(x + h / 2 * k2, ω1); k4 = f(x + h * k3, ω1)
                x += h / 6 * (k1 + 2k2 + 2k3 + k4)
            end
            return x
        end
        K = length(m.mtshape.flip); τ = m.tm[j] / K
        amp(k) = m.mtshape.flip[k] * cosd(m.mtshape.phase[k]) / sqrt(sum(abs2, m.mtshape.flip) / K)
        ω1e = 2π * 42.577478518 * B1 * m.B1e[j]
        function trmap(z)
            x = SVector(0.0, 0.0, cos(α) * z[1], z[2])                 # excitation, spoiled
            x = run(x, 0.0, m.TR[j] - m.tm[j] - m.ts[j], 1e-5)        # free recovery
            for k in 1:K
                x = run(x, ω1e * amp(k), τ, dt)                   # the MT pulse, segment by segment
            end
            x = SVector(0.0, 0.0, x[3], x[4])                         # spoiled
            x = run(x, 0.0, m.ts[j], 1e-5)
            return SVector(x[3], x[4])
        end
        z0 = trmap(SVector(0.0, 0.0))
        P = hcat(trmap(SVector(1.0, 0.0)) - z0, trmap(SVector(0.0, 1.0)) - z0)
        z = (I - P) \ z0
        out[j] = M0f * sin(α) * z[1]
    end
    return out
end
θ = (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1.0, R1f = 1.2, R1m = 1.0, R1obs = 1.1, B1 = 0.95, B0 = 30.0)
gauss = RFPulse(exp.(-((1:64) .- 32.5) .^ 2 ./ (2 * 12.0^2)), zeros(64), 0.015)    # a 15-ms Gaussian, 64 segments
offs = [1e3, 2e3, 5e3, 1e3, 2e3, 5e3, 0.0]
b1e = [fill(4.0, 3); fill(8.0, 3); 0.0]
for (name, mp) in (("block", QMTSPGRPortnoy(offset = offs, B1e = b1e, tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)),
                   ("Gaussian", QMTSPGRPortnoy(offset = offs, B1e = b1e, tm = 0.015, ts = 0.002, TR = 0.040, FA = 10, mtshape = gauss)))
    s, r = signal(mp, θ), rk4_signal(mp, θ)
    println(rpad(name, 9), "largest relative difference against RK4: ", maximum(abs.(s ./ r .- 1)))
end
# against Yarnykh's model: the same acquisition (a block pulse), offset by offset
offy = [1e3, 2e3, 2.5e3, 5e3, 10e3, 20e3, 50e3, 100e3]
for b in (4.0, 8.0)
    a = (offset = [offy; 0.0], B1e = [fill(b, length(offy)); 0.0], tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
    p, y = signal(QMTSPGRPortnoy(; a...), θ), signal(QMTSPGRYarnykh(; a...), θ)
    println("B1e $b µT, Yarnykh/MAMT − 1 by offset: ", join((@sprintf("%g kHz %+.2e", o / 1e3, y[k] / p[k] - 1) for (k, o) in enumerate(offy)), ", "), "; MT off: ", @sprintf("%.1e", y[end] / p[end] - 1))
end
