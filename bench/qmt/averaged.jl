# The time-averaged (first order in TR) pulsed-MT SPGR form against the exact QMTSPGRYarnykh (docs/design.md,
# item 6): excitation and MT pulse as continuous rates over TR, 0 = (A − s·W + ln(C)/TR)·M + R1·M0, s = tm/TR.
#   julia --project=test bench/qmt/averaged.jl
using EQUIP, StaticArrays, LinearAlgebra, Printf, Random
import EQUIP: MRIModel, parameters, signal!
struct AvgSPGR{M} <: MRIModel
    m::M
end
parameters(::Type{<:AvgSPGR}) = parameters(QMTCWHenkelman)
EQUIP.nobs(a::AvgSPGR) = nobs(a.m)
function signal!(out, a::AvgSPGR, p)
    (; offset, B1e, tm, TR, FA, lineshape) = a.m
    (; PSR, kmf, T2f, T2m, M0f, R1f, R1m, B1, B0) = p
    kfm = kmf * PSR
    A = SMatrix{2,2}(-(R1f + kfm), kfm, kmf, -(R1m + kmf))
    r = SVector(R1f, R1m * PSR)
    for j in eachindex(out)
        α = B1 * deg2rad(FA[j]); ω1² = (EQUIP._GAMMA_H * B1 * B1e[j])^2; Δ = offset[j] - B0
        Wf = iszero(B1e[j]) ? 0.0 : π * ω1² * lorentzian(T2f, Δ)
        Wm = iszero(B1e[j]) ? 0.0 : π * ω1² * lineshape(T2m, Δ)
        s = tm[j] / TR[j]
        X = A - SMatrix{2,2}(s * Wf, 0, 0, s * Wm) + SMatrix{2,2}(log(cos(α)) / TR[j], 0, 0, 0)
        Mz = -(X \ r)
        out[j] = M0f * sin(α) * Mz[1]
    end
    out
end
off = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
m = QMTSPGRYarnykh(offset = [off; off; 0], B1e = [fill(4.0, 6); fill(8.0, 6); 0], tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
a = AvgSPGR(m)
# 1. under Yarnykh's assumptions (R1F = R1B, no direct saturation of the free pool), the averaged
#    solve's MT/MT-off ratio against his P, Q formula (Mossahebi et al. 2014, Eqs. 8-10)
θ1 = (PSR = 0.15, kmf = 20.0, T2f = 1e-9, T2m = 12e-6, M0f = 1.0, R1f = 1.1, R1m = 1.1, R1obs = 1.1, B1 = 1.0, B0 = 0.0)
y = signal(a, θ1); f = 0.15 / 1.15; k = 20 * 0.15; R1 = 1.1; lc = log(cosd(10)) / 0.030
P = (R1 - (1 - f) * lc) / f; Q = (R1 - lc) / k + 1
WB = [π * (EQUIP._GAMMA_H * b)^2 * superlorentzian(12e-6, o) for (b, o) in zip(m.B1e, m.offset)] .* (0.012 / 0.030)
yy = (P .+ (Q - 1) .* WB) ./ (P .+ Q .* WB)
@printf("averaged solve vs Yarnykh's P/Q formula (its assumptions): %.1e\n", maximum(abs.(y[1:12] ./ y[13] .- yy[1:12])))
# 2. the averaged model against the exact one, white matter, R1f tied to R1obs
for (lab, kmf) in (("kmf 10", 10.0), ("kmf 20", 20.0), ("kmf 40", 40.0))
    θ = merge(nominal(m), (PSR = 0.15, kmf = kmf, T2f = 0.04, T2m = 12e-6, M0f = 1.0, R1obs = 1.1))
    ye, ya = signal(m, θ), signal(a, θ)
    @printf("%s: signal relative difference (MT images over MT-off) max %.2f%%\n", lab, 100maximum(abs.((ya[1:12] ./ ya[13]) ./ (ye[1:12] ./ ye[13]) .- 1)))
    r = fit(a, ye; R1obs = 1.1)
    @printf("   the averaged model fitted to exact noise-free data: PSR %+.1f%%, kmf %+.1f%%, T2m %+.1f%%, T2f %+.1f%%\n",
            100(r.params.PSR / 0.15 - 1), 100(r.params.kmf / kmf - 1), 100(r.params.T2m / 12e-6 - 1), 100(r.params.T2f / 0.04 - 1))
end
