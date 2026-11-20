# QMTSIRSimple (docs/design.md, item 6): julia -t 1 --project=. bench/qmtsir_simple/conditioning.jl (about 2 minutes)
# Conditioning of SIR with kmf estimated: (PSR, kmf) against (PSR, k), k = kmf·(1 + PSR), the rate
# in the signal. Dortch 2018's 5-point scheme. Same signal (QMTSIR's), two parameterizations.
using EQUIP, Random, Statistics, Printf, LinearAlgebra
import EQUIP: parameters, signal!

# QMTSIR with k in place of kmf (k's bounds: kmf's ≥ 0, so k ≥ 0)
@kwdef struct SIRk{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A
    TD::B
    magnitude::Bool = true
end
parameters(::Type{<:SIRk}) = (
    PSR = Free(start = 0.1, lower = 0, upper = 1),
    R1f = Free(start = 1.0, lower = 0.05, upper = 10),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    M0f = Free(start = :scale, lower = 0),
    k = Free(start = 12.5 * 1.1, lower = 0),          # QMTSIR's kmf start, 12.5, at PSR 0.1
    R1m = Fixed(:R1f), Sm = Fixed(0.83), Mzf_td0 = Fixed(0.0), Mzm_td0 = Fixed(0.0),
)
signal!(out, m::SIRk, p) = signal!(out, QMTSIR(m.TI, m.TD, m.magnitude),
    (; p.PSR, p.R1f, p.Sf, p.M0f, kmf = p.k / (1 + p.PSR), p.R1m, p.Sm, p.Mzf_td0, p.Mzm_td0))

ma = QMTSIR(; SIR_5POINT...)
mb = SIRk(; SIR_5POINT...)
println("| tissue | cond (PSR, kmf) | cond (PSR, k) | cor(PSR, kmf) | cor(PSR, k) | largest |cor|, kmf / k | SD PSR %, same in both |")
println("|---|---:|---:|---:|---:|---|---|")
for t in (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)
    θa = merge(nominal(ma, t), (M0f = 1.0,))
    θb = merge(Base.structdiff(θa, NamedTuple{(:kmf,)}), (k = θa.kmf * (1 + θa.PSR),))
    ca = crlb(ma, θa; snr = 100, kmf = Free())
    cb = crlb(mb, θb; snr = 100)
    Ra, Rb = cor(ca), cor(cb)
    ia = findfirst(==(:kmf), keys(ca.fit)); ib = findfirst(==(:k), keys(cb.fit))
    off(R) = maximum(abs(R[i, j]) for i in axes(R, 1), j in axes(R, 2) if i != j)
    @printf("| %s | %.1f | %.1f | %+.3f | %+.3f | %.3f / %.3f | %.2f / %.2f |\n", t, cond(ca), cond(cb),
            Ra[1, ia], Rb[1, ib], off(Ra), off(Rb), 100 * ca.sd.PSR / θa.PSR, 100 * cb.sd.PSR / θa.PSR)
end

# Fits: 10,000 voxels, truths varied, both parameterizations from the same starting point
N = 10_000
rng = MersenneTwister(5)
uu(a, b) = a .+ (b - a) .* rand(rng, N)
lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
θ = (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = lu(5, 20),
     R1m = zeros(N), Sm = fill(0.83, N), Mzf_td0 = zeros(N), Mzm_td0 = zeros(N))
θ = merge(θ, (R1m = θ.R1f,))
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
println("\n| SNR | parameters | µs/voxel | iterations | converged | not best | kmf: median |error| | kmf: same fit as the other (rel. diff. < 1e-4) |")
println("|---:|---|---:|---:|---:|---:|---:|---:|")
for snr in (50, 100, 200)
    Y = simulate(ma, θ; snr, noise = :gaussian, rng = MersenneTwister(snr))
    ra = fit(ma, Y; kmf = Free())
    rb = fit(mb, Y)
    kb = rb.params.k ./ (1 .+ rb.params.PSR)
    sy2 = vec(sum(abs2, Y; dims = 2))
    sa = ifelse.(isconverged.(ra.status), ra.sse, Inf); sb = ifelse.(isconverged.(rb.status), rb.sse, Inf)
    bs = min.(sa, sb)
    same = count(abs.(kb ./ ra.params.kmf .- 1) .< 1e-4)
    for (label, r, kmf, s) in (("PSR, kmf", ra, ra.params.kmf, sa), ("PSR, k", rb, kb, sb))
        t = best(() -> label == "PSR, kmf" ? fit(ma, Y; kmf = Free()) : fit(mb, Y))
        nb = count(i -> !(s[i] <= bs[i] + 1e-8 * sy2[i]), 1:N)
        @printf("| %d | %s | %.2f | %.1f | %.1f%% | %d | %.1f%% | %d |\n", snr, label, 1e6t / N, mean(r.iterations),
                100 * count(isconverged, r.status) / N, nb, 100 * median(abs.(kmf ./ θ.kmf .- 1)), same)
    end
end
