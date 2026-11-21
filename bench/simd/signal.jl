# Batching (docs/design.md, Performance notes): julia -t 1 --project=. bench/simd/signal.jl
# Vectorizing within a voxel: QMTSIREqualR1's signal with a branch-free exp, plain and with dual
# numbers (the exp of a dual: the value's exp, times the partials), against EQUIP's (Base.exp)
using EQUIP, ForwardDiff, Printf
import ForwardDiff: Dual, value, partials
include(joinpath(@__DIR__, "bexp.jl"))
@inline bexp(d::Dual{T,V,N}) where {T,V,N} = (e = bexp(value(d)); Dual{T,V,N}(e, e * partials(d)))
function sig!(out, m, p)
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf, Sm) = p
    k = kmf * (1 + PSR); a = (Sf + PSR * Sm) / (1 + PSR); b = PSR * (Sf - Sm) / (1 + PSR)
    @inbounds @simd for j in eachindex(out, TI, TD)
        Ei, Ed, Ek = bexp(-R1f * TI[j]), bexp(-R1f * TD[j]), bexp(-k * TI[j])
        s = M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
        out[j] = magnitude ? abs(s) : s
    end
    out
end
function bench(f!, out, m, θ, n)
    f!(out, m, θ); best = Inf
    for _ in 1:7
        t = @elapsed for _ in 1:n; f!(out, m, θ); end
        best = min(best, t / n)
    end
    best
end
eq!(out, m, θ) = EQUIP.signal!(out, m, θ)
e(i) = ntuple(j -> j == i ? 1.0 : 0.0, 4)
for (lab, acq) in (("4-point", SIR_4POINT), ("16-point", SIR_16POINT))
    m = QMTSIREqualR1(; acq...)
    θ = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, Sm = 0.83)
    θd = merge(θ, (PSR = Dual(0.12, e(1)...), R1f = Dual(1.0, e(2)...), Sf = Dual(-0.95, e(3)...), M0f = Dual(1000.0, e(4)...)))
    for (pl, x) in (("plain", θ), ("duals", θd))
        out = Vector{typeof(x.PSR)}(undef, length(m.TI)); o2 = similar(out)
        a = bench(eq!, out, m, x, 2_000_000); b = bench(sig!, o2, m, x, 2_000_000)
        eq!(out, m, x); sig!(o2, m, x)
        d = maximum(abs.(value.(out) .- value.(o2)) ./ abs.(value.(out)))
        @printf("%-8s %s: EQUIP %.0f ns, branch-free exp %.0f ns (%.2f×), largest relative difference %.0e\n", lab, pl, 1e9a, 1e9b, a / b, d)
    end
end
m = QMTSIREqualR1(; SIR_16POINT...)
θd = (PSR = Dual(0.12, e(1)...), R1f = Dual(1.0, e(2)...), Sf = Dual(-0.95, e(3)...), M0f = Dual(1000.0, e(4)...), kmf = 12.5, Sm = 0.83)
o = Vector{typeof(θd.PSR)}(undef, 16); sig!(o, m, θd)
println("allocated by one dual evaluation: ", @allocated(sig!(o, m, θd)), " bytes")
