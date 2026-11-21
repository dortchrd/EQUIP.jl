# Batching and macros (docs/design.md, roadmap item 26). Needs LoopVectorization, not one of
# EQUIP's dependencies: in an environment of its own, `Pkg.add(["LoopVectorization", "ForwardDiff"])`,
# then julia -t 1 --project=THAT bench/simd/macros.jl
# The SIR signal loop (QMTSIREqualR1's, 16 points) with each speed macro, plain and with the dual
# numbers a fit uses (4 partials), ns per evaluation; the values checked against the plain loop
using LoopVectorization, ForwardDiff, Printf
import ForwardDiff: Dual, value, partials
include(joinpath(@__DIR__, "bexp.jl"))
@inline bexp(d::Dual{T,V,N}) where {T,V,N} = (e = bexp(value(d)); Dual{T,V,N}(e, e * partials(d)))
const TI = [round.(exp.(range(log(0.010), log(2.0); length = 15)); digits = 3); 10.0]
const TD = fill(2.5, 16)
consts(p) = (k = p.kmf * (1 + p.PSR); (k, (p.Sf + p.PSR * p.Sm) / (1 + p.PSR), p.PSR * (p.Sf - p.Sm) / (1 + p.PSR)))
function plain!(out, p)                       # as EQUIP's: @inbounds only
    k, a, b = consts(p)
    @inbounds for j in eachindex(out, TI, TD)
        Ei, Ed, Ek = exp(-p.R1f * TI[j]), exp(-p.R1f * TD[j]), exp(-k * TI[j])
        out[j] = p.M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
    end
    out
end
function simd!(out, p)
    k, a, b = consts(p)
    @inbounds @simd for j in eachindex(out, TI, TD)
        Ei, Ed, Ek = exp(-p.R1f * TI[j]), exp(-p.R1f * TD[j]), exp(-k * TI[j])
        out[j] = p.M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
    end
    out
end
function fastmath!(out, p)
    k, a, b = consts(p)
    @inbounds @fastmath for j in eachindex(out, TI, TD)
        Ei, Ed, Ek = exp(-p.R1f * TI[j]), exp(-p.R1f * TD[j]), exp(-k * TI[j])
        out[j] = p.M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
    end
    out
end
function turbo!(out, p)                       # LoopVectorization: its own vector exp
    k, a, b = consts(p)
    R1f, M0f = p.R1f, p.M0f
    @turbo for j in eachindex(out)
        Ei = exp(-R1f * TI[j]); Ed = exp(-R1f * TD[j]); Ek = exp(-k * TI[j])
        out[j] = M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
    end
    out
end
function bexp!(out, p)                        # the branch-free exp, @simd
    k, a, b = consts(p)
    @inbounds @simd for j in eachindex(out, TI, TD)
        Ei, Ed, Ek = bexp(-p.R1f * TI[j]), bexp(-p.R1f * TD[j]), bexp(-k * TI[j])
        out[j] = p.M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
    end
    out
end
function bench(f!, out, p, n = 2_000_000)
    f!(out, p); best = Inf
    for _ in 1:7
        t = @elapsed for _ in 1:n; f!(out, p); end
        best = min(best, t / n)
    end
    best
end
e(i) = ntuple(j -> j == i ? 1.0 : 0.0, 4)
p = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, Sm = 0.83)
pd = (PSR = Dual(0.12, e(1)...), R1f = Dual(1.0, e(2)...), Sf = Dual(-0.95, e(3)...), M0f = Dual(1000.0, e(4)...), kmf = 12.5, Sm = 0.83)
for (lab, x) in (("plain", p), ("duals", pd))
    out = Vector{typeof(x.PSR)}(undef, 16)
    ref = copy(plain!(out, x)); t0 = bench(plain!, out, x)
    print(lab, ": @inbounds (EQUIP) ", round(1e9t0; digits = 0), " ns")
    for (name, f!) in (("@simd", simd!), ("@fastmath", fastmath!), ("@turbo", turbo!), ("branch-free exp", bexp!))
        r = try
            t = bench(f!, out, x); d = maximum(abs.(value.(out) .- value.(ref)) ./ abs.(value.(ref)))
            @sprintf("%.0f ns (%.2f×, values to %.0e)", 1e9t, t0 / t, d)
        catch err
            "fails: " * first(split(sprint(showerror, err), '\n'))
        end
        print(";  ", name, " ", r)
    end
    println()
end
