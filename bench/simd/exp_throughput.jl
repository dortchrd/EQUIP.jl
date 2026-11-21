# Batching (docs/design.md, Performance notes): julia -t 1 --project=. bench/simd/exp_throughput.jl
# A branch-free exp (exp(x) = 2^n·e^r, |r| ≤ ln2/2, r by a two-part ln 2, e^r by its Taylor
# polynomial to degree 12; no range checks: for −700 < x < 700) to see what lanes of voxels could
# give, against Base.exp; ns per value over 4096 independent values
using Printf, InteractiveUtils
# (bexp is defined here and in bexp.jl: this script runs alone, without EQUIP)
@inline function bexp(x::Float64)
    n = round(x * 1.4426950408889634)
    r = muladd(n, -1.9082149292705877e-10, muladd(n, -0.6931471803691238, x))
    p = evalpoly(r, (1.0, 1.0, 0.5, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880,
                     1/3628800, 1/39916800, 1/479001600))
    return p * reinterpret(Float64, (unsafe_trunc(Int64, n) + 1023) << 52)
end
function loop!(f, y, x)
    @inbounds @simd for i in eachindex(y, x)
        y[i] = f(x[i])
    end
    y
end
per(f, n) = (f(); minimum(@elapsed(for _ in 1:n; f(); end) for _ in 1:7) / n)
x = -50 .* rand(4096); y = similar(x)
tb = per(() -> loop!(exp, y, x), 20_000) / 4096
tv = per(() -> loop!(bexp, y, x), 20_000) / 4096
err = maximum(abs.(bexp.(x) ./ exp.(x) .- 1))
s = sprint(code_llvm, loop!, (typeof(bexp), Vector{Float64}, Vector{Float64}))
@printf("Base.exp %.2f ns, branch-free exp %.2f ns per value (%.1f×), largest relative error %.1e, vector width %s\n",
        1e9tb, 1e9tv, tb / tv, err, join(unique(m.match for m in eachmatch(r"<\d+ x double>", s)), ", "))
