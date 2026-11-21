# A branch-free exp, included by the scripts beside it (docs/design.md, Performance notes, "Batching")
@inline function bexp(x::Float64)
    n = round(x * 1.4426950408889634)
    r = muladd(n, -1.9082149292705877e-10, muladd(n, -0.6931471803691238, x))
    p = evalpoly(r, (1.0, 1.0, 0.5, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880,
                     1/3628800, 1/39916800, 1/479001600))
    return p * reinterpret(Float64, (unsafe_trunc(Int64, n) + 1023) << 52)
end
