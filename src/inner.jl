# =================================================================================================
# inner: EQUIP's dot product
#
#   inner(a, b) = a[1]·b[1] + a[2]·b[2] + … + a[N]·b[N], summed in that order
#
# Every dot product in EQUIP's code is written `inner(a, b)`: a gradient direction's cosine with a
# fiber, g·n; gᵀDg and DKI's kurtosis term, weights against the model's parameters; the solver's
# predicted decrease; crlb's sums. Not LinearAlgebra's `⋅` (StaticArrays' method, for static
# vectors), which (docs/design.md, item 30; bench/dot/dot.jl):
#   - calls LinearAlgebra's scalar `dot` once per term, a function not marked @inline, which the
#     compiler declines to inline for dual numbers of many partials (a DKI fit's 22; crlb's nested
#     duals): each term a call returning a large struct through memory, 1.75–3× slower;
#   - starts from a zero term, two products and a sum of zeros for nothing;
#   - reorders its sum (@simd), so its last bits depend on the vector width, and differ between
#     machines.
# Here the products are inlined (ForwardDiff's are), there is one accumulator, and the terms are
# summed in order: the same bits on every machine, nothing allocated. Measured against ⋅,
# sum(a .* b), mapreduce(*, +, a, b) and the sum written out, for Float32, Float64, duals of 5 and
# 22 partials and nested duals (5×5, 22×22), at lengths 3, 6 and 15: within about 10% of the
# fastest for every type but small duals at lengths 3 and 6, where ⋅'s reordering is faster, and
# the fastest by far on the largest nested duals (one product alive at a time, where an unrolled
# sum holds all of them: 1105 against 1917 and 4935 ns at length 15).
# =================================================================================================

"""
    inner(a, b)

The dot product a₁b₁ + a₂b₂ + … + a_N·b_N, summed in that order: EQUIP's, which its models and
routines use (a gradient direction's cosine with a fiber, `ξ = inner(g[j], n)`; gᵀDg as weights
against the tensor's elements). For static vectors (`SVector`) and tuples, in any mix, and any
number type (plain, or the dual numbers of a fit's derivatives and crlb's second derivatives):
inlined, one accumulator, nothing allocated, the same result on every machine. Faster than
LinearAlgebra's `⋅` on dual numbers of many partials, which it calls one term at a time (1.75–3×,
measured: docs/design.md, item 30); `⋅` gives the same value to the last bits or so (EQUIP does
not export it: a model that uses it loads LinearAlgebra). Other arrays (a `Vector`) go to LinearAlgebra's `dot`.

```julia
n = fiberdirection(0.7, 0.3)
ξ = inner(g[j], n)                 # image j's gradient direction against the fiber
```
"""
@inline function inner(a::StaticVector{N}, b::StaticVector{N}) where {N}
    N == 0 && return zero(eltype(a)) * zero(eltype(b))
    s = a[1] * b[1]
    @inbounds for j in 2:N
        s += a[j] * b[j]
    end
    return s
end
# tuples as static vectors (their elements promoted to one type, so the sum's type is stable)
@inline inner(a::Tuple, b::Tuple) = inner(SVector(a), SVector(b))
@inline inner(a::StaticVector, b::Tuple) = inner(a, SVector(b))
@inline inner(a::Tuple, b::StaticVector) = inner(SVector(a), b)
@noinline inner(a::StaticVector, b::StaticVector) = throw(DimensionMismatch("inner: lengths " *
    "$(length(a)) and $(length(b))"))
# any other array: LinearAlgebra's (BLAS for long vectors of floats)
inner(a, b) = LinearAlgebra.dot(a, b)
