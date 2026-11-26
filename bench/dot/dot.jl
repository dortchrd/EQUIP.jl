# Dot products of plain weights against the number types EQUIP computes with (docs/design.md, item 30):
# five spellings and an ordered loop, ns per dot and bytes allocated; first, why StaticArrays' ⋅ is
# slow on many-partial duals (its dispatched methods, what is left uninlined, its LLVM).
#   julia --project=test bench/dot/dot.jl

using StaticArrays, ForwardDiff, LinearAlgebra, InteractiveUtils, Printf, Random
using ForwardDiff: Dual
D22(v) = Dual{Nothing}(v, ntuple(k -> Float64(k), 22))
w = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
x = SVector(ntuple(i -> D22(0.1i), 6))
println("dot(w, x) dispatches to: ", @which dot(w, x))
println("its scalar step, dot(w[1], x[1]): ", @which dot(w[1], x[1]))
println("conj(w[1]): ", @which conj(w[1]))
println("\n--- dot(w, x), optimized typed code: the calls left after inlining ---")
ci = first(code_typed(dot, Tuple{typeof(w),typeof(x)}; optimize = true))
for st in ci[1].code
    s = string(st)
    (occursin("invoke", s) || occursin(":call", s) || occursin("Core.Intrinsics", s) && false) && println("  ", first(s, 160))
end
println("statements: ", length(ci[1].code))
f_sum(w, x) = sum(w .* x)
ci2 = first(code_typed(f_sum, Tuple{typeof(w),typeof(x)}; optimize = true))
println("\n--- sum(w .* x): calls left ---")
for st in ci2[1].code
    s = string(st); occursin("invoke", s) && println("  ", first(s, 160))
end
println("statements: ", length(ci2[1].code))
# LLVM: count instructions, calls, vector ops
function llvmstats(f, T)
    io = IOBuffer(); code_llvm(io, f, T; debuginfo = :none); s = String(take!(io))
    lines = split(s, '\n')
    (lines = count(l -> occursin(r"^\s+%|^\s+store|^\s+call|^\s+ret", l), lines),
     calls = count(l -> occursin(r"\bcall\b", l) && !occursin("llvm.", l), lines),
     fmul = count(l -> occursin("fmul", l), lines), fadd = count(l -> occursin("fadd", l), lines),
     vec = count(l -> occursin(r"<\d+ x double>", l), lines),
     loads = count(l -> occursin(" load ", l), lines), stores = count(l -> occursin(r"^\s+store", l), lines),
     allocas = count(l -> occursin("alloca", l), lines), memcpy = count(l -> occursin("memcpy", l), lines))
end
println("\nLLVM, dot(w, x):     ", llvmstats(dot, Tuple{typeof(w),typeof(x)}))
println("LLVM, sum(w .* x):  ", llvmstats(f_sum, Tuple{typeof(w),typeof(x)}))
f_wo(w, x) = w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] + w[5] * x[5] + w[6] * x[6]
println("LLVM, written out:  ", llvmstats(f_wo, Tuple{typeof(w),typeof(x)}))
f_mr(w, x) = mapreduce(*, +, w, x)
println("LLVM, mapreduce:    ", llvmstats(f_mr, Tuple{typeof(w),typeof(x)}))

# --- the matrix ---
# Dot products of plain weights w (an SVector of Float64, or Float32 for Float32) against x of each
# number type a fit, crlb or a simulation uses, by five spellings; ns per dot (61 per call, the min
# of 5 runs), and the bytes allocated. Lengths 3 (a cosine), 6 (gᵀDg), 15 (DKI's kurtosis term).

const N61 = 61
struct FitTagLike end
dual(v, P) = Dual{FitTagLike}(v, ntuple(k -> 0.01k, P))
struct OuterTag end
nested(v, P) = Dual{OuterTag}(dual(v, P), ntuple(k -> dual(0.02k, P), P))      # a dual of duals, crlb's Hessians
spellings = (
    ("w ⋅ x (StaticArrays)", (w, x) -> w ⋅ x),
    ("sum(w .* x)",          (w, x) -> sum(w .* x)),
    ("mapreduce(*, +, w, x)", (w, x) -> mapreduce(*, +, w, x)),
    ("written out",          nothing),                                   # @generated below
    ("w ⋅ Tuple(x)",         (w, x) -> w ⋅ Tuple(x)),                    # the cosine's form: g[j] ⋅ n, n a tuple
    ("ordered loop",         (w, x) -> loopdot(w, x)),
)
@inline function loopdot(w, x)
    s = w[1] * x[1]
    @inbounds for j in 2:length(w)
        s += w[j] * x[j]
    end
    return s
end
@generated function writtenout(w::SVector{N}, x::SVector{N}) where {N}
    ex = :(w[1] * x[1]); for i in 2:N; ex = :($ex + w[$i] * x[$i]); end
    return :(@inbounds $ex)
end
function run!(out, f, ws, x)
    @inbounds for j in eachindex(out, ws)
        out[j] = f(ws[j], x)
    end
    return out
end
function bench(f, ws, x)
    out = Vector{typeof(f(ws[1], x))}(undef, N61)
    run!(out, f, ws, x)
    a = @allocated run!(out, f, ws, x)
    reps = 20000
    t = minimum(@elapsed(for _ in 1:reps; run!(out, f, ws, x); end) for _ in 1:5) / (reps * N61)
    return 1e9t, a, copy(out)
end
types = (("Float32", L -> (SVector{L,Float32}(ntuple(i -> Float32(0.1i), L)), Float32)),
         ("Float64", L -> (SVector{L}(ntuple(i -> 0.1i, L)), Float64)),
         ("Dual, 5 partials", L -> (SVector{L}(ntuple(i -> dual(0.1i, 5), L)), Float64)),
         ("Dual, 22 partials", L -> (SVector{L}(ntuple(i -> dual(0.1i, 22), L)), Float64)),
         ("nested Dual, 5×5", L -> (SVector{L}(ntuple(i -> nested(0.1i, 5), L)), Float64)),
         ("nested Dual, 22×22", L -> (SVector{L}(ntuple(i -> nested(0.1i, 22), L)), Float64)))
println("| x's numbers | length | " * join(first.(spellings), " | ") * " | equal | allocated |")
println("|---|---:|" * repeat("---:|", length(spellings)) * "---|---:|")
for (tname, mk) in types, L in (3, 6, 15)
    x, W = mk(L)
    ws = [SVector{L,W}(ntuple(i -> W(randn(MersenneTwister(j + i))), L)) for j in 1:N61]
    res = map(spellings) do (name, f)
        g = f === nothing ? writtenout : f
        bench(g, ws, x)
    end
    eq = all(r -> r[3] == res[1][3], res) ? "all" : join([string(i) for (i, r) in enumerate(res) if r[3] != res[1][3]], ",") * " differ"
    al = maximum(r -> r[2], res)
    @printf("| %s | %d | %s | %s | %d |\n", tname, L, join([@sprintf("%.1f", r[1]) for r in res], " | "), eq, al)
    flush(stdout)
end
