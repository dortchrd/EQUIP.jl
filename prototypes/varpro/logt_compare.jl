# Log coordinates vs plain, with and without VarPro, at several tolerances, from the same starts
# (EQUIP's own start pipeline). Single thread, solver time only, best of 3.
#   reached:  final SSE within 1e-6 (relative) of the lowest SSE any run found for that voxel
#   off>1SD:  voxels whose estimates differ from the SAME method's tight fit by more than one
#             noise SD (the spread of the tight full fit over voxels): stopped too early
include(joinpath(@__DIR__, "logt.jl"))
using Random, Statistics, Printf, StaticArrays

const TIGHT = LM(xtol = 1e-12, ftol = 1e-15, gtol = 1e-14)

_tolog(x::SVector{P}, mask) where {P} = SVector{P}(ntuple(j -> mask[j] ? log(x[j]) : x[j], Val(P)))
_bound(x::SVector{P}, mask) where {P} = SVector{P}(ntuple(j -> mask[j] ? log(max(x[j], 0.0)) : x[j], Val(P)))

# One method: the problem's free names Fn, the wrapped model, which full-θ entries it uses
# (keep), which of those are fitted as logs (mask), and how to read comparable values back.
struct Method{W,K,M,N}
    name::String
    Fn::Tuple
    wrap::W
    keep::K
    mask::M
    nat::N          # φ -> tuple of values to compare across methods (natural units)
end

function loop!(out, ::Val{Fn}, wrap, fixed, keep, mask, lo, hi, opts, buf, Y, θ0, y, rtol) where {Fn}
    for v in eachindex(out)
        copyto!(y, view(Y, v, :))
        prob = E.VoxelProblem{Fn}(wrap, y, fixed)
        φ0 = _tolog(θ0[v][keep], mask)
        out[v] = rtol === nothing ? E.levmar(prob, φ0, lo, hi, opts, buf) : levmar_pp(prob, φ0, lo, hi, opts, buf, rtol)
    end
end

function run(meth, cache, Y, θ0, opts; rtol = nothing)
    spec = cache.spec; n, M = size(Y)
    lo = _bound(spec.lower[meth.keep], meth.mask); hi = _bound(spec.upper[meth.keep], meth.mask)
    Pn = length(meth.Fn)
    buf = Vector{Dual{E.FitTag,Float64,Pn}}(undef, M)
    y = zeros(M)
    r1 = E.levmar(E.VoxelProblem{meth.Fn}(meth.wrap, (copyto!(y, view(Y, 1, :)); y), spec.fixed),
                  _tolog(θ0[1][meth.keep], meth.mask), lo, hi, cache.opts, buf)
    out = Vector{typeof(r1)}(undef, n)
    go() = loop!(out, Val(meth.Fn), meth.wrap, spec.fixed, meth.keep, meth.mask, lo, hi, opts, buf, Y, θ0, y, rtol)
    go()
    t = minimum(@elapsed(go()) for _ in 1:3) / n * 1e6
    return (; out, t, vals = [meth.nat(r.θ) for r in out])
end

function case(title, m, truth, methods; snr, n = 2000, seed = 1, cmp = 1:2, extra = [])
    println("\n=== ", title)
    Y = simulate(m, truth; snr, n, rng = MersenneTwister(seed))
    cache = init(m); F = E.freenames(cache.spec); P = length(F)
    S = fit(m, Y; solver = LM(maxiter = 0)).params
    θ0 = [SVector{P,Float64}(ntuple(k -> S[F[k]][v], P)) for v in 1:n]
    runs = []
    for meth in methods, (tl, opts) in (("tight", TIGHT), ("default", LM()), ("loose (1e-4, 1e-6)", LM(xtol = 1e-4, ftol = 1e-6)))
        push!(runs, (meth.name, tl, meth, run(meth, cache, Y, θ0, E.SolverOptions{Float64}(opts))))
    end
    for (meth, rtol) in extra            # per-parameter relative step test only (ftol, gtol off)
        push!(runs, (meth.name, "per-param rel. step $(rtol)", meth,
                     run(meth, cache, Y, θ0, E.SolverOptions{Float64}(LM(ftol = 1e-15, gtol = 1e-14)); rtol)))
    end
    best = [minimum(r[4].out[v].sse for r in runs) for v in 1:n]
    tightvals = Dict(name => r.vals for (name, tl, _, r) in runs if tl == "tight")
    ref = tightvals[methods[1].name]
    sd = [std(getindex.(ref, k)) for k in cmp]
    @printf("  %-22s %-24s %8s %9s %8s %8s %9s\n", "method", "tolerance", "reached", "off>1SD", "mean it", "maxiter", "µs/voxel")
    for (name, tl, meth, r) in runs
        reached = 100 * count(v -> r.out[v].sse <= best[v] * (1 + 1e-6) + 1e-12, 1:n) / n
        own = tightvals[name]
        off = count(v -> maximum(abs(r.vals[v][k] - own[v][k]) / sd[i] for (i, k) in enumerate(cmp)) > 1, 1:n)
        @printf("  %-22s %-24s %7.1f%% %9d %8.1f %8d %9.2f\n", name, tl, reached, off,
                mean(x.iterations for x in r.out), count(x -> x.status == EQUIP.MaxIterations, r.out), r.t)
    end
    return runs
end

te = collect(0.01:0.01:0.32)
sorted2(a, b) = a <= b ? (a, b) : (b, a)

# --- Biexponential: plain vs log T2s; full vs VarPro (model-supplied columns) --------------------
function col_short(c, m, p); @inbounds for j in eachindex(c, m.TE); c[j] = exp(-m.TE[j] / p.T2short); end; c; end
function col_long(c, m, p);  @inbounds for j in eachindex(c, m.TE); c[j] = exp(-m.TE[j] / p.T2long);  end; c; end
for (title, truth, seed) in (("T2BiexpDecay, 15% at 20 ms + 85% at 80 ms, SNR 100", (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08), 1),
                             ("T2BiexpDecay, 40% at 15 ms + 60% at 100 ms, SNR 100", (S0 = 1000.0, fshort = 0.4, T2short = 0.015, T2long = 0.1), 2))
    m = T2BiexpDecay(TE = te)
    cache = init(m)
    vs = vp_setup(cache, (:S0, :fshort), (col_short, col_long), (true, true))
    keepall = SVector(1, 2, 3, 4)
    methods = [Method("full", (:S0, :fshort, :T2short, :T2long), cache.model, keepall, (false, false, false, false), φ -> sorted2(φ[3], φ[4])),
               Method("full, log T2s", (:S0, :fshort, :T2short, :T2long), LogT(cache.model, (false, false, true, true)), keepall,
                      (false, false, true, true), φ -> sorted2(exp(φ[3]), exp(φ[4]))),
               Method("VarPro", vs.Fn, vs.vp, vs.keep, (false, false), φ -> sorted2(φ[1], φ[2])),
               Method("VarPro, log T2s", vs.Fn, LogT(vs.vp, (true, true)), vs.keep, (true, true), φ -> sorted2(exp(φ[1]), exp(φ[2])))]
    case(title, m, truth, methods; snr = 100, seed, extra = [(methods[4], 1e-4), (methods[4], 1e-3)])
end

# --- T2Decay with T2 much longer than the echoes: T2 vs log T2, and the R2 form for reference ----
m = T2Decay(TE = te)
cache = init(m)
methods = [Method("T2", (:S0, :T2), cache.model, SVector(1, 2), (false, false), φ -> (φ[1], 1 / φ[2])),
           Method("log T2", (:S0, :T2), LogT(cache.model, (false, true)), SVector(1, 2), (false, true), φ -> (φ[1], exp(-φ[2])))]
case("T2Decay, T2 = 2 s, TE 10-320 ms, SNR 50 (compared as R2 = 1/T2)", m, (S0 = 1000.0, T2 = 2.0), methods; snr = 50, cmp = 2:2)
m = R2Decay(TE = te); cache = init(m)
case("R2Decay, R2 = 0.5/s (T2 = 2 s), same data", m, (S0 = 1000.0, R2 = 0.5),
     [Method("R2", (:S0, :R2), cache.model, SVector(1, 2), (false, false), φ -> (φ[1], φ[2]))]; snr = 50, cmp = 2:2)

# --- T1IR: T1 vs log T1 (BestOf start) ------------------------------------------------------------
ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
for T1 in (0.1, 0.9)
    m = T1IR(TI = ti); cache = init(m)
    methods = [Method("T1", (:M0, :Sf, :T1), cache.model, SVector(1, 2, 3), (false, false, false), φ -> (φ[2], φ[3])),
               Method("log T1", (:M0, :Sf, :T1), LogT(cache.model, (false, false, true)), SVector(1, 2, 3), (false, false, true), φ -> (φ[2], exp(φ[3])))]
    case("T1IR, T1 = $T1 s, SNR 50", m, (M0 = 1000.0, Sf = -0.95, T1 = T1), methods; snr = 50)
end
