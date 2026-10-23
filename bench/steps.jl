# =================================================================================================
# PROTOTYPE (branch step-proto): EQUIP's step-control variants over the test cases
#
# Every variant of EQUIP's Levenberg-Marquardt (the LM options `scaling`, `update`, `gnfirst`,
# `guard`, `damping`; see src/fit/levmar.jl) fits every case of bench/cases.jl, without bounds and
# with the model's bounds, from the same starts and with the same stopping rule as
# bench/solvers.jl (relative step 1e-8, other tests off, at most 1000 iterations). Reported per
# variant: voxels "not best" (a clearly worse fit than the best found by any variant or by the
# reference solvers LsqFit, FastLevenbergMarquardt and, without bounds, NonlinearSolve's
# TrustRegion), mean steps per voxel, % converged, and µs per voxel.
#
# The question: which step control fixes EQUIP's misses without bounds (biexponential, poor
# starts) and its crawling voxels, without making anything worse with bounds?
#
# Run, from the repository folder:  julia -t auto --project=bench bench/steps.jl
# =================================================================================================

using EQUIP, LsqFit, NonlinearSolve, FastLevenbergMarquardt
using ForwardDiff, LinearAlgebra, Random, StaticArrays, Statistics, Printf
include("cases.jl")

const CASES = ALLCASES
const NVOXELS = 1_000
const XTOL = 1e-8
const MAXITER = 1000
BLAS.set_num_threads(1)

const BASE = (maxiter = MAXITER, xtol = XTOL, ftol = 0.0, gtol = 0.0)   # the benchmark's rule
const VARIANTS = [
    ("baseline (today's solver)", (;)),
    ("scaling current", (scaling = :current,)),
    ("Marquardt update", (update = :marquardt,)),
    ("damping 1e-6", (damping = 1e-6,)),
    ("Gauss-Newton first", (gnfirst = true,)),
    ("guard off", (guard = false,)),
    ("current + Marquardt", (scaling = :current, update = :marquardt)),
    ("current + GN first", (scaling = :current, gnfirst = true)),
    ("Marquardt + GN first", (update = :marquardt, gnfirst = true)),
    ("current + Marquardt + GN first", (scaling = :current, update = :marquardt, gnfirst = true)),
    ("hybrid scaling", (scaling = :hybrid,)),
    ("hybrid + damping 1e-6", (scaling = :hybrid, damping = 1e-6)),
    ("hybrid + Marquardt", (scaling = :hybrid, update = :marquardt)),
    ("trust radius", (trust = true,)),
    ("trust radius + damping 1e-6", (trust = true, damping = 1e-6)),
    ("trust radius + Marquardt", (trust = true, update = :marquardt)),
]

@inline allparams(::Val{F}, p, fixed) where {F} =
    EQUIP._resolve_ties(merge(NamedTuple{F}(ntuple(j -> p[j], Val(length(F)))), fixed))

# One case ready to fit (as bench/solvers.jl's `prepare`)
function prepare(name, bounds)
    c = makecase(name; n = NVOXELS)
    spec = init(c.m; c.roles...).spec
    F = EQUIP.freenames(spec)
    P = length(F)
    lower = bounds ? collect(spec.lower) : fill(-Inf, P)
    upper = bounds ? collect(spec.upper) : fill(Inf, P)
    roles(starts) = merge(c.roles, NamedTuple{F}(ntuple(j -> Free(starts[j]; lower = lower[j],
                                                                  upper = upper[j]), P)))
    st = fit(c.m, c.Y; solver = LM(maxiter = 0), roles(ntuple(_ -> nothing, P))...).params
    P0 = permutedims(reduce(hcat, [st[n] for n in F]))
    c.starts === nothing || (P0 = c.starts(MersenneTwister(2), P0, F))
    bounds && (P0 = clamp.(P0, lower, upper))
    return (; c..., F, P0, lower, upper, bounds, fixed = spec.fixed, roles)
end

function fit_equip(c, kw)
    r = fit(c.m, c.Y; solver = LM(; BASE..., kw...), c.roles(Tuple(c.P0[j, :] for j in axes(c.P0, 1)))...)
    return permutedims(reduce(hcat, [r.params[n] for n in c.F])), r
end

# The reference solvers (not timed): LsqFit, FastLevenbergMarquardt, NonlinearSolve TrustRegion
function references(c)
    vF, M, P = Val(c.F), size(c.Y, 2), length(c.F)
    out = Matrix{Float64}[]
    f!(o, x, p) = signal!(o, c.m, allparams(vF, p, c.fixed))
    kw = c.bounds ? (; lower = c.lower, upper = c.upper) : (;)
    push!(out, reduce(hcat, [try coef(curve_fit(f!, collect(1.0:M), c.Y[v, :], c.P0[:, v]; inplace = true,
                                                autodiff = AutoForwardDiff(), x_tol = XTOL, g_tol = 0.0,
                                                maxIter = MAXITER, kw...))
                             catch; fill(NaN, P) end for v in axes(c.Y, 1)]))
    resid(p, y) = SVector{M}(signal(c.m, allparams(vF, p, c.fixed))) - y
    jac(p, y) = ForwardDiff.jacobian(q -> resid(q, y), p)
    lb, ub = c.bounds ? (SVector{P}(c.lower), SVector{P}(c.upper)) : (nothing, nothing)
    push!(out, reduce(hcat, [try collect(lmsolve(resid, jac, SVector{P}(c.P0[:, v]), SVector{M}(c.Y[v, :]),
                                                 lb, ub; xtol = XTOL, ftol = 0.0, gtol = 0.0, maxit = MAXITER)[1])
                             catch; fill(NaN, P) end for v in axes(c.Y, 1)]))
    if !c.bounds
        r!(o, p, y) = (signal!(o, c.m, allparams(vF, p, c.fixed)); o .-= y; nothing)
        y = c.Y[1, :]
        cache = init(NonlinearLeastSquaresProblem(NonlinearFunction(r!; resid_prototype = zeros(M)), c.P0[:, 1], y),
                     TrustRegion(autodiff = AutoForwardDiff()); maxiters = MAXITER)
        push!(out, reduce(hcat, [try y .= c.Y[v, :]; reinit!(cache, c.P0[:, v]; p = y); collect(solve!(cache).u)
                                 catch; fill(NaN, P) end for v in axes(c.Y, 1)]))
    end
    return out
end

function sse_all(c, est)
    s = zeros(size(c.Y, 2))
    vF = Val(c.F)
    return [all(isfinite, view(est, :, v)) ?
            (signal!(s, c.m, allparams(vF, view(est, :, v), c.fixed)); sum(abs2, s .- view(c.Y, v, :))) : Inf
            for v in axes(c.Y, 1)]
end

println("Julia ", VERSION, ", ", Threads.nthreads(), " threads; ", NVOXELS, " voxels per case")
table = Dict{Any,Any}()
for name in CASES, bounded in (false, true)
    c = prepare(name, bounded)
    nvox = size(c.Y, 1)
    runs = map(VARIANTS) do (label, kw)
        est, r = fit_equip(c, kw)
        t = minimum(@elapsed(fit_equip(c, kw)) for _ in 1:3)
        (; label, sse = sse_all(c, est), steps = mean(r.iterations), conv = mean(isconverged(r)),
         us = 1e6 * t / nvox)
    end
    refs = [sse_all(c, e) for e in references(c)]
    best = [minimum(min(minimum(r.sse[v] for r in runs), minimum(s[v] for s in refs))) for v in 1:nvox]
    tol = 1e-8 .* vec(sum(abs2, c.Y; dims = 2))
    for r in runs
        table[(name, bounded, r.label)] = (; notbest = count(r.sse .> best .+ tol), r.steps, r.conv, r.us)
    end
    refnb = [count(s .> best .+ tol) for s in refs]
    table[(name, bounded, "refs")] = refnb
    @printf("  %-16s %-10s done (references not best: LsqFit %d, FastLM %d%s)\n", name,
            bounded ? "bounds" : "no bounds", refnb[1], refnb[2], bounded ? "" : ", TrustRegion $(refnb[3])")
end

for bounded in (false, true), (what, fmt) in (("not best", r -> @sprintf("%d", r.notbest)),
                                              ("mean steps", r -> @sprintf("%.1f", r.steps)),
                                              ("µs per voxel", r -> @sprintf("%.2f", r.us)))
    println("\n", bounded ? "With bounds" : "Without bounds", ": ", what)
    @printf("  %-32s", "variant")
    foreach(n -> @printf(" %15s", n), CASES)
    what == "not best" && @printf(" %8s", "total")
    println()
    for (label, _) in VARIANTS
        @printf("  %-32s", label)
        foreach(n -> @printf(" %15s", fmt(table[(n, bounded, label)])), CASES)
        what == "not best" && @printf(" %8d", sum(table[(n, bounded, label)].notbest for n in CASES))
        println()
    end
    if what == "not best"
        for (j, ref) in enumerate(bounded ? ("LsqFit", "FastLM") : ("LsqFit", "FastLM", "TrustRegion"))
            @printf("  %-32s", ref)
            foreach(n -> @printf(" %15d", table[(n, bounded, "refs")][j]), CASES)
            @printf(" %8d\n", sum(table[(n, bounded, "refs")][j] for n in CASES))
        end
    end
end
