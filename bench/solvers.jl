# =================================================================================================
# Benchmark: EQUIP vs other Julia fitting packages, over test cases from easy to hard
#
# Every case in bench/cases.jl (4-point SIR, poor starts, 16-point SIR with kmf estimated, low
# SNR, bounds that come into play, a biexponential) is fitted by every method, timed with
# BenchmarkTools, and checked: does each method find the same answer as EQUIP, a better or a
# worse one, or fail? The methods:
#   - EQUIP (its own Levenberg-Marquardt)
#   - LsqFit's Levenberg-Marquardt
#   - NonlinearSolve's general solvers LevenbergMarquardt and TrustRegion (SciML)
#   - SimpleNonlinearSolve's SimpleGaussNewton and SimpleTrustRegion, SciML's low-overhead
#     solvers for small problems (loaded with NonlinearSolve)
#   - FastLevenbergMarquardt.jl, a small Levenberg-Marquardt for static arrays (Kames,
#     github.com/kamesy/FastLevenbergMarquardt.jl), given a ForwardDiff Jacobian
#
# Everything except the solver is the same for every method:
#   - the signal: every method calls EQUIP's own `signal!` for the case's model
#   - the free and fixed parameters (EQUIP's roles for the case, e.g. QMTSIR's kmf fixed at 12.5
#     or estimated), and the starting values (EQUIP's default starts, or the case's)
#   - the bounds: as the case says (the model's bounds, or none); NonlinearSolve and the Simple
#     solvers take none, so in a case with bounds their rows say "(no bounds)"
#   - the stopping rule and the iteration limit (see "Convergence")
#   - derivatives: ForwardDiff (dual numbers) everywhere
#   - the threads, and the number of timed runs (RUNS)
#
# Convergence: EQUIP, LsqFit and FastLevenbergMarquardt stop when the proposed parameter step is
# tiny, |Δθ| ≤ XTOL·|θ|, with every other test switched off. They define "the step" slightly
# differently, and that cannot be changed from outside:
#   - EQUIP:                  ‖D^½Δθ‖ ≤ XTOL (‖D^½θ‖ + XTOL), D = Marquardt's scaling
#   - LsqFit:                 ‖Δθ‖ ≤ XTOL (‖θ‖ + XTOL)
#   - FastLevenbergMarquardt: |Δθⱼ| < XTOL (XTOL + |θⱼ|) for every parameter j
# The SciML solvers (NonlinearSolve and the Simple ones) cannot use that rule: their stopping
# machinery sees a rejected step as no movement, i.e. as converged, so a step rule makes them
# stop early on hard fits (measured on the biexponential: NonlinearSolve's TrustRegion stopped
# after ~4 steps, worse than the best answer in 1934 of 2000 voxels; with its own default, 70).
# They get NonlinearSolve's default rule for least squares instead (stop when the residual is
# tiny, or when the steps have stalled for 32 iterations; the best point seen is returned),
# set explicitly so the Simple solvers use it too (their own default tests the residual only,
# which never becomes small with more observations than parameters). Same iteration limit.
#
# The Simple solvers and FastLevenbergMarquardt need their fast setup, or they are several times
# slower: a residual returning a static vector (SVector), with `signal!` inlined into it so its
# small work vector stays off the heap. SimpleGaussNewton is pure Gauss-Newton (no damping, no
# safeguard against bad steps).
#
# Answers, voxel by voxel: "not best" = the method's fit is clearly worse (SSE higher by more than
# 1e-8 of the voxel's Σy²) than the best fit any method found for that voxel, or failed; in a case
# with bounds, the best among the methods that solve the same problem (with bounds, or without).
# "same" = every estimate within 1e-4 (relative) of EQUIP's; "lower" and "higher" = the method's
# SSE is clearly lower or higher than EQUIP's; "failed" = no finite answer (an error, or NaN). Biexponential fits are put in the
# same order first (T2short ≤ T2long), so swapped components are not counted as different.
#
# Run, from the repository folder:
#   julia --project=bench -e 'using Pkg; Pkg.instantiate()'    # once: installs the packages
#   julia -t auto --project=bench bench/solvers.jl
# With -t 1 instead of -t auto, the same comparison on one thread.
# =================================================================================================

using EQUIP, LsqFit, NonlinearSolve, FastLevenbergMarquardt, BenchmarkTools
using ForwardDiff, LinearAlgebra, Random, StaticArrays, Statistics, Printf
include("cases.jl")

# --- Settings ------------------------------------------------------------------------------------
const CASES = ALLCASES   # or some of them, e.g. (:sir4_range, :biexp); see bench/cases.jl
const NVOXELS = 2_000    # voxels per case (more: steadier numbers, but a longer run)
const RUNS = 3           # every method fits every voxel of a case this many times
# Run time: the slow methods on the hard cases dominate (NonlinearSolve's LevenbergMarquardt
# takes ~3-5 ms per voxel on the 16-point and biexponential cases). With these settings, all six
# cases take roughly 5-10 minutes on 4 threads; 1000 voxels and 1 run, about 2 minutes.
const BOUNDS = nothing   # nothing: as each case says; true or false: for every case
const XTOL = 1e-8        # the shared stopping rule: relative parameter step (see above)
const MAXITER = 1000     # the same iteration limit for every method

BLAS.set_num_threads(1)  # Julia's threads already use every core; avoid oversubscription

const EQUIP_LM = LM(maxiter = MAXITER, xtol = XTOL, ftol = 0.0, gtol = 0.0)   # step test only
# NonlinearSolve's default stopping rule for least squares (see "Convergence" above)
const SCIML_STOP = (termination_condition = AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2);
                                                                           max_stalled_steps = 32),
                    maxiters = MAXITER)

# Every parameter of the model from the free ones (a plain or static vector p, possibly of dual
# numbers) and the fixed ones. `_resolve_ties` is EQUIP's internal helper for ties such as
# QMTSIR's R1m = R1f: the same full parameter set EQUIP's own solver builds.
@inline allparams(::Val{F}, p, fixed) where {F} =
    EQUIP._resolve_ties(merge(NamedTuple{F}(ntuple(j -> p[j], Val(length(F)))), fixed))

# --- One case, ready to fit: free names F, starts P0 (one column per voxel), bounds, fixed values -
function prepare(name)
    c = makecase(name; n = NVOXELS)
    bounds = BOUNDS === nothing ? c.bounds : BOUNDS
    spec = init(c.m; c.roles...).spec                # EQUIP's setup: free names, bounds, fixed
    F = EQUIP.freenames(spec)
    P = length(F)
    lower = bounds ? collect(spec.lower) : fill(-Inf, P)
    upper = bounds ? collect(spec.upper) : fill(Inf, P)
    # EQUIP's keywords: the case's roles, every free parameter with these bounds and a start
    # (`nothing`: its default start)
    roles(starts) = merge(c.roles, NamedTuple{F}(ntuple(j -> Free(starts[j]; lower = lower[j],
                                                                  upper = upper[j]), P)))
    st = fit(c.m, c.Y; solver = LM(maxiter = 0), roles(ntuple(_ -> nothing, P))...).params
    P0 = permutedims(reduce(hcat, [st[n] for n in F]))   # EQUIP's starts (0 iterations)
    c.starts === nothing || (P0 = c.starts(MersenneTwister(2), P0, F))
    return (; c..., F, P0, lower, upper, bounds, fixed = spec.fixed, roles)
end

# --- The methods: each fits every voxel, returns estimates (P × voxels), converged, steps --------
# Model, data and settings are arguments, so they are local variables: Julia compiles fast code
# for those, while every use of a non-constant global is looked up at run time. A voxel whose fit
# throws an error keeps NaN estimates (counted as failed).

function fit_equip(c)
    r = fit(c.m, c.Y; solver = EQUIP_LM, c.roles(Tuple(c.P0[j, :] for j in axes(c.P0, 1)))...)
    return permutedims(reduce(hcat, [r.params[n] for n in c.F])), isconverged(r), r.iterations
end

function fit_lsqfit(m, Y, P0, lower, upper, bounds, vF, fixed)
    f!(out, x, p) = signal!(out, m, allparams(vF, p, fixed))   # LsqFit's model; x is not used
    x = collect(1.0:size(Y, 2))
    kw = bounds ? (; lower, upper) : (;)
    est = fill(NaN, size(P0))
    ok = falses(size(Y, 1))
    Threads.@threads for v in axes(Y, 1)
        try
            r = curve_fit(f!, x, Y[v, :], P0[:, v]; inplace = true, autodiff = AutoForwardDiff(),
                          x_tol = XTOL, g_tol = 0.0, maxIter = MAXITER, kw...)
            est[:, v] .= coef(r)
            ok[v] = r.converged
        catch
        end
    end
    return est, ok, nothing                         # LsqFit does not report its iterations
end

function fit_sciml(m, Y, P0, alg, vF, fixed)
    resid!(r, p, y) = (signal!(r, m, allparams(vF, p, fixed)); r .-= y; nothing)
    est = fill(NaN, size(P0))
    ok = falses(size(Y, 1))
    steps = zeros(Int, size(Y, 1))
    # One block of voxels per thread, each with its own solver cache, reused voxel after voxel
    # (`init` once, then `reinit!` with the next start and data: SciML's fast pattern).
    blocks = Iterators.partition(axes(Y, 1), cld(size(Y, 1), Threads.nthreads()))
    @sync for vs in blocks
        Threads.@spawn begin
            y = Y[first(vs), :]
            f = NonlinearFunction(resid!; resid_prototype = zeros(size(Y, 2)))
            cache = init(NonlinearLeastSquaresProblem(f, P0[:, first(vs)], y), alg; SCIML_STOP...)
            for v in vs
                try
                    y .= view(Y, v, :)
                    reinit!(cache, P0[:, v]; p = y)
                    sol = solve!(cache)
                    est[:, v] .= sol.u
                    ok[v] = NonlinearSolve.SciMLBase.successful_retcode(sol)
                    steps[v] = sol.stats.nsteps
                catch
                end
            end
        end
    end
    return est, ok, steps
end

# The static residual for the Simple solvers and FastLevenbergMarquardt: returns a new SVector of
# M values; `signal!` is inlined so its MVector can stay off the heap.
function static_residual(m, vF, fixed, ::Val{M}) where {M}
    function resid(p, y)
        s = MVector{M,eltype(p)}(undef)
        @inline signal!(s, m, allparams(vF, p, fixed))
        return SVector(s) - y
    end
end

function fit_simple(m, Y, P0, alg, vF::Val{F}, fixed, vM::Val{M}) where {F,M}
    P = length(F)
    f = NonlinearFunction{false}(static_residual(m, vF, fixed, vM))   # built once ({false}: out of place)
    est = fill(NaN, size(P0))
    ok = falses(size(Y, 1))
    Threads.@threads for v in axes(Y, 1)
        try
            prob = NonlinearLeastSquaresProblem{false}(f, SVector{P}(view(P0, :, v)),
                                                       SVector{M}(view(Y, v, :)))
            sol = solve(prob, alg; SCIML_STOP...)
            est[:, v] .= sol.u
            ok[v] = NonlinearSolve.SciMLBase.successful_retcode(sol)
        catch
        end
    end
    return est, ok, nothing                         # the Simple solvers do not count their steps
end

function fit_fastlm(m, Y, P0, lower, upper, bounds, vF::Val{F}, fixed, vM::Val{M}) where {F,M}
    P = length(F)
    resid = static_residual(m, vF, fixed, vM)
    jac(p, y) = ForwardDiff.jacobian(q -> resid(q, y), p)       # static M × P, dual numbers
    lb, ub = bounds ? (SVector{P}(lower), SVector{P}(upper)) : (nothing, nothing)
    est = fill(NaN, size(P0))
    ok = falses(size(Y, 1))
    steps = zeros(Int, size(Y, 1))
    Threads.@threads for v in axes(Y, 1)
        try
            x, _, converged, iterations = lmsolve(resid, jac, SVector{P}(view(P0, :, v)),
                                                  SVector{M}(view(Y, v, :)), lb, ub;
                                                  xtol = XTOL, ftol = 0.0, gtol = 0.0,
                                                  maxit = MAXITER)
            est[:, v] .= x
            ok[v] = converged > 0                   # 1-3 converged, -1 iteration limit
            steps[v] = iterations
        catch
        end
    end
    return est, ok, steps
end
# Every method for one case, as (name, function that fits all voxels, can it take bounds). Val(F)
# and Val(M) carry the free names and the number of observations as types, so the static methods
# compile for them.
function solvers(c)
    vF, vM = Val(c.F), Val(size(c.Y, 2))
    (; m, Y, P0, lower, upper, bounds, fixed) = c
    nb = bounds ? " (no bounds)" : ""               # for the solvers that cannot take bounds
    return [("EQUIP", () -> fit_equip(c), true),
            ("LsqFit (Levenberg-Marquardt)", () -> fit_lsqfit(m, Y, P0, lower, upper, bounds, vF, fixed),
             true),
            ("NonlinearSolve LevenbergMarquardt" * nb,
             () -> fit_sciml(m, Y, P0, LevenbergMarquardt(autodiff = AutoForwardDiff()), vF, fixed), false),
            ("NonlinearSolve TrustRegion" * nb,
             () -> fit_sciml(m, Y, P0, TrustRegion(autodiff = AutoForwardDiff()), vF, fixed), false),
            ("SimpleGaussNewton" * nb,
             () -> fit_simple(m, Y, P0, SimpleGaussNewton(autodiff = AutoForwardDiff()), vF, fixed, vM),
             false),
            ("SimpleTrustRegion" * nb,
             () -> fit_simple(m, Y, P0, SimpleTrustRegion(autodiff = AutoForwardDiff()), vF, fixed, vM),
             false),
            ("FastLevenbergMarquardt", () -> fit_fastlm(m, Y, P0, lower, upper, bounds, vF, fixed, vM),
             true)]
end

# Each voxel's SSE at a method's estimates (NaN for a failed voxel), with the same signal code
function sse_all(m, Y, est, vF, fixed)
    s = zeros(size(Y, 2))
    return [all(isfinite, view(est, :, v)) ?
            (signal!(s, m, allparams(vF, view(est, :, v), fixed)); sum(abs2, s .- view(Y, v, :))) : NaN
            for v in axes(Y, 1)]
end

# --- Run every case ------------------------------------------------------------------------------
println("Julia ", VERSION, ", ", Threads.nthreads(), " threads, ", Sys.cpu_info()[1].model, "; ",
        RUNS, " runs per method; EQUIP, LsqFit and FastLevenbergMarquardt stop at relative step ",
        XTOL, ", the SciML solvers by their default rule")
overview = Dict{Tuple{Symbol,String},Any}()       # (case, method) => speed and failures
methodnames = String[]
for name in CASES
    c = prepare(name)
    nvox, M = size(c.Y)
    println("\n", "="^100, "\nCase ", name, ": ", c.title)
    println("  ", nvox, " voxels, ", M, " observations, free: ", join(c.F, ", "),
            c.bounds ? "; with the model's bounds" : "; no bounds")
    println("  truth (min to max): ", join(("$n $(round(minimum(c.truth[n]); sigdigits = 3)) to " *
                                            "$(round(maximum(c.truth[n]); sigdigits = 3))"
                                            for n in c.F if haskey(c.truth, n)), ", "))
    results = map(solvers(c)) do (mname, run, takesbounds)
        est, ok, steps = run()                      # compiles it, and gives the estimates
        c.order! === nothing || c.order!(est, c.F)  # e.g. biexponential components in order
        b = @benchmark $run() samples = RUNS evals = 1 seconds = 1e6   # exactly RUNS runs
        (; name = mname, est, ok, steps, sse = sse_all(c.m, c.Y, est, Val(c.F), c.fixed),
         time = median(b.times) / 1e9, tmin = minimum(b.times) / 1e9, memory = b.memory,
         group = c.bounds && takesbounds)            # which problem it solved: bounded or not
    end

    # Speed
    t_equip = results[1].time
    @printf("\n  %-42s %10s %10s %10s %11s %8s %10s\n", "method", "median (s)", "min (s)",
            "µs/voxel", "memory", "steps", "vs EQUIP")
    for r in results
        steps = r.steps === nothing ? "—" : @sprintf("%.1f", mean(r.steps))
        @printf("  %-42s %10.4f %10.4f %10.2f %8.1f MB %8s %9.1f×\n", r.name, r.time, r.tmin,
                1e6 * r.time / nvox, r.memory / 2^20, steps, r.time / t_equip)
    end

    # Answers, voxel by voxel: against the best any method found, against EQUIP's, and the truth
    ref = results[1]
    tol = 1e-8 .* vec(sum(abs2, c.Y; dims = 2))
    finite(x) = isfinite(x) ? x : Inf
    best = Dict(g => [minimum(finite(r.sse[v]) for r in results if r.group == g) for v in 1:nvox]
                for g in unique(r.group for r in results))
    errnames = [n for n in c.F if haskey(c.truth, n)]
    @printf("\n  %-42s %9s %8s %7s %7s %7s %7s", "method", "converged", "not best", "failed", "same",
            "lower", "higher")
    foreach(n -> @printf(" %9s", string(n, " err")), errnames)
    println()
    for r in results
        same = count(v -> all(abs.(r.est[:, v] .- ref.est[:, v]) .<= 1e-4 .* (abs.(ref.est[:, v]) .+ 1e-3)),
                     1:nvox)
        failed = count(!isfinite, r.sse)
        notbest = count(v -> !isfinite(r.sse[v]) || r.sse[v] > best[r.group][v] + tol[v], 1:nvox)
        nlower = count(r.sse .< ref.sse .- tol)
        nhigher = count(r.sse .> ref.sse .+ tol)
        @printf("  %-42s %8.1f%% %8d %7d %7d %7d %7d", r.name, 100 * mean(r.ok), notbest, failed, same,
                nlower, nhigher)
        for n in errnames
            j = findfirst(==(n), c.F)
            e = abs.(r.est[j, :] .- c.truth[n]) ./ abs.(c.truth[n])
            @printf(" %8.2f%%", 100 * median(filter(isfinite, e)))
        end
        println()
        base = replace(r.name, " (no bounds)" => "")
        base in methodnames || push!(methodnames, base)
        overview[(name, base)] = (; us = 1e6 * r.time / nvox, notbest)
    end
    println("  (not best: voxels whose fit is clearly worse than the best any method found, or failed;",
            " same: every\n   estimate within 1e-4 of EQUIP's; lower/higher: SSE clearly lower/higher than",
            " EQUIP's;\n   err: median |estimate - truth| / truth", c.bounds ? "; with bounds, the methods" *
            " without them solve another problem,\n   so they are compared among themselves)" : ")")
end

# --- Summary over the cases: µs per voxel, and voxels with a worse fit than EQUIP's or no answer --
if length(CASES) > 1
    println("\n", "="^100, "\nSummary: µs per voxel (voxels not best: a clearly worse fit than the best",
            " any method found, or failed)")
    @printf("  %-42s", "method")
    foreach(n -> @printf(" %20s", n), CASES)
    println()
    for mname in methodnames
        @printf("  %-42s", mname)
        for n in CASES
            s = overview[(n, mname)]
            @printf(" %20s", @sprintf("%.2f (%d)", s.us, s.notbest))
        end
        println()
    end
end
