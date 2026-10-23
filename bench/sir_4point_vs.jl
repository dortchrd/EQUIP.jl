# =================================================================================================
# Benchmark: EQUIP vs LsqFit.jl vs NonlinearSolve.jl (SciML), 4-point SIR
#
# Fits the voxels saved by one of the example scripts (run it first) with each method, times
# each with BenchmarkTools, and checks that the methods find the same answers. The methods:
# EQUIP; LsqFit's Levenberg-Marquardt; NonlinearSolve's general solvers LevenbergMarquardt and
# TrustRegion; and SimpleNonlinearSolve's SimpleGaussNewton and SimpleTrustRegion, SciML's
# low-overhead solvers for small problems (loaded with NonlinearSolve). The data:
#   DATASET = "sir_4point_range"   examples/sir_4point_range.jl: every voxel its own tissue (PSR,
#                                  kmf, R1f over the range of Dortch 2018's tissue classes, Sf and
#                                  M0f varied too), so the fits start away from the answer
#   DATASET = "sir_wm_4point"      examples/sir_wm_4point.jl: every voxel the same white matter
#
# To keep the comparison about the fitting, everything else is the same for every method:
#   - the signal equation: LsqFit and NonlinearSolve call EQUIP's own QMTSIR `signal!`, the same
#     code EQUIP's solver uses
#   - the model: kmf fixed at 12.5 s⁻¹, R1m = R1f, Sm = 0.83 (QMTSIR's defaults, what the 4-point
#     scheme was designed for); PSR, R1f, Sf and M0f estimated
#   - the starting values: EQUIP's default starts (PSR 0.1, R1f 1, Sf -0.95, M0f scaled to the
#     voxel's data), computed once and given to every method (EQUIP included, as start maps)
#   - the bounds: none by default (`BOUNDS = false`), so every method solves the same problem;
#     with `BOUNDS = true`, EQUIP and LsqFit use QMTSIR's bounds (NonlinearSolve's least-squares
#     solvers take none)
#   - the stopping rule (see "Convergence" below), and the iteration limit
#   - derivatives: ForwardDiff (dual numbers), set explicitly for LsqFit and NonlinearSolve
#   - the threads: every method spreads the voxels over all of Julia's threads
#   - the number of runs: every method fits all voxels `RUNS` times; the median and the fastest
#     run are reported
#
# Convergence: every method stops when the parameter step is tiny, |Δθ| ≤ XTOL·|θ|, and every
# other test is switched off (EQUIP's ftol and gtol, LsqFit's g_tol). The three packages define
# "the step" slightly differently, and that cannot be changed from outside:
#   - EQUIP:          ‖D^½Δθ‖ ≤ XTOL (‖D^½θ‖ + XTOL), D = Marquardt's scaling (diag of JᵀJ)
#   - LsqFit:         ‖Δθ‖ ≤ XTOL (‖θ‖ + XTOL)
#   - NonlinearSolve: ‖Δθ‖ ≤ XTOL (‖Δθ‖ + ‖θ₀‖), θ₀ the start; its own least-squares default
#     tests the residual, which never becomes small with noisy data, so the script sets a
#     step-size rule instead (RelNormSafeBestTerminationMode, one step)
#
# SimpleNonlinearSolve needs its fast setup, or it is several times slower (measured: 15 instead
# of 2.1 µs per voxel): a residual that returns a static vector (SVector), with EQUIP's
# `signal!` inlined into it so its small work vector stays off the heap; the NonlinearFunction
# built once; and a new, cheap problem per voxel. SimpleGaussNewton is pure Gauss-Newton: no
# damping and no safeguard against bad steps, so it relies on good starts.
#
# Same answers: a method can stop at a different point than EQUIP (another local minimum, or
# not converged). For every voxel the script compares each method's SSE with EQUIP's and counts
# the voxels where it is clearly lower or higher (by more than 1e-8 of the voxel's Σy²).
#
# Run, from the repository folder:
#   julia -t auto --project=. examples/sir_4point_range.jl     # the data (once)
#   julia --project=bench -e 'using Pkg; Pkg.instantiate()'    # once: installs LsqFit, SciML, ...
#   julia -t auto --project=bench bench/sir_4point_vs.jl
# With -t 1 instead of -t auto, the same comparison on one thread.
# =================================================================================================

using EQUIP, LsqFit, NonlinearSolve, BenchmarkTools
using DelimitedFiles, LinearAlgebra, StaticArrays, Statistics, Printf

# --- Settings ------------------------------------------------------------------------------------
const DATASET = "sir_4point_range"   # or "sir_wm_4point" (see above)
const RUNS = 20          # every method fits all voxels this many times (1000: about 30 min, mostly
                         # NonlinearSolve's LevenbergMarquardt, ~1.5 s per run on 4 threads)
const BOUNDS = false     # false: no bounds for anyone; true: QMTSIR's bounds for EQUIP and LsqFit
const XTOL = 1e-8        # the shared stopping rule: relative parameter step (see above)
const MAXITER = 1000     # the same iteration limit for every method

BLAS.set_num_threads(1)  # Julia's threads already use every core; avoid oversubscription

# --- The data saved by the example script --------------------------------------------------------
const DIR = joinpath(@__DIR__, "..", "examples", "output", DATASET)
isfile(joinpath(DIR, "data.csv")) ||
    error("no data in $DIR: run examples/$DATASET.jl first")

readcsv(name) = readdlm(joinpath(DIR, name), ',', Float64; header = true)   # (values, header)
Y, _ = readcsv("data.csv")                         # one row per voxel, one column per observation
prot, _ = readcsv("protocol.csv")                  # columns TI, TD (s)
m = QMTSIR(TI = prot[:, 1], TD = prot[:, 2])
nvox = size(Y, 1)

# The truth, one value per voxel (a file with one row, the same tissue everywhere, is repeated)
tvals, tnames = readcsv("truth.csv")
θtrue = NamedTuple{Tuple(Symbol.(vec(tnames)))}(
    Tuple(size(tvals, 1) == 1 ? fill(tvals[1, j], nvox) : tvals[:, j] for j in axes(tvals, 2)))

# --- Bounds, starting values and stopping rule, shared by every method ---------------------------
const NAMES = (:PSR, :R1f, :Sf, :M0f)               # the free parameters, in this order everywhere
roles = parameters(m)
lower = BOUNDS ? [something(roles[n].lower, -Inf) for n in NAMES] : fill(-Inf, 4)
upper = BOUNDS ? [something(roles[n].upper, Inf) for n in NAMES] : fill(Inf, 4)
# EQUIP's roles: each free parameter with these bounds (the default start unless one is given)
equiproles(starts = (nothing, nothing, nothing, nothing)) =
    NamedTuple{NAMES}(ntuple(j -> Free(starts[j]; lower = lower[j], upper = upper[j]), 4))

starts = fit(m, Y; solver = LM(maxiter = 0), equiproles()...).params   # 0 iterations: the starts
P0 = permutedims(reduce(hcat, [starts[n] for n in NAMES]))   # 4 × voxels: column v = voxel v

const EQUIP_LM = LM(maxiter = MAXITER, xtol = XTOL, ftol = 0.0, gtol = 0.0)   # step test only
const SCIML_STOP = (termination_condition = RelNormSafeBestTerminationMode(Base.Fix2(norm, 2);
                                                                           max_stalled_steps = 1),
                    reltol = XTOL, maxiters = MAXITER)

# The signal for LsqFit and NonlinearSolve: EQUIP's QMTSIR equation, parameters in a plain vector
# p = [PSR, R1f, Sf, M0f], the fixed ones written out. p may hold dual numbers (for derivatives).
sirparams(p) = (PSR = p[1], R1f = p[2], Sf = p[3], M0f = p[4],
                kmf = 12.5, R1m = p[2], Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)

# --- Each method: fit every voxel; return estimates (4 × voxels), converged, steps per voxel ------
# Each is a function, so the model and data are local variables, not globals: Julia compiles fast
# code for locals, while every use of a non-constant global is looked up at run time.

function fit_equip(m, Y, P0)
    r = fit(m, Y; solver = EQUIP_LM, equiproles(Tuple(P0[j, :] for j in 1:4))...)
    return permutedims(reduce(hcat, [r.params[n] for n in NAMES])), isconverged(r), r.iterations
end

function fit_lsqfit(m, Y, P0, lower, upper)
    f!(F, x, p) = signal!(F, m, sirparams(p))       # LsqFit's in-place model; x is not used
    bounds = BOUNDS ? (; lower, upper) : (;)
    est = similar(P0)
    ok = falses(size(Y, 1))
    Threads.@threads for v in axes(Y, 1)
        r = curve_fit(f!, m.TI, Y[v, :], P0[:, v]; inplace = true, autodiff = AutoForwardDiff(),
                      x_tol = XTOL, g_tol = 0.0, maxIter = MAXITER, bounds...)
        est[:, v] .= coef(r)
        ok[v] = r.converged
    end
    return est, ok, nothing                         # LsqFit does not report its iterations
end

function fit_sciml(m, Y, P0, alg)
    resid!(r, p, y) = (signal!(r, m, sirparams(p)); r .-= y; nothing)   # residuals, in place
    est = similar(P0)
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
                y .= view(Y, v, :)
                reinit!(cache, P0[:, v]; p = y)
                sol = solve!(cache)
                est[:, v] .= sol.u
                ok[v] = NonlinearSolve.SciMLBase.successful_retcode(sol)
                steps[v] = sol.stats.nsteps
            end
        end
    end
    return est, ok, steps
end

# SimpleNonlinearSolve, set up the fast way (see the header): one voxel per problem, static vectors
function fit_simple(m, Y, P0, alg)
    function resid(p, y)                            # out of place: returns a new SVector
        s = MVector{4,eltype(p)}(undef)
        @inline signal!(s, m, sirparams(p))         # inlined, so `s` can stay off the heap
        return SVector(s) - y
    end
    f = NonlinearFunction{false}(resid)             # built once ({false}: out of place)
    est = similar(P0)
    ok = falses(size(Y, 1))
    Threads.@threads for v in axes(Y, 1)
        prob = NonlinearLeastSquaresProblem{false}(f, SVector{4}(view(P0, :, v)),
                                                   SVector{4}(view(Y, v, :)))
        sol = solve(prob, alg; SCIML_STOP...)
        est[:, v] .= sol.u
        ok[v] = NonlinearSolve.SciMLBase.successful_retcode(sol)
    end
    return est, ok, nothing                         # the Simple solvers do not count their steps
end

methods = [("EQUIP", () -> fit_equip(m, Y, P0)),
           ("LsqFit (Levenberg-Marquardt)", () -> fit_lsqfit(m, Y, P0, lower, upper)),
           ("NonlinearSolve LevenbergMarquardt",
            () -> fit_sciml(m, Y, P0, LevenbergMarquardt(autodiff = AutoForwardDiff()))),
           ("NonlinearSolve TrustRegion",
            () -> fit_sciml(m, Y, P0, TrustRegion(autodiff = AutoForwardDiff()))),
           ("NonlinearSolve SimpleGaussNewton",
            () -> fit_simple(m, Y, P0, SimpleGaussNewton(autodiff = AutoForwardDiff()))),
           ("NonlinearSolve SimpleTrustRegion",
            () -> fit_simple(m, Y, P0, SimpleTrustRegion(autodiff = AutoForwardDiff())))]

# Each voxel's SSE at a method's estimates, with the same signal code for every method
function sse_all(m, Y, est)
    s = zeros(size(Y, 2))
    return [(signal!(s, m, sirparams(view(est, :, v))); sum(abs2, s .- view(Y, v, :)))
            for v in axes(Y, 1)]
end

# --- Run: each method once (compiles it, and gives the estimates), then RUNS timed runs ----------
println("Julia ", VERSION, ", ", Threads.nthreads(), " threads, ", Sys.cpu_info()[1].model)
println("QMTSIR, SIR_4POINT, dataset ", DATASET, ", ", nvox, " voxels, ",
        BOUNDS ? "with bounds" : "no bounds", ", stop at relative step ", XTOL, ", ",
        RUNS, " runs per method")
println("truth (min to max over voxels): ",
        join(("$n $(round(minimum(θtrue[n]); sigdigits = 3)) to $(round(maximum(θtrue[n]); sigdigits = 3))"
              for n in (NAMES..., :kmf)), ", "), "\n")
results = map(methods) do (name, run)
    est, ok, steps = run()
    # exactly RUNS runs: the time limit (seconds) is set far beyond what they need
    b = @benchmark $run() samples = RUNS evals = 1 seconds = 1e6
    @printf("  %-36s median %8.4f s per run\n", name, median(b.times) / 1e9)
    (; name, est, ok, steps, sse = sse_all(m, Y, est), time = median(b.times) / 1e9,
     tmin = minimum(b.times) / 1e9, memory = b.memory)
end

# --- Speed ---------------------------------------------------------------------------------------
t_equip = results[1].time
println("\nSpeed: fitting all ", nvox, " voxels, ", RUNS, " runs each")
@printf("  %-36s %10s %10s %10s %11s %10s %16s\n", "method", "median (s)", "min (s)", "µs/voxel",
        "memory", "steps", "EQUIP faster by")
for r in results
    steps = r.steps === nothing ? "—" : @sprintf("%.1f", mean(r.steps))
    @printf("  %-36s %10.4f %10.4f %10.2f %8.1f MB %10s %15.1f×\n", r.name, r.time, r.tmin,
            1e6 * r.time / nvox, r.memory / 2^20, steps, r.time / t_equip)
end
println("  (µs/voxel: median time / voxels; steps: mean iterations per voxel, rejected ones too)")

# --- Answers: do the methods reach the same minimum, and how close to the truth? -----------------
ref = results[1]
tol = 1e-8 .* vec(sum(abs2, Y; dims = 2))          # "clearly different SSE": 1e-8 of each Σy²
println("\nAnswers, compared with EQUIP's voxel by voxel, and with the truth")
@printf("  %-36s %10s %10s %10s %10s %11s %11s %11s\n", "method", "converged", "same as", "lower SSE",
        "higher SSE", "PSR error", "R1f error", "Sf error")
@printf("  %-36s %10s %10s %10s %10s %11s %11s %11s\n", "", "", "EQUIP", "than EQUIP", "than EQUIP",
        "(median)", "(median)", "(median)")
for r in results
    same = count(v -> maximum(abs.(r.est[:, v] .- ref.est[:, v]) ./ abs.(ref.est[:, v])) <= 1e-4, 1:nvox)
    lower_sse = count(r.sse .< ref.sse .- tol)
    higher_sse = count(r.sse .> ref.sse .+ tol)
    err(j) = 100 * median(abs.(r.est[j, :] .- θtrue[NAMES[j]]) ./ abs.(θtrue[NAMES[j]]))
    @printf("  %-36s %9.1f%% %9.1f%% %10d %10d %10.2f%% %10.2f%% %10.2f%%\n", r.name,
            100 * mean(r.ok), 100 * same / nvox, lower_sse, higher_sse, err(1), err(2), err(3))
end
println("  (same as EQUIP: every estimate within 1e-4 relative; lower/higher SSE: voxels where the ",
        "method's\n   fit is clearly better/worse than EQUIP's; error: median |estimate - truth| / ",
        "truth, with kmf\n   fixed at 12.5 while the true kmf varies, so it includes that model error)")
