# =================================================================================================
# Benchmark: EQUIP vs LsqFit.jl vs NonlinearSolve.jl (SciML), 4-point SIR in white matter
#
# Fits the 10 000 voxels saved by examples/sir_wm_4point.jl (run that first) with each method,
# times each with BenchmarkTools, and compares the estimates. To keep the comparison about the
# fitting, everything else is the same for every method:
#   - the signal equation: LsqFit and NonlinearSolve call EQUIP's own QMTSIR `signal!`, the same
#     code EQUIP's solver uses
#   - the model: kmf fixed at 12.5 s⁻¹, R1m = R1f, Sm = 0.83 (QMTSIR's defaults, what the 4-point
#     scheme was designed for); PSR, R1f, Sf and M0f estimated
#   - the starting values: EQUIP's per-voxel starts, computed once, given to every method
#     (EQUIP included, as start maps)
#   - the bounds: none by default (`BOUNDS = false`), so every method solves the same problem;
#     with `BOUNDS = true`, EQUIP and LsqFit use QMTSIR's bounds (NonlinearSolve's least-squares
#     solvers take none)
#   - the stopping rule (see "Convergence" below), and the iteration limit
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
# Derivatives: ForwardDiff (dual numbers) for every method, set explicitly for LsqFit and
# NonlinearSolve (`autodiff = AutoForwardDiff()`). EQUIP uses ForwardDiff's dual numbers itself.
#
# Run, from the repository folder:
#   julia --project=bench -e 'using Pkg; Pkg.instantiate()'    # once: installs LsqFit, SciML, ...
#   julia -t auto --project=bench bench/sir_wm_4point_vs.jl
# With -t 1 instead of -t auto, the same comparison on one thread.
# =================================================================================================

using EQUIP, LsqFit, NonlinearSolve, BenchmarkTools
using DelimitedFiles, LinearAlgebra, Statistics, Printf

# --- Settings ------------------------------------------------------------------------------------
const RUNS = 20          # every method fits all voxels this many times (1000: about 30 min, mostly
                         # NonlinearSolve's LevenbergMarquardt, ~1.5 s per run on 4 threads)
const BOUNDS = false     # false: no bounds for anyone; true: QMTSIR's bounds for EQUIP and LsqFit
const XTOL = 1e-8        # the shared stopping rule: relative parameter step (see above)
const MAXITER = 1000     # the same iteration limit for every method

BLAS.set_num_threads(1)  # Julia's threads already use every core; avoid oversubscription

# --- The data saved by examples/sir_wm_4point.jl -------------------------------------------------
const DIR = joinpath(@__DIR__, "..", "examples", "output", "sir_wm_4point")
isfile(joinpath(DIR, "data.csv")) || error("no data in $DIR: run examples/sir_wm_4point.jl first")

readcsv(name) = readdlm(joinpath(DIR, name), ',', Float64; header = true)   # (values, header)
Y, _ = readcsv("data.csv")                         # one row per voxel, one column per observation
prot, _ = readcsv("protocol.csv")                  # columns TI, TD (s)
tvals, tnames = readcsv("truth.csv")
θtrue = NamedTuple{Tuple(Symbol.(vec(tnames)))}(Tuple(tvals))   # (PSR = 0.12, R1f = 1.0, ...)
m = QMTSIR(TI = prot[:, 1], TD = prot[:, 2])

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

methods = [("EQUIP", () -> fit_equip(m, Y, P0)),
           ("LsqFit (Levenberg-Marquardt)", () -> fit_lsqfit(m, Y, P0, lower, upper)),
           ("NonlinearSolve LevenbergMarquardt",
            () -> fit_sciml(m, Y, P0, LevenbergMarquardt(autodiff = AutoForwardDiff()))),
           ("NonlinearSolve TrustRegion",
            () -> fit_sciml(m, Y, P0, TrustRegion(autodiff = AutoForwardDiff())))]

# --- Run: each method once (compiles it, and gives the estimates), then RUNS timed runs ----------
nvox = size(Y, 1)
println("Julia ", VERSION, ", ", Threads.nthreads(), " threads, ", Sys.cpu_info()[1].model)
println("QMTSIR, SIR_4POINT, ", nvox, " voxels, ", BOUNDS ? "with bounds" : "no bounds",
        ", stop at relative step ", XTOL, ", ", RUNS, " runs per method")
println("truth: ", θtrue, "\n")
results = map(methods) do (name, run)
    est, ok, steps = run()
    # exactly RUNS runs: the time limit (seconds) is set far beyond what they need
    b = @benchmark $run() samples = RUNS evals = 1 seconds = 1e6
    @printf("  %-36s median %8.4f s per run\n", name, median(b.times) / 1e9)
    (; name, est, ok, steps, time = median(b.times) / 1e9, tmin = minimum(b.times) / 1e9,
     memory = b.memory)
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

# --- Estimates: do the methods find the same answers? --------------------------------------------
ref = results[1].est
println("\nEstimates: mean ± SD over voxels (truth: ",
        join(("$n = $(θtrue[n])" for n in NAMES), ", "), ")")
@printf("  %-36s %10s %18s %18s %18s %18s %10s\n", "method", "converged", "PSR", "R1f", "Sf", "M0f",
        "max diff")
for r in results
    cells = [@sprintf("%.4f ± %.4f", mean(r.est[j, :]), std(r.est[j, :])) for j in eachindex(NAMES)]
    rel = maximum(abs.(r.est .- ref) ./ abs.(ref))          # worst voxel and parameter vs EQUIP
    @printf("  %-36s %9.1f%% %18s %18s %18s %18s %10.1e\n", r.name, 100 * mean(r.ok), cells..., rel)
end
println("  (max diff: largest relative difference from EQUIP's estimate, any voxel or parameter)")
