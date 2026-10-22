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
#   - the starting values: EQUIP's per-voxel starts, computed once and given to the others for
#     free (EQUIP's timed fit computes its own starts, so its time includes that work)
#   - the bounds: the same for EQUIP and LsqFit; NonlinearSolve's least-squares solvers take none
#     (here they would rarely be reached)
#   - the threads: every method spreads the voxels over all of Julia's threads
#   - derivatives: automatic (dual numbers, ForwardDiff) for all three
#
# Run, from the repository folder:
#   julia --project=bench -e 'using Pkg; Pkg.instantiate()'    # once: installs LsqFit, SciML, ...
#   julia -t auto --project=bench bench/sir_wm_4point_vs.jl
# With -t 1 instead of -t auto, the same comparison on one thread.
# =================================================================================================

using EQUIP, LsqFit, NonlinearSolve, BenchmarkTools
using DelimitedFiles, LinearAlgebra, Statistics, Printf

BLAS.set_num_threads(1)           # Julia's threads already use every core; avoid oversubscription
const SECONDS = 10                # time budget per method for BenchmarkTools (at least 1 run each)

# --- The data saved by examples/sir_wm_4point.jl -------------------------------------------------
const DIR = joinpath(@__DIR__, "..", "examples", "output", "sir_wm_4point")
isfile(joinpath(DIR, "data.csv")) || error("no data in $DIR: run examples/sir_wm_4point.jl first")

readcsv(name) = readdlm(joinpath(DIR, name), ',', Float64; header = true)   # (values, header)
Y, _ = readcsv("data.csv")                         # one row per voxel, one column per observation
prot, _ = readcsv("protocol.csv")                  # columns TI, TD (s)
tvals, tnames = readcsv("truth.csv")
θtrue = NamedTuple{Tuple(Symbol.(vec(tnames)))}(Tuple(tvals))   # (PSR = 0.12, R1f = 1.0, ...)
m = QMTSIR(TI = prot[:, 1], TD = prot[:, 2])

# --- Starting values and bounds, shared by every method ------------------------------------------
const NAMES = (:PSR, :R1f, :Sf, :M0f)               # the free parameters, in this order everywhere
starts = fit(m, Y; solver = LM(maxiter = 0)).params  # EQUIP's starts: 0 iterations = no fitting
P0 = permutedims(reduce(hcat, [starts[n] for n in NAMES]))   # 4 × voxels: column v = voxel v
roles = parameters(m)
lower = [something(roles[n].lower, -Inf) for n in NAMES]     # PSR ≥ 0, R1f ≥ 0.05, Sf ≥ -1.2, M0f ≥ 0
upper = [something(roles[n].upper, Inf) for n in NAMES]      # PSR ≤ 1, R1f ≤ 10, Sf ≤ 0

# The signal for LsqFit and NonlinearSolve: EQUIP's QMTSIR equation, parameters in a plain vector
# p = [PSR, R1f, Sf, M0f], the fixed ones written out. p may hold dual numbers (for derivatives).
sirparams(p) = (PSR = p[1], R1f = p[2], Sf = p[3], M0f = p[4],
                kmf = 12.5, R1m = p[2], Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)

# --- Each method: fit every voxel, return the estimates (4 × voxels) and which converged ---------
# Each is a function, so the model and data are local variables, not globals: Julia compiles fast
# code for locals, while every use of a non-constant global is looked up at run time.

function fit_equip(m, Y)
    r = fit(m, Y)                                   # threads, starts and bounds are built in
    return permutedims(reduce(hcat, [r.params[n] for n in NAMES])), isconverged(r)
end

function fit_lsqfit(m, Y, P0, lower, upper)
    f!(F, x, p) = signal!(F, m, sirparams(p))       # LsqFit's in-place model; x is not used
    est = similar(P0)
    ok = falses(size(Y, 1))
    Threads.@threads for v in axes(Y, 1)
        r = curve_fit(f!, m.TI, Y[v, :], P0[:, v]; inplace = true, lower, upper)
        est[:, v] .= coef(r)
        ok[v] = r.converged
    end
    return est, ok
end

function fit_sciml(m, Y, P0, alg)
    resid!(r, p, y) = (signal!(r, m, sirparams(p)); r .-= y; nothing)   # residuals, in place
    est = similar(P0)
    ok = falses(size(Y, 1))
    # One block of voxels per thread, each with its own solver cache, reused voxel after voxel
    # (`init` once, then `reinit!` with the next start and data: SciML's fast pattern).
    blocks = Iterators.partition(axes(Y, 1), cld(size(Y, 1), Threads.nthreads()))
    @sync for vs in blocks
        Threads.@spawn begin
            y = Y[first(vs), :]
            f = NonlinearFunction(resid!; resid_prototype = zeros(size(Y, 2)))
            cache = init(NonlinearLeastSquaresProblem(f, P0[:, first(vs)], y), alg)
            for v in vs
                y .= view(Y, v, :)
                reinit!(cache, P0[:, v]; p = y)
                sol = solve!(cache)
                est[:, v] .= sol.u
                ok[v] = NonlinearSolve.SciMLBase.successful_retcode(sol)
            end
        end
    end
    return est, ok
end

methods = [("EQUIP", () -> fit_equip(m, Y)),
           ("LsqFit (Levenberg-Marquardt)", () -> fit_lsqfit(m, Y, P0, lower, upper)),
           ("NonlinearSolve LevenbergMarquardt", () -> fit_sciml(m, Y, P0, LevenbergMarquardt())),
           ("NonlinearSolve TrustRegion", () -> fit_sciml(m, Y, P0, TrustRegion()))]

# --- Run: each method once (compiles it, and gives the estimates), then the timing ---------------
nvox = size(Y, 1)
println("Julia ", VERSION, ", ", Threads.nthreads(), " threads, ", Sys.cpu_info()[1].model)
println("QMTSIR, SIR_4POINT, ", nvox, " voxels; truth: ", θtrue, "\n")
results = map(methods) do (name, run)
    est, ok = run()
    b = @benchmark $run() seconds = SECONDS evals = 1
    @printf("  %-34s %3d runs, median %8.4f s\n", name, length(b.times), median(b.times) / 1e9)
    (; name, est, ok, time = median(b.times) / 1e9, memory = b.memory)
end

# --- Speed ---------------------------------------------------------------------------------------
t_equip = results[1].time
println("\nSpeed: fitting all ", nvox, " voxels (BenchmarkTools median)")
@printf("  %-34s %10s %12s %12s %16s\n", "method", "time (s)", "µs/voxel", "memory", "EQUIP faster by")
for r in results
    @printf("  %-34s %10.4f %12.2f %9.1f MB %15.0f×\n", r.name, r.time, 1e6 * r.time / nvox,
            r.memory / 2^20, r.time / t_equip)
end

# --- Estimates: do the methods find the same answers? --------------------------------------------
ref = results[1].est
println("\nEstimates: mean ± SD over voxels (truth: ",
        join(("$n = $(θtrue[n])" for n in NAMES), ", "), ")")
@printf("  %-34s %10s %18s %18s %18s %18s %12s\n", "method", "converged", "PSR", "R1f", "Sf", "M0f",
        "max diff")
for r in results
    cells = [@sprintf("%.4f ± %.4f", mean(r.est[j, :]), std(r.est[j, :])) for j in eachindex(NAMES)]
    rel = maximum(abs.(r.est .- ref) ./ abs.(ref))          # worst voxel and parameter vs EQUIP
    @printf("  %-34s %9.1f%% %18s %18s %18s %18s %12.1e\n", r.name, 100 * mean(r.ok), cells..., rel)
end
println("  (max diff: the largest relative difference from EQUIP's estimate, any voxel or parameter)")
