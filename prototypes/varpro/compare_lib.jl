# Full LM (EQUIP today) vs VarPro (linear parameters solved inside every evaluation), from the same
# starting points, on noisy data. Single thread, solver time only (starts computed beforehand by
# EQUIP's own start pipeline, `fit(...; solver = LM(maxiter = 0))`, so both methods start from
# exactly the same nonlinear values). "reached min": the final SSE is within 1e-6 (relative) of
# the lowest SSE any method found for that voxel, from any start.
include(joinpath(@__DIR__, "vp.jl"))
using Random, Statistics, Printf, StaticArrays

const RELTOL = 1e-6

# The timed loops, behind function barriers: the names arrive as Val{F}, so each loop compiles
# for them (no dynamic dispatch per voxel), as in EQUIP's own dataset loop.
function loop_full!(out, ::Val{F}, cache, Y, θ0, y) where {F}
    spec = cache.spec
    for v in eachindex(out)
        copyto!(y, view(Y, v, :))
        out[v] = E.levmar(E.VoxelProblem{F}(cache.model, y, spec.fixed), θ0[v], spec.lower, spec.upper, cache.opts, cache.buf)
    end
end
function loop_vp!(out, ::Val{Fn}, vp, fixed, keep, lo, hi, opts, Y, θ0, y, grid) where {Fn}
    for v in eachindex(out)
        copyto!(y, view(Y, v, :))
        prob = E.VoxelProblem{Fn}(vp, y, fixed)
        t0 = θ0[v][keep]
        if grid !== nothing                                             # best pair on the grid
            best = Inf
            for g in grid
                sse, = E.normal_equations(prob, g, nothing)
                sse < best && (best = sse; t0 = g)
            end
        end
        out[v] = E.levmar(prob, t0, lo, hi, opts, vp.cols[1])
    end
end

function solve_all(m, Y, kw, cache, vs; grid = nothing)
    spec = cache.spec; F = E.freenames(spec); P = length(F)
    S = fit(m, Y; solver = LM(maxiter = 0), kw...).params              # EQUIP's starts
    n, M = size(Y)
    θ0 = [SVector{P,Float64}(ntuple(k -> S[F[k]][v], P)) for v in 1:n]
    y = zeros(M)
    copyto!(y, view(Y, 1, :))                                            # one fit each, for the result types
    r1 = E.levmar(E.VoxelProblem{F}(cache.model, y, spec.fixed), θ0[1], spec.lower, spec.upper, cache.opts, cache.buf)
    r2 = E.levmar(E.VoxelProblem{vs.Fn}(vs.vp, y, vs.fixed), θ0[1][vs.keep], vs.lo, vs.hi, cache.opts, vs.vp.cols[1])
    full = Vector{typeof(r1)}(undef, n); vpr = Vector{typeof(r2)}(undef, n)
    runfull() = loop_full!(full, Val(F), cache, Y, θ0, y)
    runvp() = loop_vp!(vpr, Val(vs.Fn), vs.vp, vs.fixed, vs.keep, vs.lo, vs.hi, cache.opts, Y, θ0, y, grid)
    runfull(); runvp()                                                  # compile
    tf = minimum(@elapsed(runfull()) for _ in 1:3) / n * 1e6
    tv = minimum(@elapsed(runvp()) for _ in 1:3) / n * 1e6
    return (; full, vp = vpr, tf, tv)
end

conv(r) = r.status in (EQUIP.ConvergedX, EQUIP.ConvergedF, EQUIP.ConvergedG)

function run_case(title, m, truth; snr, n = 2000, linear, units, nonneg, starts, grid = nothing, seed = 1)
    println("\n=== ", title)
    Y = simulate(m, truth; snr, n, rng = MersenneTwister(seed))
    cache = init(m)
    vs = vp_setup(cache, linear, units, nonneg)
    runs = [(sname, solve_all(m, Y, kw, cache, vs; grid = (sname == "grid" ? grid : nothing))) for (sname, kw) in starts]
    ref = [minimum(min(r.full[v].sse, r.vp[v].sse) for (_, r) in runs) for v in 1:n]
    hit(rs) = 100 * count(v -> rs[v].sse <= ref[v] * (1 + RELTOL) + 1e-12, 1:n) / n
    @printf("  %-26s %-6s %9s %10s %10s %11s  %s\n", "start", "method", "reached", "median it", "mean it", "µs/voxel", "status: conv / maxiter / other")
    for (sname, r) in runs
        for (meth, rs, t) in (("full", r.full, r.tf), ("VarPro", r.vp, r.tv))
            sname == "grid" && meth == "full" && continue
            its = [x.iterations for x in rs]
            nc = count(conv, rs); nm = count(x -> x.status == EQUIP.MaxIterations, rs)
            @printf("  %-26s %-6s %8.1f%% %10.0f %10.1f %11.2f  %d / %d / %d\n", sname, meth, hit(rs), median(its), mean(its), t, nc, nm, n - nc - nm)
        end
    end
    return runs
end

