# =================================================================================================
# VarPro on the benchmark's biexponential cases (bench/cases.jl: :biexp at SNR 200,
# :biexp_snr100), without bounds and with the model's bounds, against EQUIP's full fit and the
# reference solvers (LsqFit, FastLevenbergMarquardt, and without bounds NonlinearSolve's
# TrustRegion). Same starts and stopping rule as bench/solvers.jl.
#
# VarPro here: LM sees only T2short and T2long; the two amplitudes (short and long component,
# S0·fshort and S0·(1 - fshort)) are solved exactly at every evaluation, from columns the model
# supplies (exp(-TE/T2short), exp(-TE/T2long)). With bounds, the amplitudes are kept ≥ 0 (so
# S0 ≥ 0 and 0 ≤ fshort ≤ 1, as EQUIP's bounds) and the T2s ≥ EQUIP's lower bound; without
# bounds, neither.
#
# Run from the repository root: julia -t auto --project=bench prototypes/varpro/biexp_bench.jl
# =================================================================================================
include(joinpath(@__DIR__, "vp.jl"))
include(joinpath(@__DIR__, "..", "..", "bench", "cases.jl"))
using LsqFit, NonlinearSolve, FastLevenbergMarquardt, ForwardDiff, LinearAlgebra, Statistics, Printf
BLAS.set_num_threads(1)

const NVOXELS = 1_000
const BENCH_LM = LM(maxiter = 1000, xtol = 1e-8, ftol = 0.0, gtol = 0.0)   # bench/solvers.jl's rule
const F = (:S0, :fshort, :T2short, :T2long)

# The biexponential's two columns, computed by the model's own terms only
function col_short(c, m, p)
    @inbounds for j in eachindex(c, m.TE); c[j] = exp(-m.TE[j] / p.T2short); end
    return c
end
function col_long(c, m, p)
    @inbounds for j in eachindex(c, m.TE); c[j] = exp(-m.TE[j] / p.T2long); end
    return c
end

sig(m, p) = signal(m, (S0 = p[1], fshort = p[2], T2short = p[3], T2long = p[4]))
sse(m, p, y) = all(isfinite, p) ? sum(abs2, sig(m, p) .- y) : Inf

# VarPro over all voxels, threaded: one VP (with its own column buffers) per block of voxels
function fit_varpro(m, Y, P0, lower, upper, bounded)
    n = size(Y, 1)
    est = fill(NaN, 4, n)
    its = zeros(Int, n)
    ok = falses(n)
    roles = NamedTuple{F}(ntuple(j -> Free(lower = lower[j], upper = upper[j]), 4))
    blocks = Iterators.partition(1:n, cld(n, Threads.nthreads()))
    @sync for vs in blocks
        Threads.@spawn begin
            cache = init(m; solver = BENCH_LM, roles...)
            v = vp_setup(cache, (:S0, :fshort), (col_short, col_long), (bounded, bounded))
            y = zeros(size(Y, 2))
            for i in vs
                y .= view(Y, i, :)
                prob = E.VoxelProblem{v.Fn}(v.vp, y, v.fixed)
                r = E.levmar(prob, SVector(P0[3, i], P0[4, i]), v.lo, v.hi, cache.opts, v.vp.cols[1])
                a1, a2 = linear_coeffs(prob, r.θ)          # short and long amplitudes
                est[:, i] .= (a1 + a2, a1 / (a1 + a2), r.θ[1], r.θ[2])
                its[i] = r.iterations
                ok[i] = isconverged(r.status)
            end
        end
    end
    return est, its, ok
end

function fit_full(m, Y, P0, lower, upper)
    roles = NamedTuple{F}(ntuple(j -> Free(P0[j, :]; lower = lower[j], upper = upper[j]), 4))
    r = fit(m, Y; solver = BENCH_LM, roles...)
    return permutedims(reduce(hcat, [r.params[k] for k in F])), r.iterations, isconverged(r)
end

function references(m, Y, P0, lower, upper, bounded)
    n, M = size(Y)
    f!(o, x, p) = (o .= sig(m, p); o)
    kw = bounded ? (; lower, upper) : (;)
    out = [("LsqFit", reduce(hcat, [try coef(curve_fit(f!, collect(1.0:M), Y[v, :], P0[:, v]; inplace = true,
                                                       autodiff = AutoForwardDiff(), x_tol = 1e-8, g_tol = 0.0,
                                                       maxIter = 1000, kw...))
                                    catch; fill(NaN, 4) end for v in 1:n]))]
    resid(p, y) = SVector{32}(sig(m, p)) - y
    jac(p, y) = ForwardDiff.jacobian(q -> resid(q, y), p)
    lb, ub = bounded ? (SVector{4}(lower), SVector{4}(upper)) : (nothing, nothing)
    push!(out, ("FastLM", reduce(hcat, [try collect(lmsolve(resid, jac, SVector{4}(P0[:, v]), SVector{32}(Y[v, :]), lb, ub;
                                                            xtol = 1e-8, ftol = 0.0, gtol = 0.0, maxit = 1000)[1])
                                        catch; fill(NaN, 4) end for v in 1:n])))
    if !bounded
        r!(o, p, y) = (o .= sig(m, p) .- y; nothing)
        y1 = Y[1, :]
        cache = init(NonlinearLeastSquaresProblem(NonlinearFunction(r!; resid_prototype = zeros(M)), P0[:, 1], y1),
                     TrustRegion(autodiff = AutoForwardDiff()); maxiters = 1000)
        push!(out, ("TrustRegion", reduce(hcat, [try y1 .= Y[v, :]; reinit!(cache, P0[:, v]; p = y1); collect(solve!(cache).u)
                                                 catch; fill(NaN, 4) end for v in 1:n])))
    end
    return out
end

println("Julia ", VERSION, ", ", Threads.nthreads(), " threads; ", NVOXELS, " voxels per case")
for name in (:biexp, :biexp_snr100), bounded in (false, true)
    c = makecase(name; n = NVOXELS)
    m, Y, truth = c.m, c.Y, c.truth
    spec = init(m).spec
    lower = bounded ? collect(spec.lower) : fill(-Inf, 4)
    upper = bounded ? collect(spec.upper) : fill(Inf, 4)
    st = fit(m, Y; solver = LM(maxiter = 0), NamedTuple{F}(ntuple(j -> Free(lower = lower[j], upper = upper[j]), 4))...).params
    P0 = clamp.(permutedims(reduce(hcat, [st[k] for k in F])), lower, upper)

    full, fits, fok = fit_full(m, Y, P0, lower, upper)
    vp, vits, vok = fit_varpro(m, Y, P0, lower, upper, bounded)
    tf = minimum(@elapsed(fit_full(m, Y, P0, lower, upper)) for _ in 1:3)
    tv = minimum(@elapsed(fit_varpro(m, Y, P0, lower, upper, bounded)) for _ in 1:3)
    refs = references(m, Y, P0, lower, upper, bounded)
    rows = [("EQUIP full", full, mean(fits), mean(fok), 1e6 * tf / NVOXELS),
            ("EQUIP VarPro", vp, mean(vits), mean(vok), 1e6 * tv / NVOXELS),
            [(lab, R, NaN, NaN, NaN) for (lab, R) in refs]...]
    for r in rows; biexp_order!(r[2], collect(F)); end
    s = [[sse(m, r[2][:, v], Y[v, :]) for v in 1:NVOXELS] for r in rows]
    best = [minimum(x[v] for x in s) for v in 1:NVOXELS]
    tol = 1e-8 .* vec(sum(abs2, Y; dims = 2))
    println("\n=== ", c.title, bounded ? ", with bounds" : ", without bounds")
    @printf("  %-14s %9s %8s %8s %10s %12s %12s %12s\n", "method", "not best", "steps", "conv", "µs/voxel",
            "fshort err", "T2short err", "T2long err")
    for (k, r) in enumerate(rows)
        err(j, n) = 100 * median(filter(isfinite, abs.(r[2][j, :] .- truth[n]) ./ truth[n]))
        @printf("  %-14s %9d %8s %8s %10s %11.1f%% %11.1f%% %11.1f%%\n", r[1], count(s[k] .> best .+ tol),
                isnan(r[3]) ? "—" : @sprintf("%.1f", r[3]), isnan(r[4]) ? "—" : @sprintf("%.1f%%", 100r[4]),
                isnan(r[5]) ? "—" : @sprintf("%.2f", r[5]), err(2, :fshort), err(3, :T2short), err(4, :T2long))
    end
end
