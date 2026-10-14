# Benchmarks: VoxelFit vs LsqFit.jl vs NonlinearSolve.jl (SciML) on typical MRI fits.
#
#   julia -t auto --project=bench bench/benchmarks.jl
#
# Each solver fits the same 2000 noisy voxels from the same start (VoxelFit's data-driven
# initial guess, so the comparison is about the solver). Reported: mean time and heap
# allocation per voxel on one thread, agreement of the estimates with VoxelFit, and
# whole-map throughput with all threads.

using VoxelFit, LsqFit, NonlinearSolve, StaticArrays, Random, Printf, LinearAlgebra
using VoxelFit: lm_fit, LMOptions, _svec
using Base.Threads

BLAS.set_num_threads(1)
Random.seed!(42)

const NVOX = 2000

struct Problem
    name::String
    model::AbstractModel
    x::Vector{Float64}
    Y::Matrix{Float64}     # nobs × NVOX
    θ0::Matrix{Float64}    # P × NVOX
    lsq!::Function         # in-place model F .= f(x, θ) for LsqFit
    lsqjac!::Function      # in-place Jacobian
    resid!::Function       # NonlinearSolve residual r .= f(x, θ) - y
end

function monoexp_problem()
    x = collect(10.0:10.0:320.0)
    m = MonoExp()
    Y = reduce(hcat, [begin
        θ = SVector(800 + 400rand(), 40 + 80rand())
        [m(t, θ) + 5randn() for t in x]
    end for _ in 1:NVOX])
    θ0 = reduce(hcat, [Vector(initial_guess(m, x, view(Y, :, v))) for v in 1:NVOX])
    f!(F, t, p) = (@. F = p[1] * exp(-t / p[2]); F)
    function j!(J, t, p)
        @. J[:, 1] = exp(-t / p[2])
        @. J[:, 2] = p[1] * exp(-t / p[2]) * t / p[2]^2
        J
    end
    r!(r, p, (t, y)) = (@. r = p[1] * exp(-t / p[2]) - y; nothing)
    Problem("MonoExp (2 params, 32 echoes)", m, x, Y, θ0, f!, j!, r!)
end

function ivim_problem()
    x = [0, 10, 20, 40, 80, 110, 140, 170, 200, 300, 400, 500, 600, 700, 800, 1000.0]
    m = IVIM()
    Y = reduce(hcat, [begin
        θ = SVector(1000 + 200rand(), 0.05 + 0.15rand(), 0.01 + 0.04rand(), 0.6e-3 + 1.2e-3rand())
        [m(b, θ) + 10randn() for b in x]
    end for _ in 1:NVOX])
    θ0 = reduce(hcat, [Vector(initial_guess(m, x, view(Y, :, v))) for v in 1:NVOX])
    f!(F, b, p) = (@. F = p[1] * (p[2] * exp(-b * p[3]) + (1 - p[2]) * exp(-b * p[4])); F)
    function j!(J, b, p)
        @. J[:, 1] = p[2] * exp(-b * p[3]) + (1 - p[2]) * exp(-b * p[4])
        @. J[:, 2] = p[1] * (exp(-b * p[3]) - exp(-b * p[4]))
        @. J[:, 3] = -p[1] * p[2] * b * exp(-b * p[3])
        @. J[:, 4] = -p[1] * (1 - p[2]) * b * exp(-b * p[4])
        J
    end
    r!(r, p, (b, y)) = (@. r = p[1] * (p[2] * exp(-b * p[3]) + (1 - p[2]) * exp(-b * p[4])) - y; nothing)
    Problem("IVIM (4 params, 16 b-values)", m, x, Y, θ0, f!, j!, r!)
end

# --------------------------------------------------------------------------------------------
# One solver over all voxels -> (θ matrix, seconds per voxel, bytes per voxel)

run_voxelfit(p::Problem) = _run_voxelfit(p.model, p.x, p.Y, p.θ0)   # function barrier

function _run_voxelfit(m::AbstractModel{P}, x, Y, θ0) where {P}
    lo, hi = _svec(lower_bounds(m), Val(P), Float64), _svec(upper_bounds(m), Val(P), Float64)
    opts = LMOptions{Float64}()
    out = similar(θ0)
    function go()
        for v in 1:NVOX
            r = lm_fit(m, x, view(Y, :, v), SVector{P}(ntuple(j -> θ0[j, v], Val(P))), lo, hi, opts)
            out[:, v] .= r.θ
        end
    end
    go()
    stats = @timed go()
    return out, stats.time / NVOX, stats.bytes / NVOX
end

function run_lsqfit(p::Problem; analytic::Bool)
    P = size(p.θ0, 1)
    lo = collect(Float64, lower_bounds(p.model))
    hi = collect(Float64, upper_bounds(p.model))
    out = similar(p.θ0)
    function go()
        for v in 1:NVOX
            y = p.Y[:, v]
            fit = analytic ?
                curve_fit(p.lsq!, p.lsqjac!, p.x, y, p.θ0[:, v]; inplace=true, lower=lo, upper=hi) :
                curve_fit(p.lsq!, p.x, y, p.θ0[:, v]; inplace=true, autodiff=:forwarddiff, lower=lo, upper=hi)
            out[:, v] .= fit.param
        end
    end
    go()
    stats = @timed go()
    return out, stats.time / NVOX, stats.bytes / NVOX
end

function run_nonlinearsolve(p::Problem, alg)
    nobs = length(p.x)
    out = similar(p.θ0)
    y = p.Y[:, 1]
    nf = NonlinearFunction(p.resid!; resid_prototype=zeros(nobs))
    prob = NonlinearLeastSquaresProblem(nf, p.θ0[:, 1], (p.x, y))
    # Reuse one cache across voxels (the allocation-minimizing SciML pattern).
    cache = init(prob, alg)
    function go()
        for v in 1:NVOX
            y .= view(p.Y, :, v)
            reinit!(cache, p.θ0[:, v]; p=(p.x, y))
            solve!(cache)
            out[:, v] .= cache.u
        end
    end
    go()
    stats = @timed go()
    return out, stats.time / NVOX, stats.bytes / NVOX
end

# --------------------------------------------------------------------------------------------

function relerr(a, b)
    d = maximum(abs, (a .- b) ./ b; dims=2)   # worst voxel, per parameter
    med = [sort(abs.((a[j, :] .- b[j, :]) ./ b[j, :]))[end÷2] for j in axes(a, 1)]
    return med, vec(d)
end

fmt_us(t) = @sprintf("%.2f", 1e6t)
fmt_b(b) = b < 1 ? "0" : b < 1024 ? @sprintf("%.0f B", b) : @sprintf("%.1f KiB", b / 1024)

lines = String[]
println("Julia $(VERSION), $(nthreads()) threads, $(Sys.cpu_info()[1].model)\n")
push!(lines, "## Single-thread time per voxel ($NVOX voxels, Float64, identical starting points)\n")
for p in (monoexp_problem(), ivim_problem())
    ref, t_vf, b_vf = run_voxelfit(p)
    rows = [("VoxelFit (custom LM)", t_vf, b_vf, "—")]
    for (name, f) in (("LsqFit, ForwardDiff Jacobian", () -> run_lsqfit(p; analytic=false)),
                      ("LsqFit, analytic Jacobian", () -> run_lsqfit(p; analytic=true)),
                      ("NonlinearSolve LevenbergMarquardt, cached", () -> run_nonlinearsolve(p, LevenbergMarquardt())),
                      ("NonlinearSolve TrustRegion, cached", () -> run_nonlinearsolve(p, TrustRegion())))
        θ, t, b = f()
        med, worst = relerr(θ, ref)
        push!(rows, (name, t, b, @sprintf("median %.1e, worst %.1e", maximum(med), maximum(worst))))
    end
    push!(lines, "**$(p.name)**\n")
    push!(lines, "| solver | µs/voxel | heap alloc/voxel | speed-up of VoxelFit | rel. difference vs VoxelFit |")
    push!(lines, "|---|---:|---:|---:|---|")
    for (name, t, b, agree) in rows
        push!(lines, "| $name | $(fmt_us(t)) | $(fmt_b(b)) | $(@sprintf("%.0f×", t / t_vf)) | $agree |")
    end
    push!(lines, "")
end

# Whole-map throughput
nx, ny, nz = 128, 128, 64
te = collect(10.0:10.0:320.0)
T2 = 40 .+ 80 .* rand(Float32, nx, ny, nz)
S0 = 800 .+ 400 .* rand(Float32, nx, ny, nz)
data = S0 .* exp.(-reshape(Float32.(te), 1, 1, 1, :) ./ T2) .+ 5 .* randn(Float32, nx, ny, nz, length(te))
nv = nx * ny * nz
push!(lines, "## Whole map: $(nx)×$(ny)×$(nz) = $nv voxels, 32 echoes, MonoExp\n")
push!(lines, "| configuration | time (s) | µs/voxel | voxels/s |")
push!(lines, "|---|---:|---:|---:|")
for T in (Float64, Float32), nt in unique([1, nthreads()])
    fit_maps(MonoExp(), te, data[1:4, 1:4, 1:2, :]; precision=T, ntasks=nt)
    t = @elapsed fit_maps(MonoExp(), te, data; precision=T, ntasks=nt)
    push!(lines, "| VoxelFit $(T), $nt thread$(nt > 1 ? "s" : "") | $(@sprintf("%.2f", t)) | $(@sprintf("%.3f", 1e6t / nv)) | $(@sprintf("%.2e", nv / t)) |")
end
# LsqFit threaded over a subset (allocation-heavy code scales worse with threads)
let nsub = 100_000, Y = reshape(Float64.(data), nv, :)
    f!(F, t, p) = (@. F = p[1] * exp(-t / p[2]); F)
    function j!(J, t, p)
        @. J[:, 1] = exp(-t / p[2])
        @. J[:, 2] = p[1] * exp(-t / p[2]) * t / p[2]^2
        J
    end
    function go(n)
        @threads for v in 1:n
            y = Y[v, :]
            curve_fit(f!, j!, te, y, Vector(initial_guess(MonoExp(), te, y)); inplace=true,
                      lower=[0.0, 1e-6], upper=[Inf, Inf])
        end
    end
    go(100)
    t = @elapsed go(nsub)
    push!(lines, "| LsqFit (analytic Jacobian), $(nthreads()) threads, first $nsub voxels | $(@sprintf("%.2f", t)) | $(@sprintf("%.3f", 1e6t / nsub)) | $(@sprintf("%.2e", nsub / t)) |")
end

report = join(lines, "\n")
println(report)
open(joinpath(@__DIR__, "results.md"), "w") do io
    println(io, "# Benchmark results\n")
    println(io, "Julia $(VERSION), $(nthreads()) threads, $(Sys.cpu_info()[1].model)\n")
    println(io, report)
end
