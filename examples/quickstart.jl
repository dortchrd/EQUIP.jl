# Quick tour of VoxelFit.  Run with threads:  julia -t auto --project=. examples/quickstart.jl
using VoxelFit, StaticArrays, Random

Random.seed!(1)

# --- 1. A T2 map from multi-echo data (X × Y × Z × echoes, as read from NIfTI/MATLAB) -----------
te = collect(10.0:10.0:320.0)                              # ms
nx, ny, nz = 128, 128, 32
T2 = 40 .+ 80 .* rand(Float32, nx, ny, nz)
S0 = 800 .+ 400 .* rand(Float32, nx, ny, nz)
data = S0 .* exp.(-reshape(Float32.(te), 1, 1, 1, :) ./ T2) .+ 5 .* randn(Float32, nx, ny, nz, length(te))
mask = S0 .> 850                                           # skip "background"

res = fit_maps(MonoExp(), te, data; mask, precision=Float32, stderr=true)
t = @elapsed fit_maps(MonoExp(), te, data; mask, precision=Float32)
println(res)
println("  $(count(mask)) voxels in $(round(t; digits=2)) s on $(Threads.nthreads()) threads")
t2map = res[:T2]                                           # nx × ny × nz view
println("  median |T2 error| = ", round(sort(abs.(t2map[mask] .- T2[mask]))[end ÷ 2]; digits=2), " ms")

# --- 2. A custom model as a plain function ------------------------------------------------------
# Stretched exponential S = S0·exp(-(t/τ)^β); θ arrives as an SVector of (possibly dual) numbers.
stretched = PointModel((t, θ) -> θ[1] * exp(-(t / θ[2])^θ[3]), (:S0, :tau, :beta);
                       p0=(1000, 60, 1), lower=(0, 1e-3, 0.1), upper=(Inf, Inf, 2))
y = [stretched(t, SVector(1000.0, 70.0, 0.8)) + 2randn() for t in te]
r = fit_voxel(stretched, te, y)
println("\nstretched exponential: θ = ", round.(r.θ; sigdigits=4), "  status = ", r.status)

# --- 3. The same from a string (what the CLI/Python/MATLAB front ends use) -----------------------
m = expr_model("S0 * exp(-(TE / tau)^beta)"; params=(:S0, :tau, :beta), x=:TE,
               p0=(1000, 60, 1), lower=(0, 1e-3, 0.1))
r = Base.invokelatest(fit_voxel, m, te, y)   # invokelatest: the function was just compiled
println("expr_model:            θ = ", round.(r.θ; sigdigits=4))

# --- 4. Per-voxel covariates: variable flip angle T1 with a B1 map --------------------------------
fa = [2, 5, 10, 15, 20, 30.0]                              # degrees
spgr = SPGR(TR=15.0)                                       # ms
B1 = 0.8 .+ 0.4 .* rand(64, 64)
T1 = 600 .+ 1000 .* rand(64, 64)
sig = [spgr(fa[k], SVector(5000.0, T1[I]), SVector(B1[I])) + 3randn() for I in CartesianIndices(B1), k in eachindex(fa)]
with = fit_maps(spgr, fa, sig; covariates=B1)
without = fit_maps(spgr, fa, sig)
relerr(a) = round(100 * sort(vec(abs.(a .- T1) ./ T1))[end ÷ 2]; digits=2)
println("\nVFA T1 median error: $(relerr(with[:T1]))% with B1, $(relerr(without[:T1]))% without")

# --- 5. GPU ---------------------------------------------------------------------------------------
# using CUDA                       # or AMDGPU / Metal / oneAPI
# res = fit_maps(MonoExp(), te, data; mask, precision=Float32, backend=CUDABackend())
