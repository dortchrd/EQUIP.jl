# Quick tour of VoxelFit.  Run with threads:  julia -t auto --project=. examples/quickstart.jl
using VoxelFit, Random

Random.seed!(1)

# --- 1. A T2 map from multi-echo data (X × Y × Z × echoes, as read from NIfTI/MATLAB) ----------
te = collect(10.0:10.0:320.0)                              # ms; a vector works for one-setting models
nx, ny, nz = 128, 128, 32
T2 = 40 .+ 80 .* rand(Float32, nx, ny, nz)
S0 = 800 .+ 400 .* rand(Float32, nx, ny, nz)
data = S0 .* exp.(-reshape(Float32.(te), 1, 1, 1, :) ./ T2) .+ 5 .* randn(Float32, nx, ny, nz, length(te))
mask = S0 .> 850                                           # skip "background"

res = fit_maps(MonoExp(), te, data; mask, precision=Float32, stderr=true)
t = @elapsed fit_maps(MonoExp(), te, data; mask, precision=Float32)
println(res)
println("  $(count(mask)) voxels in $(round(t; digits=2)) s on $(Threads.nthreads()) threads")
println("  median |T2 error| = ", round(sort(abs.(res[:T2][mask] .- T2[mask]))[end ÷ 2]; digits=2), " ms")

# --- 2. SIR-FSE qMT: kmf assumed vs estimated, and a single voxel with a reusable cache ----------
acq = (TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])  # s
truth = (pmf = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 15.0, R1m = 1.0, Sm = 0.83)
y = signal(SIRFSE(), acq, truth) .+ 2 .* randn(5)
cache = init(SIRFSE(), acq)                                # kmf fixed at 12.5 s⁻¹ by default
println("\nSIR-FSE, kmf fixed at 12.5: ", fit!(cache, y).params)
println("SIR-FSE, kmf estimated:     ", fit!(init(SIRFSE(), acq; free = (:kmf,)), y).params)
println("SIR-FSE, kmf fixed at 15:   ", fit!(init(SIRFSE(), acq; fixed = (kmf = 15.0,)), y).params)

# --- 3. A custom model from a string, with two settings per observation ---------------------------
m = expr_model("S0 * exp(-TE / T2) * (1 - exp(-TR / T1))"; protocol = (:TE, :TR),
               params = (S0 = Estimate(900.0; lower=0), T2 = Estimate(50.0; lower=0), T1 = Fixed(1200.0)))
acq2 = (TE = repeat([10.0, 30.0, 60.0, 90.0], inner=4), TR = repeat([300.0, 800.0, 2000.0, 5000.0], 4))
y2 = Base.invokelatest(signal, m, acq2, (S0 = 1000.0, T2 = 70.0, T1 = 1200.0)) .+ randn(16)
r = Base.invokelatest(() -> fit!(init(m, acq2), y2))      # invokelatest: the model was just compiled
println("\nexpr_model with T1 fixed:   ", r.params)

# --- 4. A per-voxel map: variable flip angle T1 with a B1 map -------------------------------------
vfa = (FA = [2, 5, 10, 15, 20, 30.0], TR = 15.0)           # degrees, ms
B1 = 0.8 .+ 0.4 .* rand(64, 64)
T1 = 600 .+ 1000 .* rand(64, 64)
sig = similar(B1, 64, 64, 6)
for I in CartesianIndices(B1)
    sig[I, :] .= signal(SPGR(), vfa, (M0 = 5000.0, T1 = T1[I], B1 = B1[I])) .+ 3 .* randn(6)
end
with = fit_maps(SPGR(), vfa, sig; fixed = (B1 = B1,))
without = fit_maps(SPGR(), vfa, sig)
relerr(a) = round(100 * sort(vec(abs.(a .- T1) ./ T1))[end ÷ 2]; digits=2)
println("\nVFA T1 median error: $(relerr(with[:T1]))% with the B1 map, $(relerr(without[:T1]))% without")
