# A whole-curve model: extended Tofts DCE-MRI.
#
#   Ct(t) = vp·Cp(t) + Ktrans ∫₀ᵗ Cp(τ) exp(-kep (t - τ)) dτ
#
# Evaluated point by point the convolution costs O(m) per time point, O(m²) per curve. As an
# AbstractCurveModel it is computed for all time points in one O(m) recursive pass. curve!
# receives θ as dual numbers, so derivatives still come from automatic differentiation.
#
#   julia -t auto --project=. examples/dce_tofts_curve_model.jl
using VoxelFit, StaticArrays, Random

struct ExtendedTofts <: AbstractCurveModel{3}
    aif::Vector{Float64}      # plasma concentration Cp at the sample times (curve models run on
end                           # the CPU, so array fields are fine here)

VoxelFit.paramnames(::ExtendedTofts) = (:Ktrans, :kep, :vp)
VoxelFit.lower_bounds(::ExtendedTofts) = (0.0, 1e-4, 0.0)
VoxelFit.upper_bounds(::ExtendedTofts) = (5.0, 10.0, 1.0)
VoxelFit.initial_guess(::ExtendedTofts, t, y) = SVector(0.1, 0.5, 0.02)

function VoxelFit.curve!(out, m::ExtendedTofts, t, θ)
    Ktrans, kep, vp = θ[1], θ[2], θ[3]
    cp = m.aif
    conv = zero(Ktrans)
    out[1] = vp * cp[1]
    for i in 2:length(t)
        dt = t[i] - t[i-1]
        E = exp(-kep * dt)
        # trapezoidal update of the exponentially weighted convolution
        conv = conv * E + Ktrans * dt / 2 * (cp[i] + cp[i-1] * E)
        out[i] = vp * cp[i] + conv
    end
    return out
end

# Parker-like population AIF (bi-exponential bolus), time in minutes
t = collect(range(0, 6; length=80))
aif = [ti < 0.5 ? 0.0 : 6.0 * (ti - 0.5) * exp(-4 * (ti - 0.5)) + 1.0 * (1 - exp(-3 * (ti - 0.5))) * exp(-0.1 * (ti - 0.5))
       for ti in t]
model = ExtendedTofts(aif)

# Synthetic 64×64 slice
Random.seed!(2)
n = 64
truth = (Ktrans=0.05 .+ 0.25 .* rand(n, n), kep=0.2 .+ 1.0 .* rand(n, n), vp=0.01 .+ 0.05 .* rand(n, n))
data = zeros(n, n, length(t))
buf = zeros(length(t))
for I in CartesianIndices((n, n))
    VoxelFit.curve!(buf, model, t, SVector(truth.Ktrans[I], truth.kep[I], truth.vp[I]))
    data[I, :] .= buf .+ 0.005 .* randn(length(t))
end

fit_maps(model, t, data[1:2, 1:2, :])                    # compile
el = @elapsed res = fit_maps(model, t, data)
println(res, "\n  ", n * n, " voxels in ", round(el; digits=3), " s on ", Threads.nthreads(), " threads")
for p in (:Ktrans, :kep, :vp)
    err = sort(vec(abs.(res[p] .- truth[p]) ./ truth[p]))[end ÷ 2]
    println("  median relative error ", rpad(p, 7), round(100err; digits=2), " %")
end
