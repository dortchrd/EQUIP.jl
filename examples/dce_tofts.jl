# A whole-curve model: extended Tofts DCE-MRI.
#
#   Ct(t) = vp·Cp(t) + Ktrans ∫₀ᵗ Cp(τ) exp(-kep (t - τ)) dτ
#
# signal! computes all time points in one O(m) recursive pass over the convolution (point by
# point it would be O(m²)). The arterial input function is part of the protocol, since it is the
# same for every voxel. Derivatives still come from automatic differentiation.
#
#   julia -t auto --project=. examples/dce_tofts.jl
using VoxelFit, Random

struct ExtendedTofts <: AbstractModel end
VoxelFit.parameters(::ExtendedTofts) = (Ktrans = Estimate(0.1; lower=0, upper=5),
                                        kep = Estimate(0.5; lower=1e-4, upper=10),
                                        vp = Estimate(0.02; lower=0, upper=1))
VoxelFit.protocol(::ExtendedTofts) = (:t, :Cp)

function VoxelFit.signal!(out, ::ExtendedTofts, acq, p)
    (; Ktrans, kep, vp) = p
    t, cp = acq.t, acq.Cp
    conv = zero(Ktrans)
    out[1] = vp * cp[1]
    for i in 2:length(t)
        dt = t[i] - t[i-1]
        E = exp(-kep * dt)
        conv = conv * E + Ktrans * dt / 2 * (cp[i] + cp[i-1] * E)   # trapezoidal update
        out[i] = vp * cp[i] + conv
    end
    return out
end

# Parker-like population AIF (bi-exponential bolus), time in minutes
t = collect(range(0, 6; length=80))
Cp = [ti < 0.5 ? 0.0 : 6.0 * (ti - 0.5) * exp(-4 * (ti - 0.5)) + 1.0 * (1 - exp(-3 * (ti - 0.5))) * exp(-0.1 * (ti - 0.5))
      for ti in t]
acq = (t = t, Cp = Cp)

# Synthetic 64×64 slice
Random.seed!(2)
n = 64
truth = (Ktrans = 0.05 .+ 0.25 .* rand(n, n), kep = 0.2 .+ 1.0 .* rand(n, n), vp = 0.01 .+ 0.05 .* rand(n, n))
data = zeros(n, n, length(t))
for I in CartesianIndices((n, n))
    data[I, :] .= signal(ExtendedTofts(), acq, (Ktrans = truth.Ktrans[I], kep = truth.kep[I], vp = truth.vp[I])) .+
                  0.005 .* randn(length(t))
end

fit_maps(ExtendedTofts(), acq, data[1:2, 1:2, :])           # compile
el = @elapsed res = fit_maps(ExtendedTofts(), acq, data)
println(res, "\n  ", n * n, " voxels in ", round(el; digits=3), " s on ", Threads.nthreads(), " threads")
for p in (:Ktrans, :kep, :vp)
    err = sort(vec(abs.(res[p] .- truth[p]) ./ truth[p]))[end ÷ 2]
    println("  median relative error ", rpad(p, 7), round(100err; digits=2), " %")
end
