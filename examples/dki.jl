# =================================================================================================
# Diffusion kurtosis imaging (DKI), the built-in model: 22 parameters, and the kurtosis indices
# (2026-11-05; the built-in DKI, in DIPY's parameterization, since 2026-11-07)
#
#   S = S0·exp(−b·gᵀDg + b²·Σᵢⱼₖₗ gᵢgⱼgₖgₗVᵢⱼₖₗ / 6),   V = MD²·W        (Jensen et al., MRM 2005)
#
# D is the diffusion tensor (6 distinct elements, mm²/s), W the kurtosis tensor (15 distinct
# elements), MD = tr(D)/3. EQUIP's `DKI` (src/models/diffusion/dki.jl) fits V = MD²·W, as DIPY
# writes DKI: ln S is linear in all 22 parameters, so each voxel starts from the two-pass WLS
# (`:loglinear`) and the fit takes about 4 iterations. What is read off it comes from `derived`:
# D's MD, FA, AD, RD and principal eigenvector, and the kurtosis indices as DIPY's fit methods give
# them by default, MK, AK and RK (Tabesh et al., MRM 2011, analytic) and MKT (Hansen et al., MRM
# 2013), clipped to [−3/7, 10]. The measurements, and what remains to do (crlb at 22 parameters,
# the constraints), are in docs/design.md, roadmap items 17 and 19.
#
# The constraints (K ≥ 0 in every direction) are not box bounds: checked on the results, below.
#
# Run from the repository folder (about 15 s, most of it compiling for 22 parameters):
#   julia -t 4 --project=. examples/dki.jl
# =================================================================================================

using EQUIP, LinearAlgebra, StaticArrays, Statistics, Random, Printf

# --- the acquisition: 6 b = 0, 30 directions at b = 1000 and at 2000 s/mm² (66 images) ----------
# (from files: m = DKI(; diffusionencoding("dwi.bval", "dwi.bvec")...))
g = directionset(30)
bval = [fill(0.0, 6); fill(1000.0, 30); fill(2000.0, 30)]
bvec = reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), 6); g; g])     # 3×66, as a .bvec file holds it
m = DKI(; diffusionencoding(bval, bvec)...)                      # b, gx, gy, gz

# --- the truth: D's eigenvalues 1.7, 0.4, 0.3e-3 mm²/s along x, y, z; isotropic kurtosis K = 1 --
# For isotropic kurtosis W = K·(δᵢⱼδₖₗ + δᵢₖδⱼₗ + δᵢₗδⱼₖ)/3: W1111 = W2222 = W3333 = K and
# W1122 = W1133 = W2233 = K/3, the others 0; in V = MD²·W, with MD = 0.8e-3 mm²/s.
MD2 = ((1.7e-3 + 0.4e-3 + 0.3e-3) / 3)^2
θ = (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0,
     V1111 = MD2, V2222 = MD2, V3333 = MD2, V1112 = 0.0, V1113 = 0.0, V1222 = 0.0, V2223 = 0.0,
     V1333 = 0.0, V2333 = 0.0, V1122 = MD2 / 3, V1133 = MD2 / 3, V2233 = MD2 / 3,
     V1123 = 0.0, V1223 = 0.0, V1233 = 0.0)
truth = derived(m, θ)                       # the indices of the truth itself
# W is isotropic, so MKT (the mean of W(n)) is K = 1; but MK, the mean of the apparent kurtosis
# K(n) = MD²·W(n)/D(n)², is 2.04: D is anisotropic, and K(n) is large where D(n) is small (RK 5.4).
@printf("%d images, %d parameters; the truth: MK %.3f, AK %.3f, RK %.3f, MKT %.3f\n", nobs(m),
        length(parameters(m)), truth.MK, truth.AK, truth.RK, truth.MKT)

# --- fit 2000 simulated voxels, SNR 100 --------------------------------------------------------------
Y = simulate(m, θ; snr = 100, n = 2000, noise = :gaussian, rng = MersenneTwister(1))
t = @elapsed fit(m, Y[1:10, :])                 # (voxels × images)
@printf("first fit, compiling for 22 parameters: %.1f s\n", t)
t = @elapsed (r = fit(m, Y))
@printf("2000 voxels: %.0f µs per voxel, %d threads; converged %d of 2000, %.1f iterations on average\n",
        t / 2000 * 1e6, Threads.nthreads(), count(isconverged, r.status), mean(r.iterations))
d = derived(r)                                  # maps (here vectors): MD, FA, ..., MK, AK, RK, MKT
for k in (:MD, :FA, :MK, :AK, :RK, :MKT)
    @printf("  %-4s mean %10.4g  SD %9.3g  (truth %.4g)\n", k, mean(d[k]), std(d[k]), truth[k])
end

# --- the constraint box bounds cannot state: K(g) ≥ 0 along every measured direction ------------
# K(g) = MD²·gᵀWg / (gᵀDg)² = gᵀVg / (gᵀDg)²; checked on the results rather than imposed in the fit.
# (EQUIP._kurtosisweights: the 15 monomials with their multiplicities, as the model's signal uses.)
function kmin(p, dirs)
    V = SVector(p.V1111, p.V2222, p.V3333, p.V1112, p.V1113, p.V1222, p.V2223, p.V1333, p.V2333,
                p.V1122, p.V1133, p.V2233, p.V1123, p.V1223, p.V1233)
    return minimum(dirs) do (x, y, z)
        gDg = x^2 * p.Dxx + y^2 * p.Dyy + z^2 * p.Dzz + 2x * y * p.Dxy + 2x * z * p.Dxz + 2y * z * p.Dyz
        (EQUIP._kurtosisweights(x, y, z) ⋅ V) / gDg^2
    end
end
neg = count(i -> kmin(map(v -> v[i], r.params), g) < 0, 1:2000)
println("voxels with K < 0 along some measured direction: ", neg, " of 2000")
