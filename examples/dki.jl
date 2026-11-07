# =================================================================================================
# Diffusion kurtosis imaging (DKI), written out by hand: 22 parameters (2026-11-05)
#
#   S = S0·exp(−b·gᵀDg + b²·MD²·Σᵢⱼₖₗ gᵢgⱼgₖgₗWᵢⱼₖₗ / 6)          (Jensen et al., MRM 2005)
#
# D is the diffusion tensor (6 distinct elements, mm²/s), W the kurtosis tensor (fully symmetric:
# 15 distinct elements of its 81, dimensionless), MD = tr(D)/3. Along one direction it is
# ADCKurtosis with ADC = gᵀDg and K = MD²·gᵀWg/ADC². The question this answers (the owner's:
# "I feel like that would be a mess"): how a model this size reads and runs in EQUIP. The
# measurements and what remains to do are roadmap item 17 in docs/design.md.
#
#   - the signal is short; the length is in the 15 names and their multiplicities in the sum
#   - the start: the log-linear fit, `Free(start = :loglinear)`, which EQUIP computes from this
#     signal (no start written here): about 5 iterations per voxel, against 16 from W = 0
#     (docs/design.md, "The log-linear start")
#   - not here: crlb. Its first call does not finish compiling at 22 parameters (stopped after
#     23 minutes); the fix, measured (46 s), is in design.md, item 17
#   - the indices: MKT, linear in W, below (SD 0.04 at SNR 100, where W1111's is 0.25); MK, AK
#     and RK need W in D's eigenframe
#   - the constraints (K ≥ 0 in every direction) are not box bounds: checked on the results
#
# The acquisition as decided for directional models (design.md, "Directions per image"): b and
# three vectors gx, gy, gz, made by `diffusionencoding` from a .bval and a .bvec (here, made-up
# arrays in the same layout).
#
# Run from the repository folder (about 15 s, most of it compiling for 22 parameters):
#   julia -t 4 --project=. examples/dki.jl
# =================================================================================================

using EQUIP, LinearAlgebra, StaticArrays, Statistics, Random, Printf
import EQUIP: parameters, signal!

# --- the model: a struct (the acquisition), its parameters, its signal ---------------------------
@kwdef struct DKI{A,X,Y,Z} <: MRIModel
    b::A                     # b-values (s/mm²)
    gx::X                    # gradient directions (unit vectors): x, y and z components
    gy::Y
    gz::Z
end

# The 15 distinct elements of W, by their indices (1 = x, 2 = y, 3 = z). Every element of D and W
# starts from the weighted log-linear fit (`:loglinear`: Gauss–Newton steps on ln y from 0; W's
# columns are 0 until D has moved, so W waits a step). No bounds, since K ≥ 0 is a condition on
# all of them at once, not on each.
const LOGLIN = Free(start = :loglinear)        # one role, the same for all 21 elements
parameters(::Type{<:DKI}) = (
    S0 = Free(start = :scale, lower = 0),
    Dxx = LOGLIN, Dyy = LOGLIN, Dzz = LOGLIN, Dxy = LOGLIN, Dxz = LOGLIN, Dyz = LOGLIN,   # mm²/s
    W1111 = LOGLIN, W2222 = LOGLIN, W3333 = LOGLIN, W1112 = LOGLIN, W1113 = LOGLIN,
    W1222 = LOGLIN, W2223 = LOGLIN, W1333 = LOGLIN, W2333 = LOGLIN, W1122 = LOGLIN,
    W1133 = LOGLIN, W2233 = LOGLIN, W1123 = LOGLIN, W1223 = LOGLIN, W1233 = LOGLIN,
)

# Σᵢⱼₖₗ gᵢgⱼgₖgₗWᵢⱼₖₗ over all 81 terms is a sum over the 15 distinct elements, each times its
# monomial and the number of index orders that give it: 1 for W1111, 4 for W1112 (1112, 1121,
# 1211, 2111), 6 for W1122, 12 for W1123. These are the 15 monomials with their multiplicities, in
# the order of the parameters above. A static vector (SVector), so no allocation.
@inline kurtosisweights(x, y, z) = SVector(x^4, y^4, z^4,
    4x^3 * y, 4x^3 * z, 4x * y^3, 4y^3 * z, 4x * z^3, 4y * z^3,
    6x^2 * y^2, 6x^2 * z^2, 6y^2 * z^2,
    12x^2 * y * z, 12x * y^2 * z, 12x * y * z^2)

function signal!(out, m::DKI, p)
    (; b, gx, gy, gz) = m
    (; S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz) = p
    W = SVector(p.W1111, p.W2222, p.W3333, p.W1112, p.W1113, p.W1222, p.W2223, p.W1333, p.W2333,
                p.W1122, p.W1133, p.W2233, p.W1123, p.W1223, p.W1233)
    MD = (Dxx + Dyy + Dzz) / 3
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        x, y, z = gx[j], gy[j], gz[j]
        Dg = x^2 * Dxx + y^2 * Dyy + z^2 * Dzz + 2x * y * Dxy + 2x * z * Dxz + 2y * z * Dyz   # gᵀDg
        Wg = dot(kurtosisweights(x, y, z), W)                                             # gᵀWg
        out[j] = S0 * exp(-b[j] * Dg + b[j]^2 * MD^2 * Wg / 6)
    end
    return out
end

# The mean kurtosis tensor, the mean of gᵀWg over all directions (Hansen et al., MRM 2013):
# linear in W, so simple. MK, AK and RK need W rotated into D's eigenframe (and, in Tabesh's
# closed forms, MRM 2011, Carlson's elliptic integrals); with roadmap item 16, `propagate` would
# give any of them its error from the fit's covariance.
mkt(p) = (p.W1111 + p.W2222 + p.W3333 + 2 * (p.W1122 + p.W1133 + p.W2233)) / 5

# --- the acquisition: 6 b = 0, 30 directions at b = 1000 and at 2000 s/mm² (66 images) ----------
g = directionset(30)
bval = [fill(0.0, 6); fill(1000.0, 30); fill(2000.0, 30)]
bvec = reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), 6); g; g])     # 3×66, as a .bvec file holds it
m = DKI(; diffusionencoding(bval, bvec)...)                      # b, gx, gy, gz

# --- the truth: D's eigenvalues 1.7, 0.4, 0.3e-3 mm²/s along x, y, z; isotropic kurtosis K = 1 --
# For isotropic kurtosis W = K·(δᵢⱼδₖₗ + δᵢₖδⱼₗ + δᵢₗδⱼₖ)/3: W1111 = W2222 = W3333 = K and
# W1122 = W1133 = W2233 = K/3, the others 0. MKT = K.
θ = (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0,
     W1111 = 1.0, W2222 = 1.0, W3333 = 1.0, W1112 = 0.0, W1113 = 0.0, W1222 = 0.0, W2223 = 0.0,
     W1333 = 0.0, W2333 = 0.0, W1122 = 1 / 3, W1133 = 1 / 3, W2233 = 1 / 3,
     W1123 = 0.0, W1223 = 0.0, W1233 = 0.0)
println(nobs(m), " images, ", length(parameters(m)), " parameters; MKT of the truth ", mkt(θ))

# --- fit 2000 simulated voxels, SNR 100 --------------------------------------------------------------
Y = simulate(m, θ; snr = 100, n = 2000, noise = :gaussian, rng = MersenneTwister(1))
t = @elapsed fit(m, Y[1:10, :])                 # (voxels × images)
@printf("first fit, compiling for 22 parameters: %.1f s\n", t)
t = @elapsed (r = fit(m, Y))
@printf("2000 voxels: %.0f µs per voxel, %d threads; converged %d of 2000, %.1f iterations on average\n",
        t / 2000 * 1e6, Threads.nthreads(), count(isconverged, r.status), mean(r.iterations))
for k in (:Dxx, :Dyy, :W1111, :W1122, :W1123)
    @printf("  %-6s mean %10.4g  SD %9.3g  (truth %.4g)\n", k, mean(r.params[k]), std(r.params[k]), θ[k])
end
M = [mkt(map(v -> v[i], r.params)) for i in 1:2000]
@printf("  MKT    mean %10.4g  SD %9.3g  (truth %.4g)\n", mean(M), std(M), mkt(θ))

# --- the constraint box bounds cannot state: K(g) ≥ 0 along every measured direction ------------
# K(g) = MD²·gᵀWg / (gᵀDg)²; checked on the results rather than imposed in the fit.
function kmin(p, dirs)
    MD = (p.Dxx + p.Dyy + p.Dzz) / 3
    W = SVector(p.W1111, p.W2222, p.W3333, p.W1112, p.W1113, p.W1222, p.W2223, p.W1333, p.W2333,
                p.W1122, p.W1133, p.W2233, p.W1123, p.W1223, p.W1233)
    return minimum(dirs) do d
        x, y, z = d
        Dg = x^2 * p.Dxx + y^2 * p.Dyy + z^2 * p.Dzz + 2x * y * p.Dxy + 2x * z * p.Dxz + 2y * z * p.Dyz
        MD^2 * dot(kurtosisweights(x, y, z), W) / Dg^2
    end
end
neg = count(i -> kmin(map(v -> v[i], r.params), g) < 0, 1:2000)
println("voxels with K < 0 along some measured direction: ", neg, " of 2000")
