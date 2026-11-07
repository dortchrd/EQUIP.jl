# =================================================================================================
# Diffusion kurtosis imaging (DKI), written out by hand: 22 parameters (2026-11-05; in DIPY's
# parameterization since 2026-11-07)
#
#   S = S0·exp(−b·gᵀDg + b²·MD²·Σᵢⱼₖₗ gᵢgⱼgₖgₗWᵢⱼₖₗ / 6)          (Jensen et al., MRM 2005)
#     = S0·exp(−b·gᵀDg + b²·Σᵢⱼₖₗ gᵢgⱼgₖgₗVᵢⱼₖₗ / 6),   V = MD²·W
#
# D is the diffusion tensor (6 distinct elements, mm²/s), W the kurtosis tensor (fully symmetric:
# 15 distinct elements of its 81, dimensionless), MD = tr(D)/3. Along one direction it is
# ADCKurtosis with ADC = gᵀDg and K = MD²·gᵀWg/ADC². The question this answers (the owner's:
# "I feel like that would be a mess"): how a model this size reads and runs in EQUIP. The
# measurements and what remains to do are roadmap item 17 in docs/design.md.
#
# The fit is in V = MD²·W, as DIPY writes DKI (`dki_design_matrix`): ln S is then linear in all
# 22 parameters, and W = V/MD² is computed after the fit. The same model and the same fits as in
# W (W within 7e-7), and 1.8× faster (docs/design.md, "DKI in DIPY's parameterization"):
#   - the start, `Free(start = :loglinear)`, which EQUIP computes from this signal (no start
#     written here), is the weighted log-linear fit in a step, where in W it took about five
#   - the kurtosis terms no longer depend on D, so each step's derivatives cost less
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

# The 15 distinct elements of V = MD²·W, by their indices (1 = x, 2 = y, 3 = z), in (mm²/s)².
# Every element of D and V starts from the weighted log-linear fit (`:loglinear`). No bounds,
# since K ≥ 0 is a condition on all of them at once, not on each.
const LOGLIN = Free(start = :loglinear)        # one role, the same for all 21 elements
parameters(::Type{<:DKI}) = (
    S0 = Free(start = :scale, lower = 0),
    Dxx = LOGLIN, Dyy = LOGLIN, Dzz = LOGLIN, Dxy = LOGLIN, Dxz = LOGLIN, Dyz = LOGLIN,   # mm²/s
    V1111 = LOGLIN, V2222 = LOGLIN, V3333 = LOGLIN, V1112 = LOGLIN, V1113 = LOGLIN,
    V1222 = LOGLIN, V2223 = LOGLIN, V1333 = LOGLIN, V2333 = LOGLIN, V1122 = LOGLIN,
    V1133 = LOGLIN, V2233 = LOGLIN, V1123 = LOGLIN, V1223 = LOGLIN, V1233 = LOGLIN,
)
const VNAMES = (:V1111, :V2222, :V3333, :V1112, :V1113, :V1222, :V2223, :V1333, :V2333,
                :V1122, :V1133, :V2233, :V1123, :V1223, :V1233)

# Σᵢⱼₖₗ gᵢgⱼgₖgₗVᵢⱼₖₗ over all 81 terms is a sum over the 15 distinct elements, each times its
# monomial and the number of index orders that give it: 1 for V1111, 4 for V1112 (1112, 1121,
# 1211, 2111), 6 for V1122, 12 for V1123. These are the 15 monomials with their multiplicities, in
# the order of the parameters above. A static vector (SVector), so no allocation.
@inline kurtosisweights(x, y, z) = SVector(x^4, y^4, z^4,
    4x^3 * y, 4x^3 * z, 4x * y^3, 4y^3 * z, 4x * z^3, 4y * z^3,
    6x^2 * y^2, 6x^2 * z^2, 6y^2 * z^2,
    12x^2 * y * z, 12x * y^2 * z, 12x * y * z^2)

function signal!(out, m::DKI, p)
    (; b, gx, gy, gz) = m
    (; S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz) = p
    V = SVector(p.V1111, p.V2222, p.V3333, p.V1112, p.V1113, p.V1222, p.V2223, p.V1333, p.V2333,
                p.V1122, p.V1133, p.V2233, p.V1123, p.V1223, p.V1233)
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        gDg = gx[j]^2 * Dxx + gy[j]^2 * Dyy + gz[j]^2 * Dzz +
              2gx[j] * gy[j] * Dxy + 2gx[j] * gz[j] * Dxz + 2gy[j] * gz[j] * Dyz
        gVg = dot(kurtosisweights(gx[j], gy[j], gz[j]), V)          # Σ gᵢgⱼgₖgₗ·Vᵢⱼₖₗ
        out[j] = S0 * exp(-b[j] * gDg + b[j]^2 * gVg / 6)
    end
    return out
end

# After the fit: W = V / MD², element by element (MD = tr(D)/3, from the same fit), named
# W1111, W2222, ... (`Symbol("W", ...)` builds each name from V's).
const WNAMES = Tuple(Symbol("W", string(v)[2:end]) for v in VNAMES)
function kurtosistensor(p)
    MD = (p.Dxx + p.Dyy + p.Dzz) / 3
    return NamedTuple{WNAMES}(Tuple(p[v] / MD^2 for v in VNAMES))
end

# The mean kurtosis tensor, the mean of gᵀWg over all directions (Hansen et al., MRM 2013):
# linear in W, so simple. MK, AK and RK need W rotated into D's eigenframe (and, in Tabesh's
# closed forms, MRM 2011, Carlson's elliptic integrals); with roadmap item 16, `propagate` would
# give any of them its error from the fit's covariance.
mkt(W) = (W.W1111 + W.W2222 + W.W3333 + 2 * (W.W1122 + W.W1133 + W.W2233)) / 5

# --- the acquisition: 6 b = 0, 30 directions at b = 1000 and at 2000 s/mm² (66 images) ----------
g = directionset(30)
bval = [fill(0.0, 6); fill(1000.0, 30); fill(2000.0, 30)]
bvec = reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), 6); g; g])     # 3×66, as a .bvec file holds it
m = DKI(; diffusionencoding(bval, bvec)...)                      # b, gx, gy, gz

# --- the truth: D's eigenvalues 1.7, 0.4, 0.3e-3 mm²/s along x, y, z; isotropic kurtosis K = 1 --
# For isotropic kurtosis W = K·(δᵢⱼδₖₗ + δᵢₖδⱼₗ + δᵢₗδⱼₖ)/3: W1111 = W2222 = W3333 = K and
# W1122 = W1133 = W2233 = K/3, the others 0. MKT = K. In V = MD²·W, MD = 0.8e-3 mm²/s.
MD2 = ((1.7e-3 + 0.4e-3 + 0.3e-3) / 3)^2
θ = (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.0, Dxz = 0.0, Dyz = 0.0,
     V1111 = MD2, V2222 = MD2, V3333 = MD2, V1112 = 0.0, V1113 = 0.0, V1222 = 0.0, V2223 = 0.0,
     V1333 = 0.0, V2333 = 0.0, V1122 = MD2 / 3, V1133 = MD2 / 3, V2233 = MD2 / 3,
     V1123 = 0.0, V1223 = 0.0, V1233 = 0.0)
Wtrue = kurtosistensor(θ)
println(nobs(m), " images, ", length(parameters(m)), " parameters; MKT of the truth ", mkt(Wtrue))

# --- fit 2000 simulated voxels, SNR 100 --------------------------------------------------------------
Y = simulate(m, θ; snr = 100, n = 2000, noise = :gaussian, rng = MersenneTwister(1))
t = @elapsed fit(m, Y[1:10, :])                 # (voxels × images)
@printf("first fit, compiling for 22 parameters: %.1f s\n", t)
t = @elapsed (r = fit(m, Y))
@printf("2000 voxels: %.0f µs per voxel, %d threads; converged %d of 2000, %.1f iterations on average\n",
        t / 2000 * 1e6, Threads.nthreads(), count(isconverged, r.status), mean(r.iterations))
Ws = [kurtosistensor(map(v -> v[i], r.params)) for i in 1:2000]     # W of each voxel, from V
for k in (:Dxx, :Dyy)
    @printf("  %-6s mean %10.4g  SD %9.3g  (truth %.4g)\n", k, mean(r.params[k]), std(r.params[k]), θ[k])
end
for k in (:W1111, :W1122, :W1123)
    w = getindex.(Ws, k)
    @printf("  %-6s mean %10.4g  SD %9.3g  (truth %.4g)\n", k, mean(w), std(w), Wtrue[k])
end
M = mkt.(Ws)
@printf("  MKT    mean %10.4g  SD %9.3g  (truth %.4g)\n", mean(M), std(M), mkt(Wtrue))

# --- the constraint box bounds cannot state: K(g) ≥ 0 along every measured direction ------------
# K(g) = MD²·gᵀWg / (gᵀDg)² = gᵀVg / (gᵀDg)²; checked on the results rather than imposed in the fit.
function kmin(p, dirs)
    V = SVector(Tuple(p[v] for v in VNAMES))
    return minimum(dirs) do d
        x, y, z = d
        gDg = x^2 * p.Dxx + y^2 * p.Dyy + z^2 * p.Dzz +
              2x * y * p.Dxy + 2x * z * p.Dxz + 2y * z * p.Dyz
        dot(kurtosisweights(x, y, z), V) / gDg^2
    end
end
neg = count(i -> kmin(map(v -> v[i], r.params), g) < 0, 1:2000)
println("voxels with K < 0 along some measured direction: ", neg, " of 2000")
