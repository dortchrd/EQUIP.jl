# =================================================================================================
# Directional models: every image, no spherical mean (a sketch for discussion, 2026-11-04 and 06)
#
#   1. sticks and a zeppelin along one fiber direction n(θ, φ), the same parallel diffusivity,
#      the tortuosity limit for the extra-axonal radial one (Kaden's tissue, without averaging)
#   2. the standard tensor (DTI), S0·exp(−b·gᵀDg)
#   3. the two together: DTI's principal direction as the fibers' direction, fixed (per-voxel
#      maps) or as the start of a fit of θ and φ
#
# The acquisition is each image's b and gradient direction (the .bval and .bvec files), as four
# vectors b, gx, gy, gz, made by `diffusionencoding` (from the files, or here from arrays) and
# passed to any directional model as keywords. The cosine between gradient and fiber, ξ = g·n, is
# computed in the model, because it joins the acquisition (g) and the parameters (n); the
# compartments take it as a number: `n = fiberdirection(θ, φ)` once per voxel, `ξ = g ⋅ n`
# (Julia's dot) for each image, g = (gx[j], gy[j], gz[j]). The stick + zeppelin is written by
# hand for that reason: written with `@mrimodel`, whose body runs once per image, its sines and cosines are
# computed for every image and its fit takes about twice as long (both are timed at the end).
# DTI has no angles, and the macro suits it. A stick and a zeppelin are tensors (D = D∥·nnᵀ;
# D = D⊥·I + (D∥ − D⊥)·nnᵀ): checked below.
#
# Run from the repository folder (about 30 s):
#   julia -t 4 --project=. examples/directional_models.jl
# =================================================================================================

using EQUIP, Random, LinearAlgebra, StaticArrays, Statistics, Printf
import EQUIP: parameters, signal!
rng = MersenneTwister(11)

# --- the acquisition: b and the gradient direction of every image (bval, bvec) ---------------------
g = directionset(64)                                        # 64 unit vectors, z ≥ 0
bval = [fill(0.0, 4); fill(1000.0, 64); fill(2500.0, 64)]
bvec = reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), 4); g; g])   # 3×132, as a .bvec (0 0 0 at b = 0)
acq = diffusionencoding(bval, bvec)                         # (b, gx, gy, gz), checked;
                                                            # from files: diffusionencoding("dwi.bval", "dwi.bvec")

# --- 1. stick + zeppelin along one fiber direction n(θ, φ), written by hand ------------------------
@kwdef struct StickZeppelin{A,X,Y,Z} <: MRIModel
    b::A                     # b-values (s/mm²), one per image
    gx::X                    # gradient direction of each image: x, y and z components
    gy::Y
    gz::Z
end
parameters(::Type{<:StickZeppelin}) = (
    S0  = Free(start = :scale, lower = 0),
    vax = Free(start = 0.5, lower = 0, upper = 1),
    Dax = Free(start = 2.0e-3, lower = 0, upper = 3.0e-3),                 # mm²/s
    θ   = Free(start = 0.5, lower = 0, upper = π),                         # the fiber's polar angle
    φ   = Free(start = 0.0, lower = -π, upper = π),                        # and azimuth
)
function signal!(out, m::StickZeppelin, p)
    (; b, gx, gy, gz) = m
    (; S0, vax, Dax, θ, φ) = p
    n = fiberdirection(θ, φ)                                         # once per voxel
    Dperp = (1 - vax) * Dax                                          # tortuosity
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        g = (gx[j], gy[j], gz[j])                                    # image j's gradient direction
        ξ = g ⋅ n                                                    # the cosine, gradient to fiber
        out[j] = S0 * (vax * stick(b[j], Dax, ξ) + (1 - vax) * zeppelin(b[j], Dax, Dperp, ξ))
    end
    return out
end

# the same model with the macro: the body is one image's signal, so n is computed for every image
@mrimodel function StickZeppelinMacro(b, gx, gy, gz; S0 = Free(start = :scale, lower = 0),
                                      vax = Free(start = 0.5, lower = 0, upper = 1),
                                      Dax = Free(start = 2.0e-3, lower = 0, upper = 3.0e-3),
                                      θ = Free(start = 0.5, lower = 0, upper = π),
                                      φ = Free(start = 0.0, lower = -π, upper = π))
    ξ = (gx, gy, gz) ⋅ fiberdirection(θ, φ)                         # the cosine, gradient to fiber
    Dperp = (1 - vax) * Dax                                          # tortuosity
    S0 * (vax * stick(b, Dax, ξ) + (1 - vax) * zeppelin(b, Dax, Dperp, ξ))
end

# --- 2. the standard tensor (DTI): S0·exp(−b·gᵀDg), six elements ---------------------------------
@mrimodel DTI(b, gx, gy, gz; S0 = Free(start = :scale, lower = 0),
              Dxx = Free(start = 1e-3), Dyy = Free(start = 1e-3), Dzz = Free(start = 1e-3),
              Dxy = Free(start = 0.0), Dxz = Free(start = 0.0), Dyz = Free(start = 0.0)) =
    S0 * exp(-b * (gx^2 * Dxx + gy^2 * Dyy + gz^2 * Dzz + 2gx * gy * Dxy + 2gx * gz * Dxz + 2gy * gz * Dyz))

m = StickZeppelin(; acq...)                                 # `...`: b, gx, gy, gz as keywords
display(m); println()

# a stick and a zeppelin are tensors: D = D∥·nnᵀ, and D⊥·I + (D∥ − D⊥)·nnᵀ
n = SVector(fiberdirection(0.7, 0.3)); gg = g[5]; ξ5 = gg ⋅ n
Dst = 2e-3 * n * n'; Dzp = 0.8e-3 * I + (2e-3 - 0.8e-3) * n * n'
println("stick as a tensor:    ", stick(2500.0, 2e-3, ξ5) ≈ exp(-2500.0 * gg' * Dst * gg))
println("zeppelin as a tensor: ", zeppelin(2500.0, 2e-3, 0.8e-3, ξ5) ≈ exp(-2500.0 * gg' * Dzp * gg))

# --- simulate 300 voxels, fibers in random directions --------------------------------------------
nv = 300
ns = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:nv]; ns = [x[3] < 0 ? -x : x for x in ns]
θt, φt = acos.(getindex.(ns, 3)), atan.(getindex.(ns, 2), getindex.(ns, 1))
truth = (S0 = fill(1000.0, nv), vax = fill(0.6, nv), Dax = fill(1.9e-3, nv), θ = θt, φ = φt)
Y = simulate(m, truth; σ = 10.0, noise = :gaussian, rng)          # SNR 100 at b = 0

# --- fit DTI on b ≤ 1000, take its principal direction ---------------------------------------------
lo = findall(<=(1000), bval)
mdti = DTI(; diffusionencoding(bval[lo], bvec[:, lo])...)
rd = fit(mdti, Y[:, lo])
function principal(p, i)
    D = Symmetric(SMatrix{3,3}(p.Dxx[i], p.Dxy[i], p.Dxz[i], p.Dxy[i], p.Dyy[i], p.Dyz[i], p.Dxz[i], p.Dyz[i], p.Dzz[i]))
    e = eigen(D); v = e.vectors[:, 3]; v = v[3] < 0 ? -v : v
    return acos(clamp(v[3], -1, 1)), atan(v[2], v[1]), (e.values[3] - mean(e.values)) * sqrt(1.5) / norm(e.values)
end
pr = [principal(rd.params, i) for i in 1:nv]
θd, φd, fa = first.(pr), getindex.(pr, 2), last.(pr)
angerr(θ1, φ1, i) = acosd(clamp(abs(dot(SVector(sin(θ1) * cos(φ1), sin(θ1) * sin(φ1), cos(θ1)), ns[i])), 0, 1))
@printf("DTI (b ≤ 1000): FA %.2f, its direction %.1f° from the fibers (median)\n", median(fa), median(angerr.(θd, φd, 1:nv)))

# --- stick + zeppelin: orientation fixed at DTI's, or fitted from it -------------------------------
rf = fit(m, Y; θ = θd, φ = φd)                                       # orientation fixed (maps)
rs = fit(m, Y; θ = Free(start = θd), φ = Free(start = φd))           # fitted, from DTI's
@printf("stick+zeppelin, orientation fixed at DTI's:  vax %.3f ± %.3f, Dax %.3e (truth 0.6, 1.9e-3)\n",
        mean(rf.params.vax), std(rf.params.vax), mean(rf.params.Dax))
@printf("stick+zeppelin, orientation fitted from it: vax %.3f ± %.3f, Dax %.3e; direction %.2f° from the fibers\n",
        mean(rs.params.vax), std(rs.params.vax), mean(rs.params.Dax), median(angerr.(rs.params.θ, rs.params.φ, 1:nv)))
mm = StickZeppelinMacro(; acq...)
rm = fit(mm, Y; θ = Free(start = θd), φ = Free(start = φd))
println("the macro's version gives the same fits: ", maximum(abs.(rm.params.vax .- rs.params.vax)) < 1e-8)
t = minimum(@elapsed(fit(m, Y; θ = Free(start = θd), φ = Free(start = φd))) for _ in 1:5) / nv * 1e6
tm = minimum(@elapsed(fit(mm, Y; θ = Free(start = θd), φ = Free(start = φd))) for _ in 1:5) / nv * 1e6
@printf("fit time, µs per voxel (5 parameters, 132 images, %d threads): by hand %.1f, with the macro %.1f\n",
        Threads.nthreads(), t, tm)
