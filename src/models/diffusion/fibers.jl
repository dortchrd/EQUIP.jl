# =================================================================================================
# One fiber direction, every image (no spherical mean): StickZeppelin and BallStick
#
#   StickZeppelin   S = S0·[vax·stick(b, Dax, ξ) + (1 − vax)·zeppelin(b, Dax, D⊥, ξ)],
#                   D⊥ = (1 − vax)·Dax (the tortuosity limit)
#   BallStick       S = S0·[(1 − f)·iso(b, d) + f·stick(b, d, ξ)]          (Behrens et al., 2003)
#
#   ξ = g·n, the cosine between image j's gradient direction g = (gx, gy, gz) and the fiber's
#   n = (sin θ cos φ, sin θ sin φ, cos θ) (`fiberdirection`); stick(b, D, ξ) = exp(−b·D·ξ²),
#   zeppelin(b, D∥, D⊥, ξ) = exp(−b·D⊥ − b·(D∥ − D⊥)·ξ²), iso(b, D) = exp(−b·D)
#
# The acquisition is DTI's: each image's b and gradient direction, b, gx, gy, gz (from a .bval
# and a .bvec by `diffusionencoding`). The models are written by hand, n once per voxel and
# ξ = g ⋅ n per image: written with `@mrimodel`, whose body runs once per image, the sines and
# cosines are computed for every image, and the fit took about twice as long (examples/
# directional_models.jl, where these were sketched; docs/design.md, roadmap item 17).
#
# The direction, two angles:
#   - its start, from the voxel's own diffusion tensor (`datastart`, `_tensordirection` below):
#     the log-linear tensor fit (weighted by y², a few µs), its principal eigenvector. A fiber's
#     direction is what DTI gets right even where the tensor is the wrong model;
#   - no bounds: the signal depends on ξ² only, so n and −n are one fiber and the angles repeat
#     every π and 2π. A bound would stop a fit at an angle where the signal does not stop (a fiber
#     near the z axis, θ near 0, moves through it). After the fit, `relabel` reports the one
#     direction in the upper hemisphere, z ≥ 0 (as DTI's V1 and `directionset`), θ in [0, π/2]
#     and φ in (−π, π];
#   - at θ = 0 the azimuth φ has no effect (the fiber along z): its map is noise there, as any
#     azimuth's is. The derivative is 0, and the solver's damping copes.
# =================================================================================================

"""
    StickZeppelin(; b, gx, gy, gz)

One fiber bundle, every image: intra-axonal water as a stick and extra-axonal water as a
zeppelin along the same direction n(θ, φ), with the same parallel diffusivity and the radial
one from the tortuosity limit (the standard model's kernel without its orientation dispersion;
Kaden et al., NeuroImage 2016, without the spherical mean: [`SMT`](@ref) fits the same tissue
from the directions' average),

    S  = S0·[vax·exp(−b·Dax·ξ²) + (1 − vax)·exp(−b·D⊥ − b·(Dax − D⊥)·ξ²)]
    D⊥ = (1 − vax)·Dax,   ξ = g·n,   n = (sin θ cos φ, sin θ sin φ, cos θ)

- acquisition: `b` (s/mm²) and the gradient direction `gx`, `gy`, `gz` of every image, from a
  .bval and a .bvec by [`diffusionencoding`](@ref)
- free by default: `S0`; `vax`, the intra-axonal fraction (0 to 1); `Dax`, the axial
  diffusivity (mm²/s, 0 to 4e-3); `theta` and `phi`, the fiber's polar angle and azimuth
  (radians, unbounded: reported in the upper hemisphere, z ≥ 0)
- derived: `Dperp`, the extra-axonal radial diffusivity (1 − vax)·Dax ([`derived`](@ref))

The direction starts from the voxel's diffusion tensor (its log-linear fit, the principal
eigenvector): `theta = Free(start = :data)`. To fix it from a DTI fit instead, give its maps:
`fit(m, data; theta = θmap, phi = φmap)`. The data must have images at two b-values or more
(b = 0 and a shell, or two shells), and directions enough for the tensor (six, not in a plane).

```julia
acq = diffusionencoding("dwi.bval", "dwi.bvec")
res = fit(StickZeppelin(; acq...), dwi; mask)
```
"""
@kwdef struct StickZeppelin{A<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,Z<:AbstractVector} <: MRIModel
    b::A                                # b-values (s/mm²)
    gx::X                               # gradient direction of each image (unit vectors):
    gy::Y                               #   its x, y and z components
    gz::Z
end

parameters(::Type{<:StickZeppelin}) = (
    S0 = Free(start = :scale, lower = 0),
    vax = Free(start = 0.5, lower = 0, upper = 1),
    Dax = Free(start = 2.0e-3, lower = 0, upper = _DMAX),
    theta = Free(start = :data),        # the fiber's direction: from the voxel's tensor,
    phi = Free(start = :data),          #   unbounded (see the header)
)

function signal!(out, m::StickZeppelin, p)
    (; b, gx, gy, gz) = m
    (; S0, vax, Dax, theta, phi) = p
    n = fiberdirection(theta, phi)                          # once per voxel
    Dperp = (1 - vax) * Dax                                 # the tortuosity limit
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        ξ = (gx[j], gy[j], gz[j]) ⋅ n                       # the cosine, gradient to fiber
        out[j] = S0 * (vax * stick(b[j], Dax, ξ) + (1 - vax) * zeppelin(b[j], Dax, Dperp, ξ))
    end
    return out
end

derived(::StickZeppelin, p) = (Dperp = (1 - p.vax) * p.Dax,)
datastart(m::StickZeppelin, y, p) = _tensordirection(m, y)
relabel(::StickZeppelin, θ) = _upperdirection(θ)

"""
    BallStick(; b, gx, gy, gz)

The ball-and-stick model (Behrens et al., Magn Reson Med 2003;50:1077; FSL's bedpostx with
one fiber): an isotropic compartment and a stick along n(θ, φ), with one diffusivity `d` for
both,

    S = S0·[(1 − f)·exp(−b·d) + f·exp(−b·d·ξ²)],   ξ = g·n,   n = (sin θ cos φ, sin θ sin φ, cos θ)

- acquisition: `b` (s/mm²) and the gradient direction `gx`, `gy`, `gz` of every image, from a
  .bval and a .bvec by [`diffusionencoding`](@ref)
- free by default: `S0`; `f`, the stick's fraction (0 to 1); `d`, the diffusivity (mm²/s, 0 to
  4e-3); `theta` and `phi`, the fiber's polar angle and azimuth (radians, unbounded: reported in
  the upper hemisphere, z ≥ 0)

The direction starts from the voxel's diffusion tensor, as [`StickZeppelin`](@ref)'s. One
stick: crossing fibers (bedpostx's two or three sticks, with their fractions driven to 0 where
the data do not support them) are not modelled.

```julia
res = fit(BallStick(; diffusionencoding("dwi.bval", "dwi.bvec")...), dwi; mask)
```
"""
@kwdef struct BallStick{A<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,Z<:AbstractVector} <: MRIModel
    b::A
    gx::X
    gy::Y
    gz::Z
end

parameters(::Type{<:BallStick}) = (
    S0 = Free(start = :scale, lower = 0),
    f = Free(start = 0.5, lower = 0, upper = 1),
    d = Free(start = 1.0e-3, lower = 0, upper = _DMAX),
    theta = Free(start = :data),
    phi = Free(start = :data),
)

function signal!(out, m::BallStick, p)
    (; b, gx, gy, gz) = m
    (; S0, f, d, theta, phi) = p
    n = fiberdirection(theta, phi)
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        ξ = (gx[j], gy[j], gz[j]) ⋅ n
        out[j] = S0 * ((1 - f) * iso(b[j], d) + f * stick(b[j], d, ξ))
    end
    return out
end

datastart(m::BallStick, y, p) = _tensordirection(m, y)
relabel(::BallStick, θ) = _upperdirection(θ)

# The direction's start: the principal eigenvector of the voxel's diffusion tensor, from the
# log-linear fit ln y = ln S0 − b·gᵀDg (DTI's), each image weighted by y² (the weighted fit's
# usual weights: ln y's variance is about σ²/y²), by the normal equations: seven unknowns,
# x = (ln S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz), static arrays, no allocation. Images with y ≤ 0 are
# left out. Where the tensor cannot be fitted (too few directions, all of y ≤ 0, the placeholder
# data `setup` asks with), the x axis: (θ, φ) = (π/2, 0).
function _tensordirection(m, y)
    (; b, gx, gy, gz) = m
    T = eltype(y)
    A = zero(SMatrix{7,7,T})
    r = zero(SVector{7,T})
    @inbounds for j in eachindex(y, b, gx, gy, gz)
        y[j] > 0 || continue
        x, yy, z, bj = T(gx[j]), T(gy[j]), T(gz[j]), T(b[j])
        row = SVector{7,T}(1, -bj * x * x, -bj * yy * yy, -bj * z * z, -2bj * x * yy, -2bj * x * z, -2bj * yy * z)
        w = y[j]^2
        A += w * row * row'
        r += (w * log(y[j])) * row
    end
    L, ok = _cholesky(A)
    ok || return (theta = T(π) / 2, phi = zero(T))
    c = _chol_solve(L, r)
    D = SMatrix{3,3,T}(c[2], c[5], c[6], c[5], c[3], c[7], c[6], c[7], c[4])
    all(isfinite, D) || return (theta = T(π) / 2, phi = zero(T))
    _, U = _symeigen(D)
    n = _upperhemisphere(U[:, 3])                          # the largest eigenvalue's vector
    return (theta = acos(clamp(n[3], -one(T), one(T))), phi = atan(n[2], n[1]))
end

# After the fit: the same fiber as the direction in the upper hemisphere (n and −n give the same
# signal, which depends on ξ² only), θ in [0, π/2], φ in (−π, π]. Only when both angles are free:
# a fixed one pins the labels.
function _upperdirection(θ)
    haskey(θ, :theta) && haskey(θ, :phi) || return θ
    n = _upperhemisphere(SVector(fiberdirection(θ.theta, θ.phi)))
    return merge(θ, (theta = acos(clamp(n[3], -one(n[3]), one(n[3]))), phi = atan(n[2], n[1])))
end
