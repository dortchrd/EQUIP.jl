# =================================================================================================
# Multi-compartment models along one direction, every image (no spherical mean): StickZeppelin,
# StickZeppelinBall and BallStick. "Multi-compartment": compartments added together, each a fraction of the water
# (Panagiotaki et al., NeuroImage 2012, whose taxonomy names them by their compartments); the
# compartments are EQUIP's (compartments.jl). The direction is the stick's, a fiber bundle's
# orientation in the papers ("fibre orientation", Behrens et al. 2003): a voxel's, not a tract.
#
#   StickZeppelin   S = S0·[vax·stick(b, Dax, ξ) + (1 − vax)·zeppelin(b, Dax, D⊥, ξ)],
#                   D⊥ = (1 − vax)·Dax (the tortuosity limit)
#   StickZeppelinBall  S = S0·[(1 − viso)·(vax·stick(b, Dax, ξ) + (1 − vax)·zeppelin(b, Dax, D⊥, ξ))
#                          + viso·iso(b, Diso)],  D⊥ = (1 − vax)·Dax: StickZeppelin with a ball (CSF),
#                   NODDI without dispersion (its κ → ∞)
#   BallStick       S = S0·[(1 − f)·iso(b, d) + f·stick(b, d, ξ)]          (Behrens et al., 2003)
#
#   ξ = g·n, the cosine between image j's gradient direction g[j] (a unit vector) and the fiber's
#   n = (sin θ cos φ, sin θ sin φ, cos θ) (`fiberdirection`); stick(b, D, ξ) = exp(−b·D·ξ²),
#   zeppelin(b, D∥, D⊥, ξ) = exp(−b·D⊥ − b·(D∥ − D⊥)·ξ²), iso(b, D) = exp(−b·D)
#
# The acquisition is DTI's: each image's b and gradient direction, b and g (from a .bval and a
# .bvec by `diffusionencoding`). The models are written by hand, n once per voxel and
# ξ = inner(g[j], n) per image: written with `@mrimodel`, whose body runs once per image, the sines and
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
    StickZeppelin(; b, g)

One fiber bundle, every image: intra-axonal water as a stick and extra-axonal water as a
zeppelin along the same direction n(θ, φ), with the same parallel diffusivity and the radial
one from the tortuosity limit (the standard model's kernel without its orientation dispersion;
Kaden et al., NeuroImage 2016, without the spherical mean: [`SMT`](@ref) fits the same tissue
from the directions' average),

    S  = S0·[vax·exp(−b·Dax·ξ²) + (1 − vax)·exp(−b·D⊥ − b·(Dax − D⊥)·ξ²)]
    D⊥ = (1 − vax)·Dax,   ξ = g·n,   n = (sin θ cos φ, sin θ sin φ, cos θ)

- acquisition: `b` (s/mm²) and the gradient direction `g` (a unit vector) of every image, from a
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
@kwdef struct StickZeppelin{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A                                # b-values (s/mm²)
    g::G                                # gradient direction of each image (a unit SVector{3})
    StickZeppelin(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end

parameters(::Type{<:StickZeppelin}) = (
    S0 = Free(start = :scale, lower = 0),
    vax = Free(start = 0.5, lower = 0, upper = 1),
    Dax = Free(start = 2.0e-3, lower = 0, upper = _DMAX),
    theta = Free(start = :data),        # the fiber's direction: from the voxel's tensor,
    phi = Free(start = :data),          #   unbounded (see the header)
)

function signal!(out, m::StickZeppelin, p)
    (; b, g) = m
    (; S0, vax, Dax, theta, phi) = p
    n = fiberdirection(theta, phi)                          # once per voxel
    Dperp = (1 - vax) * Dax                                 # the tortuosity limit
    @inbounds for j in eachindex(out, b, g)
        ξ = inner(g[j], n)                                        # the cosine, gradient to fiber
        out[j] = S0 * (vax * stick(b[j], Dax, ξ) + (1 - vax) * zeppelin(b[j], Dax, Dperp, ξ))
    end
    return out
end

derived(::StickZeppelin, p) = (Dperp = (1 - p.vax) * p.Dax,)
datastart(m::StickZeppelin, y, p) = _tensordirection(m, y)
relabel(::StickZeppelin, θ) = _upperdirection(θ)

"""
    StickZeppelinBall(; b, g)

[`StickZeppelin`](@ref) with a ball, free water (CSF) in the voxel as well: three compartments,
the tissue's two along n(θ, φ) and an isotropic one with the free diffusivity,

    S  = S0·[(1 − viso)·(vax·exp(−b·Dax·ξ²) + (1 − vax)·exp(−b·D⊥ − b·(Dax − D⊥)·ξ²)) + viso·exp(−b·Diso)]
    D⊥ = (1 − vax)·Dax,   ξ = g·n

[`NODDI`](@ref) without orientation dispersion (its limit ODI → 0, κ → ∞), with the axial
diffusivity fitted: a fiber bundle's model where partial volume with CSF matters (near the
ventricles, in atrophy), as the free-water models do for the tensor (Pasternak et al., Magn
Reson Med 2009).

- acquisition: as [`StickZeppelin`](@ref)'s (`b`, `g`); two shells or more, for the
  ball to be told from the tissue
- free by default: `S0`; `vax`, the intra-axonal fraction of the tissue (0 to 1; NODDI's `vic`);
  `viso`, the ball's fraction of the voxel (0 to 1); `Dax`, the axial diffusivity (mm²/s, 0 to
  4e-3); `theta` and `phi`, the fiber's direction (radians, unbounded: reported in the upper
  hemisphere, z ≥ 0)
- fixed by default: `Diso` = 3.0e-3 mm²/s, free water at body temperature (NODDI's; Zhang 2012)
- derived: `Dperp`, the extra-axonal radial diffusivity (1 − vax)·Dax

The direction starts from the voxel's diffusion tensor, as [`StickZeppelin`](@ref)'s.

```julia
res = fit(StickZeppelinBall(; diffusionencoding("dwi.bval", "dwi.bvec")...), dwi; mask)
```
"""
@kwdef struct StickZeppelinBall{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A
    g::G
    StickZeppelinBall(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end

parameters(::Type{<:StickZeppelinBall}) = (
    S0 = Free(start = :scale, lower = 0),
    vax = Free(start = 0.5, lower = 0, upper = 1),
    viso = Free(start = 0.1, lower = 0, upper = 1),
    Dax = Free(start = 2.0e-3, lower = 0, upper = _DMAX),
    theta = Free(start = :data),
    phi = Free(start = :data),
    Diso = Fixed(3.0e-3; lower = 0, upper = _DMAX),
)

function signal!(out, m::StickZeppelinBall, p)
    (; b, g) = m
    (; S0, vax, viso, Dax, theta, phi, Diso) = p
    n = fiberdirection(theta, phi)
    Dperp = (1 - vax) * Dax
    @inbounds for j in eachindex(out, b, g)
        ξ = inner(g[j], n)
        tissue = vax * stick(b[j], Dax, ξ) + (1 - vax) * zeppelin(b[j], Dax, Dperp, ξ)
        out[j] = S0 * ((1 - viso) * tissue + viso * iso(b[j], Diso))
    end
    return out
end

derived(::StickZeppelinBall, p) = (Dperp = (1 - p.vax) * p.Dax,)
datastart(m::StickZeppelinBall, y, p) = _tensordirection(m, y)
relabel(::StickZeppelinBall, θ) = _upperdirection(θ)

"""
    BallStick(; b, g)

The ball-and-stick model (Behrens et al., Magn Reson Med 2003;50:1077; FSL's bedpostx with
one fiber): an isotropic compartment and a stick along n(θ, φ), with one diffusivity `d` for
both,

    S = S0·[(1 − f)·exp(−b·d) + f·exp(−b·d·ξ²)],   ξ = g·n,   n = (sin θ cos φ, sin θ sin φ, cos θ)

- acquisition: `b` (s/mm²) and the gradient direction `g` (a unit vector) of every image, from a
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
@kwdef struct BallStick{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A
    g::G
    BallStick(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end

parameters(::Type{<:BallStick}) = (
    S0 = Free(start = :scale, lower = 0),
    f = Free(start = 0.5, lower = 0, upper = 1),
    d = Free(start = 1.0e-3, lower = 0, upper = _DMAX),
    theta = Free(start = :data),
    phi = Free(start = :data),
)

function signal!(out, m::BallStick, p)
    (; b, g) = m
    (; S0, f, d, theta, phi) = p
    n = fiberdirection(theta, phi)
    @inbounds for j in eachindex(out, b, g)
        ξ = inner(g[j], n)
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
    U = _tensorvectors(m, y)
    T = eltype(y)
    U === nothing && return (theta = T(π) / 2, phi = zero(T))
    n = _upperhemisphere(U[:, 3])                          # the largest eigenvalue's vector
    return (theta = acos(clamp(n[3], -one(T), one(T))), phi = atan(n[2], n[1]))
end
# The voxel's log-linear tensor's eigenvectors (columns, eigenvalues ascending), or nothing
function _tensorvectors(m, y)
    (; b, g) = m
    T = eltype(y)
    A = zero(SMatrix{7,7,T})
    r = zero(SVector{7,T})
    @inbounds for j in eachindex(y, b, g)
        y[j] > 0 || continue
        bj = T(b[j])
        row = SVector{7,T}(1, (-bj .* _tensorweights(T.(g[j])...))...)   # [1, −b·w(g)]: a B-matrix row (dti.jl)
        w = y[j]^2
        A += w * row * row'
        r += (w * log(y[j])) * row
    end
    L, ok = _cholesky(A)
    ok || return nothing
    c = _chol_solve(L, r)
    D = SMatrix{3,3,T}(c[2], c[5], c[6], c[5], c[3], c[7], c[6], c[7], c[4])
    all(isfinite, D) || return nothing
    _, U = _symeigen(D)
    return U
end

# After the fit: the same fiber as the direction in the upper hemisphere (n and −n give the same
# signal, which depends on ξ² only), θ in [0, π/2], φ in (−π, π]. Only when both angles are free:
# a fixed one pins the labels.
function _upperdirection(θ)
    haskey(θ, :theta) && haskey(θ, :phi) || return θ
    n = _upperhemisphere(SVector(fiberdirection(θ.theta, θ.phi)))
    return merge(θ, (theta = acos(clamp(n[3], -one(n[3]), one(n[3]))), phi = atan(n[2], n[1])))
end
