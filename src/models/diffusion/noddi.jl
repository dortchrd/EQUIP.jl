# =================================================================================================
# NODDI, the four forms: sticks and a zeppelin dispersed over orientations, and a ball
#
#   S = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·A_iso]
#   A_ic  = the sticks averaged over the orientation distribution (intra-neurite)
#   A_ec  = the zeppelin, D⊥ = (1 − vic)·D∥ (the tortuosity limit), averaged (extra-neurite)
#   A_iso = iso(b, Diso)                      a ball (CSF, free water)
#
#                            Watson (one κ about μ)    Bingham (κ, β: a fan about μ)
#   the zeppelin's tensor    NODDI                     BinghamNODDI
#     averaged (Zhang's)
#   the zeppelin's signal    NODDIConvolved            BinghamNODDIConvolved
#     averaged (dmipy's)
#
# Written with the compartments and their dispersed views (dispersion.jl, which has the integrals:
# Watson's stick by its series to b·D∥ = 12.75, by quadrature above; Bingham's by quadrature).
# The four share D∥ and Diso, fixed at the paper's in vivo values, 1.7e-3 and 3.0e-3 mm²/s
# (Zhang 2012), as the NODDI toolbox fixes them, ODI's bounds and the orientation's start from the
# voxel's diffusion tensor.
#
# --- Watson: NODDI (Zhang et al., NeuroImage 2012;61:1000) and NODDIConvolved --------------------
#
#   A_ic  = stick(b, D∥, w(ξ))                sticks averaged over W
#   A_ec  = zeppelin(b, D∥ₑ, D⊥ₑ, ξ)          NODDI: a zeppelin along μ with the tensor averaged over W,
#           (D∥ₑ, D⊥ₑ) = watsontensor(D∥, D⊥, w)    an approximation NODDI makes (Eq. 5)
#   A_ec  = zeppelin(b, D∥, D⊥, w(ξ))         NODDIConvolved: the zeppelin's signal averaged, as dmipy
#   w = watson(κ): W(n) ∝ exp(κ·(μ·n)²) about μ(θ, φ),  ξ = g·μ,  ODI = (2/π)·arctan(1/κ)
#
# At κ → ∞ (ODI → 0) NODDI is StickZeppelinBall with Dax = D∥ (multicompartment.jl), at κ = 0
# (ODI = 1) the sticks' spherical mean with an isotropic tensor.
#
# --- Bingham: BinghamNODDI (Tariq et al., NeuroImage 2016;133:207) and BinghamNODDIConvolved ------
#
#   NODDI with anisotropic dispersion, the neurites fanning more in one plane than across it:
#   A_ic = stick(b, D∥, B(g))                 sticks averaged over B
#   A_ec = exp(−b·gᵀD_ec·g),  D_ec = binghamtensor(D∥, (1 − vic)·D∥, B)   BinghamNODDI: the tensor
#                                             averaged over B, of three eigenvalues
#   A_ec = zeppelin(b, D∥, D⊥, B(g))          BinghamNODDIConvolved: the signal averaged
#   B = bingham(κ, β, μ, μβ): B(n) ∝ exp(κ·(μ·n)² + β·(μβ·n)²),  0 ≤ β ≤ κ
#
# μ(θ, φ) the mean orientation, μβ ⊥ μ the direction the neurites fan toward (the angle ψ about
# μ), μ3 = μ × μβ the third axis. β = 0 is Watson (NODDI); as β grows the distribution spreads in
# the plane of μ and μβ, and at β = κ it is uniform in that plane (a girdle about μ3). The
# parameterization is dmipy's (its `SD2Bingham`: κ = cot(π·ODI/2), β = betafraction·κ, ψ about
# μ); Tariq 2016 is its source.
# =================================================================================================

# The four forms' ODI: its lower bound, κ = 64 (the NODDI toolbox's largest concentration), and κ
# from ODI = (2/π)·arctan(1/κ): κ = cot(π·ODI/2), 0 at ODI = 1
const _ODI_MIN = 2 / π * atan(1 / 64)
@inline _kappa(ODI) = cot(oftype(float(ODI), π) * ODI / 2)

# =================================================================================================
# Watson: NODDI and NODDIConvolved (the header)
# =================================================================================================

"""
    NODDI(; b, g)

Neurite orientation dispersion and density imaging (Zhang et al., NeuroImage 2012;61:1000):
sticks (neurites) and a zeppelin (the space around them, its radial diffusivity from the
tortuosity limit) spread over orientations about μ(θ, φ) by a Watson distribution, and a ball
(CSF),

    S = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·exp(−b·Diso)]

- acquisition: `b` (s/mm²) and the gradient direction `g` (a unit vector) of every image, from a
  .bval and a .bvec by [`diffusionencoding`](@ref); two shells or more (e.g. b = 700 or 1000
  and 2000 or more), as NODDI needs
- free by default: `S0`; `vic`, the intra-neurite fraction of the tissue (the neurite density
  index, NDI; the toolbox's `ficvf`); `viso`, the isotropic (CSF) fraction (`fiso`); `ODI`, the
  orientation dispersion index (2/π)·arctan(1/κ), from 0.01 (κ = 64, the toolbox's bound) to 1
  (isotropic); `theta` and `phi`, the mean orientation μ (radians, unbounded: reported in the
  upper hemisphere, z ≥ 0)
- fixed by default: `Dpar` = 1.7e-3 mm²/s, the intrinsic diffusivity along a neurite (also the
  zeppelin's parallel one); `Diso` = 3.0e-3 mm²/s, the ball's (Zhang 2012's in vivo values)
- derived: `kappa`, the Watson concentration κ = 1/tan(π·ODI/2)

The orientation starts from the voxel's diffusion tensor, as [`StickZeppelin`](@ref)'s. The
dispersed sticks are computed from their Legendre series up to b·D∥ = 12.75 (b = 7500 s/mm² at
D∥ = 1.7e-3) and by quadrature above: about 3 µs a signal at 63 images, a brain's 300,000 voxels
in about 10 s on 4 threads, whether the b-values are exact or vary by direction as scanners
write them (NODDIConvolved: 23 s).

```julia
res = fit(NODDI(; diffusionencoding("dwi.bval", "dwi.bvec")...), dwi; mask)
```
"""
@kwdef struct NODDI{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A
    g::G
    NODDI(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end

parameters(::Type{<:NODDI}) = (
    S0 = Free(start = :scale, lower = 0),
    vic = Free(start = 0.5, lower = 0, upper = 1),
    viso = Free(start = 0.1, lower = 0, upper = 1),
    ODI = Free(start = 0.3, lower = _ODI_MIN, upper = 1),
    theta = Free(start = :data),
    phi = Free(start = :data),
    Dpar = Fixed(1.7e-3; lower = 0, upper = _DMAX),
    Diso = Fixed(3.0e-3; lower = 0, upper = _DMAX),
)

function signal!(out, m::NODDI, p)
    (; b, g) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    μ = fiberdirection(theta, phi)                      # once per voxel
    w = watson(_kappa(ODI))                             # the neurites' distribution
    Dpare, Dperpe = watsontensor(Dpar, (1 - vic) * Dpar, w)   # the extra-neurite tensor, averaged
    @inbounds for j in eachindex(out, b, g)
        ξ = g[j] ⋅ μ
        Aic = stick(b[j], Dpar, w(ξ))
        Aec = zeppelin(b[j], Dpare, Dperpe, ξ)
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

derived(::NODDI, p) = (kappa = _kappa(p.ODI),)
datastart(m::NODDI, y, p) = _tensordirection(m, y)
relabel(::NODDI, θ) = _upperdirection(θ)

"""
    NODDIConvolved(; b, g)

[`NODDI`](@ref) with the extra-neurite zeppelin's *signal* averaged over the Watson distribution,
as the sticks' is, in place of Zhang's averaged tensor:

    A_ec = zeppelin(b, D∥, D⊥, w(ξ)) = ⟨exp(−b·D⊥ − b·(D∥ − D⊥)·(g·n)²)⟩_W,   D⊥ = (1 − vic)·D∥

The same parameters, bounds, starts and fixed diffusivities as NODDI's. Every compartment is then
averaged the same way, as a voxel's signal is the sum of its parts' (two regions of differently
oriented fibers give the mean of their signals), as in SMT and the Standard Model (Novikov et
al., NeuroImage 2019) and as dmipy disperses NODDI. Zhang's tensor average is the limit where
each extra-neurite water molecule samples many orientations within the diffusion time. The two
differ by 2–4% at b = 2500, which the residuals cannot show, and shift white matter's vic and ODI
by about 1.5–2 SDs at SNR 100 (docs/design.md, item 7): a choice to make, and to report.
"""
@kwdef struct NODDIConvolved{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A
    g::G
    NODDIConvolved(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end
parameters(::Type{<:NODDIConvolved}) = parameters(NODDI)

function signal!(out, m::NODDIConvolved, p)
    (; b, g) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    μ = fiberdirection(theta, phi)
    Dperp = (1 - vic) * Dpar
    w = watson(_kappa(ODI))                             # the neurites' distribution
    @inbounds for j in eachindex(out, b, g)
        ξ = g[j] ⋅ μ
        Aic = stick(b[j], Dpar, w(ξ))
        Aec = zeppelin(b[j], Dpar, Dperp, w(ξ))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

derived(::NODDIConvolved, p) = (kappa = _kappa(p.ODI),)
datastart(m::NODDIConvolved, y, p) = _tensordirection(m, y)
relabel(::NODDIConvolved, θ) = _upperdirection(θ)

# =================================================================================================
# Bingham: BinghamNODDI and BinghamNODDIConvolved (the header)
# =================================================================================================

"""
    BinghamNODDI(; b, g)

NODDI with anisotropic orientation dispersion, the Bingham distribution (Tariq et al.,
NeuroImage 2016;133:207): neurites that fan more in one plane than across it, as where fibers
fan out or bend,

    S = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·exp(−b·Diso)],   B(n) ∝ exp(κ(μ·n)² + β(μβ·n)²)

- acquisition: as [`NODDI`](@ref)'s (`b`, `g`; two shells or more)
- free by default: `S0`, `vic`, `viso`, `ODI` (κ = cot(π·ODI/2), 0.01 to 1, as NODDI's), and
  `betafraction`, β/κ from 0 (Watson: [`NODDI`](@ref)) to 1 (spread evenly in the fan's plane);
  `theta`, `phi`, the mean orientation μ, and `psi`, the fan's direction μβ as an angle about μ
  (radians, unbounded: reported with μ in the upper hemisphere and ψ in [0, π))
- fixed by default: `Dpar` = 1.7e-3, `Diso` = 3.0e-3 mm²/s, as NODDI's
- derived, Tariq 2016's indices (Eqs. 17–22): `kappa`, `beta`; `ODIP` = (2/π)·arctan(1/(κ − β)),
  the dispersion along the primary dispersion orientation μβ (the fan's plane); `ODItot` =
  (2/π)·arctan(1/√((κ − β)·κ)), the overall dispersion; `DAB` = (2/π)·arctan(β/(κ − β)), the
  dispersion anisotropy, 0 for Watson and 1 at β = κ; `DAT` = (τ2 − τ3)/τ1, the planarity of the
  orientation tensor ⟨nnᵀ⟩ (eigenvalues τ1 ≥ τ2 ≥ τ3). The fitted `ODI` is Tariq's ODI_S, the
  dispersion along μ3 (across the fan), (2/π)·arctan(1/κ)

The orientations start from the voxel's diffusion tensor: μ its principal eigenvector, as
[`StickZeppelin`](@ref)'s, and ψ from its second (the fan spreads the diffusion along μβ).
"""
@kwdef struct BinghamNODDI{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A
    g::G
    BinghamNODDI(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end

"""
    BinghamNODDIConvolved(; b, g)

[`BinghamNODDI`](@ref) with the extra-neurite zeppelin's *signal* averaged over the Bingham
distribution, `zeppelin(b, D∥, D⊥, B(g))`, as the sticks' is, in place of the averaged tensor:
the Bingham counterpart of [`NODDIConvolved`](@ref) (its reasons there). The same parameters, starts
and derived quantities as BinghamNODDI's. Two quadratures an image (the sticks' and the
zeppelin's), so about twice BinghamNODDI's time.
"""
@kwdef struct BinghamNODDIConvolved{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A
    g::G
    BinghamNODDIConvolved(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end
# the methods they share: the parameters, derived quantities, start and labels
const _BinghamNODDIs = Union{BinghamNODDI,BinghamNODDIConvolved}

parameters(::Type{<:_BinghamNODDIs}) = (
    S0 = Free(start = :scale, lower = 0),
    vic = Free(start = 0.5, lower = 0, upper = 1),
    viso = Free(start = 0.1, lower = 0, upper = 1),
    ODI = Free(start = 0.3, lower = _ODI_MIN, upper = 1),
    betafraction = Free(start = 0.3, lower = 0, upper = 1),
    theta = Free(start = :data),
    phi = Free(start = :data),
    psi = Free(start = :data),
    Dpar = Fixed(1.7e-3; lower = 0, upper = _DMAX),
    Diso = Fixed(3.0e-3; lower = 0, upper = _DMAX),
)

function signal!(out, m::BinghamNODDI, p)
    (; b, g) = m
    (; S0, vic, viso, ODI, betafraction, theta, phi, psi, Dpar, Diso) = p
    κ = _kappa(ODI)
    μ, μβ, _ = _binghamaxes(theta, phi, psi)
    B = bingham(κ, betafraction * κ, μ, μβ)             # once per voxel
    Dec = binghamtensor(Dpar, (1 - vic) * Dpar, B)      # the extra-neurite tensor, averaged
    @inbounds for j in eachindex(out, b, g)
        gj = g[j]                                       # image j's gradient direction
        Aic = stick(b[j], Dpar, B(gj))
        Aec = exp(-b[j] * (gj' * Dec * gj))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

function signal!(out, m::BinghamNODDIConvolved, p)
    (; b, g) = m
    (; S0, vic, viso, ODI, betafraction, theta, phi, psi, Dpar, Diso) = p
    κ = _kappa(ODI)
    μ, μβ, _ = _binghamaxes(theta, phi, psi)
    B = bingham(κ, betafraction * κ, μ, μβ)
    Dperp = (1 - vic) * Dpar
    @inbounds for j in eachindex(out, b, g)
        gj = g[j]
        Aic = stick(b[j], Dpar, B(gj))
        Aec = zeppelin(b[j], Dpar, Dperp, B(gj))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

# Tariq et al. 2016's indices (Eqs. 17–22; checked against their Table 1 in the tests), written
# with two-argument arctangents, arctan(y/x) = atan(y, x), so that they stay defined, with finite
# derivatives, where κ − β = 0 (β = κ) or κ = 0 (ODI = 1): ODI_P = (2/π)·atan(1, κ − β), DA_B =
# (2/π)·atan(β, κ − β) = (2/π)·atan(f, 1 − f) with f = β/κ. (Their Eq. 18 prints ODI_S with
# κ − β; their text and Table 1 have 1/κ, the fitted ODI.) τ, the orientation tensor's eigenvalues,
# are the distribution's moments along its axes.
function derived(::_BinghamNODDIs, p)
    κ = _kappa(p.ODI)
    f = p.betafraction
    β = f * κ
    c = 2 / oftype(κ, π)
    _, τ1, τ2, τ3 = _binghammoments(κ, β)
    return (kappa = κ, beta = β, ODIP = c * atan(one(κ), κ - β), ODItot = c * atan(one(κ), sqrt(κ * (κ - β))),
            DAB = c * atan(f, 1 - f), DAT = (τ2 - τ3) / τ1)
end
# The start from the voxel's tensor: μ its principal eigenvector, μβ its second (a fan spreads the
# diffusion most along μ, then along the fan); ψ the second's angle about μ
function datastart(m::_BinghamNODDIs, y, p)
    T = eltype(y)
    U = _tensorvectors(m, y)
    U === nothing && return (theta = T(π) / 2, phi = zero(T), psi = zero(T))
    n = _upperhemisphere(U[:, 3])
    θ, φ = acos(clamp(n[3], -one(T), one(T))), atan(n[2], n[1])
    _, eθ, eφ = _binghamaxes(θ, φ, zero(T))                      # ψ = 0: μβ = e_θ, μ3 = e_φ
    v = U[:, 2]
    return (theta = θ, phi = φ, psi = mod(atan(v ⋅ eφ, v ⋅ eθ), T(π)))
end

# μ in the upper hemisphere (as the other directional models), and ψ for the same fan: μβ (or −μβ,
# the same fan) as an angle about the new μ from its e_θ, in [0, π)
function relabel(::_BinghamNODDIs, θ)
    haskey(θ, :theta) && haskey(θ, :phi) && haskey(θ, :psi) || return θ
    μ, μβ, _ = _binghamaxes(θ.theta, θ.phi, θ.psi)
    u = _upperhemisphere(μ)
    t, f = acos(clamp(u[3], -one(u[3]), one(u[3]))), atan(u[2], u[1])
    _, eθ, eφ = _binghamaxes(t, f, zero(t))
    return merge(θ, (theta = t, phi = f, psi = mod(atan(μβ ⋅ eφ, μβ ⋅ eθ), oftype(t, π))))
end
