# =================================================================================================
# Bingham–NODDI: NODDI with anisotropic orientation dispersion (Tariq et al., NeuroImage
# 2016;133:207), the neurites fanning more in one plane than across it
#
#   S = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·iso(b, Diso)]      NODDI's (noddi.jl)
#   A_ic = stick(b, D∥, B(g))                    sticks averaged over B
#   A_ec = exp(−b·gᵀD_ec·g),  D_ec = binghamtensor(D∥, (1 − vic)·D∥, B)   the tensor averaged over B
#   B = bingham(κ, β, μ, μβ): B(n) ∝ exp(κ·(μ·n)² + β·(μβ·n)²),  0 ≤ β ≤ κ
#
# μ(θ, φ) the mean orientation, μβ ⊥ μ the direction the neurites fan toward (the angle ψ about
# μ), μ3 = μ × μβ the third axis. β = 0 is Watson (NODDI); as β grows the distribution spreads in
# the plane of μ and μβ, and at β = κ it is uniform in that plane (a girdle about μ3). The
# parameterization is dmipy's (its `SD2Bingham`: κ = cot(π·ODI/2), β = betafraction·κ, ψ about
# μ); Tariq 2016 is its source. Written with the compartments and the Bingham view
# (dispersion.jl, which has the integrals); the extra-neurite space as NODDI's, the zeppelin's
# tensor averaged, here of three eigenvalues and so written as a tensor's signal.
# =================================================================================================

"""
    BinghamNODDI(; b, gx, gy, gz)

NODDI with anisotropic orientation dispersion, the Bingham distribution (Tariq et al.,
NeuroImage 2016;133:207): neurites that fan more in one plane than across it, as where fibers
fan out or bend,

    S = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·exp(−b·Diso)],   B(n) ∝ exp(κ(μ·n)² + β(μβ·n)²)

- acquisition: as [`NODDI`](@ref)'s (`b`, `gx`, `gy`, `gz`; two shells or more)
- free by default: `S0`, `vic`, `viso`, `ODI` (κ = cot(π·ODI/2), 0.01 to 1, as NODDI's), and
  `betafraction`, β/κ from 0 (Watson: [`NODDI`](@ref)) to 1 (spread evenly in the fan's plane);
  `theta`, `phi`, the mean orientation μ, and `psi`, the fan's direction μβ as an angle about μ
  (radians, unbounded: reported with μ in the upper hemisphere and ψ in [0, π))
- fixed by default: `Dpar` = 1.7e-3, `Diso` = 3.0e-3 mm²/s, as NODDI's
- derived: `kappa`, `beta`; `ODIfan`, the dispersion within the fan's plane, (2/π)·arctan(1/(κ − β))
  (ODI itself is the dispersion across it); `DA`, the dispersion anisotropy (2/π)·arctan(β/(κ − β)),
  0 for Watson and 1 at β = κ

The orientations start from the voxel's diffusion tensor: μ its principal eigenvector, as
[`StickZeppelin`](@ref)'s, and ψ from its second (the fan spreads the diffusion along μβ).
"""
@kwdef struct BinghamNODDI{A<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,Z<:AbstractVector} <: MRIModel
    b::A
    gx::X
    gy::Y
    gz::Z
end

"""
    BinghamNODDISignal(; b, gx, gy, gz)

[`BinghamNODDI`](@ref) with the extra-neurite zeppelin's *signal* averaged over the Bingham
distribution, `zeppelin(b, D∥, D⊥, B(g))`, as the sticks' is, in place of the averaged tensor:
the Bingham counterpart of [`NODDISignal`](@ref) (its reasons there). The same parameters, starts
and derived quantities as BinghamNODDI's. Two quadratures an image (the sticks' and the
zeppelin's), so about twice BinghamNODDI's time.
"""
@kwdef struct BinghamNODDISignal{A<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,Z<:AbstractVector} <: MRIModel
    b::A
    gx::X
    gy::Y
    gz::Z
end
# the methods they share: the parameters, derived quantities, start and labels
const _BinghamNODDIs = Union{BinghamNODDI,BinghamNODDISignal}

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
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, betafraction, theta, phi, psi, Dpar, Diso) = p
    κ = _kappa(ODI)
    μ, μβ, _ = _binghamaxes(theta, phi, psi)
    B = bingham(κ, betafraction * κ, μ, μβ)             # once per voxel
    Dec = binghamtensor(Dpar, (1 - vic) * Dpar, B)      # the extra-neurite tensor, averaged
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        g = SVector(gx[j], gy[j], gz[j])
        Aic = stick(b[j], Dpar, B(g))
        Aec = exp(-b[j] * (g' * Dec * g))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

function derived(::_BinghamNODDIs, p)
    κ = _kappa(p.ODI)
    β = p.betafraction * κ
    return (kappa = κ, beta = β, ODIfan = 2 / oftype(κ, π) * atan(1 / (κ - β)),
            DA = 2 / oftype(κ, π) * atan(β / (κ - β)))
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

function signal!(out, m::BinghamNODDISignal, p)
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, betafraction, theta, phi, psi, Dpar, Diso) = p
    κ = _kappa(ODI)
    μ, μβ, _ = _binghamaxes(theta, phi, psi)
    B = bingham(κ, betafraction * κ, μ, μβ)
    Dperp = (1 - vic) * Dpar
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        g = SVector(gx[j], gy[j], gz[j])
        Aic = stick(b[j], Dpar, B(g))
        Aec = zeppelin(b[j], Dpar, Dperp, B(g))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end
