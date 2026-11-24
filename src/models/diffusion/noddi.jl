# =================================================================================================
# NODDI: neurite orientation dispersion and density imaging (Zhang et al., NeuroImage 2012;61:1000)
#
#   S = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·A_iso]
#   A_ic  = stick(b, D∥, w(ξ))                sticks averaged over W (intra-neurite)
#   A_ec  = zeppelin(b, D∥ₑ, D⊥ₑ, ξ)          a zeppelin along μ with the tensor averaged over W
#           (D∥ₑ, D⊥ₑ) = watsontensor(D∥, D⊥, w), D⊥ = (1 − vic)·D∥ (the tortuosity limit)
#   A_iso = iso(b, Diso)                      a ball (CSF, free water)
#   w = watson(κ): W(n) ∝ exp(κ·(μ·n)²) about μ(θ, φ),  ξ = g·μ,  ODI = (2/π)·arctan(1/κ)
#
# Written with the compartments and their dispersed view (dispersion.jl, which has the integrals:
# the stick's series to b·D∥ = 12.75, its quadrature above). The extra-neurite space is Zhang's:
# the zeppelin's *tensor* averaged over W, an approximation NODDI makes (Eq. 5), which is again a
# zeppelin along μ; not the zeppelin's signal averaged, `zeppelin(b, D∥, D⊥, w(ξ))`, as dmipy has it.
# At κ → ∞ (ODI → 0) the model is StickZeppelin with a ball (multicompartment.jl), at κ = 0 (ODI =
# 1) the sticks' spherical mean with an isotropic tensor. D∥ and Diso fixed at the paper's in vivo
# values, 1.7e-3 and 3.0e-3 mm²/s (Zhang 2012), as the NODDI toolbox fixes them.
# =================================================================================================

"""
    NODDI(; b, gx, gy, gz)

Neurite orientation dispersion and density imaging (Zhang et al., NeuroImage 2012;61:1000):
sticks (neurites) and a zeppelin (the space around them, its radial diffusivity from the
tortuosity limit) spread over orientations about μ(θ, φ) by a Watson distribution, and a ball
(CSF),

    S = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·exp(−b·Diso)]

- acquisition: `b` (s/mm²) and the gradient direction `gx`, `gy`, `gz` of every image, from a
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
D∥ = 1.7e-3) and by quadrature above: about 4 µs a signal at 63 images, a brain's 300,000 voxels
in 12–15 s on 4 threads.

```julia
res = fit(NODDI(; diffusionencoding("dwi.bval", "dwi.bvec")...), dwi; mask)
```
"""
@kwdef struct NODDI{A<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,Z<:AbstractVector} <: MRIModel
    b::A
    gx::X
    gy::Y
    gz::Z
end

# ODI's lower bound: κ = 64, the NODDI toolbox's largest concentration
const _ODI_MIN = 2 / π * atan(1 / 64)

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
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    μ = fiberdirection(theta, phi)                      # once per voxel
    w = watson(_kappa(ODI), b, Dpar)                    # the neurites' distribution; the stick's
                                                        #  series kept for each shell
    Dpare, Dperpe = watsontensor(Dpar, (1 - vic) * Dpar, w)   # the extra-neurite tensor, averaged
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        ξ = (gx[j], gy[j], gz[j]) ⋅ μ
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
    NODDISignal(; b, gx, gy, gz)

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
@kwdef struct NODDISignal{A<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,Z<:AbstractVector} <: MRIModel
    b::A
    gx::X
    gy::Y
    gz::Z
end
parameters(::Type{<:NODDISignal}) = parameters(NODDI)

function signal!(out, m::NODDISignal, p)
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    μ = fiberdirection(theta, phi)
    Dperp = (1 - vic) * Dpar
    w = watson(_kappa(ODI), b, Dpar)                    # the sticks': their series per shell
    wz = watson(w, b, Dpar - Dperp)                     # the same distribution, the zeppelin's excess
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        ξ = (gx[j], gy[j], gz[j]) ⋅ μ
        Aic = stick(b[j], Dpar, w(ξ))
        Aec = zeppelin(b[j], Dpar, Dperp, wz(ξ))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

derived(::NODDISignal, p) = (kappa = _kappa(p.ODI),)
datastart(m::NODDISignal, y, p) = _tensordirection(m, y)
relabel(::NODDISignal, θ) = _upperdirection(θ)

# κ from ODI = (2/π)·arctan(1/κ): κ = cot(π·ODI/2), 0 at ODI = 1
@inline _kappa(ODI) = cot(oftype(float(ODI), π) * ODI / 2)
