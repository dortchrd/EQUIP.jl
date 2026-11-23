# =================================================================================================
# NODDI: neurite orientation dispersion and density imaging (Zhang et al., NeuroImage 2012;61:1000)
#
#   S    = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·A_iso]
#   A_ic = ∫ W(n)·exp(−b·D∥·(g·n)²) dn                 sticks, dispersed (intra-neurite)
#   A_ec = exp(−b·gᵀD_ec·g)                            a zeppelin, dispersed (extra-neurite):
#          D_ec = ∫ W(n)·[D⊥·I + (D∥ − D⊥)·nnᵀ] dn,   D⊥ = (1 − vic)·D∥ (the tortuosity limit)
#   A_iso = exp(−b·Diso)                               a ball (CSF, free water)
#   W(n) ∝ exp(κ·(μ·n)²)                               the Watson distribution about μ(θ, φ)
#   ODI = (2/π)·arctan(1/κ)                            0: every neurite along μ; 1: isotropic
#
# The model of StickZeppelin plus a ball (multicompartment.jl), with the stick and the zeppelin
# spread over orientations by a Watson distribution: at κ → ∞ (ODI → 0) it is that model (a
# stick, a zeppelin, a ball), at κ = 0 (ODI = 1) its spherical mean (SMTFreeWater's tissue, with
# D∥ fixed). D∥ and Diso fixed at the paper's in vivo values, 1.7e-3 and 3.0e-3 mm²/s (Zhang
# 2012), as the NODDI toolbox fixes them.
#
# The dispersed zeppelin: averaging the tensor over W gives, with τ = ⟨(μ·n)²⟩ the Watson mean of
# cos² (Zhang 2012, Eq. 5),
#   gᵀD_ec·g = D⊥ + (D∥ − D⊥)·[τ·t² + (1 − τ)·(1 − t²)/2],   t = g·μ
#   τ(κ) = 1/(2√κ·F(√κ)) − 1/(2κ)                     F: Dawson's integral; 1/3 at κ = 0
# (τ from ∫₀¹u²e^(κu²)du / ∫₀¹e^(κu²)du, u = μ·n uniform on the sphere; below κ = 1 from the two
# power series, where the closed form loses digits to cancellation).
#
# The dispersed stick, with β = b·D∥: A_ic = ∫ exp(nᵀMn) dn / ∫ exp(κ(μ·n)²) dn, M = κμμᵀ − βggᵀ,
# a quadratic form of rank two (in the plane of μ and g). In M's eigenvectors, the third axis z
# perpendicular to that plane, ∫ exp(λ1x² + λ2y²) dn = 2π·∫₋₁¹ e^(r·T/2)·I0(r·q) dz, r = 1 − z²,
# with T = λ1 + λ2 = κ − β and q = (λ1 − λ2)/2 = √(T²/4 + κβ(1 − t²)) (the azimuth's integral
# done exactly: ∫₀²π e^(c·cos 2ψ) dψ = 2π·I0(c)). The z integral by Gauss–Legendre on [0, 1]
# (the integrand is even), numerator and denominator (β = 0) with the same nodes, so A_ic = 1 at
# b = 0 exactly and the quadrature's error divides out at leading order. I0 scaled, e^(−x)·I0(x)
# (`_i01e`, the Rician code's Chebyshev series, with exact derivatives on dual numbers), and every
# exponential relative to e^κ, so nothing overflows at κ up to the ODI bound's 64. Checked
# against a two-dimensional quadrature over the sphere (the tests), and at κ = 0 against the
# stick's spherical mean, √π·erf(√β)/(2√β).
#
# The NODDI toolbox computes A_ic from a spherical-harmonic series of the Watson distribution
# (Zhang 2012, appendix); DIPY and dmipy fit the same model. The quadrature is chosen here for
# being checkable: one integral, exact limits, no series whose terms cancel.
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

The orientation starts from the voxel's diffusion tensor, as [`StickZeppelin`](@ref)'s.

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
    κ = _kappa(ODI)
    μ = fiberdirection(theta, phi)                      # once per voxel
    τ = _watsoncos2(κ)                                  # ⟨(μ·n)²⟩
    den = _watsonnorm(κ)                                # the stick integral at b = 0
    Dperp = (1 - vic) * Dpar                            # the tortuosity limit
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        t = (gx[j], gy[j], gz[j]) ⋅ μ
        β = b[j] * Dpar
        Aic = iszero(b[j]) ? one(β) : _watsonstick(β, t, κ) / den
        gDg = Dperp + (Dpar - Dperp) * (τ * t^2 + (1 - τ) * (1 - t^2) / 2)
        Aec = exp(-b[j] * gDg)
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

derived(::NODDI, p) = (kappa = _kappa(p.ODI),)
datastart(m::NODDI, y, p) = _tensordirection(m, y)
relabel(::NODDI, θ) = _upperdirection(θ)

# κ from ODI = (2/π)·arctan(1/κ): κ = cot(π·ODI/2), 0 at ODI = 1
@inline _kappa(ODI) = cot(oftype(float(ODI), π) * ODI / 2)

# τ(κ) = ⟨(μ·n)²⟩ under the Watson distribution (the header): the two power series below κ = 1,
# Σκᵏ/(k!(2k+3)) / Σκᵏ/(k!(2k+1)) (to k = 20: the terms then below 1e-19), Dawson's above.
function _watsoncos2(κ)
    if κ < 1
        num, den, term = zero(κ), zero(κ), one(κ)
        for k in 0:20
            num += term / (2k + 3)
            den += term / (2k + 1)
            term *= κ / (k + 1)
        end
        return num / den
    end
    s = sqrt(κ)
    return 1 / (2s * dawson(s)) - 1 / (2κ)
end

# The Gauss–Legendre rule on [0, 1] for the stick's z integral, by Newton's method on Pₙ from the
# Chebyshev guesses, made once when EQUIP is precompiled. 16 nodes: the largest relative error
# 2e-8 over κ from 0 to 64, b·D∥ from 0.5 to 10 (b to 5900 s/mm² at D∥ = 1.7e-3) and every
# angle, against 24 nodes, which agree with a 2-D quadrature over the sphere to 1e-14 (measured
# 2026-11-23: 12 nodes 6e-6, 14 8e-8, 20 8e-12): far below any noise, at two thirds of 24's cost.
function _gausslegendre01(n)
    x, w = zeros(n), zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        for _ in 1:100
            p0, p1 = 1.0, z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = n * (z * p1 - p0) / (z^2 - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        p0, p1 = 1.0, z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = n * (z * p1 - p0) / (z^2 - 1)
        x[i] = (1 - z) / 2
        w[i] = 1 / ((1 - z^2) * dp^2)                 # 2/((1 − z²)·P′²), halved for [0, 1]
    end
    return Tuple(x), Tuple(w)
end
const _WATSON_Z, _WATSON_W = _gausslegendre01(16)

# ∫₀¹ e^(r·(T/2 + q) − κ)·[e^(−rq)·I0(rq)] dz, r = 1 − z²: the stick's integral (the header) up to
# the factor 2π·e^κ, which the normalization divides out
@inline function _watsonstick(β, t, κ)
    T = κ - β
    q = sqrt(T^2 / 4 + κ * β * (1 - t^2))
    a = T / 2 + q
    s = zero(a)
    @inbounds for k in eachindex(_WATSON_Z)
        r = 1 - _WATSON_Z[k]^2
        s += _WATSON_W[k] * exp(r * a - κ) * first(_i01e(r * q))
    end
    return s
end
# ... at β = 0 (T/2 + q = κ, q = κ/2): its normalization
@inline function _watsonnorm(κ)
    s = zero(κ)
    @inbounds for k in eachindex(_WATSON_Z)
        r = 1 - _WATSON_Z[k]^2
        s += _WATSON_W[k] * exp((r - 1) * κ) * first(_i01e(r * κ / 2))
    end
    return s
end
