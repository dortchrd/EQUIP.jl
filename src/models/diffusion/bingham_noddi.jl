# =================================================================================================
# Bingham–NODDI: NODDI with anisotropic orientation dispersion (Tariq et al., NeuroImage
# 2016;133:207), the neurites fanning more in one plane than across it
#
#   S    = S0·[(1 − viso)·(vic·A_ic + (1 − vic)·A_ec) + viso·exp(−b·Diso)]      NODDI's (noddi.jl)
#   B(n) ∝ exp(κ·(μ·n)² + β·(μβ·n)²),  0 ≤ β ≤ κ                         the Bingham distribution
#
# μ(θ, φ) the mean orientation, μβ ⊥ μ the direction the neurites fan toward (the angle ψ about
# μ), μ3 = μ × μβ the third axis. β = 0 is Watson (NODDI: the same in every plane through μ); as β
# grows the distribution spreads in the plane of μ and μβ, and at β = κ it is uniform in that plane
# (a girdle about μ3). The parameterization is dmipy's (Fick et al., Neuroimage 2019; its
# `SD2Bingham`: κ = cot(π·ODI/2), β = betafraction·κ, ψ about μ); Tariq 2016 is its source.
#
# The dispersed stick, with b·D∥ = s: A_ic = ∫ exp(nᵀMn) dn / ∫ exp(nᵀM₀n) dn, M = κμμᵀ + βμβμβᵀ −
# s·ggᵀ, M₀ = M at s = 0. A quadratic form on the sphere depends only on M's eigenvalues λ1 ≥ λ2 ≥
# λ3: nᵀMn = λ3 + a1·x² + a2·y² (a = λ − λ3 ≥ 0, x, y along the first two eigenvectors), so, with the
# third eigenvector as the axis z and r = 1 − z²,
#   ∫ exp(nᵀMn) dn = 2π·e^λ3·∫₋₁¹ e^(r(a1 + a2)/2)·I0(r(a1 − a2)/2) dz
# NODDI's integral (noddi.jl) with a rank-three form: the azimuth exact (2π·I0), the z integral by
# the same 16 Gauss–Legendre nodes, numerator and normalization (eigenvalues κ, β, 0) on them, every
# exponential relative to e^κ (λ1 ≤ κ). Per image only the smallest eigenvalue is solved for (the
# cubic's trigonometric solution; on dual numbers its derivative by first-order perturbation), the
# others entering through M's invariants (`_binghamstick`).
#
# The dispersed zeppelin, as NODDI's, by the tensor averaged over B (Zhang 2012, Eq. 5):
#   gᵀD_ec·g = D⊥ + (D∥ − D⊥)·[⟨x²⟩·(g·μ)² + ⟨y²⟩·(g·μβ)² + ⟨z²⟩·(g·μ3)²],   D⊥ = (1 − vic)·D∥
# its moments along the three axes from the normalization's integral (z along μ3):
#   ⟨z²⟩ = ∫z²·e^(r(κ+β)/2)·I0 dz / ∫e^(r(κ+β)/2)·I0 dz,   ⟨x²⟩ − ⟨y²⟩ = ∫r·e^(r(κ+β)/2)·I1 dz / (…)
# (I0, I1 at r(κ − β)/2; ∫₀²π cos 2ψ·e^(c·cos 2ψ) dψ = 2π·I1(c)), and ⟨x²⟩ + ⟨y²⟩ = 1 − ⟨z²⟩.
#
# Checked against a two-dimensional quadrature over the sphere, and against NODDI at β = 0 (the
# tests). About as costly as NODDI's quadrature (16 Bessel functions an image) plus the eigenvalues:
# there is no Legendre series here, since B is not axially symmetric (docs/design.md, item 7).
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

parameters(::Type{<:BinghamNODDI}) = (
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

# The Bingham's axes: μ(θ, φ), μβ at ψ about μ from e_θ, μ3 = μ × μβ
@inline function _binghamaxes(theta, phi, psi)
    st, ct, sp, cp = sin(theta), cos(theta), sin(phi), cos(phi)
    μ = SVector(st * cp, st * sp, ct)
    eθ = SVector(ct * cp, ct * sp, -st)
    eφ = SVector(-sp, cp, zero(sp))
    sψ, cψ = sin(psi), cos(psi)
    return μ, cψ * eθ + sψ * eφ, cψ * eφ - sψ * eθ
end

function signal!(out, m::BinghamNODDI, p)
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, betafraction, theta, phi, psi, Dpar, Diso) = p
    κ = _kappa(ODI)
    β = betafraction * κ
    μ, μβ, μ3 = _binghamaxes(theta, phi, psi)
    den, mx2, my2, mz2 = _binghammoments(κ, β)                 # normalization; ⟨x²⟩, ⟨y²⟩, ⟨z²⟩
    Q = κ * μ * μ' + β * μβ * μβ'                               # the distribution's form, M₀
    Dperp = (1 - vic) * Dpar
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        g = SVector(gx[j], gy[j], gz[j])
        s = b[j] * Dpar
        Aic = iszero(b[j]) ? one(den) : _binghamstick(Q - s * g * g', κ) / den
        gDg = Dperp + (Dpar - Dperp) * (mx2 * (g ⋅ μ)^2 + my2 * (g ⋅ μβ)^2 + mz2 * (g ⋅ μ3)^2)
        Aec = exp(-b[j] * gDg)
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

# ∫ exp(nᵀMn) dn / (2π·e^κ) for a symmetric 3×3 M with eigenvalues at most κ (the header). Only the
# smallest eigenvalue λ3 is solved for; the rest from M's invariants, tr = λ1 + λ2 + λ3 and c2 (the
# sum of its principal 2×2 minors, λ1λ2 + λ1λ3 + λ2λ3): h = (a1 + a2)/2 = (tr − 3λ3)/2 and
# q² = ((a1 − a2)/2)² = h² − a1·a2, a1·a2 = c2 − 2λ3·tr + 3λ3² (q enters through I0, even in q).
@inline function _binghamstick(M, κ)
    λ3 = _smallesteig(M)
    tr = M[1, 1] + M[2, 2] + M[3, 3]
    c2 = M[1, 1] * M[2, 2] + M[1, 1] * M[3, 3] + M[2, 2] * M[3, 3] - M[1, 2]^2 - M[1, 3]^2 - M[2, 3]^2
    h = (tr - 3λ3) / 2
    Q = max(h^2 - (c2 - 2λ3 * tr + 3λ3^2), zero(h))             # q²
    s = zero(h)
    if _plain(Q) < 1e-12
        # q at 0 (two eigenvalues equal at the top: ODI = 1, or b = 0): I0 by its series in q²,
        # e^(rh)·I0(rq) = e^(rh)·(1 + r²q²/4 + r⁴q⁴/64 + …), so the derivatives stay finite (√q²
        # has none at 0)
        @inbounds for k in eachindex(_WATSON_Z)
            r = 1 - _WATSON_Z[k]^2
            s += _WATSON_W[k] * exp(λ3 + r * h - κ) * (1 + r^2 * Q / 4 + r^4 * Q^2 / 64)
        end
        return s
    end
    q = sqrt(Q)
    @inbounds for k in eachindex(_WATSON_Z)
        r = 1 - _WATSON_Z[k]^2
        s += _WATSON_W[k] * exp(λ3 + r * (h + q) - κ) * first(_i01e(r * q))
    end
    return s
end
# a number's value through any layers of dual numbers
@inline _plain(x::Dual) = _plain(value(x))
@inline _plain(x::Real) = x

# The smallest eigenvalue of a symmetric 3×3 matrix: the trigonometric solution of the cubic
# (Smith, Commun ACM 1961). On dual numbers, the value's, and its derivative by first-order
# perturbation, dλ3 = u3ᵀ·dM·u3 (u3 its unit eigenvector, from the cross products of the rows of
# M − λ3·I, the largest taken; where λ3 is a double eigenvalue every vector of the pair's plane is
# one, and the stick's integral, a symmetric function of the eigenvalues, does not depend on which).
@inline function _smallesteig(M::SMatrix{3,3,<:Real})
    q = (M[1, 1] + M[2, 2] + M[3, 3]) / 3
    p1 = M[1, 2]^2 + M[1, 3]^2 + M[2, 3]^2
    p2 = (M[1, 1] - q)^2 + (M[2, 2] - q)^2 + (M[3, 3] - q)^2 + 2p1
    iszero(p2) && return q                                       # M = q·I
    p = sqrt(p2 / 6)
    B = (M - q * I) / p
    r = clamp(det(B) / 2, -one(q), one(q))
    return q + 2p * cos(acos(r) / 3 + 2 * oftype(q, π) / 3)
end
@inline function _smallesteig(M::SMatrix{3,3,Dual{Tg,V,N}}) where {Tg,V,N}
    Mv = map(value, M)
    λ = _smallesteig(Mv)
    A = Mv - λ * I
    r1, r2, r3 = A[1, :], A[2, :], A[3, :]
    c1, c2, c3 = cross(r1, r2), cross(r1, r3), cross(r2, r3)
    n1, n2, n3 = sum(abs2, c1), sum(abs2, c2), sum(abs2, c3)
    c = n1 >= n2 && n1 >= n3 ? c1 : n2 >= n3 ? c2 : c3
    nc = max(n1, n2, n3)
    u = iszero(nc) ? _anyeigvec(Mv, λ) : c / sqrt(nc)
    return Dual{Tg}(λ, ntuple(k -> u' * map(x -> partials(x, k), M) * u, Val(N)))
end
# where every cross product vanishes (λ3 a double or triple eigenvalue), the decomposition's own
_anyeigvec(Mv, λ) = (U = last(_symeigen(Mv)); U[:, 1])

# The normalization (eigenvalues κ, β, 0: z along μ3), and the moments ⟨x²⟩, ⟨y²⟩, ⟨z²⟩ along μ,
# μβ, μ3, from the same nodes (the header)
@inline function _binghammoments(κ, β)
    h, q = (κ + β) / 2, (κ - β) / 2
    qa = abs(q)                                                  # q ≥ 0 within the bounds (β ≤ κ)
    s0 = sz = sd = zero(h)
    @inbounds for k in eachindex(_WATSON_Z)
        z = _WATSON_Z[k]
        r = 1 - z^2
        e = _WATSON_W[k] * exp(r * (h + qa) - κ)
        x = r * qa
        i0, i1 = _i01e(x)
        s0 += e * i0
        sz += e * z^2 * i0
        # I1(rq) = rq·[I1(x)/x], the ratio even: right in sign and derivative at q = 0 (β = κ),
        # where it is the series e^(−x)·(1/2 + x²/16 + x⁴/384)
        ratio = _plain(x) < 1e-3 ? exp(-x) * (one(x) / 2 + x^2 / 16 + x^4 / 384) : i1 / x
        sd += e * r * (r * q) * ratio
    end
    mz2 = sz / s0
    d = sd / s0                                                  # ⟨x²⟩ − ⟨y²⟩
    return s0, (1 - mz2 + d) / 2, (1 - mz2 - d) / 2, mz2
end

function derived(::BinghamNODDI, p)
    κ = _kappa(p.ODI)
    β = p.betafraction * κ
    return (kappa = κ, beta = β, ODIfan = 2 / oftype(κ, π) * atan(1 / (κ - β)),
            DA = 2 / oftype(κ, π) * atan(β / (κ - β)))
end
# The start from the voxel's tensor: μ its principal eigenvector, μβ its second (a fan spreads the
# diffusion most along μ, then along the fan); ψ the second's angle about μ
function datastart(m::BinghamNODDI, y, p)
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
function relabel(::BinghamNODDI, θ)
    haskey(θ, :theta) && haskey(θ, :phi) && haskey(θ, :psi) || return θ
    μ, μβ, _ = _binghamaxes(θ.theta, θ.phi, θ.psi)
    u = _upperhemisphere(μ)
    t, f = acos(clamp(u[3], -one(u[3]), one(u[3]))), atan(u[2], u[1])
    _, eθ, eφ = _binghamaxes(t, f, zero(t))
    return merge(θ, (theta = t, phi = f, psi = mod(atan(μβ ⋅ eφ, μβ ⋅ eθ), oftype(t, π))))
end
