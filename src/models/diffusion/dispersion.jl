# =================================================================================================
# Dispersed views: a compartment's signal averaged over a distribution of fiber directions
#
#   w = watson(κ)               the Watson distribution W(n) ∝ exp(κ·(μ·n)²) about an axis μ
#   w = watson(κ, b, D)         the same, with the stick's series kept for the shells b at D (speed)
#   stick(b, D, w(ξ))           the stick averaged over W, along a gradient at cosine ξ = g·μ
#   zeppelin(b, D∥, D⊥, w(ξ))   the zeppelin averaged over W: exp(−b·D⊥)·stick(b, D∥ − D⊥, w(ξ))
#   B = bingham(κ, β, μ, μβ)    the Bingham distribution B(n) ∝ exp(κ(μ·n)² + β(μβ·n)²), 0 ≤ β ≤ κ
#   stick(b, D, B(g))           the stick averaged over B, along the gradient direction g
#   zeppelin(b, D∥, D⊥, B(g))   the zeppelin, likewise
#   watsontensor(D∥, D⊥, w)     a zeppelin's tensor averaged over W (NODDI's extra-neurite space)
#   binghamtensor(D∥, D⊥, B)    the same over B, a 3×3 tensor (Bingham–NODDI's)
#
# A compartment's view says which orientations its signal is averaged over (compartments.jl): a
# cosine ξ, one direction; `powder`, every direction equally (the spherical mean); and here a
# distribution about an axis. The three are one family: a single direction is Watson at κ → ∞,
# powder is Watson at κ = 0 (the tests check both limits), and Watson is Bingham at β = 0. Every
# view averages the compartment's *signal*, as the spherical mean does: so `zeppelin` with a
# distribution is the zeppelin's signal averaged (as dmipy disperses it), not the signal of the
# zeppelin's averaged tensor. NODDI (Zhang et al., NeuroImage 2012, Eq. 5) uses the second for
# its extra-neurite space: an axially symmetric tensor still, a zeppelin along μ with averaged
# diffusivities, `zeppelin(b, watsontensor(D∥, D⊥, w)..., ξ)`; Bingham–NODDI the same with a
# tensor of three eigenvalues, `binghamtensor`, written as exp(−b·gᵀDg) (EQUIP has no tensor
# compartment, roadmap item 17).
#
# The Watson stick, A(ξ) = ∫ W(n)·exp(−β·(g·n)²) dn, β = b·D:
#   - to β = 12.75 (b = 7500 at D = 1.7e-3) by the Funk–Hecke series of two axially symmetric
#     functions, A(ξ) = Σₗ (2l + 1)/2 · kₗ(β) · wₗ(κ)/w₀(κ) · Pₗ(ξ), l even ≤ 40, with wₗ(κ) =
#     ∫₋₁¹ e^(κ(u² − 1))·Pₗ(u) du the distribution's Legendre coefficients (made by `watson(κ)`,
#     once per voxel) and kₗ(β) = ∫₋₁¹ e^(−βv²)·Pₗ(v) dv the stick's (made per call, or kept per
#     shell by `watson(κ, b, D)`: eight shells, a ninth made each time), both by Gauss–Legendre
#     with the Legendre values at the nodes tabulated once (the distribution's in u = 1 − s², its
#     nodes where it peaks), not by the closed forms, which cancel at small κ and β. The even
#     series is summed in z = 2ξ² − 1: P₂ₙ(ξ) = Pₙ^(0,−1/2)(z), Jacobi polynomials, by Clenshaw's
#     recurrence in 20 steps. Its error grows with β (the stick sharpens): 1e-13 to 8.5, 6e-9 to
#     12.75, 3e-5 at 17 (measured against the quadrature, κ to 64);
#   - above, by quadrature: ∫ exp(nᵀMn) dn with M = κμμᵀ − βggᵀ, of rank two; in its
#     eigenvectors (the third perpendicular to μ and g) the azimuth is exact, 2π·I0, and
#     ∫ = 2π·∫₋₁¹ e^(r·T/2)·I0(r·q) dz, r = 1 − z², T = κ − β, q = √(T²/4 + κβ(1 − ξ²)), by 16
#     Gauss–Legendre nodes, its normalization (β = 0) on the same nodes, every exponential relative
#     to e^κ; I0 scaled (`_i01e`, the Rician code's Chebyshev series, exact derivatives on duals).
# Both checked against a two-dimensional quadrature over the sphere (2e-8); validated for κ ≤ 64
# (NODDI's ODI ≥ 0.01), where the series' coefficients are accurate.
#
# The Bingham stick has no such series (B is not axially symmetric; roadmap): M = κμμᵀ + βμβμβᵀ −
# b·D·ggᵀ is of rank three, its integral from its eigenvalues (λ3 + a1x² + a2y² on the sphere)
# by the same one-dimensional Bessel integral along the third eigenvector; only the smallest
# eigenvalue solved for (the cubic's trigonometric solution; on dual numbers its derivative by
# perturbation), the rest from M's invariants. The tensor's moments ⟨(μ·n)²⟩, ⟨(μβ·n)²⟩,
# ⟨(μ3·n)²⟩ from the normalization's integral with I0 and I1.
#
# A view is made per image (`w(ξ)`: the distribution and the cosine); the distribution per voxel.
# In an `@mrimodel` model, whose body runs per image, `watson(κ)` is then made per image too:
# correct, and slower (as a fiber's direction there).
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# The machinery: quadrature rules, the Watson series, the integrals (the types and views below)
# -------------------------------------------------------------------------------------------------

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

# The Gauss–Legendre rule on [0, 1] for the stick's z integral (Watson's and Bingham's), by Newton's method on Pₙ from the
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

# --- Watson: the dispersed stick by its series (the header) ---------------------------------------
const _WATSON_L = 40                                     # even l to 40
const _WATSON_NL = _WATSON_L ÷ 2 + 1                      # 21 coefficients
const _WATSON_βMAX = 12.75                               # b·D∥ up to which the series is used

# P₀, P₂, …, P₄₀ at the nodes x, times the weights w (a column per node), by the recurrence
function _evenlegendretable(x, w)
    tab = zeros(_WATSON_NL, length(x))
    for k in eachindex(x)
        u = x[k]
        p0, p1 = 1.0, u
        tab[1, k] = w[k]
        for l in 2:_WATSON_L
            p0, p1 = p1, ((2l - 1) * u * p1 - (l - 1) * p0) / l
            iseven(l) && (tab[l ÷ 2 + 1, k] = w[k] * p1)
        end
    end
    return tab
end
# wₗ by 64 nodes in s, u = 1 − s² (du = 2s ds; the even integrand on [0, 1], its factor 2 divides
# out of wₗ/w₀); kₗ by 40 on [0, 1], times 2 for [−1, 1]
const _WATSON_S, _WATSON_SW = _gausslegendre01(64)
const _WATSON_U = map(s -> 1 - s^2, _WATSON_S)
const _WATSON_TAB = _evenlegendretable(_WATSON_U, map((s, w) -> 2s * w, _WATSON_S, _WATSON_SW))
const _STICK_V, _STICK_VW = _gausslegendre01(40)
const _STICK_TAB = _evenlegendretable(_STICK_V, map(w -> 2w, _STICK_VW))

# Σₖ tab[:, k]·f(x[k]): the coefficients for every even l at once (a stack array, no allocation)
@inline function _tablesum(tab, x, f)
    c = zero(MVector{_WATSON_NL,typeof(f(first(x)))})
    @inbounds for k in eachindex(x)
        e = f(x[k])
        for l in 1:_WATSON_NL
            c[l] += tab[l, k] * e
        end
    end
    return SVector(c)
end
# a shell's series coefficients, (2l + 1)/2 · kₗ(β) · wₗ/w₀, in the number type C
@inline function _shellcoeffs(β, w, C)
    k = _tablesum(_STICK_TAB, _STICK_V, v -> exp(-β * v^2))
    return SVector{_WATSON_NL,C}(ntuple(i -> (4(i - 1) + 1) / 2 * k[i] * w[i], Val(_WATSON_NL)))
end
# Jacobi Pₙ^(0,−1/2): Pₙ = (Aₙz + Bₙ)·Pₙ₋₁ − Cₙ·Pₙ₋₂ (the standard recurrence at a = 0, b = −1/2)
_jacobiD(n) = 2n * (n - 0.5) * (2n - 2.5)
const _JA = ntuple(n -> (2n - 1.5) * (2n - 0.5) * (2n - 2.5) / _jacobiD(n), _WATSON_NL + 1)
const _JB = ntuple(n -> -0.25 * (2n - 1.5) / _jacobiD(n), _WATSON_NL + 1)
const _JC = ntuple(n -> 2 * (n - 1) * (n - 1.5) * (2n - 0.5) / _jacobiD(n), _WATSON_NL + 1)
# Σₙ cₙ·Pₙ(z), n = 0 … 20, by Clenshaw's recurrence
@inline function _jacobisum(c, z)
    b1 = zero(z * c[1]); b2 = b1
    @inbounds for k in (_WATSON_NL - 1):-1:1
        b1, b2 = c[k+1] + (_JA[k+1] * z + _JB[k+1]) * b1 - _JC[k+2] * b2, b1
    end
    return c[1] + (_JA[1] * z + _JB[1]) * b1 - _JC[2] * b2
end
# the series at one (β, t, κ), as the signal sums it (for the tests)
function _watsonseries(β, t, κ)
    w0 = _tablesum(_WATSON_TAB, _WATSON_U, u -> exp(κ * (u^2 - 1)))
    w = w0 / w0[1]
    return _jacobisum(_shellcoeffs(β, w, typeof(first(w) * β)), 2t^2 - 1)
end

# --- Watson: the dispersed stick by quadrature (above the series' range; the tests' reference) -----
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

# The Bingham's axes: μ(θ, φ), μβ at ψ about μ from e_θ, μ3 = μ × μβ
@inline function _binghamaxes(theta, phi, psi)
    st, ct, sp, cp = sin(theta), cos(theta), sin(phi), cos(phi)
    μ = SVector(st * cp, st * sp, ct)
    eθ = SVector(ct * cp, ct * sp, -st)
    eφ = SVector(-sp, cp, zero(sp))
    sψ, cψ = sin(psi), cos(psi)
    return μ, cψ * eθ + sψ * eφ, cψ * eφ - sψ * eθ
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

# -------------------------------------------------------------------------------------------------
# The distributions and their views
# -------------------------------------------------------------------------------------------------

"""
    watson(κ)
    watson(κ, b, D)

The Watson distribution of fiber directions, W(n) ∝ exp(κ·(μ·n)²) about an axis μ (κ ≥ 0: 0 is
isotropic, a large κ concentrated about μ; NODDI's ODI = (2/π)·arctan(1/κ)), as a compartment's
view: `w = watson(κ)` once per voxel, then `stick(b, D, w(ξ))` and `zeppelin(b, D∥, D⊥, w(ξ))`
for an image whose gradient makes the cosine ξ = g·μ with the axis: the compartment's signal
averaged over the distribution. `watson(κ, b, D)` also keeps the stick's series for the b-values
`b` at diffusivity `D` (a model's acquisition and its stick's D), which a model loops over: then
`stick(b[j], D, w(ξ))` costs a short sum. Accurate (2e-8) for κ up to 64 and b·D to 12.75, and
by quadrature above.

```julia
w = watson(5.0)                          # about an axis μ
stick(1000.0, 1.7e-3, w(0.8))            # the dispersed stick, the gradient at cosine 0.8 to μ
stick(1000.0, 1.7e-3, watson(0.0)(0.8))  # = stick(1000.0, 1.7e-3, powder): κ = 0 is isotropic
```
"""
function watson end

# The distribution: κ, wₗ/w₀ (l even to 40), τ = ⟨(μ·n)²⟩, and the shells kept (b-values `bs`, the
# series coefficients `cs`, at diffusivity `D`; `n` of them, none for `watson(κ)`)
struct Watson{T,R,S,B,C}
    κ::T
    w::SVector{_WATSON_NL,R}          # (R may be wider than T: Float64 nodes in a Float32 fit)
    τ::T
    D::S
    n::Int
    bs::NTuple{8,B}
    cs::NTuple{8,SVector{_WATSON_NL,C}}
end
# A view: the distribution, and the cosine between the gradient and its axis
struct WatsonView{W<:Watson,X}
    dist::W
    ξ::X
end
@inline (w::Watson)(ξ) = WatsonView(w, ξ)

# The distribution's Legendre coefficients wₗ/w₀
@inline function _watsoncoeffs(κ)
    w0 = _tablesum(_WATSON_TAB, _WATSON_U, u -> exp(κ * (u^2 - 1)))
    return w0 / w0[1]
end

@inline function watson(κ)
    w = _watsoncoeffs(κ)
    return Watson(κ, w, _watsoncos2(κ), nothing, 0, ntuple(_ -> 0.0, Val(8)), ntuple(_ -> zero(w), Val(8)))
end
@inline function watson(κ, b::AbstractVector, D)
    w = _watsoncoeffs(κ)
    C = typeof(first(w) * D)
    # the shells as tuples, each new one set in place (Base.setindex: a new tuple, on the stack; an
    # MVector converted to a tuple at the end was put on the heap, 80 bytes a voxel)
    bs = ntuple(_ -> zero(eltype(b)), Val(8))
    cs = ntuple(_ -> zero(SVector{_WATSON_NL,C}), Val(8))
    n = 0
    @inbounds for bj in b
        n == 8 && break
        (iszero(bj) || abs(bj * D) > _WATSON_βMAX) && continue
        seen = false
        for s in 1:n
            bs[s] == bj && (seen = true; break)
        end
        seen && continue
        n += 1
        bs = Base.setindex(bs, bj, n)
        cs = Base.setindex(cs, _shellcoeffs(bj * D, w, C), n)
    end
    return Watson(κ, w, _watsoncos2(κ), D, n, bs, cs)
end

# The stick averaged over W (the header): 1 at b = 0; the series, its coefficients kept or made;
# the quadrature above β = 12.75
@inline function stick(b, D, v::WatsonView)
    w, ξ = v.dist, v.ξ
    β = b * D
    iszero(b) && return one(ξ * β * first(w.w))
    if abs(β) <= _WATSON_βMAX
        return _jacobisum(_watsonshell(w, b, D, β), 2ξ^2 - 1)
    end
    return _watsonstick(β, ξ, w.κ) / _watsonnorm(w.κ)
end
@inline function _watsonshell(w::Watson{T,R,S,B,C}, b, D, β) where {T,R,S,B,C}
    if S !== Nothing && D == w.D
        @inbounds for s in 1:w.n
            w.bs[s] == b && return w.cs[s]
        end
    end
    return _shellcoeffs(β, w.w, typeof(first(w.w) * β))
end
# the zeppelin averaged over W: every direction sees D⊥, the excess D∥ − D⊥ is a stick's
@inline zeppelin(b, Dpar, Dperp, v::WatsonView) = exp(-b * Dperp) * stick(b, Dpar - Dperp, v)

"""
    watsontensor(D∥, D⊥, w) -> (D∥ₑ, D⊥ₑ)

A zeppelin's diffusion tensor averaged over the Watson distribution `w` ([`watson`](@ref)): again a
zeppelin along the axis, D∥ₑ = D⊥ + (D∥ − D⊥)·τ and D⊥ₑ = D⊥ + (D∥ − D⊥)·(1 − τ)/2, τ = ⟨(μ·n)²⟩
(Zhang et al., NeuroImage 2012, Eq. 5: NODDI's extra-neurite space, `zeppelin(b, D∥ₑ, D⊥ₑ, ξ)`).
Not the zeppelin's signal averaged over `w`, `zeppelin(b, D∥, D⊥, w(ξ))`: the approximation NODDI
makes, written where it is used.
"""
@inline watsontensor(Dpar, Dperp, w::Watson) = (Dperp + (Dpar - Dperp) * w.τ, Dperp + (Dpar - Dperp) * (1 - w.τ) / 2)

"""
    bingham(κ, β, μ, μβ)

The Bingham distribution of fiber directions, B(n) ∝ exp(κ·(μ·n)² + β·(μβ·n)²), about the axis μ
and fanning toward μβ (a unit vector perpendicular to μ), 0 ≤ β ≤ κ: β = 0 is [`watson`](@ref)`(κ)`,
β = κ spread evenly in the plane of μ and μβ (Tariq et al., NeuroImage 2016). As a view:
`B = bingham(κ, β, μ, μβ)` once per voxel, then `stick(b, D, B(g))` and `zeppelin(b, D∥, D⊥, B(g))`
for an image with gradient direction `g` (three components): the compartment's signal averaged
over the distribution. By quadrature, about 0.6 µs an image.
"""
function bingham end

# The distribution: κ, β, its axes, its form Q = κμμᵀ + βμβμβᵀ, the normalization, the moments
struct Bingham{T,V,Q,R}
    κ::T
    β::T
    μ::V
    μβ::V
    μ3::V
    Q::Q
    den::R                            # (R may be wider than T: Float64 nodes in a Float32 fit)
    m::NTuple{3,R}
end
struct BinghamView{B<:Bingham,G}
    dist::B
    g::G
end
@inline (B::Bingham)(g) = BinghamView(B, SVector{3}(g[1], g[2], g[3]))

@inline function bingham(κ, β, μ, μβ)
    κ, β = promote(κ, β)
    μv, μβv = SVector{3}(μ[1], μ[2], μ[3]), SVector{3}(μβ[1], μβ[2], μβ[3])
    den, mx, my, mz = _binghammoments(κ, β)
    return Bingham(κ, β, μv, μβv, cross(μv, μβv), κ * μv * μv' + β * μβv * μβv', den, (mx, my, mz))
end
@inline function stick(b, D, v::BinghamView)
    B, g = v.dist, v.g
    iszero(b) && return one(B.den * b * D)
    return _binghamstick(B.Q - (b * D) * g * g', B.κ) / B.den
end
@inline zeppelin(b, Dpar, Dperp, v::BinghamView) = exp(-b * Dperp) * stick(b, Dpar - Dperp, v)

"""
    binghamtensor(D∥, D⊥, B) -> 3×3 tensor

A zeppelin's diffusion tensor averaged over the Bingham distribution `B` ([`bingham`](@ref)):
D⊥·I + (D∥ − D⊥)·(⟨x²⟩·μμᵀ + ⟨y²⟩·μβμβᵀ + ⟨z²⟩·μ3μ3ᵀ), the moments of B along its axes; three
eigenvalues, so a tensor and not a zeppelin: its signal exp(−b·gᵀDg) (Bingham–NODDI's
extra-neurite space, as NODDI's `watsontensor`).
"""
@inline function binghamtensor(Dpar, Dperp, B::Bingham)
    mx, my, mz = B.m
    return Dperp * I + (Dpar - Dperp) * (mx * B.μ * B.μ' + my * B.μβ * B.μβ' + mz * B.μ3 * B.μ3')
end
