# =================================================================================================
# DKI: diffusion kurtosis imaging (Jensen et al., Magn Reson Med 2005;53:1432), written in DIPY's
# parameterization
#
#   S = S0·exp(−b·gᵀDg + b²·MD²·Σᵢⱼₖₗ gᵢgⱼgₖgₗWᵢⱼₖₗ / 6)
#     = S0·exp(−b·gᵀDg + b²·Σᵢⱼₖₗ gᵢgⱼgₖgₗVᵢⱼₖₗ / 6),   V = MD²·W
#
# D is the diffusion tensor (six distinct elements, mm²/s), W the kurtosis tensor (fully
# symmetric: 15 distinct elements of its 81, no units), MD = tr(D)/3. Along one direction it is
# ADCKurtosis with ADC = gᵀDg and K = MD²·gᵀWg/ADC². The fit is in V = MD²·W, as DIPY's
# `dki_design_matrix` writes it: ln S is then linear in all 22 parameters, so the `:loglinear`
# start is the two-pass WLS exactly, and the kurtosis terms do not carry D's derivatives through
# MD² (1.8× faster than in W, the same fits: docs/design.md, "DKI in DIPY's parameterization").
# Nobody reads W's elements; W = V/MD² is computed with the indices.
#
# The indices (`derived`) are DIPY's, computed as its fit methods compute them by default
# (`fit.mk()`, `fit.ak()`, `fit.rk()`, `fit.mkt()`, DIPY 1.12; checked against its values in
# the tests, MK to DIPY's own error, below): the mean kurtosis MK, the mean of K(n) over all
# directions, and the radial RK, its mean over the directions perpendicular to D's principal
# one, in the closed forms of Tabesh et al. (Magn Reson Med 2011;65:823), with Carlson's
# elliptic integrals R_F and R_D and DIPY's limits where eigenvalues are (nearly) equal (G1 and
# G2 rewritten without their division, exactly: they lost digits near equal eigenvalues); the
# axial AK, K along the principal direction; the mean kurtosis tensor MKT (Hansen et al., Magn
# Reson Med 2013;69:1754), linear in W. Each clipped to [−3/7, 10], DIPY's defaults (its reason:
# large negative kurtosis is not physically plausible and makes artifacts in the kurtosis
# measures; the upper limit keeps them in a plausible range); MK, AK and RK are 0 where an
# eigenvalue of D is not positive, as DIPY's.
# Beside them, D's own (DTI's): MD, FA, AD, RD and the principal eigenvector.
#
# D's diagonal elements are from 0 to 4e-3 mm²/s, as DTI's (not negative, as a diffusivity;
# `_DMAX`, src/models/builtin.jl: above free water; the owner, 2026-11-09). The constraints that box bounds cannot state (K ≥ 0 in
# every direction, the signal not rising with b) are not imposed: check them on the results (examples/dki.jl), or mask; in CSF the
# protocol has no information on kurtosis (docs/design.md, item 17's review).
# =================================================================================================

"""
    DKI(; b, g)

Diffusion kurtosis imaging (Jensen et al., Magn Reson Med 2005), in DIPY's parameterization,
V = MD²·W:

    S = S0·exp(−b·gᵀDg + b²·Σᵢⱼₖₗ gᵢgⱼgₖgₗVᵢⱼₖₗ / 6)

- acquisition: `b` (s/mm²) and the gradient direction `g` (a unit vector) of every image, from a
  .bval and a .bvec by [`diffusionencoding`](@ref); at least two shells besides b = 0 (e.g.
  1000 and 2000 s/mm²) and 15 directions on the highest
- free by default: `S0`, D's six elements `Dxx`, `Dyy`, `Dzz`, `Dxy`, `Dxz`, `Dyz` (mm²/s), and
  V's fifteen, `V1111`, `V2222`, `V3333`, `V1112`, `V1113`, `V1222`, `V2223`, `V1333`, `V2333`,
  `V1122`, `V1133`, `V2233`, `V1123`, `V1223`, `V1233` ((mm²/s)²; W = V/MD²). D's diagonal
  elements from 0 to 4e-3 mm²/s, as [`DTI`](@ref)'s (for b in s/mm²; `Dxx = Free(upper = Inf)`
  and the others remove the cap); no other bounds.
  Every element starts from the weighted log-linear fit (`:loglinear`, the two-pass WLS: in V,
  ln S is linear in all of them)

Derived ([`derived`](@ref), written by [`writemaps`](@ref)): MD, FA, AD, RD and the principal
eigenvector (V1x, V1y, V1z), as DTI's; MK, AK, RK as DIPY's fit methods give them by default
(Tabesh et al., Magn Reson Med 2011, analytic), and MKT (Hansen et al., Magn Reson Med 2013),
each clipped to [−3/7, 10]; MK, AK and RK 0 where an eigenvalue is not positive.

A truth for simulation from derived quantities ([`nominal`](@ref), [`fromderived`](@ref)): D as
DTI's (axially symmetric, from `MD` and `FA` or `AD` and `RD`, and `direction`), and W from `MKT`
alone (isotropic) or from `MK`, `AK` and `RK` (axially symmetric about the same axis), or V's
elements as parameters: `nominal(DKI; S0 = 1000, AD = 1.7e-3, RD = 0.35e-3, direction = (1, 0, 0),
MK = 1.0, AK = 0.8, RK = 1.3)`.

```julia
acq = diffusionencoding("dwi.bval", "dwi.bvec")
r = fit(DKI(; acq...), dwi; mask)
d = derived(r)           # (MD = ..., FA = ..., ..., MK = ..., AK = ..., RK = ..., MKT = ...)
```
"""
@kwdef struct DKI{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
    b::A                                # b-values (s/mm²)
    g::G                                # gradient direction of each image (a unit SVector{3})
    DKI(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))   # checked (directions.jl)
end

parameters(::Type{<:DKI}) = (
    S0 = Free(start = :scale, lower = 0),
    # D's six elements (mm²/s), then V's fifteen ((mm²/s)²), all from the two-pass WLS
    Dxx = Free(start = :loglinear, lower = 0, upper = _DMAX),
    Dyy = Free(start = :loglinear, lower = 0, upper = _DMAX),
    Dzz = Free(start = :loglinear, lower = 0, upper = _DMAX),
    Dxy = Free(start = :loglinear), Dxz = Free(start = :loglinear), Dyz = Free(start = :loglinear),
    V1111 = Free(start = :loglinear), V2222 = Free(start = :loglinear), V3333 = Free(start = :loglinear),
    V1112 = Free(start = :loglinear), V1113 = Free(start = :loglinear), V1222 = Free(start = :loglinear),
    V2223 = Free(start = :loglinear), V1333 = Free(start = :loglinear), V2333 = Free(start = :loglinear),
    V1122 = Free(start = :loglinear), V1133 = Free(start = :loglinear), V2233 = Free(start = :loglinear),
    V1123 = Free(start = :loglinear), V1223 = Free(start = :loglinear), V1233 = Free(start = :loglinear),
)

# Σᵢⱼₖₗ gᵢgⱼgₖgₗVᵢⱼₖₗ over all 81 terms is a sum over the 15 distinct elements, each times its
# monomial and the number of index orders that give it: 1 for V1111, 4 for V1112 (1112, 1121,
# 1211, 2111), 6 for V1122, 12 for V1123. These are the 15 monomials with their multiplicities, in
# the order of the parameters above (DIPY's order too). A static vector: no allocation. It takes
# the direction itself, g = (x, y, z), a static vector or a tuple (as `_tensorweights`, dti.jl).
@inline function _kurtosisweights(g)
    x, y, z = g
    return SVector(x^4, y^4, z^4,
                   4x^3 * y, 4x^3 * z, 4x * y^3, 4y^3 * z, 4x * z^3, 4y * z^3,
                   6x^2 * y^2, 6x^2 * z^2, 6y^2 * z^2,
                   12x^2 * y * z, 12x * y^2 * z, 12x * y * z^2)
end

# V's fifteen elements, in that order (for `derived`, which takes the parameters whole).
@inline _kurtosisvector(p) = SVector(p.V1111, p.V2222, p.V3333, p.V1112, p.V1113, p.V1222, p.V2223,
                                     p.V1333, p.V2333, p.V1122, p.V1133, p.V2233, p.V1123, p.V1223, p.V1233)

function signal!(out, m::DKI, p)
    (; b, g) = m
    (; S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz,
       V1111, V2222, V3333, V1112, V1113, V1222, V2223, V1333, V2333, V1122, V1133, V2233, V1123, V1223, V1233) = p
    D = SVector(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)                       # in _tensorweights' order
    V = SVector(V1111, V2222, V3333, V1112, V1113, V1222, V2223, V1333, V2333,
                V1122, V1133, V2233, V1123, V1223, V1233)           # in _kurtosisweights' order
    @inbounds for j in eachindex(out, b, g)
        gDg = inner(_tensorweights(g[j]), D)            # image j's gradient direction: gᵀDg (dti.jl)
        gVg = inner(_kurtosisweights(g[j]), V)          #   and Σ gᵢgⱼgₖgₗ·Vᵢⱼₖₗ
        out[j] = S0 * exp(-b[j] * gDg + b[j]^2 * gVg / 6)
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# The indices
# -------------------------------------------------------------------------------------------------

const _KMIN = -3 / 7       # DIPY's range for K in MK, AK, RK and MKT (its fit methods' defaults)
const _KMAX = 10.0
const _EVALMIN = 2e-7      # mm²/s: below it an eigenvalue is not positive (DIPY's `_positive_evals`)

function derived(::DKI, p)
    q, λa, R = _tensorindices(_tensor(p))          # D's quantities, as DTI's; eigen, ascending
    MD = q.MD
    W = _kurtosisvector(p) / MD^2                   # the kurtosis tensor's 15 elements
    # MKT (Hansen et al. 2013): the mean of W(n) over all directions, linear in W
    MKT = (W[1] + W[2] + W[3] + 2 * (W[10] + W[11] + W[12])) / 5
    MK, AK, RK = _kurtosisindices(λa, R, W, MD)
    clip(k) = clamp(k, _KMIN, _KMAX)
    return merge(q, (MK = clip(MK), AK = clip(AK), RK = clip(RK), MKT = clip(MKT)))
end

# MK, AK and RK of D (eigenvalues λa ascending, eigenvectors the columns of R) and W, before
# clipping: linear in W for a fixed D (which `fromderived` uses to solve for W).
function _kurtosisindices(λa, R, W, MD)
    # D's eigenframe in DIPY's order, λ1 ≥ λ2 ≥ λ3, e1 the principal direction (an eigenvector's
    # sign does not matter here: W's elements in the frame are even in each). Where λ1 = λ2 the
    # principal direction is not defined, and AK and RK depend on which vector of the plane the
    # eigensolver returns: they then differ from DIPY's (numpy's choice), as they would between
    # any two programs; MK and MKT do not depend on the frame.
    λ1, λ2, λ3 = λa[3], λa[2], λa[1]
    e1, e2, e3 = R[:, 3], R[:, 2], R[:, 1]
    # W in the eigenframe: Ŵᵢᵢᵢᵢ = W(eᵢ, eᵢ, eᵢ, eᵢ), the quartic form, and Ŵᵢᵢⱼⱼ = W(eᵢ, eᵢ, eⱼ, eⱼ),
    # by polarization of the quartic form: W(u,u,v,v) = [W₄(u+v) + W₄(u−v) − 2W₄(u) − 2W₄(v)]/12
    W4(n) = inner(_kurtosisweights(n), W)
    Wuuvv(u, v) = (W4(u + v) + W4(u - v) - 2W4(u) - 2W4(v)) / 12
    W1111, W2222, W3333 = W4(e1), W4(e2), W4(e3)
    W1122, W1133, W2233 = Wuuvv(e1, e2), Wuuvv(e1, e3), Wuuvv(e2, e3)
    if λ1 > _EVALMIN && λ2 > _EVALMIN && λ3 > _EVALMIN
        # MK (Tabesh et al. 2011; DIPY's `mean_kurtosis`)
        MK = _F1(λ1, λ2, λ3) * W1111 + _F1(λ2, λ1, λ3) * W2222 + _F1(λ3, λ2, λ1) * W3333 +
             _F2(λ1, λ2, λ3) * W2233 + _F2(λ2, λ1, λ3) * W1133 + _F2(λ3, λ2, λ1) * W1122
        AK = W1111 * MD^2 / λ1^2                    # K along e1
        RK = _G1(λ1, λ2, λ3) * W2222 + _G1(λ1, λ3, λ2) * W3333 + _G2(λ1, λ2, λ3) * W2233
    else
        MK = AK = RK = zero(W1111)                  # DIPY's, where an eigenvalue is not positive
    end
    # Derivatives where two of D's eigenvalues are equal (`_symeigen`: the pair's eigenvectors
    # then do not turn within their plane): right for what does not depend on the basis chosen
    # in that plane, which MK and RK do not when W is symmetric about the third axis in it (the
    # truths `fromderived` makes, D and W axially symmetric); otherwise NaN, as AK's and RK's
    # where e1 is not defined (λ1 = λ2). Checked only where a pair is equal (`&&`).
    top, bottom = _equaleigen(λa, 2, 3), _equaleigen(λa, 1, 2)    # λ1 = λ2; λ2 = λ3
    sym23 = !bottom || _symmetricin(W4, Wuuvv, e1, e2, e3)
    sym12 = !top || _symmetricin(W4, Wuuvv, e3, e1, e2)
    return _nopartials(MK, !(sym23 && sym12)), _nopartials(AK, top), _nopartials(RK, top || !sym23)
end

# Is W symmetric about k in the plane of u and v (k, u, v orthonormal), as far as MK and RK see
# it: its quartic form constant around that plane's circle, and W(k, k, n, n) too? On the circle
# n(φ) = cos φ·u + sin φ·v a quartic form has Fourier terms 0, 2 and 4 in φ, a quadratic 0 and 2:
# constant when equal at 5 and at 3 angles spread over half a turn. To 1e-6 of their size.
function _symmetricin(W4, Wuuvv, k, u, v)
    n(φ) = cos(φ) * u + sin(φ) * v
    q = ntuple(i -> W4(n((i - 1) * π / 5)), 5)
    m = ntuple(i -> Wuuvv(k, n((i - 1) * π / 3)), 3)
    tol = 1e-6 * max(maximum(abs, q), maximum(abs, m))
    return all(x -> abs(x - q[1]) <= tol, q) && all(x -> abs(x - m[1]) <= tol, m)
end

# The reverse, for `nominal` (`fromderived`): a truth from derived quantities. D as DTI's, axially
# symmetric about `direction` (from MD and FA, or AD and RD), and W from
#   - MKT alone: W isotropic, W = MKT·(δᵢⱼδₖₗ + δᵢₖδⱼₗ + δᵢₗδⱼₖ)/3, its quartic form MKT·|g|⁴
#     (K(n) = MD²·MKT/D(n)², so MK ≠ MKT where D is anisotropic: examples/dki.jl);
#   - MK, AK and RK: W axially symmetric about the same axis, three numbers in the axis' frame,
#     W∥ = Ŵ1111, W⊥ = Ŵ2222 = Ŵ3333 = 3Ŵ2233 and W∥⊥ = Ŵ1122 = Ŵ1133, its quartic form
#     W∥ξ⁴ + 6W∥⊥ξ²ρ² + W⊥ρ⁴ (ξ = g·n, ρ² = |g|² − ξ²). For a fixed D, MK, AK and RK are linear
#     in W (`_kurtosisindices`), so the three come from a 3×3 solve: AK from W∥ alone, RK from
#     W⊥ alone, MK from all three.
#   - neither: V's fifteen elements from the call (`V1111 = ...`).
# V = MD²·W. The values must be within DIPY's range [−3/7, 10] (`derived` clips them), and MK,
# AK, RK need RD above `_EVALMIN` (they are 0 below it).
fromderived(::Type{<:DKI}) = (:MD, :FA, :AD, :RD, :direction, :MK, :AK, :RK, :MKT)
function fromderived(::Type{<:DKI}, q::NamedTuple)
    kurt = (:MK, :AK, :RK, :MKT)
    t = _axialtensor(DKI, q)
    given = Tuple(n for n in kurt if haskey(q, n))
    isempty(given) && return t.elements
    sets = "MKT (W isotropic), or MK, AK and RK (W axially symmetric about the direction)"
    given in ((:MKT,), (:MK, :AK, :RK)) ||
        throw(ArgumentError("nominal(DKI; ...): the kurtosis: give $sets; got " *
            "$(join(given, ", "))" * (given == (:MK, :RK) ? ": give AK too (the `DKI_` tissue " *
            "presets have MK and RK only: Lätt et al. 2013 did not report AK)" : "")))
    for k in given
        _KMIN <= q[k] <= _KMAX || throw(ArgumentError("nominal(DKI; ...): $k = $(q[k]) is " *
            "outside [−3/7, 10], DIPY's range, where `derived` clips it"))
    end
    if given == (:MKT,)
        a = b = float(q.MKT)
        c = a / 3
    else
        t.RD > _EVALMIN || throw(ArgumentError("nominal(DKI; ...): MK, AK and RK need RD above " *
            "$(_EVALMIN) mm²/s (they are 0 where an eigenvalue is not positive)"))
        λa, R = _symeigen(SMatrix{3,3}(t.elements.Dxx, t.elements.Dxy, t.elements.Dxz,
                                       t.elements.Dxy, t.elements.Dyy, t.elements.Dyz,
                                       t.elements.Dxz, t.elements.Dyz, t.elements.Dzz))
        # MK, AK and RK (in that order) of each of the three shapes: the columns of the map
        col(j) = SVector(_kurtosisindices(λa, R, _axialkurtosis(ntuple(i -> Float64(i == j), 3)..., t.n), t.MD)...)
        A = hcat(col(1), col(2), col(3))
        a, b, c = A \ SVector(float(q.MK), float(q.AK), float(q.RK))
    end
    V = t.MD^2 * _axialkurtosis(a, b, c, t.n)
    return merge(t.elements, NamedTuple{keys(parameters(DKI))[8:22]}(Tuple(V)))
end

# W's fifteen elements (in `_kurtosisweights`' order) for the quartic form
# W∥ξ⁴ + 6W∥⊥ξ²ρ² + W⊥ρ⁴, ξ = g·n, ρ² = |g|² − ξ²: from its values in 30 directions spread over
# the sphere (Fibonacci points; any 15 independent directions fix the fifteen), by least squares,
# exact to rounding: the form is a sum of the fifteen monomials.
function _axialkurtosis(Wpar, Wperp, Wpp, n)
    g = directionset(30; iterations = 0)
    A = [_kurtosisweights(x)[j] for x in g, j in 1:15]
    w = map(g) do x
        ξ = inner(x, n)
        ρ2 = 1 - ξ^2                                  # the directions are unit vectors
        Wpar * ξ^4 + 6Wpp * ξ^2 * ρ2 + Wperp * ρ2^2
    end
    return SVector{15}(A \ w)
end

# Tabesh et al.'s functions of the eigenvalues, written as DIPY writes them (`_F1m`, `_F2m`,
# dipy/reconst/dki.py), each with its limits where eigenvalues are (nearly) equal: their general
# forms divide by the differences. The thresholds are DIPY's, 2.5% of an eigenvalue. A case
# DIPY's conditions leave out (a difference exactly at the threshold) gives 0, as there. The
# limits take the mean of the two nearly equal eigenvalues, so that their derivatives are split
# between them evenly. G1 and G2 (`_G1m`, `_G2m`), below, in a form without the division.
function _F1(a, b, c)
    er = 2.5e-2
    if abs(a - b) >= a * er && abs(a - c) >= a * er
        RF, RD = _carlson_rf(a / b, a / c, one(a)), _carlson_rd(a / b, a / c, one(a))
        return (a + b + c)^2 / (18 * (a - b) * (a - c)) *
               (sqrt(b * c) / a * RF + (3a^2 - a * b - a * c - b * c) / (3a * sqrt(b * c)) * RD - 1)
    elseif abs(a - b) < a * er && abs(a - c) > a * er
        L = (a + b) / 2
        return _F2(c, L, L) / 2
    elseif abs(a - c) < a * er && abs(a - b) > a * er
        L = (a + c) / 2
        return _F2(b, L, L) / 2
    elseif abs(a - c) < a * er && abs(a - b) < a * er
        return zero(a) + 1 / 5
    end
    return zero(a)
end

function _F2(a, b, c)
    er = 2.5e-2
    if abs(b - c) > b * er
        RF, RD = _carlson_rf(a / b, a / c, one(a)), _carlson_rd(a / b, a / c, one(a))
        return (a + b + c)^2 / (3 * (b - c)^2) *
               ((b + c) / sqrt(b * c) * RF + (2a - b - c) / (3 * sqrt(b * c)) * RD - 2)
    elseif abs(b - c) < b * er && abs(a - b) > b * er
        L1, L3 = a, (c + b) / 2
        x = 1 - L1 / L3
        α = x > 0 ? atanh(sqrt(x)) / sqrt(x) : atan(sqrt(-x)) / sqrt(-x)
        return 6 * (L1 + 2L3)^2 / (144 * L3^2 * (L1 - L3)^2) * (L3 * (L1 + 2L3) + L1 * (L1 - 4L3) * α)
    elseif abs(b - c) < b * er && abs(a - b) < b * er
        return zero(a) + 6 / 15
    end
    return zero(a)
end

# G1 and G2, RK's (DIPY's `_G1m`, `_G2m`) without their cancellation. DIPY's divide by (b − c)²,
# where their numerators also vanish as (b − c)², and take the limit only within 1e5·eps (mm²/s)
# of b = c: before it they lose digits (RK 3% off at a relative gap of 1e-7, its derivative 12%
# off at 1e-5; docs/design.md). The factor cancels exactly: with s = √(c/b), the numerator
# 2b + (c² − 3bc)/√(bc) = b(s − 1)²(s + 2) and (b − c)² = b²(1 − s)²(1 + s)², so
#   G1(a, b, c) = (a + b + c)²·(s + 2) / (18b²(1 + s)²)
# and with (b + c)/√(bc) − 2 = (√b − √c)²/√(bc),
#   G2(a, b, c) = (a + b + c)² / (3√(bc)·(√b + √c)²)
# DIPY's values where theirs are accurate (to rounding), and their limits at b = c,
# (a + 2b)²/(24b²) and (a + 2b)²/(12b²), with no branch. b, c > 0 (`_EVALMIN`).
_G1(a, b, c) = (s = sqrt(c / b); (a + b + c)^2 * (s + 2) / (18 * b^2 * (1 + s)^2))
_G2(a, b, c) = (a + b + c)^2 / (3 * sqrt(b * c) * (sqrt(b) + sqrt(c))^2)

# Carlson's symmetric elliptic integrals, R_F(x, y, z) and R_D(x, y, z) (Carlson, Numer Algorithms
# 1995;10:13), by the duplication theorem: each step moves the three arguments together by λ and
# divides by 4, until their spread (Q, scaled by 4⁻ⁿ) is small against their mean, then a Taylor
# series in the remaining spread. DIPY's `carlson_rf` and `carlson_rd`, but with a tighter
# tolerance: DIPY's (3e-4 and 1e-4) leave R_F about 1e-6 off, which MK's formula amplifies to
# 4e-5 (measured, 2026-11-07: DIPY's MK 1.985127 for a tensor whose MK, the sphere average of
# K(n) by quadrature, is 1.985204). At 1e-8, R_F was still 3e-10 off and MK 9e-9; at 1e-12 the
# closed form gives the sphere average to 12 digits, for about one more duplication step (the
# cost per call is a few hundred ns either way). So EQUIP's MK differs from DIPY's by DIPY's
# error, at most a few 1e-5. Generic in the number
# type: the loop's test compares values, so dual numbers pass through.
function _carlson_rf(x, y, z; errtol = 1e-12)
    xn, yn, zn = x, y, z
    An = (xn + yn + zn) / 3
    Q = (3 * errtol)^(-1 / 6) * max(abs(An - xn), abs(An - yn), abs(An - zn))
    n = 0
    while 4.0^(-n) * Q > abs(An)
        xr, yr, zr = sqrt(xn), sqrt(yn), sqrt(zn)
        λ = xr * (yr + zr) + yr * zr
        n += 1
        xn, yn, zn, An = (xn + λ) / 4, (yn + λ) / 4, (zn + λ) / 4, (An + λ) / 4
    end
    X = 1 - xn / An
    Y = 1 - yn / An
    Z = -X - Y
    E2, E3 = X * Y - Z^2, X * Y * Z
    return An^(-1 / 2) * (1 - E2 / 10 + E3 / 14 + E2^2 / 24 - 3 / 44 * E2 * E3)
end

function _carlson_rd(x, y, z; errtol = 1e-12)
    xn, yn, zn = x, y, z
    A0 = (xn + yn + 3zn) / 5
    An = A0
    Q = (errtol / 4)^(-1 / 6) * max(abs(An - xn), abs(An - yn), abs(An - zn))
    s = zero(An)
    n = 0
    while 4.0^(-n) * Q > abs(An)
        xr, yr, zr = sqrt(xn), sqrt(yn), sqrt(zn)
        λ = xr * (yr + zr) + yr * zr
        s += 4.0^(-n) / (zr * (zn + λ))
        n += 1
        xn, yn, zn, An = (xn + λ) / 4, (yn + λ) / 4, (zn + λ) / 4, (An + λ) / 4
    end
    X = (A0 - x) / (4.0^n * An)
    Y = (A0 - y) / (4.0^n * An)
    Z = -(X + Y) / 3
    E2, E3 = X * Y - 6Z^2, (3X * Y - 8Z^2) * Z
    E4, E5 = 3 * (X * Y - Z^2) * Z^2, X * Y * Z^3
    return 4.0^(-n) * An^(-3 / 2) * (1 - 3 / 14 * E2 + 1 / 6 * E3 + 9 / 88 * E2^2 - 3 / 22 * E4 -
                                     9 / 52 * E2 * E3 + 3 / 26 * E5) + 3s
end
