# =================================================================================================
# DTI: the diffusion tensor (Basser, Mattiello and LeBihan, Biophys J 1994;66:259)
#
#   S    = S0·exp(−b·gᵀDg)
#   gᵀDg = gx²·Dxx + gy²·Dyy + gz²·Dzz + 2gx·gy·Dxy + 2gx·gz·Dxz + 2gy·gz·Dyz
#
# One image per gradient direction: the acquisition is each image's b-value and gradient
# direction, the vectors b, gx, gy, gz (from a .bval and a .bvec by `diffusionencoding`). gᵀDg is
# the diffusivity along the gradient direction g; along one direction this is ADCDecay with
# ADC = gᵀDg. D is symmetric, so it has six distinct elements, and each off-diagonal one appears
# twice in the 3×3 sum (Dxy and Dyx): hence the 2s.
#
# Written out, in the standard format (decided with the owner, docs/design.md, roadmap item 17):
#   - the six elements are fitted, not eigenvalues and angles: the eigen form is degenerate where
#     two eigenvalues are equal, its angles wrap, and it needs a relabel. FA, MD, AD and RD are
#     functions of the six, computed from the fit (`derived`, below; their errors in crlb and
#     montecarlo come from the same hook, roadmap item 19).
#   - no `tensor` compartment (2026-11-07): the compartments (iso, stick, zeppelin) are what
#     biophysical models add together, and a sum of full tensors is hard to fit (two tensors
#     swap, and parallel ones cannot be told apart); fibers are zeppelins with a direction.
#   - the diagonal elements, each the diffusivity along an axis, from 0 to 4e-3 mm²/s: not
#     negative (a diffusivity cannot be: D is positive semi-definite, gᵀDg ≥ 0 for every g, and
#     the diagonal is gᵀDg along the axes), and at most 4e-3 (`_DMAX`, src/models/builtin.jl:
#     above free water; the owner, 2026-11-09), so a voxel whose signal is lost in the noise (CSF)
#     ends on a bound instead of running away. The off-diagonal ones unbounded: they may be
#     negative, and the eigenvalues being positive is a condition on all six at once, which box
#     bounds cannot state.
#
# Each voxel starts from the weighted log-linear fit, `:loglinear` (src/fit/starts.jl): ln S is
# linear in ln S0 and the six elements, so EQUIP's Gauss–Newton step on ln y, with the derivatives
# of this `signal!`, lands on it at once. No start is written here, and none needs tuning; S0 is
# then set by least squares.
# =================================================================================================

"""
    DTI(; b, gx, gy, gz)

The diffusion tensor (Basser, Mattiello and LeBihan, Biophys J 1994;66:259): one image per
gradient direction, the signal decaying with the diffusivity along that direction,

    S    = S0·exp(−b·gᵀDg),
    gᵀDg = gx²·Dxx + gy²·Dyy + gz²·Dzz + 2gx·gy·Dxy + 2gx·gz·Dxz + 2gy·gz·Dyz

- acquisition: `b` (s/mm²) and the gradient direction `gx`, `gy`, `gz` (a unit vector) of every
  image, from a .bval and a .bvec by [`diffusionencoding`](@ref)
- free by default: `S0` (the signal at b = 0) and the tensor's six elements `Dxx`, `Dyy`, `Dzz`,
  `Dxy`, `Dxz`, `Dyz` (mm²/s). The diagonal elements from 0 (a diffusivity along an axis is not
  negative) to 4e-3 mm²/s (above free water's 3.0e-3), so data whose decay is lost in the noise
  end on a bound (`onbound`); for b in s/mm², `Dxx = Free(upper = Inf)` (and Dyy, Dzz) removes
  the cap. The off-diagonal ones unbounded: a tensor's eigenvalues being positive is a
  condition on all six at once, so check it on the results.

The data must determine the six: at least six gradient directions, not all in one plane (or on
one cone), and, for S0, images at a second b-value (b = 0, or a second shell): on one shell
alone, S0 and the tensor's trace give the same signal change.

Each voxel starts from the log-linear fit, ln S = ln S0 − b·gᵀDg, linear in ln S0 and the six
elements, in two passes: OLS, then WLS weighted by OLS's predicted signal² (DIPY's WLS; Chung
et al., NeuroImage 2006): `Free(start = :loglinear)`, which
EQUIP computes from this signal. When the data cannot give the six, they start at 0.

A truth for simulation from derived quantities ([`nominal`](@ref), [`fromderived`](@ref)): an
axially symmetric tensor, from `MD` and `FA` or `AD` and `RD`, and the direction of its axis,
`nominal(DTI; S0 = 1000, MD = 0.8e-3, FA = 0.7, direction = fiberdirection(θ, φ))`.

```julia
acq = diffusionencoding("dwi.bval", "dwi.bvec")    # b, gx, gy, gz
res = fit(DTI(; acq...), dwi; mask)
```
"""
@kwdef struct DTI{A<:AbstractVector,X<:AbstractVector,Y<:AbstractVector,Z<:AbstractVector} <: MRIModel
    b::A                                # b-values (s/mm²)
    gx::X                               # gradient direction of each image (unit vectors):
    gy::Y                               #   its x, y and z components
    gz::Z
end

parameters(::Type{<:DTI}) = (
    S0 = Free(start = :scale, lower = 0),
    Dxx = Free(start = :loglinear, lower = 0, upper = _DMAX),
    Dyy = Free(start = :loglinear, lower = 0, upper = _DMAX),
    Dzz = Free(start = :loglinear, lower = 0, upper = _DMAX),
    Dxy = Free(start = :loglinear), Dxz = Free(start = :loglinear), Dyz = Free(start = :loglinear),
)

function signal!(out, m::DTI, p)
    (; b, gx, gy, gz) = m
    (; S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz) = p
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        gDg = gx[j]^2 * Dxx + gy[j]^2 * Dyy + gz[j]^2 * Dzz +
              2gx[j] * gy[j] * Dxy + 2gx[j] * gz[j] * Dxz + 2gy[j] * gz[j] * Dyz
        out[j] = S0 * exp(-b[j] * gDg)
    end
    return out
end

# The quantities read off a tensor (`derived`: the maps beside the six elements): the mean
# diffusivity MD = tr(D)/3; the fractional anisotropy FA = √(3/2)·‖D − MD·I‖/‖D‖ (Basser &
# Pierpaoli, J Magn Reson B 1996; Frobenius norms, so no eigenvalues: equal to the eigenvalue
# form to rounding); the axial diffusivity AD = λ1 and the radial RD = (λ2 + λ3)/2; and the
# principal eigenvector, e1, as its three components V1x, V1y, V1z (FSL's V1), the fibers'
# direction. Eigenvalues and eigenvectors from StaticArrays' closed form for a symmetric 3×3
# matrix (no allocation), their derivatives by perturbation theory (`_symeigen`, below). An
# eigenvector's sign is arbitrary (e1 and −e1 are one direction): it is put in the upper
# hemisphere, z ≥ 0 (then y ≥ 0, then x ≥ 0 on the boundary), as `directionset`'s, so that
# neighbouring voxels agree. Where λ1 = λ2 (an oblate or isotropic tensor) e1 is not defined:
# any vector of the plane is one, and the values there are arbitrary; so are AD's and RD's
# derivatives (the largest of two equal eigenvalues has none), which are then NaN, so that crlb
# says so rather than give a number. A tensor that noise made not positive definite gives the
# values as computed (an eigenvalue can be negative); a zero tensor, FA = NaN.
derived(::DTI, p) = first(_tensorindices(_tensor(p)))

# The tensor as a symmetric 3×3 static matrix, from the six elements of `p` (DTI's, DKI's).
@inline _tensor(p) = SMatrix{3,3}(p.Dxx, p.Dxy, p.Dxz, p.Dxy, p.Dyy, p.Dyz, p.Dxz, p.Dyz, p.Dzz)

# The tensor's quantities (above), and its eigenvalues (ascending) and eigenvectors (columns, in
# the same order), which DKI's kurtosis indices need too.
@inline function _tensorindices(D)
    λ, U = _symeigen(D)                                          # values ascending
    e1 = _upperhemisphere(U[:, 3])
    MD = tr(D) / 3
    FA = sqrt(oftype(MD, 3) / 2) * norm(D - MD * I) / norm(D)
    top = _equaleigen(λ, 2, 3)                                   # λ1 = λ2: no derivatives
    q = (MD = MD, FA = FA, AD = _nopartials(λ[3], top), RD = _nopartials((λ[1] + λ[2]) / 2, top),
         V1x = _nopartials(e1[1], top), V1y = _nopartials(e1[2], top), V1z = _nopartials(e1[3], top))
    return q, λ, U
end

# The reverse, for `nominal` (`fromderived`): a truth from derived quantities. Six elements
# need six numbers, so the tensor is taken axially symmetric, the most common prescription
# (white matter's fibers along one axis): AD along `direction`, RD across it,
#
#   D = RD·I + (AD − RD)·nnᵀ          (a zeppelin, n the unit vector of `direction`)
#
# from AD and RD, or from MD and FA: λ1 = MD + 2δ, λ2 = λ3 = MD − δ (their mean is MD), and
# FA² = 3δ²/(MD² + 2δ²) gives δ = MD·FA/√(3 − 2FA²), with RD > 0 for every FA < 1. A general
# tensor (three different eigenvalues): give its six elements.
fromderived(::Type{<:DTI}, q::NamedTuple) = _axialtensor(DTI, q).elements
fromderived(::Type{<:DTI}) = (:MD, :FA, :AD, :RD, :direction)

# The six elements of an axially symmetric tensor from `q` (DTI's and DKI's: the names the model
# takes are `fromderived(M)`), with AD, RD, MD and the unit vector n.
function _axialtensor(M, q::NamedTuple)
    model = nameof(M)
    takes = "MD and FA, or AD and RD, with the direction of the tensor's axis " *
            "(`direction = (x, y, z)`, e.g. `fiberdirection(θ, φ)`)"
    others = Tuple(n for n in fromderived(M) if !(n in (:MD, :FA, :AD, :RD, :direction)))
    unknown = Tuple(n for n in keys(q) if !(n in fromderived(M)))
    isempty(unknown) || throw(ArgumentError("nominal($model; ...): unknown " *
        "$(join(("`$n`" for n in unknown), ", ")): neither a parameter of $model " *
        "($(join(keys(parameters(M)), ", "))) nor a derived quantity it takes ($takes" *
        (isempty(others) ? "" : "; and $(join(others, ", "))") * ")"))
    for n in keys(q)
        n === :direction || q[n] isa Real || throw(ArgumentError("nominal($model; ...): `$n` " *
            "must be a number (derived quantities make one voxel's truth), got $(repr(q[n]))"))
    end
    mf, ar = haskey(q, :MD) || haskey(q, :FA), haskey(q, :AD) || haskey(q, :RD)
    mf && ar && throw(ArgumentError("nominal($model; ...): give MD and FA, or AD and RD, not both"))
    mf || ar || throw(ArgumentError("nominal($model; ...): the tensor: give $takes"))
    pair = mf ? (:MD, :FA) : (:AD, :RD)
    all(n -> haskey(q, n), pair) || throw(ArgumentError("nominal($model; ...): give " *
        "$(pair[1]) and $(pair[2]) together (one alone does not fix the tensor)"))
    haskey(q, :direction) || throw(ArgumentError("nominal($model; ...): give the direction " *
        "of the tensor's axis: `direction = (x, y, z)` (any length), e.g. `fiberdirection(θ, φ)`"))
    d = q.direction
    (length(d) == 3 && all(x -> x isa Real && isfinite(x), d) && any(!iszero, d)) ||
        throw(ArgumentError("nominal($model; ...): `direction` must be three finite numbers, " *
                            "not all 0 (got $(repr(d)))"))
    n = normalize(SVector{3,Float64}(d...))
    if mf
        MD, FA = float(q.MD), float(q.FA)
        (MD > 0 && 0 <= FA < 1) || throw(ArgumentError("nominal($model; ...): MD > 0 and " *
            "0 ≤ FA < 1 (an axially symmetric tensor with FA = 1 has RD = 0: give AD and RD)"))
        δ = MD * FA / sqrt(3 - 2FA^2)
        AD, RD = MD + 2δ, MD - δ
    else
        AD, RD = float(q.AD), float(q.RD)
        (AD > 0 && 0 <= RD <= AD) || throw(ArgumentError("nominal($model; ...): AD > 0 and " *
            "0 ≤ RD ≤ AD (AD is the largest eigenvalue; an oblate tensor, RD > AD: give its " *
            "six elements)"))
        MD = (AD + 2RD) / 3
    end
    D = RD * I + (AD - RD) * n * n'
    elements = (Dxx = D[1, 1], Dyy = D[2, 2], Dzz = D[3, 3], Dxy = D[1, 2], Dxz = D[1, 3],
                Dyz = D[2, 3])
    return (; elements, AD, RD, MD, n)
end

# v or −v, whichever has z > 0 (on the equator y > 0, then x > 0): one sign for one direction.
@inline function _upperhemisphere(v)
    flip = v[3] < 0 || (v[3] == 0 && (v[2] < 0 || (v[2] == 0 && v[1] < 0)))
    return flip ? -v : v
end

# -------------------------------------------------------------------------------------------------
# The eigendecomposition and its derivatives
#
# For numbers, StaticArrays' closed form (`eigen(Symmetric(D))`). For dual numbers (crlb's delta
# method, the fit's standard errors) the values' decomposition, and each partial by first-order
# perturbation theory, for D + dD with eigenvalues λᵢ and orthonormal eigenvectors uᵢ:
#
#   dλᵢ = uᵢᵀ dD uᵢ                       (Hellmann–Feynman)
#   duᵢ = Σⱼ≠ᵢ uⱼ (uⱼᵀ dD uᵢ)/(λᵢ − λⱼ)
#
# Rather than differentiating the closed form, whose trigonometric solution of the cubic divides
# by zero where two eigenvalues are equal (a prolate tensor, λ2 = λ3: every derivative NaN, the
# eigenvalues' too). Here a pair of equal eigenvalues (equal to `_EIGTOL`, relative: the closed
# form splits an exact pair by up to 1.5e-8) has its term dropped: the eigenvectors of a pair
# have no derivative within its plane (any basis of the plane is one), so the pair's vectors
# turn only out of it. What is right then: every quantity that does not depend on which basis of
# the plane was chosen and is smooth there, which is what the pair's eigenvalues give as a sum
# (RD = (λ2 + λ3)/2 of a prolate tensor), or what a kurtosis tensor symmetric in that plane gives
# (DKI's `_symmetricin`). The rest is marked by `_nopartials`. Nested duals (second derivatives)
# work through the recursion: the values' decomposition is itself one of duals.
# -------------------------------------------------------------------------------------------------

const _EIGTOL = 1e-6

@inline function _symeigen(D::SMatrix{3,3,<:Real})
    E = eigen(Symmetric(D))
    return E.values, E.vectors
end

@inline function _symeigen(D::SMatrix{3,3,Dual{Tg,V,N}}) where {Tg,V,N}
    λ, U = _symeigen(map(value, D))
    # Uᵀ dD U for each partial k: its diagonal the eigenvalues' derivatives, its off-diagonal
    # elements the eigenvectors' turns, divided by the gaps
    M = ntuple(k -> U' * map(x -> partials(x, k), D) * U, Val(N))
    # G[j, i] = 1/(λᵢ − λⱼ), 0 on the diagonal and for an equal pair (element n of a 3×3 matrix,
    # column by column: row (n − 1) % 3 + 1, column (n − 1) ÷ 3 + 1)
    gap(i, j) = _equaleigen(λ, i, j) ? zero(λ[i]) : inv(λ[i] - λ[j])
    G = SMatrix{3,3}(ntuple(n -> gap((n - 1) ÷ 3 + 1, (n - 1) % 3 + 1), Val(9)))
    dual(x, ∂) = Dual{Tg}(x, ForwardDiff.Partials(∂))
    λd = SVector{3}(ntuple(i -> dual(λ[i], ntuple(k -> M[k][i, i], Val(N))), Val(3)))
    # duᵢ = Σⱼ uⱼ·Mⱼᵢ/(λᵢ − λⱼ): column i of U·(M .* G)
    dU = ntuple(k -> U * (M[k] .* G), Val(N))
    Ud = SMatrix{3,3}(ntuple(n -> dual(U[n], ntuple(k -> dU[k][n], Val(N))), Val(9)))
    return λd, Ud
end

# Are eigenvalues i and j (of λ, ascending) equal, to `_EIGTOL` of the largest in size?
@inline _equaleigen(λ, i, j) = abs(λ[i] - λ[j]) <= _EIGTOL * maximum(abs, λ)

# x with its value, and NaN for every derivative: a quantity that has none at this point (e1
# where λ1 = λ2), so that the delta method gives NaN rather than a number. Numbers unchanged.
# `_nopartials(x, when)`: so only when `when` is true.
_nopartials(x::Real) = x
_nopartials(x::Dual{T}) where {T} =
    Dual{T}(_nopartials(value(x)), ForwardDiff.Partials(map(p -> p * NaN, partials(x).values)))
@inline _nopartials(x, when::Bool) = when ? _nopartials(x) : x
