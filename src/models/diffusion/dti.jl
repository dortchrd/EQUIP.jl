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
#     functions of the six, computed from the fit (their errors: `propagate`, item 16).
#   - no `tensor` compartment (2026-11-07): the compartments (iso, stick, zeppelin) are what
#     biophysical models add together, and a sum of full tensors is hard to fit (two tensors
#     swap, and parallel ones cannot be told apart); fibers are zeppelins with a direction.
#   - no bounds on the elements: a tensor's eigenvalues being positive is a condition on all six
#     at once, which box bounds cannot state.
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
  `Dxy`, `Dxz`, `Dyz` (mm²/s). No bounds: a tensor's eigenvalues being positive is a condition
  on all six at once, so check it on the results.

The data must determine the six: at least six gradient directions, not all in one plane (or on
one cone), and, for S0, images at a second b-value (b = 0, or a second shell): on one shell
alone, S0 and the tensor's trace give the same signal change.

Each voxel starts from the log-linear fit, ln S = ln S0 − b·gᵀDg, linear in ln S0 and the six
elements, in two passes: OLS, then WLS weighted by OLS's predicted signal² (DIPY's WLS; Chung
et al., NeuroImage 2006): `Free(start = :loglinear)`, which
EQUIP computes from this signal. When the data cannot give the six, they start at 0.

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
    Dxx = Free(start = :loglinear), Dyy = Free(start = :loglinear), Dzz = Free(start = :loglinear),
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
# matrix: no allocation, and differentiable with dual numbers (checked against finite
# differences). An eigenvector's sign is arbitrary (e1 and −e1 are one direction): it is put in
# the upper hemisphere, z ≥ 0 (then y ≥ 0, then x ≥ 0 on the boundary), as `directionset`'s,
# so that neighbouring voxels agree. Where λ1 = λ2 (an oblate or isotropic tensor) e1 is not
# defined: any vector of the plane is one, and the values there are arbitrary. A tensor that
# noise made not positive definite gives the values as computed (an eigenvalue can be
# negative); a zero tensor, FA = NaN.
derived(::DTI, p) = first(_tensorindices(_tensor(p)))

# The tensor as a symmetric 3×3 static matrix, from the six elements of `p` (DTI's, DKI's).
@inline _tensor(p) = SMatrix{3,3}(p.Dxx, p.Dxy, p.Dxz, p.Dxy, p.Dyy, p.Dyz, p.Dxz, p.Dyz, p.Dzz)

# The tensor's quantities (above), and its eigenvalues (ascending) and eigenvectors (columns, in
# the same order), which DKI's kurtosis indices need too.
@inline function _tensorindices(D)
    E = eigen(Symmetric(D))                                      # values ascending
    λ, e1 = E.values, _upperhemisphere(E.vectors[:, 3])
    MD = tr(D) / 3
    FA = sqrt(oftype(MD, 3) / 2) * norm(D - MD * I) / norm(D)
    q = (MD = MD, FA = FA, AD = λ[3], RD = (λ[1] + λ[2]) / 2, V1x = e1[1], V1y = e1[2], V1z = e1[3])
    return q, λ, E.vectors
end

# v or −v, whichever has z > 0 (on the equator y > 0, then x > 0): one sign for one direction.
@inline function _upperhemisphere(v)
    flip = v[3] < 0 || (v[3] == 0 && (v[2] < 0 || (v[2] == 0 && v[1] < 0)))
    return flip ? -v : v
end
