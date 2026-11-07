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
