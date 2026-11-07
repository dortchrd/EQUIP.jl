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
# Each voxel starts from the weighted log-linear fit (`datastart`, below), so no start needs
# tuning, and S0 is then set by least squares.
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
elements, weighted by S² (Basser et al., J Magn Reson B 1994); from an isotropic tensor,
1/mean(b), when the data cannot give the six.

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
    Dxx = Free(start = :data), Dyy = Free(start = :data), Dzz = Free(start = :data),   # mm²/s
    Dxy = Free(start = :data), Dxz = Free(start = :data), Dyz = Free(start = :data),   # (datastart)
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

# The start: ln S is linear in ln S0 and the six elements, one equation per image j,
#
#   ln yⱼ = ln S0 − bⱼgxⱼ²·Dxx − bⱼgyⱼ²·Dyy − bⱼgzⱼ²·Dzz − 2bⱼgxⱼgyⱼ·Dxy − 2bⱼgxⱼgzⱼ·Dxz − 2bⱼgyⱼgzⱼ·Dyz
#
# (the B-matrix design matrix: DIPY's `design_matrix`), solved by weighted least squares with
# weights yⱼ²: the noise in ln y is about σ/y, so this weights each image by its precision (as
# `_loglinear_rate`, builtin.jl, and Basser et al., J Magn Reson B 1994). Images with y ≤ 0 are
# skipped; y is divided by its largest value, since only the weights' ratios matter.
#
# The normal equations AᵀWA·u = AᵀW·ln y are 7×7: static arrays, summed image by image, no
# allocation; in Float64 at least (Float32 data too: it costs the same on a CPU), and scaled to a
# unit diagonal before the Cholesky factorization, which puts every entry near 1 whatever the
# units of b. A design that cannot give the six (fewer than six directions, all of them in one
# plane or on one cone, one shell with no second b-value) makes the system singular, which
# rounding turns into a tiny pivot rather than a zero one: measured (2026-11-07, scratch
# pivots.jl), the smallest pivot is ≤ 3e-7 for those, ≥ 0.02 for every determined design tried
# (one b = 0 and six directions, directions within 3° of a plane, anisotropic tensors, SNR 30).
# So a pivot below eps^(1/4) (1.2e-4) means undetermined, and the start is then isotropic,
# 1/mean(b), as ADCDecay's. ln S0 is not needed (S0 is set by least squares afterwards); fixed
# elements are not used here (the fit holds them).
function datastart(m::DTI, y, p)
    (; b, gx, gy, gz) = m
    T = float(eltype(y))                             # the working precision, for the result
    S = promote_type(Float64, T)                     # the precision of the solve
    ymax = S(maximum(y))
    AWA, AWl = zero(SMatrix{7,7,S}), zero(SVector{7,S})
    @inbounds for j in eachindex(y, b, gx, gy, gz)
        y[j] > 0 || continue
        bj, X, Y, Z = S(b[j]), S(gx[j]), S(gy[j]), S(gz[j])
        a = SVector(one(S), -bj * X^2, -bj * Y^2, -bj * Z^2, -2bj * X * Y, -2bj * X * Z, -2bj * Y * Z)
        w = (S(y[j]) / ymax)^2
        AWA += w * a * a'
        AWl += w * log(S(y[j]) / ymax) * a
    end
    d = diag(AWA)                                    # 0 for a column no image measures
    if all(>(0), d)
        s = 1 ./ sqrt.(d)                            # the scaling to a unit diagonal
        F = cholesky(Symmetric(AWA .* (s * s')); check = false)
        if issuccess(F) && minimum(diag(F.U)) > sqrt(sqrt(eps(S)))
            u = s .* (F \ (s .* AWl))
            if all(isfinite, u)
                return (Dxx = T(u[2]), Dyy = T(u[3]), Dzz = T(u[4]), Dxy = T(u[5]), Dxz = T(u[6]), Dyz = T(u[7]))
            end
        end
    end
    D0 = maximum(b) > 0 ? 1 / T(mean(b)) : T(1e-3)   # no diffusion weighting: a nominal value
    return (Dxx = D0, Dyy = D0, Dzz = D0, Dxy = zero(T), Dxz = zero(T), Dyz = zero(T))
end
