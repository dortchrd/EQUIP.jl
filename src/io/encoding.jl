# =================================================================================================
# Diffusion encoding: a .bval and a .bvec as the acquisition of a directional model
#
#   diffusionencoding(bval, bvec) -> (b = ..., gx = ..., gy = ..., gz = ...)
#
# A directional diffusion model (one image per gradient direction: DTI, a stick and a zeppelin
# along a fiber, ...) takes four acquisition fields, each a vector with one value per image: the
# b-value `b` (s/mm²) and the three components of the gradient direction, `gx`, `gy`, `gz`. This
# is the rule every per-observation field follows (a vector of numbers), so EQUIP's generic code
# (lengths, the Float32 conversion, the protocol designer) handles them as it handles TE or TI.
#
# The files hold them otherwise: an FSL .bval, one b per image, and a .bvec, three lines (x, y,
# z) of one value per image, read by MRImages (`readvalues`, `readbvec`, the file's 3×N matrix).
# This is the one place that converts them, with the checks that need both together: a direction
# for every b-value, and unit vectors wherever the direction matters (b above a threshold; at
# b = 0 it does not, and FSL writes 0 0 0 there). The b-values are kept as given: a b of 5 s/mm²
# is a real b-value, and the model is more accurate with it than with 0 (DIPY's `b0_threshold`
# likewise marks the b = 0 images without changing the values). Names decided with the owner
# (docs/design.md, "Directions per image").
# =================================================================================================

"""
    diffusionencoding(bval, bvec; b0threshold = 50) -> (b = ..., gx = ..., gy = ..., gz = ...)

The acquisition of a directional diffusion model from a .bval and a .bvec: the b-value of every
image (s/mm²) and the three components of its gradient direction, each a vector with one value
per image. Give the file names, or the values already read: a vector of b-values and a 3×N or
N×3 matrix of directions (a .bvec as `readbvec` returns it, or a table with one direction per
row; a 3×3 matrix is read as a .bvec, one column per image). The result goes into any model
whose fields are `b`, `gx`, `gy`, `gz`:

```julia
acq = diffusionencoding("dwi.bval", "dwi.bvec")
m = DTI(; acq...)                      # `...` passes the four as keywords
```

Checked: one direction for every b-value, and a unit vector wherever b > `b0threshold`
(within 1e-2, as DIPY checks; a direction at b ≤ `b0threshold` may be anything, e.g. FSL's
0 0 0 at b = 0). The values are kept as given (a small b is not set to 0) and copied, so changing
the matrix later (rotating directions for motion, flipping an axis) does not change a model
made from it: make a new one.
"""
function diffusionencoding(bval::AbstractVector{<:Real}, bvec::AbstractMatrix{<:Real};
                           b0threshold::Real = 50)
    G = size(bvec, 1) == 3 ? bvec :                                    # 3 × N, as a .bvec
        size(bvec, 2) == 3 ? permutedims(bvec) :                       # N × 3: transposed
        throw(DimensionMismatch("the directions must be 3×N or N×3 (x, y, z for each image), " *
                                "got $(join(size(bvec), "×"))"))
    length(bval) == size(G, 2) || throw(DimensionMismatch("$(length(bval)) b-values but " *
        "$(size(G, 2)) gradient directions: there must be one direction per image"))
    b = Vector{Float64}(bval)
    gx, gy, gz = Vector{Float64}(G[1, :]), Vector{Float64}(G[2, :]), Vector{Float64}(G[3, :])
    for j in eachindex(b, gx, gy, gz)
        b[j] > b0threshold || continue
        len = hypot(gx[j], gy[j], gz[j])
        abs(len - 1) <= 1e-2 || throw(ArgumentError("image $j (b = $(b[j])): the gradient " *
            "direction ($(gx[j]), $(gy[j]), $(gz[j])) has length $(round(len; sigdigits = 4)), " *
            "not 1: the directions must be unit vectors"))
    end
    return (b = b, gx = gx, gy = gy, gz = gz)
end
diffusionencoding(bvalfile::AbstractString, bvecfile::AbstractString; kwargs...) =
    diffusionencoding(readvalues(bvalfile), readbvec(bvecfile); kwargs...)
