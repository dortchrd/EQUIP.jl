# =================================================================================================
# Diffusion encoding: a .bval and a .bvec as the acquisition of a directional model
#
#   diffusionencoding(bval, bvec) -> (b = ..., g = ...)
#
# A directional diffusion model (one image per gradient direction: DTI, a stick and a zeppelin
# along a fiber, ...) takes two acquisition fields, each a vector with one value per image: the
# b-value `b` (s/mm²) and the gradient direction `g`, a unit vector (an `SVector{3}`: x, y, z).
# A per-observation field holds numbers or static vectors of numbers, so EQUIP's generic code
# (lengths, the Float32 conversion, the protocol designer) handles `g` as it handles TE or TI;
# a model reads one image's direction as `g[j]`, and its cosine with a fiber as `g[j] ⋅ n`.
#
# The files hold them otherwise: an FSL .bval, one b per image, and a .bvec, three lines (x, y,
# z) of one value per image, read by MRImages (`readvalues`, `readbvec`, the file's 3×N matrix).
# This is the one place that converts them, with the checks that need both together: a direction
# for every b-value, and unit vectors wherever the direction matters (b above a threshold; at
# b = 0 it does not, and FSL writes 0 0 0 there). The b-values are kept as given: a b of 5 s/mm²
# is a real b-value, and the model is more accurate with it than with 0 (DIPY's `b0_threshold`
# likewise marks the b = 0 images without changing the values). Names decided with the owner
# (docs/design.md, "Directions per image"); `g` one field since 2026-11-25 (until then three,
# gx, gy, gz: "Directions per image, revised").
# =================================================================================================

"""
    diffusionencoding(bval, bvec; scaleb = false) -> (b = ..., g = ...)

The acquisition of a directional diffusion model from a .bval and a .bvec: the b-value of every
image (s/mm²), `b`, and its gradient direction, `g`, a unit vector (`SVector{3}`) per image.
Give the file names, or the values already read: a vector of b-values and a 3×N or N×3 matrix
of directions (a .bvec as `readbvec` returns it, or a table with one direction per row; a 3×3
matrix is read as a .bvec, one column per image), or a vector of directions (`directionset`'s,
say). The result goes into any model whose fields are `b` and `g`:

```julia
acq = diffusionencoding("dwi.bval", "dwi.bvec")
m = DTI(; acq...)                      # `...` passes the two as keywords
```

Checked: one direction for every b-value, and a unit vector wherever b > 50 s/mm² (within
1e-2, as DIPY checks; a direction at b ≤ 50 may be anything, e.g. FSL's 0 0 0 at b = 0): the
check every directional model's constructor makes too, so a model given `b` and `g` by hand is
held to the same rule. The values are kept as given, unless `scaleb = true`: for protocols that
set several b-values under one nominal b by the gradient vector's length (the .bval's b for
every image, the .bvec's vectors shorter where b is lower), each image whose vector is not a
unit vector gets b·|g|² as its b-value and g/|g| as its direction (a zero vector, b = 0), as
MRtrix does by default; unit vectors (within the check's 1e-2), and every image at b ≤ 50 (where
the direction does not matter: FSL's 0 0 0 at b = 0), are kept as they are. Not the
default: a non-unit vector more often means a wrong or mismatched file, which is then refused.
Nothing else is changed (a small b is not set to 0), and the values are copied, so changing
the matrix later (rotating directions for motion, flipping an axis) does not change a model
made from it: make a new one.
"""
function diffusionencoding(bval::AbstractVector{<:Real}, bvec::AbstractMatrix{<:Real}; scaleb::Bool = false)
    G = size(bvec, 1) == 3 ? bvec :                                    # 3 × N, as a .bvec
        size(bvec, 2) == 3 ? permutedims(bvec) :                       # N × 3: transposed
        throw(DimensionMismatch("the directions must be 3×N or N×3 (x, y, z for each image), " *
                                "got $(join(size(bvec), "×"))"))
    b = Vector{Float64}(bval)
    g = [SVector{3,Float64}(G[1, j], G[2, j], G[3, j]) for j in axes(G, 2)]
    length(b) == length(g) && scaleb && _scaleb!(b, g)
    return (b = b, g = checkencoding(b, g))      # one per b-value, unit vectors (directions.jl)
end
# directions as a vector of 3-vectors (`directionset`'s, or g itself), one per image
diffusionencoding(bval::AbstractVector{<:Real}, g::AbstractVector{<:AbstractVector{<:Real}}; kwargs...) =
    diffusionencoding(bval, reduce(hcat, [collect(x) for x in g]); kwargs...)
# b scaled by the vector's squared length, and the vector made a unit one, where it is not one
# (`scaleb`): b·|g|², g/|g|; a zero vector's image gets b = 0. Where b ≤ 50 the direction does
# not matter (checkencoding's rule; FSL's 0 0 0 at b = 0), and the image is kept as given.
function _scaleb!(b, g)
    for j in eachindex(b, g)
        n = norm(g[j])
        (abs(n - 1) <= 1e-2 || b[j] <= _B0THRESHOLD) && continue   # kept as given
        b[j] *= n^2
        n > 0 && (g[j] = g[j] / n)
    end
    return b, g
end
diffusionencoding(bvalfile::AbstractString, bvecfile::AbstractString; kwargs...) =
    diffusionencoding(readvalues(bvalfile), readbvec(bvecfile); kwargs...)
