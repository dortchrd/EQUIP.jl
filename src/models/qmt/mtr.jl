# =================================================================================================
# MTR, the magnetization transfer ratio: a quantity of the experiment, calculated from images
#
#   MTR = 1 − S/S_ref = (S_ref − S)/S_ref
#
# S an image with the MT pulse (one MTR for every MT image: every offset and power), S_ref a
# reference acquired the same way without the MT effect (the MT pulse off, or far off resonance,
# e.g. 100 kHz). A fraction, as every fraction in EQUIP (×100 for percent units).
#
# Not a model (the owner, 2026-11-27: "there are no parameters or physics. It's just a percent
# diff ... more a derived quantity for the qMT experiment ... you get one for every offset and
# power assuming you have a ref scan"): nothing is fitted. From data, two matched arrays (an array
# alone does not say which image is the reference). The physics is in the qMT models: what MTR a
# tissue shows under a protocol is their noise-free signals' MTR, and its errors come from the
# images' noise (docs/design.md, roadmap item 5).
#
# Where there is no answer, NaN: a reference that is 0 (outside the field of view, a resampled
# image's edge), negative or not finite, or an image that is not finite. No threshold on how
# small a reference may be: noise in the background gives noise in MTR, which a mask removes.
# =================================================================================================

"""
    mtr(S, Sref; mask = nothing)

The magnetization transfer ratio, MTR = 1 − S/S_ref, of an image with the MT pulse `S` against
its reference `Sref` (acquired the same way without the MT effect: the MT pulse off, or far off
resonance), voxel by voxel. A fraction (×100 for percent units).

`S` and `Sref` are numbers or arrays of matching size; as in Julia's broadcasting, a stack of MT
images (x × y × z × n) against one reference (x × y × z) gives one MTR map per MT image.
`NaN` where there is no answer: the reference 0, negative or not finite, or `S` not finite; and
outside `mask` where it is given (zero or false outside, as `fit`'s; the size of the images, or
of one image of a stack). Integer data (a scanner's) gives floating-point MTR; Float32 data,
Float32 MTR.

```julia
mtr(700.0, 1000.0)                        # 0.3
M = mtr(mt, ref)                          # one map: two images of the same size
M = mtr(mtstack, ref; mask = brain)       # one map per MT image, NaN outside the brain
```
"""
mtr(S::Real, Sref::Real) = _mtr(S, Sref)
function mtr(S::AbstractArray{<:Real}, Sref::AbstractArray{<:Real}; mask = nothing)
    r = _mtr.(S, Sref)                    # broadcast: one reference for a stack of MT images
    if mask !== nothing
        size(mask) == size(r)[1:min(ndims(mask), ndims(r))] && ndims(mask) <= ndims(r) ||
            throw(DimensionMismatch("mtr: the mask has size $(size(mask)), the MTR maps $(size(r))"))
        r .= ifelse.(iszero.(mask), eltype(r)(NaN), r)    # the mask's dimensions first: one per image
    end
    return r
end

# One voxel: in floating point (a scanner's integers too), NaN where the reference cannot divide
@inline function _mtr(S::Real, Sref::Real)
    s, r = promote(float(S), float(Sref))
    return isfinite(s) && isfinite(r) && r > 0 ? 1 - s / r : oftype(s, NaN)
end
