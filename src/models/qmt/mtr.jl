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
# alone does not say which image is the reference), or a qMT acquisition's images, whose
# references it finds (below). The physics is in the qMT models: what MTR a tissue shows under a
# protocol is their noise-free signals' MTR (`mtr(m, θ)`), and its errors come from the images'
# noise (docs/design.md, roadmap item 5).
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

# --- MTR from a qMT acquisition: the references, then MTR predicted or from the data -------------
#
# A qMT model's acquisition says each image's MT offset and power, so it can say which images are
# references and pair them (decided with the owner, 2026-11-27, docs/design.md item 5):
#
#   which:   an image with the MT pulse off (power 0) is always a reference; others only when
#            named: `reference = :maxoffset` (at each power, the images at the largest |offset|,
#            where that power has images nearer resonance too), indices, or a Bool per image. Not by default: in a protocol without a reference the
#            largest offset is a data point (a Z-spectrum ending at 16 kHz is 15% saturated there
#            at 8 µT).
#   pairing: each MT image is divided by the mean of its references, those acquired as it was
#            but for the offset (the same MT pulse, TR and excitation: a reference per power, as
#            sequence constraints often make them); else those with the MT pulse off at its TR
#            and flip angle; else an error naming it.
#
# Acquisition values are compared exactly: they are the protocol's numbers, not measurements.

# The models whose acquisitions give each image's MT offset and power
const _MTRMODELS = Union{QMTSPGRYarnykh,QMTSPGRRamani,QMTCWHenkelman}

# Each image's MT power, by the model's own name for it (µT)
_mtpowername(::Union{QMTSPGRYarnykh,QMTSPGRRamani}) = :B1e
_mtpowername(::QMTCWHenkelman) = :B1cw
_mtpower(m, j) = getfield(m, _mtpowername(m))[j]

# Two images acquired the same way but for the offset (the same MT pulse, TR and excitation), or
# with the same readout (TR and flip angle: what an MT-off reference must share with an MT image)
_samereadout(m::Union{QMTSPGRYarnykh,QMTSPGRRamani}, j, k) = m.TR[j] == m.TR[k] && m.FA[j] == m.FA[k]
_samereadout(::QMTCWHenkelman, j, k) = true
_sameacquisition(m::Union{QMTSPGRYarnykh,QMTSPGRRamani}, j, k) =
    m.B1e[j] == m.B1e[k] && m.tm[j] == m.tm[k] && m.ts[j] == m.ts[k] && _samereadout(m, j, k)
_sameacquisition(m::QMTCWHenkelman, j, k) = m.B1cw[j] == m.B1cw[k]

# Which images are references (a Bool per image), from the `reference` keyword
function _isreference(m, reference)
    n = nobs(m)
    isref = [iszero(_mtpower(m, j)) for j in 1:n]          # the MT pulse off: always
    if reference === :maxoffset                             # at each power, the largest |offset|
        for j in 1:n
            iszero(_mtpower(m, j)) && continue
            group = [abs(m.offset[k]) for k in 1:n if _sameacquisition(m, j, k)]
            # only where the power also has images nearer resonance: a power's only image (or a
            # ±offset pair alone) cannot be both its reference and its data, so it stays an MT image
            isref[j] |= abs(m.offset[j]) == maximum(group) > minimum(group)
        end
    elseif reference isa Symbol
        throw(ArgumentError("mtr: `reference = :$reference`; give :maxoffset, the reference images' indices, or a Bool per image"))
    elseif reference isa AbstractVector{Bool}
        length(reference) == n || throw(DimensionMismatch("mtr: `reference` has $(length(reference)) Bools for $n images"))
        isref .|= reference
    elseif reference !== nothing
        for k in reference
            k isa Integer && 1 <= k <= n ||
                throw(ArgumentError("mtr: reference image $k; the acquisition's images are 1 to $n"))
            isref[k] = true
        end
    end
    return isref
end

# The MT images (those not references) and each one's references
function _mtrpairs(m, reference)
    n, p = nobs(m), _mtpowername(m)
    isref = _isreference(m, reference)
    listing() = join(("$j: $(m.offset[j]) Hz, $p $(_mtpower(m, j))" for j in 1:n), "; ")
    any(isref) || throw(ArgumentError("mtr: no reference image (none with the MT pulse off): name them, " *
        "`reference = :maxoffset` (at each power, the image at the largest offset) or their indices. Images: " * listing()))
    images = findall(!, isref)
    isempty(images) && throw(ArgumentError("mtr: every image is a reference; none has an MTR"))
    refs = map(images) do j
        r = [k for k in 1:n if isref[k] && _sameacquisition(m, j, k)]               # its own power's
        isempty(r) && (r = [k for k in 1:n if isref[k] && iszero(_mtpower(m, k)) && _samereadout(m, j, k)])
        isempty(r) && throw(ArgumentError("mtr: image $j ($(m.offset[j]) Hz, $p $(_mtpower(m, j))) has no " *
            "reference: none acquired as it was but for the offset, none with the MT pulse off at its TR and " *
            "flip angle; name one with `reference`"))
        r
    end
    return images, refs
end

"""
    MTRResult

What [`mtr`](@ref) gives with a qMT acquisition: `mtr`, the MTR values (predicted for a tissue)
or maps (from data, one per MT image along the last dimension), and per MT image `image` (its
index in the acquisition), `offset` (Hz), its power by the model's name (`B1e`, or `B1cw`; also
`power`), and `reference` (the images it was divided by); `model`, the acquisition's model, and
`given`, the `reference` keyword as given. [`writemaps`](@ref) writes the maps.
"""
struct MTRResult{V<:AbstractArray,M<:MRIModel}
    mtr::V                          # the values, or the maps (the MT images along the last dimension)
    image::Vector{Int}              # per MT image: its index in the acquisition,
    offset::Vector{Float64}         #   its offset (Hz),
    power::Vector{Float64}          #   its MT power (µT: B1e, or B1cw),
    reference::Vector{Vector{Int}}  #   and the reference images it was divided by
    model::M
    given::Any                      # the `reference` keyword as given (for the sidecar)
end
# `r.B1e` (or `r.B1cw`): the power by the model's own name for it. Julia's dot syntax calls
# `getproperty`, so this method makes that name a synonym of the field `power`.
Base.getproperty(r::MTRResult, s::Symbol) =
    s === _mtpowername(getfield(r, :model)) ? getfield(r, :power) : getfield(r, s)
Base.propertynames(r::MTRResult) = (:mtr, :image, :offset, _mtpowername(getfield(r, :model)), :power,
                                    :reference, :model, :given)
function Base.show(io::IO, ::MIME"text/plain", r::MTRResult)
    p, maps = _mtpowername(r.model), ndims(r.mtr) > 1
    print(io, "MTRResult: ", nameof(typeof(r.model)), ", ", length(r.image), " MT images, ",
          maps ? "maps " * join(size(r.mtr)[1:end-1], "×") : "predicted")
    println(io); print(io, "  image  offset (Hz)  $p (µT)  reference", maps ? "" : "      MTR")
    for (i, j) in enumerate(r.image)
        println(io)
        print(io, lpad(j, 7), lpad(round(r.offset[i]; digits = 1), 13), lpad(r.power[i], 9 + length(String(p))),
              "  ", rpad(join(r.reference[i], ", "), 12), maps ? "" : lpad(round(r.mtr[i]; digits = 4), 8))
    end
end
_mtrresult(m, v, images, refs, given) =
    MTRResult(v, images, m.offset[images], getfield(m, _mtpowername(m))[images], refs, m, given)

"""
    mtr(m, θ; reference = nothing)
    mtr(m, data; reference = nothing, mask = nothing)

MTR from a qMT acquisition (`m`: [`QMTSPGRYarnykh`](@ref), [`QMTSPGRRamani`](@ref),
[`QMTCWHenkelman`](@ref)), one for every MT image, each against its references: predicted for a
tissue from the model's noise-free signals (`θ`, every parameter's value, as for `signal`), or
from data (an array with the images along its last dimension, as for `fit`: one map per MT image
along the last dimension; NaN outside `mask`, and where a reference is 0, negative or not
finite). The references:

- an image with the MT pulse off (power 0) always is one; others only when named:
  `reference = :maxoffset` (at each power, the images at the largest |offset|, e.g. 100 kHz;
  not a power's only image, nor a ±offset pair with nothing nearer resonance: those stay MT images),
  their indices (`reference = [13, 26]`), or a Bool per image (`reference = m.offset .>= 50e3`)
- each MT image is divided by the mean of its references acquired as it was but for the offset
  (the same MT pulse, TR and flip angle: one per power); else of those with the MT pulse off at
  its TR and flip angle; else it is an error

An [`MTRResult`](@ref): `mtr` (the values, or the maps), and per MT image `image` (its index),
`offset`, its power (`B1e`, or `B1cw`), and `reference` (the images it was divided by);
[`writemaps`](@ref) writes the maps.

```julia
m = QMTSPGRYarnykh(offset = [offsets; 100e3; offsets; 100e3], B1e = [fill(4.0, 7); fill(8.0, 7)],
                   tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
θ = (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1.0, R1f = 1.2, R1m = 1.0, R1obs = 1.1, B1 = 1.0, B0 = 0.0)
p = mtr(m, θ; reference = :maxoffset)                    # predicted: p.mtr, p.offset, p.B1e
M = mtr(m, data; reference = :maxoffset, mask = brain)   # maps: M.mtr[:, :, :, k] is image M.image[k]
```
"""
function mtr(m::_MTRMODELS, θ::NamedTuple; reference = nothing)
    images, refs = _mtrpairs(m, reference)
    S = signal(m, θ)
    v = [_mtr(S[j], sum(k -> S[k], r) / length(r)) for (j, r) in zip(images, refs)]
    return _mtrresult(m, v, images, refs, reference)
end
function mtr(m::_MTRMODELS, data::AbstractArray{<:Real}; reference = nothing, mask = nothing)
    d, n = ndims(data), nobs(m)
    size(data, d) == n || throw(DimensionMismatch("mtr: the data has $(size(data, d)) images along its " *
                                                  "last dimension; the acquisition, $n"))
    images, refs = _mtrpairs(m, reference)
    spatial = size(data)[1:(d-1)]
    mask === nothing || size(mask) == spatial ||
        throw(DimensionMismatch("mtr: the mask has size $(size(mask)), the data's spatial size is $spatial"))
    out = similar(data, float(eltype(data)), (spatial..., length(images)))
    for (i, (j, r)) in enumerate(zip(images, refs))
        Sref = sum(k -> float.(selectdim(data, d, k)), r) ./ length(r)     # the references' mean
        selectdim(out, d, i) .= _mtr.(selectdim(data, d, j), Sref)
    end
    mask === nothing || (out .= ifelse.(iszero.(mask), eltype(out)(NaN), out))   # one mask, every map
    return _mtrresult(m, out, images, refs, reference)
end
