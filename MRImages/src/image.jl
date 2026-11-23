# =================================================================================================
# MRImage: an image's values, where they are, and what is known about how they were acquired
#
#   img.data     the values (NIfTI's intensity scaling applied), in a floating-point type
#   img.header   the NIfTI header: voxel size, orientation and position, and the rest of the
#                file's fields, kept when maps are written in this geometry; `nothing` for an
#                array from a MATLAB file, which has no geometry
#   img.meta     metadata: the BIDS JSON sidecar's entries (EchoTime, FlipAngle, ...), empty if
#                there is no sidecar
#   img.path     the file it came from ("" for an image made in memory)
#
# As nibabel's images (img.get_fdata(), img.affine, img.header in Python): one object for any
# format. Not an AbstractArray (as nibabel's images are not arrays either): `img.data` is the
# array, so what an operation returns is never in doubt (an array, without the geometry), and
# showing an image prints a summary, not millions of numbers.
# =================================================================================================

"""
The NIfTI header type (NIfTI.jl's `NIfTI1Header` or `NIfTI2Header`): the geometry an
[`MRImage`](@ref) carries.
"""
const NIfTIHeader = NIfTI.NIfTIHeader

"""
    MRImage(data; meta = Dict())                       values without a geometry
    MRImage(data, like::MRImage; meta, descrip = "")   values in another image's geometry
    MRImage(data, header; meta, descrip = "")          ... or in a NIfTI header's

An image: `data` (its values), `header` (its geometry: a NIfTI header, or `nothing`), `meta`
(a `Dict{String,Any}` of metadata: a BIDS sidecar's entries) and `path` (the file it was read
from, or `""`). [`readimage`](@ref) makes one from a file; [`writeimage`](@ref) writes one.

With `like`, the values get that image's geometry (its voxel size, orientation and position: a
map computed from the data is in the data's place), and the header's fields that describe the
values are reset: the intensity scaling (the values are written as they are), the display range,
the intent; `descrip` (up to 79 characters) says what the values are. The values' first three
dimensions must be the size of `like`'s images.

```julia
img = readimage("t2.nii.gz")
t2map = MRImage(T2, img; meta = Dict("Units" => "s"), descrip = "T2 (s)")
writeimage("T2map.nii.gz", t2map)          # and T2map.json, from `meta`
```
"""
struct MRImage{T<:Number,N}
    data::Array{T,N}
    header::Union{Nothing,NIfTIHeader}
    meta::Dict{String,Any}
    path::String
end

MRImage(data::AbstractArray; meta::AbstractDict = Dict{String,Any}()) =
    MRImage(_array(data), nothing, Dict{String,Any}(meta), "")
MRImage(data::AbstractArray, like::MRImage; kwargs...) = MRImage(data, like.header; kwargs...)
function MRImage(data::AbstractArray, header::Union{Nothing,NIfTIHeader};
                 meta::AbstractDict = Dict{String,Any}(), descrip::AbstractString = "")
    header === nothing && return MRImage(data; meta)
    _checkspatial(size(data), header)
    return MRImage(_array(data), _valueheader(header, descrip), Dict{String,Any}(meta), "")
end

# An Array of the values (a view or a range is copied; an Array is kept as it is).
_array(a::Array) = a
_array(a::AbstractArray) = collect(a)

# The size of one image in a header (its first three dimensions; 1 for those it does not have).
_spatial(h::NIfTIHeader) = ntuple(i -> h.dim[1] >= i ? Int(h.dim[i+1]) : 1, 3)
function _checkspatial(sz::Dims, h::NIfTIHeader)
    s = ntuple(i -> i <= length(sz) ? sz[i] : 1, 3)
    s == _spatial(h) || throw(DimensionMismatch("the values are $(join(sz, "×")); the " *
        "geometry is for images of $(join(_spatial(h), "×"))"))
    return nothing
end

# A header for other values in the same geometry: the voxel size, orientation and position kept;
# what describes the values reset (they are written as they are: no intensity scaling); `descrip`
# set. The size and data type are set from the values when they are written (NIfTI.jl).
function _valueheader(h::NIfTIHeader, descrip::AbstractString)
    m = deepcopy(h)
    m.scl_slope = 1
    m.scl_inter = 0
    m.cal_max = 0
    m.cal_min = 0
    m.intent_code = 0
    m.descrip = _text80(descrip)
    return m
end
_text80(s) = (d = codeunits(first(s, 79)); ntuple(i -> i <= length(d) ? d[i] : 0x00, 80))

Base.size(img::MRImage) = size(img.data)
Base.size(img::MRImage, d) = size(img.data, d)
Base.ndims(img::MRImage) = ndims(img.data)
Base.eltype(::Type{<:MRImage{T}}) where {T} = T

"""
    hasgeometry(img) -> Bool

Whether the image knows where its voxels are (a NIfTI header): an array from a MATLAB file
does not.
"""
hasgeometry(img::MRImage) = img.header !== nothing

function _geometry(img::MRImage, what)
    hasgeometry(img) && return img.header
    throw(ArgumentError("$(_name(img)) has no geometry (it is from a MATLAB file, or was made " *
                        "without one), so it has no $what"))
end
_name(img::MRImage) = isempty(img.path) ? "this image" : img.path

"""
    affine(img) -> Matrix{Float64}

The 4×4 matrix from voxel indices (0-based, as in NIfTI) to scanner coordinates in mm:
`[x, y, z, 1] = affine(img) * [i, j, k, 1]`. The NIfTI file's sform when it has one, else its
qform (NIfTI's methods 3 and 2), as nibabel's `img.affine`.
"""
affine(img::MRImage) = Float64.(NIfTI.getaffine(_geometry(img, "affine")))

"""
    voxelsize(img) -> NTuple{3,Float64}

The voxel size in mm (from the NIfTI header's `pixdim`, in its units; units the header leaves
unknown are taken as mm, as nibabel and FSL do).
"""
function voxelsize(img::MRImage)
    v = _voxelsize(_geometry(img, "voxel size"))        # info.jl: unknown units taken as mm
    return ntuple(i -> i <= length(v) ? v[i] : 1.0, 3)
end

"""
    samegeometry(a, b; atol = 1e-4) -> Bool

Whether two images are in the same place: the same size of image and voxel size, and the same
orientation and position (both of the NIfTI files' affines, the sform and the qform, within
`atol` mm), so that their voxels can be compared one to one. Images without geometry (from
MATLAB files) are the same if they have the same size.
"""
samegeometry(a::MRImage, b::MRImage; atol::Real = 1e-4) =
    !hasgeometry(a) && !hasgeometry(b) ? size(a)[1:min(3, end)] == size(b)[1:min(3, end)] :
    hasgeometry(a) && hasgeometry(b) && _samegeometry(a.header, b.header; atol)
function _samegeometry(a::NIfTIHeader, b::NIfTIHeader; atol::Real = 1e-4)
    near(x, y) = all(isapprox.(Float64.(x), Float64.(y); atol))
    return _spatial(a) == _spatial(b) && near(a.pixdim[2:4], b.pixdim[2:4]) &&
           a.qform_code == b.qform_code && a.sform_code == b.sform_code &&
           near(a.srow_x, b.srow_x) && near(a.srow_y, b.srow_y) && near(a.srow_z, b.srow_z) &&
           near((a.quatern_b, a.quatern_c, a.quatern_d, a.qoffset_x, a.qoffset_y, a.qoffset_z),
                (b.quatern_b, b.quatern_c, b.quatern_d, b.qoffset_x, b.qoffset_y, b.qoffset_z))
end

# Shown as a summary: the size, type, voxel size, file, and the metadata's keys.
function Base.show(io::IO, img::MRImage{T}) where {T}
    print(io, "MRImage(", join(size(img), "×"), " ", T)
    isempty(img.path) || print(io, ", ", repr(img.path))
    print(io, ")")
end
function Base.show(io::IO, ::MIME"text/plain", img::MRImage{T}) where {T}
    print(io, "MRImage ", join(size(img), "×"), " ", T)
    hasgeometry(img) ? print(io, ", ", join(_short.(voxelsize(img)), "×"), " mm voxels") :
                       print(io, ", no geometry")
    isempty(img.path) || print(io, "\n  from ", img.path)
    if !isempty(img.meta)
        ks = sort!(collect(keys(img.meta)))
        print(io, "\n  meta: ", join(first(ks, 8), ", "), length(ks) > 8 ? ", ... ($(length(ks)) keys)" : "")
    end
end
_short(x) = isinteger(x) ? string(Int(x)) : string(round(x; sigdigits = 4))
