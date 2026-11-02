# =================================================================================================
# NIfTI files (.nii, .nii.gz), through NIfTI.jl
#
# Reading: the values with the file's intensity scaling applied (value = scl_slope × stored +
# scl_inter; a slope of 0 means no scaling, as the NIfTI standard says), in a floating-point
# type, and the header (the geometry). Writing: the values as they are (no scaling), with the
# image's header (its geometry and other fields), the size and data type set from the values.
# =================================================================================================

function _readnifti(path::AbstractString, ::Type{T}) where {T<:AbstractFloat}
    isfile(path) || throw(ArgumentError("$path: no such file"))
    v = niread(path)
    return MRImage(_array(_scaled(v, T)), v.header, readsidecar(path), String(path))
end

# The values with the file's intensity scaling applied, in precision T: each value scaled in
# Float64, then stored (the broadcast is fused, so no Float64 copy of the data is made on the
# way, and Float32 values are the Float64 ones rounded).
function _scaled(v::NIVolume, ::Type{T}) where {T}
    s, b = Float64(v.header.scl_slope), Float64(v.header.scl_inter)
    out = similar(v.raw, T)
    s == 0 || (s == 1 && b == 0) ? (out .= v.raw) : (out .= v.raw .* s .+ b)
    return out
end

# The image as a NIfTI file: its header when it has one (deep-copied, so the image is not
# changed), else NIfTI.jl's default (1 mm voxels at the origin); no intensity scaling (the values
# are written as they are). True/false values are written as UInt8 (0, 1): NIfTI's 1-bit type is
# read by few tools.
function _writenifti(path::AbstractString, img::MRImage)
    a = eltype(img.data) == Bool ? UInt8.(img.data) : img.data
    if img.header === nothing
        vol = NIVolume(a)
    else
        h = deepcopy(img.header)
        h.scl_slope = 1
        h.scl_inter = 0
        vol = NIVolume(h, a)
    end
    niwrite(path, vol)
    return path
end
