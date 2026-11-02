# =================================================================================================
# readimage and writeimage: each format chosen by the path's extension
# =================================================================================================

"""
    readimage(path; precision = Float64) -> MRImage

An image from a file, the format by its extension:

- **NIfTI** (`.nii`, `.nii.gz`): the values with the file's intensity scaling applied, the
  header (the geometry), and the BIDS JSON sidecar beside it (`name.json`), if there is one, as
  `meta`.
- **MATLAB** (`file.mat:name`, or `file.mat` holding one variable; after `using MAT`): the
  array, any size; no geometry, no metadata.

The values are in `precision` (`Float32` for half the memory: each value is computed in Float64,
then rounded, so the values are those of `Float32.(readimage(path).data)`).
"""
function readimage(path::AbstractString; precision::Type{<:AbstractFloat} = Float64)
    f = _format(path)
    f === :nifti && return _readnifti(path, precision)
    return _readmatimage(path, precision)
end

"""
    writeimage(path, img::MRImage) -> String
    writeimage(path, data; like = nothing, meta = Dict(), descrip = "")

Write an image as a NIfTI file (`.nii`, or `.nii.gz`, compressed): the values as they are (no
intensity scaling), in their element type (true/false as 0/1 in UInt8), with the image's
geometry (its header), or for an image without one, NIfTI.jl's default (1 mm voxels at the
origin). The image's `meta`, when it has entries, is written beside it as a JSON sidecar
(`name.json`). The directory is made if needed. Returns the image's file.

The second form writes values in the geometry of `like` (an image): `MRImage(data, like; meta,
descrip)`, written. For a MATLAB file, see [`writemat`](@ref).
"""
function writeimage(path::AbstractString, img::MRImage)
    _isnifti(path) || throw(ArgumentError("$path: writeimage writes NIfTI files (.nii, " *
        ".nii.gz)" * (_ismat(path) ? "; for a MATLAB file, writemat(path, Dict(name => array))" : "")))
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    _writenifti(path, img)
    isempty(img.meta) || writesidecar(path, img.meta)
    return String(path)
end
writeimage(path::AbstractString, data::AbstractArray; like::Union{Nothing,MRImage} = nothing,
           meta::AbstractDict = Dict{String,Any}(), descrip::AbstractString = "") =
    writeimage(path, like === nothing ? MRImage(data; meta) : MRImage(data, like; meta, descrip))
