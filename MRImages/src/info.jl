# =================================================================================================
# What is known about an image, to look at and to search
#
#   findmeta(img, "echo")    the sidecar's entries whose names contain "echo" (any case; nested
#                            entries too, as "Parent.Child")
#   headerinfo(img)          the NIfTI header's fields that matter, readable: size, voxel size,
#   headerinfo(path)         axes, affine, how the values are stored; from a path, the header
#                            alone is read (fast, whatever the image's size)
#
# The sidecar is the image's data dictionary: dcm2niix writes there what the DICOM headers hold,
# in BIDS names and units (EchoTime in s, FlipAngle in degrees, ...), the vendors' differences
# resolved. MRImages keeps all of it, unchanged (`img.meta`); these functions help find things
# in it. (DICOM itself is converted with dcm2niix: docs/design.md, "DICOM through dcm2niix".)
# =================================================================================================

"""
    findmeta(img, pattern) -> Vector{Pair{String,Any}}
    findmeta(meta, pattern)

The entries of an image's metadata (its sidecar: `img.meta`, or any dictionary) whose names
match `pattern`: a string found anywhere in the name, ignoring case (`"echo"` finds `EchoTime`,
`EchoTrainLength`, `EchoNumber`), or a regular expression (`r"^Echo"`). Entries of nested
objects are searched too, named by their path (`"Parent.Child"`). Sorted by name.

```julia
findmeta(readimage("sub-01_T2w.nii.gz"), "time")
# "AcquisitionTime" => "10:42:11.5", "EchoTime" => 0.012, "RepetitionTime" => 3.2, ...
```
"""
findmeta(img::MRImage, pattern::Union{AbstractString,Regex}) = findmeta(img.meta, pattern)
function findmeta(meta::AbstractDict, pattern::Union{AbstractString,Regex})
    found = Pair{String,Any}[]
    _findmeta!(found, meta, "", pattern)
    return sort!(found; by = first)
end
_matches(name, p::AbstractString) = occursin(lowercase(p), lowercase(name))
_matches(name, p::Regex) = occursin(p, name)
# A key matches by its own name (not its parent's: "Parent" finds the object, not each entry in it).
function _findmeta!(found, d::AbstractDict, prefix, pattern)
    for (k, v) in d
        name = prefix * string(k)
        _matches(string(k), pattern) && push!(found, name => v)
        v isa AbstractDict && _findmeta!(found, v, name * ".", pattern)
    end
    return found
end

"""
    headerinfo(img) -> Vector{Pair{String,String}}
    headerinfo(path)

The NIfTI header's fields that matter, as readable text: the size, the voxel size (mm), the
axes (the direction each voxel index increases toward, in scanner coordinates: R, A, S or L, P,
I, as nibabel's `aff2axcodes`), the affine (voxel indices to mm), the sform and qform codes, how
the values are stored (the data type and the scaling), the time between volumes, the
description. From a path, only the header is read (fast for any size of image).
"""
headerinfo(img::MRImage) = hasgeometry(img) ? headerinfo(img.header) :
    ["size" => join(size(img), "×"), "geometry" => "none (an array from a MATLAB file)"]
function headerinfo(path::AbstractString)
    _isnifti(path) || throw(ArgumentError("$path: headerinfo reads NIfTI files (.nii, .nii.gz)"))
    isfile(path) || throw(ArgumentError("$path: no such file"))
    return headerinfo(_readheader(path))
end

# The header alone (NIfTI.jl's own functions: the file opened, decompressed as it is read, and
# only the header's bytes read).
function _readheader(path::AbstractString)
    io = NIfTI.niopen(path, "r")
    try
        return first(NIfTI.read_header(io))
    finally
        close(io)
    end
end

const _XFORM = ("unknown", "scanner", "aligned", "Talairach", "MNI", "template")
_xform(c) = 0 <= c <= 5 ? "$c ($(_XFORM[c + 1]))" : string(c)

function headerinfo(h::NIfTIHeader)
    nd = Int(h.dim[1])
    sz = ntuple(i -> Int(h.dim[i+1]), nd)
    A = Float64.(NIfTI.getaffine(h))
    vs = NIfTI.voxel_size(h)
    info = Pair{String,String}[
        "size" => join(sz, "×"),
        "voxel size" => join(_short.(Float64.(vs)), "×") * " mm",
        "axes" => join(_axcodes(A), ", ") * "  (the direction each index increases toward)",
        "affine" => join((join(_short.(A[i, :]), " ") for i in 1:3), "; "),
        "sform, qform" => _xform(h.sform_code) * ", " * _xform(h.qform_code),
        "stored as" => string(NIfTI.to_eltype(h.datatype)) * ", " * _scaling(h)]
    if nd >= 4 && h.pixdim[5] > 0
        push!(info, "time step" => _short(NIfTI.time_step(h) / 1000) * " s (pixdim[5])")
    end
    h.intent_code != 0 && push!(info, "intent" => string(h.intent_code))
    d = _text(h.descrip)
    isempty(d) || push!(info, "description" => d)
    push!(info, "format" => h isa NIfTI.NIfTI2Header ? "NIfTI-2" : "NIfTI-1")
    return info
end

_scaling(h) = (h.scl_slope == 0 || (h.scl_slope == 1 && h.scl_inter == 0)) ? "no scaling" :
    "value = $(_short(h.scl_slope)) × stored + $(_short(h.scl_inter))"
_text(t) = strip(String(UInt8[c for c in t if c != 0x00]))

# The direction each voxel axis increases toward: the scanner axis its affine column points
# along most, with its sign (NIfTI's scanner coordinates are RAS+: x right, y anterior, z
# superior).
function _axcodes(A)
    codes = (('L', 'R'), ('P', 'A'), ('I', 'S'))
    return ntuple(3) do j
        col = A[1:3, j]
        i = argmax(abs.(col))
        codes[i][col[i] >= 0 ? 2 : 1]
    end
end
