# =================================================================================================
# What a path names: its format (by its extension), its stem (the name without the extension,
# which a sidecar or a .bval file shares), and for a MATLAB file, the variable.
#
#   _format("t2.nii.gz")      :nifti       (.nii, .nii.gz)
#   _format("data.mat:S")     :mat         (.mat, with or without a variable)
#   filestem("t2.nii.gz")     "t2"         ("t2.json" is its sidecar, "t2.bval" its b-values)
#   _matpath("data.mat:S")    ("data.mat", "S");  _matpath("data.mat") → ("data.mat", nothing)
#
# A MATLAB file holds several arrays, so a path can pick one: `file.mat:name`. The variable is
# split off only after `.mat`, so a Windows drive letter (`C:\data\x.mat`) is not mistaken for one.
# =================================================================================================

const _NIFTI_EXTENSIONS = (".nii.gz", ".nii")

# The file and the variable of a `.mat` path (nothing when none is named), or nothing when the
# path is not a `.mat` path. A MATLAB name: a letter, then letters, digits or underscores.
function _matpath(path::AbstractString)
    m = match(r"^(.*\.mat)(?::([A-Za-z][A-Za-z0-9_]*))?$"i, path)
    m === nothing && return nothing
    return String(m.captures[1]), m.captures[2] === nothing ? nothing : String(m.captures[2])
end

_isnifti(path::AbstractString) = any(e -> endswith(lowercase(path), e), _NIFTI_EXTENSIONS)
_ismat(path::AbstractString) = _matpath(path) !== nothing

function _format(path::AbstractString)
    _isnifti(path) && return :nifti
    _ismat(path) && return :mat
    throw(ArgumentError("$path: not a format MRImages reads; it reads NIfTI (.nii, .nii.gz) " *
                        "and MATLAB (.mat, or file.mat:variable, after `using MAT`)"))
end

"""
    filestem(path) -> String

The path without its image extension (`.nii.gz`, `.nii`, `.mat`, and a `.mat` file's
`:variable`): the name that files belonging to the image share, as BIDS and dcm2niix write them.
`filestem("sub-01_dwi.nii.gz") * ".bval"` is `"sub-01_dwi.bval"`.
"""
function filestem(path::AbstractString)
    mp = _matpath(path)
    mp !== nothing && return first(mp)[1:end-4]
    for e in _NIFTI_EXTENSIONS
        endswith(lowercase(path), e) && return String(path[1:end-length(e)])
    end
    return String(path)
end
