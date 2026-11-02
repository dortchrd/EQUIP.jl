"""
    MRImages

MRI images in and out of files: an image's values, where they are (its geometry) and what is
known about how they were acquired (its metadata), as one [`MRImage`](@ref), whatever the
format. Like Python's nibabel, so that analysis packages (EQUIP, ...) do not each handle the
formats' details themselves.

- [`readimage`](@ref)`(path)`: a NIfTI file (`.nii`, `.nii.gz`) with its BIDS JSON sidecar, or
  an array in a MATLAB file (`data.mat:S`, after `using MAT`).
- [`writeimage`](@ref)`(path, img)`: a NIfTI file, and a JSON sidecar with the image's metadata.
- [`MRImage`](@ref)`(data, like)`: new values (a map) in another image's geometry.
- [`affine`](@ref), [`voxelsize`](@ref), [`hasgeometry`](@ref), [`samegeometry`](@ref): the
  geometry.
- [`readsidecar`](@ref), [`writesidecar`](@ref): BIDS JSON sidecars.
- [`readvalues`](@ref) (numbers from a text file: an FSL `.bval`, a list of TIs; or from a
  `.mat` variable), [`readbvec`](@ref) (FSL `.bvec`).
- [`readmat`](@ref), [`writemat`](@ref): a MATLAB file's variables (after `using MAT`).
"""
module MRImages

# --- Dependencies -------------------------------------------------------------------------------
# NIfTI.jl reads and writes the NIfTI files; JSON.jl the BIDS sidecars. MATLAB files come from
# MAT.jl, a package extension (ext/MRImagesMATExt.jl): it is loaded only when the user loads MAT
# (`using MAT`), so that reading NIfTI does not pay MAT's 0.4 s of loading (it brings HDF5).
import NIfTI
using NIfTI: niread, niwrite, NIVolume
import JSON
using PrecompileTools: @setup_workload, @compile_workload

export MRImage, readimage, writeimage
export affine, voxelsize, hasgeometry, samegeometry
export readsidecar, writesidecar, sidecarpath, filestem
export readvalues, readbvec
export readmat, writemat

include("paths.jl")          # what a path names: its format, its stem, a .mat file's variable
include("image.jl")          # MRImage: values, geometry, metadata
include("nifti.jl")          # NIfTI files
include("sidecar.jl")        # BIDS JSON sidecars
include("mat.jl")            # MATLAB files, through the MAT extension
include("values.jl")         # numbers from text files: lists of values, .bval, .bvec
include("readwrite.jl")      # readimage and writeimage: each format by the path's extension
include("precompile.jl")     # compiled once, when the package is installed

end
