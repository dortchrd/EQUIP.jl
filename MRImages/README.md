# MRImages.jl

*MRI images in and out of files: the values, where they are, and how they were acquired.*

One object for an image whatever the file format, like Python's
[nibabel](https://nipy.org/nibabel/), so that analysis packages do not each handle the formats'
details. Written for [EQUIP](../) and usable on its own. For now it lives in EQUIP's
repository; it will move to one of its own.

```julia
using MRImages

img = readimage("sub-01_T2w.nii.gz")    # the values (scaling applied), the header, the sidecar
img.data                                # an Array: 96×96×40×8 Float64
img.meta["EchoTime"]                    # the BIDS JSON sidecar's entries
affine(img), voxelsize(img)             # voxel indices → scanner mm; voxel size in mm

t2 = MRImage(T2values, img; meta = Dict("Units" => "s"), descrip = "T2 (s)")  # in img's geometry
writeimage("T2map.nii.gz", t2)          # and T2map.json, from `meta`

using MAT                               # MATLAB files: an extension, loaded with MAT
S = readimage("data.mat:S").data        # the variable S (no geometry in a .mat file)
writemat("maps.mat", Dict("T2" => T2values))
```

| | |
|---|---|
| `readimage(path; precision)` | NIfTI (`.nii`, `.nii.gz`) with its sidecar; MATLAB (`file.mat:NAME`) |
| `writeimage(path, img)` | NIfTI, the values as they are (no scaling), and the sidecar from `img.meta` |
| `MRImage(data, like)` | values in another image's geometry (the scaling and display range reset) |
| `affine`, `voxelsize`, `hasgeometry`, `samegeometry` | the geometry |
| `readsidecar`, `writesidecar`, `sidecarpath`, `filestem` | BIDS JSON sidecars and companion files |
| `readvalues`, `readbvec` | numbers from text files (lists, FSL `.bval`), `.bvec` directions; a `.mat` variable |
| `readmat`, `writemat` | a MATLAB file's variables (after `using MAT`) |

- `MRImage` is not an array: `img.data` is. What an operation on the values returns is then
  plain (an array, without the geometry), and showing an image prints a summary.
- Nothing is converted: BIDS sidecars are read as they are (seconds and degrees).
- MATLAB files are written as v7.3 (MAT.jl), which MATLAB's `load` reads.

Tests: `julia --project=MRImages -e 'using Pkg; Pkg.test()'`.

Next: more formats (DICOM through DICOM.jl, PAR/REC, MGH, NumPy), orientation handling
(reorienting to a standard orientation and back), memory-mapped reading of large files.
