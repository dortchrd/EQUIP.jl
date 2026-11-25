# =================================================================================================
# Data in: the files of one dataset as one array (observations last), and what is needed to
# write maps back and to build the model (the acquisition values found beside the files)
#
#   data, info = readdata(paths; precision)   the data, 4-D, observations last; `info`: the
#                                             geometry, the size, the acquisition values found
#   readmap(path, info)                       one image of the data's size: a mask, a B1 map, ...
#
# The files are read with MRImages (NIfTI with BIDS sidecars; MATLAB files after `using MAT`);
# here, what is particular to fitting: images stacked into observations, and sidecar keys
# mapped to the models' field names. Units: seconds and degrees, as in EQUIP and in BIDS
# sidecars (EchoTime, InversionTime and RepetitionTime in s, FlipAngle in degrees), so nothing
# is converted.
# =================================================================================================

"""
    DataInfo

What [`readdata`](@ref) returns beside the data:

- `files`: the files read;
- `header`: the geometry maps are written with (the first file's NIfTI header), or `nothing`
  for data from a MATLAB file;
- `spatial`: the size of one image as fitted (three dimensions: the data are 4-D);
- `shape`: the size of one image as stored (for a MATLAB array, its own dimensions: maps are
  written back in that shape);
- `found`: the acquisition values found beside the files, by the models' field names
  (`found[:TE]`): numbers, or for `g` the gradient directions (an `SVector{3}` per image).

One concrete type whatever was found, so the functions that take it are compiled once (a named
tuple of the values found would be a new type for each set of names, and compile them again).
"""
struct DataInfo
    files::Vector{String}
    header::Union{Nothing,MRImages.NIfTIHeader}
    spatial::NTuple{3,Int}
    shape::Vector{Int}
    found::Dict{Symbol,Union{Vector{Float64},Vector{SVector{3,Float64}}}}
end
# what `found` holds: one number per observation, or one direction (g)
const _Found = Dict{Symbol,Union{Vector{Float64},Vector{SVector{3,Float64}}}}

"""
    readdata(paths; precision = Float64) -> (data, info)

The data of one dataset, as a 4-D array with the observations along the 4th dimension, and
`info` (a [`DataInfo`](@ref)): the geometry to write maps with, and the acquisition values
found beside the files. From:

- **NIfTI files** (`.nii`, `.nii.gz`): one 4-D file (the observations along its 4th dimension),
  or several (one per observation, 3-D, or 4-D each), stacked along the 4th dimension in the
  order given; the files' intensity scaling (`scl_slope`, `scl_inter`) applied. Several files
  must have the same size and geometry.
- **A MATLAB file** (`data.mat:S`, the variable `S`, or `data.mat` holding one array; after
  `using MAT`): an array with the observations along its last dimension and up to three spatial
  dimensions before (`nx×ny×nz×N`, `nx×ny×N`, or `nvoxels×N` for voxels already masked). It is
  reshaped to 4-D (`nvoxels×1×1×N`), and maps are written back in its own shape. No geometry.

`precision = Float32` keeps the data in half the memory (each value scaled in Float64, then
rounded: the same as `Float32.(readdata(paths)[1])`), for a fit in Float32.

The acquisition values found (`info.found`), one per observation, by the models' field names:

- BIDS JSON sidecars (`name.json` beside `name.nii.gz`, as dcm2niix writes them): `EchoTime` →
  `TE`, `InversionTime` → `TI`, `RepetitionTime` → `TR` (s), `FlipAngle` → `FA` (degrees). A
  value given for a 4-D file holds for each of its volumes; a key some files lack is left out.
- An FSL `.bval` file beside a single file (dcm2niix writes them for diffusion): `b` (s/mm²);
  with a `.bvec` beside it too, the gradient directions `g`, checked as
  [`diffusionencoding`](@ref) checks them.

```julia
data, info = readdata("sub-01_T2map.nii.gz")      # TE from sub-01_T2map.json
res = fit(T2Decay(TE = info.found[:TE]), data; mask = readmap("brain.nii.gz", info) .> 0)
writemaps("results/sub-01_", res, info)
```
"""
readdata(path::AbstractString; kwargs...) = readdata([path]; kwargs...)
function readdata(paths::AbstractVector{<:AbstractString};
                  precision::Type{<:AbstractFloat} = Float64)
    isempty(paths) && throw(ArgumentError("no data files given"))
    if any(MRImages._ismat, paths)
        length(paths) == 1 || throw(ArgumentError("a MATLAB file holds the whole dataset " *
            "(observations last): give one, not $(length(paths)) files"))
        return _loadmat(paths[1], precision)
    end
    foreach(_check_nifti, paths)
    imgs = [readimage(p; precision) for p in paths]
    for img in imgs
        ndims(img) in (3, 4) || throw(ArgumentError("$(img.path): a 3-D image or a 4-D series " *
            "is expected, this file has $(ndims(img)) dimensions"))
    end
    spatial = size(imgs[1])[1:3]
    for img in imgs
        size(img)[1:3] == spatial || throw(DimensionMismatch("$(img.path) is " *
            "$(join(size(img)[1:3], "×")); $(paths[1]) is $(join(spatial, "×")): the files " *
            "must have the same size"))
        samegeometry(img, imgs[1]) || throw(ArgumentError("$(img.path) and $(paths[1]) have " *
            "different geometry (voxel size or position): they must be the same images"))
    end
    data = length(imgs) == 1 && ndims(imgs[1]) == 4 ? imgs[1].data :
           cat((reshape(img.data, spatial..., :) for img in imgs)...; dims = 4)
    size(data, 4) >= 2 || throw(ArgumentError("one image is one observation: give a 4-D file, " *
                                              "or one 3-D file per observation"))
    found = _found(paths, [img.meta for img in imgs], [size(img, 4) for img in imgs])
    return data, DataInfo(collect(paths), imgs[1].header, spatial, collect(spatial), found)
end

function _check_nifti(p)
    MRImages._isnifti(p) || throw(ArgumentError("$p: give NIfTI files (.nii, .nii.gz), or a " *
                                                "MATLAB file (data.mat:NAME)"))
    isfile(p) || throw(ArgumentError("$p: no such file"))
end

# The data in a MATLAB file: an array, observations last, reshaped to 4-D.
function _loadmat(path, precision)
    img = readimage(path; precision)
    a = img.data
    ndims(a) in 2:4 || throw(ArgumentError("$path is $(join(size(a), "×")): give the " *
        "observations along the last dimension, after up to three spatial ones (nx×ny×nz×N, " *
        "nx×ny×N, or nvoxels×N)"))
    shape = collect(size(a)[1:end-1])
    spatial = (shape..., ones(Int, 3 - length(shape))...)
    data = reshape(a, spatial..., size(a, ndims(a)))
    size(data, 4) >= 2 || throw(ArgumentError("$path has one observation (its last " *
        "dimension is $(size(data, 4))): the observations go along the last dimension"))
    return data, DataInfo([String(path)], nothing, NTuple{3,Int}(spatial), shape, _Found())
end

# BIDS sidecar keys and the model fields they give (seconds, degrees: no conversion).
const _BIDS = (EchoTime = :TE, InversionTime = :TI, RepetitionTime = :TR, FlipAngle = :FA)

# The acquisition values beside the files: one per observation, by field name.
function _found(paths, cars, counts)
    found = _Found()
    for (key, field) in pairs(_BIDS)
        all(c -> haskey(c, String(key)), cars) || continue
        vals = Float64[]
        for (p, c, n) in zip(paths, cars, counts)
            v = c[String(key)]
            v isa Real ? append!(vals, fill(Float64(v), n)) :
            v isa AbstractVector && length(v) == n ? append!(vals, Float64.(v)) :
            throw(ArgumentError("$(sidecarpath(p)): $key has $(length(v)) values for $n images"))
        end
        found[field] = vals
    end
    if length(paths) == 1
        bval = filestem(paths[1]) * ".bval"
        if isfile(bval)
            b = readvalues(bval)
            length(b) == counts[1] || throw(ArgumentError("$bval has $(length(b)) b-values; " *
                                                          "$(paths[1]) has $(counts[1]) images"))
            found[:b] = b
            # the gradient directions beside it (FSL's .bvec): g, checked with the
            # b-values by `diffusionencoding` (one per image, unit vectors where b > 50)
            bvec = filestem(paths[1]) * ".bvec"
            if isfile(bvec)
                acq = try
                    diffusionencoding(b, readbvec(bvec))
                catch e                                    # said with the file's name
                    e isa Union{ArgumentError,DimensionMismatch} || rethrow()
                    throw(ArgumentError("$bvec (with $bval): $(e.msg)"))
                end
                found[:g] = acq.g
            end
        end
    end
    return found
end

"""
    readmap(path, info) -> Array{Float64,3}

One image with the data's size and geometry (`info` from [`readdata`](@ref)): a mask, or a map
for a fixed parameter (B1, Sm, ...). A NIfTI file (a 4-D file with one volume is taken as 3-D;
its geometry is checked against the data's), or a MATLAB array (`file.mat:NAME`) of the data's
shape. Returned in the shape the data are fitted in (3-D).
"""
function readmap(path::AbstractString, info::DataInfo)
    img = readimage(path)
    a = img.data
    if MRImages._ismat(path)
        (collect(size(a)) == info.shape || size(a) == info.spatial) ||
            throw(DimensionMismatch("$path is $(join(size(a), "×")); the data's images are " *
                                    "$(join(info.shape, "×"))"))
        return reshape(a, info.spatial)
    end
    ndims(a) == 4 && size(a, 4) == 1 && (a = dropdims(a; dims = 4))
    ndims(a) == 3 || throw(ArgumentError("$path: one 3-D image is expected, got " *
                                         "$(join(size(a), "×"))"))
    size(a) == info.spatial || throw(DimensionMismatch("$path is $(join(size(a), "×")); the " *
        "data are $(join(info.spatial, "×"))"))
    info.header === nothing || MRImages._samegeometry(img.header, info.header) ||
        throw(ArgumentError("$path has a different geometry (voxel size or position) from " *
                            "the data"))
    return a
end
