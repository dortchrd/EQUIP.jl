# =================================================================================================
# Data in, maps out: NIfTI files (NIfTI.jl) with their BIDS JSON sidecars, and values from text
# files. (MATLAB .mat files come next, with MAT.jl.)
#
#   data, info = loaddata(paths; precision)   the data, observations last, and `info`: the geometry,
#                                         to write maps back, and the acquisition values found
#   loadmap(path, info)                   one image of the data's size: a mask, a B1 map, ...
#   readvalues(path)                      numbers from a text file (an FSL .bval, a list of TIs)
#   savemaps(prefix, res, info)           one NIfTI file per map, in the input's geometry, each
#                                         with a JSON sidecar saying how it was made
#
# Units: seconds and degrees, as in EQUIP and in BIDS sidecars (EchoTime, InversionTime and
# RepetitionTime in s, FlipAngle in degrees), so nothing is converted.
# =================================================================================================

"""
    DataInfo

What [`loaddata`](@ref) returns beside the data: `files`, `header` (the first file's NIfTI
header: the geometry maps are written with), `spatial` (the size of one image) and `found` (the
acquisition values found beside the files, by the model's field names: `found[:TE]`). One
concrete type whatever was found, so the functions that take it are compiled once (a named
tuple of the values found would be a new type for each set of names, and compile them again).
"""
struct DataInfo
    files::Vector{String}
    header::NIfTI.NIfTIHeader
    spatial::NTuple{3,Int}
    found::Dict{Symbol,Vector{Float64}}
end

"""
    loaddata(paths; precision = Float64) -> (data, info)

The data in one NIfTI file (`.nii` or `.nii.gz`, 4-D: the observations along the 4th
dimension) or in several (one per observation, 3-D, or 4-D each, stacked along the 4th dimension
in the order given), as an array with the observations last, the files' intensity scaling
(`scl_slope`, `scl_inter`) applied. Several files must have the same size and geometry.
`precision = Float32` keeps the data in half the memory (each value scaled in Float64, then
rounded: the same as `Float32.(loaddata(paths)[1])`), for a fit in Float32.

`info` (a [`DataInfo`](@ref)) holds what writing maps back needs (`header`, the first file's;
`spatial`, the size of one image) and the acquisition values found beside the files, one per
observation, by the model's field names (`info.found[:TE]`):

- BIDS JSON sidecars (`name.json` beside `name.nii.gz`, as dcm2niix writes them): `EchoTime` →
  `TE`, `InversionTime` → `TI`, `RepetitionTime` → `TR` (s), `FlipAngle` → `FA` (degrees). A
  value given for a 4-D file holds for each of its volumes; a key some files lack is left out.
- An FSL `.bval` file beside a single file (dcm2niix writes them for diffusion): `b` (s/mm²).
"""
loaddata(path::AbstractString; kwargs...) = loaddata([path]; kwargs...)
function loaddata(paths::AbstractVector{<:AbstractString};
                  precision::Type{<:AbstractFloat} = Float64)
    isempty(paths) && throw(ArgumentError("no data files given"))
    foreach(_check_nifti, paths)
    vols = [niread(p) for p in paths]
    arrays = [_scaled(v, precision) for v in vols]
    for (p, a) in zip(paths, arrays)
        ndims(a) in (3, 4) || throw(ArgumentError("$p: a 3-D image or a 4-D series is expected, " *
                                                  "this file has $(ndims(a)) dimensions"))
    end
    spatial = size(arrays[1])[1:3]
    for (p, v, a) in zip(paths, vols, arrays)
        size(a)[1:3] == spatial || throw(DimensionMismatch("$p is $(join(size(a)[1:3], "×")); " *
            "$(paths[1]) is $(join(spatial, "×")): the files must have the same size"))
        _samegeometry(v.header, vols[1].header) || throw(ArgumentError("$p and $(paths[1]) " *
            "have different geometry (voxel size or position): they must be the same images"))
    end
    data = length(arrays) == 1 && ndims(arrays[1]) == 4 ? arrays[1] :
           cat((reshape(a, spatial..., :) for a in arrays)...; dims = 4)
    size(data, 4) >= 2 || throw(ArgumentError("one image is one observation: give a 4-D file, " *
                                              "or one 3-D file per observation"))
    found = _found(paths, [size(reshape(a, spatial..., :), 4) for a in arrays])
    return data, DataInfo(collect(paths), vols[1].header, spatial, found)
end

_isnifti(p) = endswith(p, ".nii") || endswith(p, ".nii.gz")
function _check_nifti(p)
    isfile(p) || throw(ArgumentError("$p: no such file"))
    _isnifti(p) || throw(ArgumentError("$p: only NIfTI files (.nii, .nii.gz) for now"))
end

# The image values with the file's intensity scaling applied (a slope of 0 means none, as the
# NIfTI standard says), in precision T: each value scaled in Float64, then stored (the broadcast
# is fused, so no Float64 copy of the data is made on the way).
function _scaled(v, ::Type{T} = Float64) where {T}
    s, b = Float64(v.header.scl_slope), Float64(v.header.scl_inter)
    out = similar(v.raw, T)
    s == 0 || (s == 1 && b == 0) ? (out .= v.raw) : (out .= v.raw .* s .+ b)
    return out
end

# The same voxel size and position (both affines, within rounding).
function _samegeometry(a, b)
    near(x, y) = all(isapprox.(Float64.(x), Float64.(y); atol = 1e-4))
    return near(a.pixdim[2:4], b.pixdim[2:4]) && a.qform_code == b.qform_code &&
           a.sform_code == b.sform_code && near(a.srow_x, b.srow_x) && near(a.srow_y, b.srow_y) &&
           near(a.srow_z, b.srow_z) &&
           near((a.quatern_b, a.quatern_c, a.quatern_d, a.qoffset_x, a.qoffset_y, a.qoffset_z),
                (b.quatern_b, b.quatern_c, b.quatern_d, b.qoffset_x, b.qoffset_y, b.qoffset_z))
end

# BIDS sidecar keys and the model fields they give (seconds, degrees: no conversion).
const _BIDS = (EchoTime = :TE, InversionTime = :TI, RepetitionTime = :TR, FlipAngle = :FA)

# The acquisition values beside the files: one per observation, by field name.
function _found(paths, counts)
    found = Dict{Symbol,Vector{Float64}}()
    cars = [_sidecar(p) for p in paths]
    for (key, field) in pairs(_BIDS)
        all(c -> c !== nothing && haskey(c, String(key)), cars) || continue
        vals = Float64[]
        for (p, c, n) in zip(paths, cars, counts)
            v = c[String(key)]
            v isa Real ? append!(vals, fill(Float64(v), n)) :
            v isa AbstractVector && length(v) == n ? append!(vals, Float64.(v)) :
            throw(ArgumentError("$(_sidecarpath(p)): $key has $(length(v)) values for $n images"))
        end
        found[field] = vals
    end
    if length(paths) == 1
        bval = _bvalpath(paths[1])
        if isfile(bval)
            b = readvalues(bval)
            length(b) == counts[1] || throw(ArgumentError("$bval has $(length(b)) b-values; " *
                                                          "$(paths[1]) has $(counts[1]) images"))
            found[:b] = b
        end
    end
    return found
end

_stem(p) = endswith(p, ".nii.gz") ? p[1:end-7] : endswith(p, ".nii") ? p[1:end-4] : p
_sidecarpath(p) = _stem(p) * ".json"
_bvalpath(p) = _stem(p) * ".bval"
_sidecar(p) = isfile(_sidecarpath(p)) ? JSON.parsefile(_sidecarpath(p)) : nothing

"""
    loadmap(path, info) -> Array{Float64,3}

One image with the data's size and geometry (`info` from [`loaddata`](@ref)): a mask, or a map
for a fixed parameter (B1, Sm, ...). A 4-D file with one volume is taken as 3-D.
"""
function loadmap(path::AbstractString, info::DataInfo)
    _check_nifti(path)
    v = niread(path)
    a = _scaled(v)
    ndims(a) == 4 && size(a, 4) == 1 && (a = dropdims(a; dims = 4))
    ndims(a) == 3 || throw(ArgumentError("$path: one 3-D image is expected, got " *
                                         "$(join(size(a), "×"))"))
    size(a) == info.spatial || throw(DimensionMismatch("$path is $(join(size(a), "×")); the " *
        "data are $(join(info.spatial, "×"))"))
    _samegeometry(v.header, info.header) || throw(ArgumentError("$path has a different " *
        "geometry (voxel size or position) from the data"))
    return a
end

"""
    readvalues(path) -> Vector{Float64}

Numbers from a text file: one per line, or several on a line separated by spaces, tabs or commas
(an FSL `.bval` file as it is). Empty lines, and lines starting with `#`, are skipped.
"""
function readvalues(path::AbstractString)
    isfile(path) || throw(ArgumentError("$path: no such file"))
    vals = Float64[]
    for (k, line) in enumerate(eachline(path))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        for t in split(s, r"[\s,]+"; keepempty = false)
            x = tryparse(Float64, t)
            x === nothing && throw(ArgumentError("$path, line $k: `$t` is not a number"))
            push!(vals, x)
        end
    end
    isempty(vals) && throw(ArgumentError("$path: no numbers"))
    return vals
end

# BIDS derivative names for the maps that have one (BIDS qMRI: T1map, R1map, ...); any other
# parameter by its own name.
const _MAPNAMES = (T1 = "T1map", R1 = "R1map", T2 = "T2map", R2 = "R2map", T2star = "T2starmap",
                   R2star = "R2starmap", M0 = "M0map", S0 = "S0map", B1 = "B1map")
_mapname(p::Symbol) = haskey(_MAPNAMES, p) ? _MAPNAMES[p] : String(p)

"""
    savemaps(prefix, res, info; stderror = false, sidecar = Dict()) -> Vector{String}

Write a dataset fit's maps as NIfTI files in the data's geometry (`info` from
[`loaddata`](@ref)): one per free parameter (`prefix * "T1map.nii.gz"`, BIDS names where they
exist, else the parameter's name), the fit's status (`status`, the [`FitStatus`](@ref) as an
integer: 0 not fitted, ...) and its SSE, and with `stderror = true` each parameter's standard
error (`T1map_SE`). Beside each, a JSON sidecar with `sidecar`'s entries (how the map was made).
Values are Float32, NaN outside the mask. Returns the files written.
"""
savemaps(prefix::AbstractString, res::FitResult, info::DataInfo; stderror::Bool = false,
         sidecar = Dict{String,Any}()) = _savemaps(String(prefix), res, info, stderror, sidecar)
# Compiled once for any FitResult (`@nospecialize`): a result's type records the fit's settings
# (which flags were given, a mask or not), so each combination would compile this again (0.55 s
# a command); reading its fields dynamically costs microseconds.
function _savemaps(prefix::String, @nospecialize(res::FitResult), info::DataInfo, stderror::Bool,
                   sidecar)
    dir = dirname(prefix)
    isempty(dir) || mkpath(dir)
    size(res.status) == info.spatial || throw(DimensionMismatch("the fit's maps are " *
        "$(join(size(res.status), "×")); the data were $(join(info.spatial, "×"))"))
    maps = Pair{String,Array{Float32,3}}[]
    for (p, v) in pairs(res.params)
        push!(maps, _mapname(p) => Float32.(v))
    end
    push!(maps, "status" => Float32.(Int.(res.status)))
    push!(maps, "SSE" => Float32.(res.sse))
    if stderror
        se = EQUIP.stderror(res)
        for (p, v) in pairs(se)
            push!(maps, _mapname(p) * "_SE" => Float32.(v))
        end
    end
    files = String[]
    for (name, a) in maps
        f = prefix * name * ".nii.gz"
        niwrite(f, NIVolume(_mapheader(info.header, name), a))
        open(io -> JSON.json(io, merge(Dict{String,Any}("Map" => name), sidecar); pretty = true),
             _sidecarpath(f), "w")
        push!(files, f)
    end
    return files
end

# The data's header for a map: its geometry kept, the intensity scaling and display range reset
# (the map's values are written as they are), the map named in `descrip`. The size and data type
# are set from the map when it is written (NIfTI.jl's `niupdate`).
function _mapheader(h, name)
    m = deepcopy(h)
    m.scl_slope = 1
    m.scl_inter = 0
    m.cal_max = 0
    m.cal_min = 0
    m.intent_code = 0
    d = codeunits(first("EQUIP " * name, 79))
    m.descrip = ntuple(i -> i <= length(d) ? d[i] : 0x00, 80)
    return m
end
