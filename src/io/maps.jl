# =================================================================================================
# Maps out: a dataset fit's maps as files, in the data's geometry, each with a JSON sidecar
# saying how it was made (the model, its acquisition, the fit's settings, EQUIP's version)
#
#   writemaps(prefix, res, info)    one NIfTI file per map (BIDS names where they exist), or for
#                                   data from a MATLAB file, one .mat file of all the maps
# =================================================================================================

# BIDS derivative names for the maps that have one (BIDS qMRI: T1map, R1map, ...); any other
# parameter by its own name.
const _MAPNAMES = (T1 = "T1map", R1 = "R1map", T2 = "T2map", R2 = "R2map", T2star = "T2starmap",
                   R2star = "R2starmap", M0 = "M0map", S0 = "S0map", B1 = "B1map")
_mapname(p::Symbol) = haskey(_MAPNAMES, p) ? _MAPNAMES[p] : String(p)

"""
    writemaps(prefix, res, info; stderror = false, sidecar = Dict(), like = nothing,
              format = :auto) -> Vector{String}

Write a dataset fit's maps (`res` from `fit(m, data)`, `info` from [`readdata`](@ref)): each
free parameter (BIDS names where they exist: `T1map`, `R1map`, `T2map`, `S0map`, ...; else the
parameter's name), the quantities the model derives from them ([`derived`](@ref): DTI's `MD`,
`FA`, `AD`, `RD`, `V1x`, `V1y`, `V1z`), the fit's status (the [`FitStatus`](@ref) as an integer: 0 not fitted, ...)
and its SSE, and with `stderror = true` each parameter's standard error (`T2map_SE`). Values
are Float32, NaN outside the mask. Returns the files written.

- `format = :nifti` (the default for NIfTI data): one file per map, `prefix * "T2map.nii.gz"`,
  in the data's geometry (or in `like`'s, an [`MRImage`](@ref) of the same size: for data from
  a MATLAB file), each with a JSON sidecar.
- `format = :mat` (the default for data from a MATLAB file; needs `using MAT`): one file,
  `prefix * "maps.mat"`, a variable per map in the data's own shape, and `prefix * "maps.json"`.

The sidecar records how the maps were made: the model, its acquisition, the fit's settings
(the parameters' roles, the precision, the weights), EQUIP's version; `sidecar`'s entries are
added (and win: the command line adds the command and the data files).
"""
writemaps(prefix::AbstractString, res::FitResult, info::DataInfo; stderror::Bool = false,
          sidecar::AbstractDict = Dict{String,Any}(), like::Union{Nothing,MRImage} = nothing,
          format::Symbol = :auto) =
    _writemaps(String(prefix), res, info, stderror, Dict{String,Any}(sidecar), like, format)

# Compiled once for any FitResult (`@nospecialize`): a result's type records the fit's settings
# (which keywords were given, a mask or not), so each combination would compile this again (0.55
# s a command, measured); reading its fields dynamically costs microseconds.
function _writemaps(prefix::String, @nospecialize(res::FitResult), info::DataInfo, stderror::Bool,
                    sidecar::Dict{String,Any}, like::Union{Nothing,MRImage}, format::Symbol)
    _onevoxel(res) && throw(ArgumentError("writemaps writes a dataset fit's maps; this result " *
                                          "is one voxel's"))
    sz = size(res.status)
    sz == info.spatial || throw(DimensionMismatch("the fit's maps are $(join(sz, "×")); the " *
                                                  "data were $(join(info.spatial, "×"))"))
    format === :auto && (format = info.header === nothing && like === nothing ? :mat : :nifti)
    format in (:nifti, :mat) || throw(ArgumentError("format: :nifti or :mat, got :$format"))
    maps = Pair{String,Array{Float32,3}}[]
    for (p, v) in pairs(res.params)
        push!(maps, _mapname(p) => Float32.(v))
    end
    d = derived(res)                                     # the model's derived quantities, if any
    if d !== nothing
        for (k, v) in pairs(d)
            any(m -> first(m) == String(k), maps) && throw(ArgumentError("$(nameof(typeof(res.model))): " *
                "the derived quantity `$k` has a parameter map's name"))
            push!(maps, String(k) => Float32.(v))
        end
    end
    push!(maps, "status" => Float32.(Int.(res.status)))
    push!(maps, "SSE" => Float32.(res.sse))
    if stderror
        for (p, v) in pairs(EQUIP.stderror(res))
            push!(maps, _mapname(p) * "_SE" => Float32.(v))
        end
    end
    car = merge(_provenance(res), sidecar)
    dir = dirname(prefix)
    isempty(dir) || mkpath(dir)
    format === :mat && return _savemat(prefix, maps, info, car)
    geometry = like !== nothing ? like.header : info.header
    if like !== nothing
        size(like)[1:3] == info.spatial || throw(DimensionMismatch("`like` is " *
            "$(join(size(like), "×")); the maps are $(join(info.spatial, "×"))"))
    end
    files = String[]
    for (name, a) in maps
        img = MRImage(a, geometry; meta = merge(Dict{String,Any}("Map" => name), car),
                      descrip = "EQUIP " * name)
        push!(files, writeimage(prefix * name * ".nii.gz", img))
    end
    return files
end

# The maps in one MATLAB file, each in the data's own shape (as the array was stored), and the
# sidecar beside it.
function _savemat(prefix, maps, info, car)
    f = prefix * "maps.mat"
    writemat(f, Dict{String,Any}(name => reshape(a, info.shape...) for (name, a) in maps))
    writesidecar(prefix * "maps.json", merge(Dict{String,Any}("Maps" => first.(maps)), car))
    return [f]
end

# How the maps were made, from the result: the model and its acquisition (its fields), the fit's
# settings (each parameter's role as written, `Free(start = 0.07)`, `Fixed(12.5)`, or "a map"; the
# precision; the weights), and EQUIP's version.
function _provenance(@nospecialize(res::FitResult))
    m = res.model
    acq = Dict{String,Any}(String(k) => _jsonvalue(getfield(m, k)) for k in fieldnames(typeof(m)))
    fit = Dict{String,Any}()
    for (k, v) in pairs(res.settings)
        k === :mask && continue
        fit[String(k)] = k === :weights ? _jsonvalue(v) : v isa AbstractArray ? "a map" :
                         v isa Real ? v : repr(v)
    end
    return Dict{String,Any}("Model" => string(nameof(typeof(m))), "Acquisition" => acq,
                            "Fit" => fit, "Software" => "EQUIP $(pkgversion(EQUIP))")
end
_jsonvalue(v::Union{Real,AbstractString,Symbol}) = v isa Symbol ? String(v) : v
_jsonvalue(v::AbstractVector{<:Real}) = collect(v)
_jsonvalue(v) = repr(v)
