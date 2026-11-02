# =================================================================================================
# BIDS JSON sidecars: the metadata beside an image (`sub-01_T2w.json` beside
# `sub-01_T2w.nii.gz`), as dcm2niix writes them from DICOM: EchoTime, RepetitionTime,
# InversionTime (s), FlipAngle (degrees), and the scanner's details. Read into a
# `Dict{String,Any}` (nested objects too), as they are: no keys renamed, no units converted.
# =================================================================================================

"""
    sidecarpath(path) -> String

The JSON sidecar's path for an image: its [`filestem`](@ref) and `.json`
(`sub-01_T2w.nii.gz` → `sub-01_T2w.json`).
"""
sidecarpath(path::AbstractString) = filestem(path) * ".json"

"""
    readsidecar(path) -> Dict{String,Any}

The JSON sidecar of the image at `path` (or the JSON file `path` itself): its entries, empty if
there is no sidecar. Numbers, strings, lists (`Vector{Any}`) and nested objects
(`Dict{String,Any}`) as in the file.
"""
function readsidecar(path::AbstractString)
    p = endswith(lowercase(path), ".json") ? path : sidecarpath(path)
    isfile(p) || return Dict{String,Any}()
    d = try
        JSON.parsefile(p; dicttype = Dict{String,Any})
    catch e
        throw(ArgumentError("$p: not a valid JSON file ($(sprint(showerror, e)))"))
    end
    d isa Dict{String,Any} || throw(ArgumentError("$p: a JSON sidecar is an object " *
                                                  "({\"key\": value, ...}); this file is not"))
    return d
end

"""
    writesidecar(path, meta) -> String

Write `meta` (a dictionary: the keys as strings) as the JSON sidecar of the image at `path` (or
to `path` itself, if it ends in `.json`), indented to be read. Returns the file written.
"""
function writesidecar(path::AbstractString, meta::AbstractDict)
    p = endswith(lowercase(path), ".json") ? path : sidecarpath(path)
    open(io -> JSON.json(io, meta; pretty = true), p, "w")
    return p
end
