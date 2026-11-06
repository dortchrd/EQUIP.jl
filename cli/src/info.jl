# =================================================================================================
# equip info FILE... [--find PATTERN]: what is known about data files
#
#   equip info sub-01_T2w.nii.gz               the header, readable; the sidecar's entries; the
#                                              files beside it (.bval, .bvec); what EQUIP reads
#   equip info sub-01_T2w.nii.gz --find echo   only the sidecar's entries whose names contain
#                                              "echo" (any case; nested entries too)
#   equip info data.mat                        a MATLAB file's variables: their sizes and types
#
# A NIfTI file's header is read alone (fast for any size of image). DICOM is converted first,
# with dcm2niix, whose sidecar holds what the DICOM headers say, in BIDS names and units: the
# data dictionary this command shows and searches.
# =================================================================================================

const _INFO_HELP = """
equip info FILE... [--find PATTERN]

What is known about data files: a NIfTI file's header (readable), its JSON sidecar's entries,
the files beside it (.bval, .bvec), and the values EQUIP reads from them for the models' fields;
a MATLAB file's variables (data.mat, or data.mat:NAME). --find PATTERN shows only the entries
whose names contain PATTERN (any case). DICOM: convert it first with dcm2niix
(dcm2niix -b y -z y -o out/ dicomdir/), which writes the sidecar.
"""

function cmd_info(words; io::IO = stdout, err::IO = stderr)
    a = parseargs(words)
    has(a, "help") && (print(io, _INFO_HELP); return 0)
    for (n, _) in a.flags
        n in ("find", "help") || throw(ArgumentError("`--$n` is not a flag of `equip info`; " *
                                                     "its flags: --find PATTERN, --help"))
    end
    isempty(a.positional) && throw(ArgumentError("give a file: `equip info sub-01_T2w.nii.gz` " *
                                                 "(NIfTI), or `equip info data.mat`"))
    pattern = has(a, "find") ? value(a, "find") : nothing
    for (k, f) in enumerate(a.positional)
        k > 1 && println(io)
        MRImages._ismat(f) ? _matinfo(io, f, pattern) : _niftiinfo(io, f, pattern)
    end
    return 0
end

function _niftiinfo(io, path, pattern)
    MRImages._isnifti(path) || throw(ArgumentError("$path: equip info reads NIfTI files (.nii, " *
        ".nii.gz) and MATLAB files (.mat); DICOM: convert it with dcm2niix first"))
    isfile(path) || throw(ArgumentError("$path: no such file"))
    car = MRImages.sidecarpath(path)
    meta = MRImages.readsidecar(path)
    println(io, path)
    if pattern !== nothing                       # --find: the matching entries, nothing else
        isfile(car) || (println(io, "  no sidecar (", basename(car), ")"); return)
        found = findmeta(meta, pattern)
        isempty(found) ? println(io, "  no entry of ", basename(car), " matches `", pattern, "`") :
                         _rows(io, found)
        return
    end
    h = MRImages._readheader(path)
    _rows(io, headerinfo(h))
    println(io)
    if isfile(car)
        println(io, "sidecar ", basename(car), " (", length(meta), " entries)")
        _rows(io, sort!(collect(meta); by = first))
    else
        println(io, "no sidecar (", basename(car), "): EQUIP takes the acquisition from flags")
    end
    stem = filestem(path)
    for ext in (".bval", ".bvec")
        isfile(stem * ext) && println(io, "beside it: ", basename(stem * ext))
    end
    # What loaddata finds for the models' fields: one value per volume.
    n = h.dim[1] >= 4 ? Int(h.dim[5]) : 1
    found = try
        EQUIP._found([path], [meta], [n])
    catch e
        e isa ArgumentError || rethrow()
        println(io, "\nEQUIP cannot use these values: ", e.msg)
        return
    end
    isempty(found) && return
    println(io, "\nEQUIP reads, for the models' fields (", n, n == 1 ? " volume)" : " volumes)")
    source = Dict{Symbol,String}(f => String(k) for (k, f) in pairs(EQUIP._BIDS))
    source[:b] = basename(stem * ".bval")
    foreach(f -> source[f] = basename(stem * ".bvec"), (:gx, :gy, :gz))
    rows = [String(f) * " (from " * source[f] * ")" => v
            for (f, v) in sort!(collect(found); by = first)]
    _rows(io, rows)
end

# A MATLAB file's variables (or one), with their sizes and types. MAT is loaded already: the
# command's words named a .mat file (mat.jl).
function _matinfo(io, path, pattern)
    file, name = MRImages._matpath(path)
    vars = name === nothing ? MRImages.readmat(file) : Dict{String,Any}(name => MRImages.readmat(file, name))
    println(io, path)
    names = sort!(collect(keys(vars)))
    pattern === nothing || filter!(n -> occursin(lowercase(pattern), lowercase(n)), names)
    isempty(names) && (println(io, "  no variable matches `", pattern, "`"); return)
    _rows(io, [n => _matsummary(vars[n]) for n in names])
end
_matsummary(x::AbstractArray) = length(x) <= 6 && ndims(x) <= 2 && eltype(x) <: Real ?
    "$(join(size(x), "×")) $(eltype(x)): " * join(_cell.(vec(x)), ", ") :
    "$(join(size(x), "×")) $(eltype(x))"
_matsummary(x) = _value(x)

# Name–value rows, the names aligned; long values shortened.
function _rows(io, rows)
    w = min(30, maximum(r -> length(first(r)), rows; init = 0))
    for (k, v) in rows
        println(io, "  ", rpad(k, w), "  ", v isa AbstractString ? _clip(v) : _value(v))
    end
end
function _value(v)
    s = v isa AbstractVector ?
        (length(v) <= 6 ? "[" * join(_cell.(v), ", ") * "]" :
         "[" * join(_cell.(v[1:5]), ", ") * ", …]  ($(length(v)) values)") :
        v isa AbstractDict ? "{…}  ($(length(v)) entries: --find to list them)" : _cell(v)
    return _clip(s)
end
_cell(x) = x isa AbstractString ? x : x isa AbstractFloat ? repr(x) : string(x)
_clip(s) = length(s) > 100 ? first(s, 99) * "…" : s
