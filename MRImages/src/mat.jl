# =================================================================================================
# MATLAB files (.mat), through MAT.jl: a package extension (ext/MRImagesMATExt.jl), loaded when
# the user loads MAT (`using MAT`). MAT brings HDF5 (MATLAB's v7.3 files are HDF5) and takes
# about 0.4 s to load, so only those who read MATLAB files pay for it.
#
# Here: the functions the extension adds methods to (`_matread`, `_matwrite`, `_matnames`), and
# the public functions, which check that the extension is loaded (and say how to load it), then
# call it. They call it through `invokelatest`, so that the extension's methods are found even
# when MAT was loaded after the calling code started (a program that loads MAT only when given a
# `.mat` file: the `equip` command).
# =================================================================================================

function _matread end         # (file, name) -> the variable's value
function _matwrite end        # (file, vars) -> writes the variables, a new file
function _matnames end        # (file) -> the variables' names

_matloaded() = Base.get_extension(@__MODULE__, :MRImagesMATExt) !== nothing
function _needmat(path)
    _matloaded() && return nothing
    throw(ArgumentError("$path: MATLAB files need MAT.jl: `using MAT` (installed once with " *
                        "`import Pkg; Pkg.add(\"MAT\")`)"))
end

"""
    readmat(path) -> Dict{String,Any}
    readmat(path, name)

A MATLAB file's variables, by name (as MAT.jl's `matread`), or one of them. Needs `using MAT`.
"""
function readmat(path::AbstractString)
    _needmat(path)
    isfile(path) || throw(ArgumentError("$path: no such file"))
    names = Base.invokelatest(_matnames, path)::Vector{String}
    return Dict{String,Any}(n => Base.invokelatest(_matread, path, n) for n in names)
end
function readmat(path::AbstractString, name::AbstractString)
    _needmat(path)
    isfile(path) || throw(ArgumentError("$path: no such file"))
    names = Base.invokelatest(_matnames, path)::Vector{String}
    name in names || throw(ArgumentError("$path has no variable `$name`; its variables: " *
                                         "$(join(sort(names), ", "))"))
    return Base.invokelatest(_matread, path, String(name))
end

"""
    writemat(path, vars) -> String

Write variables (a dictionary from names to values: arrays, numbers, strings) to a new MATLAB
file (v7.3, which MATLAB reads with `load`). Needs `using MAT`.
"""
function writemat(path::AbstractString, vars::AbstractDict)
    _needmat(path)
    for k in keys(vars)
        occursin(r"^[A-Za-z][A-Za-z0-9_]*$", string(k)) || throw(ArgumentError("`$k` is not " *
            "a MATLAB variable name (a letter, then letters, digits or underscores)"))
    end
    Base.invokelatest(_matwrite, String(path), Dict{String,Any}(string(k) => v for (k, v) in vars))
    return path
end

# The array a `.mat` path names (`file.mat:S`, or `file.mat` with one variable), as precision T.
function _readmatimage(path::AbstractString, ::Type{T}) where {T<:AbstractFloat}
    file, name = _matpath(path)
    _needmat(file)
    isfile(file) || throw(ArgumentError("$file: no such file"))
    names = Base.invokelatest(_matnames, file)::Vector{String}
    if name === nothing
        length(names) == 1 || throw(ArgumentError("$file has $(length(names)) variables; name " *
            "the one to read: $file:NAME (its variables: $(join(sort(names), ", ")))"))
        name = names[1]
    end
    x = readmat(file, name)
    return MRImage(_matarray(x, "$file:$name", T), nothing, Dict{String,Any}(), String(path))
end

# A MATLAB variable as an array of precision T: numbers (MATLAB's integer and logical types
# too: images are often uint16), not a struct, a string or complex values.
function _matarray(x, what, ::Type{T}) where {T}
    x isa Real && return fill(T(x))
    x isa AbstractArray{<:Real} || throw(ArgumentError("$what is not a real numeric array " *
        "(it is a $(nameof(typeof(x))))"))
    return T.(x)
end
