# =================================================================================================
# Numbers from text files: a list of acquisition values (TIs, flip angles, ...), an FSL .bval
# (b-values, s/mm², on one line) or .bvec (gradient directions: three lines, x, y and z), as
# dcm2niix writes them beside a diffusion series; or from a variable of a MATLAB file.
# =================================================================================================

"""
    readvalues(path) -> Vector{Float64}

Numbers from a text file: one per line, or several on a line separated by spaces, tabs or
commas (an FSL `.bval` file as it is); empty lines, and lines starting with `#`, are skipped. Or
a MATLAB file's variable (`protocol.mat:TI`, after `using MAT`), as a vector.
"""
function readvalues(path::AbstractString)
    if _ismat(path)
        file, name = _matpath(path)
        name === nothing && throw(ArgumentError("$path: name the variable to read, " *
                                                "$file:NAME"))
        x = readmat(file, name)
        return vec(_matarray(x, path, Float64))
    end
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

"""
    readbvec(path) -> Matrix{Float64}

Gradient directions from an FSL `.bvec` file: a 3×N matrix, one column per volume (the file's
three lines, x, y and z). A file with one direction per line (N lines of three) is read too.
"""
function readbvec(path::AbstractString)
    isfile(path) || throw(ArgumentError("$path: no such file"))
    rows = Vector{Float64}[]
    for (k, line) in enumerate(eachline(path))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        r = Float64[]
        for t in split(s, r"[\s,]+"; keepempty = false)
            x = tryparse(Float64, t)
            x === nothing && throw(ArgumentError("$path, line $k: `$t` is not a number"))
            push!(r, x)
        end
        push!(rows, r)
    end
    isempty(rows) && throw(ArgumentError("$path: no numbers"))
    n = length(rows[1])
    all(r -> length(r) == n, rows) || throw(ArgumentError("$path: the lines have different " *
                                                          "numbers of values"))
    length(rows) == 3 && return permutedims(reduce(hcat, rows))     # FSL: x, y, z lines
    n == 3 && return reduce(hcat, rows)                             # one direction per line
    throw(ArgumentError("$path: a .bvec file has three lines (x, y, z), or three values a " *
                        "line; this one has $(length(rows)) lines of $n"))
end
