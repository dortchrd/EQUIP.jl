# =================================================================================================
# The command line's words: flags (`--TI 0.01,0.05`), config files made of flags (FSL's
# `--config=` style: one argument per line, `#` comments), and the values flags hold (numbers,
# lists, ranges, text files, parameters' roles).
#
# Every flag but the boolean ones takes exactly one value, the next word, so a negative number
# (`--Sf -0.95`) is a value, never mistaken for a flag. `--name=value` works too.
# =================================================================================================

const _BOOLEAN = ("help", "stderror", "compress")   # flags without a value

"""
    Args

A parsed command line: the positional words, and the flags in order (`name => value`; a flag
given twice keeps both, and the last wins). Words from `--config` files come first, so flags
typed on the command line override them.
"""
struct Args
    positional::Vector{String}
    flags::Vector{Pair{String,String}}
end

has(a::Args, name) = any(f -> first(f) == name, a.flags)
value(a::Args, name) = last(a.flags[findlast(f -> first(f) == name, a.flags)])
value(a::Args, name, default) = has(a, name) ? value(a, name) : default
# Each flag once, in the order first given, with its last value (what --save writes).
function lastflags(a::Args)
    names = unique(first.(a.flags))
    return [n => value(a, n) for n in names]
end

function parseargs(words::AbstractVector{<:AbstractString})
    words = String.(words)
    # 1. The config files' words first, wherever --config appears.
    front, rest = String[], String[]
    i = 1
    while i <= length(words)
        w = words[i]
        if w == "--config"
            i < length(words) || throw(ArgumentError("`--config` needs a file"))
            append!(front, configwords(words[i+1]))
            i += 2
        elseif startswith(w, "--config=")
            append!(front, configwords(w[10:end]))
            i += 1
        else
            push!(rest, w)
            i += 1
        end
    end
    # 2. Flags and positional words (and MAT loaded, if a word names a MATLAB file).
    all = [front; rest]
    _loadmat(all)
    positional, flags = String[], Pair{String,String}[]
    i = 1
    while i <= length(all)
        w = all[i]
        if startswith(w, "--") && length(w) > 2
            name, eq, val = _split_eq(w[3:end])
            name == "config" && throw(ArgumentError("a config file cannot read another (`--config` in one)"))
            if eq
                push!(flags, name => val)
                i += 1
            elseif name in _BOOLEAN
                push!(flags, name => "true")
                i += 1
            else
                i < length(all) || throw(ArgumentError("`--$name` needs a value"))
                push!(flags, name => all[i+1])
                i += 2
            end
        elseif w == "-h"
            push!(flags, "help" => "true")
            i += 1
        else
            push!(positional, w)
            i += 1
        end
    end
    return Args(positional, flags)
end

function _split_eq(s)
    k = findfirst('=', s)
    k === nothing ? (s, false, "") : (s[1:k-1], true, s[k+1:end])
end

"""
    configwords(path) -> Vector{String}

The words of a config file: what one would type on the command line, one flag per line or
several, `#` starting a comment, double quotes around a value with spaces.
"""
function configwords(path::AbstractString)
    isfile(path) || throw(ArgumentError("config file $path: no such file"))
    words = String[]
    for line in eachline(path)
        s = strip(replace(line, r"(^|\s)#.*$" => ""))
        isempty(s) && continue
        for m in eachmatch(r"\"([^\"]*)\"|(\S+)", s)
            push!(words, something(m.captures[1], m.captures[2]))
        end
    end
    return words
end

"""
    saveconfig(path, flags, header)

Write `flags` (`name => value` pairs) as a config file, one per line, after a `#` header: what
`--config path` reads back.
"""
function saveconfig(path::AbstractString, flags, header::AbstractString)
    open(path, "w") do io
        for line in split(header, '\n')
            println(io, "# ", line)
        end
        for (name, val) in flags
            v = occursin(r"\s", val) || isempty(val) ? "\"" * val * "\"" : val
            println(io, "--", name, " ", v)
        end
    end
    return path
end

"""
    numbers(s, what) -> Vector{Float64}

The numbers a flag gives: a comma list (`0.01,0.05,0.056`), ranges as `start:step:stop`
(`0.01:0.01:0.32`, mixed with lists too), or `@file`, a text file of numbers
([`readvalues`](@ref): one per line, or separated by spaces, an FSL `.bval` as it is), or a
MATLAB file's variable (`@protocol.mat:TE`). `what` names the flag in errors.
"""
function numbers(s::AbstractString, what)
    startswith(s, "@") && return readvalues(s[2:end])
    vals = Float64[]
    for piece in split(s, ','; keepempty = false)
        if occursin(':', piece)
            parts = split(piece, ':')
            length(parts) == 3 || throw(ArgumentError("`--$what`: a range is start:step:stop " *
                                                      "(e.g. 0.01:0.01:0.32), got `$piece`"))
            a, step, b = (_number(p, what) for p in parts)
            r = a:step:b
            isempty(r) && throw(ArgumentError("`--$what`: the range `$piece` is empty"))
            append!(vals, collect(r))
        else
            push!(vals, _number(piece, what))
        end
    end
    isempty(vals) && throw(ArgumentError("`--$what`: no numbers in `$s`"))
    return vals
end
function _number(s, what)
    x = tryparse(Float64, strip(s))
    x === nothing && throw(ArgumentError("`--$what`: `$(strip(s))` is not a number"))
    return x
end

"""
    role(name, s, params, info) -> a parameter's setting for `fit`

A parameter's flag, by `fit`'s rules: a number fixes it (`--kmf 12.5`); `free` estimates it,
with a start or bounds if given (`--R1f free,start=0.8,lower=0`); another parameter's name ties
it (`--R1m R1f`); a NIfTI file or a MATLAB array is a map of fixed values (`--Sm sm.nii.gz`,
`--B1 maps.mat:B1`, checked against the data: `info` from `readdata`; `nothing` where there are
no data, so no maps).
"""
function role(name, s::AbstractString, params, info)
    t = strip(s)
    if t == "free" || startswith(t, "free,")
        kw = Pair{Symbol,Float64}[]
        for item in split(t, ',')[2:end]
            k, eq, v = _split_eq(strip(item))
            (eq && k in ("start", "lower", "upper")) || throw(ArgumentError("`--$name $s`: " *
                "after `free`, give start=, lower= or upper=, e.g. `free,start=0.8,lower=0`"))
            push!(kw, Symbol(k) => _number(v, name))
        end
        return Free(; kw...)
    end
    x = tryparse(Float64, t)
    x === nothing || return x
    Symbol(t) in params && return Symbol(t)
    if MRImages._isnifti(t) || MRImages._ismat(t)
        info === nothing && throw(ArgumentError("`--$name $s`: a map needs data (maps are for " *
                                                "`equip fit`); give a number"))
        return readmap(t, info)
    end
    throw(ArgumentError("`--$name $s`: give a number (fixed), `free` (estimated; " *
        "`free,start=0.8,lower=0`), another parameter's name (a tie: `--R1m R1f`), or a map " *
        "(.nii, .nii.gz, or file.mat:NAME)"))
end
