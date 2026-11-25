# =================================================================================================
# The models the command line offers, and building one from the flags: its acquisition from
# `--TI ...` (one flag per field), a named protocol (`--protocol SIR_5POINT`), and the values
# found beside the data (BIDS sidecars, .bval). Priority: the flags, then a named protocol, then
# the files. `--help` after a model lists its fields and parameters, generated from the model.
# =================================================================================================

# The built-in models, by name (a user's own model is used from Julia).
const MODELS = (; T1IR, R1IR, T1IRTD, R1IRTD, T1SR, R1SR, T1MFA, R1MFA, T2Decay, R2Decay,
                  T2DecayOffset, R2DecayOffset, T2BiexpDecay, R2BiexpDecay, ADCDecay, ADCKurtosis,
                  DTI, DKI, StickZeppelin, StickZeppelinBall, BallStick, NODDI, NODDIConvolved, BinghamNODDI,
                  BinghamNODDIConvolved, IVIM, QMTSIR, QMTSIREqualR1)
# The directional models: their acquisition is each image's b and gradient direction, read from
# an FSL .bval and .bvec beside the data (as dcm2niix writes them), or given as flags.
_directional(M) = (:gx, :gy, :gz) ⊆ fieldnames(M)
# Named protocols (acquisitions), by name, and the models each is for.
const _SIR = (QMTSIR, QMTSIREqualR1)
const PROTOCOLS = (; SIR_5POINT = (_SIR, SIR_5POINT), SIR_4POINT = (_SIR, SIR_4POINT),
                     SIR_16POINT = (_SIR, SIR_16POINT))

function modeltype(name::AbstractString)
    haskey(MODELS, Symbol(name)) && return MODELS[Symbol(name)]
    throw(ArgumentError("unknown model `$name`; the models: $(join(keys(MODELS), ", "))"))
end
paramnames(M) = keys(parameters(M))

"""
    modelarg(a, pos, cmd, synopsis, io) -> the model's name, or nothing

The command's model: `--model NAME`, or the first positional word (taken from `pos`). Without
one, `--help` prints the command's own help (what it takes, the models, how to see one model's
flags) and gives `nothing`, so that `equip fit --help` works as `equip fit QMTSIR --help` does;
otherwise a mistake naming the models.
"""
function modelarg(a::Args, pos, cmd, synopsis, io::IO)
    has(a, "model") && return value(a, "model")
    isempty(pos) || return popfirst!(pos)
    models = join(keys(MODELS), ", ")
    if has(a, "help")
        println(io, "equip $cmd $synopsis\n\nThe models: $models\n\nA model's flags (its " *
                    "acquisition and parameters) and the options: equip $cmd MODEL --help")
        return nothing
    end
    throw(ArgumentError("give the model: `equip $cmd $(split(synopsis, ':')[1])` (the models: $models)"))
end

"""
    buildmodel(M, a::Args, found; warn) -> model

The model of type `M` with its acquisition from, in order of priority: the flags named after
its fields (`--TI 0.01,0.05`; a Bool field, `--magnitude false`), a named protocol
(`--protocol SIR_5POINT`), and the values found beside the data (`found`, from `readdata`).
A field the flags give that the files also give differently is reported to `warn`.
"""
function buildmodel(M, a::Args, found; warn::IO = stderr)
    fields = fieldnames(M)
    vals = Dict{Symbol,Any}()
    for (k, v) in pairs(found)                                # 3. the files
        k in fields && (vals[k] = v)
    end
    if has(a, "protocol")                                     # 2. a named protocol
        p = Symbol(value(a, "protocol"))
        haskey(PROTOCOLS, p) || throw(ArgumentError("unknown protocol `$p`; the protocols: " *
                                                    "$(join(keys(PROTOCOLS), ", "))"))
        PM, acq = PROTOCOLS[p]
        M in PM || throw(ArgumentError("protocol `$p` is for $(join(nameof.(PM), " or ")), not $(nameof(M))"))
        merge!(vals, Dict(pairs(acq)))
    end
    for k in fields                                           # 1. the flags
        has(a, String(k)) || continue
        s = value(a, String(k))
        v = fieldtype(M, k) === Bool ? _bool(s, k) : (x = numbers(s, k); length(x) == 1 ? x[1] : x)
        if haskey(found, k) && !(length(found[k]) == length(v) && isapprox(found[k], v; rtol = 1e-6))
            println(warn, "equip: warning: --$k differs from the value in the data's sidecar files " *
                          "($(found[k])); using --$k")
        end
        vals[k] = v
    end
    try
        return M(; vals...)
    catch e
        e isa UndefKeywordError || rethrow()
        k = e.var
        _directional(M) && throw(ArgumentError("$(nameof(M)) needs `$k`: put the FSL .bval and " *
            ".bvec files beside the data, with the data's name (dwi.nii.gz: dwi.bval, dwi.bvec; " *
            "dcm2niix writes them), or give --b, --gx, --gy and --gz (one value per image: a " *
            "comma list, start:step:stop, or @file); they were not found beside the data"))
        hint = M === QMTSIR ? ", or a named protocol (`--protocol SIR_5POINT`)" : ""
        throw(ArgumentError("$(nameof(M)) needs `$k`: give --$k (values in seconds: a comma " *
            "list, start:step:stop, or @file)$hint; the data's sidecar files did not have it"))
    end
end

function _bool(s, k)
    lowercase(s) in ("true", "yes", "1") && return true
    lowercase(s) in ("false", "no", "0") && return false
    throw(ArgumentError("`--$k`: true or false, got `$s`"))
end

# The flags a subcommand accepts for model M: its fields, its parameters, the options.
function _check_flags(a::Args, M, options, cmd)
    known = Set([String.(fieldnames(M))...; String.(paramnames(M))...; options...])
    for (name, _) in a.flags
        name in known && continue
        throw(ArgumentError("`--$name` is not a flag of `equip $cmd $(nameof(M))`. Its " *
            "acquisition: $(join("--" .* String.(fieldnames(M)), ", ")); its parameters: " *
            "$(join("--" .* String.(paramnames(M)), ", ")); options: " *
            "$(join("--" .* options, ", ")). `equip $cmd $(nameof(M)) --help` explains them"))
    end
end

"""
    modelhelp(io, M, cmd)

What `equip <cmd> <model> --help` prints: the model's acquisition fields and parameters (from
the model itself: its fields, `parameters(M)`), and how to set them.
"""
function modelhelp(io::IO, M, cmd)
    println(io, "equip $cmd $(nameof(M)): flags named after the model's fields and parameters")
    println(io, "\nAcquisition (times in seconds, angles in degrees, b in s/mm²; a comma list, " *
                "start:step:stop, or @file):")
    _directional(M) && println(io, "  (read from the .bval and .bvec beside the data, FSL's, " *
                               "when they are there; the flags override them)")
    for k in fieldnames(M)
        t = fieldtype(M, k) === Bool ? "true or false" : "numbers"
        println(io, "  --", rpad(k, 12), t)
    end
    ps = [p for (p, (PM, _)) in pairs(PROTOCOLS) if M in PM]
    isempty(ps) || println(io, "  --", rpad("protocol", 12), join(ps, " | "), " (named acquisitions)")
    println(io, "\nParameters (the model's defaults; change any):")
    for (n, r) in pairs(parameters(M))
        println(io, "  --", rpad(n, 12), _describe(r))
    end
    names = String.(paramnames(M))
    p, q = names[end], names[1]
    println(io, "\nTo change one, e.g.:")
    println(io, "  --", rpad("$p 1.0", 22), "fix at a number")
    println(io, "  --", rpad("$p free", 22), "estimate it (free,start=1.0,lower=0,upper=2: with a start or bounds)")
    length(names) > 1 && println(io, "  --", rpad("$p $q", 22), "tie it to another parameter")
    println(io, "  --", rpad("$p $(lowercase(p))map.nii.gz", 22), "fix it to a map (equip fit)")
end
function _describe(r::Free)
    start = r.start isa Function ? "from the acquisition" :
            r.start === :scale ? "scaled to the data" : r.start === :data ? "from the data" :
            r.start === :loglinear ? "the log-linear fit to the data" :
            r.start isa BestOf ? "the best of $(length(r.start.values)) values" : string(r.start)
    return "free, start $start, bounds [$(something(r.lower, -Inf)), $(something(r.upper, Inf))]"
end
_describe(r::Fixed) = (r.value isa Symbol ? "fixed: tied to $(r.value)" : "fixed at $(r.value)") *
    (r.lower === nothing && r.upper === nothing ? "" :
     "; bounds [$(something(r.lower, -Inf)), $(something(r.upper, Inf))] when freed")
