# =================================================================================================
# equip fit MODEL DATA... [flags]: maps from data files
#
#   equip fit T2Decay t2.nii.gz                                       TE from the BIDS sidecar
#   equip fit QMTSIR sir.nii.gz --protocol SIR_5POINT --kmf 12.5 --mask brain.nii.gz
#   equip fit ADCDecay dwi.nii.gz --weights @nex.txt --out results/sub-01_
#
# The data are read (io.jl), the model built from the flags and the files (models.jl), the
# parameters' flags turned into `fit`'s keywords (args.jl), and the maps written in the data's
# geometry with a sidecar recording how they were made.
# =================================================================================================

const _FIT_OPTIONS = ["model", "data", "protocol", "mask", "weights", "precision", "out",
                      "stderror", "save", "help"]

function cmd_fit(words; io::IO = stdout, err::IO = stderr)
    a = parseargs(words)
    # The model and the data: positional (`equip fit MODEL DATA...`) or flags (`--model`,
    # `--data a.nii,b.nii`, as a config file holds them).
    pos = copy(a.positional)
    model = has(a, "model") ? value(a, "model") :
            isempty(pos) ? throw(ArgumentError("give the model: `equip fit MODEL DATA...` " *
                                               "(the models: $(join(keys(MODELS), ", ")))")) :
            popfirst!(pos)
    M = modeltype(model)
    if has(a, "help")
        modelhelp(io, M, "fit")
        println(io, "\nOptions: --mask FILE, --weights VALUES, --precision Float32, --out PREFIX, " *
                    "--stderror, --config FILE, --save FILE")
        return 0
    end
    _check_flags(a, M, _FIT_OPTIONS, "fit")
    files = has(a, "data") ? String.(split(value(a, "data"), ',')) : pos
    isempty(files) && throw(ArgumentError("give the data: `equip fit $model DATA.nii.gz` " *
                                          "(one 4-D file, or one file per observation)"))

    precision = _precision(value(a, "precision", "Float64"))
    data, info = loaddata(files; precision)
    m = buildmodel(M, a, info.found; warn = err)
    nobs(m) == size(data, 4) || throw(DimensionMismatch("$model has $(nobs(m)) observations " *
        "($(join(("$k: $(length(getfield(m, k)))" for k in fieldnames(M) if getfield(m, k) isa AbstractVector), ", "))); " *
        "the data have $(size(data, 4)) images"))
    pnames = paramnames(M)
    roles = (; (Symbol(n) => role(n, v, pnames, info) for (n, v) in lastflags(a) if Symbol(n) in pnames)...)
    mask = has(a, "mask") ? loadmap(value(a, "mask"), info) .> 0 : nothing
    weights = has(a, "weights") ? numbers(value(a, "weights"), "weights") : nothing
    prefix = value(a, "out", _stem(files[1]) * "_")

    # The command as a config file, before fitting: a record even if the fit is stopped.
    if has(a, "save")
        acq = [String(k) => _valuestring(getfield(m, k)) for k in fieldnames(M)]
        flags = [["model" => model, "data" => join(files, ',')]; acq;
                 [f for f in lastflags(a) if !(first(f) in ("model", "data", "save", "help",
                                                          "protocol", String.(fieldnames(M))...))]]
        saveconfig(value(a, "save"), flags, "equip fit, saved $(Dates.today()) with EQUIP " *
                   "$(pkgversion(EQUIP))\nrun again: equip fit --config $(value(a, "save"))")
    end

    res = fit(m, data; mask, precision, weights, roles...)
    car = Dict{String,Any}(
        "Model" => model,
        "Acquisition" => Dict(String(k) => getfield(m, k) for k in fieldnames(M)),
        "Fit" => Dict(n => v for (n, v) in lastflags(a) if Symbol(n) in pnames),
        "Data" => files, "Mask" => value(a, "mask", nothing), "Weights" => weights,
        "Precision" => string(precision),
        "Command" => "equip fit " * join(words, " "),
        "Software" => "EQUIP $(pkgversion(EQUIP)) (EQUIPCLI $(pkgversion(@__MODULE__)))")
    written = savemaps(prefix, res, info; stderror = has(a, "stderror"), sidecar = car)
    show(io, MIME"text/plain"(), res)
    println(io, "\n\nwrote ", length(written), " maps: ", join(basename.(written), ", "))
    println(io, "in ", abspath(dirname(written[1])))
    return 0
end

# A field's value as a flag holds it: numbers as a comma list (shortest exact form), a Bool.
_valuestring(v::AbstractVector) = join(repr.(Float64.(v)), ',')
_valuestring(v::Real) = repr(v isa Bool ? v : Float64(v))

function _precision(s)
    s in ("Float64", "64", "double") && return Float64
    s in ("Float32", "32", "single") && return Float32
    throw(ArgumentError("`--precision`: Float64 or Float32, got `$s`"))
end
