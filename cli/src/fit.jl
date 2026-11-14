# =================================================================================================
# equip fit MODEL DATA... [flags]: maps from data files
#
#   equip fit T2Decay t2.nii.gz                                       TE from the BIDS sidecar
#   equip fit QMTSIR sir.nii.gz --protocol SIR_5POINT --kmf 12.5 --mask brain.nii.gz
#   equip fit ADCDecay dwi.nii.gz --weights @nex.txt --out results/sub-01_
#   equip fit DTI dwi.nii.gz --likelihood rician --sigma 50           magnitude data, σ known
#   equip fit QMTSIR sir.nii --TI ... --TD ... --format both             NIfTI maps and one .mat
#
# The data are read (EQUIP's `readdata`), the model built from the flags and the files
# (models.jl), the parameters' flags turned into `fit`'s keywords (args.jl), and the maps written
# in the data's geometry with a sidecar recording how they were made (EQUIP's `writemaps`): NIfTI
# for NIfTI data, one .mat file for MATLAB data (or NIfTI in another image's geometry: --like);
# `--format nifti|mat|both` chooses (MATLAB's `load` reads the .mat without a toolbox: the
# MATLAB wrapper, matlab/equip_fit.m, asks for both).
# =================================================================================================

const _FIT_OPTIONS = ["model", "data", "protocol", "mask", "weights", "precision", "out",
                      "like", "stderror", "save", "likelihood", "sigma", "method", "format", "help"]

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
                    "--stderror, --like REF.nii.gz (NIfTI maps for MATLAB data), --format nifti, mat " *
                    "or both (the maps as NIfTI, as one PREFIX_maps.mat, or both), --config FILE, " *
                    "--save FILE; for magnitude data at low SNR, --likelihood rician --sigma SD " *
                    "(the noise SD of each channel: a number, or a map as FILE.nii.gz or " *
                    "FILE.mat:NAME; not estimated) and --method ml (the default), mean or wmean")
        return 0
    end
    _check_flags(a, M, _FIT_OPTIONS, "fit")
    files = has(a, "data") ? String.(split(value(a, "data"), ',')) : pos
    isempty(files) && throw(ArgumentError("give the data: `equip fit $model DATA.nii.gz` " *
                                          "(one 4-D file, or one file per observation; or " *
                                          "data.mat:NAME)"))

    precision = _precision(value(a, "precision", "Float64"))
    data, info = readdata(files; precision)
    m = buildmodel(M, a, info.found; warn = err)
    nobs(m) == size(data, 4) || throw(DimensionMismatch("$model has $(nobs(m)) observations " *
        "($(join(("$k: $(length(getfield(m, k)))" for k in fieldnames(M) if getfield(m, k) isa AbstractVector), ", "))); " *
        "the data have $(size(data, 4)) images"))
    pnames = paramnames(M)
    roles = (; (Symbol(n) => role(n, v, pnames, info) for (n, v) in lastflags(a) if Symbol(n) in pnames)...)
    mask = has(a, "mask") ? readmap(value(a, "mask"), info) .> 0 : nothing
    weights = has(a, "weights") ? numbers(value(a, "weights"), "weights") : nothing
    noise = _fitnoise(a, info)                   # (;), or the Rician fit's keywords
    like = has(a, "like") ? readimage(value(a, "like")) : nothing
    format = _mapformat(a, info, like)
    prefix = value(a, "out", filestem(files[1]) * "_")

    # The command as a config file, before fitting: a record even if the fit is stopped.
    if has(a, "save")
        acq = [String(k) => _valuestring(getfield(m, k)) for k in fieldnames(M)]
        flags = [["model" => model, "data" => join(files, ',')]; acq;
                 [f for f in lastflags(a) if !(first(f) in ("model", "data", "save", "help",
                                                          "protocol", String.(fieldnames(M))...))]]
        saveconfig(value(a, "save"), flags, "equip fit, saved $(Dates.today()) with EQUIP " *
                   "$(pkgversion(EQUIP))\nrun again: equip fit --config $(value(a, "save"))")
    end

    res = fit(m, data; mask, precision, weights, noise..., roles...)
    # The sidecar: what writemaps records (the model, its acquisition, the fit's settings), and
    # the command's own: the flags as typed, the files.
    car = Dict{String,Any}(
        "Fit" => Dict(n => v for (n, v) in lastflags(a) if Symbol(n) in pnames),
        "Data" => files, "Mask" => value(a, "mask", nothing), "Weights" => weights,
        "Precision" => string(precision),
        "Likelihood" => value(a, "likelihood", "gaussian"),
        "Command" => "equip fit " * join(words, " "),
        "Software" => "EQUIP $(pkgversion(EQUIP)) (EQUIPCLI $(pkgversion(@__MODULE__)))")
    if haskey(noise, :σ)                         # σ as typed: a number, or the map's file
        car["Sigma"] = value(a, "sigma")
        car["Method"] = String(noise.method)
    end
    writeas(f) = writemaps(prefix, res, info; stderror = has(a, "stderror"), sidecar = car, like,
                           format = f)
    written = format === :both ? [writeas(:nifti); writeas(:mat)] : writeas(format)
    show(io, MIME"text/plain"(), res)
    nii = filter(!MRImages._ismat, written)
    mat = filter(MRImages._ismat, written)
    isempty(nii) || println(io, "\n\nwrote ", length(nii), " maps: ", join(basename.(nii), ", "))
    isempty(mat) || println(io, isempty(nii) ? "\n\n" : "", "wrote the maps to ", basename(mat[1]))
    println(io, "in ", abspath(dirname(written[1])))
    return 0
end

# How the maps are written (`--format`): as writemaps chooses by default (NIfTI for NIfTI data or
# with --like, one .mat for MATLAB data), or as asked. NIfTI maps need a NIfTI geometry: the
# data's, or --like's. A .mat file needs MAT, loaded here when the command named no .mat file.
function _mapformat(a, info, like)
    has(a, "format") || return :auto
    f = lowercase(value(a, "format"))
    f in ("nifti", "mat", "both") || throw(ArgumentError("`--format`: nifti, mat or both, got `$f`"))
    f in ("nifti", "both") && info.header === nothing && like === nothing &&
        throw(ArgumentError("`--format $f`: NIfTI maps of MATLAB data need a geometry: give " *
                            "--like REF.nii.gz"))
    f in ("mat", "both") && _requiremat()
    return Symbol(f)
end

# The noise model the fit assumes (EQUIP's `fit(m, data; likelihood, σ, method)`): least squares
# by default, `(;)`; with `--likelihood rician`, the Rician fit of magnitude data, which needs
# `--sigma`, the noise SD of each channel: a number, or a map read as a fixed parameter's (a NIfTI
# file or file.mat:NAME, checked against the data; voxels where it is 0 or NaN are not fitted).
# σ is the user's: EQUIP does not estimate it (docs/design.md, tier 2 step 3, on hold: background
# estimates do not hold on clinical images). `--method`: ml (maximum likelihood, the default),
# mean (least squares on the Rician mean) or wmean (the same, weighted by 1/Var).
function _fitnoise(a::Args, info)
    lik = value(a, "likelihood", "gaussian")
    lik in ("gaussian", "rician") || throw(ArgumentError("`--likelihood`: gaussian (least " *
        "squares, the default) or rician (magnitude data, with --sigma), got `$lik`"))
    if lik == "gaussian"
        for f in ("sigma", "method")
            has(a, f) && throw(ArgumentError("`--$f` is for `--likelihood rician` (magnitude " *
                                             "data); least squares, the default, takes none"))
        end
        return (;)
    end
    has(a, "sigma") || throw(ArgumentError("`--likelihood rician` needs `--sigma`, the noise SD " *
        "of each channel: a number, or a map (sigma.nii.gz, or file.mat:NAME); EQUIP does not " *
        "estimate it"))
    s = strip(value(a, "sigma"))
    σ = MRImages._isnifti(s) || MRImages._ismat(s) ? readmap(s, info) : _number(s, "sigma")
    method = value(a, "method", "ml")
    method in ("ml", "mean", "wmean") || throw(ArgumentError("`--method`: ml (maximum " *
        "likelihood, the default), mean (least squares on the Rician mean) or wmean (the same, " *
        "weighted by 1/Var), got `$method`"))
    return (; likelihood = :rician, σ, method = Symbol(method))
end

# A field's value as a flag holds it: numbers as a comma list (shortest exact form), a Bool.
_valuestring(v::AbstractVector) = join(repr.(Float64.(v)), ',')
_valuestring(v::Real) = repr(v isa Bool ? v : Float64(v))

function _precision(s)
    s in ("Float64", "64", "double") && return Float64
    s in ("Float32", "32", "single") && return Float32
    throw(ArgumentError("`--precision`: Float64 or Float32, got `$s`"))
end
