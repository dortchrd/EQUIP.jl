# =================================================================================================
# equip design MODEL [flags]: a protocol optimized for a set of scenarios (Eq. 5 of Dortch et al.,
# MRM 2018, by default), starting from one given
#
#   equip design QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I,SIR_WM_II,SIR_GM,SIR_lesion \
#       --snr 200 --kmf 12.5 --acquisition TI=0.01..10,TD=0.01..10 --readout 0.3 --out sir4.cnf
#   equip fit --config sir4.cnf sir.nii.gz                 # fit data acquired with it
#
# The scenarios as for `equip crlb` (tissues, --truth, --vary for a box of true values, the fit's
# flags); the acquisition to optimize with its bounds (`NAME=lower..upper`, `..` so a bound is
# not read as a range), `--shared TD` for one value for every point; the scan time by
# `--readout` (SIR-FSE: Σ(TI + TD + readout)) or `--scantime`, a Julia expression of the model's
# fields (`"sum(TI .+ TD .+ 0.3)"`, run as given); `--constraint` likewise, its values ≥ 0.
# Parameter weights (Eq. 5's Wⱼ) are `--weights PSR=1,R1f=0`; numbers (`--weights 1,2,4`) are the
# fit's weights per observation, as elsewhere.
# =================================================================================================

const _DESIGN_OPTIONS = ["model", "protocol", "tissue", "truth", "vary", "weights", "snr", "sigma",
                         "averages", "acquisition", "shared", "readout", "scantime", "constraint",
                         "budget", "aggregate", "cov", "starts", "generations", "seed", "out",
                         "save", "help"]

function cmd_design(words; io::IO = stdout, err::IO = stderr)
    a = parseargs(words)
    pos = copy(a.positional)
    model = modelarg(a, pos, "design", "MODEL [flags]: a protocol optimized for a set of scenarios", io)
    model === nothing && return 0                # `equip design --help`: printed
    isempty(pos) || throw(ArgumentError("`equip design` takes no files (`$(pos[1])`)"))
    M = modeltype(model)
    if has(a, "help")
        modelhelp(io, M, "design")
        println(io, """

        The starting protocol: the acquisition flags or --protocol (it sets the number of points).
        The scenarios: --tissue NAME,NAME,..., --truth NAME=value,..., --vary NAME=values, the fit's flags.
        The noise: --snr or --sigma, --averages. To optimize: --acquisition NAME=lower..upper,...,
        --shared NAME. The scan time: --readout SECONDS (Σ(TI + TD + readout)) or --scantime EXPR
        (the model's fields by name: "sum(TI .+ TD .+ 0.3)"); --constraint EXPR (values ≥ 0).
        Eq. 5: --weights PSR=1,R1f=0 (per parameter), --budget SECONDS, --aggregate max or mean,
        --cov sandwich or biased. The search: --starts N, --generations N, --seed N.
        --out FILE: the optimized protocol as a config file (equip fit --config FILE data.nii.gz).""")
        return 0
    end
    _check_flags(a, M, _DESIGN_OPTIONS, "design")
    m = buildmodel(M, a, Dict{Symbol,Vector{Float64}}(); warn = err)
    params = paramnames(M)
    roles, fitweights = _fitflags(a, M)
    pweights = has(a, "weights") && occursin('=', value(a, "weights")) ?
               _assignments(value(a, "weights"), "weights", params) : (;)
    truths, several, truthaxes = _truths(m, a)
    vary = several ? merge(truthaxes, _vary(a, params)) : _vary(a, params)
    set = scenarios(M, several ? truths : [truths]; vary, weights = fitweights, roles...)
    lvl = _noiselevel(a)
    haskey(lvl, :snr) && lvl.snr isa AbstractVector &&
        throw(ArgumentError("`equip design` takes one noise level (--snr 200)"))
    has(a, "acquisition") || throw(ArgumentError("give what to optimize, with its bounds: " *
        "--acquisition $(join(("$k=lower..upper" for k in fieldnames(M) if fieldtype(M, k) !== Bool), ","))"))
    acquisition = _bounds(value(a, "acquisition"), M)
    shared = has(a, "shared") ? Tuple(Symbol.(split(value(a, "shared"), ','))) : ()
    scantime = _scantime(a, M)
    constraints = has(a, "constraint") ? _expression(M, value(a, "constraint"), "constraint") : nothing
    budget = has(a, "budget") ? _number(value(a, "budget"), "budget") : nothing
    agg = value(a, "aggregate", "max")
    aggregate = agg in ("max", "maximum") ? maximum : agg == "mean" ? (v -> sum(v) / length(v)) :
        throw(ArgumentError("`--aggregate`: max or mean, got `$agg`"))
    cov = Symbol(value(a, "cov", "sandwich"))
    if has(a, "save")
        acq = [String(k) => _valuestring(getfield(m, k)) for k in fieldnames(M)]
        flags = [["model" => model]; acq;
                 [f for f in lastflags(a) if !(first(f) in ("model", "save", "help", "protocol",
                                                          String.(fieldnames(M))...))]]
        saveconfig(value(a, "save"), flags, "equip design, saved $(Dates.today()) with EQUIP " *
                   "$(pkgversion(EQUIP))\nrun again: equip design --config $(value(a, "save"))")
    end

    d = ProtocolDesign(m, set; lvl..., acquisition, scantime, constraints, shared,
                       weights = pweights, budget, aggregate, cov)
    start = d(m)
    starts = round(Int, _number(value(a, "starts", string(2 * Threads.nthreads())), "starts"))
    generations = round(Int, _number(value(a, "generations", "1000"), "generations"))
    rng = has(a, "seed") ? Xoshiro(round(Int, _number(value(a, "seed"), "seed"))) : Xoshiro()
    res = design(d; starts, generations, rng)
    show(io, MIME"text/plain"(), res)
    println(io, "\n  start:    ", join(("$k = $(getfield(m, k))" for k in keys(acquisition)), ", "),
            ": value ", round(start; sigdigits = 6))
    println(io, "  the rounded protocol's value is ", round(100 * (1 - res.rounded_value / start); digits = 2),
            "% below the start's")
    if has(a, "out")
        flags = [["model" => model]; [String(k) => _valuestring(getfield(res.rounded, k)) for k in fieldnames(M)]]
        saveconfig(value(a, "out"), flags, "equip design: $model optimized, Eq. 5 value " *
                   "$(round(res.rounded_value; sigdigits = 6)) (start $(round(start; sigdigits = 6)))\n" *
                   "fit data acquired with it: equip fit --config $(value(a, "out")) DATA.nii.gz")
        println(io, "\nwrote ", value(a, "out"))
    end
    return 0
end

# `NAME=lower..upper,...`: the acquisition fields to optimize, with their bounds.
function _bounds(s, M)
    out = Pair{Symbol,Tuple{Float64,Float64}}[]
    for item in split(s, ','; keepempty = false)
        k, eq, v = _split_eq(strip(item))
        r = split(v, "..")
        (eq && length(r) == 2) || throw(ArgumentError("`--acquisition`: give NAME=lower..upper, " *
                                                     "e.g. TI=0.01..10; got `$item`"))
        Symbol(k) in fieldnames(M) || throw(ArgumentError("`--acquisition`: `$k` is not a field " *
            "of $(nameof(M)) ($(join(fieldnames(M), ", ")))"))
        push!(out, Symbol(k) => (_number(r[1], "acquisition"), _number(r[2], "acquisition")))
    end
    return (; out...)
end

# The scan time: --readout (SIR-FSE: Σ(TI + TD + readout)) or a Julia expression of the fields.
function _scantime(a, M)
    has(a, "scantime") && return _expression(M, value(a, "scantime"), "scantime")
    if has(a, "readout")
        (:TI in fieldnames(M) && :TD in fieldnames(M)) || throw(ArgumentError("`--readout` is " *
            "for a protocol of TI and TD (Σ(TI + TD + readout)); give --scantime EXPR for $(nameof(M))"))
        r = _number(value(a, "readout"), "readout")
        return m -> sum(m.TI .+ m.TD .+ r)
    end
    throw(ArgumentError("give the scan time: --readout SECONDS (Σ(TI + TD + readout), SIR-FSE) " *
        "or --scantime EXPR, a Julia expression of $(nameof(M))'s fields " *
        "($(join(fieldnames(M), ", "))), e.g. \"sum(TI .+ TD .+ 0.3)\""))
end

# A Julia expression of the model's fields, as a function of the protocol (run as given: the
# user's own command). Made at run time, so called through `invokelatest`.
function _expression(M, s, what)
    ex = Meta.parse(s; raise = false)
    ex isa Expr && ex.head in (:error, :incomplete) &&
        throw(ArgumentError("`--$what`: `$s` is not a Julia expression"))
    lets = [:($k = getfield(m, $(QuoteNode(k)))) for k in fieldnames(M)]
    f = Core.eval(@__MODULE__, :(m -> let $(lets...); $ex; end))
    return m -> Base.invokelatest(f, m)
end
