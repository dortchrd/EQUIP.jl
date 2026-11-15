# =================================================================================================
# equip crlb and equip montecarlo: a fit's errors (bias, SD, RMSE), predicted from derivatives
# or measured by simulating and fitting, for a protocol, a truth (or several) and a fit
#
#   equip crlb QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I --snr 200 --kmf 12.5
#   equip crlb QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I,SIR_GM,SIR_lesion --snr 100,200
#   equip crlb T2Decay --TE 0.01:0.01:0.32 --truth S0=1000,T2=0.08 --sigma 10 --vary T2=0.04:0.02:0.12
#   equip montecarlo QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I --snr 200 --noise rician --kmf 12.5
#   equip crlb T1MFA --FA 2,5,10,15,20,30 --TR 0.015 --truth T1=1 --snr 1000 --B1 1.05 --uncertain B1=0.03
#
# The truth: `--tissue` (a preset, or several, comma-separated: one scenario each) and/or
# `--truth NAME=value,...` (values changed from the tissue's, or from the model's own); the fit:
# the parameters' flags, as for `equip fit` (no maps: one voxel); the noise: `--snr` (a list is a
# sweep) or `--sigma` (one, or one per observation), `--averages`; `--vary NAME=values` (true
# values; repeat for several parameters); `--uncertain NAME=SD,...` (a fixed value that was
# measured, a B1 map: its measurement's SD, in the parameter's units). The table goes to `--out`
# as TSV.
# =================================================================================================

const _ERROR_OPTIONS = ["model", "protocol", "tissue", "truth", "vary", "weights", "snr", "sigma",
                        "averages", "uncertain", "out", "save", "help"]

# `NAME=value,NAME=value` into a named tuple: numbers, or a parameter's name (a tie).
function _assignments(s::AbstractString, what, params)
    pairs_ = Pair{Symbol,Any}[]
    for item in split(s, ','; keepempty = false)
        k, eq, v = _split_eq(strip(item))
        eq || throw(ArgumentError("`--$what`: give NAME=value, e.g. `--$what PSR=0.15,R1f=1.2`; got `$item`"))
        Symbol(k) in params || throw(ArgumentError("`--$what`: `$k` is not a parameter; the " *
                                                   "parameters: $(join(params, ", "))"))
        x = tryparse(Float64, v)
        push!(pairs_, Symbol(k) => (x !== nothing ? x : Symbol(v) in params ? Symbol(v) :
            throw(ArgumentError("`--$what $k=$v`: a number, or another parameter's name (a tie)"))))
    end
    return (; pairs_...)
end

# `--vary NAME=values` (repeatable) into crlb's `vary`: true values, one axis per parameter.
function _vary(a::Args, params)
    axes = Pair{Symbol,Vector{Float64}}[]
    for (n, s) in a.flags
        n == "vary" || continue
        k, eq, v = _split_eq(s)
        eq || throw(ArgumentError("`--vary`: give NAME=values, e.g. `--vary kmf=5:2.5:20` or " *
                                  "`--vary Sm=0.78,0.88`; one parameter per --vary"))
        Symbol(k) in params || throw(ArgumentError("`--vary`: `$k` is not a parameter; the " *
                                                   "parameters: $(join(params, ", "))"))
        push!(axes, Symbol(k) => numbers(v, "vary $k"))
    end
    return (; axes...)
end

# `--uncertain NAME=SD,NAME=SD` into crlb's and montecarlo's `uncertain`: each a parameter the fit
# holds at a measured value (`--B1 1.05`), with that measurement's SD, a number (not `--truth`'s
# ties; EQUIP checks the rest: that the fit holds it at a number, that the SD is ≥ 0).
function _uncertainflag(a::Args, params)
    has(a, "uncertain") || return (;)
    sds = Pair{Symbol,Float64}[]
    for item in split(value(a, "uncertain"), ','; keepempty = false)
        k, eq, v = _split_eq(strip(item))
        eq || throw(ArgumentError("`--uncertain`: give NAME=SD, e.g. `--uncertain B1=0.03`; " *
                                  "got `$item`"))
        Symbol(k) in params || throw(ArgumentError("`--uncertain`: `$k` is not a parameter; the " *
                                                   "parameters: $(join(params, ", "))"))
        push!(sds, Symbol(k) => _number(v, "uncertain $k"))
    end
    return (; sds...)
end

# The noise level: --snr (one, or several: a sweep) or --sigma (one, or one per observation),
# and --averages.
function _noiselevel(a::Args)
    has(a, "snr") && has(a, "sigma") && throw(ArgumentError("give --snr or --sigma, not both"))
    lvl = if has(a, "snr")
        x = numbers(value(a, "snr"), "snr")
        x = all(isinteger, x) ? Int.(x) : x                 # 200, not 200.0, in the tables
        (; snr = length(x) == 1 ? x[1] : x)
    elseif has(a, "sigma")
        x = numbers(value(a, "sigma"), "sigma")
        (; σ = length(x) == 1 ? x[1] : x)
    else
        throw(ArgumentError("give the noise level: --snr (relative to the model's scale " *
                            "parameter, e.g. 200) or --sigma (in signal units)"))
    end
    has(a, "averages") || return lvl
    x = numbers(value(a, "averages"), "averages")
    return merge(lvl, (; averages = length(x) == 1 ? x[1] : x))
end

# The truths: one (a named tuple of every parameter's value: a single result), or several
# tissues, as a list of names for `scenarios`, with the values `--truth` changes as one-value
# axes of `vary` (the same change in every tissue, and a column in the tables). A list, not a
# named tuple of truths by tissue: those names would be part of its type, and each set of
# tissues would compile the commands again. `nominal`'s own message says what is missing; the
# hint says how to give it here.
function _truths(m, a::Args)
    params = paramnames(typeof(m))
    changes = has(a, "truth") ? _assignments(value(a, "truth"), "truth", params) : (;)
    one(t) = try
        t === nothing ? nominal(m; changes...) : nominal(m, t; changes...)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError(e.msg * "\n(on the command line: --truth NAME=value,..., or a tissue: --tissue NAME)"))
    end
    has(a, "tissue") || return one(nothing), false, (;)
    ts = Symbol.(split(value(a, "tissue"), ','; keepempty = false))
    length(ts) == 1 && return one(ts[1]), false, (;)        # one truth: a single result
    foreach(one, ts)                                        # each checked, with its message
    return ts, true, map(v -> [v], changes)
end

# The fit's settings: the parameters' flags (numbers, free, ties; no maps) and the weights.
function _fitflags(a::Args, M)
    params = paramnames(M)
    roles = (; (Symbol(n) => role(n, v, params, nothing) for (n, v) in lastflags(a) if Symbol(n) in params)...)
    weights = has(a, "weights") && !occursin('=', value(a, "weights")) ?
              numbers(value(a, "weights"), "weights") : nothing
    return roles, weights
end

# The results as a table: a sweep's rows, or one result's (no setting columns).
_table(r::Sweep) = r.rows
_table(r) = EQUIP._rows([(;)], [r])

# What both commands share: the model, the truths, the fit, the noise; then the command's own call.
function _errors_command(cmd, words, extra, run; io::IO, err::IO)
    a = parseargs(words)
    pos = copy(a.positional)
    what = cmd == "crlb" ? "a fit's bias, SD and RMSE, predicted (Cramér–Rao)" :
           "a fit's bias, SD and RMSE, measured by simulating and fitting"
    model = modelarg(a, pos, cmd, "MODEL [flags]: $what", io)
    model === nothing && return 0                # `equip crlb --help`: printed
    isempty(pos) || throw(ArgumentError("`equip $cmd` takes no files (`$(pos[1])`): it judges a " *
        "protocol on a truth; give the truth with --tissue or --truth"))
    M = modeltype(model)
    if has(a, "help")
        modelhelp(io, M, cmd)
        println(io, "\nThe truth: --tissue NAME (or several: NAME,NAME), --truth NAME=value,...; ",
                "the noise: --snr or --sigma, --averages; --vary NAME=values (true values); ",
                "--uncertain NAME=SD (a fixed value that was measured, e.g. --B1 1.05 --uncertain ",
                "B1=0.03: its SD); --weights; --out table.tsv; --config FILE, --save FILE", extra.help)
        return 0
    end
    _check_flags(a, M, [_ERROR_OPTIONS; extra.options], cmd)
    m = buildmodel(M, a, Dict{Symbol,Vector{Float64}}(); warn = err)
    roles, weights = _fitflags(a, M)
    truths, several, truthaxes = _truths(m, a)
    lvl = _noiselevel(a)
    vary = _vary(a, paramnames(M))
    several && (vary = merge(truthaxes, vary))              # --vary wins for a parameter in both
    if has(a, "save")
        acq = [String(k) => _valuestring(getfield(m, k)) for k in fieldnames(M)]
        flags = [["model" => model]; acq;
                 [f for f in lastflags(a) if !(first(f) in ("model", "save", "help", "protocol",
                                                          String.(fieldnames(M))...))]]
        saveconfig(value(a, "save"), flags, "equip $cmd, saved $(Dates.today()) with EQUIP " *
                   "$(pkgversion(EQUIP))\nrun again: equip $cmd --config $(value(a, "save"))")
    end
    res = run(M, m, a, truths, several, roles, weights, lvl, vary)
    show(io, MIME"text/plain"(), res)
    println(io)
    if has(a, "out")
        f = writetable(value(a, "out"), _table(res))
        println(io, "\nwrote ", f)
    end
    return 0
end

cmd_crlb(words; io::IO = stdout, err::IO = stderr) =
    _errors_command("crlb", words, (options = ["cov"], help = "; --cov sandwich or biased"),
                    _run_crlb; io, err)
function _run_crlb(M, m, a, truths, several, roles, weights, lvl, vary)
    cov = Symbol(value(a, "cov", "sandwich"))
    uncertain = _uncertainflag(a, paramnames(M))
    several || return crlb(m, truths; lvl..., cov, vary, uncertain, weights, roles...)   # (vary: true values)
    return crlb(m, scenarios(M, truths; vary, weights, roles...); lvl..., cov, uncertain)
end

cmd_montecarlo(words; io::IO = stdout, err::IO = stderr) =
    _errors_command("montecarlo", words,
                    (options = ["noise", "averaging", "n", "seed", "starts", "precision"],
                     help = "; --noise rician or gaussian (required); --averaging; --n copies " *
                            "(10000); --seed; --starts truth or model; --precision"),
                    _run_montecarlo; io, err)
function _run_montecarlo(M, m, a, truths, several, roles, weights, lvl, vary)
    has(a, "noise") || throw(ArgumentError("give the noise's distribution: --noise rician " *
        "(magnitude images) or --noise gaussian (real-valued data)"))
    noise = Symbol(value(a, "noise"))
    averaging = has(a, "averaging") ? Symbol(value(a, "averaging")) : nothing
    n = round(Int, _number(value(a, "n", "10000"), "n"))
    rng = has(a, "seed") ? Xoshiro(round(Int, _number(value(a, "seed"), "seed"))) : Xoshiro()
    starts = Symbol(value(a, "starts", "truth"))
    precision = _precision(value(a, "precision", "Float64"))
    uncertain = _uncertainflag(a, paramnames(M))
    several || return montecarlo(m, truths; lvl..., noise, averaging, n, rng, starts, precision,
                                 vary, uncertain, weights, roles...)
    return montecarlo(m, scenarios(M, truths; vary, weights, starts, roles...); lvl..., noise,
                      averaging, n, rng, precision, uncertain)
end
