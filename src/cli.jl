# Command-line interface and helpers for other languages.
#
#   julia -t auto --project=/path/to/VoxelFit.jl -m VoxelFit fit --model MonoExp --x 10,20,30 \
#         --data echoes.nii.gz --mask brain.nii.gz --out maps/sub01
#
# (or bin/voxelfit, which adds the flags). NIfTI in → NIfTI out (header and affine copied);
# .npy in → .npy out with the same memory order. Uncompressed inputs are memory-mapped.

using NIfTI: NIfTI, niread, niwrite, NIVolume

const CLI_HELP = """
voxelfit: voxel-wise nonlinear least-squares fitting of MRI data (VoxelFit.jl)

Usage:
  voxelfit fit --data FILE (--model SPEC | --expr EXPR --params NAMES) PROTOCOL --out PREFIX [options]
  voxelfit models             list built-in models with their protocol and parameters
  voxelfit help               show this message

Data:
  --data FILE           .nii, .nii.gz or .npy; observations along the last axis by default
  --obs-axis last|first axis of --data that holds the observations (in the file's own axis order)
  --mask FILE           fit only voxels where the mask is non-zero

Protocol (acquisition settings; vectors are per observation, single values constant):
  --x LIST              values of a one-setting protocol, e.g. echo times: --x 10,20,30
  --x-file FILE         the same from a text file (one value per line)
  --acq NAME=LIST       one protocol setting (repeatable): --acq TI=0.01,0.1,1 --acq TD=2.5
  --acq-file FILE       text table, header line with the names, one row per observation

Model:
  --model SPEC          built-in model, e.g. MonoExp, SIRFSE, 'SIRFSE(magnitude=false)', 'IVIM(bthresh=200)'
  --expr EXPR           custom model for one observation, e.g. 'S0*exp(-TE/T2) + C'
  --params NAMES        parameter names for --expr, e.g. S0,T2,C
  --protocol NAMES      names of the acquisition settings in --expr (default: x)
  --p0, --lower, --upper LIST   starting values and bounds for --expr parameters

Parameters:
  --fixed NAME=VALUE|FILE   hold a parameter fixed at a number or a per-voxel map (repeatable),
                            e.g. --fixed kmf=12 --fixed B1=b1.nii.gz
  --free NAMES              also estimate parameters the model fixes by default, e.g. --free kmf
  --start NAME=VALUE|FILE   starting value or map for an estimated parameter (repeatable)
  --bounds NAME=LO,HI       bounds for an estimated parameter (repeatable; -inf/inf for none)

Solver:
  --precision single|double   working/output precision (default: single)
  --maxiter N (100)  --xtol X  --ftol X  --gtol X
  --weights LIST|FILE   per-observation weights
  --stderr              also write standard-error maps (PREFIX_stderr_<param>)
  --chunksize N         voxels per CPU work item (128)

Output:
  --out PREFIX          writes PREFIX_<param>, PREFIX_sse, PREFIX_iterations, PREFIX_status
  --format nii|nii.gz|npy   output format (default: same as --data)
  --quiet               no summary

Status codes: 0 not fitted (mask), 1-3 converged (step, cost, gradient), 4 max iterations,
              5 failed, 6 invalid (non-finite data or fixed-map value).
CPU threads are set when Julia starts: julia -t auto (bin/voxelfit does this).
"""

struct CLIError <: Exception
    msg::String
end
Base.showerror(io::IO, e::CLIError) = print(io, e.msg)
_clierr(msg) = throw(CLIError(msg))

const _FLAGS = Set(["--stderr", "--quiet", "--help", "-h"])
const _REPEATABLE = Set(["--acq", "--fixed", "--start", "--bounds"])

function _parse_args(args)
    opts = Dict{String,Any}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "-") || _clierr("unexpected argument '$a'")
        if a in _FLAGS
            opts[a] = true
            i += 1
            continue
        end
        i + 1 <= length(args) || _clierr("option $a needs a value")
        v = args[i+1]
        if a in _REPEATABLE
            push!(get!(opts, a, String[]), v)
        else
            opts[a] = v
        end
        i += 2
    end
    return opts
end

_parse_number(s) = (t = lowercase(strip(s));
                    t in ("inf", "+inf") ? Inf : t == "-inf" ? -Inf : parse(Float64, t))
_parse_list(s::AbstractString) = [_parse_number(t) for t in split(s, r"[,\s]+") if !isempty(t)]
_parse_names(s::AbstractString) = [Symbol(strip(t)) for t in split(s, ',') if !isempty(strip(t))]

function _name_value(s)
    k = findfirst('=', s)
    k === nothing && _clierr("expected NAME=VALUE, got '$s'")
    return Symbol(strip(s[1:k-1])), strip(s[k+1:end])
end

function _read_lines(path)
    isfile(path) || _clierr("file not found: $path")
    return [strip(l) for l in eachline(path) if !isempty(strip(l)) && !startswith(strip(l), "#")]
end

function _read_column(path)
    rows = [_parse_list(l) for l in _read_lines(path)]
    isempty(rows) && _clierr("no values in $path")
    length(rows) == 1 && return rows[1]
    all(length(r) == 1 for r in rows) ||
        _clierr("$path has several columns; use --acq-file with a header line naming them")
    return [r[1] for r in rows]
end

function _read_acq_table(path)
    lines = _read_lines(path)
    length(lines) >= 2 || _clierr("$path needs a header line and at least one row")
    names = Symbol.(split(lines[1], r"[,\s]+"; keepempty=false))
    rows = [_parse_list(l) for l in lines[2:end]]
    all(length(r) == length(names) for r in rows) || _clierr("rows of $path do not match its header")
    return NamedTuple{Tuple(names)}(Tuple([r[k] for r in rows] for k in eachindex(names)))
end

_isnifti(p) = endswith(p, ".nii") || endswith(p, ".nii.gz")
_isimage(p) = _isnifti(p) || endswith(p, ".npy")

# Load an image. Returns (array, reversed, nifti_header_or_nothing).
function load_image(path::AbstractString)
    isfile(path) || _clierr("file not found: $path")
    if endswith(path, ".npy")
        A, rev = read_npy(path)
        return A, rev, nothing
    elseif _isnifti(path)
        vol = niread(path; mmap=endswith(path, ".nii"))
        h = vol.header
        A = vol.raw
        if h.scl_slope != 0 && (h.scl_slope != 1 || h.scl_inter != 0)
            A = Float32.(A) .* Float32(h.scl_slope) .+ Float32(h.scl_inter)
        end
        return A, false, h
    end
    _clierr("unsupported file type: $path (use .nii, .nii.gz or .npy)")
end

# Bring an auxiliary image (mask/map) into the data's axis order.
_match_order(A, rev, data_rev) = rev == data_rev || ndims(A) <= 1 ? A : permutedims(A, ndims(A):-1:1)

function _load_map(path, data_rev)
    A, rev, _ = load_image(path)
    return _match_order(A, rev, data_rev)
end

function _value_or_map(s, data_rev)
    _isimage(s) ? _load_map(s, data_rev) : _parse_number(s)
end

_roledesc(r::Estimate) = r.init === :auto ? "estimated" : "estimated (start $(r.init))"
_roledesc(r::Fixed{<:Real}) = "fixed = $(r.value)"
_roledesc(r::Fixed{<:Function}) = "fixed (tied)"
_roledesc(r::Fixed) = "fixed"

function list_models(io::IO=stdout)
    println(io, "Built-in models:")
    for name in sort!(collect(keys(BUILTIN_MODELS)))
        m = BUILTIN_MODELS[name]()
        println(io, "\n  ", name, "    protocol: ", join(protocol(m), ", "))
        for (k, r) in pairs(parameters(m))
            println(io, "      ", rpad(k, 6), _roledesc(r))
        end
    end
    println(io, "\nOptions: 'InversionRecovery(magnitude=false)', 'SIRFSE(magnitude=false)', 'IVIM(bthresh=200)'.")
end

function _status_summary(io, status, nfit, elapsed)
    counts = zeros(Int, 7)
    for s in status
        counts[s+1] += 1
    end
    println(io, "fitted $nfit voxels in $(round(elapsed; digits=2)) s ",
            "($(round(1e6 * elapsed / max(nfit, 1); digits=2)) µs/voxel, $(Threads.nthreads()) threads)")
    for (k, c) in enumerate(counts)
        (k == 1 || c == 0) && continue
        println(io, "  ", rpad(string(FitStatus(k - 1)), 14), c)
    end
end

function _cli_model(o)
    if haskey(o, "--model")
        try
            return parse_model(o["--model"])
        catch e
            _clierr(sprint(showerror, e))
        end
    elseif haskey(o, "--expr")
        haskey(o, "--params") || _clierr("--expr needs --params")
        get_list(k) = haskey(o, k) ? _parse_list(o[k]) : nothing
        return expr_model(o["--expr"]; params=_parse_names(o["--params"]),
                          protocol=Tuple(_parse_names(get(o, "--protocol", "x"))),
                          p0=get_list("--p0"), lower=get_list("--lower"), upper=get_list("--upper"))
    end
    _clierr("give --model or --expr")
end

function _cli_protocol(o)
    acq = haskey(o, "--x") ? _parse_list(o["--x"]) :
          haskey(o, "--x-file") ? _read_column(o["--x-file"]) : nothing
    parts = Pair{Symbol,Any}[]
    haskey(o, "--acq-file") && append!(parts, collect(pairs(_read_acq_table(o["--acq-file"]))))
    for s in get(o, "--acq", String[])
        n, v = _name_value(s)
        vals = _parse_list(v)
        push!(parts, n => (length(vals) == 1 ? vals[1] : vals))
    end
    if !isempty(parts)
        acq === nothing || _clierr("give either --x/--x-file or --acq/--acq-file, not both")
        acq = NamedTuple(parts)
    end
    acq === nothing && _clierr("give the protocol with --x, --x-file, --acq or --acq-file")
    return acq
end

function cli_fit(args)
    o = _parse_args(args)
    if haskey(o, "--help") || haskey(o, "-h")
        print(CLI_HELP)
        return 0
    end
    for req in ("--data", "--out")
        haskey(o, req) || _clierr("missing required option $req")
    end
    quiet = get(o, "--quiet", false)
    T = get(o, "--precision", "single") in ("single", "float32", "Float32") ? Float32 : Float64

    model = _cli_model(o)
    acq = _cli_protocol(o)
    data, rev, hdr = load_image(o["--data"])
    axis = get(o, "--obs-axis", "last")
    axis in ("last", "first") || _clierr("--obs-axis must be 'last' or 'first'")
    obsdim = (axis == "last") == !rev ? ndims(data) : 1
    m = size(data, obsdim)

    mask = haskey(o, "--mask") ? _load_map(o["--mask"], rev) : nothing
    fixed = NamedTuple(Pair{Symbol,Any}[(n = _name_value(s); n[1] => _value_or_map(n[2], rev))
                                        for s in get(o, "--fixed", String[])])
    start = NamedTuple(Pair{Symbol,Any}[(n = _name_value(s); n[1] => _value_or_map(n[2], rev))
                                        for s in get(o, "--start", String[])])
    free = Any[n for n in _parse_names(get(o, "--free", ""))]
    roles = parameters(model)
    for s in get(o, "--bounds", String[])
        n, v = _name_value(s)
        haskey(roles, n) || _clierr("unknown parameter `$n` in --bounds; parameters are $(keys(roles))")
        lohi = _parse_list(v)
        length(lohi) == 2 || _clierr("--bounds $n needs LO,HI")
        old = roles[n]
        init0 = old isa Estimate ? old.init : (old.value isa Real ? Float64(old.value) : :auto)
        filter!(f -> f !== n, free)
        push!(free, n => Estimate(init0; lower=lohi[1], upper=lohi[2]))
    end
    weights = !haskey(o, "--weights") ? nothing :
              isfile(o["--weights"]) ? _read_column(o["--weights"]) : _parse_list(o["--weights"])

    # Outputs
    prefix = o["--out"]
    mkpath(dirname(abspath(prefix)))
    fmt = get(o, "--format", endswith(o["--data"], ".npy") ? "npy" :
                             endswith(o["--data"], ".nii") ? "nii" : "nii.gz")
    fmt in ("npy", "nii", "nii.gz") || _clierr("--format must be npy, nii or nii.gz")
    spatial = _spatial_size(size(data), obsdim)
    spec = try
        setup(model, T; fixed, free=Tuple(free), start)
    catch e
        e isa ArgumentError ? _clierr(e.msg) : rethrow()
    end
    names = collect(freenames(spec))
    res = MapResult{T}(names, spatial; stderr=get(o, "--stderr", false))
    files = String[]
    fname(tag) = (f = "$(prefix)_$(tag).$(fmt)"; push!(files, f); f)

    nfit = mask === nothing ? prod(spatial) : count(!iszero, mask)
    quiet || println("VoxelFit: $(model), estimating $(Tuple(names)), fixed $(keys(spec.fixed)); ",
                     "$(join(size(data), "×")) data, $m observations, $nfit voxels, $T, ",
                     "$(Threads.nthreads()) threads")
    t0 = time()
    try
        Base.invokelatest(fit_maps!, res, model, acq, data; obsdim, mask, fixed, free=Tuple(free),
                          start, weights,
                          maxiter=parse(Int, get(o, "--maxiter", "100")),
                          xtol=haskey(o, "--xtol") ? _parse_number(o["--xtol"]) : nothing,
                          ftol=haskey(o, "--ftol") ? _parse_number(o["--ftol"]) : nothing,
                          gtol=haskey(o, "--gtol") ? _parse_number(o["--gtol"]) : nothing,
                          chunksize=parse(Int, get(o, "--chunksize", "128")))
    catch e
        e isa Union{ArgumentError,DimensionMismatch} ? _clierr(sprint(showerror, e)) : rethrow()
    end
    elapsed = time() - t0

    for (j, nm) in enumerate(names)
        _write_map(fname(string(nm)), selectdim(res.params, length(spatial) + 1, j), fmt, rev, hdr)
        if res.stderr !== nothing
            _write_map(fname("stderr_$(nm)"), selectdim(res.stderr, length(spatial) + 1, j), fmt, rev, hdr)
        end
    end
    _write_map(fname("sse"), res.sse, fmt, rev, hdr)
    _write_map(fname("iterations"), res.iterations, fmt, rev, hdr)
    _write_map(fname("status"), res.status, fmt, rev, hdr)
    if fmt == "npy"   # parameter order, for programs reading the outputs back
        write(push!(files, "$(prefix)_params.txt")[end], join(string.(names), "\n"), "\n")
    end
    if !quiet
        _status_summary(stdout, res.status, nfit, elapsed)
        println("wrote ", join(files, ", "))
    end
    return 0
end

function _write_map(path, A, fmt, rev, hdr)
    if fmt == "npy"
        write_npy(path, A; reversed=rev)
    else
        h = hdr === nothing ? NIVolume(collect(A)).header : deepcopy(hdr)
        h.scl_slope = 1
        h.scl_inter = 0
        h.cal_min = 0
        h.cal_max = 0
        niwrite(path, NIVolume(h, collect(A)))
    end
end

"""
    main(args) -> exit code

Entry point of the command-line tool (`julia -m VoxelFit ...` on Julia ≥ 1.12, or
`bin/voxelfit`). Run with `help` for usage.
"""
function main(args::AbstractVector{<:AbstractString}=ARGS)
    try
        if isempty(args) || args[1] in ("help", "-h", "--help")
            print(CLI_HELP)
            return 0
        elseif args[1] == "models"
            list_models(stdout)
            return 0
        elseif args[1] == "fit"
            return cli_fit(args[2:end])
        else
            _clierr("unknown command '$(args[1])'; try 'voxelfit help'")
        end
    catch e
        e isa CLIError || rethrow()
        println(stderr, "error: ", e.msg)
        return 1
    end
end

@static if isdefined(Base, Symbol("@main"))
    @main
end

# --------------------------------------------------------------------------------------------
# Helpers for foreign callers (Python via juliacall).

"""
    wrap_array(addr, dtype, shape) -> Array

Wrap foreign C-ordered memory (e.g. a contiguous NumPy array) as a Julia `Array` of reversed
shape, without copying. `dtype` is a NumPy type string such as `"<f4"`. The caller must keep the
buffer alive while the array is in use.
"""
function wrap_array(addr::Integer, dtype::AbstractString, shape)
    code = lstrip(dtype, ['<', '>', '|', '='])
    haskey(_NPY_DTYPES, code) || throw(ArgumentError("unsupported dtype $dtype"))
    startswith(dtype, ">") && throw(ArgumentError("big-endian arrays are not supported"))
    T = _NPY_DTYPES[code]
    dims = Tuple(Int(s) for s in reverse(collect(shape)))
    return unsafe_wrap(Array, Ptr{T}(UInt(addr)), dims; own=false)
end

"""
    named(; kw...) -> NamedTuple

Build a named tuple from keyword arguments (lets Python pass dicts as `named(**d)`).
"""
named(; kw...) = NamedTuple(kw)

"""
    fit_into!(model, acq, data, params, sse, iterations, status, stderr; kwargs...)

Fit observation-major `data` (nobs × spatial...) into caller-provided output arrays; used by the
Python front end with arrays from [`wrap_array`](@ref). Keywords as in [`fit_maps`](@ref).
"""
function fit_into!(model::AbstractModel, acq, data::AbstractArray, params::Array{T}, sse::Array{T},
                   iterations::Array{Int32}, status::Array{UInt8}, stderr; free=(), kwargs...) where {T}
    free = Tuple(f isa AbstractString ? Symbol(f) : f for f in free)
    names = freenames(setup(model, T; fixed=get(kwargs, :fixed, (;)), free))
    res = MapResult(collect(Symbol, names), params, sse, iterations, status, stderr)
    fit_maps!(res, model, acq, data; obsdim=1, free, kwargs...)
    return nothing
end

"""
    free_names(model; fixed = (;), free = ()) -> Vector{String}

Names of the parameters a fit would estimate (for front ends that allocate outputs).
"""
free_names(model::AbstractModel; fixed=(;), free=()) =
    [string(n) for n in freenames(setup(model, Float64; fixed,
                                        free=Tuple(f isa AbstractString ? Symbol(f) : f for f in free)))]
