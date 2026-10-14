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
  voxelfit fit --data FILE (--model SPEC | --expr EXPR --params NAMES) (--x LIST | --x-file FILE)
               --out PREFIX [options]
  voxelfit models             list built-in models
  voxelfit help               show this message

Data:
  --data FILE           .nii, .nii.gz or .npy; observations along the last axis by default
  --obs-axis last|first axis of --data that holds the observations (in the file's own axis order)
  --x LIST              comma-separated acquisition values (TE, b, TI, flip angle...), one per observation
  --x-file FILE         text file, one row per observation (several columns = several settings)
  --mask FILE           fit only voxels where the mask is non-zero
  --covariate FILE      per-voxel covariate map (repeatable), e.g. a relative B1 map for SPGR
  --weights LIST|FILE   per-observation weights

Model:
  --model SPEC          built-in model, e.g. MonoExp, 'SPGR(TR=15)', 'IVIM(bthresh=200)'
  --expr EXPR           custom model as a Julia expression, e.g. 'S0*exp(-TE/T2) + C'
  --params NAMES        parameter names for --expr, e.g. S0,T2,C
  --xname NAMES         name(s) of the acquisition variable(s) in --expr (default: x)
  --covariate-names N   names of the covariates in --expr
  --p0 LIST             initial guess for every voxel (default: model's data-driven guess)
  --lower LIST          lower bounds (use -inf for none)
  --upper LIST          upper bounds (use inf for none)

Solver:
  --precision single|double   working/output precision (default: single)
  --maxiter N (100)  --xtol X  --ftol X  --gtol X
  --stderr              also write standard-error maps (PREFIX_stderr_<param>)
  --backend NAME        cpu (default), cuda, amdgpu, metal or oneapi; the GPU package must be
                        installed in the active environment
  --chunksize N         voxels per CPU work item (128);  --batchsize N  voxels per GPU batch

Output:
  --out PREFIX          writes PREFIX_<param>, PREFIX_sse, PREFIX_iterations, PREFIX_status
  --format nii|nii.gz|npy   output format (default: same as --data)
  --quiet               no summary

Status codes: 0 not fitted (mask), 1-3 converged (step, cost, gradient), 4 max iterations,
              5 failed, 6 invalid (non-finite) data.
CPU threads are set when Julia starts: julia -t auto (bin/voxelfit does this).
"""

struct CLIError <: Exception
    msg::String
end
Base.showerror(io::IO, e::CLIError) = print(io, e.msg)
_clierr(msg) = throw(CLIError(msg))

const _FLAGS = Set(["--stderr", "--quiet", "--help", "-h"])
const _REPEATABLE = Set(["--covariate"])

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

function _read_table(path)
    isfile(path) || _clierr("file not found: $path")
    rows = [_parse_list(l) for l in eachline(path) if !isempty(strip(l)) && !startswith(strip(l), "#")]
    isempty(rows) && _clierr("no values in $path")
    length(rows) == 1 && return rows[1]
    all(length(r) == 1 for r in rows) && return [r[1] for r in rows]
    k = length(rows[1])
    all(length(r) == k for r in rows) || _clierr("rows of $path have different lengths")
    return permutedims(reduce(hcat, rows))   # nobs × k
end

_isnifti(p) = endswith(p, ".nii") || endswith(p, ".nii.gz")

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

# Bring an auxiliary image (mask/covariate) into the data's axis order.
function _match_order(A, rev, data_rev)
    rev == data_rev || ndims(A) <= 1 ? A : permutedims(A, ndims(A):-1:1)
end

function list_models(io::IO=stdout)
    println(io, "Built-in models (parameters):")
    for name in sort!(collect(keys(BUILTIN_MODELS)))
        ctor = BUILTIN_MODELS[name]
        m = name == "SPGR" ? ctor(TR=1.0) : ctor()
        println(io, "  ", rpad(name, 20), join(paramnames(m), ", "))
    end
    println(io, "\nSPGR needs TR, e.g. 'SPGR(TR=15)' (x = flip angles in degrees; optional B1 covariate).")
    println(io, "IVIM(bthresh=200) and InversionRecovery(magnitude=true) take optional settings.")
end

function _load_backend(name::AbstractString)
    name = lowercase(name)
    name == "cpu" && return nothing
    table = Dict("cuda" => (:CUDA, :CUDABackend), "amdgpu" => (:AMDGPU, :ROCBackend),
                 "metal" => (:Metal, :MetalBackend), "oneapi" => (:oneAPI, :oneAPIBackend))
    haskey(table, name) || _clierr("unknown backend '$name'")
    pkg, ctor = table[name]
    mod = try
        Base.require(Main, pkg)
    catch
        _clierr("backend '$name' needs the $pkg package in the active Julia environment")
    end
    return Base.invokelatest(getfield, mod, ctor)()
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

    # Model
    model = if haskey(o, "--model")
        try
            parse_model(o["--model"])
        catch e
            _clierr(sprint(showerror, e))
        end
    elseif haskey(o, "--expr")
        haskey(o, "--params") || _clierr("--expr needs --params")
        xn = _parse_names(get(o, "--xname", "x"))
        expr_model(o["--expr"]; params=_parse_names(o["--params"]),
                   x=length(xn) == 1 ? xn[1] : Tuple(xn),
                   covariates=Tuple(_parse_names(get(o, "--covariate-names", ""))))
    else
        _clierr("give --model or --expr")
    end

    # Acquisition values
    x = haskey(o, "--x") ? _parse_list(o["--x"]) :
        haskey(o, "--x-file") ? _read_table(o["--x-file"]) : _clierr("give --x or --x-file")

    data, rev, hdr = load_image(o["--data"])
    axis = get(o, "--obs-axis", "last")
    axis in ("last", "first") || _clierr("--obs-axis must be 'last' or 'first'")
    obsdim = (axis == "last") == !rev ? ndims(data) : 1
    nobs = size(data, obsdim)
    size(x, 1) == nobs || _clierr("--x has $(size(x, 1)) values but the data have $nobs observations along the $axis axis")

    mask = nothing
    if haskey(o, "--mask")
        M, mrev, _ = load_image(o["--mask"])
        mask = _match_order(M, mrev, rev)
    end
    covariates = nothing
    if haskey(o, "--covariate")
        maps = [(C = load_image(p); _match_order(C[1], C[2], rev)) for p in o["--covariate"]]
        covariates = length(maps) == 1 ? maps[1] : cat(maps...; dims=ndims(maps[1]) + 1)
    end
    weights = !haskey(o, "--weights") ? nothing :
              isfile(o["--weights"]) ? vec(_read_table(o["--weights"])) : _parse_list(o["--weights"])
    init = haskey(o, "--p0") ? _parse_list(o["--p0"]) : nothing
    lower = haskey(o, "--lower") ? _parse_list(o["--lower"]) : nothing
    upper = haskey(o, "--upper") ? _parse_list(o["--upper"]) : nothing
    backend = _load_backend(get(o, "--backend", "cpu"))

    # Outputs
    prefix = o["--out"]
    mkpath(dirname(abspath(prefix)))
    fmt = get(o, "--format", endswith(o["--data"], ".npy") ? "npy" :
                             endswith(o["--data"], ".nii") ? "nii" : "nii.gz")
    fmt in ("npy", "nii", "nii.gz") || _clierr("--format must be npy, nii or nii.gz")
    names = collect(paramnames(model))
    spatial = _spatial_size(size(data), obsdim)
    want_se = get(o, "--stderr", false)
    files = String[]
    fname(tag) = (f = "$(prefix)_$(tag).$(fmt)"; push!(files, f); f)
    res = MapResult{T}(names, spatial; stderr=want_se)

    nfit = mask === nothing ? prod(spatial) : count(!iszero, mask)
    quiet || println("VoxelFit: $(typeof(model)) $(Tuple(names)), $(join(size(data), "×")) data, ",
                     "$nobs observations, $nfit voxels, $T, ",
                     backend === nothing ? "$(Threads.nthreads()) CPU threads" : string(typeof(backend)))
    t0 = time()
    Base.invokelatest(fit_maps!, res, model, x, data; obsdim, mask, init, lower, upper,
                      covariates, weights, backend,
                      maxiter=parse(Int, get(o, "--maxiter", "100")),
                      xtol=haskey(o, "--xtol") ? _parse_number(o["--xtol"]) : nothing,
                      ftol=haskey(o, "--ftol") ? _parse_number(o["--ftol"]) : nothing,
                      gtol=haskey(o, "--gtol") ? _parse_number(o["--gtol"]) : nothing,
                      chunksize=parse(Int, get(o, "--chunksize", "128")),
                      batchsize=parse(Int, get(o, "--batchsize", string(2^18))))
    elapsed = time() - t0

    # Write maps
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
# Helpers for foreign callers (Python via juliacall, C via @ccallable builds).

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
    fit_into!(model, x, data, params, sse, iterations, status, stderr; kwargs...)

Fit observation-major `data` (nobs × spatial...) into caller-provided output arrays; used by the
Python front end with arrays from [`wrap_array`](@ref). Keywords as in [`fit_maps`](@ref).
"""
function fit_into!(model::AbstractModel, x, data::AbstractArray, params::Array{T}, sse::Array{T},
                   iterations::Array{Int32}, status::Array{UInt8}, stderr; kwargs...) where {T}
    res = MapResult(collect(Symbol, paramnames(model)), params, sse, iterations, status, stderr)
    fit_maps!(res, model, x, data; obsdim=1, kwargs...)
    return nothing
end
