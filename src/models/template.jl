# =================================================================================================
# newmodel: a model file to edit, written for you
#
#   newmodel(:StretchedDecay; acquisition = [:TE], parameters = [:S0, :T2, :beta])
#                                    a blank model: the struct, parameters and signal! with TODOs
#   newmodel(StretchedDecay)         a model made with @mrimodel, written out as plain code (the
#                                    code the macro generates, readable), to edit and extend
#
# The file is the hand-written form of "Writing a model" (README): a header block, a docstring,
# the struct, `parameters`, the signal of one observation as a function of its own, `signal!`
# looping over the observations, the optional hooks, and a first check, commented out. It is
# for models that will grow: hooks (`datastart`, `relabel`, `derived`, `fromderived`), work moved out of the
# loop, a built-in model's conventions (src/models/). A model made with @mrimodel keeps its definition
# (`_mrimodel_source`), so it can be written out when it outgrows the one-line form.
# =================================================================================================

"""
    newmodel(name; acquisition, parameters, options = (;), file = "name.jl", force = false)
    newmodel(M; file, force = false)

Write a model's file, to edit: a header block, a docstring, the struct, [`parameters`](@ref),
the signal of one observation as a function of its own, [`signal!`](@ref) looping over the
observations, the optional hooks ([`datastart`](@ref), [`relabel`](@ref), [`derived`](@ref), [`fromderived`](@ref))
and a first check, commented out. Returns the file's path; an existing file is kept unless `force = true`.

- **A blank model**: its `name`, the `acquisition`'s names (`[:TE]`, `[:TI, :TD]`), the
  `parameters` (names, `[:S0, :T2]`, or roles, `(S0 = Free(start = :scale, lower = 0),
  T2 = Free(start = 0.05))`), and `options` with their defaults (`(magnitude = true,)`). The
  signal is a TODO (it returns 0 until written).
- **A model made with [`@mrimodel`](@ref)** (`M`, its type): the code the macro generates,
  written out as plain Julia, the body as the signal of one observation, so the model can grow
  past the one-line form. Loaded with `include`, it gives the same fits.

```julia
newmodel(:StretchedDecay; acquisition = [:TE], parameters = [:S0, :T2, :beta])
newmodel(MySIR; file = "src/models/qmt/my_sir.jl")
```
"""
function newmodel(name::Union{Symbol,AbstractString}; acquisition, parameters,
                  options::NamedTuple = (;), file::AbstractString = "$(name).jl",
                  force::Bool = false)
    n = Symbol(name)
    Base.isidentifier(n) || throw(ArgumentError("newmodel: `$name` is not a Julia name"))
    acq = Symbol.(collect(acquisition))
    isempty(acq) && throw(ArgumentError("newmodel: give the acquisition's names, " *
                                        "`acquisition = [:TE]`"))
    opts = Pair{Symbol,String}[k => repr(v) for (k, v) in pairs(options)]
    optvals = Dict{Symbol,Any}(k => v for (k, v) in pairs(options))
    roles = if parameters isa NamedTuple
        Pair{Symbol,String}[k => _roletext(r) for (k, r) in pairs(parameters)]
    else
        Pair{Symbol,String}[Symbol(k) => "Free(),                    # TODO: start, bounds" for k in parameters]
    end
    isempty(roles) && throw(ArgumentError("newmodel: give the parameters, `parameters = [:S0, :T2]`"))
    names = [acq; first.(opts); first.(roles)]
    dup = [x for x in unique(names) if count(==(x), names) > 1]
    isempty(dup) || throw(ArgumentError("newmodel: `$(dup[1])` is used twice"))
    pnames = first.(roles)
    body = "return zero($(pnames[1]))           # TODO: the signal of one observation"
    return _writemodel(file, force, n, acq, opts, optvals, roles, body, nothing, false)
end
newmodel(m::MRIModel; kwargs...) = newmodel(typeof(m); kwargs...)
function newmodel(M::Type{<:MRIModel}; file::AbstractString = "$(nameof(M)).jl", force::Bool = false)
    src = _mrimodel_source(M)
    src === nothing && throw(ArgumentError("newmodel: $(nameof(M)) was not made with @mrimodel, " *
        "so its code is already written (`@which signal!(zeros(1), m, p)` finds it); for a new " *
        "model, `newmodel(:Name; acquisition, parameters)`"))
    name, acq, opts, pars, body = _mmparse(Base.remove_linenums!(deepcopy(src)))
    fields = [acq; first.(opts)]
    pnames = first.(pars)
    roles = Pair{Symbol,String}[n => _mmroletext(n, d, fields, pnames) for (n, d) in pars]
    optcode = Pair{Symbol,String}[o => string(d) for (o, d) in opts]
    optvals = Dict{Symbol,Any}(o => d for (o, d) in opts if d isa Union{Bool,Number,String})
    return _writemodel(file, force, name, acq, optcode, optvals, roles, _bodytext(body),
                       _docstring(M), true)
end

# The definition @mrimodel keeps for each model it makes (nothing for other models).
_mrimodel_source(::Type) = nothing

# A role as Julia code (for a blank model given roles): what `show` prints (`Free(start = :scale,
# lower = 0.0)`, `Fixed(12.5)`, `Fixed(:R1f)`), a function written as a TODO to fill in.
function _roletext(r)
    t = repr(r)
    return occursin("<function>", t) ? replace(t, "start = <function>" => "start = m -> TODO",
                                               "Fixed(<function>)" => "Fixed(p -> TODO)") *
                                       ",                    # TODO: the function" : t * ","
end

# An @mrimodel default as the role it makes, written as Julia code (as `_mmrole` makes it, in
# macro.jl, but with plain names: `m -> mean(m.TE)`, `Fixed(:R1f)`, `Fixed(p -> 0.8 * p.R1f)`).
function _mmroletext(n, d, fields, pnames)
    others = filter(!=(n), pnames)
    d === nothing && return "Free(),                    # TODO: a start (or give one in the fit)"
    if _mmcalls(d, :Free)
        kws = Any[]
        for a in d.args[2:end]
            a isa Expr && a.head === :parameters ? append!(kws, a.args) : push!(kws, a)
        end
        items = map(kws) do k
            key, v = k.args
            key === :start && _mmuses(v, fields) ?
                "start = m -> " * string(_mmsubst(v, fields, :m)) : "$key = $(string(v))"
        end
        return "Free(" * join(items, ", ") * "),"
    end
    _mmcalls(d, :Fixed) && return string(d) * ","
    d isa Symbol && d in others && return "Fixed(:$d),                # tie: $n equals $d"
    _mmuses(d, others) && return "Fixed(p -> " * string(_mmsubst(d, others, :p)) * "),"
    return "Fixed(" * string(d) * "),"
end

# The body as code, one statement a line, indented once (for the function of one observation).
# Julia prints `c ? a : b` as an if-block and `a, b = x, y` with parentheses; written back here
# as they are usually typed, at the top of a statement, of an assignment's right side or of a
# `return` (deeper inside an expression, Julia's own printing).
function _bodytext(body)
    stmts = body isa Expr && body.head === :block ? body.args : Any[body]
    return join((replace(_code(s), "\n" => "\n    ") for s in stmts), "\n    ")
end
_isternary(x) = x isa Expr && x.head === :if && length(x.args) == 3 &&
                !any(a -> a isa Expr && a.head === :block, x.args[2:3])
function _code(x)
    _isternary(x) && return "$(_code(x.args[1])) ? $(_code(x.args[2])) : $(_code(x.args[3]))"
    x isa Expr && x.head === :return && length(x.args) == 1 && return "return " * _code(x.args[1])
    if x isa Expr && x.head === :(=) && length(x.args) == 2
        lhs, rhs = x.args
        tup(e) = e isa Expr && e.head === :tuple ? join(_code.(e.args), ", ") : _code(e)
        return tup(lhs) * " = " * tup(rhs)
    end
    return string(x)
end

# The model's docstring, if it has one (an @mrimodel definition's), as text.
function _docstring(M)
    mod = parentmodule(M)
    md = get(Base.Docs.meta(mod), Base.Docs.Binding(mod, nameof(M)), nothing)
    md === nothing && return nothing
    texts = String[]
    for d in values(md.docs), t in d.text
        t isa AbstractString && push!(texts, strip(t))
    end
    return isempty(texts) ? nothing : join(texts, "\n\n")
end

function _writemodel(file, force, name, acq, opts, optvals, roles, body, doc, frommacro)
    isfile(file) && !force && throw(ArgumentError("newmodel: $file exists; give another " *
                                                  "`file`, or `force = true` to replace it"))
    dir = dirname(file)
    isempty(dir) || mkpath(dir)
    fn = "_" * lowercase(String(name))                    # the signal of one observation
    pnames = first.(roles)
    tvars = ["A$i" for i in eachindex(acq)]
    # Fields: the acquisition, a vector each (a type parameter each, so any vector works); the
    # options with their defaults (typed when the default is a literal).
    # A directional model (acquisition b and g) as the built-in ones: g of type `Directions`, and
    # the inner constructor that checks b and g when the model is made (`checkencoding`)
    directional = (:b, :g) ⊆ acq
    note(a) = directional && a === :b ? "# b-values (s/mm²)" :
              directional && a === :g ? "# gradient direction of each image (a unit SVector{3})" :
              "# TODO: units; one value per observation"
    fieldlines = ["    $a::$t" * " "^max(1, 33 - length("$a::$t")) * note(a) for (a, t) in zip(acq, tvars)]
    for (o, code) in opts
        v = get(optvals, o, nothing)
        line = v isa Union{Bool,Number,String} ? "    $o::$(typeof(v)) = $code" : "    $o = $code"
        push!(fieldlines, line * " "^max(1, 37 - length(line)) * "# an option")
    end
    curly = join((directional && a === :g ? "$t<:Directions" : "$t<:AbstractVector" for (a, t) in zip(acq, tvars)), ", ")
    if directional                        # the constructor, checking b and g (see `Directions`)
        cargs = join([["$a::$t" for (a, t) in zip(acq, tvars)]; string.(first.(opts))], ", ")
        cvals = join([[a === :g ? "checkencoding(b, g)" : string(a) for a in acq]; string.(first.(opts))], ", ")
        push!(fieldlines, "    $name($cargs) where {$(join(tvars, ","))} =
" *
                          "        new{$(join(tvars, ","))}($cvals)   # b and g checked (`Directions`)")
    end
    sig = "$name(; " * join([string.(acq); ["$o = $c" for (o, c) in opts]], ", ") * ")"
    w = maximum(length ∘ string, pnames)
    rolelines = ["    " * rpad(string(n), w) * " = " * r for (n, r) in roles]
    args = join([string.(acq); string.(first.(opts)); string.(pnames)], ", ")
    callargs = join([["$a[j]" for a in acq]; string.(first.(opts)); string.(pnames)], ", ")
    unpackm = join([string.(acq); string.(first.(opts))], ", ")
    a1, an = acq[1], acq[end]
    onevalue = length(acq) < 2 ? "" : "# One value of `$an` for every observation can be allowed by a method of its own (as\n" *
        "# T1MFA does for TR): $name($(join(acq[1:end-1], ", ")), $an::Real$(isempty(opts) ? "" : ", " * join(first.(opts), ", "))) =\n" *
        "#     $name($(join(acq[1:end-1], ", ")), fill($an, length($a1))$(isempty(opts) ? "" : ", " * join(first.(opts), ", ")))\n"
    docbody = doc === nothing ? "TODO: what the model is: the signal equation, its units, the reference." : doc
    origin = frommacro ? "from its @mrimodel definition" : "as a blank model"
    text = """
    # =================================================================================================
    # $name
    #
    # TODO: what the model is, its signal equation in the paper's notation, and the reference
    # (author, journal, year, table). Written by EQUIP's `newmodel`, $origin: edit it.
    # =================================================================================================

    using EQUIP
    import EQUIP: parameters, signal!       # the model adds methods to these functions
    # (and the packages its starts or signal use, e.g. `using Statistics: mean`)

    \"\"\"
        $sig

    $docbody

    - acquisition: $(join(("`$a` (TODO: units)" for a in acq), ", ")), one value per observation$(isempty(opts) ? "" : "\n- options: " * join(("`$o` (default $c)" for (o, c) in opts), ", "))
    - parameters: $(join(("`$n`" for n in pnames), ", "))
    \"\"\"
    @kwdef struct $name{$curly} <: MRIModel
    $(join(fieldlines, "\n"))
    end
    $onevalue
    parameters(::Type{<:$name}) = (
    $(join(rolelines, "\n"))
    )

    # The signal of one observation. Work that is the same for every observation (rates that depend
    # only on the parameters, say) can move out of it, to the top of `signal!`, done once per
    # voxel; inlined, the compiler often moves it there itself.
    @inline function $fn($args)
        $body
    end

    function signal!(out, m::$name, p)
        (; $unpackm) = m$(" "^max(1, 28 - length(unpackm)))# the acquisition, by name
        (; $(join(pnames, ", "))) = p$(" "^max(1, 28 - length(join(pnames, ", "))))# the parameters, by name (never by position)
        @inbounds for j in eachindex(out, $(join(acq, ", ")))    # checks once that the lengths match
            out[j] = $fn($callargs)
        end
        return out
    end

    # Optional hooks (see their docstrings):
    # EQUIP.datastart(m::$name, y, p) = (name = value,)    starting values from the voxel's data
    # EQUIP.relabel(m::$name, θ) = θ                       if parameters can swap labels
    # EQUIP.derived(m::$name, p) = (name = value,)       quantities computed from the parameters
    # EQUIP.fromderived(::Type{<:$name}, q) = (...)       the parameters from them, for nominal

    # A first check: the signal at the model's starting and fixed values, fitted back.
    # m = $name($(join(("$a = ..." for a in acq), ", ")))
    # y = signal(m, nominal(m))
    # fit(m, y)
    """
    open(io -> write(io, text), file, "w")
    return String(file)
end
