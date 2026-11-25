# =================================================================================================
# @mrimodel: a model from an ordinary Julia function of one observation
#
#   @mrimodel MyT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05), C = 0.0) =
#       S0 * exp(-TE / T2) + C
#
# The function's pieces become the model's (docs/design.md, roadmap item 8):
#
#   name                         the model type: MyT2(TE = ...) makes one
#   positional arguments         the acquisition: one value per observation, or one number for
#                                all (in the body: this observation's value)
#   positional, with a default   options (magnitude = true): in the body, the value as it is
#   keywords, in order           the parameters, their defaults the default roles:
#                                  none                 free, its start given in the call
#                                  Free(...)            free (a start may use the acquisition:
#                                                       Free(start = mean(TE)), TE the whole vector)
#                                  a number, Fixed(...) fixed
#                                  another parameter    a tie (R1m = R1f), or a formula of
#                                                       others (R1m = 0.8 * R1f)
#   body                         the signal of one observation
#
# What it writes (`@macroexpand` shows it): the struct (a type parameter per field, so any vector
# or number works; for a directional model, one whose acquisition is `b` and `g`, g's bound
# `DiffusionDirections` and the inner constructor that checks b and g, as the built-in models have), a
# keyword constructor, `parameters` for the type, the body as a function of
# one observation (`_obsignal`), `signal!` calling it in a loop (the lengths checked once, then
# `@inbounds`: as fast as a hand-written loop, no allocation), and the conversion to the fit's
# precision; and the definition, kept for `newmodel` (template.jl), which writes it out as plain
# code. A docstring before `@mrimodel` documents the type. Hooks (`datastart`, `relabel`, `derived`,
# `fromderived`) are ordinary methods added afterwards. A signal that shares work across observations (QMTSIR's
# rates, computed once per voxel) is written by hand, as the built-in models are.
# =================================================================================================

"""
    @mrimodel Name(acquisition...; parameters...) = signal of one observation
    @mrimodel function Name(acquisition..., options...; parameters...) ... end

A model from an ordinary function of one observation: the function's name becomes the model
type, its positional arguments the acquisition, its keywords the parameters (their defaults the
default roles), its body the signal of one observation.

```julia
@mrimodel StretchedT2(TE; S0 = Free(start = :scale, lower = 0),
                          T2 = Free(start = 0.05, lower = 1e-6),
                          beta = Free(start = 0.9, lower = 0.1, upper = 1)) =
    S0 * exp(-(TE / T2)^beta)

m = StretchedT2(TE = 0.01:0.01:0.32)
res = fit(m, data; mask)                    # as any model: crlb, montecarlo, design too
```

- **Acquisition**: the positional arguments, given by name when the model is made
  (`StretchedT2(TE = ...)`), each one value per observation or one number for every
  observation. In the body, the observation's value.
- **Options**: positional arguments with a default (`magnitude = true`): optional when the model
  is made; in the body, the value as it is.
- **Parameters**: the keywords, in order (the order of the outputs). Their defaults are the
  default roles, which a fit changes as for any model (`fit(m, data; C = Free())`):
  - no default: free, with no start of its own (give one in the call:
    `fit(m, data; T2 = Free(start = 0.05))`);
  - `Free(start = ..., lower = ..., upper = ...)`: free. The start may use the acquisition, as
    Julia's keyword defaults may use earlier arguments: `T2 = Free(start = mean(TE))`, where
    `TE` is the whole vector;
  - a number, or `Fixed(...)`: fixed;
  - another parameter's name: tied to it (`R1m = R1f`), or a formula of others
    (`R1m = 0.8 * R1f`).
- **Diffusion directions**: a model whose acquisition is `b` and `g` (b-values and gradient directions,
  as [`diffusionencoding`](@ref) gives them) is directional: in the body, `g` is the
  observation's direction, an `SVector{3}` (`ξ = g ⋅ n`). The macro writes the struct as the
  built-in directional models are written, `g` of type [`DiffusionDirections`](@ref) and checked by
  [`checkencoding`](@ref) when the model is made (a direction per b-value, unit vectors where
  b > 50 s/mm²).
- **The body** is the signal of one observation: plain Julia, any number of lines, the last
  value (or `return`) the signal. As for any model: no allocation, and generic in the number
  type (the fit passes dual numbers to get derivatives).

A docstring before `@mrimodel` documents the model. Hooks are ordinary methods on the type,
added afterwards: `EQUIP.datastart(m::StretchedT2, y, p) = ...`. `@macroexpand` shows the code
written. A signal that shares work across observations (QMTSIR computes its exchange rates
once per voxel) is faster written by hand: see "Writing a model" in the README.
"""
macro mrimodel(ex)
    return esc(_mrimodel(ex))
end

# Errors while the macro expands, naming the macro and saying what to write instead.
_mmerror(msg) = throw(ArgumentError("@mrimodel: " * msg))

# The pieces of the function: name, acquisition, options (name => default), parameters
# (name => default, or nothing), body.
function _mmparse(ex)
    (ex isa Expr && ex.head in (:(=), :function) && length(ex.args) == 2) ||
        _mmerror("write a function definition: `@mrimodel Name(TE; S0, T2) = S0 * exp(-TE / T2)`, " *
                 "or `@mrimodel function Name(...) ... end`")
    call, body = ex.args
    call isa Expr && call.head === :where &&
        _mmerror("no `where` clause: the model's types come from the values given")
    (call isa Expr && call.head === :call) ||
        _mmerror("write a function definition: `@mrimodel Name(TE; S0, T2) = ...`")
    name = call.args[1]
    name isa Symbol || _mmerror("the model's name must be a plain name (`MyT2`), got `$name`")
    rest = call.args[2:end]
    kws = !isempty(rest) && rest[1] isa Expr && rest[1].head === :parameters ? popfirst!(rest).args : Any[]
    acq, opts, pars = Symbol[], Pair{Symbol,Any}[], Pair{Symbol,Any}[]
    for a in rest
        if a isa Symbol
            isempty(opts) || _mmerror("`$a`: the acquisition comes before the options (the " *
                                      "positional arguments with a default)")
            push!(acq, a)
        elseif a isa Expr && a.head === :kw && a.args[1] isa Symbol
            push!(opts, a.args[1] => a.args[2])
        else
            _mmerror("`$a`: the positional arguments are the acquisition's names (`TE`), or " *
                     "options with a default (`magnitude = true`); no types (they come from the values)")
        end
    end
    for k in kws
        if k isa Symbol
            push!(pars, k => nothing)
        elseif k isa Expr && k.head === :kw && k.args[1] isa Symbol
            push!(pars, k.args[1] => k.args[2])
        else
            _mmerror("`$k`: the keywords are the parameters, by name, with a default role or " *
                     "none (`S0`, `T2 = Free(start = 0.05)`, `kmf = 12.5`); no types")
        end
    end
    isempty(acq) && _mmerror("$name has no acquisition: give it as the first positional " *
                             "argument, `$name(TE; ...)`")
    isempty(pars) && _mmerror("$name has no parameters: give them as keywords, after a " *
                              "semicolon, `$name(TE; S0, T2)`")
    names = [acq; first.(opts); first.(pars)]
    dup = [n for n in unique(names) if count(==(n), names) > 1]
    isempty(dup) || _mmerror("`$(dup[1])` is used twice: the acquisition, options and " *
                             "parameters need different names")
    return name, acq, opts, pars, body
end

# Whether an expression uses any of `names`: as a variable, not as a keyword's name
# (`Free(start = ...)`), a field (`x.TE`) or a quoted symbol.
_mmuses(ex::Symbol, names) = ex in names
function _mmuses(ex::Expr, names)
    ex.head === :kw && return _mmuses(ex.args[2], names)
    ex.head === :. && return _mmuses(ex.args[1], names)
    ex.head === :quote && return false
    return any(a -> _mmuses(a, names), ex.args)
end
_mmuses(ex, names) = false

# The expression with each parameter name `n` replaced by `p.n` (a tie's formula).
_mmsubst(ex::Symbol, names, p) = ex in names ? Expr(:., p, QuoteNode(ex)) : ex
function _mmsubst(ex::Expr, names, p)
    ex.head === :kw && return Expr(:kw, ex.args[1], _mmsubst(ex.args[2], names, p))
    ex.head === :. && return Expr(:., _mmsubst(ex.args[1], names, p), ex.args[2:end]...)
    ex.head === :quote && return ex
    return Expr(ex.head, map(a -> _mmsubst(a, names, p), ex.args)...)
end
_mmsubst(ex, names, p) = ex

_mmcalls(ex, f) = ex isa Expr && ex.head === :call && (ex.args[1] === f ||
    (ex.args[1] isa Expr && ex.args[1].head === :. && ex.args[1].args[end] == QuoteNode(f)))

# A parameter's default as its role.
function _mmrole(n, d, fields, pnames, model)
    Free_, Fixed_ = GlobalRef(EQUIP, :Free), GlobalRef(EQUIP, :Fixed)
    d === nothing && return :($Free_())                     # free, its start given in the call
    others = filter(!=(n), pnames)
    if _mmcalls(d, :Free)
        # Its keywords, whether written `Free(start = ...)` or `Free(; start = ...)`.
        kws = Any[]
        for a in d.args[2:end]
            a isa Expr && a.head === :parameters ? append!(kws, a.args) : push!(kws, a)
        end
        out = Any[]
        for k in kws
            (k isa Expr && k.head === :kw) || _mmerror("`$n = $d`: Free takes keywords " *
                                                       "(start, lower, upper)")
            key, v = k.args
            _mmuses(v, others) && _mmerror("`$n = $d`: a start or bound cannot depend on " *
                                           "another parameter (only on the acquisition)")
            if key === :start && _mmuses(v, fields)
                # A start that uses the acquisition: a function of the model, called once per fit
                # with the model, its fields bound to their names (as Free documents).
                m = gensym(:m)
                binds = [Expr(:(=), f, :(getfield($m, $(QuoteNode(f))))) for f in fields if _mmuses(v, [f])]
                v = Expr(:->, m, Expr(:let, Expr(:block, binds...), v))
            elseif _mmuses(v, fields)
                _mmerror("`$n = $d`: only the start may use the acquisition; bounds are numbers")
            end
            push!(out, Expr(:kw, key, v))
        end
        return Expr(:call, Free_, out...)
    end
    _mmuses(d, fields) && _mmerror("`$n = $d`: a fixed value cannot depend on the acquisition " *
        "(a start can: `$n = Free(start = ...)`)")
    _mmcalls(d, :Fixed) && return d
    d isa Symbol && d in others && return :($Fixed_($(QuoteNode(d))))      # a tie by name
    d === n && _mmerror("`$n = $n`: a parameter cannot be tied to itself")
    if _mmuses(d, others)                                                  # a tie by formula
        p = gensym(:p)
        return :($Fixed_($p -> $(_mmsubst(d, others, p))))
    end
    return :($Fixed_($d))                                                  # a value
end

function _mrimodel(ex)
    name, acq, opts, pars, body = _mmparse(ex)
    fields = [acq; first.(opts)]
    pnames = first.(pars)
    E = EQUIP                                            # EQUIP's functions, whatever the caller imported
    tparams = [Symbol("T", i) for i in eachindex(fields)]
    # A directional model (acquisition b and g): g bound to `DiffusionDirections` (SVector{3}s), and an
    # inner constructor that checks b and g when the model is made (as the built-in ones do)
    directional = (:b, :g) ⊆ acq
    bounds = [directional && f === :g ? Expr(:<:, t, GlobalRef(E, :DiffusionDirections)) : t
              for (f, t) in zip(fields, tparams)]
    inner = directional ?
        :($name($((Expr(:(::), f, t) for (f, t) in zip(fields, tparams))...)) where {$(tparams...)} =
              new{$(tparams...)}($((f === :g ? :($E.checkencoding(b, g)) : f for f in fields)...))) :
        nothing
    structdef = Expr(:struct, false, Expr(:<:, Expr(:curly, name, bounds...), GlobalRef(E, :MRIModel)),
                     Expr(:block, (Expr(:(::), f, t) for (f, t) in zip(fields, tparams))..., inner))
    # The keyword constructor: the acquisition required, the options with their defaults.
    kwargs = Expr(:parameters, acq..., (Expr(:kw, o, d) for (o, d) in opts)...)
    kwcons = :($name($kwargs) = $name($(fields...)))
    # The default roles, for the type.
    roles = Expr(:tuple, (Expr(:(=), n, _mmrole(n, d, fields, pnames, name)) for (n, d) in pars)...)
    paramsdef = :($E.parameters(::Type{<:$name}) = $roles)
    # The body: a function of one observation (the model first, for dispatch).
    obsdef = :(@inline function $E._obsignal(::$name, $(fields...), $(pnames...))
                   $body
               end)
    # signal!: the per-observation values checked against `out` once, then the loop.
    out, m, p, j = gensym(:out), gensym(:m), gensym(:p), gensym(:j)
    fvars = [gensym(f) for f in fields]
    pvars = [gensym(n) for n in pnames]
    nacq = length(acq)
    args = [i <= nacq ? :($E._obs($v, $j)) : v for (i, v) in enumerate(fvars)]
    sigdef = quote
        function $E.signal!($out, $m::$name, $p)
            $((:($v = getfield($m, $(QuoteNode(f)))) for (v, f) in zip(fvars, fields))...)
            $((:($E._checkobs($out, $v, $(QuoteNode(f)), $(QuoteNode(name)))) for (v, f) in zip(fvars[1:nacq], acq))...)
            $((:($v = $p.$n) for (v, n) in zip(pvars, pnames))...)
            @inbounds for $j in eachindex($out)
                $out[$j] = $E._obsignal($m, $(args...), $(pvars...))
            end
            return $out
        end
    end
    # The fit's precision: the acquisition converted (vectors, and single numbers too), the
    # options kept as they are.
    T = gensym(:T)
    conv = [i <= nacq ? :($E._acq_precision(getfield($m, $(QuoteNode(f))), $T)) :
                        :(getfield($m, $(QuoteNode(f)))) for (i, f) in enumerate(fields)]
    precdef = :($E._with_precision($m::$name, ::Type{$T}) where {$T} = $name($(conv...)))
    # The definition itself, kept: `newmodel(Name)` writes it out as plain code (template.jl).
    srcdef = :($E._mrimodel_source(::Type{<:$name}) = $(QuoteNode(ex)))
    return quote
        Base.@__doc__ $structdef
        $kwcons
        $paramsdef
        $obsdef
        $sigdef
        $precdef
        $srcdef
        $name
    end
end

# The generated code's helpers. `_obsignal` gets a method per model; `_obs` reads one
# observation's value (a vector's element, or the one number for every observation);
# `_checkobs` checks a vector's length against `out` once, before the loop (which makes
# `@inbounds` safe); `_acq_precision` converts the acquisition to the fit's precision.
function _obsignal end
Base.@propagate_inbounds _obs(x::AbstractVector, j) = x[j]
_obs(x, j) = x
function _checkobs(out, x::AbstractVector, name, model)
    axes(x, 1) == axes(out, 1) || throw(DimensionMismatch("$model: `$name` has $(length(x)) " *
        "values for $(length(out)) observations"))
    return nothing
end
_checkobs(out, x, name, model) = nothing                 # one number for every observation
_acq_precision(x::Real, ::Type{T}) where {T} = x isa Bool ? x : T(x)
_acq_precision(x, ::Type{T}) where {T} = _to_precision(x, T)
