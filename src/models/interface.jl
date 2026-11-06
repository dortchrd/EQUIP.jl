# =================================================================================================
# The model interface
#
# Everything a model author needs: the supertype `MRIModel`, the two parameter roles `Free` and
# `Fixed` (with `BestOf`, a kind of start), and the two functions every model defines,
# `parameters` and `signal!`. Plus the optional hooks `datastart` and `relabel`, `nobs` (number of
# observations) and the simulation wrapper `signal`.
#
# Two separate inputs reach a signal equation:
#
#   model  m    the physics: acquisition + options    a struct:      QMTSIR(TI = [...], TD = 2.5)
#   p           the tissue/model parameter VALUES     a NamedTuple:  (PSR = 0.1, R1f = 1.0, ...)
#
# The model holds everything that is the same for every voxel. It never stores parameter values:
# the same model object is reused for millions of voxels, while `p` changes at every step of
# every fit.
# =================================================================================================

"""
    MRIModel

Supertype of all signal models. A model is three pieces:

1. A struct `<: MRIModel` holding everything that is the same for every voxel: the acquisition
   (one value per observation, e.g. `TE`) and options (e.g. `magnitude::Bool`).
2. [`parameters`](@ref)`(m)`: every parameter of the signal equation with its default role,
   [`Free`](@ref) or [`Fixed`](@ref), as a named tuple.
3. [`signal!`](@ref)`(out, m, p)`: one voxel's signal at every observation.

Example (see `src/models/` for more):

```julia
@kwdef struct MyDecay{A<:AbstractVector} <: MRIModel
    TE::A                          # echo times (s), one per observation
end

parameters(::Type{<:MyDecay}) = (S0 = Free(start = :scale, lower = 0),
                                 T2 = Free(start = 0.05, lower = 1e-6))

function signal!(out, m::MyDecay, p)
    (; TE) = m                     # unpack the acquisition by name ...
    (; S0, T2) = p                 # ... and the parameters
    @inbounds for j in eachindex(out, TE)   # checks once that out and TE match
        out[j] = S0 * exp(-TE[j] / T2)
    end
    return out
end
```
"""
abstract type MRIModel end
# `abstract type` declares a type that has no instances of its own; it only groups concrete
# types. `struct MyDecay <: MRIModel` says "MyDecay is a kind of MRIModel", so any function
# written for `m::MRIModel` (like `signal` and `fit`) works for it.
#
# `@kwdef` (built into Julia) in front of `struct` adds a keyword constructor, so a model is made
# with `MyDecay(TE = [...])`. A field written with a default, `magnitude::Bool = true`, becomes
# an optional keyword; a field without one is required (leaving it out is an error).

# -------------------------------------------------------------------------------------------------
# Parameter roles
#
# Each parameter of a model is either Free (estimated by the fit) or Fixed (held at a value).
# Each role is a tiny struct that carries what the fitter needs to know about that parameter.
# -------------------------------------------------------------------------------------------------

"""
    Free(; start, lower, upper)

Role of a parameter that is estimated: its starting value and its bounds, each given by name.

In a model's [`parameters`](@ref), the defaults: `Free(start = 0.1, lower = 0, upper = 1)`. A
bound left out means none. In a fit, what is left out keeps the model's setting; give a bound as
infinite (`lower = -Inf`, `upper = Inf`) to remove one the model sets. The start says where
each voxel's starting value comes from:

- a number, or a map (one value per voxel, in a fit): start there;
- `:scale`, for the parameter that multiplies the whole signal (S0, M0): the least-squares
  value given the other starts, so the data need no normalization. Only the model declares
  it: the rule assumes the signal is proportional to the parameter (true for M0, not for T1).
  A fit can turn it off (`M0 = Free(start = 1000.0)`) but cannot move it to another parameter;
- `:data`: the model's own estimate from the voxel's data (see [`datastart`](@ref)), e.g. a
  log-linear fit for T2. Only for parameters the model's `datastart` estimates;
- [`BestOf`](@ref)`(values)`: try each value and start from the one that fits best;
- a function of the model, `m -> ...`, for a start that depends on the acquisition: called
  once per fit with the model, it gives any of the above, e.g. ADCDecay's
  `ADC = Free(start = m -> 1 / mean(m.b))` and T1IR's
  `T1 = Free(start = m -> BestOf(_t1grid(m.TI)))`. This keeps [`parameters`](@ref) a property
  of the model's type, so it needs no acquisition.

In a fit, to estimate a parameter or change its start or bounds; whatever is left out keeps the
model's default:

```julia
fit(m, data; kmf = Free())                 # estimate kmf, fixed by default (starts at its value)
fit(m, data; R1f = Free(start = 0.8))      # another start: a number, or a map (one per voxel)
fit(m, data; T1 = Free(start = BestOf([0.3, 0.8, 1.5])))   # try three values, start from the best
fit(m, data; Sf = Free(lower = -1.5))      # another lower bound
fit(m, data; PSR = Free(lower = -Inf))     # no lower bound (the model's is 0)
```
"""
struct Free
    start::Any                         # a number, a map, :scale, :data, a BestOf, a function of the
                                       # model (m -> ...), or nothing (keep the default)
    lower::Union{Nothing,Float64}      # nothing: keep the default (no bound in a model's parameters)
    upper::Union{Nothing,Float64}
    # The one constructor, by keyword: `Free()`, `Free(start = 0.8)`, `Free(start = :scale,
    # lower = 0)`. Written inside the struct (an "inner constructor"), it replaces the positional
    # `Free(start, lower, upper)` Julia would otherwise define, so every Free is made here: the
    # start is checked, numbers become Float64, and `new` (only available inside) builds it.
    function Free(; start = nothing, lower = nothing, upper = nothing)
        start === nothing || start === :scale || start === :data || start isa Real ||
            start isa AbstractArray || start isa BestOf || start isa Function ||
            throw(ArgumentError("Free: the start must be a number, a map, :scale, :data, " *
                                "BestOf(values) or a function of the model, got $(repr(start))"))
        tofloat(x) = x === nothing ? nothing : Float64(x)
        return new(start isa Real ? Float64(start) : start, tofloat(lower), tofloat(upper))
    end
end
"""
    BestOf(values)

A start for a [`Free`](@ref) parameter: try each value and start from the one that fits best. In
every voxel the fit evaluates the signal at each value (the other parameters at their starts, a
`:scale` parameter at its least-squares value for each) and runs one fit from the value with the
lowest sum of squared residuals. For fits that can end in a wrong local minimum from a single
start, such as magnitude inversion recovery. Values outside the parameter's bounds are dropped.
One parameter per fit can start from `BestOf` (several, trying every combination, are planned).

```julia
parameters(::Type{<:T1IR}) = (..., T1 = Free(start = m -> BestOf(_t1grid(m.TI)), lower = _TMIN))
fit(m, data; T1 = Free(start = BestOf(logrange(0.05, 5, 50))))                  # other values
```

The values are a list, not a map: `Free(start = [0.3, 0.8, 1.5])` would be a map, one start per
voxel, which is why they are wrapped.
"""
struct BestOf
    values::Vector{Float64}
    # An inner constructor (inside the struct) replaces the default one, so every BestOf is
    # checked: at least one value, all finite. Any list of numbers works (a vector, a range).
    function BestOf(values)
        v = Float64[x for x in values]
        isempty(v) && throw(ArgumentError("BestOf: give at least one value"))
        all(isfinite, v) || throw(ArgumentError("BestOf: the values must be finite"))
        return new(v)
    end
end
# How roles print, e.g. in `parameters(m)`: only what is set, by name, as they are written:
# `Free(start = :scale, lower = 0.0)` rather than the raw fields `Free(:scale, 0.0, nothing)`.
# A map prints as its summary (`64×64 Matrix{Float64}`), not its values.
_showvalue(x::AbstractArray) = summary(x)
_showvalue(x::Function) = "<function>"             # a tie such as p -> 0.8 * p.R1f
_showvalue(x) = repr(x)
function Base.show(io::IO, f::Free)
    parts = String[]
    f.start === nothing || push!(parts, "start = " * _showvalue(f.start))
    f.lower === nothing || push!(parts, "lower = " * repr(f.lower))
    f.upper === nothing || push!(parts, "upper = " * repr(f.upper))
    print(io, "Free(", join(parts, ", "), ")")
end

# Printed in full when short, else summarized: BestOf(30 values from 0.01 to 12.0).
function Base.show(io::IO, b::BestOf)
    v = b.values
    r(x) = round(x; sigdigits = 3)
    length(v) <= 6 ? print(io, "BestOf(", v, ")") :
        print(io, "BestOf(", length(v), " values from ", r(minimum(v)), " to ", r(maximum(v)), ")")
end

# A positional start, `Free(0.8)` or `Free(:scale; lower = 0)`, is an error that shows the call
# written by name. (`args...` collects any positional arguments; a call with none goes to the
# constructor above, the more specific method.)
function Free(args...; kwargs...)
    names = (:start, :lower, :upper)
    length(args) <= 3 ||
        throw(ArgumentError("Free takes keywords: Free(start = ..., lower = ..., upper = ...)"))
    parts = ["$(names[i]) = $(_showvalue(args[i]))" for i in eachindex(args)]
    append!(parts, ["$k = $(_showvalue(v))" for (k, v) in kwargs])
    throw(ArgumentError("Free takes keywords, so the start is named like the bounds: write " *
                        "Free($(join(parts, ", ")))"))
end

"""
    Fixed(value)
    Fixed(value; free = Free(...))

Role of a parameter that is not estimated. `value` is

- a number, the same for every voxel: `Fixed(12.5)`
- a per-voxel map, an array with one value per voxel (e.g. a B1 map): `Fixed(b1map)`
- a tie to another parameter, given by its name: `Fixed(:R1f)` ("always equal to R1f")
- a tie given by a function of the other parameters: `Fixed(p -> 0.8 * p.R1f)`

A tie may use free or fixed parameters (including per-voxel maps), but not another tie.

`free`, in a model's [`parameters`](@ref): a part of the model that is switched off by default
(an offset fixed at 0, a second component's fraction at 0, a compartment's T2 at `Inf`), and the
role it takes when a fit switches it on with `Free()`, its start and bounds where the call leaves
them out: `C = Fixed(0.0; free = Free(start = 0.0))`, then `fit(m, y; C = Free())`. A part
switched off also needs no value where a complete set is asked for (`simulate`, `signal`,
`nominal`, a scenario's truth): it is filled in at its value. See "Building blocks" in
src/models/blocks.jl.
"""
struct Fixed{T}
    value::T
    free::Union{Nothing,Free}    # a part switched off: its role when switched on (else nothing)
end
Fixed(value; free::Union{Nothing,Free} = nothing) = Fixed(value, free)
# `Fixed(12.5)` keeps working: the keyword method above, with `free = nothing`. The struct's own
# two-argument constructor, `Fixed(value, free)`, is the one it calls.

# A fixed parameter with a role for when it is freed (`Fixed(value; free = ...)`): filled in at its
# value where a complete set leaves it out. It is a part switched off when that value removes the
# part, 0 (a fraction, an offset, a rate) or Inf (a T2): shown as `off`, and left out of leakage
# (a relative error of 0 or Inf has no meaning). SMTFreeWater's Diso = 3e-3 is switchable, not off.
_switchable(r) = r isa Fixed && r.free !== nothing
_isoff(r) = _switchable(r) && r.value isa Real && (iszero(r.value) || isinf(r.value))
# `Fixed{T}` is a *parametric* type: T is the type of `value`, so `Fixed(12.5)` is a
# `Fixed{Float64}`, `Fixed(b1map)` a `Fixed{Matrix{Float64}}`, `Fixed(:R1f)` a `Fixed{Symbol}`
# and `Fixed(p -> 0.8 * p.R1f)` a `Fixed{var"#1#2"}` (every anonymous function has its own type).
# Keeping the exact type lets the compiler generate fast, specialized code for each case.
Base.show(io::IO, f::Fixed) = print(io, "Fixed(", _showvalue(f.value), ")")   # e.g. Fixed(12.5), Fixed(:R1f)

"""
    parameters(m) -> NamedTuple
    parameters(M::Type) -> NamedTuple
    parameters(m; changes...) -> NamedTuple

Every parameter of a model with its default role, [`Free`](@ref) or [`Fixed`](@ref). The order
is the order of the outputs. A fit can change any role with a keyword named after the parameter:
`fit(m, data; kmf = 15.0)` (fixed at a value), `fit(m, data; kmf = Free())` (estimated).

With keywords, the model's roles with those changes, by the same rules as `fit`'s keywords (a
value fixes, `Free(...)` estimates and keeps the defaults it leaves out): a complete set of
roles made by a call, to keep and pass on, as `nominal(m; kmf = 10.0)` makes a complete set of
values. `fit` and `init` take it after the data:

```julia
roles = parameters(m; kmf = 10.0, Sm = 0.9)     # every parameter's role, two of them changed
fit(m, y, roles)                                 # the same as fit(m, y; kmf = 10.0, Sm = 0.9)
```

For `crlb` and `montecarlo`, a fit is judged against a truth: a [`Scenario`](@ref) builds its
roles from the truth (`Scenario(m, θ; kmf = 10.0)`: the fixed parameters at their true values,
then the changes), and `s.roles` fits data the same way: `fit(m, y, s.roles)`.

The roles are a property of the model's type, the same for every acquisition, so a model
declares them for its type, and asking a model (an instance) asks its type:

```julia
parameters(::Type{<:T1SR}) = (M0 = Free(start = :scale, lower = 0),
                              T1 = Free(start = 1.0, lower = 1e-6))
parameters(T1SR)                     # no acquisition needed
parameters(T1SR(TD = [0.1, 1.0]))    # the same
```

`Type{<:T1SR}` ("T1SR or any of its subtypes") rather than `Type{T1SR}`: the models are
parametric types, and a model's type is e.g. `T1SR{Vector{Float64}}`, which `Type{T1SR}` does
not match. A start that depends on the acquisition is a function of the model, called once per
fit (see `Free`): asked of a model, `parameters` gives the start computed for its acquisition
(T1IR's `BestOf` values for its TIs); asked of the type, the function. A model whose roles
cannot be written that way can define `parameters(m::MyModel)` instead; it then needs a model
wherever roles are asked for.

Values rather than roles: [`defaults`](@ref)`(m)` (the fixed parameters' values),
[`nominal`](@ref)`(m)` or `nominal(m, tissue)` (every value, from the model or from a tissue),
[`resolve`](@ref)`(m, p)` (a complete set checked, ties turned into numbers). Typing `m` at the
REPL shows the roles as a table. `keys(parameters(m))` lists the names.
"""
function parameters end
# `function name end` declares a generic function with no methods yet. Each model adds its own
# method, e.g. `parameters(::Type{<:T1MFA}) = ...`; Julia picks the method by the type of the
# argument ("multiple dispatch"). `::Type{<:T1MFA}` without a variable name means "an argument
# that is the type T1MFA or one of its subtypes, which we do not need to look at".
#
# The fallback, for a model (an instance): ask its type, as Base does for `eltype`
# (`eltype(x) = eltype(typeof(x))`), then compute the starts that depend on the acquisition for
# this model's, so `parameters(T1IR(TI = ...))` shows the BestOf values for those TIs. A model's
# own `parameters(m::MyModel)` method, if it has one, is more specific and wins.
#
# With keywords, the changes are applied as `setup` applies a fit's keywords (`_apply_roles`,
# fit/setup.jl). A model's own `parameters(m::MyModel)` takes no keywords, so a call with
# keywords comes here (Julia dispatches keyword calls to the methods that take keywords), and
# `parameters(m)` below then asks the model's own method.
function parameters(m::MRIModel; changes...)
    isempty(changes) && return map(r -> _at_acquisition(r, m), parameters(typeof(m)))
    roles = parameters(m)
    return _apply_roles(m, roles, changes, n -> ArgumentError("`$n` is not a parameter of " *
        "$(nameof(typeof(m))); the parameters are $(join(keys(roles), ", "))"))
end

# A model or a model type: what `nominal` and `defaults` accept.
const _ModelOrType = Union{MRIModel,Type{<:MRIModel}}

# The roles with each start computed from the acquisition (`m -> ...`) replaced by the start it
# gives for this model: for the fit's setup (setup.jl), the model's display and `nominal(m)`
# (again, for a model with its own `parameters(m::MyModel)` method). For a type there is no
# acquisition: the functions stay (a start not known yet).
_at_acquisition(r::Free, m) =
    r.start isa Function ? Free(start = r.start(m), lower = r.lower, upper = r.upper) : r
_at_acquisition(r, m) = r
_roles(m::MRIModel) = map(r -> _at_acquisition(r, m), parameters(m))
function _roles(M::Type{<:MRIModel})
    applicable(parameters, M) && return parameters(M)
    throw(ArgumentError("$(nameof(M)) defines its parameters for a model with its acquisition " *
        "(`parameters(m::$(nameof(M)))`), not for its type: give a model, $(nameof(M))(...)"))
end
_modelname(m::MRIModel) = nameof(typeof(m))
_modelname(M::Type{<:MRIModel}) = nameof(M)

"""
    signal!(out, m, p) -> out

Write one voxel's signal at every observation into `out` and return `out`.

- `m`: the model: its acquisition (e.g. `m.TI`, one value per observation) and options.
- `p`: every parameter by name (`p.T1`, `p.B1`, ...), free and fixed alike, as plain numbers.

Unpack both by name at the top: `(; TI, TD, magnitude) = m` and `(; PSR, R1f, ...) = p`. (Do
not rely on the ORDER of `p`: it depends on which parameters are free in a given fit.)

Loop with `for j in eachindex(out, TI, TD)`, listing every per-observation field: it checks
once that their lengths match `out` (an error otherwise), which also makes `@inbounds` safe.

Rules: no allocation, no state, and generic in the number type of `p`. The fitter calls it
with dual numbers (value + derivatives), so do not force values to `Float64`.

Called directly, nothing is checked or filled in (that would cost time in the fitter's inner
loop): a tie (`R1m = :R1f`) or a missing value fails inside the model. Complete and resolve the
values once with [`resolve`](@ref), then reuse them:

```julia
p = resolve(m, nominal(m, :SIR_WM_II))     # every value a number, R1m = R1f
out = zeros(nobs(m))
signal!(out, m, p)
```
"""
function signal! end
# The `!` is a Julia naming convention: the function modifies (one of) its arguments, here `out`.

"""
    datastart(m, y, p) -> nothing or (name = value, ...)

Optional: starting values computed from one voxel's data `y` (plain numbers, in the working
precision), e.g. a quick linearized fit, for the parameters that start at `:data`
(`T2 = Free(start = :data, ...)` in [`parameters`](@ref), or `T2 = Free(start = :data)` in a
fit). `p` holds the voxel's fixed parameters by name (numbers, e.g. `p.B1`, which a DESPOT1 estimate
needs; ties appear as ties). Return a finite value for each of them, with a fallback when the
data give no estimate (T2Decay: mean(TE) if the data do not decay). Values outside the bounds
are clamped; a NaN makes that voxel's fit fail (status `Failed`). A `:scale` parameter is then
set by least squares, and [`BestOf`](@ref) values are scored, from these starts. Must not
allocate: it runs once per voxel. The default, `nothing`, means no data start.

```julia
function datastart(m::T2Decay, y, p)
    R2 = _loglinear_rate(m.TE, y)                 # NaN if the data do not decay
    return (T2 = isnan(R2) ? mean(m.TE) : 1 / R2,)
end
```
"""
datastart(::MRIModel, y, p) = nothing

"""
    relabel(m, θ) -> θ

Optional: for models whose parameters can describe the same signal in more than one way by
swapping labels, choose the one to report. `θ` holds one voxel's fitted free parameters by name;
return the same names, in the same order. Called once per voxel after the fit; the signal, and
so the SSE, must not change. Example: the two components of [`T2BiexpDecay`](@ref) can swap,
`(fshort, T2short, T2)` ↔ `(1 - fshort, T2, T2short)`, so it puts the shorter T2 first.
The default returns `θ` unchanged.
"""
relabel(::MRIModel, θ) = θ

# -------------------------------------------------------------------------------------------------
# The model's acquisition: number of observations and working precision
# -------------------------------------------------------------------------------------------------

# The model's fields as a tuple, e.g. (TI vector, TD vector, magnitude) for QMTSIR.
# `fieldnames(typeof(m))` lists the field names; `getfield(m, f)` reads one.
_fields(m) = map(f -> getfield(m, f), fieldnames(typeof(m)))

"""
    nobs(m) -> Int

Number of observations: the common length of the model's vector fields (its per-observation
acquisition settings, e.g. `TI` and `TD`). Throws an error if they differ. A model with vector
fields of another kind can define its own method.
"""
function nobs(m::MRIModel)
    _checkfields(m)
    # Each field's length if it is a vector, -1 if not: a tuple (`map` over the tuple of fields,
    # each field's type known), so nothing is allocated. It is called for every noise level
    # crlb is given; collecting the lengths in an array cost 35–65 ns and 80–240 bytes a call.
    lens = map(x -> x isa AbstractVector ? length(x) : -1, _fields(m))
    i = findfirst(>=(0), lens)
    i === nothing && throw(ArgumentError("$(nameof(typeof(m))) has no per-observation (vector) " *
                                         "fields; define `nobs` for it"))
    n = lens[i]
    all(l -> l < 0 || l == n, lens) ||
        throw(DimensionMismatch("$(nameof(typeof(m))): the per-observation fields have " *
                                "different lengths, $(filter(>=(0), collect(lens)))"))
    return n
end
# A per-observation field holds numbers, one per observation: every generic routine relies on
# it (the lengths, the signal's number type, the conversion to the fit's precision). A vector of
# anything else (3-vectors as gradient directions, one per image) is said here, where every fit,
# signal and crlb starts, with what to write instead; otherwise it would fail later, obscurely
# (`float(Any)`), or stay Float64 in a Float32 fit. The test is on the fields' types, so the
# compiler settles it when compiling: nothing at run time. A matrix is not checked: it is not a
# per-observation field (a lookup table, say), and `nobs` does not count it.
@inline _checkfields(m) =
    any(x -> x isa AbstractVector && !(eltype(x) <: Number), _fields(m)) ? _badfield(m) : nothing
@noinline function _badfield(m)
    names = fieldnames(typeof(m))
    f = names[findfirst(f -> (x = getfield(m, f); x isa AbstractVector && !(eltype(x) <: Number)),
                        names)]
    T = eltype(getfield(m, f))
    throw(ArgumentError("$(nameof(typeof(m))): the field `$f` holds $T values; a field with " *
        "one value per observation must hold numbers. Gradient directions go in as three " *
        "fields of numbers, gx, gy and gz: `diffusionencoding(bval, bvec)` makes them (and b) " *
        "from a .bval and a .bvec, or from a 3×N matrix"))
end

# `nobs` is StatsAPI's function (imported in EQUIP.jl); this adds a method for every
# MRIModel. A specific model can add its own, more specific method, e.g.
# `nobs(m::T2Spectrum) = length(m.TE)`, and Julia will pick that one for it.
#
# `cond && expr` runs `expr` only if `cond` is true; `cond || expr` only if it is false. They are
# common shorthands for "if ... throw an error".

# The model with its numeric vectors converted to Vector{T}, so that a Float32 fit computes in
# Float32 (and ranges or integer vectors become plain float vectors). Every other field (e.g.
# `magnitude`) is kept. The struct is rebuilt with its positional constructor:
# `Base.typename(typeof(m)).wrapper` is the bare type, e.g. `QMTSIR` without the `{...}` part,
# so Julia picks the type parameters for the converted fields itself.
function _with_precision(m::MRIModel, ::Type{T}) where {T}
    return Base.typename(typeof(m)).wrapper(map(x -> _to_precision(x, T), _fields(m))...)
end
_to_precision(x::AbstractVector{<:Real}, ::Type{T}) where {T} = Vector{T}(x)
_to_precision(x::AbstractVector{Bool}, ::Type{T}) where {T} = x     # flags stay flags
_to_precision(x, ::Type{T}) where {T} = x
# `::Type{T}` with `where {T}` means "this argument is a type, call it T". When several methods
# match, Julia picks the most specific one: a Vector{Bool} matches the second method.
# `f(xs...)` "splats" a tuple: its elements become separate arguments.

# -------------------------------------------------------------------------------------------------
# Building the complete parameter set
# -------------------------------------------------------------------------------------------------

# A tie by name, `Fixed(:R1f)`, is turned into `_Tie{:R1f}()` before fitting. The name is a
# *type parameter* (like F in FitSpec), so the compiler knows which field to read: resolving the
# tie is a plain field access. (Reading a field whose name is only known at run time would be
# slow, because the compiler could not know its type.)
struct _Tie{S} end
_Tie(name::Symbol) = _Tie{name}()
Base.show(io::IO, ::_Tie{S}) where {S} = print(io, "Fixed(:", S, ")")

# Ties are resolved with the parameter set itself:
# `(R1f = 1.2, R1m = _Tie{:R1f}())` and `(R1f = 1.2, R1m = p -> p.R1f)` both become
# `(R1f = 1.2, R1m = 1.2)`. Julia picks the `_resolve_tie` method by the type of `x`.
@inline _resolve_tie(::_Tie{S}, p) where {S} = getfield(p, S)   # tie by name: read field S
@inline _resolve_tie(x::Function, p) = x(p)                     # tie by function: call it
@inline _resolve_tie(x, p) = x                                  # not a tie: keep it
@inline _resolve_ties(p::NamedTuple) = map(x -> _resolve_tie(x, p), p)
# `map(f, namedtuple)` applies f to every value and keeps the names. Because the field types are
# known to the compiler, this compiles to plain field reads: no loop, no allocation.

# Ties by name are turned into `_Tie`s; everything else is kept as it is.
_tie(x::Symbol) = _Tie(x)
_tie(x) = x

_istie(x) = x isa Union{Symbol,_Tie,Function}

# Back to the form a user writes, for results and display: `_Tie{:R1f}()` → `:R1f`.
_untie(::_Tie{S}) where {S} = S
_untie(x) = x

# Check the ties among the values `vals` (a named tuple of parameter values or fixed values):
# a tie by name must name another parameter that is not itself a tie. (Function ties cannot be
# checked before they run.) One `_check_tie` call per value, by `map` over the names and the
# values: the values have different types (numbers, a Symbol), and `map` over a named tuple is
# compiled for each, where a `for` loop would look each one's type up at run time (0.8 µs for
# QMTSIR; checked on every truth a crlb sweep meets).
function _check_ties(vals::NamedTuple{K}) where {K}
    map((n, v) -> _check_tie(n, v, vals), NamedTuple{K}(K), vals)
    return nothing
end
function _check_tie(n, v::Symbol, vals)
    v === n && throw(ArgumentError("`$n` cannot be tied to itself"))
    haskey(vals, v) ||
        throw(ArgumentError("`$n` is tied to `$v`, which is not a parameter; the parameters are $(keys(vals))"))
    _istie(vals[v]) &&
        throw(ArgumentError("`$n` is tied to `$v`, which is itself a tie; tie `$n` to what `$v` is tied to"))
    return nothing
end
_check_tie(n, v, vals) = nothing          # not a tie by name

function _check_names(kind, given, known)
    for n in given
        n in known || throw(ArgumentError("unknown parameter `$n` in `$kind`; the parameters are $known"))
    end
end

# Roles are not values: `parameters(m)` says how each parameter is fitted (start, bounds), so
# passing it, or a `Free(...)`/`Fixed(...)`, where values belong is an easy mistake that would
# otherwise fail deep inside the model.
function _reject_roles(p::NamedTuple)
    any(values(map(x -> x isa Union{Free,Fixed}, p))) || return nothing   # the usual case, quickly
    for (n, x) in pairs(p)
        x isa Union{Free,Fixed} || continue
        hint = x isa Fixed && x.value isa Real ? "write `$n = $(x.value)`" : "give a number or a map"
        throw(ArgumentError("`$n` is a role, $(repr(x)), not a value: $hint. `signal` and " *
            "`simulate` take values; `parameters(m)` says how each parameter is fitted, " *
            "`nominal(m)` gives the model's own values and `nominal(m, tissue)` a tissue's"))
    end
end

"""
    resolve(m, p) -> named tuple

Every parameter of `m` as a plain value, in the model's order: `p` must give them all (numbers,
or ties), and ties are resolved (`R1m = :R1f` becomes R1f's value; a function tie is called
with the other values). Checked: every name is a parameter of `m`, none is missing (the error
lists them all), no value is a role (`Free`, `Fixed`), and every tie names a parameter that is
not itself a tie. [`signal`](@ref) does this before calling [`signal!`](@ref); call it yourself
to use `signal!` directly, e.g. in a loop with one output buffer.

```julia
m = QMTSIR(; SIR_4POINT...)
resolve(m, nominal(m))       # (PSR = 0.1, R1f = 1.0, ..., R1m = 1.0, ...)
```
"""
function resolve(m::MRIModel, p::NamedTuple)
    q = _complete(nameof(typeof(m)), parameters(m), p)   # checked; model order, ties kept
    return _resolve_ties(map(_tie, q))
end

# The checks of `resolve`, for model `M` (its name) with the roles `roles`: `p` in the model's
# order, as given (ties kept). Also used by `Scenario` (errors/scenario.jl), which knows the
# model's type but not its acquisition. The names K come from the roles' type (`where {K}`), so
# the compiler knows them and the reordering is a plain copy.
function _complete(M::Symbol, roles::NamedTuple{K}, p::NamedTuple) where {K}
    _check_names("p", keys(p), K)
    _reject_roles(p)
    # A switchable parameter (`Fixed(0.0; free = ...)`, a part off) needs no value: filled in at its own.
    if !all(n -> haskey(p, n) || _switchable(roles[n]), K)
        missing = [n for n in K if !haskey(p, n) && !_switchable(roles[n])]
        throw(ArgumentError("$M needs a value for every parameter; missing: " *
            "$(join(missing, ", ")). For a complete set, start from `nominal(m)` (the model's " *
            "starting and fixed values) or `nominal(m, tissue)`, and change values by keyword: " *
            "`nominal(m; $(first(missing)) = ...)`"))
    end
    q = NamedTuple{K}(map(n -> haskey(p, n) ? p[n] : roles[n].value, K))   # every parameter, in model order
    _check_ties(q)                       # e.g. R1m = :R1f must name a parameter that is not a tie
    return q
end

"""
    defaults(m) -> named tuple

The values of the parameters that [`parameters`](@ref)`(m)` fixes by default: the model's fitting
assumptions, e.g. for QMTSIR `(kmf = 12.5, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)`
(`R1m = :R1f`: a tie, R1m equals R1f). Empty for a model with none. `m` may be a model or its
type (`defaults(QMTSIR)`).

Only the fixed parameters: for a complete set to simulate with, use [`nominal`](@ref)`(m)` (these
plus the free parameters' starts) or `nominal(m, tissue)`. Or merge them into your own values of
the free parameters (later values win):

```julia
simulate(m, merge(defaults(m), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0)); noise = :rician, snr = 200)
```
"""
defaults(m::_ModelOrType) = (; (n => r.value for (n, r) in pairs(_roles(m)) if r isa Fixed)...)

# The name of the parameter the model declares `Free(start = :scale)` (M0, S0, M0f), or nothing:
# from a model, or from its own roles. Used by the fit's checks (`_check_scale`, fit/setup.jl)
# and by the noise level: `snr` is relative to this parameter (sim/noise.jl).
_scalename(m::MRIModel) = _scalename(parameters(m))
function _scalename(roles::NamedTuple)
    i = findfirst(r -> r isa Free && r.start === :scale, values(roles))
    return i === nothing ? nothing : keys(roles)[i]
end

# The value a role gives `nominal` (sim/tissues.jl) when neither a keyword nor a tissue does:
# a fixed value (a tie kept as written), 1 for the scale parameter, and a number start if
# `starts` (no tissue given); `nothing` when there is none.
_model_value(r::Fixed, starts) = r.value
_model_value(r::Free, starts) =
    r.start === :scale ? 1.0 : starts && r.start isa Real ? r.start : nothing

"""
    signal(m, p) -> Vector

Simulate one voxel's signal. Allocating version of [`signal!`](@ref), for simulation and
plotting. `p` gives a value for every parameter of the model: a number, or a tie to another
parameter (`R1m = :R1f`, or a function of the others); ties are resolved (see
[`resolve`](@ref)). Nothing is filled in silently: [`nominal`](@ref)`(m)` gives every value from
the model (its starts and fixed values), `nominal(m, tissue)` from a tissue, and
[`defaults`](@ref)`(m)` the fixed ones alone. Roles (`Free`, `Fixed`) are not values: passing
`parameters(m)` itself is an error.

```julia
m = QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)
signal(m, nominal(m))                               # the model's starting and fixed values
signal(m, nominal(m; PSR = 0.15, kmf = 10.0))       # a couple changed
signal(m, nominal(m, :SIR_WM_II))                   # every value from a tissue
signal(m, merge(defaults(m), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0)))  # your free values
```
"""
function signal(m::MRIModel, p::NamedTuple)
    q = resolve(m, p)
    n = nobs(m)                                    # also checks the acquisition's lengths
    return signal!(Vector{_signaltype(m, q)}(undef, n), m, q)
end

# Element type of the signal for values q (resolved: numbers): wide enough for every parameter
# and acquisition value (e.g. Float64; or a dual number if someone differentiates through
# `signal`). Shared with `simulate`, which writes into one buffer of this type.
function _signaltype(m::MRIModel, q::NamedTuple)
    vectors = filter(x -> x isa AbstractVector, _fields(m))
    return float(promote_type(map(typeof, values(q))..., map(eltype, vectors)...))
end

# -------------------------------------------------------------------------------------------------
# How a model prints at the REPL
# -------------------------------------------------------------------------------------------------

# Typing a model's name at the REPL (or `display(m)`) shows its acquisition and options, and a
# table of its parameters: role, start or fixed value, bounds, and the value `nominal(m)` takes.
# For that multi-line display Julia calls `show(io, MIME"text/plain"(), x)`; the one-line
# `show(io, m)` (inside a vector, in string interpolation) keeps Julia's default, the
# constructor call. One method for every model, built-in or not:
#
#   QMTSIR with 4 observations
#     TI = [0.01, 0.01, 0.278, 1.007]
#     TD = [0.684, 4.171, 2.73, 0.01]
#     magnitude = true
#   Parameters:
#     name     role   start/value  bounds        nominal(m)
#     PSR      free   0.1          [0.0, 1.0]    0.1
#     M0f      free   :scale       [0.0, Inf]    1.0
#     R1m      fixed  :R1f (tie)                 :R1f
#     ...
function Base.show(io::IO, ::MIME"text/plain", m::MRIModel)
    n = try
        nobs(m)
    catch                                  # vector fields of different lengths: no count
        nothing
    end
    print(io, nameof(typeof(m)), n === nothing ? "" : " with $n observation$(n == 1 ? "" : "s")")
    for f in fieldnames(typeof(m))
        print(io, "\n  ", f, " = ", _acqvalue(getfield(m, f)))
    end
    roles = try
        _roles(m)
    catch e                                # a new model, before its `parameters` method
        e isa MethodError && e.f === parameters || rethrow()
        return
    end
    vals = map(r -> _model_value(r, true), roles)     # what nominal(m) takes
    print(io, "\nParameters:\n")
    _print_parameters(io, roles, vals, "nominal(m)")
    need = [k for (k, v) in pairs(vals) if v === nothing]
    give = join(("$n = value" for n in need), ", ")
    isempty(need) || print(io, "\n`nominal(m)` needs a value for ", join(need, ", "),
                           ": `nominal(m; ", give, ")`")
end

# An acquisition field, short: a range as written, one value repeated as "2.5, every
# observation", a list of up to 8 values in full, a longer one as its count and extent (like
# BestOf). Numbers to 6 significant digits (`:compact`). Anything else as `_showvalue` does.
function _acqvalue(x::AbstractVector{<:Real})
    c(v) = repr(v; context = :compact => true)
    x isa AbstractRange && return c(x)
    length(x) > 1 && all(==(first(x)), x) && return c(first(x)) * ", every observation"
    length(x) <= 8 && return "[" * join(map(c, x), ", ") * "]"
    return "$(length(x)) values from $(c(minimum(x))) to $(c(maximum(x)))"
end
_acqvalue(x) = _showvalue(x)

# The parameter table, for the model's display and for `nominal`'s error: one row per parameter
# with its role, its start (free) or value (fixed), its bounds (a bound left out prints as
# ±Inf, as the fit uses it), and `vals[n]`, the value `nominal` takes (`nothing`: required), in a
# column headed `label` (`nominal(m)`, or `nominal(m, :WM)` with a tissue).
function _print_parameters(io::IO, roles::NamedTuple, vals::NamedTuple, label)
    cell(x) = x isa Real ? repr(x; context = :compact => true) : _showvalue(x)
    setting(r::Free) = r.start === nothing ? "" : cell(r.start)
    setting(r::Fixed) = _istie(r.value) ? cell(r.value) * " (tie)" : cell(r.value)
    bounds(r::Free) = "[" * cell(something(r.lower, -Inf)) * ", " *
                      cell(something(r.upper, Inf)) * "]"
    bounds(r::Fixed) = ""
    rows = [["name", "role", "start/value", "bounds", label]]
    for (n, r) in pairs(roles)
        v = vals[n]
        push!(rows, [string(n), r isa Free ? "free" : _isoff(r) ? "off" : "fixed", setting(r), bounds(r),
                     v === nothing ? "required" : cell(v)])
    end
    widths = [maximum(row -> textwidth(row[j]), rows) for j in 1:5]   # each column's width
    print(io, join((rstrip("  " * join(rpad.(row, widths), "  ")) for row in rows), "\n"))
end
