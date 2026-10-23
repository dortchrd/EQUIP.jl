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

parameters(::MyDecay) = (S0 = Free(:scale; lower = 0), T2 = Free(0.05; lower = 1e-6))

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
    Free(start; lower, upper)
    Free(; start, lower, upper)

Role of a parameter that is estimated: its starting value and its bounds.

In a model's [`parameters`](@ref), the defaults: `Free(0.1; lower = 0, upper = 1)`. A bound left
out means none. The start says where each voxel's starting value comes from:

- a number, or a map (one value per voxel, in a fit): start there;
- `:scale`, for the parameter that multiplies the whole signal (S0, M0): the least-squares
  value given the other starts, so the data need no normalization. Only the model declares
  it: the rule assumes the signal is proportional to the parameter (true for M0, not for T1).
  A fit can turn it off (`M0 = Free(start = 1000.0)`) but cannot move it to another parameter;
- `:data`: the model's own estimate from the voxel's data (see [`datastart`](@ref)), e.g. a
  log-linear fit for T2. Only for parameters the model's `datastart` estimates;
- [`BestOf`](@ref)`(values)`: try each value and start from the one that fits best.

In a fit, to estimate a parameter or change its start or bounds; whatever is left out keeps the
model's default:

```julia
fit(m, data; kmf = Free())                 # estimate kmf, fixed by default (starts at its value)
fit(m, data; R1f = Free(start = 0.8))      # another start: a number, or a map (one per voxel)
fit(m, data; T1 = Free(start = BestOf([0.3, 0.8, 1.5])))   # try three values, start from the best
fit(m, data; Sf = Free(lower = -1.5))      # another lower bound
```
"""
struct Free
    start::Any                         # a number, a map, :scale, :data, a BestOf, or nothing (keep the default)
    lower::Union{Nothing,Float64}      # nothing: keep the default (no bound in a model's parameters)
    upper::Union{Nothing,Float64}
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
parameters(m::T1IR) = (..., T1 = Free(BestOf(_t1grid(m.TI)); lower = _TMIN))   # the model's default
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
# How roles print, e.g. in `parameters(m)`: only what is set, by name, so
# `Free(start = :scale, lower = 0.0)` rather than the positional `Free(:scale, 0.0, nothing)`.
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

# Julia automatically defines the "default constructor" `Free(start, lower, upper)` (all three
# fields, positional). The method below adds the friendlier forms: `Free()`, `Free(0.8)`,
# `Free(start = 0.8)`, `Free(0.8; lower = 0)`. The keyword `start` defaults to the positional one,
# so either spelling works. It checks the start and converts numbers to Float64.
function Free(s = nothing; start = s, lower = nothing, upper = nothing)
    s !== nothing && start !== s && throw(ArgumentError("give the start once: Free(0.8) or Free(start = 0.8)"))
    start === nothing || start === :scale || start === :data || start isa Real ||
        start isa AbstractArray || start isa BestOf ||
        throw(ArgumentError("Free: the start must be a number, a map, :scale, :data or BestOf(values), got $(repr(start))"))
    return Free(start isa Real ? Float64(start) : start, lower === nothing ? nothing : Float64(lower),
                upper === nothing ? nothing : Float64(upper))
end

"""
    Fixed(value)

Role of a parameter that is not estimated. `value` is

- a number, the same for every voxel: `Fixed(12.5)`
- a per-voxel map, an array with one value per voxel (e.g. a B1 map): `Fixed(b1map)`
- a tie to another parameter, given by its name: `Fixed(:R1f)` ("always equal to R1f")
- a tie given by a function of the other parameters: `Fixed(p -> 0.8 * p.R1f)`

A tie may use free or fixed parameters (including per-voxel maps), but not another tie.
"""
struct Fixed{T}
    value::T
end
# `Fixed{T}` is a *parametric* type: T is the type of `value`, so `Fixed(12.5)` is a
# `Fixed{Float64}`, `Fixed(b1map)` a `Fixed{Matrix{Float64}}`, `Fixed(:R1f)` a `Fixed{Symbol}`
# and `Fixed(p -> 0.8 * p.R1f)` a `Fixed{var"#1#2"}` (every anonymous function has its own type).
# Keeping the exact type lets the compiler generate fast, specialized code for each case.
Base.show(io::IO, f::Fixed) = print(io, "Fixed(", _showvalue(f.value), ")")   # e.g. Fixed(12.5), Fixed(:R1f)

"""
    parameters(m) -> NamedTuple

Every parameter of model `m` with its default role, [`Free`](@ref) or [`Fixed`](@ref). The order
is the order of the outputs. A fit can change any role with a keyword named after the parameter:
`fit(m, data; kmf = 15.0)` (fixed at a value), `fit(m, data; kmf = Free())` (estimated).

```julia
parameters(::T1SR) = (M0 = Free(:scale; lower = 0), T1 = Free(1.0; lower = 1e-6))
```
"""
function parameters end
# `function name end` declares a generic function with no methods yet. Each model adds its own
# method, e.g. `parameters(::T1MFA) = ...`; Julia picks the method by the model's type
# ("multiple dispatch"). `::T1MFA` without a variable name means "an argument of type T1MFA that
# we do not need to look at".

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
"""
function signal! end
# The `!` is a Julia naming convention: the function modifies (one of) its arguments, here `out`.

"""
    datastart(m, y, p) -> nothing or (name = value, ...)

Optional: starting values computed from one voxel's data `y` (plain numbers, in the working
precision), e.g. a quick linearized fit, for the parameters that start at `:data`
(`T2 = Free(:data; ...)` in [`parameters`](@ref), or `T2 = Free(start = :data)` in a fit). `p`
holds the voxel's fixed parameters by name (numbers, e.g. `p.B1`, which a DESPOT1 estimate
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
`(fshort, T2short, T2long)` ↔ `(1 - fshort, T2long, T2short)`, so it puts the shorter T2 first.
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
    lens = [length(x) for x in _fields(m) if x isa AbstractVector]
    name = nameof(typeof(m))
    isempty(lens) && throw(ArgumentError("$name has no per-observation (vector) fields; define `nobs` for it"))
    all(==(lens[1]), lens) ||
        throw(DimensionMismatch("$name: the per-observation fields have different lengths, $lens"))
    return lens[1]
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

# Check the ties among the values `vals` (a named tuple of parameter values or fixed values):
# a tie by name must name another parameter that is not itself a tie. (Function ties cannot be
# checked before they run.)
function _check_ties(vals::NamedTuple)
    for (n, v) in pairs(vals)
        v isa Symbol || continue
        v === n && throw(ArgumentError("`$n` cannot be tied to itself"))
        haskey(vals, v) ||
            throw(ArgumentError("`$n` is tied to `$v`, which is not a parameter; the parameters are $(keys(vals))"))
        _istie(vals[v]) &&
            throw(ArgumentError("`$n` is tied to `$v`, which is itself a tie; tie `$n` to what `$v` is tied to"))
    end
end

function _check_names(kind, given, known)
    for n in given
        n in known || throw(ArgumentError("unknown parameter `$n` in `$kind`; the parameters are $known"))
    end
end

# Every parameter of `m`, in model order, from `p`, which must give them all (no silent defaults:
# a simulation's truth should be visible in the script), ties resolved. Used by `signal`
# (simulation), not by the fitter.
function _allparams(m::MRIModel, p::NamedTuple)
    roles = parameters(m)
    _check_names("p", keys(p), keys(roles))
    # Roles are not values: `parameters(m)` says how each parameter is fitted (start, bounds), so
    # passing it here is an easy mistake that would otherwise fail deep inside the model.
    for (n, x) in pairs(p)
        x isa Union{Free,Fixed} || continue
        hint = x isa Fixed && x.value isa Real ? "write `$n = $(x.value)`" : "give a number or a map"
        throw(ArgumentError("`$n` is a role, $(repr(x)), not a value: $hint. `signal` and " *
            "`simulate` take values; `parameters(m)` says how each parameter is fitted, and " *
            "`truth(m, tissue)` gives values for a tissue"))
    end
    missing = Tuple(n for n in keys(roles) if !haskey(p, n))
    if !isempty(missing)
        M = nameof(typeof(m))
        d = defaults(m)
        hint = isempty(d) ? "" : " The model's fitting defaults are `defaults(m)` = $d; to simulate " *
            "with them, merge them in: `merge(defaults(m), p)`."
        throw(ArgumentError("$M needs a value for every parameter; missing: $(join(missing, ", ")).$hint"))
    end
    q = NamedTuple{keys(roles)}(map(n -> p[n], keys(roles)))   # every parameter, in model order
    _check_ties(q)                       # e.g. R1m = :R1f must name a parameter that is not a tie
    return _resolve_ties(map(_tie, q))
end

"""
    defaults(m) -> named tuple

The values of the parameters that [`parameters`](@ref)`(m)` fixes by default: the model's fitting
assumptions, e.g. for QMTSIR `(kmf = 12.5, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)`
(`R1m = :R1f`: a tie, R1m equals R1f). Empty for a model with none.

[`signal`](@ref) and [`simulate`](@ref) need every parameter's value; to use these, merge them in,
so the choice is visible in the script (later values win):

```julia
simulate(m, merge(defaults(m), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0)); snr = 200)
```
"""
defaults(m::MRIModel) = (; (n => r.value for (n, r) in pairs(parameters(m)) if r isa Fixed)...)

"""
    signal(m, p) -> Vector

Simulate one voxel's signal. Allocating version of [`signal!`](@ref), for simulation and
plotting. `p` gives a value for every parameter of the model: a number, or a tie to another
parameter (`R1m = :R1f`, or a function of the others); ties are resolved. Nothing is filled in
silently: the model's fitting defaults (QMTSIR's `kmf = 12.5`, ...) are [`defaults`](@ref)`(m)`,
merged in explicitly, and [`truth`](@ref) gives every value for a tissue. Roles (`Free`,
`Fixed`) are not values: passing `parameters(m)` itself is an error.

```julia
m = QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)
θ = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0)
signal(m, merge(defaults(m), θ))                  # with the model's defaults for the rest
signal(m, merge(defaults(m), θ, (kmf = 15.0,)))   # one of them changed
signal(m, truth(m, :SIR_WM_II))                   # every value from a tissue
```
"""
function signal(m::MRIModel, p::NamedTuple)
    q = _allparams(m, p)
    n = nobs(m)                                    # also checks the acquisition's lengths
    # Element type of the output: wide enough for every parameter and acquisition value
    # (e.g. Float64; or a dual number if someone differentiates through `signal`).
    vectors = filter(x -> x isa AbstractVector, _fields(m))
    T = float(promote_type(map(typeof, values(q))..., map(eltype, vectors)...))
    return signal!(Vector{T}(undef, n), m, q)
end
