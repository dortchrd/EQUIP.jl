# =================================================================================================
# The model interface
#
# Everything a model author needs: the supertype `MRIModel`, the two parameter roles `Free` and
# `Fixed`, and the two functions every model defines, `parameters` and `signal!`. Plus `nobs`
# (number of observations) and the simulation wrapper `signal`.
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
    Free(init; lower = -Inf, upper = Inf)

Role of a parameter that is estimated: its starting value `init` and its bounds.

`init` is a number, or `:scale` for a parameter that multiplies the whole signal (S0, M0):
it then starts, in every voxel, at the least-squares value given the other starting values,
so the data need no normalization.

```julia
Free(0.1; lower = 0, upper = 1)
Free(:scale; lower = 0)
```
"""
struct Free
    init::Union{Float64,Symbol}    # a number, or :scale
    lower::Float64
    upper::Float64
end
# Julia automatically defines the "default constructor" `Free(init, lower, upper)` (all three
# fields, positional). The method below adds a friendlier one with keywords and defaults.
# Different number of positional arguments -> different method, so the two do not clash. It
# checks `init`, converts numbers to Float64, and calls the default constructor.
function Free(init; lower = -Inf, upper = Inf)
    init isa Real || init === :scale ||
        throw(ArgumentError("Free(init): init must be a number or :scale, got $(repr(init))"))
    return Free(init isa Real ? Float64(init) : init, lower, upper)
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

"""
    parameters(m) -> NamedTuple

Every parameter of model `m` with its default role, [`Free`](@ref) or [`Fixed`](@ref). The order
is the order of the outputs. A fit can override the roles with the `free` and `fixed` keywords.

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
    startgrid(m) -> nothing or (name = values,)

Optional: a grid of starting values to try for ONE free parameter, e.g. `(T1 = values,)`. For
each voxel, the fit evaluates the signal at every grid value (the other parameters at their
starts, a `Free(:scale)` parameter at its least-squares value for that grid value) and starts
from the one that fits the data best. Use it for models whose fits can end in a wrong local
minimum from a single start, such as magnitude inversion recovery. The grid is skipped when
the parameter is fixed or given a `start`. The default, `nothing`, means no grid.

```julia
startgrid(m::T1IR) = (T1 = exp.(range(log(0.01), log(10.0); length = 30)),)
```
"""
startgrid(::MRIModel) = nothing

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

# Every parameter of `m`, in model order: values given in `p`, the `Fixed` defaults of
# `parameters(m)` for the rest, ties resolved. Used by `signal` (simulation), not by the fitter.
function _allparams(m::MRIModel, p::NamedTuple)
    roles = parameters(m)
    _check_names("p", keys(p), keys(roles))
    vals = map(keys(roles)) do n          # `do n ... end` is the body of an anonymous function n -> ...
        haskey(p, n) ? p[n] :
        roles[n] isa Fixed ? roles[n].value :
        throw(ArgumentError("`p` must give `$n`: it is Free and has no fixed default"))
    end
    q = NamedTuple{keys(roles)}(vals)
    _check_ties(q)                       # e.g. R1m = :R1f must name a parameter that is not a tie
    return _resolve_ties(map(_tie, q))
end

"""
    signal(m, p) -> Vector

Simulate one voxel's signal. Allocating version of [`signal!`](@ref), for simulation and
plotting. `p` must give the Free parameters; Fixed ones default to their values in
[`parameters`](@ref)`(m)` and can be overridden; ties are resolved.

```julia
m = QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)
signal(m, (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0))
signal(m, (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0, kmf = 15.0))   # override a fixed one
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
