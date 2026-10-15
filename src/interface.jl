# =================================================================================================
# The model interface
#
# Everything a model author needs: the supertype `AbstractModel`, the two parameter roles
# `Free` and `Fixed`, and the two functions every model defines, `parameters` and `signal!`.
# Plus helpers for the acquisition (`obs`, `nobs`) and the simulation wrapper `signal`.
#
# Three separate inputs reach a signal equation, and each has its own home:
#
#   model  m    WHICH physics, plus options       a struct:      SIRFSE(magnitude = true)
#   acq         HOW the experiment was run        a NamedTuple:  (TI = [...], TD = [...])
#   p           the tissue/model parameter VALUES a NamedTuple:  (PSR = 0.1, R1f = 1.0, ...)
#
# The model never stores parameter values: the same model object is reused for millions of
# voxels, while `p` changes at every step of every fit.
# =================================================================================================

"""
    AbstractModel

Supertype of all signal models. A model is three pieces:

1. A struct `<: AbstractModel` holding options that are the same for every voxel
   (e.g. `magnitude::Bool`). Many models have no options and so no fields.
2. [`parameters`](@ref)`(m)`: every parameter of the signal equation with its default role,
   [`Free`](@ref) or [`Fixed`](@ref), as a named tuple.
3. [`signal!`](@ref)`(out, m, acq, p)`: one voxel's signal at every observation.

Example (see `src/models.jl` for more):

```julia
struct ExpDecay <: AbstractModel end
parameters(::ExpDecay) = (S0 = Free(1.0; lower = 0), T2 = Free(0.05; lower = 1e-6))
function signal!(out, ::ExpDecay, acq, p)
    for j in eachindex(out)
        (; TE) = obs(acq, j)
        out[j] = p.S0 * exp(-TE / p.T2)
    end
    return out
end
```
"""
abstract type AbstractModel end
# `abstract type` declares a type that has no instances of its own; it only groups concrete
# types. `struct ExpDecay <: AbstractModel` says "ExpDecay is a kind of AbstractModel", so any
# function written for `m::AbstractModel` (like `signal` and `fit_maps`) works for it.

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
- a tie, a function of the other parameters: `Fixed(p -> p.R1f)`. A tie may use free or
  fixed parameters, but not another tie.
"""
struct Fixed{T}
    value::T
end
# `Fixed{T}` is a *parametric* type: T is the type of `value`, so `Fixed(12.5)` is a
# `Fixed{Float64}`, `Fixed(b1map)` a `Fixed{Matrix{Float64}}`, `Fixed(p -> p.R1f)` a
# `Fixed{var"#1#2"}` (every anonymous function has its own type). Keeping the exact type lets
# the compiler generate fast, specialized code for each case.

"""
    parameters(m) -> NamedTuple

Every parameter of model `m` with its default role, [`Free`](@ref) or [`Fixed`](@ref). The order
is the order of the outputs. A fit can override the roles with the `free` and `fixed` keywords.

```julia
parameters(::SaturationRecovery) = (M0 = Free(1.0; lower = 0), T1 = Free(1.0; lower = 1e-6))
```
"""
function parameters end
# `function name end` declares a generic function with no methods yet. Each model adds its own
# method, e.g. `parameters(::MFA) = ...`; Julia picks the method by the model's type
# ("multiple dispatch"). `::MFA` without a variable name means "an argument of type MFA that
# we do not need to look at".

"""
    signal!(out, m, acq, p) -> out

Write one voxel's signal at every observation into `out` and return `out`.

- `m`: the model (its options, e.g. `m.magnitude`).
- `acq`: the acquisition, a named tuple. Vector fields have one value per observation, scalar
  fields are the same for all; `(; TI, TD) = obs(acq, j)` gives observation `j`'s values.
- `p`: every parameter by name (`p.T1`, `p.B1`, ...), free and fixed alike, as plain numbers.

Rules: no allocation, no state, and generic in the number type of `p`. The fitter calls it
with dual numbers (value + derivatives), so do not force values to `Float64`.
"""
function signal! end
# The `!` is a Julia naming convention: the function modifies (one of) its arguments, here `out`.

# -------------------------------------------------------------------------------------------------
# The acquisition
# -------------------------------------------------------------------------------------------------

"""
    obs(acq, j) -> NamedTuple

The acquisition settings of observation `j`: vector fields are indexed, scalars passed through.

```julia
acq = (FA = [2, 5, 10], TR = 0.015)
obs(acq, 2)        # (FA = 5, TR = 0.015)
```
"""
@inline obs(acq::NamedTuple, j) = map(v -> v isa AbstractVector ? (@inbounds v[j]) : v, acq)
# `map(f, namedtuple)` applies f to every value and keeps the names. Because the field types are
# known to the compiler, this compiles to plain field reads: no loop, no allocation.
# `@inline` asks the compiler to paste the body into the caller (it is tiny and called in the
# innermost loop). `@inbounds` skips the index bounds check (safe: `nobs` checked the lengths).

"""
    nobs(acq) -> Int

Number of observations: the common length of the vector fields of `acq`.
"""
function nobs(acq::NamedTuple)
    lens = [length(v) for v in values(acq) if v isa AbstractVector]
    isempty(lens) && throw(ArgumentError("the acquisition has no per-observation (vector) settings"))
    all(==(lens[1]), lens) ||
        throw(DimensionMismatch("acquisition vectors have different lengths: $lens"))
    return lens[1]
end
# `cond && expr` runs `expr` only if `cond` is true; `cond || expr` only if it is false. They are
# common shorthands for "if ... throw an error".

# Convert the acquisition's numbers to the working precision T (Float64 or Float32), so the
# signal is computed in one precision. Vectors are copied; anything else is passed through.
_convert_acq(acq::NamedTuple, ::Type{T}) where {T} =
    map(v -> v isa AbstractVector{<:Real} ? Vector{T}(v) : v isa Real ? T(v) : v, acq)
# `::Type{T}` with `where {T}` means "this argument is a type, call it T", so `T(v)` converts.

# -------------------------------------------------------------------------------------------------
# Building the complete parameter set
# -------------------------------------------------------------------------------------------------

# Ties are resolved by calling the function with the parameter set itself:
# `(R1f = 1.2, R1m = p -> p.R1f)` becomes `(R1f = 1.2, R1m = 1.2)`.
# Two methods of `_resolve1`: Julia picks the first when `x` is a Function, the second otherwise.
@inline _resolve1(x::Function, p) = x(p)
@inline _resolve1(x, p) = x
@inline _resolve(p::NamedTuple) = map(x -> _resolve1(x, p), p)

function _check_names(kind, given, known)
    for n in given
        n in known || throw(ArgumentError("unknown parameter `$n` in `$kind`; the parameters are $known"))
    end
end

# Every parameter of `m`, in model order: values given in `p`, the `Fixed` defaults of
# `parameters(m)` for the rest, ties resolved. Used by `signal` (simulation), not by the fitter.
function _allparams(m::AbstractModel, p::NamedTuple)
    roles = parameters(m)
    _check_names("p", keys(p), keys(roles))
    vals = map(keys(roles)) do n          # `do n ... end` is the body of an anonymous function n -> ...
        haskey(p, n) ? p[n] :
        roles[n] isa Fixed ? roles[n].value :
        throw(ArgumentError("`p` must give `$n`: it is Free and has no fixed default"))
    end
    return _resolve(NamedTuple{keys(roles)}(vals))
end

"""
    signal(m, acq, p) -> Vector

Simulate one voxel's signal. Allocating version of [`signal!`](@ref), for simulation and
plotting. `p` must give the Free parameters; Fixed ones default to their values in
[`parameters`](@ref)`(m)` and can be overridden; ties are resolved.

```julia
signal(SIRFSE(), (TI = TI, TD = TD), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0))
signal(SIRFSE(), (TI = TI, TD = TD), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1.0, kmf = 15.0))
```
"""
function signal(m::AbstractModel, acq::NamedTuple, p::NamedTuple)
    q = _allparams(m, p)
    # Element type of the output: wide enough for every parameter and acquisition value
    # (e.g. Float64; or a dual number if someone differentiates through `signal`).
    T = float(promote_type(map(typeof, values(q))..., map(eltype, values(acq))...))
    return signal!(Vector{T}(undef, nobs(acq)), m, acq, q)
end
