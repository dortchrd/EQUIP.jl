# Model interface: a struct (settings), `parameters` (roles) and `signal!` (the equation).

"""
    AbstractModel

Supertype of signal models. A model is defined by three pieces:

1. A struct holding settings that are the same for every voxel (e.g. `magnitude::Bool`).
2. [`parameters`](@ref)`(m)`: every parameter of the signal equation with its default role,
   [`Estimate`](@ref) or [`Fixed`](@ref), as a named tuple.
3. [`signal!`](@ref)`(out, m, acq, p)`: one voxel's signal at every observation.

Optional: [`initial_guess`](@ref) (data-driven starting values for `Estimate(:auto)`
parameters) and [`protocol`](@ref) (names of the per-observation settings).
"""
abstract type AbstractModel end

"""
    Estimate(init = :auto; lower = -Inf, upper = Inf)

Role of a parameter that is estimated. `init` is the starting value, or `:auto` to take it
from the model's [`initial_guess`](@ref).
"""
struct Estimate{T}
    init::T
    lower::Float64
    upper::Float64
end
Estimate(init=:auto; lower=-Inf, upper=Inf) = Estimate(init, Float64(lower), Float64(upper))

"""
    Fixed(value)

Role of a parameter that is not estimated. `value` is a number (same for every voxel), an
array over the voxel grid (a per-voxel map, e.g. B1), or a function of the other parameters
(a tie, e.g. `p -> p.R1f`). A tie may refer to estimated or fixed parameters, but not to
another tie.
"""
struct Fixed{T}
    value::T
end

"""
    parameters(m) -> NamedTuple

Every parameter of model `m` with its default role ([`Estimate`](@ref) or [`Fixed`](@ref)).
Roles can be overridden per fit with the `free` and `fixed` keywords.
"""
function parameters end

"""
    protocol(m) -> Tuple of Symbols

Names of the acquisition settings `m` reads from `acq`, e.g. `(:TE,)` or `(:FA, :TR)`. They are
checked before fitting. When a model has exactly one, a plain vector can be passed instead of
a named tuple.
"""
protocol(::AbstractModel) = ()

"""
    signal!(out, m, acq, p) -> out

Write the signal of one voxel at every observation into `out`. `acq` is the protocol (a named
tuple: vectors are per observation, scalars constant across observations; see [`obs`](@ref)),
`p` holds every parameter by name (`p.T2`, `p.B1`, ...). Must be generic in the number type of
`p`: it is called with dual numbers to obtain derivatives.
"""
function signal! end

"""
    signal(m, acq, p) -> Vector

Allocating version of [`signal!`](@ref), for simulation and plotting. `p` must give every
parameter (estimated and fixed) by name; ties (functions) are resolved.
"""
function signal(m::AbstractModel, acq, p::NamedTuple)
    a = prepare_protocol(m, acq, Float64)
    q = _resolve(p)
    T = promote_type(Float64, map(typeof, filter(x -> x isa Number, values(q)))...)
    return signal!(Vector{T}(undef, nobs(a)), m, a, q)
end

"""
    obs(acq, i) -> NamedTuple

Settings of observation `i`: vector fields of the protocol are indexed, scalars repeated.
"""
@inline obs(acq::NamedTuple, i) = map(v -> v isa AbstractVector ? (@inbounds v[i]) : v, acq)

"""
    initial_guess(m, acq, y, p) -> NamedTuple

Starting values for one voxel with data `y`, for the parameters declared `Estimate(:auto)`.
`p` holds the voxel's fixed values (e.g. its B1). Only called when some estimated parameter has
no numeric starting value.
"""
initial_guess(m::AbstractModel, acq, y, p) =
    throw(ArgumentError("$(typeof(m)) has no initial_guess; give numeric starting values " *
                        "with Estimate(value) or the `start` keyword"))

# ---------------------------------------------------------------------------------------------
# Protocol handling

nobs(acq::NamedTuple) = _nobs(values(acq))
_nobs(vals) = maximum((v isa AbstractVector ? length(v) : 0 for v in vals); init=0)

_conv(v::AbstractVector{<:Real}, ::Type{T}) where {T} = Vector{T}(v)
_conv(v::Real, ::Type{T}) where {T} = T(v)
_conv(v, ::Type{T}) where {T} = v

"""
    prepare_protocol(m, acq, T) -> NamedTuple

Normalize a protocol: a plain vector becomes `(name = acq,)` for one-setting models, numeric
fields are converted to `T`, and vector fields must all have the same length.
"""
function prepare_protocol(m, acq, ::Type{T}) where {T}
    if !(acq isa NamedTuple)
        names = protocol(m)
        length(names) == 1 ||
            throw(ArgumentError("$(typeof(m)) needs the protocol as a named tuple with fields $(names)"))
        acq = NamedTuple{names}((acq,))
    end
    for n in protocol(m)
        haskey(acq, n) || throw(ArgumentError("protocol is missing `$n` (needed by $(typeof(m)))"))
    end
    a = map(v -> _conv(v, T), acq)
    lens = [length(v) for v in values(a) if v isa AbstractVector]
    isempty(lens) && throw(ArgumentError("protocol has no per-observation (vector) settings"))
    all(==(lens[1]), lens) || throw(DimensionMismatch("protocol vectors have different lengths: $lens"))
    return a
end

# ---------------------------------------------------------------------------------------------
# Parameter assembly: estimated values + fixed values -> every parameter by name.

@inline _resolve1(x::Function, p) = x(p)
@inline _resolve1(x, p) = x
@inline _resolve(p::NamedTuple) = map(x -> _resolve1(x, p), p)

@inline _params(::Val{F}, θ, fixed) where {F} = _resolve(merge(NamedTuple{F}(Tuple(θ)), fixed))

# ---------------------------------------------------------------------------------------------
# Function-backed models

"""
    FunctionModel(f!, parameters; protocol = ())

A model from a plain function `f!(out, acq, p)` (same contract as [`signal!`](@ref)) and a
named tuple of parameter roles, without defining a type.

```julia
m = FunctionModel((out, acq, p) -> (out .= p.S0 .* exp.(-acq.TE ./ p.T2)),
                  (S0 = Estimate(1000.0; lower = 0), T2 = Estimate(50.0; lower = 0));
                  protocol = (:TE,))
```
"""
struct FunctionModel{F,R,N} <: AbstractModel
    f!::F
    roles::R
end
FunctionModel(f!, roles::NamedTuple; protocol=()) =
    FunctionModel{typeof(f!),typeof(roles),Tuple(Symbol.(protocol))}(f!, roles)
parameters(m::FunctionModel) = m.roles
protocol(::FunctionModel{F,R,N}) where {F,R,N} = N
signal!(out, m::FunctionModel, acq, p) = m.f!(out, acq, p)
Base.show(io::IO, m::FunctionModel) = print(io, "FunctionModel", keys(m.roles))

# Pointwise function f(o, p) for one observation, used by expr_model.
struct _Pointwise{F}
    f::F
end
@inline function (pw::_Pointwise)(out, acq, p)
    @inbounds for i in eachindex(out)
        out[i] = pw.f(obs(acq, i), p)
    end
    return out
end

module ExprModels end  # sandbox in which user expressions are compiled

"""
    expr_model(expr; params, protocol = (:x,), p0 = nothing, lower = nothing, upper = nothing)

A model from a Julia expression for the signal at one observation, e.g.

```julia
m = expr_model("S0 * exp(-TE / T2) + C"; params = (:S0, :T2, :C), protocol = (:TE,),
               p0 = (1000, 50, 0), lower = (0, 0, 0))
```

`params` is either names (all estimated, with optional `p0`, `lower`, `upper`) or a named
tuple of roles (e.g. `(S0 = Estimate(1000.0), B1 = Fixed(1.0))`). Names in `protocol` are the
acquisition settings. The expression is compiled as Julia code, so only use trusted input.
"""
function expr_model(expr::AbstractString; params, protocol=(:x,), p0=nothing, lower=nothing,
                    upper=nothing)
    roles = if params isa NamedTuple
        params
    else
        names = Tuple(Symbol.(collect(params)))
        n = length(names)
        get_i(v, i, d) = v === nothing ? d : Float64(v[i])
        for v in (p0, lower, upper)
            v === nothing || length(v) == n || throw(ArgumentError("expected $n values, got $(length(v))"))
        end
        NamedTuple{names}(ntuple(i -> Estimate(p0 === nothing ? :auto : Float64(p0[i]);
                                               lower=get_i(lower, i, -Inf),
                                               upper=get_i(upper, i, Inf)), n))
    end
    body = Meta.parse(expr)
    o, p = gensym(:o), gensym(:p)
    assigns = Expr[]
    for n in Symbol.(collect(protocol))
        push!(assigns, :($n = $o.$n))
    end
    for n in keys(roles)
        push!(assigns, :($n = $p.$n))
    end
    fexpr = Expr(:->, Expr(:tuple, o, p), Expr(:block, Expr(:let, Expr(:block, assigns...), body)))
    f = Core.eval(ExprModels, fexpr)
    return FunctionModel(_Pointwise(f), roles; protocol=Tuple(Symbol.(collect(protocol))))
end
