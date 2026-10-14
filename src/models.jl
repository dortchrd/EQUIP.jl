# Model interface.
#
# A model with P parameters is evaluated one observation at a time: `m(x, θ)` returns the
# predicted signal at acquisition setting `x` (an echo time, b-value, flip angle, or a small
# tuple/SVector of settings) for parameters `θ::SVector{P}`. Evaluating pointwise lets the
# solver stream over observations and accumulate JᵀJ and Jᵀr in registers without ever forming
# the m×P Jacobian, and it is what makes the same code run inside a GPU kernel.

"""
    AbstractModel{P}

Supertype of signal models with `P` free parameters.

Required:

- `(m::MyModel)(x, θ)`: predicted signal for one observation, `θ::SVector{P}`. Must be generic
  in the element type of `θ` (it is called with `ForwardDiff.Dual` numbers).

Optional (all have defaults):

- [`paramnames`](@ref)`(m)`: tuple of `P` Symbols.
- [`initial_guess`](@ref)`(m, x, y)`: per-voxel starting point from the data.
- [`lower_bounds`](@ref)`(m)`, [`upper_bounds`](@ref)`(m)`: tuples of `P` numbers.
- `(m::MyModel)(x, θ, c)`: model with per-voxel covariates `c::SVector{K}` (e.g. a B1 map).
- `VoxelFit.value_gradient(m, x, θ)`: analytic `(f, ∇θ f)` to bypass automatic differentiation.

Models must be `isbits` (no array fields) to run on a GPU.
"""
abstract type AbstractModel{P} end

"""
    AbstractCurveModel{P} <: AbstractModel{P}

Supertype for models that compute the whole signal curve at once (e.g. extended phase graphs),
where evaluating observation by observation would repeat work. Implement
[`curve!`](@ref)`(out, m, x, θ)` (and optionally `curve!(out, m, x, θ, c)` for covariates)
instead of the pointwise call. Curve models run on the CPU only.
"""
abstract type AbstractCurveModel{P} <: AbstractModel{P} end

"""
    curve!(out, m::AbstractCurveModel, x::AbstractVector, θ[, c])

Write the predicted signal for every observation `x[i]` into `out[i]`.
"""
function curve! end

nparams(::AbstractModel{P}) where {P} = P
nparams(::Type{<:AbstractModel{P}}) where {P} = P

"""
    paramnames(m) -> NTuple{P,Symbol}
"""
paramnames(::AbstractModel{P}) where {P} = ntuple(i -> Symbol(:p, i), Val(P))

"""
    lower_bounds(m) -> NTuple{P,Float64}

Default lower bounds used by the fitters (`-Inf` means unbounded).
"""
lower_bounds(::AbstractModel{P}) where {P} = ntuple(_ -> -Inf, Val(P))

"""
    upper_bounds(m) -> NTuple{P,Float64}
"""
upper_bounds(::AbstractModel{P}) where {P} = ntuple(_ -> Inf, Val(P))

"""
    initial_guess(m, x, y[, c]) -> SVector{P}

Starting point for one voxel computed from its data `y` (and covariates `c`). Built-in models use
linearized closed-form estimates. Must not allocate if the model is to run on a GPU.
"""
initial_guess(m::AbstractModel, x, y, c) = initial_guess(m, x, y)
initial_guess(m::AbstractModel, x, y) =
    throw(ArgumentError("no initial_guess method for $(typeof(m)); pass `init=` to the fitter"))

# ---------------------------------------------------------------------------------------------
# Automatic derivatives with ForwardDiff duals seeded on an SVector: no allocation, GPU-safe.

struct VoxelFitTag end

@inline function _seed(θ::SVector{P,T}) where {P,T}
    SVector(ntuple(j -> Dual{VoxelFitTag}(θ[j], ntuple(k -> ifelse(k == j, one(T), zero(T)), Val(P))),
                   Val(P)))
end

@inline _vg(f::Dual, ::Val{P}, ::Type{T}) where {P,T} =
    (T(value(f)), SVector{P,T}(partials(f).values))
@inline _vg(f::Real, ::Val{P}, ::Type{T}) where {P,T} = (T(f), zero(SVector{P,T}))

"""
    value_gradient(m, x, θ[, c]) -> (f, ∇f::SVector{P})

Signal and its gradient with respect to `θ` at one observation. The default uses forward-mode
automatic differentiation; overload it to supply an analytic gradient.
"""
@inline value_gradient(m::AbstractModel{P}, x, θ::SVector{P,T}) where {P,T} =
    _vg(m(x, _seed(θ)), Val(P), T)
@inline value_gradient(m::AbstractModel{P}, x, θ::SVector{P,T}, c) where {P,T} =
    _vg(m(x, _seed(θ), c), Val(P), T)

# ---------------------------------------------------------------------------------------------
# Covariates: bind a voxel's covariate vector to the model so the solver only sees m(x, θ).

struct WithCovariates{P,M<:AbstractModel{P},C} <: AbstractModel{P}
    model::M
    c::C
end
@inline (w::WithCovariates)(x, θ) = w.model(x, θ, w.c)
@inline value_gradient(w::WithCovariates{P}, x, θ::SVector{P}) where {P} =
    value_gradient(w.model, x, θ, w.c)
initial_guess(w::WithCovariates, x, y) = initial_guess(w.model, x, y, w.c)

@inline bind_covariates(m::AbstractModel, ::Nothing) = m
@inline bind_covariates(m::AbstractModel{P}, c) where {P} = WithCovariates{P,typeof(m),typeof(c)}(m, c)

# Curve models dispatch through a trait so that wrappers like WithCovariates keep it.
abstract type ModelStyle end
struct Pointwise <: ModelStyle end
struct Curve <: ModelStyle end
modelstyle(::AbstractModel) = Pointwise()
modelstyle(::AbstractCurveModel) = Curve()
modelstyle(w::WithCovariates) = modelstyle(w.model)

_curve!(out, m::AbstractModel, x, θ) = curve!(out, m, x, θ)
_curve!(out, w::WithCovariates, x, θ) = curve!(out, w.model, x, θ, w.c)

# Per-task scratch space: a buffer of duals for curve models, nothing for pointwise models.
workspace(m::AbstractModel, ::Type{T}, nobs) where {T} = _workspace(modelstyle(m), m, T, nobs)
_workspace(::Pointwise, m, T, nobs) = nothing
_workspace(::Curve, ::AbstractModel{P}, ::Type{T}, nobs) where {P,T} =
    Vector{Dual{VoxelFitTag,T,P}}(undef, nobs)

# ---------------------------------------------------------------------------------------------
# Generic function-backed model.

"""
    PointModel(f, names; p0=nothing, lower=nothing, upper=nothing, init=nothing)

Wrap a function `f(x, θ)` (or `f(x, θ, c)` when covariates are used) as a model with parameters
`names` (a tuple or vector of Symbols). Starting values come from `init(x, y)` if given,
otherwise the constant `p0`.

```julia
m = PointModel((t, θ) -> θ[1] * exp(-t / θ[2]) + θ[3], (:S0, :T2, :C);
               p0 = (1000.0, 50.0, 0.0), lower = (0, 0, 0))
```

For GPU use, `f` must not capture arrays (plain anonymous functions are fine).
"""
struct PointModel{P,F,I} <: AbstractModel{P}
    f::F
    names::NTuple{P,Symbol}
    p0::NTuple{P,Float64}
    lower::NTuple{P,Float64}
    upper::NTuple{P,Float64}
    init::I
end

function PointModel(f, names; p0=nothing, lower=nothing, upper=nothing, init=nothing)
    P = length(names)
    tup(v, d) = v === nothing ? ntuple(_ -> d, P) :
                (length(v) == P ? ntuple(i -> Float64(v[i]), P) :
                 throw(ArgumentError("expected $P values, got $(length(v))")))
    PointModel{P,typeof(f),typeof(init)}(f, ntuple(i -> Symbol(names[i]), P), tup(p0, NaN),
                                         tup(lower, -Inf), tup(upper, Inf), init)
end

@inline (m::PointModel)(x, θ) = m.f(x, θ)
@inline (m::PointModel)(x, θ, c) = m.f(x, θ, c)
paramnames(m::PointModel) = m.names
lower_bounds(m::PointModel) = m.lower
upper_bounds(m::PointModel) = m.upper
initial_guess(m::PointModel, x, y) = _pm_init(m.init, m, x, y)
initial_guess(m::PointModel, x, y, c) = _pm_init(m.init, m, x, y)
_pm_init(init, m, x, y) = init(x, y)
function _pm_init(::Nothing, m::PointModel{P}, x, y) where {P}
    any(isnan, m.p0) && throw(ArgumentError("PointModel has no `p0` or `init`; pass `init=` to the fitter"))
    SVector{P,Float64}(m.p0)
end

Base.show(io::IO, m::PointModel{P}) where {P} = print(io, "PointModel{", P, "}", m.names)

# ---------------------------------------------------------------------------------------------
# Models from strings (for the CLI, Python and MATLAB front ends).

module ExprModels end  # sandbox in which user expressions are compiled

"""
    expr_model(expr; params, x=:x, covariates=(), p0=nothing, lower=nothing, upper=nothing)

Build a [`PointModel`](@ref) from a Julia expression string, e.g.

```julia
m = expr_model("S0 * exp(-TE / T2) + C"; params = (:S0, :T2, :C), x = :TE,
               p0 = (1000, 50, 0), lower = (0, 0, 0))
```

`x` names the independent variable; give several names (`x = (:TR, :alpha)`) when each
observation has several settings (pass `x` to the fitter as an `nobs × k` matrix). Names in
`covariates` refer to per-voxel covariate maps. The expression is compiled as Julia code, so
only use trusted input.
"""
function expr_model(expr::AbstractString; params, x=:x, covariates=(), p0=nothing, lower=nothing,
                    upper=nothing)
    body = Meta.parse(expr)
    θ, xv, cv = gensym(:θ), gensym(:x), gensym(:c)
    assigns = Expr[]
    for (i, p) in enumerate(params)
        push!(assigns, :($(Symbol(p)) = $θ[$i]))
    end
    if x isa Union{Symbol,AbstractString}
        push!(assigns, :($(Symbol(x)) = $xv))
    else
        for (k, xn) in enumerate(x)
            push!(assigns, :($(Symbol(xn)) = $xv[$k]))
        end
    end
    args = Any[xv, θ]
    if !isempty(covariates)
        push!(args, cv)
        for (k, cn) in enumerate(covariates)
            push!(assigns, :($(Symbol(cn)) = $cv[$k]))
        end
    end
    fexpr = Expr(:->, Expr(:tuple, args...), Expr(:block, Expr(:let, Expr(:block, assigns...), body)))
    f = Core.eval(ExprModels, fexpr)
    return PointModel(f, Tuple(Symbol.(params)); p0, lower, upper)
end
