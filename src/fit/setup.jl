# =================================================================================================
# Fit setup: from "what the user asked for" to resolved parameter roles
#
#   parameters(m)  +  keywords named after the parameters (kmf = 15.0, R1f = Free(start = 0.8))
#        │
#        ▼  setup        merge the user's choices into the model's default roles, then split the
#   FitSpec              parameters into Free (estimated) and Fixed. Done ONCE per fit call.
#
# The next steps are in cache.jl (init -> FitCache, fit! -> one voxel) and fit.jl (all voxels).
# =================================================================================================

"""
    FitSpec{F,P,T}

The resolved parameter roles of one fit (made by `setup`):

- `F`: names of the free parameters, e.g. `(:PSR, :R1f, :Sf, :M0f)`, in model order
- `P = length(F)`, `T`: the working float type
- `lower`, `upper`: bounds of the free parameters (`SVector{P,T}`)
- `start`: starting values of the free parameters (numbers or per-voxel maps), names `F`
- `scale`: position in `F` of the `Free(:scale)` parameter, or 0 if there is none
- `fixed`: the fixed parameters (numbers, per-voxel maps or ties), by name
- `grid`, `gridindex`: starting values to try for free parameter number `gridindex` (from the
  model's [`startgrid`](@ref)); `gridindex = 0` means no grid
- `userstart`: for each free parameter, whether the call gave it a start (then the model's
  [`datastart`](@ref) and [`startgrid`](@ref) leave it alone)

`F` and `P` are type parameters rather than fields: the compiler then knows the names and the
number of free parameters and generates code specialized for them (fixed-size vectors and
compile-time named tuples instead of name lookups at run time).
"""
struct FitSpec{F,P,T,S,X}
    lower::SVector{P,T}
    upper::SVector{P,T}
    start::S        # NamedTuple with names F (a :scale parameter holds 1 here)
    scale::Int      # which free parameter starts at the least-squares scale (0 = none)
    fixed::X        # NamedTuple with the fixed parameters' names
    grid::Vector{T} # starting values to try for one free parameter (empty = none)
    gridindex::Int  # which free parameter the grid is for (0 = none)
    userstart::NTuple{P,Bool}   # true for free parameters given a start in the call
end

freenames(::FitSpec{F}) where {F} = F       # read the names back out of the type

# --- How the keywords become roles ---------------------------------------------------------------

# Keywords of `fit` and `init` that are options, not parameters. A model's parameter cannot have
# one of these names: the keyword would be ambiguous.
const _OPTIONS = (:mask, :precision, :solver)

# A keyword `name = x` turns the parameter's default role into the fit's role:
#   Free(...)       estimated; whatever it leaves out (start, bounds) keeps the default
#   Fixed(...)      as given
#   anything else   fixed at that value: a number, a per-voxel map, or a tie (`R1m = :R1f`, or a
#                   function of the other parameters)
_role(default, x::Free, name) = _override(default, x, name)
_role(default, x::Fixed, name) = x
_role(default, x, name) = Fixed(x)

_keep(new, default) = new === nothing ? default : new
# Over a default Free: the new start and bounds where given, the default's elsewhere.
_override(d::Free, x::Free, name) = Free(_keep(x.start, d.start), _keep(x.lower, d.lower), _keep(x.upper, d.upper))
# A parameter fixed by default, now estimated: it starts at its fixed value unless given a start;
# no bounds unless given.
function _override(d::Fixed, x::Free, name)
    x.start !== nothing && return x
    d.value isa Real ||
        throw(ArgumentError("`$name` is fixed by default to $(repr(d.value)); give it a start: $name = Free(start = ...)"))
    return Free(d.value, x.lower, x.upper)
end

# Numbers are converted to the working precision; a tie by name becomes a `_Tie` (see
# models/interface.jl); maps and function ties are kept as they are.
_value(x::Real, ::Type{T}) where {T} = T(x)
_value(x::Symbol, ::Type{T}) where {T} = _Tie(x)
_value(x, ::Type{T}) where {T} = x

"""
    setup(model, T = Float64; params...) -> FitSpec

Combine the model's default roles with the fit's keywords, and split the parameters into free
and fixed. Each keyword is named after a parameter of the model:

- a value fixes it: a number, a per-voxel map, or a tie to another parameter, by name
  (`R1m = :R1f`) or as a function (`R1m = p -> 0.8 * p.R1f`);
- `Free(...)` estimates it, with the start (a number or a map) and bounds given, and the model's
  defaults for anything left out: `kmf = Free()`, `R1f = Free(start = 0.8)`, `Sf = Free(lower = -1.5)`.
"""
function setup(model::MRIModel, ::Type{T} = Float64; params...) where {T}
    # 1. The model's defaults, e.g. (PSR = Free(0.1; ...), ..., kmf = Fixed(12.5), ...).
    roles = parameters(model)
    names = keys(roles)
    for n in names
        n in _OPTIONS && throw(ArgumentError("$(nameof(typeof(model))) has a parameter named `$n`, " *
            "which is also an option of fit ($(join(_OPTIONS, ", "))); rename the parameter"))
    end

    # 2. Apply the keywords, one parameter each. `merge(a, b)` replaces a's entry by b's with the
    #    same name and keeps a's order: (…, kmf = Fixed(12.5), …) becomes (…, kmf = Fixed(15.0), …).
    userstart = Symbol[]                                 # parameters given a start in the call
    for (n, x) in pairs(params)
        haskey(roles, n) || throw(ArgumentError(
            n === :mask ? "`mask` is for datasets: fit(m, data; mask) with `data` holding many voxels" :
            n in fieldnames(LM) ? "`$n` is a solver setting: solver = LM($n = ...)" :
            "`$n` is neither a parameter of $(nameof(typeof(model))) ($(join(names, ", "))) " *
            "nor an option ($(join(_OPTIONS, ", ")))"))
        x isa Free && x.start !== nothing && x.start !== :scale && push!(userstart, n)
        roles = merge(roles, NamedTuple{(n,)}((_role(roles[n], x, n),)))
    end

    # 3. Split by role, keeping the model's order.
    F = Tuple(n for n in names if roles[n] isa Free)     # e.g. (:PSR, :R1f, :Sf, :M0f)
    X = Tuple(n for n in names if roles[n] isa Fixed)    # e.g. (:kmf, :R1m, :Sm)
    isempty(F) && throw(ArgumentError("no free parameters: everything is fixed"))
    P = length(F)
    # A tie by name (R1m = :R1f) must name a parameter that is not itself a tie.
    _check_ties(map(r -> r isa Fixed ? r.value : r, roles))

    # 4. Starting values: each Free role's start, a number or a map. A `:scale` parameter gets the
    #    placeholder 1 here; `_fit_voxel` replaces it voxel by voxel.
    for n in F
        roles[n].start === nothing && throw(ArgumentError("`$n` has no starting value in parameters($(nameof(typeof(model))))"))
    end
    isscale(n) = roles[n].start === :scale
    count(isscale, F) <= 1 || throw(ArgumentError("only one parameter can start as :scale"))
    scale = something(findfirst(isscale, F), 0)          # findfirst gives `nothing` if none
    starts = NamedTuple{F}(map(n -> isscale(n) ? one(T) : _value(roles[n].start, T), F))

    # 5. Bounds as fixed-size vectors (none = ±Inf), and the fixed values.
    lower = SVector{P,T}(map(n -> T(something(roles[n].lower, -Inf)), F))
    upper = SVector{P,T}(map(n -> T(something(roles[n].upper, Inf)), F))
    all(lower .<= upper) || throw(ArgumentError("a lower bound exceeds its upper bound"))
    fixedvals = NamedTuple{X}(map(n -> _value(roles[n].value, T), X))

    # 6. Grid of starting values (optional, see `startgrid`): used only if its parameter is free
    #    and has no start from the call; values outside the bounds are dropped.
    grid, gridindex = T[], 0
    g = startgrid(model)
    if g !== nothing
        length(g) == 1 || throw(ArgumentError("`startgrid` must give values for one parameter"))
        k = findfirst(==(only(keys(g))), F)
        if k !== nothing && !(F[k] in userstart)
            grid = T[T(x) for x in only(values(g)) if lower[k] <= x <= upper[k]]
            gridindex = isempty(grid) ? 0 : k
        end
    end

    # The names F and their number P go into the type: FitSpec{(:PSR, :R1f, :Sf, :M0f), 4, ...}.
    return FitSpec{F,P,T,typeof(starts),typeof(fixedvals)}(lower, upper, starts, scale, fixedvals,
                                                           grid, gridindex, map(n -> n in userstart, F))
end
