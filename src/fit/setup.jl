# =================================================================================================
# Fit setup: from "what the user asked for" to resolved parameter roles
#
#   parameters(m)  +  fixed / free / start keywords
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
- `userstart`: for each free parameter, whether the user gave it a `start` (then the model's
  [`datastart`](@ref) leaves it alone)

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
    userstart::NTuple{P,Bool}   # true for free parameters with a `start` from the user
end

freenames(::FitSpec{F}) where {F} = F       # read the names back out of the type

# --- How each kind of user input becomes a role --------------------------------------------------

# `fixed = (kmf = 15.0, B1 = b1map, R1m = :R1f)`: plain values become Fixed(value).
_asfixed(x::Fixed) = x
_asfixed(x::Free) = throw(ArgumentError("use `free` (not `fixed`) to estimate a parameter"))
_asfixed(x) = Fixed(x)

# `free` accepts names, `free = (:kmf,)` or `free = :kmf`, or roles, `free = (kmf = Free(...),)`.
# The result is always a named tuple of Free roles.
_asfree(free::Symbol, roles) = _asfree((free,), roles)
_asfree(free::NamedTuple, roles) =
    map(x -> x isa Free ? x : throw(ArgumentError("`free` values must be Free(...), got $x")), free)
function _asfree(free, roles)                    # a tuple or vector of names
    names = Tuple(Symbol(n) for n in free)
    _check_names("free", names, keys(roles))
    return NamedTuple{names}(map(n -> _tofree(roles[n], n), names))
end
# Freeing a parameter by name: a Free one stays as is; a Fixed number becomes the starting value
# (with no bounds; give `name = Free(init; lower, upper)` for bounds); maps and ties need a start.
_tofree(r::Free, n) = r
_tofree(r::Fixed{<:Real}, n) = Free(r.value)
_tofree(r::Fixed, n) = throw(ArgumentError("give `$n` a starting value: free = ($n = Free(init),)"))

# Numbers are converted to the working precision; a tie by name becomes a `_Tie` (see
# models/interface.jl); maps and function ties are kept as they are.
_value(x::Real, ::Type{T}) where {T} = T(x)
_value(x::Symbol, ::Type{T}) where {T} = _Tie(x)
_value(x, ::Type{T}) where {T} = x

"""
    setup(model, T = Float64; fixed = (;), free = (), start = (;)) -> FitSpec

Combine the model's default roles with the user's choices and split the parameters into free
and fixed:

- `fixed`: named tuple of parameters to hold fixed: numbers, per-voxel maps, or ties (the
  name of another parameter, `R1m = :R1f`, or a function, `R1m = p -> 0.8 * p.R1f`).
- `free`: parameters to estimate that the model fixes by default: names (`(:kmf,)`) or a
  named tuple of roles (`(kmf = Free(12.5; lower = 0),)`).
- `start`: named tuple of starting values for free parameters: numbers or per-voxel maps.
"""
function setup(model::MRIModel, ::Type{T} = Float64; fixed = (;), free = (), start = (;)) where {T}
    # 1. The model's defaults, e.g. (PSR = Free(0.1; ...), ..., kmf = Fixed(12.5), ...).
    roles = parameters(model)
    names = keys(roles)

    # 2. Apply `fixed`: `merge(a, b)` replaces a's entries by b's with the same name and keeps
    #    a's order, so (…, kmf = Fixed(12.5), …) becomes (…, kmf = Fixed(15.0), …).
    fixed isa NamedTuple || throw(ArgumentError("`fixed` must be a named tuple, e.g. (kmf = 15.0,)"))
    _check_names("fixed", keys(fixed), names)
    roles = merge(roles, map(_asfixed, fixed))

    # 3. Apply `free` the same way.
    free = _asfree(free, roles)
    _check_names("free", keys(free), names)
    for n in keys(free)
        n in keys(fixed) && throw(ArgumentError("`$n` is in both `fixed` and `free`"))
    end
    roles = merge(roles, free)

    # 4. Split by role, keeping the model's order.
    F = Tuple(n for n in names if roles[n] isa Free)     # e.g. (:PSR, :R1f, :Sf, :M0f)
    X = Tuple(n for n in names if roles[n] isa Fixed)    # e.g. (:kmf, :R1m, :Sm)
    isempty(F) && throw(ArgumentError("no free parameters: everything is fixed"))
    P = length(F)
    # A tie by name (R1m = :R1f) must name a parameter that is not itself a tie.
    _check_ties(map(r -> r isa Fixed ? r.value : r, roles))

    # 5. Starting values: each Free role's `init`, overridden by the `start` keyword. A `:scale`
    #    parameter gets the placeholder 1 here; `_fit_voxel` replaces it voxel by voxel.
    start isa NamedTuple || throw(ArgumentError("`start` must be a named tuple, e.g. (T1 = 1.2,)"))
    _check_names("start", keys(start), F)                # only free parameters have starts
    isscale(n) = roles[n].init === :scale && !haskey(start, n)
    count(isscale, F) <= 1 || throw(ArgumentError("only one parameter can start as :scale"))
    scale = something(findfirst(isscale, F), 0)          # findfirst gives `nothing` if none
    init_value(n) = roles[n].init === :scale ? one(T) : T(roles[n].init)
    starts = merge(NamedTuple{F}(map(init_value, F)), map(x -> _value(x, T), start))

    # 6. Bounds as fixed-size vectors, and the fixed values.
    lower = SVector{P,T}(map(n -> T(roles[n].lower), F))
    upper = SVector{P,T}(map(n -> T(roles[n].upper), F))
    all(lower .<= upper) || throw(ArgumentError("a lower bound exceeds its upper bound"))
    fixedvals = NamedTuple{X}(map(n -> _value(roles[n].value, T), X))

    # 7. Grid of starting values (optional, see `startgrid`): used only if its parameter is free
    #    and has no `start` from the user; values outside the bounds are dropped.
    grid, gridindex = T[], 0
    g = startgrid(model)
    if g !== nothing
        length(g) == 1 || throw(ArgumentError("`startgrid` must give values for one parameter"))
        k = findfirst(==(only(keys(g))), F)
        if k !== nothing && !haskey(start, F[k])
            grid = T[T(x) for x in only(values(g)) if lower[k] <= x <= upper[k]]
            gridindex = isempty(grid) ? 0 : k
        end
    end

    # The names F and their number P go into the type: FitSpec{(:PSR, :R1f, :Sf, :M0f), 4, ...}.
    userstart = map(n -> haskey(start, n), F)
    return FitSpec{F,P,T,typeof(starts),typeof(fixedvals)}(lower, upper, starts, scale, fixedvals,
                                                           grid, gridindex, userstart)
end
