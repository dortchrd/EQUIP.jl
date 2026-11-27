# =================================================================================================
# ObjectiveDesign: a protocol search for an objective of your own
#
#   design(f, m; acquisition, shared, constraints, round)   f(p) → a number to minimize, p a
#                                                            candidate protocol (a model)
#
# ProtocolDesign's objective is Eq. 5, a fit's errors from crlb; some protocols are judged by
# something else: MTR's CNR between two tissues per √(scan time), from `propagate(m, θ, g)`
# (the owner, 2026-11-27: "we often want to optimize SNR efficiency or CNR efficiency for MTR
# for a given pulse sequence"; docs/design.md, item 5). The search is the same (search.jl:
# differential evolution, then Nelder–Mead, starts on threads, feasibility first); only the
# objective is the user's.
#
# The protocol space: `acquisition` names the fields to vary, each with its bounds, (lo, hi) for
# every image, or one entry per image, a range to vary or a number to keep (an MTR protocol's
# reference stays where it is: `offset = [(1e3, 20e3), 100e3]`). With `shared = (:TR,)` a
# field's varied entries take one value. Not sorted after the search, as ProtocolDesign's are:
# the objective may depend on the images' order (a reference given by its index).
#
# Not allocation-free, as ProtocolDesign is (its kernel): the objective is the user's code, and
# a few tens of microseconds an evaluation is fast enough for a search of a few variables.
# =================================================================================================

"""
    ObjectiveDesign(f, m; acquisition, shared = (), constraints = nothing)

A protocol search's objective of your own: `f(p)`, a number to minimize, for a candidate
protocol `p` (a model like `m`, its `acquisition` fields varied). [`design`](@ref)`(f, m; ...)`
makes one and searches it; `d(p)` evaluates it.

- `acquisition`: the fields to vary with their bounds, `(lo, hi)` for every image, or one entry
  per image, `(lo, hi)` to vary it or a number to keep it (`offset = [(1e3, 20e3), 100e3]`: the
  MT image's offset varied, the reference's kept)
- `shared`: fields whose varied entries take one value (`shared = (:TR, :FA)`)
- `constraints`: a function of the protocol whose values must all be ≥ 0 (a SAR limit:
  `p -> 25.6 .- p.B1e .^ 2 .* p.tm ./ p.TR`), or `nothing`

`f` may return `Inf` (or `NaN`, read as `Inf`) where a protocol is no good. The search calls it
from several threads at once (a start each): it must not share buffers between calls.
"""
struct ObjectiveDesign{F,M<:MRIModel,C}
    f::F
    model::M
    fields::Vector{Symbol}         # each variable's field,
    entries::Vector{Vector{Int}}   #   and the entries (images) it sets
    lower::Vector{Float64}
    upper::Vector{Float64}
    constraints::C
end

function ObjectiveDesign(f, m::MRIModel; acquisition = nothing, shared = (), constraints = nothing)
    acquisition === nothing && throw(ArgumentError("give `acquisition`, the fields to vary with their " *
        "bounds: `acquisition = (offset = (1e3, 20e3),)`, or one entry per image, `offset = [(1e3, 20e3), 100e3]`"))
    acquisition = _byname(acquisition, "acquisition", "(offset = (1e3, 20e3),)")
    isempty(acquisition) && throw(ArgumentError("`acquisition`: give at least one field to vary"))
    N = nobs(m)
    names = fieldnames(typeof(m))
    bounds(b) = b isa Tuple{Real,Real} && isfinite(b[1]) && isfinite(b[2]) && b[1] < b[2]
    fields, entries, lower, upper = Symbol[], Vector{Int}[], Float64[], Float64[]
    for f_ in shared
        haskey(acquisition, f_) || throw(ArgumentError("`shared`: `$f_` is not in `acquisition`"))
    end
    for (f_, b) in pairs(acquisition)
        f_ in names || throw(ArgumentError("`acquisition`: $(nameof(typeof(m))) has no field `$f_`; " *
            "its fields are $(join(names, ", "))"))
        getfield(m, f_) isa AbstractVector{<:Real} && length(getfield(m, f_)) == N ||
            throw(ArgumentError("`acquisition`: `$f_` is not a per-image field (a vector of $N numbers)"))
        spec = b isa AbstractVector ? b : fill(b, N)          # one entry per image
        length(spec) == N || throw(DimensionMismatch("`acquisition`: `$f_` has $(length(spec)) entries for $N images"))
        varied = Int[]
        for (j, s) in enumerate(spec)
            s isa Real && continue                            # kept: the protocol's own value below
            bounds(s) || throw(ArgumentError("`acquisition`: `$f_`'s entry $j: a range (lower, upper), " *
                                             "finite, lower < upper, or a number to keep"))
            push!(varied, j)
        end
        isempty(varied) && throw(ArgumentError("`acquisition`: `$f_` has no entry to vary"))
        groups = f_ in shared ? [varied] : [[j] for j in varied]
        for g in groups
            lo, hi = maximum(j -> spec[j][1], g), minimum(j -> spec[j][2], g)   # shared: the common range
            lo < hi || throw(ArgumentError("`acquisition`: `$f_`'s shared entries have no common range"))
            push!(fields, f_); push!(entries, g); push!(lower, lo); push!(upper, hi)
        end
    end
    # the kept entries take the values given in `acquisition` (numbers), the rest m's own
    base = deepcopy(m)
    for (f_, b) in pairs(acquisition)
        b isa AbstractVector || continue
        v = getfield(base, f_)
        for (j, s) in enumerate(b)
            s isa Real && (v[j] = s)
        end
    end
    d = ObjectiveDesign(f, base, fields, entries, lower, upper, constraints)
    d(base)                         # once, so a mistake in f or the constraints shows here
    return d
end

# The protocol for the values x: a copy of the model with each variable's entries set. The
# fields go back to the model's positional constructor (`W(...)`) in order, as ProtocolDesign's.
function _protocol(d::ObjectiveDesign, x::AbstractVector)
    m = d.model
    fs = Dict{Symbol,Vector{Float64}}()
    for (k, f) in enumerate(d.fields)
        v = get!(() -> Float64.(collect(getfield(m, f))), fs, f)
        for j in d.entries[k]
            v[j] = x[k]
        end
    end
    W = Base.typename(typeof(m)).wrapper
    return W((haskey(fs, n) ? fs[n] : getfield(m, n) for n in fieldnames(typeof(m)))...)
end

"""
    (d::ObjectiveDesign)(x) -> Float64
    (d::ObjectiveDesign)(p::MRIModel) -> Float64

The objective at the values `x` (the variables in `acquisition`'s order) or at a protocol `p`:
`Inf` where a constraint is broken, or where `f` gives `Inf` or `NaN`.
"""
(d::ObjectiveDesign)(x::AbstractVector{<:Real}) = _objective(_rank(d, x))
(d::ObjectiveDesign)(p::MRIModel) = _objective(_rank(d, p))
_rank(d::ObjectiveDesign, x::AbstractVector{<:Real}) = _rank(d, _protocol(d, x))
function _rank(d::ObjectiveDesign, p::MRIModel)
    v = _violation(d.constraints, p)
    v > 0 && return (v, Inf)
    y = Float64(d.f(p))
    return (0.0, isnan(y) ? Inf : y)
end

# The search's other needs: a copy per start (nothing to copy: the objective holds no buffers of
# EQUIP's), the protocol's own values as x, and the result's protocol (in the images' order)
Base.copy(d::ObjectiveDesign) = d
_x(d::ObjectiveDesign, m) = clamp.([Float64(getfield(m, f)[first(d.entries[k])]) for (k, f) in enumerate(d.fields)],
                                   d.lower, d.upper)
_sortedprotocol(d::ObjectiveDesign, x) = _protocol(d, x)
_varfields(d::ObjectiveDesign) = d.fields
_defaultround(::ObjectiveDesign) = nothing

function Base.show(io::IO, d::ObjectiveDesign)
    b(k) = "$(d.fields[k])[$(join(d.entries[k], ","))] ∈ [$(d.lower[k]), $(d.upper[k])]"
    print(io, "ObjectiveDesign(", nameof(typeof(d.model)), ", ", nobs(d.model), " images: ",
          join((b(k) for k in eachindex(d.fields)), ", "), ")")
end
