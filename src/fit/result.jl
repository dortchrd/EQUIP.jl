# =================================================================================================
# FitResult: what `fit` and `fit!` return, the same type for one voxel and for a dataset
#
#   one voxel:  res.params.T2 is a number,  res.status is a FitStatus
#   a dataset:  res.params.T2 is a map,     res.status is a map of FitStatus (1 byte per voxel)
#
# The result also keeps the model and the settings it was fitted with (references, not copies),
# so a fitted curve can be rebuilt (`fitted`), and a dataset refitted with the same choices.
# =================================================================================================

"""
    FitResult

The result of [`fit`](@ref) and [`fit!`](@ref): the same fields for one voxel (numbers) and for
a dataset (maps with the spatial size of the data).

- `params`: named tuple of the free parameters, e.g. `res.params.T1`
- `sse`: sum of squared residuals
- `iterations`: solver iterations
- `status`: [`FitStatus`](@ref), e.g. `ConvergedX`; `UInt8.(res.status)` gives the codes
  (e.g. for writing a status map to a file)
- `model`: the model as fitted (its acquisition in the working precision)
- `settings`: the keywords of the call: the parameter keywords, `precision`, `solver`, and
  `mask` for a dataset. [`refit`](@ref)`(res, data)` fits again with them, starting from these
  results

For a dataset, voxels that were not fitted hold `NaN` parameters and SSE, 0 iterations and
status `NotFitted` (outside the mask) or `InvalidData`. [`isconverged`](@ref)`(res)` and
[`fitted`](@ref)`(res)` / `fitted(res, i)` work on both.
"""
struct FitResult{P,S,I,St,M,K}
    params::P           # named tuple: numbers (one voxel) or maps (a dataset)
    sse::S
    iterations::I
    status::St
    model::M
    settings::K         # named tuple of the call's keywords
end

# One voxel or a dataset? The status field says: a FitStatus, or an array of them. Julia knows
# the field's type when compiling, so these checks cost nothing.
_onevoxel(r::FitResult) = r.status isa FitStatus

isconverged(r::FitResult) = _onevoxel(r) ? isconverged(r.status) : isconverged.(r.status)

# -------------------------------------------------------------------------------------------------
# Fitted curves
# -------------------------------------------------------------------------------------------------

"""
    fitted(res)           one-voxel result: the fitted signal, one value per observation
    fitted(res, i...)     dataset result: the fitted signal of one voxel, by linear index
                          (`fitted(res, 1234)`) or by position (`fitted(res, 40, 52, 10)`)

Rebuilt from `res.model`, the fitted parameters and the fixed values in `res.settings` (a
number, or that voxel's value of a map). `fitted` is StatsAPI's function name.
"""
function fitted(res::FitResult)
    _onevoxel(res) || throw(ArgumentError("a dataset result: give the voxel, e.g. fitted(res, i)"))
    fixed = _fixed_settings(res)
    any(x -> x isa AbstractArray, values(fixed)) &&
        throw(ArgumentError("fixed maps: the voxel is not known; fit the dataset with `fit` instead"))
    return signal(res.model, merge(fixed, res.params))
end

function fitted(res::FitResult, I...)
    _onevoxel(res) && throw(ArgumentError("a one-voxel result: use fitted(res)"))
    v = LinearIndices(res.status)[I...]                  # the voxel's linear index
    fixed = map(x -> _pick_value(x, v), _fixed_settings(res))
    return signal(res.model, merge(fixed, map(x -> x[v], res.params)))
end

"""
    residuals(res, y)            one-voxel result: `y .- fitted(res)`
    residuals(res, data, i...)   dataset result: the data minus the fit, for one voxel

`y` / `data` are the data that were fitted (a dataset with observations last).
"""
residuals(res::FitResult, y::AbstractVector) = y .- fitted(res)

function residuals(res::FitResult, data::AbstractArray, I...)
    v = LinearIndices(res.status)[I...]
    y = view(reshape(data, length(res.status), :), v, :)   # that voxel's observations
    return y .- fitted(res, I...)
end

# The parameters the call fixed (keywords that are parameters of the model and not `Free`), by
# name: numbers, maps or ties. The model's own fixed defaults are filled in later by `signal`.
_fixed_settings(res) =
    (; (n => _value_of(x) for (n, x) in pairs(res.settings) if haskey(parameters(res.model), n) && !(x isa Free))...)
_value_of(x::Fixed) = x.value                            # `kmf = Fixed(15.0)`
_value_of(x) = x                                         # `kmf = 15.0`, a map, a tie
_pick_value(x::AbstractArray, v) = x[v]
_pick_value(x, v) = x

# -------------------------------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------------------------------

# One line, e.g. inside a vector of results or in `print`.
function Base.show(io::IO, r::FitResult)
    name = nameof(typeof(r.model))
    if _onevoxel(r)
        vals = join(("$n = $(round(x; sigdigits = 4))" for (n, x) in pairs(r.params)), ", ")
        print(io, "FitResult(", name, ", one voxel: ", vals, "; ", r.status, ")")
    else
        nfit = count(!=(NotFitted), r.status)
        pct = nfit == 0 ? 0.0 : 100 * count(isconverged, r.status) / nfit
        print(io, "FitResult(", name, ", ", join(size(r.status), "×"), " voxels, ",
              round(pct; digits = 1), "% converged)")
    end
end

# The full summary, shown at the REPL.
function Base.show(io::IO, ::MIME"text/plain", r::FitResult)
    name = nameof(typeof(r.model))
    free = keys(r.params)
    fixed = Tuple(n for n in keys(parameters(r.model)) if !(n in free))
    w = maximum(n -> length(string(n)), free)            # width, to line up the names
    if _onevoxel(r)
        println(io, "FitResult: ", name, ", one voxel")
        print(io, "  status   ", r.status, " after ", r.iterations, " iterations, SSE ",
              round(r.sse; sigdigits = 5))
        for (n, x) in pairs(r.params)
            print(io, "\n  ", rpad(n, w), " = ", round(x; sigdigits = 6))
        end
        _print_fixed(io, r, fixed)
    else
        nvox = length(r.status)
        nfit = count(!=(NotFitted), r.status)
        conv = isconverged.(r.status)
        println(io, "FitResult: ", name, ", ", join(size(r.status), "×"), " voxels")
        println(io, "  fitted   ", nfit, " of ", nvox, " voxels", nfit < nvox ? " (mask)" : "",
                ", converged ", nfit == 0 ? 0.0 : round(100 * count(conv) / nfit; digits = 1), "%")
        counts = [(s, count(==(s), r.status)) for s in instances(FitStatus) if s != NotFitted]
        println(io, "  status   ", join(("$s $c" for (s, c) in counts if c > 0), ", "))
        print(io, "  medians over converged voxels (interquartile range):")
        for (n, x) in pairs(r.params)
            xs = x[conv]
            if isempty(xs)
                print(io, "\n    ", rpad(n, w), "   none converged")
            else
                q1, q2, q3 = quantile(vec(xs), (0.25, 0.5, 0.75))
                print(io, "\n    ", rpad(n, w), " = ", round(q2; sigdigits = 4), "   (",
                      round(q1; sigdigits = 4), " to ", round(q3; sigdigits = 4), ")")
            end
        end
        _print_fixed(io, r, fixed)
    end
end

# A last line: "fixed    kmf = 12.5, R1m = R1f (tied), Sm = 0.83, B1 = map", or "fixed    none".
function _print_fixed(io, r, fixed)
    given = _fixed_settings(r)
    roles = parameters(r.model)
    describe(n) = begin
        x = _value_of(haskey(given, n) ? given[n] : roles[n].value)
        x isa Number ? "$n = $x" : x isa Symbol ? "$n = $x (tied)" :
        x isa AbstractArray ? "$n = map" : "$n = function of others"
    end
    print(io, "\n  fixed    ", isempty(fixed) ? "none" : join(map(describe, fixed), ", "))
end
