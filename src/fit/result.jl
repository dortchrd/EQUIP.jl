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
- `fixed`: named tuple of the fixed parameters' values used in the fit, the model's defaults
  and the call's: numbers, ties (`R1m = :R1f`), and for a dataset maps; for one voxel fitted
  with a map (`fit!(c, y; voxel = v)`), that voxel's value
- `sse`: sum of squared residuals
- `iterations`: solver iterations
- `status`: [`FitStatus`](@ref), e.g. `ConvergedX`; `UInt8.(res.status)` gives the codes
  (e.g. for writing a status map to a file)
- `model`: the model as fitted (its acquisition in the working precision)
- `settings`: the keywords of the call: the parameter keywords, `precision`, `solver`, and
  `mask` for a dataset. [`refit`](@ref)`(res, data)` fits again with them, starting from these
  results

For a dataset, voxels that were not fitted hold `NaN` parameters and SSE, 0 iterations and
status `NotFitted` (outside the mask) or `InvalidData`. [`isconverged`](@ref)`(res)`,
[`fitted`](@ref)`(res)` / `fitted(res, i)` and [`resolve`](@ref)`(res)` / `resolve(res, i)`
(every parameter's value, fitted and fixed) work on both.
"""
struct FitResult{P,X,S,I,St,M,K}
    params::P           # named tuple: numbers (one voxel) or maps (a dataset)
    fixed::X            # named tuple: the fixed values used (numbers, ties; maps for a dataset)
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

Rebuilt from `res.model` and every parameter's value, [`resolve`](@ref)`(res)`: the fitted ones
and the fixed values used (a number, or that voxel's value of a map). `fitted` is StatsAPI's
function name.
"""
fitted(res::FitResult) = signal(res.model, resolve(res))
fitted(res::FitResult, I...) = signal(res.model, resolve(res, I...))

"""
    resolve(res)          one-voxel result: every parameter's value, fitted and fixed
    resolve(res, i...)    dataset result: the same for one voxel, by linear index or position

The complete set of values behind a fit, in the model's order, ties evaluated: the fitted
parameters and the fixed values used (`res.fixed`; a map's value at the voxel).
`fitted(res)` is `signal(res.model, resolve(res))`.

```julia
r = fit(QMTSIR(; SIR_4POINT...), y)
resolve(r)        # (PSR = ..., R1f = ..., Sf = ..., M0f = ..., kmf = 12.5, R1m = (R1f's value), ...)
```
"""
function resolve(res::FitResult)
    _onevoxel(res) || throw(ArgumentError("a dataset result: give the voxel, e.g. resolve(res, i)"))
    return resolve(res.model, merge(res.fixed, res.params))
end

function resolve(res::FitResult, I...)
    _onevoxel(res) && throw(ArgumentError("a one-voxel result: use resolve(res)"))
    v = LinearIndices(res.status)[I...]                  # the voxel's linear index
    fixed = map(x -> _pick_value(x, v), res.fixed)      # a map's value at the voxel
    return resolve(res.model, merge(fixed, map(x -> x[v], res.params)))
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
    w = maximum(n -> length(string(n)), free)            # width, to line up the names
    if _onevoxel(r)
        println(io, "FitResult: ", name, ", one voxel")
        print(io, "  status   ", r.status, " after ", r.iterations, " iterations, SSE ",
              round(r.sse; sigdigits = 5))
        for (n, x) in pairs(r.params)
            print(io, "\n  ", rpad(n, w), " = ", round(x; sigdigits = 6))
        end
        _print_fixed(io, r)
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
        _print_fixed(io, r)
    end
end

# A last line: "fixed    kmf = 12.5, R1m = R1f (tied), Sm = 0.83, B1 = map", or "fixed    none".
function _print_fixed(io, r)
    describe((n, x)) = x isa Number ? "$n = $x" : x isa Symbol ? "$n = $x (tied)" :
                       x isa AbstractArray ? "$n = map" : "$n = function of others"
    list = isempty(r.fixed) ? "none" : join(map(describe, collect(pairs(r.fixed))), ", ")
    print(io, "\n  fixed    ", list)
end
