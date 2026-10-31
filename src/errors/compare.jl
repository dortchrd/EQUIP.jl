# =================================================================================================
# compare: the errors of a dataset fit (bias, SD, RMSE) against the truth it was simulated from,
# e.g. simulated maps: what montecarlo measures for one voxel, measured over maps and labels
# =================================================================================================

"""
    Comparison

The result of [`compare`](@ref): `rows`, one per (group, parameter), each a named tuple with

- `group`: `:all`, or a label
- `parameter`, `voxels` (voxels fitted in the group), `converged` (fraction)
- `truth`: the median true value
- `mean`, `bias` (mean of estimate − truth), `relbias` (mean of the relative error,
  (estimate − truth)/|truth|: the sign of the bias, also for a negative truth such as Sf)
- `sd`: standard deviation of the errors (of the estimates, when the truth is one value)
- `rmse`, `relrmse`, `medrel` (median of the absolute relative error)

With one true value for every voxel, the statistics are [`montecarlo`](@ref)'s (`relbias` =
bias/|truth|, as there and in [`crlb`](@ref)); with maps, each voxel's error is relative to its
own true value.

Statistics use every fitted voxel, converged or not. `rows` is a table: e.g.
`DataFrame(c.rows)` with DataFrames.jl.
"""
struct Comparison
    rows::Vector{NamedTuple}
end

"""
    compare(res, truth; labels = nothing, mask = nothing) -> Comparison

Bias, SD, RMSE (absolute and relative) and % converged of a dataset fit, for every free
parameter that `truth` gives (numbers, maps or ties, as for [`simulate`](@ref)). Over all fitted
voxels, and per label when `labels` (an integer map: tissue labels, a segmentation) is given; `mask`
restricts the voxels further. Speed is not measured here (see the benchmarks in `bench/`).
"""
function compare(res::FitResult, truth::NamedTuple; labels = nothing, mask = nothing)
    res.status isa FitStatus && throw(ArgumentError("compare needs a dataset result (maps)"))
    use = (res.status .!= NotFitted) .& (res.status .!= InvalidData)
    mask === nothing || (use = use .& (mask .!= 0))
    conv = isconverged(res)
    groups = Any[(:all, use)]
    if labels !== nothing
        size(labels) == size(res.status) || throw(DimensionMismatch("`labels` must have the size of the maps"))
        for l in sort(unique(labels[use]))
            push!(groups, (l, use .& (labels .== l)))
        end
    end
    rows = NamedTuple[]
    for (g, sel) in groups, n in keys(res.params)
        haskey(truth, n) || continue
        count(sel) == 0 && continue
        est = res.params[n][sel]
        t = truth[n] isa Symbol ? truth[truth[n]] : truth[n]    # a tie (R1m = :R1f): the value it names
        tv = t isa AbstractArray ? t[sel] : fill(float(t), count(sel))
        err = est .- tv
        rel = err ./ abs.(tv)        # relative to |truth|, as crlb and montecarlo: the bias's sign
        push!(rows, (group = g, parameter = n, voxels = count(sel), converged = count(conv .& sel) / count(sel),
                     truth = median(tv), mean = mean(est), bias = mean(err), relbias = mean(rel),
                     sd = length(err) > 1 ? std(err) : NaN, rmse = sqrt(mean(abs2, err)),
                     relrmse = sqrt(mean(abs2, rel)), medrel = median(abs.(rel))))
    end
    return Comparison(rows)
end

function Base.show(io::IO, ::MIME"text/plain", c::Comparison)
    println(io, "Comparison with the truth (", length(c.rows), " rows)")
    f(x) = string(round(x; sigdigits = 4))
    pct(x) = string(round(100 * x; sigdigits = 3), "%")
    header = ("group", "parameter", "voxels", "conv", "truth", "mean", "bias", "rel bias", "SD", "RMSE",
              "rel RMSE")
    cells = [(string(r.group), string(r.parameter), string(r.voxels), pct(r.converged), f(r.truth),
              f(r.mean), f(r.bias), pct(r.relbias), f(r.sd), f(r.rmse), pct(r.relrmse)) for r in c.rows]
    w = [maximum(length, (header[j], (row[j] for row in cells)...)) for j in eachindex(header)]
    print(io, "  ", join((rpad(header[j], w[j]) for j in eachindex(header)), "  "))
    for row in cells
        print(io, "\n  ", join((rpad(row[j], w[j]) for j in eachindex(row)), "  "))
    end
end

Base.show(io::IO, c::Comparison) = print(io, "Comparison(", length(c.rows), " rows)")
