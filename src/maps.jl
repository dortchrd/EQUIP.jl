# =================================================================================================
# fit_maps: fit every voxel of a dataset, in parallel
#
# Data layout: observations along the LAST dimension, X×Y×Z×M (as NIfTI and MATLAB store them;
# any number of spatial dimensions works, including a plain voxels×M matrix).
#
# The dataset is never copied. `reshape(data, nvox, M)` is a 2-D *view* of the same memory:
# row v holds voxel v's M observations. A mask becomes a list of voxel indices.
#
# Threads: the list of voxels is cut into chunks; each chunk runs as a task with its own copy of
# the cache (its own buffers). Several chunks per thread balance the load when some voxels need
# more iterations than others. Start Julia with threads (`julia -t auto`) to use them.
# =================================================================================================

"""
    fit_maps(model, acq, data; mask = nothing, fixed, free, start, precision, maxiter, ...)

Fit `model` to every voxel of `data` (observations along the last dimension). Returns a named
tuple of arrays with the spatial size of `data`:

- `params`: one map per free parameter, e.g. `res.params.T1`
- `sse`: sum of squared residuals
- `status`: [`FitStatus`](@ref) codes (`UInt8`); 0 outside the mask, 6 for invalid inputs

Voxels that are not fitted hold `NaN`. Keywords:

- `mask`: array with one value per voxel; voxels where it is zero/false are skipped.
- `fixed`, `free`, `start`, `precision`, solver settings: as for [`init`](@ref). Per-voxel maps
  have the spatial size of `data`, e.g. `fixed = (B1 = b1map,)`, `start = (M0 = m0map,)`.

`res.params` can be passed back as `start` to refit from a previous result.

```julia
res = fit_maps(MFA(), (FA = [3, 10, 20], TR = 0.015), data; mask, fixed = (B1 = b1map,))
res.params.T1
```
"""
function fit_maps(model::AbstractModel, acq::NamedTuple, data::AbstractArray; mask = nothing, kwargs...)
    cache = init(model, acq; kwargs...)              # setup once; all other keywords go to init
    spatial = Base.front(size(data))                 # size without the last dimension
    nvox = prod(spatial)
    m = size(data, ndims(data))
    m == length(cache.y) ||
        throw(DimensionMismatch("the acquisition has $(length(cache.y)) observations, the data have $m"))
    _check_maps(cache.spec.fixed, nvox, "fixed")
    _check_maps(cache.spec.start, nvox, "start")
    voxels = if mask === nothing
        1:nvox                                       # every voxel
    else
        length(mask) == nvox || throw(DimensionMismatch("the mask has $(length(mask)) voxels, the data have $nvox"))
        findall(!iszero, vec(mask))                  # linear indices of the voxels to fit
    end
    Y = reshape(data, nvox, m)                       # a view: row v = voxel v's observations
    return _fit_maps(cache, Y, voxels, spatial)
end

function _check_maps(nt::NamedTuple, nvox, kind)
    for (name, x) in pairs(nt)
        x isa AbstractArray && length(x) != nvox &&
            throw(DimensionMismatch("$kind map `$name` has $(length(x)) voxels, the data have $nvox"))
    end
end

# The work, behind a "function barrier": the cache's concrete type is known inside this
# function, so the loops below compile to fast specialized code.
function _fit_maps(cache::FitCache{M,A,S,T}, Y, voxels, spatial) where {M,A,S,T}
    F = freenames(cache.spec)
    # Outputs: NaN / NotFitted everywhere, overwritten for the fitted voxels.
    params = NamedTuple{F}(ntuple(_ -> fill(T(NaN), spatial), length(F)))
    sse = fill(T(NaN), spatial)
    status = fill(UInt8(NotFitted), spatial)

    n = length(voxels)
    if n > 0
        chunksize = cld(n, 8 * Threads.nthreads())   # about 8 chunks per thread
        # `@sync` waits for every task `Threads.@spawn`ed inside it. Each task writes different
        # voxels of the output arrays, so they never write to the same element.
        @sync for chunk in Iterators.partition(voxels, chunksize)
            Threads.@spawn _fit_chunk!(params, sse, status, _copycache(cache), Y, chunk)
        end
    end
    return (; params, sse, status)                   # same as (params = params, sse = sse, ...)
end

function _fit_chunk!(params, sse, status, cache, Y, voxels)
    for v in voxels
        r = fit!(cache, view(Y, v, :); voxel = v)
        # Store each free parameter in its map: `map` walks the two named tuples (maps and
        # fitted values) together, name by name.
        map((a, x) -> (@inbounds a[v] = x), params, r.params)
        @inbounds sse[v] = r.sse
        @inbounds status[v] = UInt8(r.status)
    end
    return nothing
end
