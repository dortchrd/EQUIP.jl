# =================================================================================================
# fit: the one function users call
#
# Two methods, chosen by the shape of the data (multiple dispatch):
#
#   fit(m, y::AbstractVector)    one voxel  -> VoxelResult: r.params.T2 is a number
#   fit(m, data::AbstractArray)  many voxels, observations along the LAST dimension
#                                (voxels × M, X×Y×M, X×Y×Z×M, ...) -> maps: r.params.T2 is an array
#
# A Vector matches both signatures; Julia picks the more specific one (AbstractVector), so 1-D
# data always means "one voxel".
#
# `fit` and `fit!` are not our own functions: they belong to StatsAPI.jl (the interface package
# behind StatsBase, GLM, Distributions). We add methods for our types, so `using MRISignalModels,
# StatsBase` works without a name clash.
#
# Many voxels: the dataset is never copied. `reshape(data, nvox, M)` is a 2-D *view* of the same
# memory: row v holds voxel v's M observations. A mask becomes a list of voxel indices.
#
# Threads: the list of voxels is cut into chunks; each chunk runs as a task with its own copy of
# the cache (its own buffers). Many chunks per thread balance the load when some voxels need
# more iterations than others. Start Julia with threads (`julia -t auto`) to use them.
# =================================================================================================

"""
    fit(model, y::AbstractVector; fixed, free, start, precision, maxiter, ...) -> VoxelResult

Fit one voxel: `y` has one value per observation. Returns a [`VoxelResult`](@ref) with plain
numbers, e.g. `r.params.T1`, `r.status`. Keywords as for [`init`](@ref).

For many single-voxel fits in a loop, make the cache once with `init` and call `fit!` instead.

```julia
m = SIRFSE(TI = [0.010, 0.050, 0.277, 0.843], TD = 2.5)
r = fit(m, y; fixed = (kmf = 15.0,))
r.params.PSR
```
"""
fit(model::MRIModel, y::AbstractVector; kwargs...) = fit!(init(model; kwargs...), y)

"""
    fit(model, data::AbstractArray; mask = nothing, fixed, free, start, precision, maxiter, ...)

Fit every voxel of `data`, an array with the observations along the LAST dimension: voxels × M,
X×Y×M, X×Y×Z×M, ... (as NIfTI and MATLAB store volumes). A matrix is voxels × observations: one
row per voxel. Returns a named tuple of arrays with the spatial size of `data` (all dimensions
but the last):

- `params`: one map per free parameter, e.g. `res.params.T1`
- `sse`: sum of squared residuals
- `status`: [`FitStatus`](@ref) codes (`UInt8`); 0 outside the mask, 6 for invalid inputs

Voxels that are not fitted hold `NaN`. Keywords:

- `mask`: array with one value per voxel; voxels where it is zero/false are skipped.
- `fixed`, `free`, `start`, `precision`, solver settings: as for [`init`](@ref). Per-voxel maps
  have the spatial size of `data`, e.g. `fixed = (B1 = b1map,)`, `start = (M0 = m0map,)`.

`res.params` can be passed back as `start` to refit from a previous result.

```julia
res = fit(MFA(FA = [3, 10, 20], TR = 0.015), data; mask, fixed = (B1 = b1map,))
res.params.T1
```
"""
function fit(model::MRIModel, data::AbstractArray; mask = nothing, kwargs...)
    cache = init(model; kwargs...)                   # setup once; all other keywords go to init
    spatial = Base.front(size(data))                 # size without the last dimension
    nvox = prod(spatial)
    m = size(data, ndims(data))
    m == length(cache.y) ||
        throw(DimensionMismatch("the model has $(length(cache.y)) observations, the data have $m"))
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
function _fit_maps(cache::FitCache{M,S,T}, Y, voxels, spatial) where {M,S,T}
    F = freenames(cache.spec)
    # Outputs: NaN / NotFitted everywhere, overwritten for the fitted voxels.
    params = NamedTuple{F}(ntuple(_ -> fill(T(NaN), spatial), length(F)))
    sse = fill(T(NaN), spatial)
    status = fill(UInt8(NotFitted), spatial)

    n = length(voxels)
    if n > 0
        # Chunk size: about 64 chunks per thread, but at least 16 voxels per chunk.
        # - Many chunks: the run ends when the last chunk ends, so the wait at the end is about
        #   one chunk's duration. Small chunks keep that short when slow voxels (noise,
        #   background) are clustered in one region.
        # - Not too small: each task costs ~1 µs to create and schedule, compared with a few µs
        #   per voxel fit, so a chunk needs at least a few dozen voxels.
        # (Measured, 4 threads, 200k voxels with a clustered slow quarter: 1 chunk/thread 546 ms,
        # 8/thread 258 ms, 64/thread 218 ms; 2k voxels with 1 voxel/chunk: 2.4× slower.)
        chunksize = max(16, cld(n, 64 * Threads.nthreads()))
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
