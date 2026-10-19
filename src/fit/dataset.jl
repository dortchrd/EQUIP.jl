# =================================================================================================
# Fitting a dataset: every voxel, spread over threads (and `refit`, at the end of this file)
#
#   fit(m, data::AbstractArray)  many voxels, observations along the LAST dimension
#                                (voxels × M, X×Y×M, X×Y×Z×M, ...): r.params.T2 is a map
#
# The one-voxel method, fit(m, y::AbstractVector), is in voxel.jl. Julia picks between the two
# by the shape of the data (multiple dispatch): a Vector matches both signatures, and Julia picks
# the more specific one (AbstractVector), so 1-D data always means "one voxel". Both return a
# FitResult (result.jl), with the same fields.
#
# The dataset is never copied. `reshape(data, nvox, M)` is a 2-D *view* of the same memory: row v
# holds voxel v's M observations. A mask becomes a list of voxel indices.
#
# Threads: the list of voxels is cut into chunks; each chunk runs as a task with its own copy of
# the cache (its own buffers). Many chunks per thread balance the load when some voxels need
# more iterations than others. Start Julia with threads (`julia -t auto`) to use them.
# =================================================================================================

"""
    fit(model, data::AbstractArray; mask = nothing, precision, solver, params...) -> FitResult

Fit every voxel of `data`, an array with the observations along the LAST dimension: voxels × M,
X×Y×M, X×Y×Z×M, ... (as NIfTI and MATLAB store volumes). A matrix is voxels × observations: one
row per voxel. Returns a [`FitResult`](@ref) whose fields are maps with the spatial size of
`data` (all dimensions but the last):

- `params`: one map per free parameter, e.g. `res.params.T1`
- `sse`: sum of squared residuals
- `iterations`: solver iterations
- `status`: [`FitStatus`](@ref): `NotFitted` outside the mask, `InvalidData` for invalid inputs

Voxels that are not fitted hold `NaN`. Keywords:

- parameters, as for one voxel; maps have the spatial size of `data`: `B1 = b1map` (fixed),
  `T1 = Free(start = t1map)` (a start per voxel).
- `mask`: array with one value per voxel; voxels where it is zero/false are skipped. Writing
  `mask` alone after the `;` is Julia's shorthand for `mask = mask`.
- `precision`, `solver`: as for one voxel.

[`refit`](@ref)`(res, data)` fits again, starting from a result's values.

```julia
res = fit(T1MFA(FA = [3, 10, 20], TR = 0.015), data; mask = mask, B1 = b1map)
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
    params, sse, iterations, status = _fit_maps(cache, Y, voxels, spatial)
    settings = merge(cache.settings, (; mask))       # the keywords of this call, mask included
    return FitResult(params, sse, iterations, status, cache.model, settings)
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
    # Outputs: NaN / 0 / NotFitted everywhere, overwritten for the fitted voxels. A status map
    # of FitStatus values takes 1 byte per voxel, like UInt8 codes, but prints names.
    params = NamedTuple{F}(ntuple(_ -> fill(T(NaN), spatial), length(F)))
    sse = fill(T(NaN), spatial)
    iterations = zeros(Int32, spatial)
    status = fill(NotFitted, spatial)

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
            Threads.@spawn _fit_chunk!(params, sse, iterations, status, _copycache(cache), Y, chunk)
        end
    end
    return params, sse, iterations, status
end

function _fit_chunk!(params, sse, iterations, status, cache, Y, voxels)
    for v in voxels
        r = _fit_voxel(cache, cache.spec, view(Y, v, :), v)
        # Store each free parameter in its map: `map` walks the two named tuples (maps and
        # fitted values) together, name by name.
        map((a, x) -> (@inbounds a[v] = x), params, r.params)
        @inbounds sse[v] = r.sse
        @inbounds iterations[v] = r.iterations
        @inbounds status[v] = r.status
    end
    return nothing
end

"""
    refit(res, data; changes...) -> FitResult

Fit `data` again with the model and settings of an earlier result `res`, starting every free
parameter from its value in `res` (numbers for one voxel, maps for a dataset): to polish a fit,
or to see the effect of one change. Keywords change settings, as for [`fit`](@ref), and win
over the stored ones; a parameter changed by a keyword starts from what that keyword says.

```julia
res = fit(m, data; mask = mask)
res2 = refit(res, data; kmf = 15.0)        # the same fit with kmf fixed at 15, from res's maps
```

Voxels without a value in `res` (NaN: outside its mask, or invalid) have no start, and are
marked invalid; fit them with [`fit`](@ref) instead.
"""
function refit(res::FitResult, data; changes...)
    settings = merge(res.settings, (; changes...))           # later keywords win
    starts = Pair{Symbol,Free}[]
    for n in keys(res.params)                                # the parameters fitted in `res`
        haskey(changes, n) && continue                       # changed: as the keyword says
        current = get(settings, n, Free())                   # this fit's role for n, if given
        current isa Free || continue
        push!(starts, n => Free(res.params[n], current.lower, current.upper))
    end
    return fit(res.model, data; merge(settings, (; starts...))...)   # the starts win
end
