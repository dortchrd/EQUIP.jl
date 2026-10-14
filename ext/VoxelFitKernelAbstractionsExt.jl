# GPU (and other KernelAbstractions backends) for fit_maps.
#
# One work item fits one voxel, running exactly the same lm_fit code as the CPU path. Data are
# staged in voxel-major batches (batch × nobs), so adjacent work items read adjacent addresses at
# every observation index: coalesced loads. Batching bounds device memory use for any dataset
# size. Pointwise models only (curve models need per-voxel scratch buffers).

module VoxelFitKernelAbstractionsExt

using VoxelFit
using VoxelFit: AbstractModel, Pointwise, ModelInit, ConstInit, MapInit, Covariates,
                VoxelMajor, ObsMajor, InvalidData, lm_fit, standard_errors, bind_covariates,
                _covariate, _start, _allfinite_data, modelstyle
using KernelAbstractions
using StaticArrays

# Wrappers holding arrays are rebuilt inside the kernel from raw device arrays, so that the
# backend's argument conversion (host array -> device array) applies to every array.
@inline _mkinit(init, ::Nothing) = init
@inline _mkinit(init, A) = MapInit(A)
@inline _mkcov(::Nothing, ::Val) = nothing
@inline _mkcov(C, ::Val{K}) where {K} = Covariates{K,typeof(C)}(C)

@kernel function _lm_kernel!(θout, sse, iters, status, se, @Const(Y), @Const(x), model, init0,
                             initA, lo, hi, opts, w, C, valK, n)
    k = @index(Global, Linear)
    if k <= n
        init = _mkinit(init0, initA)
        covs = _mkcov(C, valK)
        T = eltype(Y)
        P = length(lo)
        y = view(Y, k, :)
        if !_allfinite_data(y)
            # Overwrite every output: device buffers are reused across batches.
            @inbounds begin
                for j in 1:P
                    θout[k, j] = T(NaN)
                    se === nothing || (se[k, j] = T(NaN))
                end
                sse[k] = T(NaN)
                iters[k] = Int32(0)
                status[k] = UInt8(InvalidData)
            end
        else
            m = bind_covariates(model, _covariate(covs, k, T))
            θ0 = _start(init, m, x, y, k, lo, hi)
            r = lm_fit(m, x, y, θ0, lo, hi, opts, w, nothing)
            @inbounds begin
                for j in 1:P
                    θout[k, j] = r.θ[j]
                end
                sse[k] = r.sse
                iters[k] = r.iterations
                status[k] = UInt8(r.status)
                if se !== nothing
                    s = standard_errors(r, length(y))
                    for j in 1:P
                        se[k, j] = s[j]
                    end
                end
            end
        end
    end
end

_device(backend, a::AbstractArray) = copyto!(KernelAbstractions.allocate(backend, eltype(a), size(a)), a)
_device(backend, ::Nothing) = nothing

function VoxelFit._fit_backend!(backend::KernelAbstractions.Backend, out, model::AbstractModel,
                                xs, Y, layout, idx, initspec, lo::SVector{P,T}, hi, opts, w,
                                covs, batchsize) where {P,T}
    modelstyle(model) isa Pointwise ||
        throw(ArgumentError("GPU fitting supports pointwise models only"))
    isbitstype(typeof(model)) ||
        throw(ArgumentError("model $(typeof(model)) is not isbits; it cannot be passed to a GPU kernel"))
    n = length(idx)
    n == 0 && return out
    nobs = length(xs)
    bs = min(batchsize, n)
    alloc(S, dims...) = KernelAbstractions.allocate(backend, S, dims)

    xd = _device(backend, xs)
    wd = _device(backend, w)
    Yh, Yd = Matrix{T}(undef, bs, nobs), alloc(T, bs, nobs)
    θh, θd = Matrix{T}(undef, bs, P), alloc(T, bs, P)
    sseh, ssed = Vector{T}(undef, bs), alloc(T, bs)
    ith, itd = Vector{Int32}(undef, bs), alloc(Int32, bs)
    sth, std = Vector{UInt8}(undef, bs), alloc(UInt8, bs)
    seh, sed = out.stderr === nothing ? (nothing, nothing) : (Matrix{T}(undef, bs, P), alloc(T, bs, P))
    Ih, Id = initspec isa MapInit ? (Matrix{T}(undef, bs, P), alloc(T, bs, P)) : (nothing, nothing)
    K = covs === nothing ? 0 : size(covs.C, 2)
    Ch, Cd = covs === nothing ? (nothing, nothing) : (Matrix{T}(undef, bs, K), alloc(T, bs, K))
    init0 = initspec isa MapInit ? nothing : initspec

    kernel! = _lm_kernel!(backend)
    for start in 1:bs:n
        r = start:min(start + bs - 1, n)
        nb = length(r)
        _stage!(Yh, Y, layout, idx, r)
        copyto!(Yd, Yh)
        if Ih !== nothing
            @inbounds for j in 1:P, (k, kk) in enumerate(r)
                Ih[k, j] = initspec.A[idx[kk], j]
            end
            copyto!(Id, Ih)
        end
        if Ch !== nothing
            @inbounds for j in 1:K, (k, kk) in enumerate(r)
                Ch[k, j] = covs.C[idx[kk], j]
            end
            copyto!(Cd, Ch)
        end
        kernel!(θd, ssed, itd, std, sed, Yd, xd, model, init0, Id, lo, hi, opts, wd, Cd, Val(K), nb;
                ndrange=nb)
        KernelAbstractions.synchronize(backend)
        copyto!(θh, θd); copyto!(sseh, ssed); copyto!(ith, itd); copyto!(sth, std)
        seh === nothing || copyto!(seh, sed)
        @inbounds for (k, kk) in enumerate(r)
            v = idx[kk]
            for j in 1:P
                out.params[v, j] = θh[k, j]
            end
            out.sse[v] = sseh[k]
            out.iterations[v] = ith[k]
            out.status[v] = sth[k]
            if seh !== nothing
                for j in 1:P
                    out.stderr[v, j] = seh[k, j]
                end
            end
        end
    end
    return out
end

function _stage!(Yh::Matrix{T}, Y, ::VoxelMajor, idx, r) where {T}
    @inbounds for i in axes(Y, 2), (k, kk) in enumerate(r)
        Yh[k, i] = T(Y[idx[kk], i])
    end
    # Rows past the batch end keep stale values; the kernel never reads them.
    return Yh
end
function _stage!(Yh::Matrix{T}, Y, ::ObsMajor, idx, r) where {T}
    @inbounds for (k, kk) in enumerate(r), i in axes(Y, 1)
        Yh[k, i] = T(Y[i, idx[kk]])
    end
    return Yh
end

end # module
