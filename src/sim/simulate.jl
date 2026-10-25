# =================================================================================================
# simulate: noisy data from a model, for Monte Carlo, phantoms, benchmarks and tests
#
# The noise level is set by `snr` or `σ`:
#   snr   relative to the model's scale parameter (M0, S0, M0f: the parameter declared
#         `Free(start = :scale)`), σ = scale / snr: the "SNR at thermal equilibrium" of Dortch
#         et al. 2018. It does not depend on the protocol, as on a scanner, where σ is set by the
#         hardware.
#   σ     absolute, in the units of the signal (e.g. estimated from an image's background); one
#         number, or one per observation.
#   Σ     the covariance matrix of the noise between observations (correlated noise).
# =================================================================================================

"""
    simulate(m, truth; snr or σ or Σ, noise, n, rng) -> data

Noisy data for model `m`, with the observations along the last dimension (the layout `fit`
takes).

- `truth`: a value for every parameter of the model, as for [`signal`](@ref)`(m, truth)` (see
  [`nominal`](@ref) for every value from the model or a tissue, [`defaults`](@ref)`(m)` for the
  model's fitting defaults alone). Each value is a number, a map, or a tie (`R1m = :R1f`);
  numbers and ties apply to every voxel, and all maps have the same size.
- noise level, one of:
  - `snr`: relative to the model's scale parameter (`M0`, `S0`, `M0f`), σ = scale/snr, voxel by
    voxel (the SNR at thermal equilibrium, as in Dortch et al., MRM 2018);
  - `σ`: absolute, in the units of the signal, the same in every voxel: a number, or one SD per
    observation (independent noise, e.g. different bandwidths or numbers of averages);
  - `Σ`: the covariance matrix of the noise between observations (correlated noise), the same
    in every voxel. For Rician noise, the real and imaginary channels each have covariance Σ.
- `noise`: `:rician` (magnitude images: |S + σ·(n₁ + i·n₂)|), or `:gaussian` (S + σ·n, for
  signed data). The default is `:gaussian` for a model with `magnitude = false`, else `:rician`.
- `n`: number of noisy copies when no value in `truth` is a map (Monte Carlo).
- `rng`: the random number generator, e.g. `MersenneTwister(1)` for repeatable data.

The result is a vector (one voxel), an `n × M` matrix (`n` copies), or an array with the size
of the maps plus a last dimension of `M` observations. Voxels with a NaN parameter (e.g. the
background of a [`phantom`](@ref)) give NaN data.

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
y = simulate(m, (S0 = 1000.0, T2 = 0.08); snr = 50)                      # one voxel
Y = simulate(m, (S0 = 1000.0, T2 = 0.08); snr = 50, n = 10_000)          # Monte Carlo
data = simulate(m, phantom(labelshapes((64, 64)), tissues); snr = 50)    # a phantom
```
"""
function simulate(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
                  noise::Symbol = _default_noise(m), n::Union{Nothing,Integer} = nothing,
                  rng::AbstractRNG = Random.default_rng())
    count(!isnothing, (snr, σ, Σ)) == 1 ||
        throw(ArgumentError("give the noise level as `snr`, `σ` or `Σ` (one of them)"))
    noise in (:rician, :gaussian) || throw(ArgumentError("`noise` must be :rician or :gaussian"))
    scale = _scalename(m)
    snr !== nothing && scale === nothing &&
        throw(ArgumentError("$(nameof(typeof(m))) has no `Free(start = :scale)` parameter to " *
                            "define the SNR; give `σ`"))

    # The voxels: the size of the maps, or n copies, or one voxel.
    maps = filter(x -> x isa AbstractArray, Tuple(values(truth)))
    sz = isempty(maps) ? (n === nothing ? () : (Int(n),)) : size(first(maps))
    all(x -> size(x) == sz, maps) || throw(DimensionMismatch("all maps in `truth` must have the same size"))
    isempty(maps) || n === nothing ||
        throw(ArgumentError("`n` is for numbers only: with maps, the maps give the voxels"))

    M = nobs(m)
    σ isa AbstractVector && length(σ) != M &&
        throw(DimensionMismatch("`σ` has $(length(σ)) values; the model has $M observations"))
    L = Σ === nothing ? nothing : _noisefactor(Σ, M)   # Σ = LLᵀ: correlated noise is L·z
    out = Array{Float64}(undef, sz..., M)
    Y = reshape(out, :, M)                           # a view: row v = voxel v's observations
    for v in axes(Y, 1)
        p = map(x -> x isa AbstractArray ? x[v] : x, truth)    # this voxel's parameters
        s = signal(m, p)
        if L === nothing
            # Independent noise: one σ, or one per observation. (The order of the random draws
            # is the same either way, so one σ given as a vector gives the same data.)
            σv = σ === nothing ? p[scale] / snr : σ
            for j in 1:M
                σj = σv isa AbstractVector ? σv[j] : σv
                Y[v, j] = noise === :rician ? hypot(s[j] + σj * randn(rng), σj * randn(rng)) :
                                              s[j] + σj * randn(rng)
            end
        else
            # Correlated noise: L·z, z independent standard normal, has covariance LLᵀ = Σ.
            e1 = L * randn(rng, M)
            if noise === :rician
                e2 = L * randn(rng, M)
                Y[v, :] .= hypot.(s .+ e1, e2)
            else
                Y[v, :] .= s .+ e1
            end
        end
    end
    return out
end

# The lower Cholesky factor of a noise covariance matrix, checked: M×M, symmetric, positive
# definite.
function _noisefactor(Σ::AbstractMatrix, M)
    size(Σ) == (M, M) ||
        throw(DimensionMismatch("`Σ` is $(join(size(Σ), "×")); the model has $M observations"))
    isapprox(Σ, Σ'; rtol = 1e-10) || throw(ArgumentError("`Σ` must be symmetric"))
    c = cholesky(Symmetric(Matrix{Float64}(Σ)); check = false)
    issuccess(c) || throw(ArgumentError("`Σ` must be positive definite"))
    return Matrix(c.L)
end
_noisefactor(Σ, M) = throw(ArgumentError("`Σ` is the noise covariance matrix, $M×$M"))

# Magnitude images get Rician noise; models fitted to signed data (magnitude = false) Gaussian.
_default_noise(m) = hasproperty(m, :magnitude) && !m.magnitude ? :gaussian : :rician
