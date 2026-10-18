# =================================================================================================
# simulate: noisy data from a model, for Monte Carlo, phantoms, benchmarks and tests
#
# The noise level is set by `snr` or `σ`:
#   snr   relative to the model's scale parameter (M0, S0, M0f: the parameter declared
#         `Free(:scale)`), σ = scale / snr: the "SNR at thermal equilibrium" of Dortch et al. 2018.
#         It does not depend on the protocol, as on a scanner, where σ is set by the hardware.
#   σ     absolute, in the units of the signal (e.g. estimated from an image's background).
# =================================================================================================

"""
    simulate(m, truth; snr or σ, noise, n, rng) -> data

Noisy data for model `m`, with the observations along the last dimension (the layout `fit`
takes).

- `truth`: the parameters, as for [`signal`](@ref)`(m, truth)`: the free ones, and fixed ones to
  override. Each value is a number or a map; numbers apply to every voxel, and all maps have
  the same size.
- noise level, one of:
  - `snr`: relative to the model's scale parameter (`M0`, `S0`, `M0f`), σ = scale/snr, voxel by
    voxel (the SNR at thermal equilibrium, as in Dortch et al., MRM 2018);
  - `σ`: absolute, in the units of the signal, the same in every voxel.
- `noise`: `:rician` (magnitude images: |S + σ·(n₁ + i·n₂)|), or `:gaussian` (S + σ·n, for
  signed data). The default is `:gaussian` for a model with `magnitude = false`, else `:rician`.
- `n`: number of noisy copies when every value in `truth` is a number (Monte Carlo).
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
function simulate(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing,
                  noise::Symbol = _default_noise(m), n::Union{Nothing,Integer} = nothing,
                  rng::AbstractRNG = Random.default_rng())
    (snr === nothing) == (σ === nothing) &&
        throw(ArgumentError("give the noise level as `snr` or as `σ` (one of them)"))
    noise in (:rician, :gaussian) || throw(ArgumentError("`noise` must be :rician or :gaussian"))
    scale = _scalename(m)
    snr !== nothing && scale === nothing &&
        throw(ArgumentError("$(nameof(typeof(m))) has no `Free(:scale)` parameter to define the SNR; give `σ`"))

    # The voxels: the size of the maps, or n copies, or one voxel.
    maps = filter(x -> x isa AbstractArray, Tuple(values(truth)))
    sz = isempty(maps) ? (n === nothing ? () : (Int(n),)) : size(first(maps))
    all(x -> size(x) == sz, maps) || throw(DimensionMismatch("all maps in `truth` must have the same size"))
    isempty(maps) || n === nothing ||
        throw(ArgumentError("`n` is for numbers only: with maps, the maps give the voxels"))

    M = nobs(m)
    out = Array{Float64}(undef, sz..., M)
    Y = reshape(out, :, M)                           # a view: row v = voxel v's observations
    for v in axes(Y, 1)
        p = map(x -> x isa AbstractArray ? x[v] : x, truth)    # this voxel's parameters
        s = signal(m, p)
        σv = σ === nothing ? p[scale] / snr : σ
        for j in 1:M
            Y[v, j] = noise === :rician ? hypot(s[j] + σv * randn(rng), σv * randn(rng)) :
                                          s[j] + σv * randn(rng)
        end
    end
    return out
end

# Magnitude images get Rician noise; models fitted to signed data (magnitude = false) Gaussian.
_default_noise(m) = hasproperty(m, :magnitude) && !m.magnitude ? :gaussian : :rician

# The name of the parameter declared `Free(:scale)` (M0, S0, M0f), or nothing.
function _scalename(m::MRIModel)
    roles = parameters(m)
    i = findfirst(r -> r isa Free && r.start === :scale, values(roles))
    return i === nothing ? nothing : keys(roles)[i]
end
