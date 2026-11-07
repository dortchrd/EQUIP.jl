# =================================================================================================
# simulate: noisy data from a model, for Monte Carlo, maps, benchmarks and tests
#
# The noise level (`snr`, `σ` or `Σ`) is checked and turned into what is drawn by the noise model,
# sim/noise.jl, which `crlb` shares.
# =================================================================================================

"""
    simulate(m, truth; snr or σ or Σ, noise, averages, averaging, n, rng, precision) -> data

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
- `noise`: the distribution, always given: `:rician` (magnitude images: |S + σ·(n₁ + i·n₂)|)
  or `:gaussian` (S + σ·n: real-valued data, e.g. phase-corrected; the noise crlb assumes). It
  follows from how the images were reconstructed, which the model does not know.
- `averages`: the number of averages (NEX) of each observation, or one number for all, with
  `snr` or `σ` given for one average: observation j's noise is σ/√NEXⱼ (more averages at high
  b-values, say). Not with `Σ`, which is the noise of the data as fitted.
- `averaging`: how the averages were combined, given with Rician noise and `averages` (only
  there does it change the data): `:complex` (the complex images averaged, then the magnitude:
  Rician with σ/√NEX, so the noise floor falls with √NEX) or `:magnitude` (the magnitude images
  averaged, as in most diffusion: the mean of NEX Rician values with one average's σ, whose SD
  falls with √NEX but whose floor does not).
- `n`: number of noisy copies when no value in `truth` is a map (Monte Carlo).
- `rng`: the random number generator, e.g. `MersenneTwister(1)` for repeatable data.
- `precision`: the data's number type, `Float64` (the default) or `Float32` (half the memory,
  for large simulated datasets). The signal and noise are computed in Float64 either way and
  then stored, so the same seed gives the same data, rounded: `simulate(...; precision =
  Float32) == Float32.(simulate(...))`. A fit converts the data to its own precision anyway.

The result is a vector (one voxel), an `n × M` matrix (`n` copies), or an array with the size
of the maps plus a last dimension of `M` observations. Voxels with a NaN parameter (e.g. a
map's background) give NaN data.

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
y = simulate(m, (S0 = 1000.0, T2 = 0.08); snr = 50, noise = :rician)                # one voxel
Y = simulate(m, (S0 = 1000.0, T2 = 0.08); snr = 50, noise = :rician, n = 10_000)    # Monte Carlo
maps = (S0 = fill(1000.0, 64, 64), T2 = 0.04 .+ 0.08 .* rand(64, 64))
data = simulate(m, maps; snr = 50, noise = :gaussian)                                # real data
md = ADCDecay(b = [0, 1000, 2000, 3000])
simulate(md, (S0 = 1.0, ADC = 8e-4); snr = 20, averages = [1, 2, 4, 8], noise = :rician,
         averaging = :magnitude, n = 10_000)                     # magnitude images averaged
```
"""
function simulate(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
                  noise = nothing, averages = nothing, averaging = nothing,
                  n::Union{Nothing,Integer} = nothing, rng::AbstractRNG = Random.default_rng(),
                  precision::Type{<:AbstractFloat} = Float64)
    # The noise level, checked (sim/noise.jl): the scale parameter `snr` is relative to, and
    # Σ's Cholesky factor, Σ = LLᵀ (correlated noise is L·z).
    M = nobs(m)
    scale, L = _checked_noise(m, M, snr, σ, Σ, averages)
    noise = _checked_kind(noise)             # :rician or :gaussian, given (no default)
    averaging = _checked_averaging(averaging, noise, averages)
    nex = averages === nothing ? nothing : _averages(averages, M)

    # The truth, checked once, as `resolve` checks one voxel's (every parameter, no roles, ties
    # that name a parameter): in the model's order, with ties by name as `_Tie`s, so each voxel
    # only reads its numbers and follows its ties (`_voxel`, below). A truth given by derived
    # quantities (the model's `fromderived`: DTI's MD, FA and direction) becomes its parameters.
    truth = _fromderived(m, truth, keys(parameters(m)))
    q = map(_tie, _complete(nameof(typeof(m)), parameters(m), truth))

    # The voxels: the size of the maps, or n copies, or one voxel.
    maps = filter(x -> x isa AbstractArray, Tuple(values(q)))
    sz = isempty(maps) ? (n === nothing ? () : (Int(n),)) : size(first(maps))
    all(x -> size(x) == sz, maps) || throw(DimensionMismatch("all maps in `truth` must have the same size"))
    isempty(maps) || n === nothing ||
        throw(ArgumentError("`n` is for numbers only: with maps, the maps give the voxels"))

    out = Array{precision}(undef, sz..., M)          # computed in Float64, stored in `precision`
    Y = reshape(out, :, M)                           # a view: row v = voxel v's observations
    isempty(Y) && return out
    _simulate!(Y, m, q, scale === nothing ? nothing : Val(scale), snr, σ, L, noise, averaging,
               nex, rng)
    return out
end

# The voxels, one at a time, into the rows of Y. A separate function (a "function barrier"):
# Julia compiles it for the types of its arguments (the truth's, the noise level's, `averaging`
# a Symbol or nothing), which the caller only knows at run time, so the loop below is compiled
# code with known types. Nothing is allocated per voxel: one signal buffer, written by `signal!`
# (the model's own function, as `fit` calls it), and for correlated noise three more, of M
# numbers each. The random numbers are drawn in the same order as one call of `signal` and
# `randn` per voxel would (the same data, draw for draw).
function _simulate!(Y, m, q, scale, snr, σ, L, noise, averaging, nex, rng)
    M = size(Y, 2)
    s = Vector{_signaltype(m, _voxel(q, 1))}(undef, M)          # the signal, voxel by voxel
    k = L === nothing ? 0 : M                    # correlated noise only: L·z, Σ = LLᵀ
    z, e1, e2 = Vector{Float64}(undef, k), Vector{Float64}(undef, k), Vector{Float64}(undef, k)
    for v in axes(Y, 1)
        p = _voxel(q, v)                                         # this voxel's parameters
        signal!(s, m, p)
        if L === nothing
            # Independent noise: one σ, or one per observation (one average's, with
            # `averages`). (The order of the random draws is the same either way, so one σ given
            # as a vector gives the same data; so do `averages` combined in complex and
            # σ/√NEX given as a vector.)
            σv = σ === nothing ? getfield(p, _name(scale)) / snr : σ
            for j in 1:M
                σj = σv isa AbstractVector ? σv[j] : σv
                if averaging === :magnitude
                    # Magnitude images averaged: the mean of NEXⱼ Rician values, each with one
                    # average's σ (its floor does not fall with the averages).
                    acc = 0.0
                    for _ in 1:nex[j]
                        acc += hypot(s[j] + σj * randn(rng), σj * randn(rng))
                    end
                    Y[v, j] = acc / nex[j]
                else
                    # One image, or the average of complex (or real) images: σ/√NEX.
                    nex === nothing || (σj = σj / sqrt(nex[j]))
                    Y[v, j] = noise === :rician ? hypot(s[j] + σj * randn(rng), σj * randn(rng)) :
                                                  s[j] + σj * randn(rng)
                end
            end
        else
            # Correlated noise: L·z, z independent standard normal, has covariance LLᵀ = Σ.
            # (`mul!` writes L·z into e1 without a new vector; `randn!` fills z.)
            mul!(e1, L, Random.randn!(rng, z))
            if noise === :rician
                mul!(e2, L, Random.randn!(rng, z))
                @views Y[v, :] .= hypot.(s .+ e1, e2)
            else
                @views Y[v, :] .= s .+ e1
            end
        end
    end
    return Y
end

# Voxel v's parameters: the maps read at v, numbers kept, then the ties followed (`R1m = :R1f`:
# R1f's value in this voxel). Both are `map`s over named tuples, which the compiler unrolls
# field by field: no loop, no allocation.
@inline _voxel(q::NamedTuple, v) = _resolve_ties(map(x -> x isa AbstractArray ? x[v] : x, q))

# The scale parameter's name, from `Val(name)`: a type parameter, so `getfield(p, name)` is
# compiled to a plain field read (with a Symbol known only at run time it would be a lookup).
_name(::Val{S}) where {S} = S
