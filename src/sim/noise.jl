# =================================================================================================
# The noise model: what `snr`, `σ` and `Σ` mean, checked in one place for `simulate` (the noise it
# adds) and for `crlb` and the pieces (the noise they assume, `noisecovariance`)
#
# The noise level is one of:
#   snr   relative to the model's scale parameter (M0, S0, M0f: the parameter declared
#         `Free(start = :scale)`), σ = scale / snr: the "SNR at thermal equilibrium" of Dortch
#         et al. 2018. It does not depend on the protocol, as on a scanner, where σ is set by the
#         hardware.
#   σ     absolute, in the units of the signal (e.g. estimated from an image's background); one
#         number, or one per observation.
#   Σ     the covariance matrix of the noise between observations (correlated noise).
# =================================================================================================

"""
    noisecovariance(m, θ; snr or σ or Σ) -> Diagonal or Matrix
    noisecovariance(m, s::Scenario; snr or σ or Σ) -> Diagonal or Matrix

The covariance Σ of Gaussian noise between the observations of `m` (N×N), from one of:

- `snr`: relative to the model's scale parameter at the values θ (a scenario's truth), as for
  [`simulate`](@ref):
  σ = M0/snr, the same for every observation, independent: Σ = σ²I (a `Diagonal`)
- `σ`: a number (Σ = σ²I), or one SD per observation (Σ = Diagonal(σ²): independent noise, e.g.
  different numbers of averages)
- `Σ`: the covariance matrix itself (correlated noise, e.g. after denoising), checked
  (symmetric, positive definite) and returned

```julia
noisecovariance(m, nominal(m, :SIR_WM_I); snr = 200)       # 4×4 Diagonal, (M0f/200)²
noisecovariance(m, s; snr = 200)                           # the same at a scenario's truth
```
"""
function noisecovariance(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, Σ = nothing)
    N = nobs(m)
    scale, _ = _checked_noise(m, N, snr, σ, Σ)
    snr !== nothing && return Diagonal(fill((_scalevalue(m, θ, scale) / snr)^2, N))  # σ = M0/snr
    σ isa Real && return Diagonal(fill(Float64(σ)^2, N))
    σ !== nothing && return Diagonal(Float64.(collect(σ)) .^ 2)
    return Σ
end
# (The scenario's method, `noisecovariance(m, s::Scenario)`, is with the scenario's other pieces:
# errors/derivatives.jl.)

# The checks every noise level gets, the same for `simulate` and `noisecovariance`: one of `snr`,
# `σ` and `Σ`; `snr` needs a scale parameter; `σ` is a number or one SD per observation; `Σ` an
# N×N covariance matrix, checked by factoring it. Returns the scale parameter's name (with `snr`;
# else `nothing`) and Σ's Cholesky factor (with `Σ`; else `nothing`), which `simulate` draws with.
# N = nobs(m) is passed in, as the callers have it already (`nobs` collects the acquisition's
# lengths: 0.06 µs and 144 bytes a call, a tenth of a 4-point `crlb!`'s noise).
function _checked_noise(m::MRIModel, N, snr, σ, Σ)
    count(!isnothing, (snr, σ, Σ)) == 1 ||
        throw(ArgumentError("give the noise level as `snr`, `σ` or `Σ` (one of them)"))
    scale = snr === nothing ? nothing : _scalename(m)
    snr !== nothing && scale === nothing &&
        throw(ArgumentError("$(nameof(typeof(m))) has no `Free(start = :scale)` parameter to " *
                            "define the SNR; give `σ`"))
    if σ !== nothing && !(σ isa Real)
        σ isa AbstractVector || throw(ArgumentError("`σ` is a number or one SD per observation; " *
                                                    "for a covariance matrix, give `Σ`"))
        length(σ) == N ||
            throw(DimensionMismatch("`σ` has $(length(σ)) values; the model has $N observations"))
    end
    return scale, Σ === nothing ? nothing : _noisefactor(Σ, N)
end

# The scale parameter's value (M0, S0): read directly when given as a number; otherwise (a tie,
# or missing: then `resolve` says what is wrong) through `resolve`, which checks every value.
_scalevalue(m, θ, scale) =
    haskey(θ, scale) && θ[scale] isa Real ? Float64(θ[scale]) : Float64(resolve(m, θ)[scale])

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

# The noise's distribution: the user's choice, with no default. It is decided by how the images
# were made, which the model does not know: magnitude images have Rician noise,
# |S + σ(n₁ + i·n₂)|; real-valued data (phase-corrected, the real channel) Gaussian, S + σn, the
# noise crlb assumes. (It was inferred from the model's `magnitude` option, Rician unless
# `magnitude = false`; but that option says how the signal is computed, not how the data were
# reconstructed, and most models have no such option.)
function _checked_kind(noise)
    noise === nothing && throw(ArgumentError("give the noise's distribution: `noise = :rician` " *
        "(magnitude images) or `noise = :gaussian` (real-valued data; the noise crlb assumes)"))
    noise in (:rician, :gaussian) ||
        throw(ArgumentError("`noise` must be :rician or :gaussian, got $(repr(noise))"))
    return noise
end
