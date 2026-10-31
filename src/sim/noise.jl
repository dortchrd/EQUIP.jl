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
#
# With `averages` (NEX, the number of averages of each observation), `snr` and `σ` are one
# average's noise, and observation j's is σⱼ = σ/√NEXⱼ: the independent noise of NEXⱼ images
# averages down by √NEXⱼ. `Σ` is always the covariance of the data as fitted (after averaging
# and any processing, such as denoising, where correlation between observations mostly comes
# from), so it takes no `averages`. How the averages were combined matters only for Rician
# noise, and only `simulate` can show it (`averaging`, below); to first order in the noise, as
# for crlb, both give σ/√NEX.
# =================================================================================================

"""
    noisecovariance(m, θ; snr or σ or Σ, averages) -> Diagonal or Matrix
    noisecovariance(m, s::Scenario; snr or σ or Σ, averages) -> Diagonal or Matrix

The covariance Σ of Gaussian noise between the observations of `m` (N×N), from one of:

- `snr`: relative to the model's scale parameter at the values θ (a scenario's truth), as for
  [`simulate`](@ref):
  σ = M0/snr, the same for every observation, independent: Σ = σ²I (a `Diagonal`)
- `σ`: a number (Σ = σ²I), or one SD per observation (Σ = Diagonal(σ²): independent noise, e.g.
  different numbers of averages)
- `Σ`: the covariance matrix itself (correlated noise, e.g. after denoising), checked
  (symmetric, positive definite) and returned
- `averages`: the number of averages (NEX) of each observation, or one number for all, with
  `snr` or `σ` (one average's noise): σⱼ² = σ²/NEXⱼ. Not with `Σ`, which is the covariance of
  the averaged data.

```julia
noisecovariance(m, nominal(m, :SIR_WM_I); snr = 200)       # 4×4 Diagonal, (M0f/200)²
noisecovariance(m, s; snr = 200)                           # the same at a scenario's truth
noisecovariance(m, s; snr = 200, averages = [1, 2, 4, 8])   # (M0f/200)²/NEX
```
"""
function noisecovariance(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
                         averages = nothing)
    N = nobs(m)
    scale, _ = _checked_noise(m, N, snr, σ, Σ, averages)
    Σ === nothing || return Σ
    # One average's variance per observation: σ = M0/snr, or σ given (a number or one each).
    v = snr !== nothing ? fill((_scalevalue(m, θ, scale) / snr)^2, N) :
        σ isa Real ? fill(Float64(σ)^2, N) : Float64.(collect(σ)) .^ 2
    averages === nothing || (v ./= _averages(averages, N))      # σⱼ² = σ²/NEXⱼ
    return Diagonal(v)
end
# (The scenario's method, `noisecovariance(m, s::Scenario)`, is with the scenario's other pieces:
# errors/derivatives.jl.)

# The checks every noise level gets, the same for `simulate` and `noisecovariance`: one of `snr`,
# `σ` and `Σ`; `snr` needs a scale parameter; `σ` is a number or one SD per observation; `Σ` an
# N×N covariance matrix, checked by factoring it; `averages` whole numbers ≥ 1, one per
# observation or one for all, not with `Σ`. Returns the scale parameter's name (with `snr`;
# else `nothing`) and Σ's Cholesky factor (with `Σ`; else `nothing`), which `simulate` draws with.
# N = nobs(m) is passed in, as the callers have it already (`nobs` collects the acquisition's
# lengths: 0.06 µs and 144 bytes a call, a tenth of a 4-point `crlb!`'s noise).
function _checked_noise(m::MRIModel, N, snr, σ, Σ, averages = nothing)
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
    averages === nothing || _check_averages(averages, N, Σ)
    return scale, Σ === nothing ? nothing : _noisefactor(Σ, N)
end

# `averages`: the number of averages of each observation (NEX), or one number for all.
function _check_averages(a, N, Σ)
    Σ === nothing || throw(ArgumentError("`averages` with `Σ`: Σ is the covariance of the data " *
        "as fitted, after averaging; give the averaged data's Σ, without `averages`"))
    a isa Real || a isa AbstractVector ||
        throw(ArgumentError("`averages` is one number of averages per observation, or one for all"))
    a isa AbstractVector && length(a) != N &&
        throw(DimensionMismatch("`averages` has $(length(a)) values; the model has $N observations"))
    all(x -> x isa Real && isinteger(x) && x >= 1, a) ||
        throw(ArgumentError("`averages` must be whole numbers, at least 1, got $(repr(a))"))
    return nothing
end
# The averages, checked, as one number per observation.
_averages(a::Real, N) = fill(Int(a), N)
_averages(a::AbstractVector, N) = Int.(collect(a))

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

# How the averages were combined: the user's choice, as the distribution is, and needed only
# where it changes the data, Rician noise with averages:
#   :complex     the complex images averaged, then the magnitude: Rician with σ/√NEX, so the
#                noise floor falls with the averages;
#   :magnitude   the magnitude images averaged (as in most diffusion, where shot-to-shot motion
#                phase rules out complex averaging): the mean of NEX Rician values, each with one
#                average's σ. Its SD falls with √NEX but its floor stays at one average's.
# Real-valued data (Gaussian noise) average the same either way, so `averaging` is refused there,
# as without `averages`.
function _checked_averaging(averaging, noise, averages)
    if averages === nothing
        averaging === nothing || throw(ArgumentError("`averaging` says how the `averages` were " *
            "combined: give `averages` too (the number of averages of each observation)"))
    elseif noise === :rician
        averaging === nothing && throw(ArgumentError("with Rician noise and `averages`, give how " *
            "they were combined: `averaging = :complex` (complex images averaged, then the " *
            "magnitude: the noise floor falls with √NEX) or `averaging = :magnitude` (magnitude " *
            "images averaged, as in most diffusion: the floor stays at one average's)"))
        averaging in (:complex, :magnitude) || throw(ArgumentError("`averaging` must be " *
            ":complex or :magnitude, got $(repr(averaging))"))
    else
        averaging === nothing || throw(ArgumentError("`averaging` is for Rician noise (magnitude " *
            "images); real-valued data (`noise = :gaussian`) average the same either way: leave " *
            "it out"))
    end
    return averaging
end

# Where the distribution is given: where data are made (`simulate`, `montecarlo`), and nowhere
# else. The fit is least squares and crlb is first order in the noise, so both assume Gaussian
# noise whatever the data's; a scenario is the truth and the fit judged against it, not the data.
# Given to any of them anyway (keywords copied from a montecarlo call, say), `noise` is refused
# with this error, which says where the distribution matters instead. The same for `averaging`
# (how averages were combined), which only the distribution of the data shows, and for
# `averages` given to a scenario or the fit.
function _noise_refused(fname, key = :noise)
    key in (:averages, :averaging) && return ArgumentError("`$key`: $fname takes no $key. " * (
        fname == "Scenario" ?
        "A scenario is the truth and the fit judged against it; the noise belongs to the data: " *
        "give `averages` and `averaging` to `montecarlo` or `simulate` (and `averages` to `crlb`)." :
        fname == "fit" ?
        "The fit is unweighted least squares (weights, e.g. from the averages, are on the roadmap); " *
        "how averages were combined matters where data are made (`simulate`, `montecarlo`)." :
        "crlb is first order in the noise, where NEX averages give σ/√NEX however they were " *
        "combined (give `averages`); the noise floor that magnitude averaging keeps is what " *
        "`montecarlo(m, θ; noise = :rician, averages, averaging = :magnitude, ...)` measures."))
    why = fname == "fit" ?
        "The fit is least squares, which assumes Gaussian noise; with magnitude data at low SNR " *
        "the Rician floor biases it, which `montecarlo(m, θ; noise = :rician, ...)` measures " *
        "(a Rician likelihood is on the roadmap)." :
        fname == "Scenario" ?
        "A scenario is the truth and the fit judged against it; the noise belongs to the data: " *
        "give it to `montecarlo(m, s; noise = :rician, ...)` or `simulate`." :
        "crlb and what is built on it are first order in the noise, so they assume Gaussian " *
        "noise, as the least-squares fit does; the Rician floor of magnitude images is what " *
        "`montecarlo(m, θ; noise = :rician, ...)` measures, with crlb alongside."
    return ArgumentError("`noise`: $fname takes no noise distribution. " * why)
end
