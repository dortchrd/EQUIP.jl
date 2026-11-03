# =================================================================================================
# Rician noise: what taking the magnitude of noisy complex data does to its mean, its variance and
# the information it carries about the signal
#
#   ricianmean(A, σ)            E|y|: above A at low SNR (the noise floor, σ√(π/2) at A = 0)
#   ricianvar(A, σ)             Var|y|
#   ricianinformation(A, σ)     the Fisher information about A in one magnitude value, R(A/σ)/σ²
#   RicianMean(m; σ, ...)       a model whose signal is the mean of the magnitude data
#   riciannoise(m, θ; snr, ...) crlb's keywords for a least-squares fit of magnitude data: the
#                               floor's bias (`simulate_with`) and the Rician variance (`Σ`)
#   ricianbound(m, θ; snr, ...) the Cramér–Rao bound of Rician data (Devan et al. 2020, Appendix)
#
# A is the true signal amplitude, σ the noise SD of each of the real and imaginary channels, y the
# measured magnitude (Gudbjartsson & Patz, Magn Reson Med 1995;34:910). With averages: averaged
# as complex data, the magnitude of the mean is Rician with σ/√NEX (the floor falls); averaged as
# magnitudes (a spherical mean over directions is one), the mean of NEX Rician values has the
# same mean E|y| at σ (the floor stays) and the variance Var|y|/NEX.
# =================================================================================================

"""
    ricianmean(A, σ) -> E|y|

The mean of the magnitude of a complex signal of amplitude `A` with Gaussian noise of SD `σ` in
each channel (Rician; Gudbjartsson & Patz, Magn Reson Med 1995):

    E|y| = σ·√(π/2)·L₁/₂(−A²/2σ²) = σ·√(π/2)·[(1 + ν)·I₀(ν/2) + ν·I₁(ν/2)]·e^(−ν/2),  ν = A²/2σ²

(L₁/₂ a Laguerre function, I₀ and I₁ modified Bessel functions, computed scaled: no overflow at
high SNR). σ√(π/2) at A = 0 (the noise floor), A + σ²/(2A) at high SNR. Differentiable in A (dual
numbers), so it can be part of a signal model ([`RicianMean`](@ref)).
"""
function ricianmean(A, σ)
    ν = A^2 / (2σ^2)
    return σ * sqrt(oftype(float(ν), π) / 2) * ((1 + ν) * besselix(0, ν / 2) + ν * besselix(1, ν / 2))
end

"""
    ricianvar(A, σ) -> Var|y|

The variance of the magnitude of a complex signal of amplitude `A` with Gaussian noise of SD `σ`
in each channel: A² + 2σ² − (E|y|)² ([`ricianmean`](@ref)). (2 − π/2)·σ² at A = 0, σ² at high
SNR (above A/σ = 1000, its expansion σ²·(1 − σ²/2A²): the difference loses precision there).
"""
function ricianvar(A, σ)
    s = abs(A) / σ
    s > 1000 && return σ^2 * (1 - 1 / (2s^2))
    return A^2 + 2σ^2 - ricianmean(A, σ)^2
end

"""
    ricianinformation(A, σ) -> R(A/σ)/σ²

The Fisher information about the amplitude `A` in one Rician magnitude value: R(s)/σ², s = A/σ,
with R(s) = E[u²·(I₁(us)/I₀(us))²] − s² over u, the magnitude in units of σ (the factor R(S, σ) of
Devan et al., Magn Reson Imaging 2020;74:56, Appendix, in units of σ; Karlsen et al., Magn Reson
Med 1999). R is 0 at A = 0 (no information), about 0.52 at s = 1, 0.94 at s = 3, and
1 − 1/(2s²) − 1/(4s⁴) at high SNR (Gaussian: 1/σ²). Computed by quadrature over u (Simpson,
1201 points over s ± 12, where the density lies); the expansion above s = 40.
"""
ricianinformation(A::Real, σ::Real) = _ricianR(abs(A) / σ) / σ^2

function _ricianR(s::Real)
    s <= 0 && return 0.0
    s > 40 && return 1 - 1 / (2s^2) - 1 / (4s^4)
    n = 1201
    lo, hi = max(0.0, s - 12), s + 12
    h = (hi - lo) / (n - 1)
    total = 0.0
    for i in 1:n
        u = lo + (i - 1) * h
        z = u * s
        i0 = besselix(0, z)
        r = z == 0 ? 0.0 : besselix(1, z) / i0                # I₁/I₀
        pdf = u * i0 * exp(-(u - s)^2 / 2)                     # the Rician density (σ = 1)
        w = i == 1 || i == n ? 1 : (iseven(i) ? 4 : 2)
        total += w * u^2 * r^2 * pdf
    end
    return total * h / 3 - s^2
end

# -------------------------------------------------------------------------------------------------
"""
    RicianMean(m; σ, averages = nothing, averaging = :complex)

A model whose signal is the mean of the magnitude data that model `m` gives with Rician noise:
[`ricianmean`](@ref) of `m`'s signal (its magnitude), observation by observation. The same
parameters and acquisition as `m`. `σ` is the noise SD of one image, a number or one per
observation. With `averages`: complex averaging (the scanner's, usually) divides σ by √NEX, so
the floor falls; `:magnitude` (a spherical mean over directions is one) leaves the mean, so the
floor, as it is.

As the data a fit is judged against, it gives the bias of a least-squares fit of magnitude data
(the noise floor's): `crlb(m, θ; simulate_with = RicianMean(m; σ) => θ, ...)`, which
[`riciannoise`](@ref) writes for you. Fitted itself, it is least squares on the expected
magnitude, a correction of the floor for any model (with σ known).
"""
struct RicianMean{M<:MRIModel,S<:AbstractVector} <: MRIModel
    model::M
    σ::S                                # each observation's σ, after complex averaging
end

function RicianMean(m::MRIModel; σ, averages = nothing, averaging::Symbol = :complex)
    averaging in (:complex, :magnitude) || throw(ArgumentError("RicianMean: `averaging` is " *
        ":complex (the floor falls with the averages) or :magnitude (it stays), got :$averaging"))
    n = nobs(m)
    σv = σ isa Real ? fill(Float64(σ), n) : Float64.(collect(σ))
    length(σv) == n || throw(DimensionMismatch("RicianMean: $(length(σv)) σ for $n observations"))
    if averages !== nothing && averaging === :complex
        nex = averages isa Real ? fill(averages, n) : collect(averages)
        length(nex) == n || throw(DimensionMismatch("RicianMean: $(length(nex)) averages for " *
                                                    "$n observations"))
        σv = σv ./ sqrt.(nex)
    end
    return RicianMean(m, σv)
end

# The wrapped model's parameters, observations and hooks; the signal through the Rician mean.
parameters(w::RicianMean) = parameters(w.model)
nobs(w::RicianMean) = nobs(w.model)
datastart(w::RicianMean, y, p) = datastart(w.model, y, p)
relabel(w::RicianMean, θ) = relabel(w.model, θ)
_with_precision(w::RicianMean, ::Type{T}) where {T} =
    RicianMean(_with_precision(w.model, T), Vector{T}(w.σ))
function signal!(out, w::RicianMean, p)
    signal!(out, w.model, p)
    @inbounds for j in eachindex(out, w.σ)
        out[j] = ricianmean(abs(out[j]), w.σ[j])
    end
    return out
end

# -------------------------------------------------------------------------------------------------
# σ per image (one average) and the averages, from `snr` or `σ` (as `simulate` takes them).
function _rician_σ(m, θ, snr, σ, averages)
    σ1 = sqrt.(diag(noisecovariance(m, θ; snr, σ)))           # one image's σ, per observation
    nex = averages === nothing ? ones(Int, length(σ1)) :
          averages isa Real ? fill(Int(averages), length(σ1)) : Int.(collect(averages))
    length(nex) == length(σ1) || throw(DimensionMismatch("$(length(nex)) averages for " *
                                                          "$(length(σ1)) observations"))
    return σ1, nex
end

"""
    riciannoise(m, θ; snr or σ, averages = nothing, averaging = :complex) -> (; simulate_with, Σ)

[`crlb`](@ref)'s keywords for a least-squares fit of magnitude data with Rician noise, from model
`m` at truth `θ`: the data's mean, [`RicianMean`](@ref) (`simulate_with`: the noise floor's
bias), and their covariance, [`ricianvar`](@ref) of each observation (`Σ`: the sandwich's SD).
With `averages`, `averaging = :complex` (the scanner's: σ/√NEX, then Rician) or `:magnitude` (a
mean of NEX magnitudes: the floor stays, the variance is divided by NEX).

```julia
m = SMT(b = [0, 1000, 2500, 4000])
θ = nominal(m, :SMT_cBH)
crlb(m, θ; riciannoise(m, θ; snr = 20, averages = dirs, averaging = :magnitude)..., weights = dirs)
```

The bias is the floor's, exactly (the fit of the mean data); the SD first order in the noise
around it. To fit data from another model, give that model and its truth here, and the fitted
model to crlb (`crlb(SMT(b = ...), θ; riciannoise(SMTT2(...), θ2; ...)...)`).
"""
function riciannoise(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing,
                     averages = nothing, averaging::Symbol = :complex)
    σ1, nex = _rician_σ(m, θ, snr, σ, averages)
    A = abs.(signal(m, θ))
    if averaging === :complex
        σc = σ1 ./ sqrt.(nex)
        v = ricianvar.(A, σc)
        w = RicianMean(m, σc)
    else
        averaging === :magnitude || throw(ArgumentError("riciannoise: `averaging` is :complex " *
                                                        "or :magnitude, got :$averaging"))
        v = ricianvar.(A, σ1) ./ nex
        w = RicianMean(m, σ1)
    end
    return (simulate_with = w => θ, Σ = Diagonal(v))
end

"""
    ricianbound(m, θ; snr or σ, averages = nothing, averaging = :complex, changes...) -> CRLBResult

The Cramér–Rao bound of Rician magnitude data: the SD no unbiased estimator can beat, with the
Fisher information of each observation R(A/σ)/σ² ([`ricianinformation`](@ref)) in place of the
Gaussian 1/σ², as in Devan et al., Magn Reson Imaging 2020;74:56, Appendix. A maximum-likelihood
fit of the magnitudes reaches it at high enough SNR; least squares does not (its bias and SD:
[`riciannoise`](@ref)). With `averages`, averaged as complex data (one Rician value with σ/√NEX)
or as magnitudes (NEX Rician values with σ: less information at low SNR). The fit's other
settings as keywords, as for [`crlb`](@ref).

Computed as crlb's Gaussian bound with each observation's effective σ, σⱼ/√(NEXⱼ·Rⱼ) (magnitude
averaging) or σⱼ/√(NEXⱼ·Rⱼ') (complex, R at the averaged SNR), and the matching weights, which
make the fit's covariance the inverse of the Fisher information.
"""
function ricianbound(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, averages = nothing,
                     averaging::Symbol = :complex, changes...)
    averaging in (:complex, :magnitude) || throw(ArgumentError("ricianbound: `averaging` is " *
                                                               ":complex or :magnitude"))
    σ1, nex = _rician_σ(m, θ, snr, σ, averages)
    A = abs.(signal(m, θ))
    info = averaging === :complex ? ricianinformation.(A, σ1 ./ sqrt.(nex)) :
                                    nex .* ricianinformation.(A, σ1)
    σeff = 1 ./ sqrt.(info)
    return crlb(m, θ; σ = σeff, weights = info, changes...)
end
