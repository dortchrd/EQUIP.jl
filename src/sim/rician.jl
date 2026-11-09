# =================================================================================================
# Rician noise in crlb: the keywords for a least-squares fit of magnitude data, and the bound of
# Rician data
#
#   riciannoise(m, θ; snr, ...) crlb's keywords for a least-squares fit of magnitude data: the
#                               floor's bias (`simulate_with`) and the Rician variance (`Σ`)
#   ricianbound(m, θ; snr, ...) the Cramér–Rao bound of Rician data (Devan et al. 2020, Appendix)
#
# Built on the Rician mean, variance and information and on `RicianMean`, in
# src/models/rician.jl (moved there on 2026-11-09 so that the fit can use them).
# =================================================================================================

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
