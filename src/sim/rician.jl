# =================================================================================================
# The Cramér–Rao bound of Rician data
#
#   ricianbound(m, θ; snr, ...) the Cramér–Rao bound of Rician data (Devan et al. 2020, Appendix)
#
# (A least-squares fit of magnitude data is crlb's `noise = :rician`, src/errors/crlb.jl; it was
# `crlb(m, θ; riciannoise(m, θ; snr)...)` until 2026-11-12.)
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
    ricianbound(m, θ; snr or σ, averages = nothing, averaging = :complex, changes...) -> CRLBResult

The Cramér–Rao bound of Rician magnitude data: the SD no unbiased estimator can beat, with the
Fisher information of each observation R(A/σ)/σ² ([`ricianinformation`](@ref)) in place of the
Gaussian 1/σ², as in Devan et al., Magn Reson Imaging 2020;74:56, Appendix. A maximum-likelihood
fit of the magnitudes reaches it at high enough SNR (`crlb(m, θ; snr, likelihood = :rician)`
predicts that fit: the same numbers when nothing fixed is wrong); least squares does not (its
bias and SD: `crlb(m, θ; snr, noise = :rician)`). With `averages`, averaged as complex data (one
Rician value with σ/√NEX) or as magnitudes (NEX Rician values with σ: less information at low SNR). The fit's other
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
