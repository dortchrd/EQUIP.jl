# =================================================================================================
# Models with several parameters, one of them fitted
#
# Four ways to fit a single parameter while the others are known, each on simulated data, and
# what goes wrong when a fixed value is not right:
#   1. @mrimodel, one line:  normalized T2 decay, only T2 free (S0 = 1, no offset)
#   2. @mrimodel, several lines:  inversion recovery with M0 and the inversion efficiency known,
#      only T1 free; another known M0 given in the call
#   3. a fixed parameter set to a map, voxel by voxel:  ADC with S0 from a measured b = 0 image
#   4. any model, built-in too, with all but one parameter fixed in the call:  QMTSIR, PSR only
#      (R1f and M0f from separate measurements); its predicted precision (crlb), and how each
#      fixed value's error moves PSR (leakage)
#
# A parameter whose default is a plain number is fixed (`S0 = 1.0`); `Free(...)` is fitted. Any
# role changes in the call, as for any model: `fit(m, y; M0 = 950.0)` fixes M0 at another value,
# `fit(m, y; M0 = Free(start = :scale))` frees it.
#
# Run from the repository folder:
#   julia --project=. examples/one_free_parameter.jl
# =================================================================================================

using EQUIP, Random

rng = MersenneTwister(3)                  # the same seed gives the same data every time

# --- 1. One line: normalized T2 decay, only T2 fitted ---------------------------------------------
# Three parameters; S0 and C have plain numbers as defaults, so they are fixed.
@mrimodel NormT2(TE; T2 = Free(start = 0.05, lower = 1e-6), S0 = 1.0, C = 0.0) =
    S0 * exp(-TE / T2) + C

m1 = NormT2(TE = 0.01:0.01:0.32)
y1 = signal(m1, (T2 = 0.08, S0 = 1.0, C = 0.0)) .+ 0.002 .* randn(rng, nobs(m1))
r1 = fit(m1, y1)
println("1. normalized T2 decay:  T2 = ", round(r1.params.T2; digits = 4), " s (truth 0.08), ",
        "± ", round(stderror(r1).T2; digits = 4), " s")

# --- 2. Several lines: inversion recovery, M0 and the inversion efficiency known ------------------
# `function ... end`: the body is ordinary Julia for one observation; the last value (or
# `return`) is the signal. `magnitude = true` is an option (a positional argument with a default).
@mrimodel function IRT1(TI, magnitude = true; T1 = Free(start = 1.0, lower = 1e-6),
                        M0 = 1000.0, Sf = -1.0)
    s = M0 * (1 + (Sf - 1) * exp(-TI / T1))     # Sf = -1: a perfect inversion
    return magnitude ? abs(s) : s               # magnitude images lose the sign
end

m2 = IRT1(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
y2 = signal(m2, (T1 = 0.9, M0 = 1000.0, Sf = -1.0)) .+ 5 .* randn(rng, nobs(m2))
println("2. inversion recovery:   T1 = ", round(fit(m2, y2).params.T1; digits = 4), " s (truth 0.9); ",
        "with M0 = 950 known instead: ", round(fit(m2, y2; M0 = 950.0).params.T1; digits = 4), " s")

# --- 3. A fixed parameter set to a map: ADC with S0 from a measured b = 0 image -------------------
# S0 is fixed (default 1.0), and the call gives it a value per voxel: the b = 0 image.
@mrimodel ADCfromb0(b; ADC = Free(start = 1e-3, lower = 0), S0 = 1.0) = S0 * exp(-b * ADC)

m3 = ADCfromb0(b = [250.0, 500, 1000, 1500, 2000])          # s/mm²
b0map = 1000 .* (1 .+ 0.1 .* rand(rng, 4, 4))               # the measured b = 0 image
adcmap = fill(8e-4, 4, 4)                                   # mm²/s
# The noise as σ: `snr` is relative to the model's `:scale` parameter, and this one has none.
Y3 = simulate(m3, (ADC = adcmap, S0 = b0map); σ = 10.0, noise = :rician, rng)
r3 = fit(m3, Y3; S0 = b0map)
lo, hi = extrema(r3.params.ADC)
println("3. ADC with S0 = b0 map: ADC from ", round(lo; sigdigits = 3), " to ", round(hi; sigdigits = 3),
        " mm²/s (truth 8.0e-4) in ", length(r3.params.ADC), " voxels; fitted: ", keys(r3.params))

# --- 4. Any model, all but one fixed in the call: QMTSIR, PSR only --------------------------------
# No new model: the built-in's roles change in the call. R1f and M0f are fixed to maps (as if
# measured separately), Sf and kmf to the tissue's values. White matter I (Dortch et al., Magn
# Reson Med 2018, Table 1): PSR 0.20, kmf 5 s⁻¹, R1f 1.5 s⁻¹, Sf -1.
mq = QMTSIR(; SIR_5POINT...)
θ = nominal(mq, :SIR_WM_I)
truth = map(x -> x isa Real ? fill(Float64(x), 3, 3) : x, θ)   # a 3×3 slice of this tissue
Yq = simulate(mq, truth; snr = 200, noise = :rician, rng)
r1fmap, m0fmap = fill(1.5, 3, 3), fill(1.0, 3, 3)
psr = fit(mq, Yq; R1f = r1fmap, M0f = m0fmap, Sf = -1.0, kmf = 5.0)
lo, hi = extrema(psr.params.PSR)
println("4. QMTSIR, PSR only:     PSR from ", round(lo; digits = 3), " to ", round(hi; digits = 3),
        " (truth 0.2); fitted: ", keys(psr.params))

# Its predicted precision against all four of QMTSIR's free parameters fitted: fixing the others
# buys precision...
sd1 = crlb(mq, θ; snr = 200, R1f = 1.5, M0f = 1.0, Sf = -1.0).sd.PSR
sd4 = crlb(mq, θ; snr = 200).sd.PSR
println("   predicted SD of PSR (crlb, SNR 200): ", round(sd1; digits = 4), " with PSR alone, ",
        round(sd4; digits = 4), " with PSR, R1f, Sf and M0f fitted")

# ... if the fixed values are right. With Sf at -0.95 instead of -1, and kmf left at QMTSIR's
# default (12.5 s⁻¹, not the tissue's 5), PSR goes wrong:
wrong = fit(mq, Yq; R1f = r1fmap, M0f = m0fmap, Sf = -0.95)
lo, hi = extrema(wrong.params.PSR)
println("   with Sf = -0.95 and kmf = 12.5: PSR from ", round(lo; digits = 3), " to ", round(hi; digits = 3))

# `leakage` says which fixed values matter: the % change of PSR per 1% error in each fixed value
# (its elasticity), from derivatives, without noise. Here 1% in Sf moves PSR by about 7%.
println()
display(leakage(mq, θ; R1f = 1.5, M0f = 1.0, Sf = -1.0))
println()
