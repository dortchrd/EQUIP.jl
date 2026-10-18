# =================================================================================================
# Built-in models
#
# Units: all times in SECONDS (TE, TI, TD, TR) and rates in 1/s. The default starting values
# below assume this (e.g. T1 starts at 1 s). Flip angles are in degrees; b-values in s/mm², so
# the ADC is in mm²/s.
#
# Default starting values are fixed numbers (no model-specific, data-driven guesses yet), except
# for the parameter that scales the whole signal (S0, M0, M0f): it is declared `Free(:scale)` and
# starts, voxel by voxel, at the least-squares scale given the other starting values. So raw
# data (values in the hundreds or thousands) need no normalization. Override any start with the
# `start` keyword (numbers or per-voxel maps).
#
# Each model follows the same pattern:
#
#   1. @kwdef struct    fields = the acquisition (one value per observation) and options.
#                       `@kwdef` gives the keyword constructor, e.g. `QMTSIR(TI = ..., TD = ...)`.
#                       Each vector field gets its own type parameter (A, B, ...), so a field can
#                       be any kind of vector (Vector, range, ...) and Julia still knows its
#                       exact type. `init` converts them to plain float vectors for fitting.
#   2. fill method      (optional) lets one number stand for every observation, e.g. one TR.
#   3. parameters(m)    every parameter with its default role.
#   4. signal!(out, m, p)
#                       unpack the model and the parameters by name, then loop over observations
#                       with `@inbounds for j in eachindex(out, <every per-observation field>)`:
#                       eachindex checks once that the lengths match, so @inbounds is safe.
# =================================================================================================

const _TMIN = 1e-6    # lower bound for time constants, so that t/T stays finite

# Time-constant and rate versions (T1IR and R1IR, T2Decay and R2Decay, ...): the equation is
# written once, in the rate version. The time version's `signal!` converts (R1 = 1/T1) and calls
# the rate version's:
#
#     signal!(out, m::T1IR, p) = signal!(out, R1IR(m.TI, m.magnitude), (; p.M0, p.B, R1 = 1 / p.T1))
#
# - `R1IR(m.TI, m.magnitude)` makes a rate model around the same TI vector (no copy, no cost).
# - `(; p.M0, p.B, R1 = ...)` is short for `(M0 = p.M0, B = p.B, R1 = ...)`.
# - ForwardDiff carries the derivatives through 1/T1, so the fit is exactly what a loop written
#   with T1 would give (measured: identical results, bit for bit, within 2% of the speed).
# - Always time → rate, never rate → time: R1 = 0 (a flat curve) is a valid value but 1/0 is
#   not, while T1 never reaches 0 (lower bound `_TMIN`), so 1/T1 is always safe.
# The two versions keep their own structs, `parameters` (names, bounds, starts) and docstrings:
# those are what users see.

# n values from a to b, evenly spaced on a log scale (for grids of starting values).
_loggrid(a, b, n) = exp.(range(log(a), log(b); length = n))

# A grid of T1 values for starting values, spanning the inversion (or delay) times t: from a
# fifth of the shortest positive time to five times the longest, 30 values.
# The rate R of a mono-exponential decay, y = S0·exp(-R·t), from a straight-line fit to log(y)
# against t (for starting values, see `datastart`). Weighted by y²: the noise in log(y) is about
# σ/y, so this weights each point by its precision and keeps the noisy tail from dominating.
# Points with y ≤ 0 are skipped. NaN when the data give no decay (flat or rising, too few points).
function _loglinear_rate(t, y)
    T = float(eltype(y))
    sw = swt = swtt = swl = swtl = zero(T)                 # weighted sums: w, w·t, w·t², w·l, w·t·l
    @inbounds for i in eachindex(t, y)
        y[i] > 0 || continue
        w, ti, li = T(y[i])^2, T(t[i]), log(T(y[i]))
        sw += w; swt += w * ti; swtt += w * ti^2; swl += w * li; swtl += w * ti * li
    end
    R = (swt * swl - sw * swtl) / (sw * swtt - swt^2)       # minus the slope of log(y) against t
    return isfinite(R) && R > 0 ? R : T(NaN)
end

function _t1grid(t)
    tmin = minimum(x -> x > 0 ? float(x) : Inf, t)
    tmax = float(maximum(t))
    isfinite(tmin) || (tmin = tmax / 100)                  # all times 0: a nominal range
    return _loggrid(tmin / 5, 5tmax, 30)
end

# Grouped by area, one file per model (or per family of closely related models). Names put the
# quantity first, then the method: T1IR, T1MFA, T2Decay, QMTSIR, ...
include("relaxometry/t1_ir.jl")         # T1IR, R1IR, T1IRTD, R1IRTD: inversion recovery
include("relaxometry/t1_sr.jl")         # T1SR, R1SR: saturation recovery
include("relaxometry/t1_mfa.jl")        # T1MFA, R1MFA: multiple flip angles (spoiled gradient echo)
include("relaxometry/t2_decay.jl")      # T2Decay, R2Decay and their offset versions
include("diffusion/adc_decay.jl")       # ADCDecay
include("qmt/qmt_sir.jl")               # QMTSIR: selective inversion recovery
