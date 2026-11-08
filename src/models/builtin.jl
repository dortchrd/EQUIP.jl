# =================================================================================================
# Built-in models
#
# Units: all times in SECONDS (TE, TI, TD, TR) and rates in 1/s. The default starting values
# below assume this (e.g. T1 starts at 1 s). Flip angles are in degrees; b-values in s/mm², so
# the ADC is in mm²/s.
#
# Default starting values are mostly fixed numbers. The parameter that scales the whole signal
# (S0, M0, M0f) is declared `Free(start = :scale)` and starts, voxel by voxel, at the
# least-squares scale given the other starting values, so raw data (values in the hundreds or thousands) need no
# normalization. A few start from the data: T2Decay's T2, DTI and DKI (`:loglinear`, a log-linear fit)
# and the inversion-recovery T1 (`BestOf` 30 values). Override any start in the call, e.g.
# `T1 = Free(start = 0.8)` (a number or a per-voxel map).
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

# Upper bound for T2 (and T2*) in T2Decay: 2.5 s, above the longest in tissue (CSF, about 2 s;
# the owner, 2026-11-08). Data that do not decay are best fitted by T2 → ∞ (R2 = 0); without a
# bound the time form reports convergence at an enormous T2 (about 1e10 s). With it, such voxels
# end on the bound, where `onbound` flags them, as R2Decay's R2 = 0 is flagged. Widen it in a
# call: `fit(m, data; T2 = Free(upper = 10))`.
const _T2MAX = 2.5

# Time-constant and rate versions (T1IR and R1IR, T2Decay and R2Decay, ...): the equation is
# written once, in the rate version. The time version's `signal!` converts (R1 = 1/T1) and calls
# the rate version's:
#
#     signal!(out, m::T1IR, p) = signal!(out, R1IR(m.TI, m.magnitude), (; p.M0, p.Sf, R1 = 1 / p.T1))
#
# - `R1IR(m.TI, m.magnitude)` makes a rate model around the same TI vector (no copy, no cost).
# - `(; p.M0, p.Sf, R1 = ...)` is short for `(M0 = p.M0, Sf = p.Sf, R1 = ...)`.
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
function _t1grid(t)
    tmin = minimum(x -> x > 0 ? float(x) : Inf, t)
    tmax = float(maximum(t))
    isfinite(tmin) || (tmin = tmax / 100)                  # all times 0: a nominal range
    return _loggrid(tmin / 5, 5 * tmax, 30)
end

# Grouped by area, one file per model (or per family of closely related models). Names put the
# quantity first, then the method: T1IR, T1MFA, T2Decay, QMTSIR, ...
include("relaxometry/t1_ir.jl")         # T1IR, R1IR, T1IRTD, R1IRTD: inversion recovery
include("relaxometry/t1_sr.jl")         # T1SR, R1SR: saturation recovery
include("relaxometry/t1_mfa.jl")        # T1MFA, R1MFA: multiple flip angles (spoiled gradient echo)
include("relaxometry/t2_decay.jl")      # T2Decay, R2Decay and their offset versions
include("relaxometry/t2_biexp_decay.jl")  # T2BiexpDecay, R2BiexpDecay: two T2 components
include("diffusion/compartments.jl")    # the compartments: iso, stick, zeppelin and their view
                                        # powder (exported), for the diffusion models below
include("diffusion/adc_decay.jl")       # ADCDecay
include("diffusion/adc_kurtosis.jl")    # ADCKurtosis: diffusion with kurtosis, one direction
include("diffusion/dti.jl")             # DTI: the diffusion tensor, one image per gradient direction
include("diffusion/dki.jl")             # DKI: diffusion kurtosis (V = MD²·W), MK, AK, RK as DIPY's
include("diffusion/smt.jl")             # SMT, SMTFreeWater, SMTT2: the spherical mean technique; sphericalmean
include("diffusion/directions.jl")      # fiberdirection, directionset: a fiber's direction; gradient direction sets
include("qmt/qmt_sir.jl")               # QMTSIR: selective inversion recovery
