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

# Grouped by area, one file per model (or per family of closely related models). Names put the
# quantity first, then the method: T1IR, T1MFA, T2Decay, QMTSIR, ...
include("relaxometry/t1_ir.jl")         # T1IR: inversion recovery
include("relaxometry/t1_sr.jl")         # T1SR: saturation recovery
include("relaxometry/t1_mfa.jl")        # T1MFA: multiple flip angles (spoiled gradient echo)
include("relaxometry/t2_decay.jl")      # T2Decay, R2Decay and their offset versions
include("diffusion/adc_decay.jl")       # ADCDecay
include("qmt/qmt_sir.jl")               # QMTSIR: selective inversion recovery
