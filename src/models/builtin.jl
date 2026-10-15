# =================================================================================================
# Built-in models
#
# Units: all times in SECONDS (TE, TI, TD, TR) and rates in 1/s. The default starting values
# below assume this (e.g. T1 starts at 1 s). Flip angles are in degrees.
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
#                       `@kwdef` gives the keyword constructor, e.g. `SIRFSE(TI = ..., TD = ...)`.
#                       Each vector field gets its own type parameter (A, B, ...), so a field can
#                       be any kind of vector (Vector, range, ...) and Julia still knows its
#                       exact type. `init` converts them to plain float vectors for fitting.
#   2. fill method      (optional) lets one number stand for every observation, e.g. one TR.
#   3. parameters(m)    every parameter with its default role.
#   4. signal!(out, m, p)
#                       unpack the model and the parameters by name, then loop over observations.
# =================================================================================================

const _TMIN = 1e-6    # lower bound for time constants, so that t/T stays finite

# One file per model:
include("exp_decay.jl")
include("inversion_recovery.jl")
include("saturation_recovery.jl")
include("mfa.jl")
include("sirfse.jl")
