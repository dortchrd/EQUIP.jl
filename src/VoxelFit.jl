"""
    VoxelFit

Fast, allocation-free nonlinear least-squares fitting of MRI signal models, voxel by voxel,
for building parametric maps (T1, T2, T2*, ADC, IVIM, qMT, ...).

A model is a struct (settings), [`parameters`](@ref) (every parameter with its default role,
[`Estimate`](@ref) or [`Fixed`](@ref)) and [`signal!`](@ref) (the signal equation). See
`docs/design.md`.

Main entry points:

- [`fit_maps`](@ref) / [`fit_maps!`](@ref): fit every voxel of an N-d dataset (threaded).
- [`init`](@ref) + [`fit!`](@ref): a reusable cache for fitting single voxels.
- [`signal`](@ref): simulate a model's signal.
"""
module VoxelFit

using LinearAlgebra
using StaticArrays
using ForwardDiff
using ForwardDiff: Dual, value, partials
using Mmap
using PrecompileTools
import CommonSolve
using CommonSolve: init

export AbstractModel, Estimate, Fixed, parameters, protocol, signal!, signal, obs, initial_guess
export FunctionModel, expr_model
export MonoExp, MonoExpOffset, BiExp, ADC, Kurtosis, IVIM, InversionRecovery,
       SaturationRecovery, SPGR, SIRFSE
export FitStatus, LMOptions, VoxelResult, MapResult, init, fit!, fit_maps, fit_maps!
export standard_errors, isconverged, parse_model

include("models.jl")
include("lm.jl")
include("cache.jl")
include("builtin_models.jl")
include("driver.jl")
include("npy.jl")
include("cli.jl")
include("precompile.jl")

end # module
