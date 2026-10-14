"""
    VoxelFit

Fast, allocation-free nonlinear least-squares fitting of MRI signal models, voxel by voxel,
for building parametric maps (T1, T2, T2*, ADC, IVIM, ...).

Main entry points:

- [`fit_maps`](@ref) / [`fit_maps!`](@ref): fit every voxel of an N-d dataset (threaded CPU,
  or a GPU backend through KernelAbstractions).
- [`fit_voxel`](@ref): fit one signal curve.
- Built-in models: [`MonoExp`](@ref), [`MonoExpOffset`](@ref), [`BiExp`](@ref), [`ADC`](@ref),
  [`Kurtosis`](@ref), [`IVIM`](@ref), [`InversionRecovery`](@ref),
  [`SaturationRecovery`](@ref), [`SPGR`](@ref).
- Custom models: subtype [`AbstractModel`](@ref), wrap a function with [`PointModel`](@ref),
  or build one from a string with [`expr_model`](@ref).
"""
module VoxelFit

using LinearAlgebra
using StaticArrays
using ForwardDiff
using ForwardDiff: Dual, value, partials
using Mmap
using PrecompileTools

export AbstractModel, AbstractCurveModel, PointModel, expr_model
export nparams, paramnames, initial_guess, lower_bounds, upper_bounds, curve!
export MonoExp, MonoExpOffset, BiExp, ADC, Kurtosis, IVIM, InversionRecovery,
       SaturationRecovery, SPGR
export FitStatus, LMOptions, LMResult, MapResult, fit_voxel, fit_maps, fit_maps!
export isconverged, parse_model

include("models.jl")
include("lm.jl")
include("builtin_models.jl")
include("driver.jl")
include("npy.jl")
include("cli.jl")
include("precompile.jl")

end # module
