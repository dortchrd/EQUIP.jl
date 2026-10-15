"""
    VoxelFit

Fast voxel-wise nonlinear least-squares fitting of MRI signal models, for parametric maps.

A model is three pieces (see [`MRIModel`](@ref)):

1. a struct: everything that is the same for every voxel, i.e. the acquisition (`TI`, `TD`,
   ...) and options (`magnitude`), e.g. `SIRFSE(TI = [...], TD = 2.5)`,
2. [`parameters`](@ref)`(m)`: every parameter with its default role, [`Free`](@ref) or [`Fixed`](@ref),
3. [`signal!`](@ref)`(out, m, p)`: the signal equation.

Main entry points:

- [`signal`](@ref)`(m, p)`: simulate one voxel's signal.
- [`init`](@ref)`(m; ...)` + [`fit!`](@ref)`(cache, y)`: fit single voxels with a reusable cache.
- [`fit_maps`](@ref)`(m, data; ...)`: fit every voxel of a dataset (threaded).
"""
module VoxelFit

# --- Dependencies -------------------------------------------------------------------------------
# `using X` makes X's exported names available here. `import X` only makes the module name
# available, which we need when we *add methods* to a function owned by another package
# (here `CommonSolve.init`).
using LinearAlgebra                    # dot, Diagonal, diag
using StaticArrays                     # SVector / SMatrix: small fixed-size arrays on the stack
using ForwardDiff: Dual, value, partials   # dual numbers for automatic derivatives
import CommonSolve                     # owner of the generic function `init` (SciML convention)
using CommonSolve: init                # so that `init` can be re-exported below

# --- Public names -------------------------------------------------------------------------------
# `export` lists the names a user gets with `using VoxelFit`. Everything else is still reachable
# as `VoxelFit.name` (e.g. `VoxelFit.setup`), just not brought into scope automatically.
export MRIModel, Free, Fixed, parameters, signal!, signal, nobs             # model interface
export init, fit!, fit_maps, VoxelResult, FitStatus, isconverged            # fitting
export ExpDecay, InversionRecovery, SaturationRecovery, MFA, SIRFSE         # built-in models

# --- Source files, in dependency order ----------------------------------------------------------
include("interface.jl")   # MRIModel, Free, Fixed, parameters, signal!, signal, nobs
include("lm.jl")          # the Levenberg–Marquardt solver for one voxel
include("fit.jl")         # setup (roles -> FitSpec), init (-> FitCache), fit! (one voxel)
include("maps.jl")        # fit_maps: all voxels of a dataset, with threads
include("models.jl")      # the built-in models

end # module
