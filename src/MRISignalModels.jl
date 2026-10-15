"""
    MRISignalModels

MRI signal models and fast voxel-wise fitting, for quantitative maps (T1, T2, qMT, ...).

A model is three pieces (see [`MRIModel`](@ref)):

1. a struct: everything that is the same for every voxel, i.e. the acquisition (`TI`, `TD`,
   ...) and options (`magnitude`), e.g. `SIRFSE(TI = [...], TD = 2.5)`,
2. [`parameters`](@ref)`(m)`: every parameter with its default role, [`Free`](@ref) or [`Fixed`](@ref),
3. [`signal!`](@ref)`(out, m, p)`: the signal equation.

Main entry points:

- [`signal`](@ref)`(m, p)`: simulate one voxel's signal.
- [`fit`](@ref)`(m, y)`: fit one voxel (`y` a vector), or every voxel of a dataset (`y` a matrix
  or N-d array, observations last; threaded).
- [`init`](@ref)`(m; ...)` + [`fit!`](@ref)`(cache, y)`: a reusable cache, for your own loops.
"""
module MRISignalModels

# --- Dependencies -------------------------------------------------------------------------------
# `using X` makes X's exported names available here. `import X: f` lets us *add methods* to a
# function that another package owns. We do that for a few standard function names, so that our
# `fit` IS StatsBase's `fit` (and our `init` is SciML's `init`): loading those packages together
# then causes no name clash.
using LinearAlgebra                        # dot, Diagonal, diag
using StaticArrays                         # SVector / SMatrix: small fixed-size arrays on the stack
using ForwardDiff: Dual, value, partials   # dual numbers for automatic derivatives
import CommonSolve                         # owner of `init` (the SciML convention)
using CommonSolve: init
import StatsAPI: fit, fit!, nobs           # owner of `fit`, `fit!`, `nobs` (StatsBase, GLM, ...)

# --- Public names -------------------------------------------------------------------------------
# `export` lists the names a user gets with `using MRISignalModels`. Everything else is still
# reachable as `MRISignalModels.name` (e.g. `MRISignalModels.setup`), just not brought into scope.
export MRIModel, Free, Fixed, parameters, signal!, signal, nobs             # model interface
export fit, init, fit!, VoxelResult, FitStatus, isconverged                 # fitting
export ExpDecay, InversionRecovery, SaturationRecovery, MFA, SIRFSE         # built-in models

# --- Source files, in layers ----------------------------------------------------------------
# Each layer only uses the layers above it: models ← fit (← analysis ← design, later).

# models: the interface every model implements, and the built-in models
include("models/interface.jl")   # MRIModel, Free, Fixed, parameters, signal!, signal, nobs
include("models/builtin.jl")     # ExpDecay, InversionRecovery, SaturationRecovery, MFA, SIRFSE

# fit: from a model and data to fitted parameters
include("fit/lm.jl")             # the Levenberg–Marquardt solver for one voxel
include("fit/setup.jl")          # parameter roles + fixed/free/start -> FitSpec
include("fit/cache.jl")          # init -> FitCache; fit! -> one voxel
include("fit/fit.jl")            # fit: one voxel, or every voxel of a dataset (threaded)

end # module
