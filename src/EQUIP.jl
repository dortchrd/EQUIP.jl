"""
    EQUIP

**E**stimation of **QU**antitative **I**maging **P**arameters: MRI signal models and fast
voxel-wise fitting, for quantitative maps (T1, T2, qMT, ...).

A model is three pieces (see [`MRIModel`](@ref)):

1. a struct: everything that is the same for every voxel, i.e. the acquisition (`TI`, `TD`,
   ...) and options (`magnitude`), e.g. `QMTSIR(TI = [...], TD = 2.5)`,
2. [`parameters`](@ref)`(m)`: every parameter with its default role, [`Free`](@ref) or [`Fixed`](@ref),
3. [`signal!`](@ref)`(out, m, p)`: the signal equation.

Main entry points:

- [`signal`](@ref)`(m, p)`: simulate one voxel's signal.
- [`fit`](@ref)`(m, y)`: fit one voxel (`y` a vector), or every voxel of a dataset (`y` a matrix
  or N-d array, observations last; threaded).
- [`init`](@ref)`(m; ...)` + [`fit!`](@ref)`(cache, y)`: a reusable cache, for your own loops.
"""
module EQUIP

# --- Dependencies -------------------------------------------------------------------------------
# `using X` makes X's exported names available here. `import X: f` lets us *add methods* to a
# function that another package owns. We do that for a few standard function names, so that our
# `fit` IS StatsBase's `fit` (and our `init` is SciML's `init`): loading those packages together
# then causes no name clash.
using LinearAlgebra                        # dot, Diagonal, diag
using StaticArrays                         # SVector / SMatrix: small fixed-size arrays on the stack
using ForwardDiff: Dual, value, partials   # dual numbers for automatic derivatives
using Statistics: mean, median, quantile, std   # default starts; result summaries; compare
using Random: AbstractRNG, randn            # simulate: noise
import Random                               # Random.default_rng()
import CommonSolve                         # owner of `init` (the SciML convention)
using CommonSolve: init
import StatsAPI: fit, fit!, nobs, fitted, residuals   # owner of these names (StatsBase, GLM, ...)

# --- Public names -------------------------------------------------------------------------------
# `export` lists the names a user gets with `using EQUIP`. Everything else is still
# reachable as `EQUIP.name` (e.g. `EQUIP.setup`), just not brought into scope.
export MRIModel, Free, Fixed, parameters, signal!, signal, nobs             # model interface
export startgrid, datastart, relabel                                        # optional hooks
export fit, init, fit!, FitResult, FitStatus, isconverged                   # fitting
export fitted, residuals                                                    # fitted curves
export T1IR, R1IR, T1IRTD, R1IRTD, T1SR, R1SR, T1MFA, R1MFA                # relaxometry models
export T2Decay, R2Decay, T2DecayOffset, R2DecayOffset, T2BiexpDecay, R2BiexpDecay
export ADCDecay, ADCKurtosis                                               # diffusion models
export QMTSIR, SIR_5POINT, SIR_4POINT                                      # qMT models
export simulate, TISSUES, tissue, truth, phantom, labelshapes, compare, Comparison  # simulation

# --- Source files, in layers ----------------------------------------------------------------
# Each layer only uses the layers above it: models ← fit (← analysis ← design, later).

# models: the interface every model implements, and the built-in models
include("models/interface.jl")         # MRIModel, Free, Fixed, parameters, signal!, signal, nobs
include("models/builtin.jl")           # the built-in models, by area (relaxometry, diffusion, qmt)

# fit: from a model and data to fitted parameters
include("fit/levmar.jl")               # the solver for one voxel
include("fit/setup.jl")                # parameter roles + fixed/free/start -> FitSpec
include("fit/result.jl")               # FitResult: what fit and fit! return
include("fit/cache.jl")                # init -> FitCache; fit! -> one voxel
include("fit/fit.jl")                  # fit: one voxel, or every voxel of a dataset (threaded)

# sim: simulated data and accuracy against the truth (uses models and fit)
include("sim/simulate.jl")             # simulate: noisy data (Rician or Gaussian), snr or σ
include("sim/tissues.jl")              # TISSUES (published values), tissue, truth
include("sim/phantom.jl")              # phantom: parameter maps from a label image; labelshapes
include("sim/compare.jl")              # compare: bias, SD, RMSE, % converged

end # module
