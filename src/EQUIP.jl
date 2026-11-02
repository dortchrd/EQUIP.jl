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
- [`refit`](@ref)`(res, data; changes...)`: fit again from an earlier result, with changes.
"""
module EQUIP

# --- Dependencies -------------------------------------------------------------------------------
# `using X` makes X's exported names available here. `import X: f` lets us *add methods* to a
# function that another package owns. We do that for a few standard function names, so that our
# `fit` IS StatsBase's `fit` (and our `init` is SciML's `init`): loading those packages together
# then causes no name clash.
using LinearAlgebra                        # dot, Diagonal, diag
import LinearAlgebra: cond                 # cond(c): a crlb result's condition number
using StaticArrays                         # SVector / SMatrix: small fixed-size arrays on the stack
using ForwardDiff: Dual, value, partials   # dual numbers for automatic derivatives
import ForwardDiff                         # crlb: ForwardDiff.jacobian, ForwardDiff.hessian
using Statistics: mean, median, quantile, std, cov   # starts; summaries; compare; montecarlo
import Statistics: cor                     # cor(c), cor(mc): correlations of the estimates
using Random: AbstractRNG, randn, Xoshiro   # simulate: noise; montecarlo: one stream per seed
import Random                               # Random.default_rng()
import CommonSolve                         # owner of `init` (the SciML convention)
using CommonSolve: init
import StatsAPI: fit, fit!, nobs, fitted, residuals   # owner of these names (StatsBase, GLM, ...)
import StatsAPI: informationmatrix, vcov, stderror    # crlb and montecarlo results; fits
import StatsAPI: dof, dof_residual, rss, loglikelihood, aic, aicc, bic   # fit statistics
import MRImages                            # data files: NIfTI with BIDS sidecars, MATLAB (MAT)
using MRImages: MRImage, readimage, writeimage, readvalues, samegeometry, filestem, sidecarpath,
                writesidecar, writemat

# --- Public names -------------------------------------------------------------------------------
# `export` lists the names a user gets with `using EQUIP`. Everything else is still
# reachable as `EQUIP.name` (e.g. `EQUIP.setup`), just not brought into scope.
export MRIModel, Free, Fixed, BestOf, parameters, defaults                  # model interface
export resolve, signal!, signal, nobs                                       # values and signals
export datastart, relabel                                                   # optional hooks
export fit, init, fit!, refit, LM, FitResult, FitStatus, isconverged       # fitting
export fitted, residuals                                                    # fitted curves
export dof, dof_residual, rss, residualsd, loglikelihood, aic, aicc, bic   # fit statistics
export onbound
export T1IR, R1IR, T1IRTD, R1IRTD, T1SR, R1SR, T1MFA, R1MFA                # relaxometry models
export T2Decay, R2Decay, T2DecayOffset, R2DecayOffset, T2BiexpDecay, R2BiexpDecay
export ADCDecay, ADCKurtosis                                               # diffusion models
export QMTSIR, SIR_5POINT, SIR_4POINT, SIR_16POINT                         # qMT models
export simulate, TISSUES, tissue, nominal, compare, Comparison              # simulation
export Scenario, scenarios, crlb, crlb!, CRLBCache, CRLBResult, montecarlo, MonteCarloResult, Sweep
export noisecovariance, noisefree, bias, jacobian, hessian, biasgradient    # crlb's pieces
export sandwich, biasedcrlb
export cond, cor, leakage, LeakageResult                                    # diagnostics
export ProtocolDesign, optimize_protocol, DesignResult                       # protocol design
export informationmatrix, vcov, stderror                                    # StatsAPI's names
export loaddata, loadmap, DataInfo, savemaps, writetable                     # data in, maps out
export MRImage, readimage, writeimage, readvalues                           # (from MRImages)

# --- Source files, in layers ----------------------------------------------------------------
# Each layer only uses the layers above it: models ← fit, sim ← errors ← design (fit and sim
# each use models alone); io (data in, maps out) uses fit.

# models: the interface every model implements, and the built-in models
include("models/interface.jl")         # MRIModel, Free, Fixed, parameters, signal!, signal, nobs
include("models/builtin.jl")           # the built-in models, by area (relaxometry, diffusion, qmt)

# fit: from a model and data to fitted parameters
include("fit/levmar.jl")               # the solver for one voxel
include("fit/setup.jl")                # parameter roles + the keywords of a fit -> FitSpec
include("fit/result.jl")               # FitResult: what fit and fit! return
include("fit/starts.jl")               # each voxel's starting point: :data, BestOf, :scale
include("fit/cache.jl")                # init -> FitCache: everything a voxel fit needs, made once
include("fit/voxel.jl")                # one voxel: fit!(cache, y), fit(m, y::AbstractVector)
include("fit/dataset.jl")              # every voxel of a dataset, on threads: fit(m, data); refit
include("fit/statistics.jl")           # the fit's statistics: stderror, vcov, AIC/BIC, onbound

# sim: from true values to data (uses models only)
include("sim/tissues.jl")              # TISSUES (published values), tissue, nominal
include("sim/noise.jl")                # the noise model: snr, σ or Σ, checked; noisecovariance
include("sim/simulate.jl")             # simulate: noisy data (Rician or Gaussian)

# errors: a fit's errors against the truth, bias and SD, predicted from derivatives or measured
# (uses models, fit and sim)
include("errors/scenario.jl")          # Scenario: a truth and the fit judged against it
include("errors/derivatives.jl")       # crlb's pieces: CRLBCache, the sums, the formulas
include("errors/crlb.jl")              # crlb: bias, SD and RMSE without noisy fits; the kernel
include("errors/montecarlo.jl")        # montecarlo: the same measured, with crlb alongside
include("errors/compare.jl")           # compare: the same measured on a dataset (maps)
include("errors/diagnostics.jl")       # cond, cor, leakage: where the errors come from
include("errors/sweep.jl")             # Sweep: crlb or montecarlo over noise levels and truths

# design: protocols chosen by their predicted errors (uses errors)
include("design/design.jl")            # ProtocolDesign: the objective (Eq. 5), made once
include("design/search.jl")            # optimize_protocol: differential evolution, Nelder–Mead

# io: data files in, maps out (MRImages reads and writes the files; here, what fitting needs)
include("io/data.jl")                  # loaddata, loadmap, DataInfo: a dataset's files as one array
include("io/maps.jl")                  # savemaps: the maps in the data's geometry, with sidecars
include("io/tables.jl")                # writetable: crlb's and montecarlo's rows as TSV

# Compile the common cases while the package is installed (see the file).
include("precompile.jl")

end # module
