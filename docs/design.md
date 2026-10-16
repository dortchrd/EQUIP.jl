# MRISignalModels design (0.3)

The design decisions, recorded so the code and the discussion share one reference.

## Inputs: where each kind of information lives

| input | varies across voxels? | Julia form | example |
|---|---|---|---|
| **Model**: physics, acquisition and options | no | a `@kwdef` struct `<: MRIModel` | `SIRFSE(TI = [...], TD = 2.5, magnitude = true)` |
| **Parameters**: every quantity in the signal equation | each one can be free, a constant, a per-voxel map or a tie | values: a named tuple `p`; default roles: `parameters(m)` | PSR, R1f, kmf, B1, Sm |
| **Fit configuration**: which parameters are estimated in this fit | no | keywords `fixed`, `free`, `start` | `free = (:kmf,)` |
| **Data** | yes | N-d array, observations last | X×Y×Z×M |

Decisions behind this:

- **The acquisition lives in the model struct** (TI, TD, flip angles, ...), next to options
  such as `magnitude`.
  - The model alone then defines the forward model f(θ): `signal(m, p)`, `init(m)` and
    `fit(m, data)` need no extra argument.
  - The struct documents which acquisition a model needs, and a missing field fails at
    construction.
  - Protocol optimization just builds a new model per candidate, which costs nothing.
  - Front ends can build any model from named fields through the `@kwdef` keyword
    constructor.
- **The model never holds parameter values.** One model object is reused for millions of
  voxels, while `p` changes at every LM step.
- **A parameter keeps its place in the model whether it is fixed or estimated.** Its role
  (`Free(init; lower, upper)` / `Fixed(value)`) sits next to it, so the starting value, the
  bounds and the default value live in one place. A fit changes roles through keywords, so
  sensitivity analyses (e.g. fix kmf at different values) never edit the model.
- **Model structs use `@kwdef`**, with one type parameter per vector field and no inner
  constructors (no Parameters.jl):
  - The only invariant, equal lengths of the per-observation vectors, is checked generically
    by `nobs(m)` when `init` and `signal` run.
  - A field that may be one number for all observations (TD, TR) gets a one-line method that
    fills it to a vector. So `signal!` always indexes vectors (`TD[j]`), and any other code
    reading the protocol (scan time, plots) sees vectors too.
- **`signal!` unpacks by name**: `(; TI, TD, magnitude) = m` and `(; PSR, ...) = p`. Never by
  position, because the order of `p` depends on which parameters are free.
- **The supertype is `MRIModel`**, not `AbstractModel`. The generic name is used by other
  packages (e.g. AbstractMCMC.jl), and exporting the same name from two packages forces users
  to qualify it.
- **One function, `fit`, chosen by the shape of the data.** `fit(m, y::AbstractVector)` fits one
  voxel and returns plain numbers; `fit(m, data::AbstractArray)` fits every voxel (observations
  last, any number of spatial dimensions) and returns maps. Both results have `params`, `sse`
  and `status`. `init` + `fit!` remain for user-written loops.
- **Standard function names are shared, not redefined.** `fit`, `fit!` and `nobs` are StatsAPI's
  functions (the interface behind StatsBase, GLM, Distributions) and `init` is CommonSolve's
  (SciML). We add methods for our types, so loading those packages alongside causes no clash.

## Writing a model

```julia
@kwdef struct SIRFSE{A<:AbstractVector,B<:AbstractVector} <: MRIModel   # 1. acquisition + options
    TI::A
    TD::B
    magnitude::Bool = true
end
SIRFSE(TI, TD::Real, magnitude) = SIRFSE(TI, fill(TD, length(TI)), magnitude)   # one TD for all

parameters(::SIRFSE) = (                       # 2. every parameter with its default role
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf  = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),             # starts at the least-squares signal scale
    kmf = Fixed(12.5),
    R1m = Fixed(p -> p.R1f),                   # tie
    Sm  = Fixed(0.83),
)

function signal!(out, m::SIRFSE, p)            # 3. one voxel's signal at every observation
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf, R1m, Sm) = p
    ...                                        #    loop j over observations: TI[j], TD[j]
end
```

Rules for `signal!`: no allocation, no state, generic in the number type of `p` (dual numbers
pass through it). `signal(m, p)` (provided) allocates the output and fills in fixed defaults,
for simulation.

## Parameter roles

- `Free(init; lower, upper)`: estimated. `init` is a number, or `:scale` for the parameter that
  multiplies the whole signal. A `:scale` parameter starts, per voxel, at
  (s·y)/(s·s), where s is the signal at scale 1 given the other starts. This one generic rule
  replaces data normalization. Model-specific guesses are deferred.
- `Fixed(value)`: not estimated. `value` is a number, a per-voxel map, or a tie (a function of
  the other parameters, resolved after the free values are merged in, so derivatives flow
  through it). A tie may not refer to another tie.

Fit-time overrides: `fixed = (name = value, ...)`; `free = (:name, ...)` or
`free = (name = Free(...), ...)`; `start = (name = number_or_map, ...)`. A non-finite value in
the data, a fixed map or a start map marks that voxel invalid (status 6).

## Fitting pipeline

1. `setup` merges the fit's keywords into `parameters(m)`, splits free from fixed, and returns
   a `FitSpec{F,P,T}`. The free names `F` and their count `P` are type parameters, so the
   compiler generates code specialized to them. Without that, building
   `NamedTuple{F}(θ)` from a run-time list of names would be type-unstable and slow.
2. `init` checks the acquisition lengths (`nobs`) and rebuilds the model with its numeric
   vectors converted to the working precision. Together with solver options and buffers this
   gives a `FitCache`, one per thread.
3. `fit!` handles one voxel (`fit(m, y)` with a vector is `fit!(init(m), y)`): pick its map
   values, build θ0 (and the `:scale` start), run LM.
   Each LM evaluation does `p = resolve_ties(merge(NamedTuple{F}(seeded duals), fixed))`, then
   `signal!(buf, m, p)`, then sums SSE, JᵀJ and Jᵀr from `buf`. The m×P Jacobian is never
   stored and nothing is allocated.
4. `fit(m, data)` reshapes the data to voxels × observations (a view) and turns the mask into an
   index list. It splits the list into about 64 chunks per thread (at least 16 voxels each) and
   fits each chunk in a task with its own cache. Many small chunks keep threads busy when slow
   voxels are clustered; the floor keeps the ~1 µs per-task overhead negligible. It returns
   `(params = (name = map, ...), sse, status)`.

LM details: Marquardt scaling with a monotone diagonal (MINPACK), Nielsen damping updates,
bounds by projection with an active set, Cholesky without exceptions.

## Status codes

0 not fitted (mask), 1–3 converged (step, cost, gradient), 4 max iterations,
5 failed (non-finite model at the start), 6 invalid input.

## Package structure and name

**One package, in layers.** Each layer only uses the layers before it:

```
src/
  MRISignalModels.jl   the module: dependencies, exports, includes
  models/              interface (MRIModel, Free, Fixed, parameters, signal!, signal, nobs);
                       builtin.jl + one file per model
  fit/                 lm.jl (solver), setup.jl (roles -> FitSpec), cache.jl (init, fit!),
                       fit.jl (fit: one voxel or a dataset, threaded)
  analysis/            (later) CRLB, Monte Carlo, sensitivity
  design/              (later) protocol objectives (Eq. 5, Eq. 6), scan time
ext/                   (later) package extensions: OptimExt (optimize_protocol), NIfTIExt (file I/O)
```

- Why not several packages: CRLB (biased, Eq. 3), Monte Carlo and sensitivity analysis all
  need the model, its Jacobian and the fitter. Separate packages would mean keeping versions
  in step while the API is still changing, and none of these features adds a heavy dependency.
- Heavy optional dependencies (an optimizer, file formats) go in package extensions (Julia ≥
  1.9): their code loads only when the user also loads that package.
- The protocol objectives stay in the core: the signal can already be differentiated with
  respect to TI/TD (dual numbers pass through the model's vectors), so any optimizer can use
  them.
- Python and MATLAB wrappers will be separate, thin packages (PyPI, MATLAB toolbox).
- When to split later: if another project wants only the models (e.g. model-based
  reconstruction), or loading time becomes a problem. The layering makes that a move of
  folders, following the one-repository-several-packages pattern of KomaMRI (KomaMRIBase,
  KomaMRICore, ...) and MRIReco (MRIBase, ...).

**Name.** `MRISignalModels`: the signal model is what fitting, CRLB, Monte Carlo, sensitivity
analysis and protocol design have in common. It follows Julia's conventions (CamelCase, plural
for a package built around its own types) and the `MRI…` names of the MRI ecosystem. Checked
against the General registry (Oct 2026): unregistered. `QuantitativeMRI` is taken
(aTrotier/QuantitativeMRI.jl).

## Performance notes

- **SIRFSE `signal!` stays in its step-by-step form** (recover over TD, invert, recover over
  TI), which follows the paper's derivation. A rearranged version was tested: all parameter-only
  work folded into six coefficients, so each observation needs `s = M0f + A·exp(-R1⁺·TI) +
  B·exp(-R1⁻·TI)` with A, B depending only on TD, reused when TD repeats. Same values and
  derivatives. Whole-voxel fit: 1.05× faster on the paper's optimized 5-point scheme (every point
  has its own TD), 1.6× on a 16-point scheme with one TD. Kept the readable form; revisit if
  Monte Carlo or protocol optimization on constant-TD schemes becomes speed-limited.
- **A matrix-exponential SIRFSE (Eq. 1 as written) is slower**: Base `exp` on 2×2 matrices is
  ~180× slower than the closed form and cannot take dual numbers; with StaticArrays 2×2 matrices
  it is ~3× slower and works with duals. A reasonable fallback for models without a convenient
  closed form.
- **Threads**: chunks of voxels as tasks (`Threads.@spawn`), about 64 per thread with at least 16
  voxels each; plain `Threads.@threads` was 2× slower when slow voxels cluster (it splits the
  work into one block per thread). See the comment in `src/fit/fit.jl`.

## Scope so far

In: the interface above, five models (ExpDecay, InversionRecovery, SaturationRecovery, MFA,
SIRFSE), `fit` for one voxel or a dataset, Float64/Float32, bounds, masks, per-voxel maps, ties,
threads, tests.

Out for now; earlier implementations are on the `api-v0.2` and `main` branches:
CLI / Python / MATLAB, GPU, model-specific and neighbour starting values, standard errors,
weights, other data layouts, precompilation, more models.

## Roadmap

1. **Uncertainty** (formulation from Dortch et al., Magn Reson Med 2018;80:1824):
   - Fisher information for Gaussian noise (Eq. 4): F = JᵀJ/σ², with J the derivatives of
     the signal at the true parameters with respect to the free ones. This is the JᵀJ the
     solver already accumulates.
   - Unbiased estimator (e.g. SIR with kmf free): MSE(θ̂ⱼ) = [F⁻¹]ⱼⱼ.
   - Biased estimator (e.g. SIR with kmf fixed at an assumed value) (Eq. 3):
     MSE(θ̂ⱼ) = bⱼ² + [(I + ∇b) F⁻¹ (I + ∇b)ᵀ]ⱼⱼ, with bⱼ = E[θ̂ⱼ] − θⱼ and ∇b = ∂b/∂θ. As in
     the paper, b and ∇b come from fits to noise-free data with the constrained parameter at
     its assumed value, repeated with each true parameter perturbed (finite differences).
   - Protocol design helpers: the minimax objective (Eq. 5) and sensitivity efficiency
     (Eq. 6). The acquisition can be differentiated too, since `signal` accepts dual numbers
     in the model's vectors.
   - Monte Carlo with Gaussian/Rician noise (bias, SD, RMSE, SD/CRLB).
   - Validation targets: Table 1 tissue classes, Table 2 schemes, Fig. 2B.
2. **Starting values**: model-specific guesses, neighbour/previous-voxel starts with quality
   gates, refitting failed voxels.
3. **Front ends**: CLI (NIfTI), Python, MATLAB.
4. **Later**: GPU, Rician-likelihood fitting, more models.
