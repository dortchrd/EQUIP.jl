# VoxelFit 0.2 design

The design agreed for 0.2, recorded so the code and discussion have one reference.

## Model inputs: four layers

| layer | varies across voxels? | examples | where it lives |
|---|---|---|---|
| **Model**: physics and implementation choices | no | magnitude on/off | fields of the model struct, e.g. `SIRFSE(magnitude = true)` |
| **Protocol**: experimental settings | no | per observation: TE, TI, TD, flip angle; constants: TR | one named tuple `acq`, e.g. `(TI = [...], TD = 2.5)`: vectors are per observation, scalars are constant |
| **Parameters**: every quantity in the signal equation | each one: estimated, a constant, a per-voxel map, or a tie | PSR, kmf, R1f, B1, Sm | declared by the model with default roles; overridden per fit with `free` / `fixed` |
| **Data** | yes | the images | N-d array; observations along one dimension |

## Writing a model

A model has three pieces:

```julia
struct SIRFSE <: AbstractModel                 # 1. settings, the same for every voxel
    magnitude::Bool
end
SIRFSE(; magnitude = true) = SIRFSE(magnitude)

parameters(::SIRFSE) = (                       # 2. every parameter, with its default role
    pmf = Estimate(0.1; lower = 0, upper = 1),
    R1f = Estimate(1.0; lower = 0.05, upper = 10),
    Sf  = Estimate(-0.95; lower = -1.2, upper = 0),
    M0f = Estimate(:auto; lower = 0),          # :auto -> initial_guess
    kmf = Fixed(14.5),
    R1m = Fixed(p -> p.R1f),                   # tie
    Sm  = Fixed(0.83),
)

function signal!(out, m::SIRFSE, acq, p)       # 3. one voxel's signal at every observation
    ...                                        #    p.pmf, p.kmf, ... ; acq.TI[i], acq.TD
end
```

Optional: `initial_guess(m, acq, y, p)` returns a named tuple of starting values for parameters
declared `Estimate(:auto)`; `p` holds the voxel's fixed values (e.g. its B1).
`protocol(m)` names the per-observation settings, so a plain vector can stand in for a
one-setting protocol (`fit_maps(MonoExp(), te, data)`).

Rules for `signal!`: pure (no noise, no state), generic in number type (dual numbers pass
through `p`), parameters accessed by name.

`signal(m, acq, p)` (provided) allocates the output; for simulation and plotting.

## Parameter roles

- `Estimate(init; lower, upper)`: estimated. `init` is a number or `:auto`.
- `Fixed(value)`: not estimated. `value` is a number, a per-voxel map (an array over the voxel
  grid), or a function of the other parameters (a tie, e.g. `p -> p.R1f`). A tie may refer to
  estimated or fixed parameters, but not to another tie.

Overrides at fit time:

```julia
fit_maps(SIRFSE(), acq, data; fixed = (kmf = 12.0, Sm = sm_map))        # fix / set maps
fit_maps(SIRFSE(), acq, data; free = (:R1m,))                           # estimate a default-fixed one
fit_maps(SIRFSE(), acq, data; free = (:kmf => Estimate(20.0; lower = 5, upper = 50),))
fit_maps(SIRFSE(), acq, data; start = (pmf = 0.12, R1f = r1_map))       # starting values / maps
```

A non-finite value in a fixed map marks that voxel invalid (status 6), like non-finite data.

## Fitting

- Each voxel is fitted independently. For each iteration, the fitter builds the voxel's full
  parameter set `p` (estimates as dual numbers, fixed values, resolved ties), calls `signal!`
  into a per-thread buffer of dual numbers, and accumulates SSE, JᵀJ and Jᵀr from it. The
  m×P Jacobian is never stored. Levenberg–Marquardt as in 0.1 (Marquardt scaling, Nielsen
  damping, projected bounds with an active set).
- Reusable state (setup, buffers) lives in a cache:

  ```julia
  cache = init(SIRFSE(), acq; fixed = (...), free = (...))
  r = fit!(cache, y; voxel = v)        # one voxel, no allocation
  ```

  `fit_maps` creates one cache per thread. `init` is `CommonSolve.init`, so it does not clash
  with SciML packages.
- Data stay in their natural N-d shape; internally 2D views plus a list of masked voxels, and
  small per-thread chunk buffers. Outputs keep the spatial shape.
- Outputs: maps of the estimated parameters, SSE, iterations, status, optional standard errors.

## Status codes

0 not fitted (mask), 1–3 converged (step, cost, gradient), 4 max iterations,
5 failed (non-finite model at the start), 6 invalid (non-finite data or fixed-map value).

## Phases

1. **Core rewrite** (this branch): the API above, all built-in models ported, SIRFSE added,
   CLI / Python / MATLAB updated, no performance regression.
2. **Analysis** (formulation from Dortch et al., Magn Reson Med 2018;80:1824, "the 2018 paper"):
   - Fisher information for Gaussian noise (Eq. 4): F = JᵀJ/σ², J the derivatives of the
     signal at the true parameters with respect to the estimated ones (per-observation
     weights generalize this to JᵀWJ). This is the JᵀJ the solver already accumulates.
   - Unbiased estimator (e.g. SIR with kmf estimated): MSE(θ̂ⱼ) = [F⁻¹]ⱼⱼ.
   - Biased estimator (e.g. SIR with kmf fixed at an assumed value) (Eq. 3):
     MSE(θ̂ⱼ) = bⱼ² + [(I + ∇b) F⁻¹ (I + ∇b)ᵀ]ⱼⱼ, with bⱼ = E[θ̂ⱼ] − θⱼ and ∇b = ∂b/∂θ.
     As in the paper, b and ∇b are obtained numerically: fit noise-free data simulated at the
     true parameters while holding the constrained parameter at its assumed value, and
     repeat with each true parameter perturbed (finite differences). A linearized
     alternative (∇b from the implicit-function theorem, one pass) can be offered as an
     option and checked against it.
   - Helpers for protocol design: the minimax objective of Eq. 5 (normalized MSE summed over
     parameters, worst tissue class, times total scan time Σ(tI + tD + tFSE)) and the
     sensitivity efficiency of Eq. 6 (∂M_zf/∂θ / √(tI + tD + tFSE)).
   - `simulate` with Gaussian or Rician noise and a Monte Carlo summary (bias, SD, RMSE,
     SD/CRLB), as used in the paper to validate the CRLB (10,000 realizations, SNR 200).
   - Validation targets from the paper: Table 1 tissue classes (Sm = 0.83, Sf = −1,
     M0f = 1, tFSE = 0.3 s, R1m = R1f, SNR = 200), Table 2 schemes, the 16-point reference
     scheme (15 tI log-spaced 0.01–2 s plus tI = 10 s, tD = 2.5 s), the reported CRLB-predicted
     PSR efficiency gains (1.4× 5-point, 1.8× 4-point) and Fig. 2B (PSR bias vs true kmf with
     kmf fixed at 12.5 s⁻¹).
3. **Starting values and sensitivity**: candidate starts (model guess, previous voxel), a
   second pass refitting failed/outlier voxels from their neighbours, linearized sensitivity
   maps (e.g. ∂PSR/∂kmf).
4. **Later**: GPU (removed in 0.2, see 0.1 for the KernelAbstractions version; would need a
   one-point `signal` per model), coarse-to-fine starts, Rician-likelihood fitting.

## Decisions

- Keywords are `free` and `fixed`.
- Bad values in fixed maps → voxel invalid.
- GPU deferred.
- Parameter names are ASCII (they become MATLAB struct fields and CLI file names).
