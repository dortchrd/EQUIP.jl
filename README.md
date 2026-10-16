# EQUIP.jl

**E**stimation of **QU**antitative **I**maging **P**arameters: MRI signal models and fast
voxel-wise fitting, for quantitative maps (T1, T2, qMT, ...). Built for millions of small fits:
fewer than 10 parameters and fewer than 100 observations per voxel. A custom
Levenberg–Marquardt solver with automatic derivatives fits one voxel in a few microseconds
without allocating memory, and threads spread the voxels over your CPU cores.

The signal model is the center of the package. Fitting comes first; CRLB, Monte Carlo,
sensitivity analysis and protocol optimization will build on the same models (see
[Roadmap](#roadmap)).

## Install and test

```julia
] dev /path/to/EQUIP.jl
```

```sh
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

Start Julia with threads (`julia -t auto`) so fitting a dataset uses all cores.

## Two inputs

| input | what it is | Julia form | example |
|---|---|---|---|
| **model** | the physics: acquisition + options, the same for every voxel | a struct | `SIRFSE(TI = [...], TD = 2.5)` |
| **parameters** | tissue/model values | a named tuple | `(PSR = 0.1, R1f = 1.0, ...)` |

Acquisition fields have one value per observation. Where it makes sense, a single number
stands for all of them, e.g. `MFA(FA = [2, 5, 10, 15], TR = 0.015)`. **All times are in seconds.**

## Quick start

```julia
using EQUIP

m = SIRFSE(TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])

# Simulate. Fixed parameters (kmf, R1m, Sm) take their defaults unless you give them.
y = signal(m, (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0))

# Fit one voxel: y is a vector.
r = fit(m, y)
r.params            # (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)
r.status            # ConvergedX

# Fit a whole dataset: observations along the LAST dimension (voxels × M, X×Y×M, X×Y×Z×M, ...).
res = fit(m, data; mask)
res.params.PSR      # the PSR map, with the spatial size of data
res.status          # FitStatus codes per voxel
```

`fit` chooses by the shape of the data: a vector is one voxel and gives plain numbers; a
matrix or N-d array is many voxels and gives maps. A matrix is voxels × observations, one row
per voxel.

For your own loops (simulations, Monte Carlo), make a reusable cache once:

```julia
cache = init(m; fixed = (kmf = 15.0,))
for y in curves
    r = fit!(cache, y)          # no memory allocation per fit
end
```

`fit`, `fit!` and `nobs` are the same functions as in StatsBase/GLM (from StatsAPI.jl), and
`init` is SciML's (from CommonSolve.jl), so loading those packages alongside causes no clash.

## Free and fixed parameters

Every model declares each parameter's default **role** with `parameters(m)`. For SIRFSE:

```julia
parameters(::SIRFSE) = (
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf  = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),
    kmf = Fixed(12.5),
    R1m = Fixed(:R1f),           # a tie: always equal to R1f
    Sm  = Fixed(0.83),
)
```

- `Free(init; lower, upper)`: estimated, starting at `init` and kept within the bounds.
  `init = :scale` is for the parameter that scales the whole signal (S0, M0). It starts at
  the least-squares value given the other starts, so raw data need no normalization.
- `Fixed(value)`: held fixed at a number, a per-voxel map (e.g. B1), or a tie to other
  parameters: `Fixed(:R1f)` (always equal to R1f) or a function such as
  `Fixed(p -> 0.8 * p.R1f)`. A tie can't refer to another tie.

Any fit can change the roles with three keywords, accepted by `fit` and `init`:

```julia
fit(m, data; fixed = (kmf = 15.0,))              # change a fixed value
fit(m, data; fixed = (Sm = sm_map,))             # a per-voxel map
fit(m, data; free = (:kmf,))                     # estimate kmf (starts at 12.5)
fit(m, data; free = (kmf = Free(12.5; lower = 1, upper = 50),))
fit(m, data; fixed = (PSR = 0.1,))               # fix a parameter that is free by default
fit(m, data; start = (R1f = r1_map,))            # starting values: numbers or maps

mfa = MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
fit(mfa, data; fixed = (B1 = b1map,))            # B1-corrected T1 mapping
```

Mistakes raise an error before any fitting starts: misspelled names, a parameter in both
`fixed` and `free`, a missing acquisition field, acquisition vectors of different lengths,
or maps whose size doesn't match the data. A NaN or Inf in a voxel's data, maps or starts
marks that voxel invalid (status 6) instead of producing a meaningless fit.

## Outputs

One voxel, `fit(m, y)`: a `VoxelResult` with `params` (named tuple of numbers), `sse`,
`iterations` and `status`.

A dataset, `fit(m, data)`: a named tuple of arrays with the spatial size of the data:

- `params`: one map per free parameter (`res.params.T1`). You can pass it back as
  `start = res.params` to refit.
- `sse`: sum of squared residuals.
- `status`: `UInt8` codes. 0 = outside the mask, 1–3 = converged (step / cost / gradient),
  4 = max iterations, 5 = failed, 6 = invalid input. `FitStatus(code)` gives the name;
  `isconverged(code)` tests for 1–3.

Voxels that are not fitted hold NaN.

## Built-in models

| model | signal | free | fixed |
|---|---|---|---|
| `ExpDecay(TE = ...)` | S0·exp(−TE/T2) | S0, T2 | |
| `InversionRecovery(TI = ..., magnitude = true)` | \|M0·(1 − B·exp(−TI/T1))\| | M0, B, T1 | |
| `SaturationRecovery(TD = ...)` | M0·(1 − exp(−TD/T1)) | M0, T1 | |
| `MFA(FA = ..., TR = ...)` | spoiled gradient echo, flip angle B1·FA (degrees); TR one number or one per FA | M0, T1 | B1 = 1 |
| `SIRFSE(TI = ..., TD = ..., magnitude = true)` | two-pool qMT (Dortch et al., MRM 2018, Eq. 1); TD one number or one per TI | PSR, R1f, Sf, M0f | kmf = 12.5, R1m = R1f, Sm = 0.83 |

## Writing a model

A model is a struct, a `parameters` method and a `signal!` method:

```julia
using EQUIP
import EQUIP: parameters, signal!     # we add methods to these functions

@kwdef struct T2Offset{A<:AbstractVector} <: MRIModel
    TE::A                                # echo times (s), one per observation
end

parameters(::T2Offset) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(0.05; lower = 1e-6),
    C  = Free(0.0; lower = 0),           # a noise floor
)

function signal!(out, m::T2Offset, p)
    (; TE) = m                           # unpack the acquisition by name ...
    (; S0, T2, C) = p                    # ... and the parameters
    for j in eachindex(out)
        out[j] = S0 * exp(-TE[j] / T2) + C
    end
    return out
end

m = T2Offset(TE = 0.01:0.01:0.32)        # @kwdef gives the keyword constructor
```

- **Fields.** Each vector field gets its own type parameter (`A`, `B`, ...), so any kind of
  vector works, and `init` converts it to plain floats in the working precision. Options with
  defaults (`magnitude::Bool = true`) become optional keywords.
- **Optional single values.** To let one number stand for every observation, add a one-line
  method, as `MFA` does for TR: `MFA(FA, TR::Real) = MFA(FA, fill(TR, length(FA)))`.
- **Rules for `signal!`.** It must not allocate, and must work for any number type, because the
  fitter passes dual numbers to get derivatives. Unpack `p` by name, never by position: its
  order depends on which parameters are free in a given fit. It never needs to know which
  parameters are free or fixed.
- **`nobs`** (the number of observations) works for every model automatically: it is the common
  length of the model's vector fields. A model with other kinds of vector fields can define its
  own method.

## How it works

```
parameters(m) + fixed/free/start  ──setup──▶  FitSpec   free names, bounds, starts, fixed values
FitSpec + model + buffers         ──init───▶  FitCache  made once (one copy per thread)
one voxel's data                  ──fit!───▶  VoxelResult
a dataset                         ──fit────▶  maps      threads over chunks of voxels
```

Inside each LM step, the free values θ become dual numbers and are merged with the fixed values
into the full named tuple `p`. Then `signal!` fills a buffer of duals (values plus derivatives),
and JᵀJ, Jᵀr and the SSE are summed from it. The Jacobian matrix itself is never stored.

The code is in layers; each only uses the ones before it:

```
src/
  EQUIP.jl                the module: dependencies, exports, includes
  models/                 interface.jl (MRIModel, Free, Fixed, signal!, ...), builtin.jl, one file per model
  fit/                    levmar.jl (solver), setup.jl (roles -> FitSpec),
                          cache.jl (init, fit!), fit.jl (fit)
```

`docs/design.md` records the design decisions.

## Roadmap

Deferred on purpose. Earlier implementations of some of these exist on the `api-v0.2` and
`main` branches.

- **One result type for `fit`**, the same for one voxel or a dataset: params, SSE, iterations
  and status (as names, e.g. `ConvergedX`), plus the model and settings used (for provenance,
  fitted curves and refits) and a summary display; later JᵀJ / standard errors and StatsAPI
  queries (`coef`, `stderror`, `vcov`, ...)
- **sim/**: `simulate` (Rician/Gaussian noise), digital phantoms (label images → parameter maps,
  synthetic shapes, later brain-like segmentations), tissue presets (the 2018 paper's Table 1
  for SIR), and `compare` against the truth; shared by Monte Carlo, benchmarks and tests
- **bench/**: BenchmarkTools speed suite (own Project.toml) and accuracy studies on phantoms
  (starting values, bounds, accuracy vs SNR), with recorded results
- **analysis/**: standard errors, CRLB for unbiased and biased estimators (Dortch et al. 2018,
  Eqs. 3–6), Monte Carlo with Gaussian/Rician noise, sensitivity analysis
- **design/**: protocol objectives (Eq. 5 minimax, Eq. 6 efficiency) and protocol optimization
  (best TI/TD), with an Optim.jl package extension
- Model-specific data-driven starting values; starts from neighbouring voxels
- Outlier detection after a dataset fit (status, a parameter at a bound, deviation from
  neighbouring voxels), then a refit from neighbour starts or replacement by a neighbourhood
  average, marked in the status map
- **io/**: read data as easily as possible (NIfTI, BIDS JSON sidecars, MAT, .npy/.npz, DICOM,
  PAR/REC and XML/REC, ISMRMRD images), one package extension per format. Acquisition values
  missing from headers (e.g. TI/TD for SIR) come from the call or a protocol file
  (TOML/JSON/CSV), converted to seconds/degrees. Write maps back in the input's format and
  geometry, with BIDS-derivative names and a JSON record of the fit settings
- Command line, Python and MATLAB front ends, built on io/
- A `levmar` method for ordinary arrays, for models with more than ~30–40 free parameters
  (chosen by dispatch; the current method uses static arrays and is faster below that)
- GPU, observation weights, other data layouts, precompilation for fast start-up
- More models (diffusion, IVIM, multi-exponential, DCE)
