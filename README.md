# VoxelFit.jl (lean)

Fast voxel-wise nonlinear least-squares fitting of MRI signal models, for parametric maps
(T1, T2, qMT, ...). Built for millions of small fits: fewer than 10 parameters and fewer than
100 observations per voxel. A custom Levenberg–Marquardt solver with automatic derivatives
fits one voxel in about 2 µs without allocating memory, and threads spread the voxels over
your CPU cores.

This is the **lean** version: only the core, so that it is easy to read and understand. CLI,
Python/MATLAB, GPU, data-driven starting values and uncertainty estimates come later (see
[Roadmap](#roadmap)).

## Install and test

```julia
] dev /path/to/VoxelFit.jl
```

```sh
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

Start Julia with threads (`julia -t auto`) so `fit_maps` uses all cores.

## Three inputs, three homes

| input | what it is | Julia form | example |
|---|---|---|---|
| **model** | which physics, plus options | a struct | `SIRFSE(magnitude = true)` |
| **acquisition** | how the experiment was run | a named tuple | `(TI = [...], TD = [...])` |
| **parameters** | tissue/model values | a named tuple | `(PSR = 0.1, R1f = 1.0, ...)` |

In the acquisition, vector fields have one value per observation and scalar fields apply to
all of them, e.g. `(FA = [2, 5, 10, 15], TR = 0.015)`. **All times are in seconds.**

## Quick start

```julia
using VoxelFit

acq = (TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])

# Simulate. Fixed parameters (kmf, R1m, Sm) take their defaults unless you give them.
y = signal(SIRFSE(), acq, (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0))

# Fit one voxel with a reusable cache.
cache = init(SIRFSE(), acq)
r = fit!(cache, y)
r.params            # (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)
r.status            # ConvergedX

# Fit a whole dataset: data are X×Y×Z×M (observations last), maps come back X×Y×Z.
res = fit_maps(SIRFSE(), acq, data; mask)
res.params.PSR      # the PSR map
res.status          # FitStatus codes per voxel
```

## Free and fixed parameters

Every model declares each parameter's default **role** with `parameters(m)`. For SIRFSE:

```julia
parameters(::SIRFSE) = (
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf  = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),
    kmf = Fixed(12.5),
    R1m = Fixed(p -> p.R1f),     # a tie
    Sm  = Fixed(0.83),
)
```

- `Free(init; lower, upper)`: estimated, starting at `init` and kept within the bounds.
  `init = :scale` is for the parameter that scales the whole signal (S0, M0). It starts at
  the least-squares value given the other starts, so raw data need no normalization.
- `Fixed(value)`: held fixed at a number, a per-voxel map (e.g. B1), or a tie, i.e. a
  function of the other parameters such as `p -> p.R1f`.

Any fit can change the roles with three keywords, accepted by `init` and `fit_maps`:

```julia
fit_maps(SIRFSE(), acq, data; fixed = (kmf = 15.0,))              # change a fixed value
fit_maps(SIRFSE(), acq, data; fixed = (Sm = sm_map,))             # a per-voxel map
fit_maps(SIRFSE(), acq, data; free = (:kmf,))                     # estimate kmf (starts at 12.5)
fit_maps(SIRFSE(), acq, data; free = (kmf = Free(12.5; lower = 1, upper = 50),))
fit_maps(SIRFSE(), acq, data; fixed = (PSR = 0.1,))               # fix a parameter that is free by default
fit_maps(SIRFSE(), acq, data; start = (R1f = r1_map,))            # starting values: numbers or maps
fit_maps(MFA(), acq, data; fixed = (B1 = b1map,))                 # B1-corrected T1 mapping
```

Mistakes such as misspelled names, a parameter in both `fixed` and `free`, or wrong map sizes
raise an error before any fitting starts. A NaN or Inf in a voxel's data, maps or starts marks
that voxel invalid (status 6) instead of producing a meaningless fit.

## Outputs

`fit_maps` returns a named tuple of arrays with the spatial size of the data:

- `params`: one map per free parameter (`res.params.T1`). You can pass it back as
  `start = res.params` to refit.
- `sse`: sum of squared residuals.
- `status`: `UInt8` codes. 0 = outside the mask, 1–3 = converged (step / cost / gradient),
  4 = max iterations, 5 = failed, 6 = invalid input. `FitStatus(code)` gives the name;
  `isconverged(code)` tests for 1–3.

Voxels that are not fitted hold NaN.

## Built-in models

| model | signal | acquisition | free | fixed |
|---|---|---|---|---|
| `ExpDecay()` | S0·exp(−TE/T2) | `TE` | S0, T2 | |
| `InversionRecovery(; magnitude = true)` | \|M0·(1 − B·exp(−TI/T1))\| | `TI` | M0, B, T1 | |
| `SaturationRecovery()` | M0·(1 − exp(−TD/T1)) | `TD` | M0, T1 | |
| `MFA()` | spoiled gradient echo, flip angle B1·FA | `FA` (deg), `TR` | M0, T1 | B1 = 1 |
| `SIRFSE(; magnitude = true)` | two-pool qMT (Dortch et al., MRM 2018, Eq. 1) | `TI`, `TD` | PSR, R1f, Sf, M0f | kmf = 12.5, R1m = R1f, Sm = 0.83 |

## Writing a model

A model is a struct, a `parameters` method and a `signal!` method:

```julia
using VoxelFit
import VoxelFit: parameters, signal!     # we add methods to these functions

struct T2Offset <: AbstractModel end     # options would be fields; this model has none

parameters(::T2Offset) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(0.05; lower = 1e-6),
    C  = Free(0.0; lower = 0),           # a noise floor
)

function signal!(out, ::T2Offset, acq, p)
    (; S0, T2, C) = p                    # every parameter by name, as numbers
    for j in eachindex(out)
        (; TE) = obs(acq, j)             # this observation's settings
        out[j] = S0 * exp(-TE / T2) + C
    end
    return out
end
```

`signal!` must not allocate, and must work for any number type, because the fitter passes dual
numbers to get derivatives. It never needs to know which parameters are free or fixed.

## How it works

```
parameters(m) + fixed/free/start  ──setup──▶  FitSpec   free names, bounds, starts, fixed values
FitSpec + acquisition + buffers   ──init───▶  FitCache  made once (one copy per thread)
one voxel's data                  ──fit!───▶  VoxelResult
all voxels                        ──fit_maps─▶ maps      threads over chunks of voxels
```

Inside each LM step, the free values θ become dual numbers and are merged with the fixed values
into the full named tuple `p`. Then `signal!` fills a buffer of duals (values plus derivatives),
and JᵀJ, Jᵀr and the SSE are summed from it. The Jacobian matrix itself is never stored. The
code is in `src/`, roughly in reading order: `interface.jl`, `fit.jl`, `lm.jl`, `maps.jl`,
`models.jl`. `docs/design.md` records the design decisions.

## Roadmap

Deferred on purpose. Earlier implementations exist on the `api-v0.2` and `main` branches.

- Uncertainty: standard errors, CRLB for unbiased and biased estimators (Dortch et al. 2018,
  Eqs. 3–6), Monte Carlo with Gaussian/Rician noise
- Model-specific data-driven starting values; starts from neighbouring voxels
- Command line (NIfTI/npy), Python and MATLAB front ends
- GPU, observation weights, other data layouts, precompilation for fast start-up
- More models (diffusion, IVIM, multi-exponential, DCE)
