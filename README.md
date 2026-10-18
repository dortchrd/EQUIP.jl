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
| **model** | the physics: acquisition + options, the same for every voxel | a struct | `QMTSIR(TI = [...], TD = 2.5)` |
| **parameters** | tissue/model values | a named tuple | `(PSR = 0.1, R1f = 1.0, ...)` |

Acquisition fields have one value per observation. Where it makes sense, a single number
stands for all of them, e.g. `T1MFA(FA = [2, 5, 10, 15], TR = 0.015)`. **All times are in seconds.**

## Quick start

```julia
using EQUIP

m = QMTSIR(TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])

# Simulate. Fixed parameters (kmf, R1m, Sm, ...) take their defaults unless you give them.
y = signal(m, (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0))

# Fit one voxel: y is a vector.
r = fit(m, y)
r.params            # (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)
r.status            # ConvergedX
fitted(r)           # the fitted curve, e.g. to plot against y

# Fit a whole dataset: observations along the LAST dimension (voxels × M, X×Y×M, X×Y×Z×M, ...).
res = fit(m, data; mask)
res.params.PSR      # the PSR map, with the spatial size of data
res.status          # a FitStatus map (ConvergedX, MaxIterations, ...)
res                 # a summary: voxels fitted, % converged, medians of each parameter
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

Every model declares each parameter's default **role** with `parameters(m)`. For QMTSIR:

```julia
parameters(::QMTSIR) = (
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf  = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),
    kmf = Fixed(12.5),
    R1m = Fixed(:R1f),           # a tie: always equal to R1f
    Sm  = Fixed(0.83),
    Mzf_td0 = Fixed(0.0),        # Mz at the start of TD, as a fraction of equilibrium
    Mzm_td0 = Fixed(0.0),
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

mfa = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
fit(mfa, data; fixed = (B1 = b1map,))            # B1-corrected T1 mapping

t2 = R2DecayOffset(TE = 0.01:0.01:0.32)
fit(t2, data; fixed = (C = 40.0,))               # a known offset instead of estimating it
```

Mistakes raise an error before any fitting starts: misspelled names, a parameter in both
`fixed` and `free`, a missing acquisition field, acquisition vectors of different lengths,
or maps whose size doesn't match the data. A NaN or Inf in a voxel's data, maps or starts
marks that voxel invalid (status `InvalidData`) instead of producing a meaningless fit.

## Outputs

`fit` (and `fit!`) return a `FitResult`, the same type for one voxel and for a dataset. For one
voxel the fields are numbers; for a dataset, maps with the spatial size of the data:

- `params`: the free parameters by name (`res.params.T1`). Pass a dataset's back as
  `start = res.params` to refit from them.
- `sse`: sum of squared residuals.
- `iterations`: solver iterations.
- `status`: a `FitStatus`: `NotFitted` (outside the mask), `ConvergedX` / `ConvergedF` /
  `ConvergedG` (step / cost / gradient), `MaxIterations`, `Failed`, `InvalidData`. A dataset's
  status map takes 1 byte per voxel; `UInt8.(res.status)` gives the codes 0–6, e.g. for
  writing a file. `isconverged(res)` is true/false, or a map of them.
- `model` and `settings`: the model as fitted and the keywords of the call, so
  `fit(res.model, data; res.settings...)` refits with the same choices.

`fitted(res)` rebuilds the fitted curve of a one-voxel result, `fitted(res, i...)` that of
voxel `i` of a dataset (linear index or position, e.g. `fitted(res, 40, 52, 10)`), and
`residuals(res, y)` / `residuals(res, data, i...)` the data minus the fit. Showing a result
prints a summary: for a dataset, the voxels fitted, % converged, counts per status, the median
and interquartile range of each parameter, and the fixed parameters.

Voxels that are not fitted hold NaN parameters and SSE and 0 iterations.

## Built-in models

Names put the quantity first, then the method, so typing `T1` + Tab lists every T1 model.
Rate (R1, R2) and time-constant (T1, T2) versions fit the same data; fit the one you will
report. Magnitude inversion recovery (`T1IR`, `T1IRTD` and their R1 versions) has local minima,
so these models first try 30 values of T1 per voxel and start from the best one (`startgrid`):
in tests, at most 2 of 200 voxels failed for T1 from 0.05 to 4 s, against all 200 from a single
start. It costs about 1 µs per voxel; giving a `start` for T1 (or R1) skips it.

| area | model | signal | free | fixed |
|---|---|---|---|---|
| relaxometry | `T1IR(TI = ..., magnitude = true)` | inversion recovery: \|M0·(1 − B·exp(−TI/T1))\| | M0, B, T1 | |
| | `R1IR(TI = ..., magnitude = true)` | the same with R1 = 1/T1 | M0, B, R1 | |
| | `T1IRTD(TI = ..., TD = ..., magnitude = true)` | inversion recovery with the delay TD modelled (Li et al., MRM 2010, Eq. 1): M0·[Sf·Mz(TD)·e^(−TI/T1) + 1 − e^(−TI/T1)], Mz(TD) = 1 − (1 − Mz_td0)·e^(−TD/T1); TD one number or one per TI | M0, Sf, T1 | Mz_td0 = 0 |
| | `R1IRTD(TI = ..., TD = ..., magnitude = true)` | the same with R1 = 1/T1 | M0, Sf, R1 | Mz_td0 = 0 |
| | `T1SR(TD = ...)` | saturation recovery: M0·(1 − exp(−TD/T1)) | M0, T1 | |
| | `R1SR(TD = ...)` | the same with R1 = 1/T1 | M0, R1 | |
| | `T1MFA(FA = ..., TR = ...)` | multiple flip angles (spoiled gradient echo), flip angle B1·FA (degrees); TR one number or one per FA | M0, T1 | B1 = 1 |
| | `R1MFA(FA = ..., TR = ...)` | the same with R1 = 1/T1 | M0, R1 | B1 = 1 |
| | `T2Decay(TE = ...)` | S0·exp(−TE/T2); also T2* (simple mono-exponential case) | S0, T2 | |
| | `R2Decay(TE = ...)` | S0·exp(−R2·TE); also R2*; the more robust form | S0, R2 | |
| | `T2DecayOffset(TE = ...)` | S0·exp(−TE/T2) + C (C: a common approximation for the noise floor) | S0, T2, C | |
| | `R2DecayOffset(TE = ...)` | S0·exp(−R2·TE) + C | S0, R2, C | |
| | `T2BiexpDecay(TE = ...)` | two T2 components (e.g. myelin water): S0·[fshort·exp(−TE/T2short) + (1 − fshort)·exp(−TE/T2long)]; the shorter T2 reported as T2short | S0, fshort, T2short, T2long | |
| | `R2BiexpDecay(TE = ...)` | the same with R2short = 1/T2short, R2long = 1/T2long | S0, fshort, R2short, R2long | |
| diffusion | `ADCDecay(b = ...)` | S0·exp(−b·ADC); b in s/mm², ADC in mm²/s | S0, ADC | |
| | `ADCKurtosis(b = ...)` | kurtosis, one direction or direction-averaged (Jensen et al., MRM 2005): S0·exp(−b·ADC + b²·ADC²·K/6); K in [0, 3]; check b_max·ADC·K < 3 | S0, ADC, K | |
| qMT | `QMTSIR(TI = ..., TD = ..., magnitude = true)` | selective inversion recovery, two-pool qMT (Li et al., MRM 2010; Dortch et al., MRM 2018); TD one number or one per TI | PSR, R1f, Sf, M0f | kmf = 12.5, R1m = R1f, Sm = 0.83, Mzf_td0 = Mzm_td0 = 0 (both pools saturated at the start of TD) |

## Writing a model

A model is a struct, a `parameters` method and a `signal!` method:

```julia
using EQUIP
import EQUIP: parameters, signal!     # we add methods to these functions

@kwdef struct StretchedDecay{A<:AbstractVector} <: MRIModel
    TE::A                                # echo times (s), one per observation
end

parameters(::StretchedDecay) = (
    S0   = Free(:scale; lower = 0),
    T2   = Free(0.05; lower = 1e-6),
    beta = Free(0.9; lower = 0.1, upper = 1),   # stretching exponent (1 = mono-exponential)
)

function signal!(out, m::StretchedDecay, p)
    (; TE) = m                           # unpack the acquisition by name ...
    (; S0, T2, beta) = p                 # ... and the parameters
    @inbounds for j in eachindex(out, TE)   # one length check, then no per-element checks
        out[j] = S0 * exp(-(TE[j] / T2)^beta)
    end
    return out
end

m = StretchedDecay(TE = 0.01:0.01:0.32)  # @kwdef gives the keyword constructor
```

- **Fields.** Each vector field gets its own type parameter (`A`, `B`, ...), so any kind of
  vector works, and `init` converts it to plain floats in the working precision. Options with
  defaults (`magnitude::Bool = true`) become optional keywords.
- **Optional single values.** To let one number stand for every observation, add a one-line
  method, as `T1MFA` does for TR: `T1MFA(FA, TR::Real) = T1MFA(FA, fill(TR, length(FA)))`.
- **Rules for `signal!`.** It must not allocate, and must work for any number type, because the
  fitter passes dual numbers to get derivatives. Unpack `p` by name, never by position: its
  order depends on which parameters are free in a given fit. It never needs to know which
  parameters are free or fixed.
- **The loop.** Write `for j in eachindex(out, TE)`, listing every per-observation field
  (`eachindex(out, TI, TD)` for QMTSIR, `eachindex(out, FA, TR)` for T1MFA). It checks once,
  before the loop, that they all have the same length as `out`, and stops with a clear
  `DimensionMismatch` error if not. That makes `@inbounds` safe: it skips the per-element bounds
  checks inside the loop. `@inbounds` is optional (it gains about 5% in `signal!`); the
  `eachindex` check costs nothing measurable, so always use it. Without both, a wrong-length
  `out` would read past the end of `TE` and return garbage instead of an error.
- **Parametrization.** Fit the quantity you will report (T2 or R2), so standard errors, CRLB and
  Monte Carlo come out directly for it. When both forms of a model exist, they are separate
  models (`T2Decay`, `R2Decay`) with their own `parameters` and docstrings, but the equation is
  written once, in the rate form; the time form converts and calls it:
  `signal!(out, m::T2Decay, p) = signal!(out, R2Decay(m.TE), (; p.S0, R2 = 1 / p.T2))`.
  Always time → rate: R2 = 0 (a flat curve) is a valid value, but 1/0 is not. Variations such as
  an added offset are separate models with their own loop.
- **`nobs`** (the number of observations) works for every model automatically: it is the common
  length of the model's vector fields. A model with other kinds of vector fields can define its
  own method.
- **`startgrid`** (optional): if fits can end in a wrong local minimum from one start, give a
  grid of values for one free parameter, e.g.
  `startgrid(m::T1IR) = (T1 = exp.(range(log(0.01), log(10.0); length = 30)),)`. Each voxel
  evaluates the signal at every value (a `:scale` parameter at its best value for each) and
  starts from the best fit. The default is no grid.
- **`datastart`** (optional): starting values computed from each voxel's data, such as a
  linearized fit. T2Decay's is a log-linear fit:
  `datastart(m::T2Decay, y) = (T2 = 1 / _loglinear_rate(m.TE, y),)`. Return NaN when the data
  give no estimate; the usual start then stays. A `start` from the user always wins.
- **`relabel`** (optional): if the parameters can describe the same signal in more than one way
  by swapping labels, choose the one to report. `T2BiexpDecay`'s two components can swap, so
  its `relabel` puts the shorter T2 first. Called once per voxel after the fit; it must not
  change the signal. The default changes nothing.

## How it works

```
parameters(m) + fixed/free/start  ──setup──▶  FitSpec   free names, bounds, starts, fixed values
FitSpec + model + buffers         ──init───▶  FitCache  made once (one copy per thread)
one voxel's data                  ──fit!───▶  FitResult numbers
a dataset                         ──fit────▶  FitResult maps; threads over chunks of voxels
```

Inside each LM step, the free values θ become dual numbers and are merged with the fixed values
into the full named tuple `p`. Then `signal!` fills a buffer of duals (values plus derivatives),
and JᵀJ, Jᵀr and the SSE are summed from it. The Jacobian matrix itself is never stored.
Before the first step, each voxel's start is set: the model's own estimates from the data, for
a model with a `datastart`; a `:scale` parameter at its least-squares value; and, for a model with
a `startgrid`, the grid value that fits that voxel best.

The code is in layers; each only uses the ones before it:

```
src/
  EQUIP.jl                the module: dependencies, exports, includes
  models/                 interface.jl (MRIModel, Free, Fixed, signal!, ...), builtin.jl (includes)
    relaxometry/          t1_ir.jl, t1_sr.jl, t1_mfa.jl, t2_decay.jl, t2_biexp_decay.jl
    diffusion/            adc_decay.jl, adc_kurtosis.jl
    qmt/                  qmt_sir.jl
  fit/                    levmar.jl (solver), setup.jl (roles -> FitSpec),
                          cache.jl (init, fit!), fit.jl (fit)
```

`docs/design.md` records the design decisions.

## Roadmap

Deferred on purpose. Earlier implementations of some of these exist on the `api-v0.2` and
`main` branches.

- **sim/**: `simulate` (Rician/Gaussian noise), digital phantoms (label images → parameter maps,
  synthetic shapes, later brain-like segmentations), tissue presets (the 2018 paper's Table 1
  for SIR), and `compare` against the truth; shared by Monte Carlo, benchmarks and tests
- **bench/**: BenchmarkTools speed suite (own Project.toml) and accuracy studies on phantoms
  (starting values, bounds, accuracy vs SNR), with recorded results
- **analysis/**: standard errors (in `FitResult`, with StatsAPI's `coef`, `stderror`, `vcov`, and
  a flag for parameters that ended at a bound), CRLB for unbiased and biased estimators (Dortch et al. 2018,
  Eqs. 3–6), Monte Carlo with Gaussian/Rician noise, sensitivity analysis (including model
  assumptions, e.g. how much Mz left after the SIR readout, `Mzf_td0`, biases PSR and kmf)
- **design/**: protocol objectives (Eq. 5 minimax, Eq. 6 efficiency) and protocol optimization
  (best TI/TD), with an Optim.jl package extension
- Starting values as candidates: the default, a user value, the model's data start and its grid
  each propose starting points, and the fit starts from the one with the lowest SSE. One option
  per fit chooses the set: `starts = :auto` (the model's recommendation, the default),
  `(:data, :grid)`, or `:default` (plain defaults, for comparisons)
- Outlier detection after a dataset fit (status, a parameter at a bound, deviation from
  neighbouring voxels), then a refit with the neighbours' values as a starting candidate or
  replacement by a neighbourhood average, marked in the status map
- **io/**: read data as easily as possible (NIfTI, BIDS JSON sidecars, MAT, .npy/.npz, DICOM,
  PAR/REC and XML/REC, ISMRMRD images), one package extension per format. Acquisition values
  missing from headers (e.g. TI/TD for SIR) come from the call or a protocol file
  (TOML/JSON/CSV), converted to seconds/degrees. Write maps back in the input's format and
  geometry, with BIDS-derivative names and a JSON record of the fit settings
- Command line, Python and MATLAB front ends, built on io/
- A `levmar` method for ordinary arrays, for models with more than ~30–40 free parameters
  (chosen by dispatch; the current method uses static arrays and is faster below that)
- GPU, observation weights, other data layouts, precompilation for fast start-up
- More models: gradient-echo decay with the through-slice sinc term for macroscopic field
  gradients (a separate model; the frequency spread from a B0 field map or fitted); diffusion
  tensor and kurtosis, IVIM, multi-exponential T2 (myelin water), DCE, more qMT methods
- More data-driven starting values (with `datastart`): DESPOT1 for T1MFA, which needs the
  voxel's B1, so `datastart` would also receive the fixed values; R2Decay and ADCDecay
- Variable projection: solve M0/S0 exactly at every iteration, not only at the start
