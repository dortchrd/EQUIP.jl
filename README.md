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

Requires Julia 1.11 or later.

```julia
] dev /path/to/EQUIP.jl
```

```sh
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

Start Julia with threads (`julia -t auto`) so fitting a dataset uses all cores.

Complete examples, Monte Carlo tests of the 4-point SIR scheme at SNR 200 (Dortch et al. 2018),
saved as CSV files: `examples/sir_wm_4point.jl` (white matter) and `examples/sir_4point_range.jl`
(tissue values over the range of the paper's tissue classes), e.g.
`julia -t auto --project=. examples/sir_4point_range.jl`. `bench/solvers.jl` compares EQUIP with
LsqFit, NonlinearSolve (SciML) and FastLevenbergMarquardt on test cases from easy to hard
(`bench/cases.jl`: T2 decay, variable flip angle and inversion recovery T1, the same 4-point data,
poor starts, 16-point SIR with kmf estimated, low SNR, a small PSR, a biexponential), each run
without bounds and with the models' bounds: speed, and whether each finds the best fit.

The first `using EQUIP` after installing (or after any change to its code) takes about 40 s:
EQUIP then compiles small fits of every built-in model and saves the result, so that in every
new session the first fit is as fast as later ones (the default roles, one voxel or a dataset,
with or without a mask). Other combinations (other free parameters, `precision = Float32`)
compile on their first use, about a second or two. When developing EQUIP, switch that step off
to keep each change to about 3 s: put

```toml
[EQUIP]
precompile_workload = false
```

in a `LocalPreferences.toml` next to `Project.toml` (it is not committed).

## Two inputs

| input | what it is | Julia form | example |
|---|---|---|---|
| **model** | the physics: acquisition + options, the same for every voxel | a struct | `QMTSIR(TI = [...], TD = 2.5)` |
| **parameters** | tissue/model values | a named tuple | `(PSR = 0.1, R1f = 1.0, ...)` |

Two words keep these apart throughout EQUIP:

- **parameters** are the quantities in the signal equation (T1, PSR, M0, B1, ...), whether a
  fit estimates them or holds them fixed;
- the **acquisition** is the protocol (TI, TE, flip angles, ...), never called "parameters".

A model is the signal equation (its type, e.g. `QMTSIR`) plus its acquisition and options (its
fields).

Acquisition fields have one value per observation. Where it makes sense, a single number
stands for all of them, e.g. `T1MFA(FA = [2, 5, 10, 15], TR = 0.015)`. **All times are in seconds.**

## Quick start

```julia
using EQUIP

m = QMTSIR(TI = [0.010, 0.050, 0.056, 0.277, 0.843], TD = [3.270, 4.489, 1.652, 2.922, 0.010])

# Simulate. Every parameter needs a value: `nominal(m)` takes them all from the model (its
# starting and fixed values: kmf = 12.5, R1m = R1f, ...), and keywords change any of them.
y = signal(m, nominal(m; PSR = 0.12, R1f = 1.0, M0f = 1000.0))

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
cache = init(m; kmf = 15.0)
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
    PSR = Free(start = 0.1, lower = 0, upper = 1),
    R1f = Free(start = 1.0, lower = 0.05, upper = 10),
    Sf  = Free(start = -0.95, lower = -1.2, upper = 0),
    M0f = Free(start = :scale, lower = 0),
    kmf = Fixed(12.5),
    R1m = Fixed(:R1f),           # a tie: always equal to R1f
    Sm  = Fixed(0.83),
    Mzf_td0 = Fixed(0.0),        # Mz at the start of TD, as a fraction of equilibrium
    Mzm_td0 = Fixed(0.0),
)
```

- `Free(; start, lower, upper)`: estimated, kept within the bounds. Each part is a keyword,
  named in the call (`Free(start = 0.1, lower = 0)`). The start says where each voxel's
  starting value comes from:
  - a number (or, in a fit, a map with one value per voxel);
  - `:scale`, for the parameter that scales the whole signal (S0, M0): the least-squares value
    given the other starts, so raw data need no normalization. Only the model declares it (the
    rule assumes the signal is proportional to the parameter): a fit can turn it off,
    `M0 = Free(start = 1000.0)`, but not move it to another parameter;
  - `:data`: the model's own estimate from the voxel's data (`datastart`), e.g. T2Decay's
    log-linear fit, `T2 = Free(start = :data, lower = ...)`;
  - `BestOf(values)`: try each value (with the `:scale` parameter solved for each) and start
    from the one that fits best, e.g. the inversion-recovery models'
    `T1 = Free(start = BestOf(_t1grid(m.TI)), lower = ...)`, 30 values spanning the TIs. One
    parameter per fit for now. `BestOf` wraps the values because a plain vector start is a map.
- `Fixed(value)`: held fixed at a number, a per-voxel map (e.g. B1), or a tie to other
  parameters: `Fixed(:R1f)` (always equal to R1f) or a function such as
  `Fixed(p -> 0.8 * p.R1f)`. A tie can't refer to another tie. `defaults(m)` gives the fixed
  values as a named tuple, e.g. `(kmf = 12.5, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)`.

Any fit can change a role with a keyword named after the parameter, in `fit` and `init`. A
value fixes the parameter; `Free(...)` estimates it, and whatever it leaves out (start, bounds)
keeps the model's default:

```julia
fit(m, data; kmf = 15.0)                         # change a fixed value
fit(m, data; Sm = sm_map)                        # a per-voxel map
fit(m, data; PSR = 0.1)                          # fix a parameter that is free by default
fit(m, data; kmf = Free())                       # estimate kmf (starts at 12.5, no bounds)
fit(m, data; kmf = Free(lower = 1, upper = 50))  # ... within bounds
fit(m, data; R1f = Free(start = r1_map))         # another start: a number or a map
fit(m, data; R1f = Free(start = BestOf([0.5, 1.0, 2.0])))   # try three, start from the best
fit(m, data; Sf = Free(lower = -1.5))            # another bound; start and upper bound as before
fit(m, data; kmf = 15.0, Sm = sm_map, mask)      # several at once (`mask` means `mask = mask`)

mfa = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
fit(mfa, data; B1 = b1map)                       # B1-corrected T1 mapping

t2 = R2DecayOffset(TE = 0.01:0.01:0.32)
fit(t2, data; C = 40.0)                          # a known offset instead of estimating it
```

Three keywords are options rather than parameters, so no model has a parameter with these
names:

- `mask`: which voxels of a dataset to fit (true = fit);
- `precision`: `Float64` (the default) or `Float32`;
- `solver`: the solver's settings, e.g. `solver = LM(maxiter = 50)`. `LM` takes `maxiter`
  (100), the tolerances `xtol`, `ftol`, `gtol` (defaults set by the precision) and the starting
  `damping` (1e-3). Most fits need none of them.

Mistakes raise an error before any fitting starts: a name that is neither a parameter nor an
option, freeing a parameter tied by default without giving it a start, a missing acquisition
field, acquisition vectors of different lengths, or maps whose size doesn't match the data. A NaN or Inf in a voxel's data, maps or starts
marks that voxel invalid (status `InvalidData`) instead of producing a meaningless fit.

## Outputs

`fit` (and `fit!`) return a `FitResult`, the same type for one voxel and for a dataset. For one
voxel the fields are numbers; for a dataset, maps with the spatial size of the data:

- `params`: the free parameters by name (`res.params.T1`).
- `sse`: sum of squared residuals.
- `iterations`: solver iterations.
- `status`: a `FitStatus`: `NotFitted` (outside the mask), `ConvergedX` / `ConvergedF` /
  `ConvergedG` (step / cost / gradient), `MaxIterations`, `Failed`, `InvalidData`. A dataset's
  status map takes 1 byte per voxel; `UInt8.(res.status)` gives the codes 0–6, e.g. for
  writing a file. `isconverged(res)` is true/false, or a map of them.
- `model` and `settings`: the model as fitted and the keywords of the call, so
  `fit(res.model, data; res.settings...)` refits with the same choices.

`refit(res, data; changes...)` fits again, starting each free parameter from the previous
estimates, with the same settings except the changes. For example, fit with kmf fixed, then
refine with kmf free, or fix kmf at the median of a first fit:

```julia
res  = fit(m, data; mask)
res2 = refit(res, data; kmf = Free())              # PSR, R1f, ... start from res's maps
res3 = refit(res, data; kmf = 14.0)
```

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
so these models first try 30 values of T1 per voxel and start from the best one (`BestOf`):
in tests, at most 2 of 200 voxels failed for T1 from 0.05 to 4 s, against all 200 from a single
start. It costs about 1 µs per voxel; giving T1 (or R1) a start, `T1 = Free(start = ...)`, skips it.

| area | model | signal | free | fixed |
|---|---|---|---|---|
| relaxometry | `T1IR(TI = ..., magnitude = true)` | inversion recovery, long TR: \|M0·[Sf·e^(−TI/T1) + 1 − e^(−TI/T1)]\|; Sf = inversion efficiency (−1 = perfect), as in T1IRTD | M0, Sf, T1 | |
| | `R1IR(TI = ..., magnitude = true)` | the same with R1 = 1/T1 | M0, Sf, R1 | |
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
    S0   = Free(start = :scale, lower = 0),
    T2   = Free(start = 0.05, lower = 1e-6),
    beta = Free(start = 0.9, lower = 0.1, upper = 1),  # stretching exponent (1 = mono-exponential)
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
- **Starting values** are declared in `parameters(m)` with each `Free` role (see Free and fixed
  parameters): a number, `:scale`, `:data` or `BestOf(values)`. If fits can end in a wrong
  local minimum from one start, give the parameter several values to try: T1IR's is
  `T1 = Free(start = BestOf(_t1grid(m.TI)), lower = _TMIN)`, 30 values log-spaced from 1/5 of the
  shortest TI to 5× the longest.
- **`datastart`** (optional): starting values computed from each voxel's data `y`, such as a
  linearized fit, for the parameters declared `:data`. `p` holds the voxel's fixed values (e.g.
  `p.B1`). T2Decay's is a log-linear fit, with mean(TE) as the fallback when the data do not
  decay: `datastart(m::T2Decay, y, p) = (T2 = isnan(R2) ? mean(m.TE) : 1 / R2,)` with
  `R2 = _loglinear_rate(m.TE, y)`. Return a finite value for each `:data` parameter: a NaN makes
  that voxel's fit fail. Another start given in the call replaces it.
- **`relabel`** (optional): if the parameters can describe the same signal in more than one way
  by swapping labels, choose the one to report. `T2BiexpDecay`'s two components can swap, so
  its `relabel` puts the shorter T2 first. Called once per voxel after the fit; it must not
  change the signal. The default changes nothing.

## Simulation

Simulated data, phantoms and published tissue values, for Monte Carlo studies, testing a
protocol, or checking a model before scanning:

```julia
m = QMTSIR(; SIR_4POINT...)                   # the 4-point scheme of Dortch et al., MRM 2018
t = truth(m, :SIR_lesion)                     # Table 1 values (PSR, R1f, Sf, M0f = 1, kmf = 20,
                                              # Sm), and the model's defaults for the rest
Y = simulate(m, t; snr = 200, n = 10_000)     # Rician noise, SNR at thermal equilibrium
res = fit(m, Y)                               # kmf fixed at 12.5: PSR biased below 5%
compare(res, t)                               # bias, SD, RMSE, % converged

m = T2Decay(TE = 0.01:0.01:0.32)
labels = labelshapes((64, 64))                # synthetic shapes: labels 0 (background) to 3
maps = phantom(labels, [1 => truth(m, :GM; S0 = 1000.0), 2 => truth(m, :WM; S0 = 1000.0),
                        3 => truth(m, :blood; S0 = 1000.0)])
res = fit(m, simulate(m, maps; snr = 100); mask = labels .> 0)
compare(res, maps; labels)                    # per tissue
```

- `simulate(m, truth; snr or σ, noise, n, rng)` and `signal(m, truth)` (one voxel, no noise):
  `truth` gives a value for every parameter, a number, a map or a tie (`R1m = :R1f`). Nothing
  is filled in silently, so the script shows everything the data were made with: `truth(m,
  tissue)` gives every value for a tissue, `nominal(m; changes...)` every value from the model
  (its starts, 1 for the scale, its fixed values), and `resolve(m, θ)` checks a set and turns its
  ties into numbers (for calling `signal!` yourself). `snr` is
  relative to the model's scale parameter (M0, S0, M0f: σ = scale/snr, "SNR at thermal
  equilibrium"), so it does not depend on the protocol; `σ` sets the noise directly, in signal
  units. Rician noise by default; Gaussian for models fitted to signed data
  (`magnitude = false`). Pass `rng = MersenneTwister(seed)` for repeatable data.
- `TISSUES`, `tissue(name)`: T1 and T2 at 3 T from Stanisz et al., MRM 2005 (in vitro, 37 °C:
  `:WM`, `:GM`, `:blood`, `:liver`, `:muscle`, ...); `:CSF` (T1 = 4.124 s, the mean of three
  in vivo inversion recovery studies reviewed by Bojorquez et al., MRI 2017; T2 = 1.75 s, a
  placeholder); and the SIR tissue classes of Dortch et al., MRM 2018, Table 1 (`:SIR_WM_I`,
  `:SIR_WM_II`, `:SIR_GM`, `:SIR_lesion`). Your own values: `truth(m, (T1 = 4.3, T2 = 2.0))`,
  or one value changed: `truth(m, merge(tissue(:CSF), (T2 = 2.0,)))`.
- `truth(m, tissue; values...)`: one model's parameters for a tissue, converting T1/T2 to rates
  where needed; the scale parameter is 1 unless given, and a parameter the model fixes takes
  its default (`defaults(m)`), shown in the result. Free values that are not tissue properties
  (an inversion efficiency `Sf`) must be given.
- `phantom(labels, tissues)`: parameter maps from a label image (NaN in the background);
  `labelshapes(dims)`: a 2-D or 3-D image of nested ellipses and a small disc.
- `compare(res, truth; labels, mask)`: per parameter, overall and per label: bias, relative
  bias, SD, RMSE, relative RMSE, median relative error and % converged. `c.rows` is a table.
- `SIR_5POINT`, `SIR_4POINT`: the optimized schemes of Dortch et al., MRM 2018, Table 2;
  `SIR_16POINT`: the earlier 16-point scheme they were compared with (15 TIs log-spaced from
  10 ms to 2 s, rounded to the ms, then 10 s; TD = 2.5 s).

## How it works

```
parameters(m) + keywords          ──setup──▶  FitSpec   free names, bounds, starts, fixed values
FitSpec + model + buffers         ──init───▶  FitCache  made once (a copy per chunk of voxels)
one voxel's data                  ──fit!───▶  FitResult numbers
a dataset                         ──fit────▶  FitResult maps; threads over chunks of voxels
```

Inside each LM step, the free values θ become dual numbers and are merged with the fixed values
into the full named tuple `p`. Then `signal!` fills a buffer of duals (values plus derivatives),
and JᵀJ, Jᵀr and the SSE are summed from it. The Jacobian matrix itself is never stored.
Before the first step, each voxel's start is set from the roles: numbers and maps as given; the
model's own estimates from the data for parameters declared `:data`; then a `BestOf` parameter at
the value that fits that voxel best, with the `:scale` parameter at its least-squares value for
each (or, without `BestOf`, for the start).

The code is in layers; each only uses the ones before it:

```
src/
  EQUIP.jl                the module: dependencies, exports, includes
  models/                 interface.jl (MRIModel, Free, Fixed, signal!, ...), builtin.jl (includes)
    relaxometry/          t1_ir.jl, t1_sr.jl, t1_mfa.jl, t2_decay.jl, t2_biexp_decay.jl
    diffusion/            adc_decay.jl, adc_kurtosis.jl
    qmt/                  qmt_sir.jl
  fit/                    levmar.jl (solver), setup.jl (roles -> FitSpec), result.jl
                          (FitResult), starts.jl (each voxel's starting point), cache.jl
                          (init), voxel.jl (fit!, and fit for one voxel), dataset.jl (fit for
                          a dataset, threaded; refit)
  sim/                    simulate.jl, tissues.jl, phantom.jl, compare.jl
```

`docs/design.md` records the design decisions.

## Roadmap

Deferred on purpose. Earlier implementations of some of these exist on the `api-v0.2` and
`main` branches.

- **sim/** (started): phantoms with smooth variation, partial volume, a B1 field and lesions;
  brain-like phantoms from segmentations (BrainWeb, KomaMRI's); a measured CSF T2 and in vivo tissue values
- **bench/**: BenchmarkTools speed suite (own Project.toml) and accuracy studies on phantoms
  (starting values, bounds, accuracy vs SNR), with recorded results
- **analysis/**: fit statistics on `FitResult` under StatsAPI's names (`stderror`, `vcov`,
  `rss`, `dof`, the residual SD per voxel, `aic` / `bic` for comparing models such as mono vs
  bi-exponential, and a flag for parameters that ended at a bound), CRLB for unbiased and biased estimators (Dortch et al. 2018,
  Eqs. 3–6), Monte Carlo with Gaussian/Rician noise, sensitivity analysis (including model
  assumptions, e.g. how much Mz left after the SIR readout, `Mzf_td0`, biases PSR and kmf)
- **design/**: protocol objectives (Eq. 5 minimax, Eq. 6 efficiency) and protocol optimization
  (best TI/TD), with an Optim.jl package extension
- `BestOf` on several parameters at once, trying every combination (with precomputed
  candidate signals when they are the same in every voxel), and `BestOf(values; fits = k)`:
  full fits from the k best local minima of the values, keeping the lowest SSE
- Outlier detection after a dataset fit (status, a parameter at a bound, deviation from
  neighbouring voxels), then a refit with the neighbours' values as a starting candidate or
  replacement by a neighbourhood average, marked in the status map
- MT ratio, MTsat (Helms) and B1 maps (DAM, AFI) as models whose data start is the exact
  closed-form answer, with B1 correction through a fixed B1 map; derived outputs (e.g. MTsat's
  empirical B1 correction, BPF from PSR); later B0 maps, with complex data and phase unwrapping
  from an existing package
- More qMT models, from approximate to exact: Henkelman's continuous-wave steady state, pulsed
  MT with SPGR (Sled & Pike, Ramani, Yarnykh, single-point MPF), MAMT (pulse pieces of constant
  amplitude, propagated), and the full two-pool equations through the sequence without
  approximations, where speed matters most; B1, B0 and R1 maps as fixed parameters
- Diffusion microstructure: SMT (spherical mean technique) and NODDI
- Custom models in one step: `@mrimodel` in front of an ordinary signal function of one
  observation (its keywords become the parameters and their default roles), a template
  generator for models that will grow, `fit(m, y, p0)` and `coef` for LsqFit users, and a
  "Coming from LsqFit, lsqcurvefit or qMRLab" page; a `SimVary`-style sensitivity sweep
- **io/**: read data as easily as possible (NIfTI, BIDS JSON sidecars, MAT, .npy/.npz, DICOM,
  PAR/REC and XML/REC, ISMRMRD images), one package extension per format. Acquisition values
  missing from headers (e.g. TI/TD for SIR) come from the call or a protocol file
  (TOML/JSON/CSV), converted to seconds/degrees. Write maps back in the input's format and
  geometry, with BIDS-derivative names and a JSON record of the fit settings
- Command line, Python and MATLAB front ends, built on io/, each call compiling nothing: a
  precompile workload covering every front-end call (checked by a test), `equip --init` to
  prepare a case outside it once (compiled code saved on disk for later calls; prototype:
  3.2–3.8 s → 0.06 s), juliacall for Python, a Julia server for MATLAB
- Solver backends through package extensions, e.g. `solver = LsqFitLM()` or a SciML solver:
  for checking EQUIP against a reference, models that cannot take dual numbers (finite
  differences and user Jacobians come with the backends), and models with many parameters;
  EQUIP's own static-array solver stays the default. Or a `levmar` method for ordinary arrays,
  for models with more than ~30–40 free parameters
- GPU, observation weights, other data layouts
- Faster start for the command line: a Julia server for many calls in a row, or a system
  image or standalone app (PackageCompiler.jl)
- More models: gradient-echo decay with the through-slice sinc term for macroscopic field
  gradients (a separate model; the frequency spread from a B0 field map or fitted); diffusion
  tensor and kurtosis, IVIM, multi-exponential T2 (myelin water), DCE, more qMT methods
- More data-driven starting values (with `datastart`): DESPOT1 for T1MFA (with the voxel's B1),
  R2Decay and ADCDecay, ADCKurtosis, saturation recovery, the offset and biexponential models
- Variable projection: the parameters the signal is linear in (M0/S0, the biexponential
  amplitudes, an offset) solved inside every LM step, chosen per parameter with a setting of
  its own, `fshort = Free(lower = 0, upper = 1, varpro = true)` (design in docs/design.md).
  Prototype measured (branch
  `varpro-proto`): biexponential 19–26% faster with more fits reaching the minimum; 0–22%
  faster with one linear parameter; no help with multiple minima, so `BestOf` stays
- Positive parameters fitted in log coordinates, with a per-parameter relative stopping test
  ("stop when every parameter changes by less than 0.01%"). Prototype measured (branch
  `varpro-proto`): with VarPro, the biexponential 40–59% faster than today and more fits
  reaching the minimum; no change for models whose starts already do the job
