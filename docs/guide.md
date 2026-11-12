# EQUIP.jl: the guide

Everything EQUIP does, part by part, with the code to run. The [README](../README.md) is the
short version; [design.md](design.md) has the reasons behind each choice and the measurements;
[status.md](status.md) where the work stands. The examples in [examples/](../examples/) run as
they are.

- [Two inputs](#two-inputs) · [Quick start](#quick-start) · [Free and fixed parameters](#free-and-fixed-parameters) · [Outputs](#outputs)
- [Built-in models](#built-in-models) · [Writing a model](#writing-a-model) · [Simulation](#simulation)
- [Accuracy before scanning: crlb and montecarlo](#accuracy-before-scanning-crlb-and-montecarlo) · [Protocol design](#protocol-design)
- [Data files](#data-files) · [Command line](#command-line) · [How it works](#how-it-works)

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

Every model declares each parameter's default **role** with `parameters`, for its type: the
roles are the same for every acquisition, so `parameters(QMTSIR)` needs no protocol, and
`parameters(m)` for a model asks its type. For QMTSIR:

```julia
parameters(::Type{<:QMTSIR}) = (
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
  - `:data`: the model's own estimate from the voxel's data (`datastart`), e.g. a closed-form
    estimate, `T2 = Free(start = :data, lower = ...)`;
  - `:loglinear`: the weighted linear fit of ln y with ln S, computed by EQUIP from the
    model's own signal in two passes, OLS then WLS weighted by OLS's predicted signal² (DIPY's
    WLS; the `:scale` parameter as the intercept), for models whose ln S is linear in the
    parameters: diffusivities and rates, and time constants through their reciprocal (T2 = 0
    is no signal, so T2 starts very long and the fit runs on 1/T2). DTI's tensor,
    `Dxx = Free(start = :loglinear)`, DKI written in V = MD²·W, T2Decay's T2. No code in the
    model; when nothing in the start varies by voxel, the first pass is made once per fit. For
    a model whose ln S is not linear in them (ADCKurtosis with ADC and K both at `:loglinear`)
    the passes are two Gauss–Newton steps on ln S, a start the fit refines, and the fit warns
    once (it compares ∂ln S/∂θ at two points);
  - `BestOf(values)`: try each value (with the `:scale` parameter solved for each) and start
    from the one that fits best, e.g. the inversion-recovery models'
    `T1 = Free(start = m -> BestOf(_t1grid(m.TI)), lower = ...)`, 30 values spanning the TIs.
    One parameter per fit for now. `BestOf` wraps the values because a plain vector start is a
    map;
  - a function of the model, `m -> ...`, for a start that depends on the acquisition, called
    once per fit with the model: ADCDecay's `ADC = Free(start = m -> 1 / mean(m.b))`, the
    `BestOf` above. Asked of a model, `parameters` shows the start it gives (`parameters(T1IR(TI
    = ti))` shows the 30 values); asked of the type, the function.
- `Fixed(value)`: held fixed at a number, a per-voxel map (e.g. B1), or a tie to other
  parameters: `Fixed(:R1f)` (always equal to R1f) or a function such as
  `Fixed(p -> 0.8 * p.R1f)`. A tie can't refer to another tie. `defaults(m)` gives the fixed
  values as a named tuple, e.g. `(kmf = 12.5, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)`.

Any fit can change a role with a keyword named after the parameter, in `fit` and `init`. A
value fixes the parameter; `Free(...)` estimates it, and whatever it leaves out (start, bounds)
keeps the model's default. To remove a bound the model sets, give it as infinite:
`Free(lower = -Inf)` or `Free(upper = Inf)`. (In `refit`, what a `Free(...)` leaves out keeps
the earlier fit's setting, and the start is the earlier estimate unless a new one is given.)

```julia
fit(m, data; kmf = 15.0)                         # change a fixed value
fit(m, data; Sm = sm_map)                        # a per-voxel map
fit(m, data; PSR = 0.1)                          # fix a parameter that is free by default
fit(m, data; kmf = Free())                       # estimate kmf (starts at 12.5, no bounds)
fit(m, data; kmf = Free(lower = 1, upper = 50))  # ... within bounds
fit(m, data; R1f = Free(start = r1_map))         # another start: a number or a map
fit(m, data; R1f = Free(start = BestOf([0.5, 1.0, 2.0])))   # try three, start from the best
fit(m, data; Sf = Free(lower = -1.5))            # another bound; start and upper bound as before
fit(m, data; PSR = Free(lower = -Inf))           # remove the model's lower bound (PSR ≥ 0)
fit(m, data; kmf = 15.0, Sm = sm_map, mask)      # several at once (`mask` means `mask = mask`)

roles = parameters(m; kmf = 15.0, Sm = 0.9)      # every role, two changed: a set to keep
fit(m, data, roles)                              # the same as fit(m, data; kmf = 15.0, Sm = 0.9)

mfa = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
fit(mfa, data; B1 = b1map)                       # B1-corrected T1 mapping

t2 = R2DecayOffset(TE = 0.01:0.01:0.32)
fit(t2, data; C = 40.0)                          # a known offset instead of estimating it
```

Four keywords are options rather than parameters, so no model has a parameter with these
names:

- `mask`: which voxels of a dataset to fit (true = fit);
- `precision`: `Float64` (the default) or `Float32`: half the memory, and on datasets faster,
  1.0–1.5× by model (measured, 4 threads; e.g. 4-point SIR 1.5×, VFA 1.45×, T2 1.4×,
  biexponential 1.1–1.25×). Float32 dataset fits flush subnormal numbers (below 1.2e-38) to zero,
  which the CPU handles slowly and which long SIR delays produce; without that some SIR fits were
  slower in Float32. The data can be any number type: each voxel is converted to the fit's
  precision as it is fitted;
- `solver`: the solver's settings, e.g. `solver = LM(maxiter = 50)`. `LM` takes `maxiter`
  (100), the tolerances `xtol`, `ftol`, `gtol` (defaults set by the precision) and the starting
  `damping` (1e-3). Most fits need none of them;
- `weights`: one weight per observation, the same in every voxel: the fit minimizes
  ½Σwᵢ(gᵢ − yᵢ)² (a weight 0 leaves an observation out; only the ratios matter for the
  estimates). With noise that differs between observations, wᵢ = 1/σᵢ² is the efficient choice:
  `weights = nex` for different numbers of averages (σ/√NEX), `weights = 1 ./ σ.^2` for σ per
  observation. On T2 decay with σ rising from 5 to 40 over 32 echoes, the weighted fit's SD of
  T2 is 1.5 times smaller than the unweighted fit's, at the bound (JᵀΣ⁻¹J)⁻¹.
- `likelihood`: the noise model the fit assumes. `:gaussian` (the default) is least squares, as
  everything above. `:rician` is for magnitude images at low SNR, where the noise floor (about
  1.25σ where the signal has decayed away) biases least squares. What decides it is each image's
  own SNR, A/σ (its signal over the noise SD), not the unweighted SNR S0/σ: the floor raises one
  image's mean by +1% at A/σ = 7, +2% at 5, +6% at 3, +14% at 2 and +55% at 1. So the Rician fit
  matters when the images that carry a parameter (the late echoes for T2, the high b-values for
  MD, the TIs near the inversion's null for T1) fall below A/σ of about 3; where they stay above
  5–7, least squares does as well. Measured (docs/design.md, the tier 2 plan, steps 0, 2 and 4),
  at S0/σ = 10: a T2 in white matter +41% by least squares (32 echoes to 320 ms, 4.6 T2s: the
  median echo's A/σ 0.9, three quarters below 3) and +0.8% fitted as Rician; DTI's MD −9% and
  +0.4% (b 1000 and 2500: median A/σ 3.6); DKI's MD in white matter hardly biased by the floor
  either way (b ≤ 2000: 2% of images below 3). It needs `σ`, the
  noise SD of each channel (as `simulate`'s): a number, or a map with one value per voxel (a σ map
  from another tool, MRtrix's `dwidenoise -noise`, read with `readmap`). `method` chooses how the
  noise model is used:
  - `:ml` (the default): the full Rician model, maximum likelihood (Sijbers et al. 1998);
  - `:mean`: least squares on the Rician mean E|y| = `ricianmean(S, σ)`, the data's average at
    signal S, a cheaper approximation;
  - `:wmean`: the same, each observation weighted by 1/`ricianvar(S, σ)`.

  ```julia
  res = fit(T2Decay(TE = te), data; mask, likelihood = :rician, σ = 12.0)             # :ml
  res = fit(DTI(; acq...), dwi; mask, likelihood = :rician, σ = σmap, method = :mean)
  ```

  Each voxel is first fitted by least squares, from the model's own starts, then with the Rician
  model from that result: the result's maps, derived quantities and `refit` are as usual, its
  `iterations` both fits', its `sse` Σ(y − E|y|)², and `fitted(res)` the data's mean under the
  fit, E|y|. Its standard errors use the known σ (the inverse Rician information for `:ml`, the
  sandwich with the Rician variance for `:mean`). Voxels with negative data (not magnitudes) or
  no σ (NaN or 0 in the map) are not fitted (`InvalidData`). `crlb` and `montecarlo` take the
  same keywords and predict or measure these fits (under "Accuracy before scanning"). Not yet
  with it: `weights`, numbers of averages, the log-likelihood and information criteria of the
  fit, a protocol design for it. Where the data have no information left (CSF at high b in DKI), no noise model
  recovers it: mask such voxels. Phase-corrected real-valued data have Gaussian noise and no
  floor: fit them as they are.

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

### Fit statistics

How uncertain the fitted values are, and how good the fit, from the fit itself, under
StatsAPI's names (GLM's and StatsBase's). Numbers for one voxel; for a dataset, maps shaped
like `res.params`, each voxel its own fit:

```julia
res = fit(T2Decay(TE = te), data; mask)
stderror(res).T2            # a map of T2's standard error: σ̂²(JᵀJ)⁻¹ at each voxel's answer
residualsd(res)             # σ̂ = √(rss/(N − P)), the noise estimated from the residuals
stderror(res; σ = 12.0)     # with the noise known (a number or a map) instead
vcov(res), cor(res)         # the covariance and correlation matrices, one per voxel
onbound(res).T2             # where T2 ended on a bound (its SE then NaN, the others with it held)
bic(fit(T2BiexpDecay(TE = te), y)) < bic(fit(T2Decay(TE = te), y))   # the second pool earns its keep
```

- `nobs`, `dof` (the free parameters and σ, as GLM counts), `dof_residual` (N − P), `rss`,
  `loglikelihood`, `aic`, `aicc`, `bic` (Gaussian, σ estimated).
- A weighted fit (`weights`) assumes variances σ²/wᵢ: `rss` is Σwr², `residualsd` the SD of an
  observation of weight 1 (one average's with `weights = nex`; about 1 with `weights = 1 ./ σ.^2`),
  the covariance σ̂²(JᵀWJ)⁻¹, `stderror(res; σ)` with that unit-weight σ, the log-likelihood with
  its ½Σ log wᵢ, and N the observations with a positive weight.
- With as many observations as free parameters (the 4-point SIR scheme) the residuals say
  nothing about the noise: `residualsd` and the information criteria are NaN; for standard
  errors, give `σ`. AICc, the one for few observations per parameter, is NaN also when
  N ≤ P + 2 (its small-sample correction is undefined there). With σ given, the standard errors
  are `crlb`'s at the fit's values (tested), so the two can be compared.
- Checked on 2000 noisy copies (T2 decay): the median standard error matches the spread of
  the estimates within 1%, and σ̂ the true σ. All of it assumes Gaussian noise: for magnitude
  data at low SNR (Rician) σ̂ and the standard errors are biased; fit such data with
  `likelihood = :rician`, whose standard errors use its σ (`residualsd`, `loglikelihood`, `aic`,
  `aicc` and `bic` are not yet defined for it).

## Built-in models

Names put the quantity first, then the method, so typing `T1` + Tab lists every T1 model.
Rate (R1, R2) and time-constant (T1, T2) versions fit the same data; fit the one you will
report. Magnitude inversion recovery (`T1IR`, `T1IRTD` and their R1 versions) has local minima,
so these models first try 30 values of T1 per voxel and start from the best one (`BestOf`):
in tests, at most 2 of 200 voxels failed for T1 from 0.05 to 4 s, against all 200 from a single
start. It costs about 1 µs per voxel; giving T1 (or R1) a start, `T1 = Free(start = ...)`, skips it.

| area | model | signal | free | fixed |
|---|---|---|---|---|
| relaxometry | `T1IR(TI = ..., magnitude = true)` | inversion recovery, long TR: \|M0·[Sf·e^(−TI/T1) + 1 − e^(−TI/T1)]\|; Sf = inversion efficiency (−1 = perfect), as in T1IRTD; T1 at most 5 s (above CSF's 4.1 s), as in every T1 model, so a recovery too slow for the TIs ends on the bound, flagged by `onbound` | M0, Sf, T1 | |
| | `R1IR(TI = ..., magnitude = true)` | the same with R1 = 1/T1; R1 at least 0.2 s⁻¹, as in every R1 model (T1 at most 5 s) | M0, Sf, R1 | |
| | `T1IRTD(TI = ..., TD = ..., magnitude = true)` | inversion recovery with the delay TD modelled (Li et al., MRM 2010, Eq. 1): M0·[Sf·Mz(TD)·e^(−TI/T1) + 1 − e^(−TI/T1)], Mz(TD) = 1 − (1 − Mz_td0)·e^(−TD/T1); TD one number or one per TI | M0, Sf, T1 | Mz_td0 = 0 |
| | `R1IRTD(TI = ..., TD = ..., magnitude = true)` | the same with R1 = 1/T1 | M0, Sf, R1 | Mz_td0 = 0 |
| | `T1SR(TD = ...)` | saturation recovery: M0·(1 − exp(−TD/T1)) | M0, T1 | |
| | `R1SR(TD = ...)` | the same with R1 = 1/T1 | M0, R1 | |
| | `T1MFA(FA = ..., TR = ...)` | multiple flip angles (spoiled gradient echo), flip angle B1·FA (degrees); TR one number or one per FA | M0, T1 | B1 = 1 |
| | `R1MFA(FA = ..., TR = ...)` | the same with R1 = 1/T1 | M0, R1 | B1 = 1 |
| | `T2Decay(TE = ...)` | S0·exp(−TE/T2); also T2* (simple mono-exponential case); T2 at most 2.5 s (CSF's is about 2 s), so data that do not decay end on the bound, flagged by `onbound` | S0, T2 | |
| | `R2Decay(TE = ...)` | S0·exp(−R2·TE); also R2*; the more robust form; R2 at least 0.4 s⁻¹ (T2 at most 2.5 s, as T2Decay) | S0, R2 | |
| | `T2DecayOffset(TE = ...)` | S0·exp(−TE/T2) + C (C: a common approximation for the noise floor); T2 at most 2.5 s | S0, T2, C | |
| | `R2DecayOffset(TE = ...)` | S0·exp(−R2·TE) + C; R2 at least 0.4 s⁻¹ | S0, R2, C | |
| | `T2BiexpDecay(TE = ...)` | two T2 components (e.g. myelin water): S0·[fshort·exp(−TE/T2short) + (1 − fshort)·exp(−TE/T2long)]; the shorter T2 reported as T2short ; each T2 at most 2.5 s | S0, fshort, T2short, T2long | |
| | `R2BiexpDecay(TE = ...)` | the same with R2short = 1/T2short, R2long = 1/T2long; each at least 0.4 s⁻¹ | S0, fshort, R2short, R2long | |
| diffusion | `ADCDecay(b = ...)` | S0·exp(−b·ADC); b in s/mm², ADC in mm²/s, at most 4e-3 (above free water's 3e-3, so data whose decay is lost in the noise end on the bound, flagged by `onbound`; `ADC = Free(upper = Inf)` removes it, as for b in other units) | S0, ADC | |
| | `ADCKurtosis(b = ...)` | kurtosis, one direction or direction-averaged (Jensen et al., MRM 2005): S0·exp(−b·ADC + b²·ADC²·K/6); K in [0, 3]; ADC at most 4e-3 mm²/s, as ADCDecay; check b_max·ADC·K < 3 | S0, ADC, K | |
| | `DTI(b = ..., gx = ..., gy = ..., gz = ...)` | the diffusion tensor (Basser et al., Biophys J 1994): one image per gradient direction, S0·exp(−b·gᵀDg), the six elements fitted, the diagonal ones from 0 (a diffusivity is not negative) to 4e-3 mm²/s (as ADCDecay); each voxel starts from the weighted log-linear fit (`:loglinear`); the acquisition from a .bval and a .bvec by `diffusionencoding` (`DTI(; acq...)`) | S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz (mm²/s) | |
| | `DKI(b = ..., gx = ..., gy = ..., gz = ...)` | diffusion kurtosis (Jensen et al., MRM 2005) in DIPY's parameterization, V = MD²·W: S0·exp(−b·gᵀDg + b²·ΣgᵢgⱼgₖgₗVᵢⱼₖₗ/6), every element from the two-pass WLS (`:loglinear`); D's diagonal from 0 to 4e-3 mm²/s, as DTI; derived: DTI's quantities and MK, AK, RK as DIPY's fit methods give them by default (Tabesh et al., MRM 2011, analytic), MKT (Hansen et al., MRM 2013), clipped to [−3/7, 10] | S0, D's six, V's fifteen | |
| | `SMT(b = ...)` | the spherical mean technique, two compartments (Kaden et al., NeuroImage 2016; as in Devan et al., Magn Reson Imaging 2020): one value per shell, the mean over its directions (`sphericalmean(data, b)`); sticks and a zeppelin with the tortuosity limit, D⊥,ex = (1 − v_ax)·D_ax; protocols `SMT_PROTOCOLS` (Devan Table 2), tissues `SMT_NWM`, `SMT_NAWM`, `SMT_T2lesion`, `SMT_cBH` (Table 1) | S0, vax, Dax (mm²/s) | |
| | `SMTFreeWater(b = ...)` | `SMT` with free water, isotropic (Devan 2020's main confound): S0·[(1 − fiso)·SMT + fiso·exp(−b·Diso)] | S0, vax, Dax, fiso | Diso = 3e-3 mm²/s |
| | `SMTT2(b = ..., TE = ...)` | `SMT` with a T2 per compartment, shells at several echo times (Veraart et al., NeuroImage 2018; McKinnon & Jensen, MRM 2019): vax the volume fraction, not SMT's T2-weighted one | S0, vax, Dax, T2ax, T2ex | |
| qMT | `QMTSIR(TI = ..., TD = ..., magnitude = true)` | selective inversion recovery, two-pool qMT (Li et al., MRM 2010; Dortch et al., MRM 2018); TD one number or one per TI | PSR, R1f, Sf, M0f | kmf = 12.5, R1m = R1f, Sm = 0.83, Mzf_td0 = Mzm_td0 = 0 (both pools saturated at the start of TD) |

## Writing a model

### In one step: `@mrimodel`

Put `@mrimodel` in front of an ordinary Julia function of one observation:

```julia
using EQUIP

"""Stretched-exponential T2 decay."""
@mrimodel StretchedDecay(TE; S0   = Free(start = :scale, lower = 0),
                             T2   = Free(start = 0.05, lower = 1e-6),
                             beta = Free(start = 0.9, lower = 0.1, upper = 1)) =
    S0 * exp(-(TE / T2)^beta)

m = StretchedDecay(TE = 0.01:0.01:0.32)
res = fit(m, data; mask)                 # crlb, montecarlo, simulate, design: as any model
```

- **The name** becomes the model type; **the positional arguments** its acquisition, given by
  name when the model is made, one value per observation or one number for every observation
  (in the body, the observation's value); positional arguments **with a default** are options
  (`magnitude = true`).
- **The keywords** are the parameters, in order, their defaults the default roles: `Free(...)`
  (its start may use the acquisition: `T2 = Free(start = mean(TE))`, `TE` the whole vector); a
  number or `Fixed(...)` for fixed; another parameter's name for a tie (`R1m = R1f`), or a
  formula of others (`R1m = 0.8 * R1f`); no default for a free parameter whose start is given
  in the call (`fit(m, data; T2 = Free(start = 0.05))`).
- **The body**, any number of lines (`@mrimodel function Name(...) ... end`), is the signal of
  one observation. The macro writes the struct, its keyword constructor, `parameters` and a
  `signal!` that calls the body for each observation; `@macroexpand` shows the code. A fit
  allocates nothing and runs as fast as the hand-written models: `QMTSIR` written this way, per
  observation, gives the same fits bit for bit, at the same speed (the compiler moves the work
  that does not depend on the observation out of the loop).
- A docstring before `@mrimodel` documents the model; hooks are added afterwards as ordinary
  methods (`EQUIP.datastart(m::StretchedDecay, y, p) = ...`).
- When the model outgrows the one-line form, `newmodel(StretchedDecay)` writes it out as a file
  of plain code (the hand-written form below, the body as the signal of one observation, the
  roles as written, the docstring kept), to edit and `include`: the same fits.
  `newmodel(:Name; acquisition = [:TE], parameters = [:S0, :T2])` writes a blank one, with
  TODOs.

### Compartments

The diffusion compartments EQUIP's own models are made of, exported by EQUIP for writing yours
(`using EQUIP` brings them; b in s/mm², D in mm²/s). The names are Microstructure.jl's (Gong &
Yendiki, Imaging Neuroscience 2026), for the compartments of Panagiotaki et al.'s taxonomy
(NeuroImage 2012), where `iso` is the "ball":

| compartment | signal |
|---|---|
| `iso(b, D)` | exp(−b·D): isotropic Gaussian diffusion (free water, an extra-cellular space, an ADC), the same in every view; with D = 0, immobile water (a "dot") |
| `stick(b, D, powder)`, `stick(b, D, ξ)` | a stick's spherical mean (powder average), √π·erf(√(bD))/(2√(bD)); along one direction at cosine ξ to the fiber, exp(−b·D·ξ²) |
| `zeppelin(b, D∥, D⊥, powder)`, `zeppelin(b, D∥, D⊥, ξ)` | an axially symmetric tensor: its spherical mean, exp(−b·D⊥)·√π·erf(√(b(D∥ − D⊥)))/(2√(b(D∥ − D⊥))); along one direction at cosine ξ to the fiber, exp(−b·D⊥)·exp(−b·(D∥ − D⊥)·ξ²) |

A stick or a zeppelin takes its view explicitly, `powder` (its spherical mean) or a cosine ξ
(along one gradient direction); without one it is an error that says which to write. A zeppelin
with D⊥ > D∥ (oblate) is allowed in both views (its spherical mean then takes erfi); to keep
D⊥ ≤ D∥ in a model that fits D⊥, give it a fraction, D⊥ = fperp·D∥ with bounds 0 and 1. The
cosine comes from two helpers, not compartments themselves:

| helper | gives |
|---|---|
| `fiberdirection(θ, φ)` | a fiber's unit direction n = (sinθ·cosφ, sinθ·sinφ, cosθ), a tuple; once per voxel, before the loop over images |
| `directionset(N)` | N gradient directions spread evenly over the sphere (electrostatic repulsion), to simulate a scheme |
| `g ⋅ n` | ξ, the cosine between one image's gradient direction g = (gx[j], gy[j], gz[j]) and the fiber n: Julia's own dot product (LinearAlgebra's `⋅`, exported by EQUIP too), on two tuples, nothing allocated |

The compartments are diffusion only. Relaxation is written in the model as its equation, as the
paper writes it: `exp(-TE / T2)`, `1 - 2exp(-TI / T1)`, and a compartment's T2 as a factor of it,
`vax * exp(-TE / T2ax) * stick(...)`, in a model whose acquisition has several echo times.
(Microstructure.jl's compartments carry a T2 each instead, used only with multi-TE data.)

```julia
using EQUIP

@mrimodel ThreePool(TE; S0 = Free(start = :scale, lower = 0), fmw = Free(start = 0.1, lower = 0, upper = 0.5),
                    fcsf = Free(start = 0.05, lower = 0, upper = 0.5), T2mw = Free(start = 0.015),
                    T2ie = Free(start = 0.08), T2csf = 2.0) =                # CSF's T2 known
    S0 * (fmw * exp(-TE / T2mw) + (1 - fmw - fcsf) * exp(-TE / T2ie) + fcsf * exp(-TE / T2csf))

@mrimodel IVIM(b; S0 = Free(start = :scale, lower = 0), f = Free(start = 0.1, lower = 0, upper = 0.5),
               Dstar = Free(start = 0.02), D = Free(start = 1e-3, lower = 0)) =
    S0 * (f * iso(b, Dstar) + (1 - f) * iso(b, D))

@mrimodel SMTwithT2(b, TE; S0 = Free(start = :scale, lower = 0), vax = Free(start = 0.5, lower = 0, upper = 1),
                    Dax = Free(start = 2e-3, lower = 0), T2ax = Free(start = 0.08), T2ex = Free(start = 0.05)) =
    S0 * (vax * exp(-TE / T2ax) * stick(b, Dax, powder) +
          (1 - vax) * exp(-TE / T2ex) * zeppelin(b, Dax, (1 - vax) * Dax, powder))

@mrimodel NerveZ(b, ξ; S0 = Free(start = :scale, lower = 0), vax = Free(start = 0.5, lower = 0, upper = 1),
                 Dax = Free(start = 2e-3, lower = 0)) =                        # ξ = g·z, fibers along z
    S0 * (vax * stick(b, Dax, ξ) + (1 - vax) * zeppelin(b, Dax, (1 - vax) * Dax, ξ))
```

The compartments are inlined: a model written with them is as fast as the equation written out.
More, each fitted to simulated data (inversion recovery, a T2 decay against the built-in T2Decay):
examples/compartments.jl.

A model with one image per gradient direction (DTI, a stick and a zeppelin along a fiber) takes
four acquisition fields, `b`, `gx`, `gy`, `gz`, each one value per image, and
`diffusionencoding` makes them from a .bval and a .bvec (checked: a direction per b-value, unit
vectors where b > 50):

```julia
acq = diffusionencoding("dwi.bval", "dwi.bvec")       # (b = ..., gx = ..., gy = ..., gz = ...)
m = DTI(; acq...)                                     # built in; any model with these fields
```

With fiber angles, write `signal!` by hand and compute the direction once per voxel,
`n = fiberdirection(θ, φ)`, then `ξ = (gx[j], gy[j], gz[j]) ⋅ n` for each image: an
`@mrimodel` body runs once per image, so its sines and cosines would be computed for every image,
and the fit takes about twice as long (examples/directional_models.jl times both).

### By hand

For a model that will grow, or one whose signal is more than a function of one observation, a
model is a struct, a `parameters` method and a `signal!` method:

```julia
using EQUIP
import EQUIP: parameters, signal!     # we add methods to these functions

@kwdef struct StretchedDecay{A<:AbstractVector} <: MRIModel
    TE::A                                # echo times (s), one per observation
end

parameters(::Type{<:StretchedDecay}) = (
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
- **`parameters`** is declared for the model's type, `parameters(::Type{<:StretchedDecay})`:
  `Type{<:StretchedDecay}` ("StretchedDecay or a subtype") because the models are parametric
  types (a model's type is `StretchedDecay{Vector{Float64}}`, which `Type{StretchedDecay}` would
  not match). Then `parameters(StretchedDecay)` and `nominal(StretchedDecay, ...)` need no
  acquisition. (A model may define `parameters(m::StretchedDecay)` on the instance instead;
  its type then cannot be asked.)
- **Starting values** are declared with each `Free` role (see Free and fixed parameters): a
  number, `:scale`, `:data`, `:loglinear`, `BestOf(values)`, or a function of the model for a start that
  depends on the acquisition. If fits can end in a wrong local minimum from one start, give the
  parameter several values to try: T1IR's is
  `T1 = Free(start = m -> BestOf(_t1grid(m.TI)), lower = _TMIN)`, 30 values log-spaced from 1/5
  of the shortest TI to 5× the longest.
- **`datastart`** (optional): starting values computed from each voxel's data `y`, such as a
  closed-form estimate, for the parameters declared `:data`. `p` holds the voxel's fixed values
  (e.g. `p.B1`). For example T2 from the first two echoes, with mean(TE) as the fallback when they
  do not decay: `datastart(m::MyT2Model, y, p) = (T2 = isfinite(t) && t > 0 ? t : mean(m.TE),)`
  with `t = (m.TE[2] - m.TE[1]) / log(y[1] / y[2])`. (For a log-linear fit, `:loglinear` needs no
  code.) Return a finite value for each `:data` parameter: a NaN makes that voxel's fit fail.
  Another start given in the call replaces it.
- **`relabel`** (optional): if the parameters can describe the same signal in more than one way
  by swapping labels, choose the one to report. `T2BiexpDecay`'s two components can swap, so
  its `relabel` puts the shorter T2 first. Called once per voxel after the fit; it must not
  change the signal. The default changes nothing.
- **`derived`** (optional): quantities computed from one voxel's parameters, reported beside
  them. DTI's: `derived(m::DTI, p) = (MD = ..., FA = ..., AD = ..., RD = ..., V1x = ..., V1y =
  ..., V1z = ...)`, from the six tensor elements (V1: the principal eigenvector, z ≥ 0). `p` holds every parameter, fitted and fixed; the rules are `signal!`'s (no
  allocation, generic in the number type, so their errors can come from the fit's). For a fit,
  `derived(res)` gives them, maps for a dataset (NaN where a voxel was not fitted), computed when
  asked; `writemaps` writes them beside the parameters' maps (`FA.nii.gz`, ...). The default,
  `nothing`, means none. DKI's are its tensor's and MK, AK, RK, MKT. crlb and montecarlo report
  them after the parameters (`c.sd.FA`, below); their standard-error maps are on the roadmap
  (docs/design.md, item 19).
- **`fromderived`** (optional): the reverse, for `nominal`: `fromderived(::Type{<:MyModel}, q)`
  returns the parameters that give the derived quantities in `q` (the call's keywords that are
  not parameters), so that a truth can be prescribed by them (Simulation, below). Derived
  quantities seldom fix the parameters alone, so a model says which sets it takes and what it
  assumes for the rest, and refuses anything else; `fromderived(::Type{<:MyModel})` lists the
  names it takes (what a tissue preset may give it, and a sweep may vary). The default,
  `nothing` and `()`, means none.

## Simulation

Simulated data and published tissue values, for Monte Carlo studies, testing a
protocol, or checking a model before scanning:

```julia
m = QMTSIR(; SIR_4POINT...)                   # the 4-point scheme of Dortch et al., MRM 2018
t = nominal(m, :SIR_lesion)                   # Table 1 values (PSR, R1f, Sf, M0f = 1, kmf = 20,
                                              # Sm), and the model's defaults for the rest
Y = simulate(m, t; noise = :rician, snr = 200, n = 10_000)     # Rician noise, SNR at thermal equilibrium
res = fit(m, Y)                               # kmf fixed at 12.5: PSR biased below 5%
compare(res, t)                               # bias, SD, RMSE, % converged

m = T2Decay(TE = 0.01:0.01:0.32)
maps = (S0 = fill(1000.0, 64, 64), T2 = 0.04 .+ 0.08 .* rand(64, 64))   # any maps
res = fit(m, simulate(m, maps; noise = :rician, snr = 100))
compare(res, maps)                            # labels = an integer map, for each tissue

m = ADCDecay(b = [0, 500, 1000, 1500, 2000, 2500, 3000])
Y = simulate(m, (S0 = 1.0, ADC = 8e-4); snr = 20, averages = [1, 1, 2, 2, 4, 4, 8],
             noise = :rician, averaging = :magnitude, n = 10_000)   # magnitude images averaged
```

Four functions give a model's parameters in different forms:

| Function | Gives | Example (QMTSIR) |
|---|---|---|
| `parameters(m)`, `parameters(M)`, `parameters(m; changes...)` | each parameter's role: free (start, bounds) or fixed; for a model or its type; with keywords, changed as a fit's keywords change them (pass the result whole: `fit(m, data, roles)`) | `PSR = Free(start = 0.1, lower = 0.0, upper = 1.0), …, kmf = Fixed(12.5)` |
| `defaults(m)` | the values of the fixed parameters alone | `(kmf = 12.5, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)` |
| `nominal(m; changes...)`, `nominal(m, tissue; changes...)` (a model or its type) | every value, for simulation: from the model (starts, 1 for the scale, fixed values) or from a tissue (a preset or your own named tuple) | `nominal(m)`: `(PSR = 0.1, R1f = 1.0, …, kmf = 12.5, …)`; `nominal(m, :SIR_WM_II)`: `(PSR = 0.12, …, kmf = 10.0, …)` |
| `resolve(m, p)`, `resolve(res)` | a complete set checked, ties turned into numbers; for a result, fitted and fixed values | `(…, R1f = 1.0, …, R1m = 1.0, …)` |

Each value of `nominal` comes from a keyword, else the tissue (by name, or converted between T1
and R1, T2 and R2), else the model: a fixed value, 1 for the scale parameter and, without a
tissue, a free parameter's start (a guess for fitting, not a tissue value). With a tissue, a
free parameter must come from the tissue or a keyword: `nominal(QMTSIR(...), :WM)` is an error
(WM has T1 and T2, no PSR), not the model's starts in disguise, and `nominal(T1IR(...), :GM; Sf =
-0.95)` needs Sf, an inversion efficiency, not a tissue property. Without a tissue, one that
starts from the data (`:data`, `:loglinear`, e.g. T2Decay's T2) or from `BestOf` values (the
inversion-recovery models' T1) must be given: `nominal(T2Decay(TE = te); T2 = 0.08)`. The error
names each and lists what `nominal` takes for every parameter.

A model with derived quantities can take them in place of its parameters (its `fromderived`).
DTI and DKI: an axially symmetric tensor, from MD and FA, or AD and RD, and the direction of its
axis; DKI's kurtosis from MKT alone (W isotropic) or from MK, AK and RK (W axially symmetric
about the same axis), or V's elements as parameters. `derived` gives them back:

```julia
θ = nominal(DTI; S0 = 1000, MD = 0.8e-3, FA = 0.7, direction = fiberdirection(0.3, 1.2))
θ = nominal(DKI; S0 = 1000, AD = 1.7e-3, RD = 0.35e-3, direction = (1, 0, 0), MK = 1.0, AK = 0.8, RK = 1.3)
derived(DKI(; acq...), θ)                     # (MD = 0.0008, FA = 0.762, ..., MK = 1.0, AK = 0.8, RK = 1.3, MKT = ...)
θ = nominal(DTI, :DKI_CC_splenium; S0 = 1000, direction = (1, 0, 0))   # a preset's MD and FA
```

The `DKI_` presets in `TISSUES` are regional values in healthy adults at 3 T (Lätt et al., J Magn
Reson Imaging 2013;37:610, Table 1: corpus callosum splenium, genu and body, internal capsule,
centrum semiovale, corona radiata, frontal white matter, thalamus, putamen, caudate): MD, FA,
MK and RK. DTI takes their MD and FA; DKI their MK and RK too, with AK given in the call (the
paper did not report it). With every index, AK included: `DKI_GM` and `DKI_WM`, whole gray and
white matter (Maiter et al., Sci Rep 2021;11:8857: MD, FA, MK, AK, RK from one analysis), and
`DKI_CSF`, free water as a truth (MD 3.0e-3 mm²/s, FA and every kurtosis 0; measured CSF differs,
through partial volume, the noise floor and flow). The direction is always yours. Wherever a truth goes, `simulate`,
`Scenario`, `crlb` and `montecarlo` (and `simulate_with`'s), it can be given by derived
quantities in the same way, and a sweep can then vary one (below).

Typing a model at the REPL shows the same table, with its acquisition:

```
julia> m = QMTSIR(TI = [0.01, 0.05, 0.277, 1.0], TD = 2.5)
QMTSIR with 4 observations
  TI = [0.01, 0.05, 0.277, 1.0]
  TD = 2.5, every observation
  magnitude = true
Parameters:
  name     role   start/value  bounds        nominal(m)
  PSR      free   0.1          [0.0, 1.0]    0.1
  R1f      free   1.0          [0.05, 10.0]  1.0
  Sf       free   -0.95        [-1.2, 0.0]   -0.95
  M0f      free   :scale       [0.0, Inf]    1.0
  kmf      fixed  12.5                       12.5
  R1m      fixed  :R1f (tie)                 :R1f
  Sm       fixed  0.83                       0.83
  Mzf_td0  fixed  0.0                        0.0
  Mzm_td0  fixed  0.0                        0.0
```

- `simulate(m, truth; snr or σ or Σ, noise, averages, averaging, n, rng, precision)` and `signal(m, truth)` (one voxel, no noise):
  `truth` gives a value for every parameter, a number, a map or a tie (`R1m = :R1f`). Nothing
  is filled in silently, so the script shows everything the data were made with: `nominal(m,
  tissue)` gives every value for a tissue, `nominal(m; changes...)` every value from the model
  (its starts, 1 for the scale, its fixed values), and `resolve(m, θ)` checks a set and turns its
  ties into numbers (for calling `signal!` yourself). `snr` is
  relative to the model's scale parameter (M0, S0, M0f: σ = scale/snr, "SNR at thermal
  equilibrium"), so it does not depend on the protocol; `σ` sets the noise directly, in signal
  units. Each image's own SNR is then its signal over σ, lower where the signal has decayed
  (the noise floor's bias depends on that one: `likelihood`, under "Free and fixed parameters"). `noise` is always given, as it follows from how the images were made, which the model
  does not know: `:rician` for magnitude images, `:gaussian` for real-valued data
  (phase-corrected, a real channel; the noise crlb assumes). A model's `magnitude` option says how
  the signal is computed, not how the data were reconstructed, so it does not choose the noise.
  Pass `rng = MersenneTwister(seed)` for repeatable data, and `precision = Float32` for data in
  half the memory (computed in Float64, then rounded: the same draws).
- **Averages:** `averages = nex` (the number of averages of each observation, or one number
  for all) makes `snr` or `σ` one average's noise, and observation j's σ/√NEXⱼ. With Rician
  noise, `averaging` says how they were combined, always given: `:complex` (complex images
  averaged, then the magnitude: Rician with σ/√NEX, so the noise floor falls with the averages)
  or `:magnitude` (magnitude images averaged, as in most diffusion, where shot-to-shot motion
  phase rules out complex averaging: the SD falls with √NEX, the floor stays at one average's).
  At 1.8σ (one average) and 8 averages, the floor is 0.106 averaged as magnitudes and 0.093 in
  complex, against a signal of 0.091, and the ADC of the example above is biased −1.96% instead of
  −0.33% (± 0.04%; SNR 20 per average at b = 0, 20 000 copies). `Σ` is the covariance of the data as fitted, after
  averaging, so it takes no `averages`.
- `TISSUES`, `tissue(name)`: T1 and T2 at 3 T from Stanisz et al., MRM 2005 (in vitro, 37 °C:
  `:WM`, `:GM`, `:blood`, `:liver`, `:muscle`, ...); `:CSF` (T1 = 4.124 s, the mean of three
  in vivo inversion recovery studies reviewed by Bojorquez et al., MRI 2017; T2 = 1.75 s, a
  placeholder); and the SIR tissue classes of Dortch et al., MRM 2018, Table 1 (`:SIR_WM_I`,
  `:SIR_WM_II`, `:SIR_GM`, `:SIR_lesion`). Your own values: `nominal(m, (T1 = 4.3, T2 = 2.0))`,
  or one value changed: `nominal(m, merge(tissue(:CSF), (T2 = 2.0,)))`.
- `nominal(m, tissue; values...)`: one model's parameters for a tissue, converting T1/T2 to rates
  where needed; the scale parameter is 1 unless given, and a parameter the model fixes takes
  its default (`defaults(m)`), shown in the result. Free values that are not tissue properties
  (an inversion efficiency `Sf`) must be given.
- `compare(res, truth; labels, mask)`: per parameter, overall and per label: bias, relative
  bias, SD, RMSE, relative RMSE, median relative error and % converged. `c.rows` is a table.
- `SIR_5POINT`, `SIR_4POINT`: the optimized schemes of Dortch et al., MRM 2018, Table 2;
  `SIR_16POINT`: the earlier 16-point scheme they were compared with (15 TIs log-spaced from
  10 ms to 2 s, rounded to the ms, then 10 s; TD = 2.5 s).

## Accuracy before scanning: crlb and montecarlo

Two ways to find a fit's bias, SD and RMSE for a protocol and a tissue: `crlb` predicts them
from derivatives (no noisy fits; microseconds), `montecarlo` measures them (simulate n noisy
copies, fit them; seconds), and shows `crlb`'s prediction alongside. Both judge a fit against
a truth, a `Scenario`, under a protocol.

```julia
θ = nominal(QMTSIR, :SIR_WM_I)                # the truth: kmf = 5 in this tissue
s = Scenario(QMTSIR, θ; kmf = 12.5)           # the fit assumes kmf = 12.5: wrong
m = QMTSIR(; SIR_4POINT...)                   # the protocol
c = crlb(m, s; snr = 200)                     # biased
c.bias.PSR, c.sd.PSR, c.relrmse.PSR           # by quantity, then parameter
crlb(m, θ; snr = 200, kmf = 12.5)            # the same, the scenario made inside
crlb(m, θ; snr = 200)                         # kmf held at its true value: unbiased
mc = montecarlo(m, s; noise = :rician, snr = 200)              # 10 000 noisy copies, Rician noise
mc.sd.PSR / mc.crlb.sd.PSR, mc.se_sd.PSR      # measured against predicted, ± standard error
```
```
julia> s
Scenario for QMTSIR: a truth and the fit judged against it
starts: the truth
  name     true  role   start/value  bounds
  PSR      0.2   free   0.2          [0.0, 1.0]
  R1f      1.5   free   1.5          [0.05, 10.0]
  Sf       -1.0  free   -1.0         [-1.2, 0.0]
  M0f      1.0   free   1.0          [0.0, Inf]
  kmf      5.0   fixed  12.5                       ← wrong
  R1m      :R1f  fixed  :R1f (tie)
  Sm       0.83  fixed  0.83
  Mzf_td0  0.0   fixed  0.0
  Mzm_td0  0.0   fixed  0.0
wrong: kmf = 12.5 (true 5.0): the fit is biased

julia> crlb(m, s; snr = 200)
crlb for QMTSIR, 4 observations, σ = 0.005, covariance: sandwich (the least-squares fit)
fixed in the fit: kmf = 12.5 (true 5.0), R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0
  parameter  true  noise-free fit  bias       rel bias  SD        rel SD  RMSE     rel RMSE
  PSR        0.2   0.1718          -0.02818   -14.1%    0.01156   5.78%   0.03046  15.2%
  R1f        1.5   1.5             5.096e-6   0.00034%  0.02581   1.72%   0.02581  1.72%
  Sf         -1.0  -1.019          -0.01897   -1.9%     0.009311  0.931%  0.02113  2.11%
  M0f        1.0   1.0             0.0002192  0.0219%   0.009748  0.975%  0.00975  0.975%
```

- **The scenario: a truth and the fit judged against it.** Made from the model's type (no
  protocol) and the truth θ (every parameter, the fixed ones included: what the tissue is). The
  fit is built from the truth: it estimates the model's free parameters (with the model's
  bounds) and holds the others at their TRUE values, so it is unbiased; changes, keywords as for
  `fit`, make it otherwise: `kmf = 12.5` (held at a wrong value, listed in `s.wrong`),
  `kmf = Free()` (estimated), `R1f = Free(start = 0.8)`, `PSR = Free(lower = 0.05)`. Fields:
  `truth`, `roles` (every parameter's role in the fit), `starts`, `wrong`, `weights`. The fit
  can be weighted, as `fit` is: `Scenario(m, θ; weights = nex)`, or `crlb(m, θ; weights = nex)`;
  crlb then gives the weighted fit's sandwich H⁻¹J₀ᵀWΣWJ₀H⁻¹ (the bound (JᵀΣ⁻¹J)⁻¹ when W = Σ⁻¹
  and nothing fixed is wrong), montecarlo fits with the weights, and side by side
  (`(plain = Scenario(m, θ), weighted = Scenario(m, θ; weights = nex))`) the two compare on the
  same data. The same scenario is
  judged under any protocol (`crlb(QMTSIR(; SIR_5POINT...), s; snr = 200)`), and `s.roles` fits
  real data the same way: `fit(m, data, s.roles)`. `crlb(m, θ; changes...)` and
  `montecarlo(m, θ; noise, changes...)` make one inside.
- **Starts:** `starts = :truth` (the default): every estimated parameter starts at its true
  value; `starts = :model`: the model's own rules (`:scale`, `BestOf`, `:data`), as `fit` starts
  a real fit, so montecarlo measures the starting rules too; a start given wins. crlb's
  noise-free fit always starts at the truth.
- **How crlb works:** the noise-free data are fitted once, starting at the truth; their answer
  θ̂ gives the bias. The SD comes from derivatives at θ̂ (ForwardDiff), first order in the
  noise. Two covariances: `cov = :sandwich` (default), the covariance of the least-squares fit
  itself (the "misspecified CRB"), and `cov = :biased`, Eq. 3 of Dortch et al., MRM 2018, a
  bound for any estimator with the same bias gradient. Both are the classical CRLB σ²(JᵀJ)⁻¹
  when nothing fixed is wrong, and equal when the fit is exact (as many observations as free
  parameters, e.g. the 4-point scheme); otherwise Eq. 3 is lower, and least squares follows
  the sandwich (16-point scheme, WM I, kmf fixed at 12.5: Monte Carlo PSR SD 0.00580 ±
  0.00003, sandwich 0.00576, Eq. 3 0.00544). Checked against the paper: Eq. 5 of the Table 2
  schemes is 0.243395 (4-point, kmf = 12.5) and 1.602062 (5-point, kmf fitted).
- **montecarlo** fits the copies with the scenario's fit (its starts and bounds; `solver` and
  `precision` as for `fit`), so local minima and Rician noise are part of what is measured; the
  statistics come with standard errors (`se_mean`, `se_sd`). The parameters are fitted to the
  same noise, so their errors move together: several ratios low in one sample are one
  fluctuation, not several.
- **Sweeps:** `snr = [100, 200, 400]` and `vary = :kmf => 5:2.5:20` (true values; several as
  `vary = [:kmf => 5:5:20, :PSR => [0.1, 0.2]]`; the fit stays the scenario's, so with
  `kmf = 12.5` this is the bias of that assumption against the truth, the paper's Fig. 2B; starts
  taken from the truth follow it; a derived quantity, with the truth given by derived
  quantities: `crlb(m, (S0 = 1000, MD = 0.8e-3, FA = 0.7, direction = (1, 0, 0)); snr = 50,
  vary = :FA => 0.2:0.1:0.9)` holds MD and the direction as given), and scenarios side by side, by name,
  `crlb(m, (assumed = s, fitted = Scenario(QMTSIR, θ; kmf = Free())); snr = 200)`, run every
  combination and return a `Sweep`: `s[i]`, `s.settings`, and `s.rows`, a table
  (`DataFrame(s.rows)`). In a Monte Carlo sweep every setting sees the same noise (one seed
  drawn from `rng`), so differences between settings are not noise: scenarios side by side are
  a paired comparison.
- **Noise:** `snr` (relative to M0), `σ` (a number, or one SD per observation, e.g. different
  bandwidths) or `Σ` (the full covariance matrix between observations, for correlated noise),
  for `crlb`, `montecarlo` and `simulate`; `averages = nex` with `snr` or `σ` (one average's:
  σ/√NEX per observation), also for `ProtocolDesign`. crlb takes no `averaging`: to first order
  averages give σ/√NEX however they were combined; the floor that magnitude averaging keeps is
  montecarlo's to measure (`montecarlo(m, θ; snr, averages, noise = :rician, averaging =
  :magnitude)`). One σ, the usual case for separately acquired images with the same readout,
  costs nothing extra.
- **Derived quantities** (a model's `derived`, above, e.g. DTI's MD, FA, AD, RD and V1,
  DKI's MK, AK, RK, MKT) come after the parameters in every named tuple, `c.sd.FA`,
  `mc.bias.MK`, in the tables and in a sweep's rows. crlb's: the noise-free fit's value g(θ̂),
  the bias g(θ̂) − g(θ), and the SD by the delta method, √(∇g·C·∇gᵀ) (C the parameters'
  covariance, ∇g by dual numbers through `derived`), first order in the noise as the
  parameters'; montecarlo's: from each copy's fit, so it also measures the noise's own bias of a
  nonlinear quantity (FA's, MK's at low SNR). Measured: DTI (6 b = 0 and 30 directions, SNR 50)
  and DKI (22 parameters, SNR 100), every SD within 1.5% of 4000–20 000 fits'. NaN where
  `derived` has no derivative: AD, RD, V1, AK and RK where λ1 = λ2 (no principal direction);
  MK and RK where two of D's eigenvalues are equal and W is not symmetric about the third axis
  (the axially symmetric truths `nominal` makes are: finite). A principal direction is folded into
  one hemisphere in each fit, so montecarlo's V1 near the fold (fibers in the x–y plane) is the
  folded vectors'. `vcov`, `stderror`, `cor` and `cond` stay the parameters'.
- **StatsAPI's names:** `informationmatrix(c)` (the Fisher information at the noise-free fit),
  `informationmatrix(c; expected = false)` (the observed information), `vcov(c)`,
  `stderror(c)`; `informationmatrix(m, θ; snr = 200)` at any values; `vcov(mc)`, `stderror(mc)`
  for a Monte Carlo.
- **Diagnostics, computed when called** (as `fitted(res)` and `residuals(res, y)` for a fit):
  `cond(c)`, the condition number of J₀ with each column scaled by its parameter's true value
  (how hard the data make the parameters to tell apart; scale-free, and independent of the
  noise level; `Inf` when they cannot be determined), and `cor(c)`, the predicted correlations
  of the estimates (`NaN` for a parameter held on a bound); `cor(mc)` the measured ones. They
  are LinearAlgebra's `cond` and Statistics' `cor`, re-exported; for a sweep,
  `cond.(s.results)`.
- **Leakage: which assumptions matter.** `leakage(m, s)` (or `leakage(m, θ; kmf = 12.5)`) gives,
  for each parameter the fit holds at a number, dθ̂/dc: how far the noise-free fit's answer
  moves per unit change in the value it assumes (the slope of Dortch 2018's Fig. 2B), at that
  fit. Times an error Δc it is the first-order bias, for every fixed parameter at once and
  without a fit per value; no noise involved. By fixed and then free parameter:
  `l.absolute.kmf.PSR` (PSR per s⁻¹), `l.relative.kmf.PSR` (per |true PSR|: −0.119, PSR 11.9%
  low per s⁻¹ of kmf assumed too high, 5-point, WM I) and `l.elasticity.kmf.PSR` (% per %, to
  compare fixed parameters with different units). From derivatives: −H_θθ⁻¹H_θc, the blocks of
  the cost's Hessian over the free and fixed parameters, which at the truth is
  −(JᵀJ)⁻¹Jᵀ∂g/∂c.
- **Uncertain inputs: a fixed value that was measured.** A B1 map, a T1 map for MTsat: the fit
  holds the parameter at a value that has its own error. `crlb(m, θ; snr, B1 = 1.05, uncertain
  = (B1 = 0.03,))` predicts a fit with a B1 map 5% high (the bias, as without `uncertain`) whose
  values scatter with an SD of 0.03 (in B1's units; independent of the image noise, another
  scan): the covariance adds D·s²·Dᵀ, D = leakage's slope at the value the fit assumes, so the
  SD, RMSE and derived quantities include it, to first order in both errors. `montecarlo` takes
  the same keyword and fits each copy with its own value, drawn from N(value, s²). Measured
  (T1MFA, flip angles 2–30°, T1 = 1 s, SNR 1000): T1's SD 0.0572 s predicted, 0.0577 ± 0.0003
  measured over 20,000 copies; the image noise alone gives 0.0223. Several at once, each
  independent of the others: QMTSIR's `uncertain = (kmf = 2.0, Sm = 0.05)`. Not yet with
  `likelihood = :rician`, `noise = :rician`, `simulate_with` or a protocol design.
- **Fast, and in named pieces.** `crlb` makes a `CRLBCache` (the scenario's fit, once) and calls
  `crlb!`; reuse the cache for many noise levels or truths (sweeps do; the fit stays the
  cache's):

  ```julia
  c = CRLBCache(m, s)
  [crlb!(c, θ; snr) for snr in (50, 100, 200)]
  ```

  `crlb(m, s)` takes 19–34 µs (4- to 16-point SIR), `crlb(m, θ; kmf = 12.5)` 30–47 µs (the
  scenario is 10 µs of it), `crlb!` 5–14 µs, a sweep 6–17 µs a setting, and the
  allocation-free kernel inside 1–16 µs.
  Each step is a function of its own, summed over the observations one row of J at a time (as
  the solver sums JᵀJ), with one-call versions from a model and a scenario (or the truth θ and
  the changes as keywords; the derivatives are taken at the truth, with the fit choosing the
  free parameters):

  | Function | Gives |
  |---|---|
  | `noisecovariance(m, s; snr)` | Σ, the noise covariance (N×N) |
  | `noisefree(m, s)` | θ̂, the fit of the noise-free data (a `FitResult`) |
  | `bias(m, s)` | b = θ̂ − θ |
  | `jacobian(m, s)`, `jacobian(r)` | J = ∂g/∂θ (N×P), at the truth or at a fit's answer |
  | `hessian(m, s, y)`, `hessian(r, y)` | H = JᵀJ + Σᵢ rᵢ∇²gᵢ, the cost's Hessian |
  | `informationmatrix(m, s; snr)` | F = JᵀΣ⁻¹J |
  | `biasgradient(m, s)` | ∇b (implicit function theorem, or `method = :unrolled`) |
  | `sandwich(H, J, Σ)` | H⁻¹JᵀΣJH⁻¹, the least-squares fit's covariance |
  | `biasedcrlb(∇b, F)` | (I + ∇b)F⁻¹(I + ∇b)ᵀ, Eq. 3 |

  ```julia
  r = noisefree(m, s)                                   # kmf held at 12.5: wrong
  sandwich(hessian(r, signal(m, θ)), jacobian(r), noisecovariance(m, s; snr = 200))
  biasedcrlb(biasgradient(m, s), informationmatrix(m, s; snr = 200))   # = crlb's two covariances
  ```
- Fits with the Rician noise model (`likelihood = :rician`, under "Free and fixed parameters"):
  give crlb and montecarlo the same keywords as the fit. `crlb(m, θ; snr = 10, likelihood =
  :rician)` predicts that fit (`method = :ml` by default; `:mean`, `:wmean`), its σ the noise
  given: for `:ml` the inverse Rician information (`ricianbound`'s) when nothing fixed is wrong,
  and with a wrong fixed value the likelihood's bias and the misspecified sandwich, to first
  order; for `:mean` the sandwich with the Rician variance. `montecarlo(m, θ; snr, noise =
  :rician, likelihood = :rician)` fits each copy that way, its crlb alongside the same fit's
  (checked: within 3 Monte Carlo standard errors for T2 decay at SNR 20, and for the 5-point SIR
  with kmf fixed at 12.5 at SNR 200). A scenario carries it too: `Scenario(m, θ; likelihood =
  :rician, method = :mean)`; sweeps follow. Not yet: protocol design, `leakage` and
  `cov = :biased` for these fits; montecarlo's with averages or σ per observation.
- Rician noise for a least-squares fit: `crlb(m, θ; snr, noise = :rician)` (with `averages`,
  `averaging` too), the same keywords as `montecarlo` and `simulate`, predicts a least-squares
  fit of magnitude data, its bias from the noise floor (exactly: the fit of the magnitudes'
  mean, `RicianMean`) and its SD (with the Rician variance). The default is `noise = :gaussian`
  (real-valued data), or `:rician` for a fit with `likelihood = :rician`; montecarlo's crlb
  alongside a least-squares fit stays the Gaussian one, so that the floor shows as the
  difference. `ricianbound(m, θ; snr, ...)` is the Cramér–Rao bound of Rician data (the information
  R(A/σ)/σ² per observation, as in Devan et al., Magn Reson Imaging 2020, Appendix), what no
  estimator beats. `averaging = :complex` (the scanner's averages: the floor falls) or
  `:magnitude` (a spherical mean over directions: it stays). Fitting `RicianMean(m; σ)` itself
  removes the floor's bias when σ is known (start it from the Gaussian fit's values; it gives the
  model's derived quantities too, DTI's MD and FA), at 1–4 times the Gaussian fit's cost; `fit(m,
  data; likelihood = :rician, σ, method = :mean)` does that in one call.
  `ricianmean`, `ricianvar`, `ricianinformation` for one value.
- Not modelled by crlb: bounds reached by noisy fits, nonlinearity at low SNR. That is what
  montecarlo is for.

## Protocol design

The protocol that minimizes the predicted error over tissues, Eq. 5 of Dortch et al., MRM
2018 by default: the worst scenario's sum of relative mean squared errors (bias² + SD², from
crlb's kernel) times the scan time. Each scenario is a tissue and what the fit assumes, so a
wrong assumption counts as the bias it causes.

```julia
tissues = [:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion]
ss = scenarios(QMTSIR, tissues; kmf = 12.5)        # the four tissues, the fit assuming kmf = 12.5
d = ProtocolDesign(QMTSIR(; SIR_4POINT..., magnitude = false), ss; snr = 200,
                   acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),   # to optimize, bounds (s)
                   constraints = m -> m.TI .+ m.TD .- 0.5,            # TR ≥ 0.8 s
                   scantime = m -> sum(m.TI .+ m.TD .+ 0.3))          # Σ(TI + TD + t_FSE)
d(QMTSIR(; SIR_4POINT..., magnitude = false))   # Eq. 5 of a protocol: 0.243395 (Table 2)
r = design(d; starts = 8, rng = MersenneTwister(1))
r.protocol, r.value             # 0.243255 (the paper's optimum before rounding: 0.2433)
r.rounded, r.rounded_value      # to the ms: 0.243312
```

- `d(x)`, x the varied values (the TIs, then the TDs), takes 11 µs here and allocates nothing:
  one cache per fit (the four tissues share one), the kernel once per scenario.
- In one call, `design(m, ss; snr = 200, acquisition, constraints, scantime, starts = 8, rng)`:
  `ProtocolDesign`'s keywords and the search's together, the objective made inside (as
  `fit(m, data)` is `init` then `fit!`). (`design` was `optimize_protocol` until 2026-11-09.)
- Options: `weights` per parameter, `budget` (a fixed scan time instead of × T),
  `aggregate` (`maximum` by default, Vmax; `mean`, a quantile, `v -> sum(w .* v)` with tissue
  weights), `shared = (:TD,)` (one TD for every point), `cov`.
- **Sets of scenarios**: `scenarios` makes the truths a design is judged over, the modes of the
  paper's `calcVmax`: one tissue, several (`'multiple'`), or a box of true values
  (`'ranges'`: every combination of the values given, plus with `midpoint = true` the center,
  2⁴ + 1 = 17 for four parameters), or tissues × conditions for a robust design. `vary` is
  crlb's (true values, every combination, the fit unchanged); the keywords are the fit
  (`Sm = 0.83`: what it assumes, whatever the true Sm):

  ```julia
  box = scenarios(QMTSIR, nominal(QMTSIR); kmf = 12.5, midpoint = true, vary = (PSR = [0.05, 0.2],
                  kmf = [5, 20], R1f = [0.5, 1.5], Sf = [-1, -0.8]))                   # 17
  robust = scenarios(QMTSIR, tissues; vary = :Sm => [0.78, 0.83, 0.88], kmf = 12.5, Sm = 0.83)  # 4 × 3
  crlb(m, robust; snr = 200)          # one crlb result per scenario (montecarlo too), a Sweep
  ```

  A set is points, not the box: corners and the midpoint need not contain a nonlinear model's
  worst case, and with `aggregate = mean` every point weighs the same (a discrete mean). The
  scale parameter (M0f) need not be varied: with `snr` relative to it, the relative errors do
  not depend on it.

  `vary` is crlb's (true values; the fit unchanged), and the result a `Sweep` whose results are
  the scenarios: `robust.settings` says which tissue and Sm each one has, and after `d(x)`,
  `robust.settings[argmax(d.values)]` is the worst. At Table 2's 4-point scheme, Eq. 5 is
  0.2434 over the tissues and 0.3274 with the true Sm 0.78–0.88 (the fit assuming 0.83), worst
  in WM I at Sm = 0.78. Each scenario costs 3–4 µs per evaluation: 243 (a 3⁵ grid) make one
  evaluation 0.8 ms.
- The search: differential evolution, then Nelder–Mead, from several starts on threads
  (repeatable with `rng`); the result rounded to the ms by a search of its own (rounding each
  value to the nearest ms can raise one of the tissues that tie at the optimum). It finds both
  Table 2 optima (4-point 0.2433, 5-point 1.6020) in seconds; docs/design.md has the
  comparison with simulated annealing and a genetic algorithm.

## Data files

`using EQUIP` reads and writes NIfTI files with their BIDS sidecars, through
[MRImages](../MRImages/) (in this repository: MRI images in and out of files, like Python's
nibabel, for any package to use). MATLAB `.mat` files work after `using MAT`. DICOM (and
Philips PAR/REC): convert it first with [dcm2niix](https://github.com/rordenlab/dcm2niix)
(`dcm2niix -b y -z y -o out/ dicomdir/`), which writes the NIfTI file and its sidecar, the
DICOM headers' values in BIDS names and units.

```julia
data, info = readdata("sub-01_MESE.nii.gz")         # 4-D, observations last; TE from the sidecar
mask = readmap("brain.nii.gz", info) .> 0          # checked against the data's size and geometry
res = fit(T2Decay(TE = info.found[:TE]), data; mask)
writemaps("results/sub-01_", res, info; stderror = true)   # T2map.nii.gz, S0map, status, SSE, ...
```

- `readdata(paths)`: one 4-D NIfTI file, or one file per observation (stacked in the order
  given; their size and geometry checked), the intensity scaling applied; `precision = Float32`
  for half the memory. `info.found` holds what the sidecars say: `EchoTime` → `TE`,
  `InversionTime` → `TI`, `RepetitionTime` → `TR` (s), `FlipAngle` → `FA` (degrees), `b`
  from an FSL `.bval` beside the data, and `gx`, `gy`, `gz` from a `.bvec` beside that
  (checked as `diffusionencoding` checks them). Nothing is converted: seconds and degrees, as in
  BIDS.
- MATLAB: `readdata("data.mat:S")` (the variable `S`; `data.mat` alone if it holds one array),
  observations along the last dimension after up to three spatial ones (`nx×ny×nz×N`,
  `nx×ny×N`, or `nvoxels×N` for voxels already masked); `readmap("data.mat:mask", info)`;
  `readvalues("protocol.mat:TI")`. `writemaps` then writes one `.mat` file, a variable per map
  in the data's own shape (or NIfTI in another image's geometry: `like = readimage(ref)`).
- `writemaps(prefix, res, info)`: one NIfTI file per map in the data's geometry, with BIDS
  names where they exist (`T1map`, `R1map`, `T2map`, `S0map`, ...), the status and SSE maps,
  and with `stderror = true` the standard errors; beside each a JSON sidecar recording how it
  was made (the model, its acquisition, each parameter's role, the precision, EQUIP's version).
- (`readdata`, `readmap` and `writemaps` were `loaddata`, `loadmap` and `savemaps` until
  2026-11-12: one verb pair, read and write, for every file function.)
- `readvalues(path)`: numbers from a text file (one per line, or an FSL `.bval` as it is);
  `readbvec(path)`: an FSL `.bvec` as its 3×N matrix; `diffusionencoding(bval, bvec)`: both,
  as a directional model's `b`, `gx`, `gy`, `gz` (from file names or from those arrays).
  `writetable(path, rows)`: crlb's and montecarlo's rows as TSV.
- Any image: `img = readimage(path)` gives `img.data`, `img.header` (the geometry), `img.meta`
  (the sidecar, every entry as it is); `writeimage(path, MRImage(values, img))` writes values
  in its geometry. `findmeta(img, "echo")` finds the sidecar's entries by name (any case,
  nested too), `headerinfo(img)` shows the header readable; from the command line,
  `equip info file.nii.gz [--find echo]`.

## Command line

The four parts as one command, `equip`, for work without writing Julia: NIfTI (or MATLAB) data
in, maps out, tables as TSV. A companion package in `cli/`. Once: `julia --project=cli -e 'using Pkg; Pkg.instantiate()'` (a few minutes, the
first time); then `cli/bin/equip` (put `cli/bin` on your `PATH`).

```bash
equip fit T2Decay t2.nii.gz                                   # TE from the BIDS sidecar
equip fit QMTSIR sir.nii.gz --protocol SIR_5POINT --kmf free,start=10 --mask brain.nii.gz
equip fit ADCDecay dwi.nii.gz --b @dwi.bval --weights @nex.txt --stderror --out res/sub-01_
equip fit DTI dwi.nii.gz                                       # b and the directions from dwi.bval, dwi.bvec
equip fit DKI dwi.nii.gz --out res/sub-01_                     # MK, AK, RK, MKT maps too
equip fit DTI dwi.nii.gz --likelihood rician --sigma sigma.nii.gz   # magnitude data at low SNR
equip crlb QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I,SIR_GM,SIR_lesion --snr 100,200 --kmf 12.5 --out crlb.tsv
equip montecarlo QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I --snr 200 --noise rician --kmf 12.5 --seed 1
equip design QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I,SIR_WM_II,SIR_GM,SIR_lesion --snr 200 \
    --kmf 12.5 --acquisition TI=0.01..10,TD=0.01..10 --readout 0.3 --out sir4.cnf
equip fit --config sir4.cnf sir4_data.nii.gz                   # fit data acquired with it
equip fit QMTSIR --help                                        # a model's flags
equip fit T2Decay data.mat:S --TE @data.mat:TE --mask data.mat:mask   # MATLAB in, maps.mat out
equip info sub-01_T2w.nii.gz --find echo                       # the header and the sidecar
```

- Flags are named after the model's fields (`--TI`, `--TE`, `--b`: a comma list,
  `start:step:stop`, or `@file`, one value per line or an FSL `.bval`) and parameters, by
  `fit`'s rules: `--kmf 12.5` fixes, `--kmf free,start=10,lower=0` estimates, `--R1m R1f` ties,
  `--Sm map.nii.gz` fixes to a map. A value given wins over the data's sidecar (with a warning
  if they differ). Times in seconds.
- Magnitude data at low SNR: `--likelihood rician --sigma SD` fits the Rician model
  (`fit(m, data; likelihood = :rician, σ, method)`), with `--sigma` the noise SD of each
  channel, a number or a map (`sigma.nii.gz`, or `file.mat:NAME`; voxels where it is 0 or NaN
  are not fitted), and `--method ml` (maximum likelihood, the default), `mean` or `wmean`.
  EQUIP does not estimate σ: give one from your own measurement or tool. The maps' sidecar
  records `Likelihood`, `Sigma` (as typed: the number or the map's file) and `Method`.
- `--save run.cnf` writes the command as a file of flags, one per line (FSL's style), the
  acquisition used included; `--config run.cnf` reads one, and flags after it win.
- The truth for `crlb`, `montecarlo` and `design`: `--tissue` (several: one scenario each) and
  `--truth NAME=value,...`; `--vary NAME=values` for true values swept or boxed. `--noise` is
  always given for `montecarlo`. `design` takes its bounds as `NAME=lower..upper`, the scan time
  as `--readout` (SIR-FSE) or `--scantime "sum(TI .+ TD .+ 0.3)"`, Eq. 5's weights as
  `--weights PSR=1,R1f=0`; `--out` writes the protocol it finds as a config file.
- A command starts in about 1–2 s (Julia starting and loading EQUIP; the work itself is
  milliseconds for crlb and small datasets). A case outside the precompiled ones (another
  model's crlb, a new combination of sweep axes) compiles once in that call, a few seconds.
  Faster start-up (a system image, a resident service) is on the roadmap.
- MATLAB files as `file.mat:NAME` wherever a file goes (the data, `--mask`, a map, `@file`
  values); the maps are then written to one `PREFIX_maps.mat`, or as NIfTI with
  `--like ref.nii.gz`. MAT is loaded only by a command that names a `.mat` file (about 0.4 s).

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
model's own estimates from the data for parameters declared `:data`; those declared `:loglinear`
from Gauss–Newton steps on ln y (the log-residuals' derivatives from the same dual numbers); then
a `BestOf` parameter at
the value that fits that voxel best, with the `:scale` parameter at its least-squares value for
each (or, without `BestOf`, for the start).

The code is in layers; each only uses the ones before it (`fit/` and `sim/` use `models/`
alone):

```
src/
  EQUIP.jl                the module: dependencies, exports, includes
  models/                 interface.jl (MRIModel, Free, Fixed, signal!, ...), builtin.jl (includes)
    relaxometry/          t1_ir.jl, t1_sr.jl, t1_mfa.jl, t2_decay.jl, t2_biexp_decay.jl
    diffusion/            adc_decay.jl, adc_kurtosis.jl
    qmt/                  qmt_sir.jl
  fit/                    levmar.jl (solver), setup.jl (roles -> FitSpec), result.jl
                          (FitResult), starts.jl (each voxel's starting point),
                          loglinear.jl (the :loglinear start), cache.jl (init), voxel.jl
                          (fit!, and fit for one voxel), dataset.jl (fit for a dataset,
                          threaded; refit)
  sim/                    from true values to data: tissues.jl (TISSUES, nominal),
                          noise.jl (snr, σ or Σ; noisecovariance), simulate.jl
  errors/                 a fit's errors against the truth (bias, SD, RMSE): scenario.jl
                          (Scenario: a truth and the fit judged against it), derivatives.jl
                          (crlb's pieces: CRLBCache, the sums, the formulas), crlb.jl (the
                          kernel, crlb, crlb!), montecarlo.jl, compare.jl (a dataset fit, e.g.
                          simulated maps), diagnostics.jl (cond, cor, leakage), sweep.jl
  design/                 design.jl (ProtocolDesign: the objective), search.jl (design)
```

`docs/design.md` records the design decisions.
