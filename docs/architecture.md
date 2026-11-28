# EQUIP's architecture: the code as it is

What is built, where it lives, the rules it keeps, how accurate it is and how fast. Read this
for the code; [status.md](status.md) for where the work stands (branches, what is in progress,
what waits on the owner); [design.md](design.md) for why each thing is as it is (the decision
log and the roadmap, indexed at its top); [guide.md](guide.md) for what users see. Updated
2026-11-28. Only what is built is described here: plans are in design.md's roadmap, rejected
approaches in its index ("Tried and not adopted").

## What EQUIP is, and the choices that shape it

Voxel-wise fitting of quantitative MRI models, an alternative to qMRLab: millions of small,
independent fits (fewer than 10 parameters and 100 observations each, up to 22 for DKI), then the
errors of those fits predicted before scanning and protocols designed from them (Dortch et al.,
Magn Reson Med 2018;80:1824 is the formulation). The owner is an MRI physicist learning Julia.

The choices everything else follows from (reasons in design.md, "Fitting pipeline" and
"Performance notes"):

- **A solver of its own**: Levenberg–Marquardt for one voxel on static arrays (`SVector`,
  `SMatrix` from StaticArrays: the number of free parameters is part of the type, the compiler
  unrolls the arithmetic, nothing is allocated). Box bounds by projection, with a dead-zone rule
  (`_lost_influence`). Measured faster than LsqFit (about 19×) and NonlinearSolve (3.7× its
  TrustRegion) on the 4-point SIR (design.md, Performance notes).
- **Exact derivatives from dual numbers** (ForwardDiff): a model writes only its signal; the
  fit's Jacobian, crlb's Hessians (nested duals) and every sensitivity come from it. Every
  function a model calls must take dual numbers (tables carry their own derivative rule).
- **Prepare once, solve many**: `init` turns a model and a fit's keywords into a `FitCache`
  (roles resolved, buffers made, the acquisition in the working precision); `fit!` then fits a
  voxel with no allocation. Datasets go over threads in chunks (`Threads.@spawn`).
- **Parameters by role**: each parameter is `Free` (estimated, with a start and bounds), `Fixed`
  (a number or a map, one value per voxel), or tied to others by a function. The user changes
  roles by keyword (`fit(m, data; kmf = Free(), B1 = b1map)`), never by editing the model.
- **Both precisions**: Float64 and Float32 (half the memory for big datasets) throughout.
- **One noise model** (`snr`, `σ` or `Σ`, src/sim/noise.jl) shared by `simulate`, `crlb` and the
  designs, so a prediction and a simulation assume the same thing.

## The layers, in the order a call goes through them

```
models/   a model: its acquisition (a struct), parameters(m) (roles, starts, bounds), signal!(out, m, p)
  │
fit/      setup.jl: roles + keywords → FitSpec      cache.jl: init → FitCache
          starts.jl, loglinear.jl: each voxel's start      levmar.jl: the solver
          voxel.jl: one voxel      dataset.jl: maps on threads, refit
          result.jl: FitResult      statistics.jl: σ̂, vcov, AIC/BIC from the fit's JᵀJ
          rician.jl: magnitude data, likelihood = :rician
  │
sim/      tissues.jl: TISSUES, nominal      noise.jl: snr/σ/Σ      simulate.jl      rician.jl: ricianbound
  │
errors/   scenario.jl: Scenario (a truth + the fit judged against it)
          derivatives.jl: crlb's pieces, allocation-free      crlb.jl: crlb, crlb! (bias, SD, RMSE)
          montecarlo.jl      compare.jl      diagnostics.jl: cond, cor, leakage
          propagate.jl: propagate(c, g)      sweep.jl: over SNR and true values; scenarios
          images.jl (unmerged): a quantity calculated from the images, propagate(m, θ, g)
  │
design/   design.jl: ProtocolDesign (Eq. 5)      search.jl: design (DE, then Nelder–Mead)
          objective.jl (unmerged): ObjectiveDesign, design(f, m)
  │
io/       encoding.jl: .bval/.bvec      data.jl: readdata, readmap      maps.jl: writemaps      tables.jl
```

`src/EQUIP.jl` includes them in this order and holds the export list; `src/precompile.jl` is the
precompile workload (most built-in models' fits: not DKI's, the SMT models' or QMTSPGRPortnoy's;
crlb, montecarlo, a design; 65 s to precompile, first calls then under 0.3 s).

## The model interface (src/models/interface.jl)

A model is a struct `<: MRIModel` holding its acquisition (one value per image in every vector
field; a pulse shape is held whole, as an `RFPulse`), and two methods:

- `parameters(::Type{<:M})`: a named tuple, each parameter's default role: `Free(start, lower,
  upper)` (start a number, `:scale`, `:data`, `:loglinear`, or `BestOf(values)`), `Fixed(value;
  lower, upper)` (bounds for when a fit frees it), a number, or a tie (`R1m = :R1f`, or a
  function of other parameters)
- `signal!(out, m, p)`: the signal of every image for the values `p` (a named tuple, every
  parameter by name), into `out`; any number type (plain or dual); allocating nothing

Optional hooks: `datastart(m, y, p)` (starts from the data), `relabel`, `derived(m, p)`
(quantities reported after the parameters, e.g. DTI's FA, qMT's MPF; with their errors by the
delta method), `fromderived` (`nominal(m; MPF = 0.12)`). `signal(m, θ)`, `resolve`, `nobs` and
`defaults` are built on these. A model in one step: `@mrimodel` (src/models/macro.jl, from a
function of one observation); as a file to edit: `newmodel` (src/models/template.jl).

Shared pieces a model may use: `inner(a, b)` (src/inner.jl, every dot product); `_QuinticTable`
(src/tables.jl, a function of one variable tabulated with its first two derivatives, with its
own dual-number rule: the super-Lorentzian's G, the Watson stick's kₗ); the diffusion
compartments `iso`, `stick`, `zeppelin`, `powder` (src/models/diffusion/compartments.jl), their
dispersed views `watson`, `bingham` (dispersion.jl), `fiberdirection`, `directionset`,
`DiffusionDirections`, `checkencoding` (directions.jl); for qMT the lineshapes `lorentzian`,
`gaussian`, `superlorentzian` (lineshapes.jl), `RFPulse`, `blockpulse`, `inversionefficiency`,
`saturationefficiency` (pulses.jl), and `_expm2`, the 2×2 matrix exponential in closed form
(twopool.jl).

## Built-in models

| area | models | file (src/models/) | where |
|---|---|---|---|
| T1 | `T1IR`, `R1IR`, `T1IRTD`, `R1IRTD` | relaxometry/t1_ir.jl | main |
| | `T1SR`, `R1SR`; `T1MFA`, `R1MFA` | relaxometry/t1_sr.jl, t1_mfa.jl | main |
| T2 | `T2Decay`, `R2Decay`, `T2DecayOffset`, `R2DecayOffset` | relaxometry/t2_decay.jl | main |
| | `T2BiexpDecay`, `R2BiexpDecay` | relaxometry/t2_biexp_decay.jl | main |
| diffusion | `ADCDecay`, `ADCKurtosis` | diffusion/adc_decay.jl, adc_kurtosis.jl | main |
| | `DTI`; `DKI` (MK, AK, RK, MKT derived as DIPY's) | diffusion/dti.jl, dki.jl | main |
| | `SMT`, `SMTFreeWater`, `SMTT2` (spherical means) | diffusion/smt.jl | main |
| | `StickZeppelin`, `StickZeppelinBall`, `BallStick` | diffusion/multicompartment.jl | main |
| | `NODDI`, `NODDIConvolved`, `BinghamNODDI`, `BinghamNODDIConvolved` | diffusion/noddi.jl | main |
| | `IVIM` | diffusion/ivim.jl | main |
| qMT | `QMTSIR`, `QMTSIREqualR1` (selective inversion recovery) | qmt/qmt_sir.jl | main |
| | `QMTCWHenkelman` (CW saturation; holds the family's parameters and R1f tie) | qmt/qmt_cw.jl | main |
| | `QMTSPGRYarnykh`, `QMTSPGRRamani` (pulsed MT with SPGR) | qmt/qmt_spgr.jl | main |
| | `QMTSPGRPortnoy` (MAMT: the two-pool Bloch equations through the MT pulse) | qmt/qmt_spgr.jl | `mamt-proto` |
| noise | `RicianMean(m)` (any model's Rician mean) | rician.jl | main |

Time and rate forms are separate types (`T2Decay` calls `R2Decay`'s signal). The pulsed-MT models
share one acquisition constructor (`_pulsedspgr`) and take Henkelman's parameters by delegation
(`parameters(::Type{<:QMTSPGRYarnykh}) = parameters(QMTCWHenkelman)`). MTR is a calculation, not a
model: `mtr` in qmt/mtr.jl (on `mtr` and `mamt-proto`, unmerged). Tissue presets (`TISSUES`, e.g.
`SIR_WM_I`, `DKI_WM`) and protocols (`SIR_5POINT`, `SMT_PROTOCOLS`) are in src/sim/tissues.jl and
the model files. The command line (`equip fit`) covers the relaxometry, diffusion (not SMT) and SIR models,
not the CW or pulsed-MT ones, nor MTR (cli/src/models.jl).

## Outside src/

- **MRImages/**: a package of its own, kept in this repository for now: images in and out of
  NIfTI files with BIDS sidecars, `.bval`/`.bvec`, MATLAB through an extension (loaded with MAT).
  EQUIP reads data through it (src/io/). DICOM is converted with dcm2niix, not read.
- **ext/EQUIPMATExt.jl**: only a precompile workload for .mat files (StaticArrays invalidates
  MRImages' own; design.md, "MRImages").
- **cli/**: the `equip` command (package EQUIPCLI, cli/bin/equip): `fit`, `crlb`, `montecarlo`,
  `design`, `info`; MAT loaded only when a command names a .mat file.
- **matlab/**: MATLAB functions over the command line for users without Julia (`equip_fit`,
  `equip_setup`, `equip`; matlab/README.md); tested in GNU Octave, not in MATLAB.
- **test/**: setup.jl (packages, test models, `CASES`: every built-in model with true values for
  every parameter, `@test_noalloc`, the independent references: SIR's matrix form, pulsed-MT SPGR
  played TR by TR, MAMT by RK4, Gauss–Legendre nodes) and the groups unit/, integration/,
  validation/ (runtests.jl's `GROUPS`; CLAUDE.md has the commands and the rules).
- **bench/**: speed.jl (the main paths), solvers.jl (against other packages, its own
  environment), and studies by area (bench/qmt/, bench/noddi/, bench/simd/, ...), each with its
  measurements in its header and in design.md. **examples/**: one script per part, runnable.

## Rules the code keeps

These are checked by the tests; a change that breaks one is a regression.

- **Allocation-free where it matters**: `fit!` (one voxel), `signal!`, crlb's kernel and pieces
  (`crlb!`, derivatives.jl), `ProtocolDesign`'s objective. Checked by `@test_noalloc`.
- **Type-stable**: `@inferred fit!(c, y)` for every model in `CASES`.
- **Both precisions**: every model in `CASES` fitted noise-free in Float64 and Float32.
- **Thread-safe**: a `FitCache` per task; models and settings immutable; no global state but the
  read-only tables.
- **Values by name**: parameters, truths, uncertain inputs and bounds are named tuples
  (`(B1 = 0.05,)`), checked in one place (`_byname`).
- **Units, fractions, readability**: the conventions in CLAUDE.md, Style (seconds and 1/s inside,
  fractions 0 to 1, each model's equations in its own `signal!`).

## Numerical accuracy

What is required, and what has been measured. A number the implementation produced is a
regression value, not a check (CLAUDE.md, Tests).

**Standards the tests hold**

| what | standard |
|---|---|
| noise-free recovery from default starts (`CASES`) | Float64 within 1e-6 relative, Float32 within 1e-3 |
| a signal against its independent reference | the reference's own accuracy (below), typically 1e-10 to 1e-14 |
| dual-number derivatives against central differences | 1e-6 relative (steps of 1e-6 of the value; 1e-4 for signals made by large matrix exponentials, whose rounding is about 1e-13) |
| crlb against montecarlo | within Monte Carlo standard errors (z-scores; test/validation/agreement.jl) |
| a refactoring | results bit for bit, or to rounding with the size stated (signals ~1e-15, fits ~1e-13), every model's starts and fits, both precisions |

**Measured accuracy of the approximations and references** (details in design.md)

| piece | accuracy | against |
|---|---|---|
| super-Lorentzian G, table (lineshapes.jl) | 1e-12 (first derivative 1e-10, second 1e-8) | QuadGK in BigFloat |
| Watson stick's kₗ, table (NODDI) | Jacobians 4e-13, Hessians 1.4e-8 of the series | the 8-shell series it replaced |
| QMTSIR closed form (Li et al. 2010) | 1e-14 (random values 1e-10) | the matrix form (Dortch 2018, Eq. 1), RK4 3e-13 |
| QMTSPGRYarnykh | 7e-15; 1e-12 | the lab's MATLAB; the sequence played TR by TR |
| QMTCWHenkelman (closed form) | 1e-12 | the two-pool linear system |
| QMTSPGRPortnoy (5×5 exponentials) | 3.6e-10 (RK4's own error); CW limit 2e-13 | RK4 of Eqs. 1a–1d; Henkelman's form |
| StaticArrays' 5×5 `exp` at large \|Hτ\| | 5e-15 (1 kHz) to 2.5e-13 (100 kHz) | BigFloat |
| Rician functions (Chebyshev Bessel fits, src/models/rician.jl) | 3e-14 | `besselix` (AMOS) |
| `inner` against LinearAlgebra's `⋅` | the last bits | (the order of the sum) |

**Known model approximations** (physics, not numerics): Yarnykh's pulsed MT is above MAMT near
resonance (white matter, 12-ms block: +20% and +79% at 1 kHz for 4 and 8 µT, +9% and +5% at
2 kHz, under 0.1% from 50 kHz); Ramani's CWPE biases PSR −29% at 10°; the excitation's saturation
of the bound pool is neglected in all pulsed models; ideal spoiling.

## Performance

Measured in cloud containers, which vary 10–25% run to run: compare rows within one run, not
across sessions. No baseline file is stored yet (roadmap item 21); `bench/speed.jl` prints the
main paths' times (about 1–2 minutes), bench/qmt/speed.jl the qMT models'.

| what | time | source |
|---|---|---|
| one voxel: QMTSIR 4-point | 0.41 µs a voxel on 4 threads | design.md, "Batching" |
| one voxel: T1IR, T1MFA | 1.7, 0.7 µs (1 thread) | Performance notes |
| 368k voxels, T2Decay 32 echoes / SIR 5-point | 0.16–0.19 s / 0.24 s on 4 threads | Performance notes |
| whole brain, 294,912 voxels, 4 threads: DTI, DKI | 1.4 s, 10.0 s | bench/qmt/brain.jl (2026-11-26) |
| … StickZeppelinBall, NODDI, NODDIConvolved | 3.2 s, 15.4 s, 30.6 s | same |
| … BinghamNODDI, BinghamNODDIConvolved | 95 s, 177 s | same; bench/noddi/four_brain.jl |
| … QMTSIR, QMTSIREqualR1 | 0.18 s, 0.13 s | bench/qmt/speed.jl (2026-11-27) |
| … QMTSPGRYarnykh, QMTSPGRRamani, QMTCWHenkelman (13 images) | 2.2 s, 0.9 s, 0.85 s | same |
| QMTSPGRPortnoy | not yet measured (status.md, next step 2); its 5×5 `exp` 0.4 µs plain, 6.6 µs with 10 partials | design.md item 6 |
| against other solvers (4-point SIR, 4 threads, µs a voxel) | EQUIP 0.94, NonlinearSolve TrustRegion 3.5, LsqFit 18, NonlinearSolve LM 155 | bench/solvers.jl |
| solver size | P = 7: 2.4 µs, P = 22: 64 µs, P = 28: 74 µs a fit | Performance notes |
| precompiling EQUIP / first calls after | 65 s / under 0.3 s | src/precompile.jl |

Where the time goes, and what did not help, is in design.md's Performance notes and its index
("Tried and not adopted").
