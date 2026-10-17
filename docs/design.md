# EQUIP design (0.3)

The design decisions, recorded so the code and the discussion share one reference.

## Inputs: where each kind of information lives

| input | varies across voxels? | Julia form | example |
|---|---|---|---|
| **Model**: physics, acquisition and options | no | a `@kwdef` struct `<: MRIModel` | `SIR(TI = [...], TD = 2.5, magnitude = true)` |
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
- **Model families share a loop, with one model per kind of measurement.** `T2Decay`,
  `T2StarDecay` and `DiffusionDecay` are one equation, S0·exp(−R·x) + C. The loop is written once
  (`_monoexp!` in `src/models/decay.jl`); each model declares only its acquisition field and its
  parameters (names, starts, bounds), about five lines, which also gives each output map its name.
  - Considered instead: one model with the parameter name as a type parameter
    (`EchoDecay{:T2star}`): the per-name information moves into lookup methods, and it needs a
    custom constructor and special handling when `init` rebuilds the model in Float32. Or models
    generated with `@eval`: shortest, but hard to read and debug.
  - Variations of one equation are parameter roles, not models: an offset is `C = Fixed(0.0)`,
    estimated with `free = (:C,)`.
  - Rate or time constant: least squares gives the same answer either way, so R2 = 1/T2 after
    the fit. Add a rate-form model (e.g. `R2StarDecay`) when the fit itself needs it, e.g. for
    nearly flat curves, where R2* settles near 0 and T2* runs off to large values.
- **Standard function names are shared, not redefined.** `fit`, `fit!` and `nobs` are StatsAPI's
  functions (the interface behind StatsBase, GLM, Distributions) and `init` is CommonSolve's
  (SciML). We add methods for our types, so loading those packages alongside causes no clash.

## Writing a model

```julia
@kwdef struct SIR{A<:AbstractVector,B<:AbstractVector} <: MRIModel   # 1. acquisition + options
    TI::A
    TD::B
    magnitude::Bool = true
end
SIR(TI, TD::Real, magnitude) = SIR(TI, fill(TD, length(TI)), magnitude)   # one TD for all

parameters(::SIR) = (                          # 2. every parameter with its default role
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf  = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),             # starts at the least-squares signal scale
    kmf = Fixed(12.5),
    R1m = Fixed(:R1f),                         # tie: always equal to R1f
    Sm  = Fixed(0.83),
    Mzf_td0 = Fixed(0.0),                      # Mz at the start of TD (fraction of equilibrium):
    Mzm_td0 = Fixed(0.0),                      # 0 = saturated by the readout
)

function signal!(out, m::SIR, p)               # 3. one voxel's signal at every observation
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf, R1m, Sm, Mzf_td0, Mzm_td0) = p
    @inbounds for j in eachindex(out, TI, TD)  #    one length check, then TI[j], TD[j]
        ...
    end
end
```

Rules for `signal!`: no allocation, no state, generic in the number type of `p` (dual numbers
pass through it). Loop with `eachindex(out, <every per-observation field>)`: one length check
per call (no measurable cost), after which `@inbounds` is safe whoever calls `signal!` (the
fitter, `signal`, or future CRLB / protocol code with its own buffers). Measured: `@inbounds`
gains ~4–7% in a `signal!` loop and up to 1.16–1.41× over whole fits (mostly the solver's own
loops); `@simd` gains nothing real here (`exp` prevents vectorization). `signal(m, p)`
(provided) allocates the output and fills in fixed defaults, for simulation.

## Parameter roles

- `Free(init; lower, upper)`: estimated. `init` is a number, or `:scale` for the parameter that
  multiplies the whole signal. A `:scale` parameter starts, per voxel, at
  (s·y)/(s·s), where s is the signal at scale 1 given the other starts. This one generic rule
  replaces data normalization. Model-specific guesses are deferred.
- `Fixed(value)`: not estimated. `value` is a number, a per-voxel map, or a tie, resolved after
  the free values are merged in, so derivatives flow through it. A tie may not refer to another
  tie.
  - **By name**, `Fixed(:R1f)`: always equal to R1f. The common case. It is checked in `setup`
    (the name must exist and must not be a tie), can be written in config files and on a command
    line later, and becomes `_Tie{:R1f}()`, so resolving it is a field read whose name the
    compiler knows.
  - **By function**, `Fixed(p -> 0.8 * p.R1f)`: for real relations; a named function for long
    ones. Cannot be checked before it runs.
  - An R1 map as a constraint on SIR needs no extra feature: with R1m = R1f the slow two-pool
    rate is exactly R1f, so `fixed = (R1f = r1map,)` does it. (A formula R1f(R1obs, …) would
    only be needed if R1m ≠ R1f; that would require auxiliary per-voxel inputs. Not planned.)

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
  EQUIP.jl             the module: dependencies, exports, includes
  models/              interface (MRIModel, Free, Fixed, parameters, signal!, signal, nobs);
                       builtin.jl + one file per model
  fit/                 levmar.jl (solver), setup.jl (roles -> FitSpec),
                       cache.jl (init, fit!), fit.jl (fit: one voxel or a dataset, threaded)
  sim/                 (later) simulate (noise models), digital phantoms, tissue presets
  analysis/            (later) compare (fit vs truth), Monte Carlo, CRLB, sensitivity
  design/              (later) protocol objectives (Eq. 5, Eq. 6), scan time
  io/                  (later) loaddata, protocol files, units, savemaps, PAR/REC reader
ext/                   (later) package extensions: OptimExt (optimize_protocol), and one per file
                       format package (NIfTI, MAT, NPZ, DICOM, JSON3, MRIFiles)
test/                  correctness and deterministic performance checks (no allocation, type
                       stability); no timings
bench/                 (later) own Project.toml (BenchmarkTools, LsqFit): speed suite + accuracy
                       studies, results.md
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

**Name.** `EQUIP`: **E**stimation of **QU**antitative **I**maging **P**arameters (the expansion
can grow to "…Parameters and Protocols" as protocol design becomes central). A pronounceable
word, specific about what the package does; ".jl" already says "in Julia" (Julia's naming
guidelines advise against "Julia"/"Ju" in names). All-caps acronyms are fine in Julia (DECAES,
DICOM). Earlier names: VoxelFit (only fitting), MRISignalModels (descriptive but long).
Alternatives considered: Quanta, Saguaro, Larmor, DESQ, QUIJOTE; avoided QUIT (an existing C++
qMRI toolbox), QUEST (an MRS method), QED (quantum-electrodynamics packages), and the
institute's name (implies endorsement). Checked against the General registry (Oct 2026):
unregistered. `QuantitativeMRI` is taken (aTrotier/QuantitativeMRI.jl).

## Performance notes

- **SIR `signal!` follows Li et al. 2010 (Eqs. 3–6) step by step**: recovery over TD from the
  start values, the inversion pulse, recovery over TI. It uses Li's names (R1⁺, R1⁻, bf⁺, bf⁻,
  and Mz as a fraction of equilibrium), so each line can be checked against the paper. Li's
  Eq. 4 already allows any starting values, so `Mzf_td0`/`Mzm_td0` (Mz at the start of TD) cost
  nothing: whole fits measured the same with and without them. Verified against the matrix form
  (Dortch 2018, Eq. 1, extended to nonzero start values; ~1e-14), a numerical (RK4) solution of
  Li's Eq. 2 (~3e-13), and finite-difference Jacobians (~6e-7, the finite differences' error).
- **What made it fast, measured with `@belapsed`** (5-point scheme / 16-point scheme, whole fits):
  - Naming each `exp` once (`E_td⁺`, …) matters: writing them out where used is 1.7–2× slower,
    because the compiler does not merge repeated `exp` calls.
  - Naming plain arithmetic does not matter: the compiler shares repeated products itself, so
    intermediate names (the old `bf1`, `bf2`) are for readability only.
  - Moving parameter-only work out of the loop gains nothing measurable (the compiler does it).
  - bf⁻ from bf⁺ + bf⁻ = Mf(0) − 1 (Li Eq. 3 at t = 0) instead of Li's full expression: ~7%
    faster (2.2 vs 2.37 µs; 4.1 vs 4.4 µs). Multiplying by 1/(R1⁺ − R1⁻) instead of dividing:
    ~3%.
  - A rearranged version (all parameter-only work folded into six coefficients, so
    `s = M0f + A·exp(-R1⁺·TI) + B·exp(-R1⁻·TI)` with A, B depending only on TD, reused when TD
    repeats) is 1.05× faster on the paper's 5-point scheme and 1.6× with one TD. Not used:
    harder to check against the paper. Revisit if Monte Carlo or protocol optimization on
    constant-TD schemes becomes speed-limited.
- **The matrix form written literally is slower**: Base `exp` on 2×2 matrices is ~180× slower
  than the closed form and cannot take dual numbers; 2×2 StaticArrays with `exp(A·t)` in closed
  form (eigen-decomposition) is 1.3–1.4× slower and works with duals. The tests use Base `exp`
  as an independent reference. A reasonable route for models without a convenient closed form.
- **Decays**: the loop multiplies by R = 1/T2 (computed once per call) instead of dividing
  every TE by T2: 5–16% faster whole fits. Adding the offset C at every echo cost ~8%, so it is
  skipped when C is a plain 0 (a free C is a dual number during the fit and is always added).
  Net against the old `ExpDecay` (minimum of 10 alternating `@belapsed` runs; this machine varies
  15–25% between single runs): 593 → 576 ns with 8 echoes, 1503 → 1283 ns with 32 echoes.
- **Solver size limits** (`levmar`, static-array method): no allocation and faster than LsqFit
  up to at least 28 free parameters (DKI-like test, 3 observations per parameter: P = 7 2.4 µs,
  P = 15 25 µs, P = 22 64 µs, P = 28 74 µs per fit; LsqFit 3.5–19× slower). Compile time grows
  (≈1 s → ≈8 s, once per model and set of free parameters). Beyond ~30–40 free parameters, add
  a `levmar` method for ordinary arrays (`mul!`, `cholesky!`); the model interface does
  not change.
- **Threads**: chunks of voxels as tasks (`Threads.@spawn`), about 64 per thread with at least 16
  voxels each; plain `Threads.@threads` was 2× slower when slow voxels cluster (it splits the
  work into one block per thread). See the comment in `src/fit/fit.jl`.

## Scope so far

In: the interface above, seven models (T2Decay, T2StarDecay, DiffusionDecay, InversionRecovery,
SaturationRecovery, MFA, SIR), `fit` for one voxel or a dataset, Float64/Float32, bounds, masks, per-voxel maps, ties,
threads, tests.

Out for now; earlier implementations are on the `api-v0.2` and `main` branches:
CLI / Python / MATLAB, GPU, model-specific and neighbour starting values, standard errors,
weights, other data layouts, precompilation, more models.

## Roadmap

1. **One result type for `fit`** (`FitResult`, replacing `VoxelResult` and the named tuple
   returned for datasets). Same fields for one voxel and for a dataset, numbers or maps:
   - `params` (named tuple), `sse`, `iterations` (datasets gain an iterations map), `status`
     as `FitStatus` (prints `ConvergedX` instead of `0x01`; still 1 byte per voxel;
     `UInt8.(res.status)` for writing files),
   - `model` (the model fitted, acquisition included) and `settings` (`fixed`, `free`, `start`
     as used), as references, not copies. Lets `savemaps` write the provenance sidecar, lets
     `fitted(res)` / `residuals(res)` (StatsAPI names) rebuild a single voxel's curve, and
     allows refitting with the same settings,
   - a summary display (model, voxels, % converged, free and fixed parameters).
   With the uncertainty work (3): JᵀJ for a single voxel, optional standard-error maps for
   datasets, a flag for parameters that ended at a bound, and StatsAPI queries (`coef`,
   `coefnames`, `rss`, `stderror`, `vcov`, `informationmatrix`). The internal `SolverResult`
   stays as it is; the solver already computes JᵀJ at the solution.
2. **Simulation, digital phantoms and benchmarks**: one simulation layer (`sim/`) shared by
   Monte Carlo, benchmarks, accuracy studies and tests, and useful to users directly.
   - `simulate(m, truth; noise = :rician | :gaussian, snr or σ, rng)`: noisy data, observations
     last. `truth` is a named tuple like `p` in `signal(m, p)`: numbers (one tissue) or maps
     (a phantom). The SNR definition is stated explicitly (e.g. relative to M0).
   - Digital phantoms: maps of tissue parameters with spatial structure. `phantom(labels,
     tissues)` turns an integer label image into parameter maps (optional smooth variation,
     partial volume, a B1 field, lesions, background); `phantom(:shapes, size; tissues)` makes
     synthetic geometry with no external data. Later, brain-like phantoms from tissue
     segmentations (BrainWeb, or the brain phantom KomaMRI uses, read from file) and real data
     (qMRLab's OSF datasets).
   - Tissue presets: the Table 1 classes of the 2018 paper for SIR; literature T1/T2 for
     WM/GM/CSF.
   - `compare(res, truth; mask, labels)`: bias, RMSE, % converged and time, overall and per
     tissue label.
   - Monte Carlo vs phantoms: Monte Carlo (independent noisy copies of chosen tissues) gives
     bias/SD/RMSE against the CRLB and accuracy vs SNR; phantoms are needed for everything that
     uses neighbours or geometry (neighbour starts, outlier detection, masks) and for realistic
     whole-volume speed, where hard voxels cluster.
   - Where things live: `test/` gets a tiny seeded phantom (e.g. 8×8×2) for an end-to-end check
     (simulate → fit → compare) with loose accuracy limits, plus `@inferred` checks next to the
     allocation checks; timings never go in tests. `bench/` (own Project.toml) gets the speed
     suite as `const SUITE = BenchmarkGroup()` (the convention PkgBenchmark.jl and
     AirspeedVelocity.jl use to compare commits), timed with BenchmarkTools (`@btime` /
     `@belapsed` with `$` interpolation; `evals = 1, samples = 5` for whole-dataset passes): one
     fit per model, `signal!` alone, dataset throughput on threads, bounds vs no bounds, scaling
     with P, LsqFit comparison. Accuracy studies are scripts there too (starting-value
     strategies, bounds, accuracy vs SNR, protocol comparisons), each writing a table to
     `bench/results.md` with machine, Julia version and thread count.
3. **Uncertainty** (formulation from Dortch et al., Magn Reson Med 2018;80:1824):
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
   - Sensitivity to model assumptions: the bias from fitting with an assumed value that is
     wrong, the same machinery as Eq. 3's b. First case: SIR's `Mzf_td0 = Mzm_td0 = 0` (both
     pools saturated at the start of TD). Simulate with nonzero values, fit with the default.
     A first check (5-point scheme, noise-free, 5% of M0 left): PSR +2%, R1f +4%, M0f −5%.
   - Validation targets: Table 1 tissue classes, Table 2 schemes, Fig. 2B.
4. **Starting values and outliers**:
   - Starting values: model-specific guesses, neighbour/previous-voxel starts with quality
     gates.
   - Outlier detection after a dataset fit, from what the fit already reports: status (hit
     `maxiter`, failed, invalid), a parameter that ended at a bound (the `FitResult` flag of
     item 1), and a large deviation from the local neighbourhood (e.g. distance from the
     median of the surrounding voxels, in units of a robust spread such as the MAD).
   - Then, per outlier, either refit starting from its neighbours' values (keep the refit if
     it converges, within bounds, with a lower SSE), or replace it with a neighbourhood
     average. Mark replaced voxels in the status map, so they are never mistaken for fitted
     values.
   - Bounds stay on by default: measured on SIR-FSE (5-point scheme, SNR 100), they cost
     nothing for good voxels (2.98 vs 3.00 µs, same iterations and results) and make bad ones
     faster and contained (noise-only voxels: 1 vs 154 hitting `maxiter`; 0 vs 874/1000 PSR
     values outside [0, 1]). Unbounded fits remain possible per fit via `free`.
5. **Data I/O**: make getting data in and maps out as easy as possible. An `io/` layer
   (dispatch on file type, protocol handling, the PAR/REC reader) plus one package extension per
   file-format package, so the core never depends on file formats.
   - Reading: `data, info = loaddata(path_or_paths; ...)` returns the data with observations
     last, plus `info`: the geometry needed to write maps back, and any acquisition values found
     (TE, TR, FA, TI, ...).

     | format | via | notes |
     |---|---|---|
     | NIfTI (.nii, .nii.gz) | NIfTI.jl | apply `scl_slope`/`scl_inter`; memory-map uncompressed files |
     | BIDS JSON sidecars | JSON3.jl | EchoTime, RepetitionTimeExcitation, FlipAngle, InversionTime (s, deg); qMRI file collections (`_echo-`, `_flip-`, `_inv-`, `_mt-` entities) stacked into one array |
     | MATLAB .mat | MAT.jl | pick the variable by name |
     | NumPy .npy / .npz | NPZ.jl | C order: reverse dimensions, or observations first |
     | DICOM | DICOM.jl | one series; TE/TR/FA/TI tags (ms → s); sort by instance/echo number |
     | Philips PAR/REC, XML/REC | own reader (text/XML + raw binary; DECAES has one to learn from, check its license) | per-slice scaling; TE/TR/FA/TI in the header |
     | ISMRMRD | MRIFiles.jl / HDF5.jl | image datasets; raw k-space must be reconstructed first (e.g. MRIReco.jl) |

   - Several files (one per echo, flip angle or inversion time): a list or pattern, stacked
     along the observation dimension and sorted by the acquisition value or BIDS entity index.
   - **Acquisition values that are not in the headers** (TI and TD for SIR rarely are):
     - Priority: values given in the call > a protocol file > values found in headers/sidecars.
     - Protocol file: TOML/JSON with named vectors, or CSV with one row per observation:
       ```toml
       model = "SIR"
       TI = [0.010, 0.050, 0.056, 0.277, 0.843]
       TD = [3.270, 4.489, 1.652, 2.922, 0.010]
       ```
     - The model is built with its keyword constructor, e.g. `SIR(; TI, TD)`, so this works
       for every model, user-defined ones included. Missing fields raise an error that names
       them and says where they were looked for.
     - Units: converted to seconds and degrees on reading (DICOM and PAR/REC store ms, BIDS
       uses s). Warn when values look like the wrong unit (e.g. TE > 10 s). The number of
       observations must match the data's last dimension.
   - Writing: `savemaps(prefix, res; like = info)` writes one file per map in the input's format
     and geometry (intensity scaling reset), with BIDS-derivative names where they exist
     (T1map, R1map, T2map, T2starmap, M0map, S0map, ...), plus status and SSE maps and a JSON
     sidecar recording the model, acquisition, fixed/free/start settings and package version.
6. **Front ends**, built on the `io/` layer: command line (data files + protocol file in, map
   files out), Python, MATLAB.
7. **`levmar` for ordinary arrays** (only when a model needs more than ~30–40 free parameters):
   a second method of the same function, chosen by dispatch on the argument types:
   `levmar(prob, θ0::Vector, lo::Vector, hi::Vector, opts, work::LevmarWorkspace)`, with a
   workspace allocated once per thread (JᵀJ, Jᵀr, step, Cholesky factor) and in-place
   `mul!`/`cholesky!`/`ldiv!`. `setup` picks static or ordinary arrays from the number of free
   parameters, once per fit (a threshold around 32), so every voxel dispatches to the right
   method at no run-time cost. Derivatives may need chunked dual numbers
   (`ForwardDiff.JacobianConfig`) at that size. The model interface does not change.
8. **Later**: GPU, Rician-likelihood fitting, more models:
   - Gradient-echo decay with the through-slice sinc term for macroscopic field gradients,
     S0·exp(−TE/T2*)·|sinc(Δf·TE)| (Fernández-Seara & Wehrli, MRM 2000), as a separate model.
     Δf, the frequency spread across the slice (Hz), is fixed from a B0 field map or fitted
     (default 0: no correction). Julia's `sinc` is the normalized sinc, matching the slice
     integral. Later: in-plane gradients (a product of sincs), non-rectangular slice profiles.
   - Rate-form decays (R2StarDecay, R2Decay) when a fit needs them (see "Model families share a
     loop" above).
   - Diffusion tensor and kurtosis, IVIM, multi-exponential T2 (myelin water), DCE.
