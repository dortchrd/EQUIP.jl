# EQUIP.jl: notes for Claude

EQUIP (Estimation of QUantitative Imaging Parameters) is a Julia package for voxel-wise MRI
fitting, an alternative to qMRLab: millions of small fits (fewer than 10 parameters, fewer than
100 observations each) with a custom Levenberg–Marquardt solver, ForwardDiff duals and
StaticArrays. The owner is an MRI physicist learning Julia.

## Commands

- Tests: `julia -t auto --project=. -e 'using Pkg; Pkg.test()'` (test/runtests.jl, about 1800 tests);
  the command line's: `julia --project=cli -e 'using Pkg; Pkg.test()'`; MRImages':
  `julia --project=MRImages -e 'using Pkg; Pkg.test()'`
- Examples (4-point SIR, SNR 200, CSV output): `julia -t auto --project=. examples/sir_4point_range.jl`
  (tissue values over the paper's range) or `examples/sir_wm_4point.jl` (white matter only);
  `examples/one_free_parameter.jl` (one parameter fitted, the others known; about 25 s)
- Benchmark vs LsqFit, NonlinearSolve and FastLevenbergMarquardt over the test cases in bench/cases.jl (`CASES` picks which): `julia -t auto --project=bench bench/solvers.jl`
  (own environment: `julia --project=bench -e 'using Pkg; Pkg.instantiate()'` once)
- Speed of the main paths (fit!, datasets, crlb!, sweeps, montecarlo, a design, fit statistics),
  in the package's own environment: `julia -t auto --project=. bench/speed.jl` (about 1 minute)
- Monte Carlo against crlb (bias and SD, z in Monte Carlo standard errors; T2, Dortch 2018
  Fig. 1, an SNR sweep): `julia -t auto --project=. bench/agreement.jl` (about 25 s)
- First setup: `julia --project=. -e 'using Pkg; Pkg.instantiate()'` (Manifest.toml is not committed)
- The first `using EQUIP` after a code change takes about 40 s (precompile workload,
  src/precompile.jl). While developing, switch it off with a `LocalPreferences.toml` (not
  committed) containing `[EQUIP]` and `precompile_workload = false`; about 3 s per change then.

## Where things are

- src/models/interface.jl: the model interface (`MRIModel`, `Free`, `Fixed`, `BestOf`,
  `parameters`, `signal!`, `datastart`); src/models/macro.jl: `@mrimodel` (a model from a
  function of one observation); src/models/template.jl: `newmodel` (a model file to edit, blank
  or written out from an @mrimodel definition); src/models/*/: the built-in models
- src/fit/, in pipeline order: setup.jl (`FitSpec`) → starts.jl → cache.jl (`init`, `FitCache`)
  → voxel.jl (one voxel) → dataset.jl (maps, threads, `refit`) → statistics.jl (`stderror`,
  `vcov`, `cor`, `residualsd`, AIC/BIC, `onbound`: from the fit's own JᵀJ); levmar.jl is the
  solver, result.jl the `FitResult`
- src/sim/, from true values to data (uses models only): `nominal` and tissue presets
  (`TISSUES`), noise.jl (the noise model: `snr`, `σ` or `Σ` checked once for
  `simulate` and crlb; `noisecovariance`), `simulate`
- src/errors/, a fit's errors against the truth (bias, SD, RMSE): scenario.jl, the `Scenario` (a truth and the fit judged against it, built
  from the truth with fit's keywords: what crlb and montecarlo take; `crlb(m, θ; kmf = 12.5)`
  makes one); derivatives.jl, crlb's pieces one function each (`CRLBCache`, `noisefree!`,
  `hessian!`, `meat`, `information`, `sandwich`, `biasgradient`, `biasedcrlb`; allocation-free,
  summed row by row from dual numbers), with one-call versions (`jacobian`, `hessian`,
  `informationmatrix`, ...); crlb.jl, the kernel `_crlb!` (their composition), `crlb!` and
  `crlb` (bias, SD and RMSE predicted from derivatives; sandwich or Eq. 3);
  montecarlo.jl, `montecarlo` (the same measured by simulating and fitting, with `crlb`
  alongside); compare.jl, `compare` (the same measured on a dataset fit, e.g. simulated maps);
  diagnostics.jl, `cond`, `cor` and `leakage` (where the errors come from: dθ̂/dc of each
  fixed parameter, from a joint cache with the fixed numbers seeded too); sweep.jl:
  crlb or montecarlo over noise levels and true values (`snr = [...]`, `vary = (...)`),
  returning a `Sweep`, and `scenarios` (a set of `Scenario`s: tissues, boxes of true values,
  or both; calcVmax's modes; a `Sweep` whose results are the scenarios), which `crlb` and
  `montecarlo` also take
- src/design/: design.jl, `ProtocolDesign` (the objective of a protocol search, Eq. 5 by
  default, made once from scenarios: one cache per fit, the kernel per scenario, allocation-free);
  search.jl, `optimize_protocol` (differential evolution, then Nelder–Mead, starts on threads;
  the result rounded to the ms by a search of its own)
- src/io/: data files in, maps out, on MRImages: data.jl (`loaddata`, `loadmap`, `DataInfo`:
  files stacked into observations, sidecar keys to model fields), maps.jl (`savemaps`: BIDS
  names, a provenance sidecar per map; one .mat file for MATLAB data), tables.jl (`writetable`);
  ext/EQUIPMATExt.jl: only a precompile workload for .mat files (loaded with MAT; StaticArrays
  invalidates MRImages' own, docs/design.md)
- MRImages/: a package of its own (in this repository for now): MRI images in and out of files,
  like nibabel (`MRImage`, `readimage`, `writeimage`, BIDS sidecars, `.bval`/`.bvec`; MATLAB
  through an extension, ext/MRImagesMATExt.jl, loaded with MAT)
- cli/: the `equip` command (package EQUIPCLI, cli/bin/equip): a subcommand per part, and
  `equip info` (a file's header and sidecar); MAT loaded only when a command names a .mat file.
  DICOM is converted with dcm2niix, not read (docs/design.md)
- docs/design.md: the decision log and the roadmap. Record design decisions, measurements and
  roadmap items there, with the reasons.

## Style

- Readable and heavily commented, so the code can be checked against the papers: a header block
  at the top of each file, comments that say why, equations in the papers' notation, citations
  (author, journal, year, table) for published values.
- Times in seconds, rates in 1/s.
- Naming: "parameters" are the quantities in the signal equation (T1, PSR, M0, B1, ...), fitted
  or fixed; the "acquisition" is the protocol (TI, TE, flip angles, ...), never "parameters".
- Match the surrounding code; explain Julia idioms (dispatch, type parameters, closures) when
  they are not obvious.

## Working agreements

- Ask before starting a long run (benchmarks, large simulations).
- Prototypes go on their own branch (e.g. `varpro-proto`), not on `lean`, until they are
  accepted.
- When refactoring, check that results are unchanged (starts and fits of every model, both
  precisions) and measure speed before and after.
- No pull requests unless asked.
