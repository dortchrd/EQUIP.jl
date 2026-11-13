# EQUIP.jl: notes for Claude

EQUIP (Estimation of QUantitative Imaging Parameters) is a Julia package for voxel-wise MRI
fitting, an alternative to qMRLab: millions of small fits (fewer than 10 parameters, fewer than
100 observations each) with a custom Levenberg–Marquardt solver, ForwardDiff duals and
StaticArrays. The owner is an MRI physicist learning Julia.

## Commands

- Tests, in groups (test/runtests.jl; about 2360 tests in unit + integration):
  - while developing, directly, without Pkg.test's setup: `julia -t auto --project=test
    test/runtests.jl unit` (about a minute), or a file by name (`... runtests.jl crlb`), or
    `integration`, `validation`, `all`. The test environment builds EQUIP without its precompile
    workload (`[preferences.EQUIP]` in test/Project.toml), so a code change costs seconds, not a
    minute; nothing to set up. CI turns the workload back on (CI.yml writes a
    test/LocalPreferences.toml, which takes precedence), to test EQUIP as users install it
  - the full check (unit + integration, as CI runs on every push):
    `julia -t auto --project=. -e 'using Pkg; Pkg.test()'` (about 6 minutes of tests, plus up to
    8 of setup after a change); a group: `Pkg.test(test_args = ["unit"])`
  - the command line's: `julia --project=cli -e 'using Pkg; Pkg.test()'`; MRImages':
    `julia --project=MRImages -e 'using Pkg; Pkg.test()'`
- Examples (4-point SIR, SNR 200, CSV output): `julia -t auto --project=. examples/sir_4point_range.jl`
  (tissue values over the paper's range) or `examples/sir_wm_4point.jl` (white matter only);
  `examples/one_free_parameter.jl` (one parameter fitted, the others known; about 25 s);
  `examples/compartments.jl` (six models written with `@mrimodel` and the compartments, fitted; about 30 s);
  `examples/directional_models.jl` (stick + zeppelin per image, with DTI's direction; about 30 s);
  `examples/dki.jl` (the built-in DKI, its indices and the K ≥ 0 check; about 15 s)
- One example per part (each under a minute): `examples/fit_maps.jl` (files to maps),
  `examples/crlb.jl`, `examples/montecarlo.jl`, `examples/protocol_design.jl` (Dortch 2018 Table 2)
- Benchmark vs LsqFit, NonlinearSolve and FastLevenbergMarquardt over the test cases in bench/cases.jl (`CASES` picks which): `julia -t auto --project=bench bench/solvers.jl`
  (own environment: `julia --project=bench -e 'using Pkg; Pkg.instantiate()'` once)
- Speed of the main paths (fit!, datasets, crlb!, sweeps, montecarlo, a design, fit statistics),
  in the package's own environment: `julia -t auto --project=. bench/speed.jl` (about 1 minute)
- Monte Carlo against crlb (bias and SD, z in Monte Carlo standard errors; T2, Dortch 2018
  Fig. 1, an SNR sweep): `julia -t auto --project=. bench/agreement.jl` (about 25 s)
- First setup: `julia --project=. -e 'using Pkg; Pkg.instantiate()'` (Manifest.toml is not committed),
  and for the direct test command `julia --project=test -e 'using Pkg; Pkg.instantiate()'`
- The first `using EQUIP` after a code change takes about a minute (precompile workload,
  src/precompile.jl: every model's fit except DKI's, crlb, montecarlo, a design; 65 s measured). While developing, switch it off with a `LocalPreferences.toml` (not
  committed) containing `[EQUIP]` and `precompile_workload = false`; about 3 s per change then.

## Where things are

- src/models/interface.jl: the model interface (`MRIModel`, `Free`, `Fixed`, `BestOf`,
  `parameters`, `signal!`, the optional hooks `datastart`, `relabel`, `derived` and `fromderived`, its reverse for `nominal`); src/models/diffusion/compartments.jl: the diffusion
  compartments exported by EQUIP for writing models (`iso`, `stick`, `zeppelin`, with the view
  `powder` or a cosine ξ);
  src/models/macro.jl: `@mrimodel` (a model from a
  function of one observation); src/models/template.jl: `newmodel` (a model file to edit, blank
  or written out from an @mrimodel definition); src/models/*/: the built-in models
  (src/models/diffusion/dti.jl: `DTI`, the diffusion tensor written out, with a log-linear
  start; dki.jl: `DKI` in V = MD²·W, with MK, AK, RK, MKT derived as DIPY's; src/models/diffusion/directions.jl: `fiberdirection`, which with Julia's `⋅` gives a
  directional compartment its view ξ = g ⋅ n; `directionset`, N gradient directions spread
  evenly, to simulate a scheme)
- src/fit/, in pipeline order: setup.jl (`FitSpec`) → starts.jl and loglinear.jl (`:loglinear`)
  → cache.jl (`init`, `FitCache`) → voxel.jl (one voxel) → dataset.jl (maps, threads, `refit`)
  → statistics.jl (`stderror`, `vcov`, `cor`, `residualsd`, AIC/BIC, `onbound`: from the fit's
  own JᵀJ); levmar.jl is the solver, result.jl the `FitResult`
- src/sim/, from true values to data (uses models only): `nominal` and tissue presets
  (`TISSUES`), noise.jl (the noise model: `snr`, `σ` or `Σ` checked once for
  `simulate` and crlb; `noisecovariance`), `simulate`; rician.jl (`ricianmean`, `ricianvar`,
  `ricianinformation`, `RicianMean` (src/models/rician.jl), and `ricianbound`; crlb takes
  `noise = :rician` for a least-squares fit of magnitude data)
- src/errors/, a fit's errors against the truth (bias, SD, RMSE): scenario.jl, the `Scenario` (a truth and the fit judged against it, built
  from the truth with fit's keywords: what crlb and montecarlo take; `crlb(m, θ; kmf = 12.5)`
  makes one); derivatives.jl, crlb's pieces one function each (`CRLBCache`, `noisefree!`,
  `hessian!`, `meat`, `information`, `sandwich`, `biasgradient`, `biasedcrlb`; allocation-free,
  summed row by row from dual numbers), with one-call versions (`jacobian`, `hessian`,
  `informationmatrix`, ...); crlb.jl, the kernel `_crlb!` (their composition), `crlb!` and
  `crlb` (bias, SD and RMSE predicted from derivatives; sandwich or Eq. 3; the model's
  `derived` quantities after the parameters, by the delta method, `_derivedcrlb`);
  montecarlo.jl, `montecarlo` (the same measured by simulating and fitting, with `crlb`
  alongside; derived quantities from each copy's fit); compare.jl, `compare` (the same measured on a dataset fit, e.g. simulated maps);
  diagnostics.jl, `cond`, `cor` and `leakage` (where the errors come from: dθ̂/dc of each
  fixed parameter, from a joint cache with the fixed numbers seeded too); propagate.jl,
  `propagate(c, g)` (crlb's errors for any function of the parameters); sweep.jl:
  crlb or montecarlo over noise levels and true values (`snr = [...]`, `vary = (...)`),
  returning a `Sweep`, and `scenarios` (a set of `Scenario`s: tissues, boxes of true values,
  or both; calcVmax's modes; a `Sweep` whose results are the scenarios), which `crlb` and
  `montecarlo` also take
- src/design/: design.jl, `ProtocolDesign` (the objective of a protocol search, Eq. 5 by
  default, made once from scenarios: one cache per fit, the kernel per scenario, allocation-free);
  search.jl, `design` (differential evolution, then Nelder–Mead, starts on threads;
  the result rounded to the ms by a search of its own)
- src/io/: data files in, maps out, on MRImages: encoding.jl (`diffusionencoding`: a .bval
  and a .bvec as a directional model's b, gx, gy, gz, checked), data.jl (`readdata`,
  `readmap`, `DataInfo`: files stacked into observations, sidecar keys, .bval and .bvec to
  model fields), maps.jl (`writemaps`: BIDS
  names, a provenance sidecar per map; one .mat file for MATLAB data), tables.jl (`writetable`);
  ext/EQUIPMATExt.jl: only a precompile workload for .mat files (loaded with MAT; StaticArrays
  invalidates MRImages' own, docs/design.md)
- MRImages/: a package of its own (in this repository for now): MRI images in and out of files,
  like nibabel (`MRImage`, `readimage`, `writeimage`, BIDS sidecars, `.bval`/`.bvec`; MATLAB
  through an extension, ext/MRImagesMATExt.jl, loaded with MAT)
- cli/: the `equip` command (package EQUIPCLI, cli/bin/equip): a subcommand per part, and
  `equip info` (a file's header and sidecar); MAT loaded only when a command names a .mat file.
  DICOM is converted with dcm2niix, not read (docs/design.md)
- README.md: the front page (short); docs/guide.md: the user's guide, every part in full (what
  the README held until 2026-11-08). A change users see goes in the guide, and in the README
  only if it changes the front page.
- docs/design.md: the decision log and the roadmap. Record design decisions, measurements and
  roadmap items there, with the reasons.
- test/: setup.jl (packages, test models, CASES, `@test_noalloc`, the independent references) and
  the groups: unit/ (models.jl: signals against their equations; roles.jl; starts_solver.jl),
  integration/ (fitting, simulation, crlb, montecarlo_sweeps, noise_weights, scenarios_leakage,
  design, io, diffusion, derived, macro), validation/ (expensive statistical checks, run apart).
  bench/: speed (bench/speed.jl), crlb against Monte Carlo (agreement.jl), other packages
  (solvers.jl); their framework is on the roadmap (docs/design.md, "Tests and benchmarks").
- docs/status.md: where the work stands (branches, what waits on the lab, open questions): the
  map to read first when coming back; design.md has the reasons.

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
- Prototypes go on their own branch (e.g. `varpro-proto`), not on `main`, until they are
  accepted.
- When refactoring, check that results are unchanged (starts and fits of every model, both
  precisions) and measure speed before and after.
- No pull requests unless asked.
- CI is a safety net, not a gate (the owner, 2026-11-09: "do we have to wait for ci every time").
  Docs-only changes: commit and push; CI skips them (CI.yml's paths-ignore). Code changes: the
  file or group being changed while working, the full local suite (Pkg.test) once before merging
  into `main`, then merge and let CI run on `main` in the background, fixing at once if it fails.
  Wait for CI before merging only when a change touches what only CI checks: Julia 1.11 and
  version compatibility (Project.toml, [compat], newer language features), the precompile
  workload, the command line or MRImages, or CI.yml itself. Long local runs go in the background
  while the next item proceeds.

## Tests

- Read the tests of the area (the file in test/unit or test/integration) before adding one; extend
  a test set rather than start a parallel one, and do not repeat what another already covers.
- A real bug fixed gets a regression test that fails without the fix, saying what it guards.
- Check against something independent of the code: an analytic form or a limit, a published value
  (paper, table), central differences, an invariance (scaling, reordering, σ), another package's
  result. A number the implementation produced is a regression value, not a check: say so in a
  comment and where it was checked, or test the property instead.
- One configuration per behavior: test a feature on one representative model and precision unless
  others take a different code path. Loops over many random voxels: one dataset fit and a few
  assertions on it, not an assertion per voxel.
- Error messages: one check per error path, matching the name or value the message is about, not
  its prose.
- Run the file or group being changed while developing (`julia -t auto --project=test
  test/runtests.jl <file or group>`); the full suite (Pkg.test) before committing to a branch to
  be merged; the validation group before merging into `main` when the change touches what it
  checks.
- No timing assertions in tests: speed goes in bench/, compared against a baseline. Allocation
  checks are fine (deterministic): `@test_noalloc`, which skips them when bounds checks are forced
  (Julia 1.11's Pkg.test).
- New expensive statistical checks (Monte Carlo at large n, convergence studies) go in
  test/validation/, listed in runtests.jl's `GROUPS.validation`.
