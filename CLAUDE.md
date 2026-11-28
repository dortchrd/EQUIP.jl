# EQUIP.jl: notes for Claude

EQUIP (Estimation of QUantitative Imaging Parameters) is a Julia package for voxel-wise MRI
fitting, an alternative to qMRLab: millions of small fits (fewer than 10 parameters, fewer than
100 observations each) with a custom Levenberg–Marquardt solver, ForwardDiff duals and
StaticArrays, and the errors of those fits predicted before scanning, for protocol design. The
owner is an MRI physicist learning Julia.

## Read first

- **docs/status.md**: where the work stands: the branches (which one to start on), the work in
  progress and its next steps, what waits on the owner. Read it before anything else.
- **docs/architecture.md**: the code as it is: the layers and where each part lives, the model
  interface, the built-in models and their branches, the rules the code keeps, its numerical
  accuracy, its speed.
- **docs/design.md**: why things are as they are: the decision log and the roadmap (items 1–31).
  Too long to read whole: its index at the top gives each section's and item's status, what was
  tried and not adopted (read it before proposing one of those again), and the decisions that
  constrain new work.
- docs/guide.md: the user's guide (README.md is the front page); docs/history.md: what was
  merged when, and what each priority tier built; the `experiments` branch's README: the
  archived prototypes.

Where to record what: a decision, a measurement or a roadmap item, with its reasons, in
design.md (the item it belongs to); the state of the work, next steps and open questions in
status.md; a change to the code's structure in architecture.md; what users see in guide.md (and
README.md only if the front page changes); a finished merge in history.md. Say a thing in one
place and point to it from the others.

## Commands

- Tests, in groups (test/runtests.jl; about 3550 tests in unit + integration at the last full
  run):
  - while developing, directly, without Pkg.test's setup: `julia -t auto --project=test
    test/runtests.jl unit` (about 5 minutes in a cloud container, 2026-11-28: the models file
    about 3 of them), or a file by name (`... runtests.jl crlb`), or `integration`,
    `validation`, `all`. The test environment builds EQUIP without its precompile workload
    (`[preferences.EQUIP]` in test/Project.toml), so a code change costs seconds, not a minute;
    nothing to set up. CI turns the workload back on (CI.yml writes a test/LocalPreferences.toml,
    which takes precedence), to test EQUIP as users install it
  - the full check (unit + integration, as CI runs on every push):
    `julia -t auto --project=. -e 'using Pkg; Pkg.test()'` (plus up to 8 minutes of setup after
    a change); a group: `Pkg.test(test_args = ["unit"])`
  - the command line's: `julia --project=cli -e 'using Pkg; Pkg.test()'`; MRImages':
    `julia --project=MRImages -e 'using Pkg; Pkg.test()'`
  - the validation group (expensive statistical checks, run apart; CI runs it nightly):
    `julia -t auto --project=test test/runtests.jl validation` (about 5 minutes), or a file:
    `agreement` (Monte Carlo against crlb, z in Monte Carlo standard errors), `rician_fits`,
    `secondorder`
- First setup: `julia --project=. -e 'using Pkg; Pkg.instantiate()'` (Manifest.toml is not
  committed), and for the direct test command `julia --project=test -e 'using Pkg;
  Pkg.instantiate()'`
- The first `using EQUIP` after a code change takes about a minute (the precompile workload,
  src/precompile.jl; 65 s measured). While developing outside the tests, switch it off with a
  `LocalPreferences.toml` (not committed) containing `[EQUIP]` and `precompile_workload = false`;
  about 3 s per change then.
- Examples, one per part, each under a minute: examples/fit_maps.jl (files to maps), crlb.jl,
  montecarlo.jl, protocol_design.jl (Dortch 2018 Table 2), sir_4point_range.jl and
  sir_wm_4point.jl (4-point SIR, CSV output), one_free_parameter.jl, compartments.jl,
  directional_models.jl, dki.jl, mtr_design.jl (on `mtr`): `julia -t auto --project=.
  examples/NAME.jl`
- Benchmarks (ask first: Working agreements, below): the main paths' speed,
  `julia -t auto --project=. bench/speed.jl` (about 1–2 minutes); against LsqFit, NonlinearSolve
  and FastLevenbergMarquardt, `julia -t auto --project=bench bench/solvers.jl` (its own
  environment: `julia --project=bench -e 'using Pkg; Pkg.instantiate()'` once); by area,
  bench/qmt/, bench/noddi/, ... (each file's header says how to run it and what it measured)

## Style

- Readable and heavily commented, so the code can be checked against the papers: a header block
  at the top of each file, comments that say why, equations in the papers' notation, citations
  (author, journal, year, table) for published values.
- Forward models self-contained (the owner, 2026-11-27): a model's equations written in its own
  `signal!` (and `derived`), readable there against the paper, not behind helper functions. A helper
  only for what is genuinely shared: pieces several models use and their derived quantities too
  (DTI's and DKI's tensor weights and `_tensorvector`), a mathematical tool (`inner`, the 2×2
  exponential `_expm2`), a tie's function, or the acquisition's construction (`_pulsedspgr`).
  For qMT especially: minimize helpers; a short formula written out in each of two models beats
  one shared through a helper.
- Units: times in seconds, rates in 1/s, b-values in s/mm², diffusivities in mm²/s, flip angles
  in degrees, MT offsets and B0 in Hz, RF amplitudes (B1e, B1cw) in µT; B1 relative (1 nominal).
- Fractions as fractions, 0 to 1, never percent (the owner, 2026-11-27: "fraction for all MTR,
  PSR, BPF, f in diffusion ... the standard across all of equip"): MTR, MPF, volume fractions
  (f, fiso, vax, ...); PSR, a ratio, likewise. Percent only in printed summaries.
- Dot products as `inner(a, b)` (EQUIP's, src/inner.jl: in order, inlined; LinearAlgebra's `⋅` is
  1.75–3× slower on dual numbers of many partials, docs/design.md item 30), never `⋅` or
  `dot(a, b)` in EQUIP's code; tests may use LinearAlgebra's `⋅` as an independent reference.
  Quadratic forms as `g' * D * g` (gᵀDg).
- Naming: "parameters" are the quantities in the signal equation (T1, PSR, M0, B1, ...), fitted
  or fixed; the "acquisition" is the protocol (TI, TE, flip angles, ...), never "parameters".
  Values by name are named tuples (`(B1 = 0.05,)`), everywhere.
- Match the surrounding code; explain Julia idioms (dispatch, type parameters, closures) when
  they are not obvious.

## Working agreements

- Ask before starting a long run (benchmarks, large simulations).
- Keep it simple and tied to the goal (the owner, 2026-11-27: "i fear you are going into
  overengineering territory. remember goals and keep it simple"): the smallest change that serves
  the goal; no new framework or API until the same need has come up twice. Discuss the
  architecture of a larger piece before implementing it, and say why a technical choice is made
  (a numerical method, a data structure) before writing the code, not after.
- Prototypes go on their own branch (e.g. `varpro-proto`), not on `main`, until they are
  accepted. Merges only when the owner says (held since 2026-11-27).
- When refactoring, check that results are unchanged (starts and fits of every model, both
  precisions) and measure speed before and after (the standards: docs/architecture.md,
  "Numerical accuracy").
- No pull requests unless asked.
- CI is a safety net, not a gate (the owner, 2026-11-09: "do we have to wait for ci every time").
  Docs-only changes: commit and push; CI skips them (CI.yml's paths-ignore). Code changes: the
  file or group being changed while working, the full local suite (Pkg.test) once before merging
  into `main`, then merge and let CI run on `main` in the background, fixing at once if it fails.
  Wait for CI before merging only when a change touches what only CI checks: Julia 1.11 and
  version compatibility (Project.toml, [compat], newer language features), the precompile
  workload, the command line or MRImages, or CI.yml itself. Long local runs go in the background
  while the next item proceeds.
- When work ends or a session is about to be replaced, update docs/status.md (and the other
  documents, as "Read first" says where), so a new session can start from them alone.

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
- A new built-in model goes in test/setup.jl's `CASES` (true values for every parameter): that
  gives it the noise-free fits in both precisions, the length checks, `fit!`'s allocation and
  inference checks.
- Error messages: one check per error path, matching the name or value the message is about, not
  its prose.
- Pitfalls met: `@test a ≈ b rtol = x` takes one comparison (with `&&` it is a parse error: split
  it, or `isapprox(a, b; rtol)`); a model's or a result struct's `==` compares identity, not
  values (compare fields); arrays with NaN compare with `isequal`; EQUIP does not export
  `sidecarpath` (`EQUIP.MRImages.sidecarpath`); every vector field of a model must have one value
  per image (a pulse shape is held as an `RFPulse`, as QMTSIR's pulse); a new qMT model joins the
  unions `_MTMODELS` (mtnames.jl) and `_MTRMODELS` (mtr.jl); central differences on a signal made
  by large matrix exponentials (QMTSPGRPortnoy's) need steps of about 1e-4 of the value, not
  1e-6: the exponential's rounding (2.5e-13 at 100 kHz) swamps a smaller step's differences.
- Run the file or group being changed while developing (`julia -t auto --project=test
  test/runtests.jl <file or group>`); the full suite (Pkg.test) before committing to a branch to
  be merged; the validation group before merging into `main` when the change touches what it
  checks.
- No timing assertions in tests: speed goes in bench/, compared against a baseline. Allocation
  checks are fine (deterministic): `@test_noalloc`, which skips them when bounds checks are forced
  (Julia 1.11's Pkg.test).
- New expensive statistical checks (Monte Carlo at large n, convergence studies) go in
  test/validation/, listed in runtests.jl's `GROUPS.validation`.
