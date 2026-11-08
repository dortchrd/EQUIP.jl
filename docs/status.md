# Where we are (2026-11-08)

The map to read first when coming back: what is where, and what is left. The reasons and the
measurements behind each item are in docs/design.md (roadmap items by number).

## Branches

Four branches are kept (the owner, 2026-11-08: "we will be left with 4 manageable branches"):

| branch | what it is |
|---|---|
| `main` | the work. GitHub's default branch; on 2026-11-08 moved from the first version to `lean`'s fae57ba (a fast-forward), so the repository's front page shows EQUIP. New work starts here; prototypes on a branch of their own, made from `main`. |
| `lean` | kept for now, at fae57ba, where `main` started; it holds nothing `main` does not. Its history is below. Whether to delete it: the owner, later. |
| `claude-oneshot` | the first version (VoxelFit, 2026-10-14, bd6f278): what `main` was until 2026-11-08. |
| `experiments` | the four prototypes never merged, in one commit (bc996fe) whose four parents are their last commits, so every commit of each is kept. Its only file is a README: what each tried, its outcome, and how to get one back (`git switch -c step-proto experiments^2`). |

Open: `prerelease` (made from `main` on 2026-11-08): tier 1 below, for the owner to look at,
not merged.

The four in `experiments` (their outcomes are in design.md):

| parent | was branch | outcome |
|---|---|---|
| 1 | `varpro-proto` | VarPro on the biexponential cases: not faster as prototyped; a design for later ("Variable projection (VarPro): proposed design") |
| 2 | `step-proto` | step-control variants for the solver: today's step control kept |
| 3 | `sens-proto` | `sensitivity` maps: built, then removed (Eq. 6 is `jacobian ./ √t`; its plot on the roadmap) |
| 4 | `prototype/solver-deadzone` | steps toward bounds: superseded by the solver's dead-zone rule (`_lost_influence`) |

**Deleted on GitHub by the owner, 2026-11-08** (every commit on them is in `main` or in
`experiments`): 45 branches. The four experiments, now in `experiments`: `varpro-proto`
`step-proto` `sens-proto` `prototype/solver-deadzone`. The chain merged into `lean` (below):
`smt-proto` `nerve-smt` `blocks-proto` `directions-proto` `loglinear-proto` `noinline-proto`
`derived-errors` `derived-truths` `derived-sweeps` `loglinear-time` `roadmap-priorities`. And
the older ones: `aicc-fix` `analysis-proto` `api-v0.2` `averaging`
`claude/sweet-thompson-ctbc4v` `cli-plan` `cli-proto` `crlb-proto` `design-proto` `diag-proto`
`float32-ftz` `kwargs-proto` `leakage-proto` `mrimages-proto` `mrimodel-proto` `names-proto`
`newmodel-proto` `nobs-proto` `noise-choice` `noise-proto` `org-proto` `result-proto`
`scen-proto` `scenario-proto` `set-proto` `sim-precision` `simulate-fast` `stats-proto`
`trim-proto` `weights-proto`. A branch named in this file or in design.md is in `main`'s history,
or, for the four experiments, in `experiments`'s; any commit hash there still works (`git switch -c
name <hash>`).

**Proposed, for the owner (not adopted):** one working branch at a time, made from `main`,
merged into `main` when the owner has seen it, then deleted; no more chains.

### How `lean` (now `main`) got here

`lean` was fast-forwarded on 2026-11-07, at the owner's request, from 5414c60 (2026-11-03) to the
tip of the chain below, `derived-sweeps`. (Earlier the same day the chain had been merged into
`lean` without the owner asking, and `lean` was reset to 5414c60; that merge commit, f3fd99d, is
in the history the fast-forward brought.)

What came in, oldest first (each branch built on the one before):

| branch | what it added |
|---|---|
| `smt-proto` | SMT, SMTFreeWater, SMTT2 |
| `nerve-smt` | Rician noise (`ricianmean`, `RicianMean`, `riciannoise`, `ricianbound`), `directionset`; the spherical mean's pipeline errors and the floor studies (taken out again on `directions-proto`, at the owner's request: roadmap item 18) |
| `blocks-proto` | the diffusion compartments; one-equation model families (undone on `directions-proto`, at the owner's request) |
| `directions-proto` | `diffusionencoding`, `fiberdirection`, `readbvec`; every model back in the standard format; the built-in `DTI` |
| `loglinear-proto` | `Free(start = :loglinear)`; the built-in `DKI` with MK, AK, RK, MKT as DIPY's; the `derived` hook |
| `noinline-proto` | crlb at many parameters (DKI's 22) |
| `derived-errors` | derived quantities in crlb and montecarlo (`c.sd.FA`) |
| `derived-truths` | `fromderived` and `nominal(DTI; MD, FA, direction)`; eigen derivatives at equal eigenvalues |
| `derived-sweeps` | tissue presets by derived quantities (`DKI_...`, Lätt et al. 2013); the truth by derived quantities in `Scenario`, `simulate`, `crlb`, `montecarlo`; sweeps over one (`vary = :FA => ...`) |

Then, fast-forwards on 2026-11-08 at the owner's request: `loglinear-time` (T2Decay on the shared
log-linear start, no `datastart` of its own; `:loglinear` for time constants; the start's first
pass made once per fit; the voxel fit called as a function above 8 free parameters: DTI −16%,
DKI −29%, T2Decay +14–28% at typical T2, converging everywhere) and `roadmap-priorities` (docs
only: the priority list below). Last full runs: EQUIP 2341, CLI 124, MRImages 103.

## Priorities (agreed with the owner, 2026-11-08)

In order. The reasons, measurements and plans are in design.md (roadmap items by number; its
"Priorities" paragraph has the same list).

**1. Finish the prerelease**: done on branch `prerelease` (2026-11-08), for the owner to look
at, not merged (design.md, "The prerelease: what was done", has the measurements)
- T2Decay's T2 at most 2.5 s (the owner): voxels with no decay end on the bound, flagged by
  `onbound`, instead of "converged" at about 1e10 s; tissue and CSF at 32 echoes unchanged.
- `equip fit DTI` and `equip fit DKI`: the directions from the `.bval` and `.bvec` beside the
  data; the derived maps written.
- The precompile workload: DTI, crlb, montecarlo, a design (their first calls 1–3 s → under
  0.3 s; precompiling 46 → 65 s). DKI left out: from the package image its fit ran 20–70%
  slower. Open: QMTSIR, precompiled since before, measured 12–27% slower the same way.
- MIT license; versions EQUIP 0.3.0, EQUIPCLI and MRImages 0.1.0; CI (.github/workflows).
- An example per part (fit_maps, crlb, montecarlo, protocol_design); the README a front page,
  its full text in docs/guide.md.
- Shipped without pulsed qMT and NODDI (the owner); `python/` was not in the repository.
- Full runs: EQUIP 2347, CLI 139, MRImages 103. CI's first run: MRImages and the CLI pass on
  Julia 1.11 and 1.13; EQUIP failed on both, from older causes in two tests, both fixed: the
  flush test's probe (a lock), and on 1.11 the zero-allocation checks under the bounds checks
  1.11's `Pkg.test` forces (`fit!` allocates nothing on 1.11 as users run it; the checks are
  skipped when bounds checks are forced, `@test_noalloc`).

**2. Bias from noise at low SNR** (items 13, 16, 20)
- Rician fitting for every model (items 13 and 20, one item): `fit(m, data; noise = :rician,
  σ)`, the Gaussian fit first, σ as a map. Measured for DTI and DKI (MD 11% low at SNR 10), not
  built.
- crlb's second-order noise bias, both terms, opt-in, tested against montecarlo: Box's bias of
  the parameters, and a derived quantity's ½ tr(∇²g·C).
- `propagate(c, g)` for any function of the parameters (the `derived` hook covers a model's
  own).
- Uncertain inputs: a measured B1 map's SD added to the covariance.

**3. Protocol design** (items 3, 14, 19)
- A protocol from a number of points (item 14).
- An objective on a derived quantity (`ProtocolDesign` for MK's SD, not W1111's).
- The search's crlb and leakage from one kernel; the worst case inside a box.

**4. Finish derived quantities** (item 19)
- Standard-error maps from the fit's own covariance (`savemaps` writing `FA_SE`).
- `compare` (a dataset fit against truth maps) for derived quantities.
- An angular error for the principal direction (the cone of uncertainty) in place of V1's
  components, whose statistics break near the hemisphere's fold.
- `vary` over derived quantities in `scenarios()` (sweeps already take them).
- Derived quantities for the other models: qMT's kfm = kmf·PSR, T1 from R1, SMT's D⊥.

**5. Models** (items 5, 6, 7, 17)
- First, in this order (the owner): stick–zeppelin (examples/directional_models.jl has it by
  hand), ball–stick (one stick; N sticks later), IVIM (the segmented fit as its `datastart`).
  The two directional ones start from DTI's principal direction.
- Then: MTR, MTsat, B1 and B0 maps (item 5); pulsed qMT (item 6); NODDI (item 7); free-water
  DTI; DKI's constraints (K ≥ 0 in every direction) imposed, not only checked.

**6. Starts, across all models** (item 4; after the models, the owner: "let's build all models
first")
- Plausible ranges per model, from the presets and the literature.
- Every model's start tested over its plausible range, both precisions; outside it, tested only
  for failing safely (flagged, never "converged" at nonsense).
- The dictionary start: `BestOf` over several parameters, the candidate signals made once per
  fit (the shared first pass is already that for models linear in ln S).
- An example `datastart` (DESPOT1's linearization, IVIM's segmented fit).

**7. Nerve SMT, when the lab sends its values** (item 18; below)
- The protocol evaluator and the search (steps 3 and 4).
- The spherical mean's pipeline errors: taken out of the code on 2026-11-06; the code, tests and
  floor studies are in `main`'s history at commit a9c7be4 (`nerve-smt`'s last), to restore
  when this resumes.

**Recorded, not ranked** (design.md has each)
- DKI's other cheap indices, RTK, KFA, Kmax, and DIPY's numerical MK, AK, RK as an option (item
  19).
- Bayesian estimation (15); VarPro, log coordinates, solver backends, GPU (11, 12).
- The other start changes: ADCKurtosis and T2DecayOffset on `:loglinear` (the owner: the offset
  not now), the second pass from the linear prediction, `BestOf(fits = k)`, neighbours as a
  second pass (item 4). R2Decay and ADCDecay keep their constant starts (measured:
  `:loglinear` made them slower, 2026-11-08).
- The B-matrix for b-tensor data, `rotinv` and Watson views, cylinder and sphere compartments,
  N sticks (item 17).
- Phantoms and BrainWeb (2); the LsqFit bridge and a "coming from" page (8); formats, protocol
  files, MRImages in its own repository and with it Julia 1.10 support (9); `--init`, a server, the Python and MATLAB front
  ends, an installer (10); IRWLS and sensitivity–efficiency plots (3); myelin water, DCE, the
  sinc gradient echo (12).

**The owner's own**
- The remote has moved (github.com/dortchrd/voxelfit.jl → dortchrd/EQUIP.jl): update the
  remote URL in clones (`git remote set-url origin https://github.com/dortchrd/EQUIP.jl.git`).
- Later: whether to delete `lean`; adopt or not one working branch at a time.
- A source with AK, and with cortical GM and CSF, for the `DKI_` presets (Lätt et al. 2013 has
  neither).

## Waiting on the lab (nerve SMT steps 3 and 4: the protocol evaluator, then the search)

- nerve v_ax, D_ax, compartment T2s, T1; the nerve's dispersion
- the b = 0 SNR, the shortest TE at each b, the time per slice, the number of slices, today's
  protocol
- complex or magnitude averaging; whether the scanner can export complex (or phase) images
- the registration SNR floor, and of which image
- the complex- and magnitude-averaged nerve DTI pair, to measure the floor in the lab's own data
