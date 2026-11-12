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

Merged into `main` (fast-forward, 2026-11-12, at the owner's request: "yes, run the full tests
and merge"): `rician`, 17 commits: tier 2 steps 0, 1, 2 and 4, 5 (the Rician fits, and crlb,
montecarlo and sweeps for them), the diffusivity, T1 and DTI/DKI-diagonal bounds, `design` (was
`optimize_protocol`), crlb's `noise = :rician`, `readdata`/`readmap`/`writemaps`, the
`:loglinear` warning and src/fit/loglinear.jl, the docstrings attached. Checked first: the full
`Pkg.test` (EQUIP 2497), the command line's (139), and CI on the merged commit, 1c38e11 (all six
jobs: EQUIP, the command line and MRImages on Julia 1.11 and 1). The branch can carry the rest
of tier 2 (steps 3, 6 to 9) or be deleted.

Merged into `main` (fast-forward, 2026-11-08, at the owner's request): `prerelease`, tier 1 below;
its branch is safe to delete on GitHub.

Merged into `main` (fast-forward, 2026-11-09, at the owner's request: "yes to both"): `t2-bounds`,
T2 ≤ 2.5 s and R2 ≥ 0.4 s⁻¹ in every T2 model, `onbound` fixed for Float32, and the presets
`DKI_GM`, `DKI_WM` (Maiter et al. 2021) and `DKI_CSF`; its branch is safe to delete on GitHub.

Merged into `main` (fast-forward, 2026-11-08, at the owner's request: "looks good"): `test-structure`,
the tests split into groups (test/unit, test/integration, test/validation), CLAUDE.md's rules for
tests, the audit's items on the roadmap (design.md item 21); its branch is safe to delete on GitHub.

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

**1. Finish the prerelease**: done (2026-11-08, branch `prerelease`, merged into `main`;
design.md, "The prerelease: what was done", has the measurements)
- T2Decay's T2 at most 2.5 s (the owner): voxels with no decay end on the bound, flagged by
  `onbound`, instead of "converged" at about 1e10 s; tissue and CSF at 32 echoes unchanged.
  Extended (2026-11-09, branch `t2-bounds`): T2 ≤ 2.5 s in every T2 model and R2 ≥ 0.4 s⁻¹ in
  every R2 model, both forms alike; and `onbound` fixed for Float32 fits.
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

**2. Bias from noise at low SNR** (items 13, 16, 20). Planned (2026-11-09), not built: design.md,
"Tier 2, bias from noise at low SNR: the plan", ten steps, each with how it is judged. Decided
(the owner, 2026-11-09): the keyword is `likelihood`; MP-PCA waits, with the other
preprocessing (item 22); the diffusivity cap, 4e-3 mm²/s, in every fit ("cap everywhere").
Done on branch `rician` (2026-11-09 to 12; merged into `main` 2026-11-12): step 0, the baseline (bench/rician_baseline.jl;
design.md, "Step 0, measured"), and the cap (ADCDecay, ADCKurtosis, DTI's and DKI's diagonal;
Gaussian fits in tissue unchanged; the Rician fit in CSF 98–100% converged with it, 70–94%
without). What step 0 found: the Rician mean removes the floor's bias in tissue from SNR 10;
weighting it gains under 2.4% of SD, so it is not the gap; the gaps are the bias at SNR ≤ 10 and
in DKI's MKT (overcorrected, −13% at SNR 10), convergence where there is no information, and
speed (4–80× the Gaussian fit). Then (the owner: "make T1 cap 5 sec"): T1 ≤ 5 s and R1 ≥ 0.2
s⁻¹ in every T1 model (T1IR in CSF had run off, +290% at SNR 10); DTI's and DKI's diagonal
from 0 (a diffusivity is not negative: the Rician fits of DKI in CSF no longer run negative). Step 1 (the owner: "do step
1"): the Rician functions fast, in src/models/rician.jl (Chebyshev Bessel functions fitted to
`besselix` at precompile time, their exact derivatives on dual numbers, R(s) tabulated): a fit
of the Rician mean 6.5–28× faster, now 1–4× the Gaussian fit's cost; results unchanged (Float64
to 1e-8, Float32 within its own tolerance); `RicianMean` gives the model's derived quantities.
Then steps 2 and 4 together (the owner: "both now, ML the default"): `fit(m, data; likelihood
= :rician, σ, method = :ml | :mean | :wmean)`, σ a number or a map; least-squares fits
unchanged value for value. Measured (design.md, "Steps 2 and 4, built and measured"): the three
methods agree where the data hold information and remove the floor's bias from SNR 10; `:ml`
converges best where fits are hard, at `:mean`'s cost; at SNR 5 the Rician fits trade bias for
variance; DKI's MKT overcorrection is not the noise model's (every method alike). Step 5 (the
owner: "do 5"): crlb, montecarlo and sweeps take `likelihood = :rician` and `method` (a
Scenario carries them) and predict or measure that fit; each method is, to first order, a
least-squares fit crlb's kernel already computes (`:ml` the magnitude weighted by the Rician
information: `ricianbound` when nothing is wrong). Checked against montecarlo (T2 decay at SNR
20; the 5-point SIR with kmf wrong at SNR 200). Not yet: a protocol design for these fits,
`leakage`, the command line, `weights`, averages. Found while answering the owner (2026-11-09): least squares on the
Rician mean weighted by 1/Var is within 4.5% of the Rician bound's SD at any SNR
(asymptotically), so maximum likelihood (step 4) is to be decided on the measurements.
2026-11-12 (names, item 24): crlb takes `noise = :rician` for least squares on magnitude data
(`riciannoise` removed); `readdata`, `readmap`, `writemaps` (were `loaddata`, `loadmap`,
`savemaps`); `:loglinear` kept, and `init` warns (once) where ln S is not linear in those
parameters, so the start is not the log-linear fit. Full runs on the branch: EQUIP 2496, CLI 139
(MRImages unchanged, not rerun).
- Rician fitting for every model (items 13 and 20, one item): `fit(m, data; likelihood
  = :ricianmean or :rician, σ)` (least squares on E|y|, then maximum likelihood), the Gaussian
  fit first, σ a fixed parameter so a map works as B1's does. Measured for DTI and DKI (MD 11%
  low at SNR 10).
- crlb's second-order noise bias, both terms, opt-in, tested against montecarlo: Box's bias of
  the parameters, and a derived quantity's ½ tr(∇²g·C).
- `propagate(c, g)` for any function of the parameters (the `derived` hook covers a model's
  own).
- Uncertain inputs: a measured B1 map's SD added to the covariance.
- With it, from the test audit (design.md item 21): the missing independent checks (the T1
  signals' analytic values, invariances, every model's Jacobian and DKI's index gradients against
  central differences, edge cases), and test/validation/ filled with the statistical checks
  (bench/agreement.jl, the Rician fits'), run nightly in CI.

**3. Protocol design** (items 3, 14, 19)
- A protocol from a number of points (item 14).
- An objective on a derived quantity (`ProtocolDesign` for MK's SD, not W1111's).
- The search's crlb and leakage from one kernel; the worst case inside a box.

**4. Finish derived quantities** (item 19)
- Standard-error maps from the fit's own covariance (`writemaps` writing `FA_SE`).
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
- From the test audit (design.md item 21): the tests consolidated area by area, when work
  touches an area (per-voxel loops into dataset fits, one error check per error path,
  regression values traced, one configuration per behavior); the benchmark framework
  (benchmark/, a BenchmarkTools suite with a baseline and first-call times) before performance
  work.
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
  ends, an installer (10); preprocessing: MP-PCA's denoising and σ maps, median–Otsu masks,
  Gibbs-ringing removal, perhaps filtering (22); a units switch for what users see (ms, 1/s,
  µm²/ms: the owner's preference; inside, the units stay s, 1/s, s/mm², mm²/s) (23); IRWLS and
  sensitivity–efficiency plots (3); a review of the exported names (24): `optimize_protocol`
  renamed `design` (2026-11-09, the owner's choice; no alias); 2026-11-12: crlb's Rician noise
  by keyword (`noise = :rician`, `riciannoise` removed), the file verbs read/write, crlb's
  pieces' exports left for the release, `compare` left; myelin water, DCE, the
  sinc gradient echo (12).

**The owner's own**
- The remote has moved (github.com/dortchrd/voxelfit.jl → dortchrd/EQUIP.jl): update the
  remote URL in clones (`git remote set-url origin https://github.com/dortchrd/EQUIP.jl.git`).
- Later: whether to delete `lean`; adopt or not one working branch at a time.
- Done (2026-11-09): `DKI_GM` and `DKI_WM` with AK (Maiter et al., Sci Rep 2021;11:8857, found
  at the owner's request) and `DKI_CSF` as free water; Lätt's regional presets still take AK in
  the call.

## Waiting on the lab (nerve SMT steps 3 and 4: the protocol evaluator, then the search)

- nerve v_ax, D_ax, compartment T2s, T1; the nerve's dispersion
- the b = 0 SNR, the shortest TE at each b, the time per slice, the number of slices, today's
  protocol
- complex or magnitude averaging; whether the scanner can export complex (or phase) images
- the registration SNR floor, and of which image
- the complex- and magnitude-averaged nerve DTI pair, to measure the floor in the lab's own data
