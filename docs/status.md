# Where we are (2026-11-17)

The map to read first when coming back: what is where, and what is left. The reasons and the
measurements behind each item are in docs/design.md (roadmap items by number).

## Branches

Four branches are kept (the owner, 2026-11-08: "we will be left with 4 manageable branches";
again 2026-11-17, after deleting the finished ones: "kept rest"):

| branch | what it is |
|---|---|
| `main` | the work. GitHub's default branch; on 2026-11-08 moved from the first version to `lean`'s fae57ba (a fast-forward), so the repository's front page shows EQUIP. New work starts here; prototypes on a branch of their own, made from `main`. |
| `lean` | kept for now, at fae57ba, where `main` started; it holds nothing `main` does not. Its history is below. Whether to delete it: the owner, later. |
| `claude-oneshot` | the first version (VoxelFit, 2026-10-14, bd6f278): what `main` was until 2026-11-08. |
| `bingham-proto` | `BinghamNODDI` (2026-11-23, the owner: "build it on a branch"): NODDI with the Bingham distribution, dmipy's parameterization (ODI, betafraction, the fan's angle psi); tested against NODDI at β = 0, a 2-D quadrature over the sphere, the derivatives at the bounds, fits against crlb; a whole brain in 85 s on 4 threads. The full suite (3237), the command line's (182) and CI pass. Waits on the owner: Tariq 2016's index definitions (ODIfan, DA) to check against the paper, which this container could not read; the merge. |
| `experiments` | the prototypes never merged, kept: an archive commit per archiving whose parents are their last commits, so every commit of each is kept (2026-11-08, bc996fe: varpro-proto, step-proto, sens-proto, prototype/solver-deadzone; 2026-11-23, 4b47226, with bc996fe its first parent: stopping-proto, geodesic-proto, combined-proto). Its only file is a README: what each tried, its outcome, and how to get one back (`git switch -c stopping-proto experiments^2`). |

Merged into `main` (fast-forward, 2026-11-23, at the owner's request: "merge into main"):
`models-proto`, four built-in diffusion models: `StickZeppelin` and `BallStick` (multi-compartment,
src/models/diffusion/multicompartment.jl: the direction from the voxel's own tensor, reported with
z ≥ 0), `NODDI` (Watson-dispersed sticks and zeppelin with a ball; the sticks by their Jacobi series
to b·D∥ = 12.75 and quadrature above; a whole brain in 12–15 s on 4 threads) and `IVIM` (the
segmented start), in the command line and both precompile workloads. Checked first: the full
`Pkg.test` (3143), the command line's (180), and CI on the last code commit, 51d0ed8 (all six jobs).
Not yet tried on the owner's data.

Merged into `main` (2026-11-23, at the owner's request: "merge both into main"): `nii-default`
(a fast-forward), maps gzipped as the data are (`.nii` maps for `.nii` data, `.nii.gz` for
`.nii.gz`; `compress`, `--compress yes/no`, `'compress'` choose), and `nifti-units` (a merge),
MRImages reading headers whose units are unknown (taken as mm, as FSL and ITK do; `equip info`
had failed on the owner's SIR data). Checked first: CI on each branch's last commit (all six
jobs), and each branch's tests locally (the full `Pkg.test`, 3013, and the command line's, 170;
MRImages', 109); and the merged tree, after the push: the full `Pkg.test` (3015), the command
line's (170), MRImages' (109).

Merged into `main` (fast-forward, 2026-11-21, at the owner's request: "merge"): `freed-bounds`,
`Fixed(value; lower, upper)`, the bounds a fixed parameter gets when a fit frees it: QMTSIR's and
QMTSIREqualR1's kmf ≥ 0 (before, 10 to 82 voxels in 10,000 ended at a negative exchange rate with
`kmf = Free()`), T1MFA's B1 ≥ 0, SMTFreeWater's Diso. Checked first: the full `Pkg.test` with the
precompile workload (its two failures, the test's own `@mrimodel` copy of QMTSIR, then fixed), the
command line's (167), the validation group (76), and CI on the branch's last commit, 0ca3ef4 (all
six jobs).

Merged into `main` (fast-forward, 2026-11-21, at the owner's request: "merge into main once CI
passes"): `qmtsir-simple`, `QMTSIREqualR1`, SIR with R1m = R1f and both pools saturated at the
start of TD, its signal in one line (named by the owner: "rename it QMTSIREqualR1"; the branch
kept its name). Checked first: the full `Pkg.test` with the precompile workload (EQUIP 3002), the
command line's (166), and CI on the branch's last commit, a26ffee (all six jobs).

**Deleted on GitHub by the owner, 2026-11-23**, each first merged into `main` or archived in
`experiments` (checked commit by commit): `models-proto`, `nii-default`, `nifti-units`,
`qmtsir-simple`, `freed-bounds` (in `main`); `stopping-proto`, `geodesic-proto`, `combined-proto`
(in `experiments`, 4b47226). Kept: `main`, `lean`, `claude-oneshot` (the owner: "keep lean and
oneshot"), `bestof-proto` (on hold), `experiments`.

**Deleted on GitHub by the owner, 2026-11-17**, each merged into `main` first (every commit is in
`main`'s history, checked on the full history): `rician` (tier 2), `matlab` (the MATLAB
functions), `sidecar-roles` (every parameter in the maps' sidecar), `matlab-fixes`. Work goes on
`main`, or on a branch of its own for a prototype, deleted once merged. (This session's GitHub
access cannot delete branches: the owner deletes them, `git push origin --delete NAME`.)

Merged into `main` (fast-forward, 2026-11-12, at the owner's request: "yes, run the full tests
and merge"): `rician`, 17 commits: tier 2 steps 0, 1, 2 and 4, 5 (the Rician fits, and crlb,
montecarlo and sweeps for them), the diffusivity, T1 and DTI/DKI-diagonal bounds, `design` (was
`optimize_protocol`), crlb's `noise = :rician`, `readdata`/`readmap`/`writemaps`, the
`:loglinear` warning and src/fit/loglinear.jl, the docstrings attached. Checked first: the full
`Pkg.test` (EQUIP 2497), the command line's (139), and CI on the merged commit, 1c38e11 (all six
jobs: EQUIP, the command line and MRImages on Julia 1.11 and 1). The branch can carry the rest
of tier 2 (steps 6 to 9; step 3, estimating σ, is on hold: the owner, 2026-11-12, "let's hold
off on different ways to estimate noise sd", σ given by the user meanwhile) or be deleted. (It
carried them, and was deleted on 2026-11-17, merged.)

Merged into `main` (fast-forward, 2026-11-17, at the owner's request: "merge matlab-fixes into
main once CI passes"): `matlab-fixes`, `equip fit|crlb|montecarlo|design --help` without a model,
MATLAB's output without terminal codes and Julia's binary called directly; and a test made honest
(directionset's second moment near I/3, 1e-4, not 1e-6: it failed on Julia 1.11.9 once
ForwardDiff 1.5.0 changed what was compiled; the set moves with rounding at that level). CI on the
branch, all six jobs. Billing fixed (2026-11-17): CI runs again.

Merged into `main` (fast-forward, 2026-11-15, at the owner's request: "yes run it and merge
into main"): `sidecar-roles`, the maps' sidecar recording every parameter as the fit used it
("Parameters", "ParameterNames"; the model's defaults included), and `equip_fit` printing them and
returning `info.parameters` and `info.sidecar`. Checked first: the full `Pkg.test` (EQUIP 2574),
the command line's (160), Octave. CI could not run: GitHub refused to start its jobs ("recent
account payments have failed or your spending limit needs to be increased": the month's Actions
minutes for a private repository), so it waits for the billing; by CLAUDE.md's rule the change
(writemaps, MATLAB) did not need CI first.

Merged into `main` (fast-forward, 2026-11-14, at the owner's request: "yes merge into main"):
`matlab` at 37d5759, the MATLAB functions over the command line (matlab/: `equip_fit`,
`equip_setup`, `equip`, README.md for a user without Julia) and `equip fit --format`, for the
first user's test with SIR data. Checked first: the command line's tests (160), the functions in
GNU Octave 8.4 (not MATLAB, not Windows), CI on the branch (all six jobs).

Merged into `main` (fast-forward, 2026-11-14, at the owner's request: "merge rician to main"):
`rician` at 3dea513, the Rician fit's second-order bias (Cox and Snell, `secondorder = true`
with `likelihood = :rician`). Checked first: the full `Pkg.test` (EQUIP 2569), CI on the branch
(all six jobs), and the validation group (rician_fits 52, secondorder and agreement 24).

Merged into `main` (fast-forward, 2026-11-13, at the owner's request: "merge into main, then
do the cox snell bias"): `rician` at bf79d5f, tier 2 steps 6 to 9 (the validation group, run
nightly in CI; `propagate`; `secondorder`; uncertain inputs) and the values-by-name rule. Checked
first: the full `Pkg.test` (EQUIP 2547), CI on the branch (all six jobs) and the validation job
run by hand on it (63 tests, 1 m 55 s, with the precompile workload). `rician` carries on with the
Rician fit's second-order bias (Cox and Snell; design.md item 16).

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

**2. Bias from noise at low SNR** (items 13, 16, 20). Built (2026-11-09 to 14, in `main`) but
step 3, estimating σ (on hold): design.md, "Tier 2, bias from noise at low SNR: the plan", ten
steps, each with how it is judged; left from it, DKI in CSF (item 20), `:wmean`'s second-order
bias, and propagation by draws (item 16). Decided
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
`leakage`, `weights`, averages. The command line: `equip fit --likelihood rician --sigma SD`
(a number or a map) `[--method ml|mean|wmean]`, 2026-11-12. Step 9, uncertain inputs (2026-11-12):
`crlb(m, θ; snr, B1 = 1.05, uncertain = (B1 = 0.03,))` adds a measured fixed value's SD through
leakage's slope; montecarlo draws each copy's value (T1's SD 0.0572 s predicted, 0.0577 ± 0.0003
measured); `equip crlb` and `equip montecarlo` take `--uncertain B1=0.03`. 2026-11-13: values by
name are named tuples everywhere (`vary` no longer takes pairs; one check, `_byname`, with a
message for the missing comma); design's `weights` (per parameter) against everyone else's
(per observation) is on the names review (item 24). Step 8 (2026-11-13): `crlb(m, θ; snr,
secondorder = true)`, the noise's own bias to second order (Box 1971, with the residual's term
when something fixed is wrong), parameters and derived quantities; montecarlo within 1 standard
error in every case measured; the validation group's first file, test/validation/secondorder.jl.
Step 7 (2026-11-13): `propagate(c, p -> 1 / p.T1)`, crlb's errors for any function of the
parameters (the derived path's numbers for DTI's FA and MD; montecarlo for PSR/(1 + PSR)).
Step 6 (2026-11-13, branch `rician`): the validation group filled and run nightly in CI
(test/validation/: rician_fits.jl, the Rician fits per model; agreement.jl, bench/agreement.jl
moved and asserted, with `secondorder`; secondorder.jl). Measured (design.md, "Step 6, built
and measured"): the Rician fit removes the floor's bias in tissue (T2 +0.76% against least
squares' +41.2% at SNR 10) at the Rician bound's SD; not better than least squares where the
floor is not the main error (T1IR in WM, DKI's MKT −12.6%); without information, converged on
the caps; DKI in CSF does not converge (2–5%). Two roadmap items from it: the Rician fit's
second-order bias (item 16; built the same day, below) and DKI in CSF (item 20). The Rician
fit's second-order bias (2026-11-13, the owner: "do the cox snell bias"; branch `rician`):
`crlb(m, θ; snr, likelihood = :rician, secondorder = true)`, Cox and Snell's, predicts the
maximum-likelihood fit's remaining bias (T2 +0.78% at SNR 10, montecarlo +0.83 ± 0.10; T1IR
+2.26%, +2.51 ± 0.12; MKT from SNR 20), so crlb tells before scanning whether maximum
likelihood or least squares is less biased. DKI's MK is not predicted: not the fit's bias but
MK's own curvature, too strong for the delta method (its SD 1.5× the delta method's at SNR
20); the parameters drawn from crlb's normal distribution give montecarlo's MK, with no fits:
a roadmap item (item 16, propagation by draws). Design.md, "The Rician fit's second-order
bias". Found while answering the owner (2026-11-09): least squares on the
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
  central differences, edge cases). (test/validation/ filled and run nightly in CI: step 6,
  done 2026-11-13.)
- Still to do from tier 2 (design.md): DKI in CSF with the Rician fit (item 20: bounds on the
  kurtosis, or a convergence test for a flat likelihood); the second-order bias for `:wmean`.

**MATLAB, for a first user's test** (2026-11-14, in `main`; item 10): `equip_fit` over
the command line, `equip_setup`, `equip`, matlab/README.md for a user without Julia; `equip fit
--format nifti|mat|both`. Tested in Octave 8.4, not in MATLAB or on Windows (design.md, item 10,
"First version built"). The first user tests it with SIR data.

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
  hand), ball–stick (one stick; N sticks later), IVIM (the segmented fit as its `datastart`). Built 2026-11-23 (branch `models-proto`).
  The two directional ones start from DTI's principal direction.
- Then: MTR, MTsat, B1 and B0 maps (item 5); pulsed qMT (item 6); NODDI (item 7); free-water
  DTI; DKI's constraints (K ≥ 0 in every direction) imposed, not only checked.
- SAGE (item 25, 2026-11-20): R2* and R2 from spin- and gradient-echo data (Keeling et al.,
  MRI 2024), the paper's Eq. 1 with δ = S0ᴵ/S0ᴵᴵ (three- or four-parameter as δ fixed or free),
  fixed starts. Prototyped and measured (design.md item 25); to decide: rates/times, R2* ≥ R2,
  the DSC file layout.

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
- Later (the owner, 2026-11-23): DKI's and DTI's Jacobian written out (about 1.5× and 1.3×,
  estimated; not worth it for SIR, 6–13% measured); safe stopping tolerances learned per model;
  the dictionary starts (`bestof-proto`, with tier 6); the branch-free `exp` (item 26: 1.39× on
  16-point SIR fits, none on the 4-point).
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
- Open (2026-11-13, tier 2 step 6): Dortch et al. 2018, Fig. 1: did it show the 5-point SIR
  (kmf fitted), and at which SNR? At 10,000 copies crlb predicts the 4-point at SNR 200 within
  Monte Carlo error, but not the 5-point: PSR's SD is 1.28 of crlb's at SNR 200, 1.04 at 400,
  within error from 800 (design.md, "Step 6, built and measured"). The old bench/agreement.jl
  said "the paper: they agree".
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
