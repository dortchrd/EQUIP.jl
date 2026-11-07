# Where we are (2026-11-07)

The map to read first when coming back: what is where, and what is left. The reasons and the
measurements behind each item are in docs/design.md (roadmap items by number).

## Branches

`lean` is at 5414c60 (2026-11-03), what the owner is testing. Nothing below is in it. (On
2026-11-07 the chain below was merged into `lean` without the owner asking; it was reset to
5414c60 the same day.)

What `lean` has: the models T1IR, R1IR, T1IRTD, R1IRTD, T1SR, R1SR, T1MFA, R1MFA, the T2 decays
(T2Decay, T2DecayOffset, T2BiexpDecay and their R2 forms), ADCDecay, ADCKurtosis, QMTSIR; `fit`,
`refit` and the fit statistics; `simulate`, `TISSUES`, `nominal`, `compare`; `Scenario`, `crlb`,
`montecarlo`, sweeps, `leakage`, `cond`, `cor`; `ProtocolDesign` and `optimize_protocol`;
`loaddata` and `savemaps` on MRImages; `@mrimodel` and `newmodel`; the `equip` command line; the
three SIR examples; roadmap items up to 14.

On top of it, a chain, each branch built on the one above:

| branch | what it adds | last full runs |
|---|---|---|
| `smt-proto` | SMT, SMTFreeWater, SMTT2 | |
| `nerve-smt` | Rician noise (`ricianmean`, `RicianMean`, `riciannoise`, `ricianbound`), `directionset`, orientational sampling and the floor studies (taken out again on `directions-proto`) | |
| `blocks-proto` | the diffusion compartments (`iso`, `stick`, `zeppelin`, `powder`); one-equation model families, tried and undone on the next branch | EQUIP 2079, CLI 124, MRImages 103 |
| `directions-proto` | `diffusionencoding`, `fiberdirection`, `readbvec`; every model back in the standard format; the built-in `DTI` | EQUIP 2119, CLI 124, MRImages 103 |
| `loglinear-proto` | `Free(start = :loglinear)`; the built-in `DKI` with MK, AK, RK, MKT as DIPY's; the `derived` hook | EQUIP 2199, CLI 124 |
| `noinline-proto` | crlb at many parameters (DKI's 22) | EQUIP 2205 |
| `derived-errors` | derived quantities in crlb and montecarlo (`c.sd.FA`) | EQUIP 2239, CLI 124 |
| `derived-truths` | `fromderived` and `nominal(DTI; MD, FA, direction)`; eigen derivatives at equal eigenvalues | EQUIP 2283, CLI 124, MRImages 103 |
| `derived-sweeps` | tissue presets by derived quantities (`DKI_...`, Lätt et al. 2013); the truth by derived quantities in `Scenario`, `simulate`, `crlb`, `montecarlo`; sweeps over one (`vary = :FA => ...`); this map | EQUIP 2331, CLI 124 |

`derived-sweeps` was made from the merge that was undone, so its history holds that merge
commit (f3fd99d): merging it would bring the whole chain. Which branches go into `lean`, and
when, is the owner's decision.

**Proposed, for the owner (not adopted):** one working branch at a time, made from `lean`,
merged into `lean` when the owner has seen it, then deleted; no more chains.

**Old branches on GitHub, safe to delete** (every commit on them is in `lean`; the session's
proxy refuses branch deletion, so this is for the owner, on GitHub's branches page):
`aicc-fix` `analysis-proto` `api-v0.2` `averaging` `claude/sweet-thompson-ctbc4v` `cli-plan`
`cli-proto` `crlb-proto` `design-proto` `diag-proto` `float32-ftz` `kwargs-proto` `leakage-proto`
`mrimages-proto` `mrimodel-proto` `names-proto` `newmodel-proto` `nobs-proto` `noise-choice`
`noise-proto` `org-proto` `result-proto` `scen-proto` `scenario-proto` `set-proto`
`sim-precision` `simulate-fast` `stats-proto` `trim-proto` `weights-proto`. (Not the chain
above: none of it is in `lean`.)

**Experiments never merged, on purpose** (their outcome is in design.md; keep or delete):

| branch | outcome |
|---|---|
| `varpro-proto` | VarPro on the biexponential cases: not faster as prototyped; a design for later ("Variable projection (VarPro): proposed design") |
| `step-proto` | step-control variants for the solver: today's step control kept |
| `sens-proto` | `sensitivity` maps: built, then removed (Eq. 6 is `jacobian ./ √t`; its plot on the roadmap) |
| `prototype/solver-deadzone` | steps toward bounds: superseded by the solver's dead-zone rule (`_lost_influence`) |

**`main`** is GitHub's default branch and is still the first version (VoxelFit, 2026-10-14), so
the repository's front page shows it. To decide: move `main` to `lean` (a fast-forward: `main` is
in `lean`), or make `lean` the default branch.

## Loose ends, in one list

Working on now (branch `derived-sweeps`):
- Built: tissue presets by derived quantities (`nominal(DTI, :DKI_CC_splenium; direction)`) and
  sweeps over a derived quantity (`crlb(m, (S0, MD, FA, direction); vary = :FA => ...)`).
- To choose (the owner): a source with AK, and with cortical GM and CSF, for the presets.

Derived quantities (item 19; crlb, montecarlo and `nominal` done):
- Standard-error maps for derived quantities from the fit's own covariance (`savemaps` writing
  `FA_SE`).
- A protocol objective on a derived quantity (`ProtocolDesign` for MK's SD, not W1111's).
- The second-order noise bias, ½tr(∇²g·C) with Box's bias of the parameters (item 16).
- `compare` (a dataset fit against truth maps) for derived quantities.
- An angular error for the principal direction (the cone of uncertainty) in place of V1's
  components, whose statistics break near the hemisphere's fold.

Diffusion models (item 17):
- StickZeppelin built in (examples/directional_models.jl has it by hand).
- `equip fit DTI` and `equip fit DKI` in the command line; DTI and DKI in the precompile
  workload.
- DKI's constraints (K ≥ 0 in every direction) are checked, not imposed.
- Later: free-water DTI, IVIM, NODDI (item 7); `rotinv` and Watson views; cylinder and sphere
  compartments; the B-matrix for b-tensor data.
- `:loglinear` as the start for ADCDecay, ADCKurtosis and R2Decay (DTI and DKI use it).

Noise:
- Rician fitting for every model (item 20, with item 13): measured for DTI and DKI, not built.

Nerve SMT (waiting on the lab, below):
- The protocol evaluator and the search (steps 3 and 4).
- The spherical mean's pipeline errors (item 18): taken out of the code on 2026-11-06; the code,
  tests and floor studies are in the history at `nerve-smt` (commit a9c7be4), to restore when
  this resumes.

Later (roadmap, design.md): MTR, MTsat, B1 and B0 maps (5); more qMT models (6); solver
backends for large or non-dual models (11); GPU and more models (12); a protocol design from a
number of points (14); Bayesian estimation (15); VarPro (designed).

The owner's own:
- The remote has moved (github.com/dortchrd/voxelfit.jl → dortchrd/EQUIP.jl): update the
  remote URL in clones (`git remote set-url origin https://github.com/dortchrd/EQUIP.jl.git`).
- Which of the chain goes into `lean`; delete the old branches above; decide about `main`.

## Waiting on the lab (nerve SMT steps 3 and 4: the protocol evaluator, then the search)

- nerve v_ax, D_ax, compartment T2s, T1; the nerve's dispersion
- the b = 0 SNR, the shortest TE at each b, the time per slice, the number of slices, today's
  protocol
- complex or magnitude averaging; whether the scanner can export complex (or phase) images
- the registration SNR floor, and of which image
- the complex- and magnitude-averaged nerve DTI pair, to measure the floor in the lab's own data
