# Where we are (2026-11-07)

The map to read first when coming back: what is where, and what is left. The reasons and the
measurements behind each item are in docs/design.md (roadmap items by number).

## Branches

| branch | what it is | state |
|---|---|---|
| `lean` | everything accepted; the prototype chain (SMT through truths from derived quantities) merged on 2026-11-07, commit f3fd99d (lean before it: 5414c60) | Last full runs: EQUIP 2283, CLI 124, MRImages 103 |
| `derived-sweeps` | the working branch, off `lean`: tissue presets by derived quantities (`DKI_...`, Lätt et al. 2013), `fromderived(M)`, the truth by derived quantities in `Scenario`, `simulate`, `crlb` and `montecarlo`, sweeps over a derived quantity (`vary = :FA => ...`) | to merge when the owner has seen it. Last full runs: EQUIP (running when committed), CLI (running when committed) |

**The rule from now on:** one working branch at a time, made from `lean`; merged into `lean`
when its suites pass and the owner has seen it; then deleted. No chains of prototypes on top of
each other.

**Old branches on GitHub, safe to delete** (every commit on them is in `lean`; the session's
proxy refuses branch deletion, so this is for the owner, on GitHub's branches page):
`aicc-fix` `analysis-proto` `api-v0.2` `averaging` `blocks-proto` `claude/sweet-thompson-ctbc4v`
`cli-plan` `cli-proto` `crlb-proto` `derived-errors` `derived-truths` `design-proto` `diag-proto`
`directions-proto` `float32-ftz` `kwargs-proto` `leakage-proto` `loglinear-proto`
`mrimages-proto` `mrimodel-proto` `names-proto` `nerve-smt` `newmodel-proto` `nobs-proto`
`noinline-proto` `noise-choice` `noise-proto` `org-proto` `result-proto` `scen-proto`
`scenario-proto` `set-proto` `sim-precision` `simulate-fast` `smt-proto` `stats-proto`
`trim-proto` `weights-proto`.

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
- Delete the old branches above; decide about `main`.

## Waiting on the lab (nerve SMT steps 3 and 4: the protocol evaluator, then the search)

- nerve v_ax, D_ax, compartment T2s, T1; the nerve's dispersion
- the b = 0 SNR, the shortest TE at each b, the time per slice, the number of slices, today's
  protocol
- complex or magnitude averaging; whether the scanner can export complex (or phase) images
- the registration SNR floor, and of which image
- the complex- and magnitude-averaged nerve DTI pair, to measure the floor in the lab's own data

## Merged into lean on 2026-11-07 (the chain, oldest first)

- SMT, SMTFreeWater, SMTT2 (Devan 2020).
- Rician noise (`ricianmean`, `RicianMean`, `riciannoise`, `ricianbound`) and `directionset`.
- The diffusion compartments (`iso`, `stick`, `zeppelin`, `powder`); one-equation model
  families tried and undone (every model in the standard format).
- `diffusionencoding` (.bval and .bvec as b, gx, gy, gz), `fiberdirection`, `readbvec`.
- `DTI` and `DKI` built in; `Free(start = :loglinear)`, the two-pass WLS from the model's own
  signal; DKI's indices as DIPY's (MK, AK, RK, MKT).
- crlb at many parameters (the kernel's pieces not inlined above P = 8).
- The `derived` hook: `derived(res)`, `savemaps`, crlb and montecarlo after the parameters
  (`c.sd.FA`).
- The `fromderived` hook: `nominal(DTI; MD, FA, direction)`; eigen derivatives at equal
  eigenvalues.
