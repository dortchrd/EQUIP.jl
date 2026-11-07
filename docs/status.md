# Where we are (2026-11-07)

A note to come back to. The decisions and measurements behind each item are in docs/design.md;
this is the map. Everything below is committed and pushed.

## Branches

Each branch builds on the one above it.

| branch | what it adds | state |
|---|---|---|
| `lean` | the CLI, MRImages, `@mrimodel`, `newmodel`, examples, roadmap items 13–14 | the owner is testing it; nothing merged into it since |
| `smt-proto` | SMT, SMTFreeWater, SMTT2 | not merged |
| `nerve-smt` | Rician noise (`ricianmean`, `RicianMean`, `riciannoise`, `ricianbound`), orientational sampling (`directionset`, `orientationcovariance`, `sphericalmeannoise`), the floor-correction studies (bench/floor_*.jl), the real-data and SMT-or-SMI notes | not merged |
| `blocks-proto` | `EQUIP.Blocks` (`decay`, `ball`, `stick`, `zeppelin`, `powder`); the T2 and SMT families as one equation each with parts switched on and off (both undone on `directions-proto`); examples/building_blocks.jl, examples/directional_models.jl, examples/dki.jl; roadmap items 15–17 | not merged; includes lean's item 14. Last full runs: EQUIP 2079, CLI 124, MRImages 103 |
| `directions-proto` | `diffusionencoding` (a .bval and a .bvec as b, gx, gy, gz, checked), `loaddata` reading a .bvec, `readbvec` exported; `fiberdirection` and Julia's `⋅` (ξ = g ⋅ n); the compartments (`iso`, `stick`, `zeppelin`, `powder`; "building blocks" until 2026-11-06) exported by EQUIP itself (no module), diffusion only, no `decay`; the T2 and SMT families undone, every model in the standard format again; `nobs` refusing a field of 3-vectors; "fibre" spelled "fiber" throughout; the spherical mean's pipeline errors (`orientationcovariance`, `RicianSphericalMean`, `sphericalmeannoise`) and the floor studies taken out, to return with the nerve protocol work (design.md, roadmap item 18); the built-in `DTI`, written out, starting from the weighted log-linear fit | not merged. Last full runs: EQUIP 2119, CLI 124, MRImages 103 |
| `loglinear-proto` | `Free(start = :loglinear)`: the weighted log-linear fit computed by EQUIP from the model's own signal (Gauss–Newton on ln y with the solver's dual numbers); DTI on it, its `datastart` removed (the same starts and fits, 4–16% slower in all); DKI from it, 4.6 iterations instead of 15.7, 11% faster | not merged. Last full runs: EQUIP 2125, CLI 124 |

## Done recently

- **`:loglinear`, the log-linear start from the model's signal** (2026-11-07, branch
  `loglinear-proto`; design.md, "The log-linear start"). The owner's point: a start should not
  restate the signal equation. Gauss–Newton on ln y with ∂ln S from `signal!`'s dual numbers;
  DTI uses it (no `datastart`), DKI converges in a third of the iterations. Not yet: the DKI
  example on it; ADCDecay, ADCKurtosis, R2Decay.
- **DTI, built in** (2026-11-07; src/models/diffusion/dti.jl, design.md "DTI built in"). The
  standard format, gᵀDg written out, the six elements fitted with no bounds; each voxel starts
  from the weighted log-linear fit (exact on noise-free data, an isotropic start when the
  design cannot give the six), 4 iterations instead of 6, about 2.5 µs per voxel. No `tensor`
  compartment and no multi-tensor models (the owner: several fibers are zeppelins with
  directions). Next for DTI: FA, MD, AD, RD and the principal direction as maps; `equip fit
  DTI`; the precompile workload.
- **Compartments** (called "building blocks" until 2026-11-06; the literature's word, after
  Panagiotaki et al.'s taxonomy). The diffusion compartments, exported by EQUIP, no module
  (src/models/diffusion/compartments.jl): `iso(b, D)` (Panagiotaki's ball; with D = 0 the dot), and
  `stick` and `zeppelin` with an explicit view,
  `powder` (the spherical mean) or a cosine ξ (one direction). The cosine comes from helpers, not
  compartments: `fiberdirection` (directions.jl) and `ξ = g ⋅ n`. Relaxation is written as its
  equation, `exp(-TE / T2)`; there is no `decay`.
- **One equation per family: tried on blocks-proto, then undone (2026-11-06, the owner: "I do not
  like how unreadable that code is, switching things on and off").** Every model is in the
  standard format again, its own struct, `parameters` and `signal!`; SMT's three are written with
  the compartments. Every built-in model's fits are bit-identical to the family code's.
- **Floor corrections for nerve SMT** (simulated):
  - Modelling the floor per direction is best when the fibers are coherent.
  - The exact per-direction inversion (`corrRicianBias` on each direction's image) is the safe
    choice when dispersion is unknown or above about ODI 0.1.
  - Phase-corrected real data removes the problem at its source.
  - The studies (bench/floor_*.jl) and the functions they use are on `nerve-smt`; taken out of
    `directions-proto` on 2026-11-06 until the nerve protocol work resumes (roadmap item 18).

## Waiting on the lab (nerve SMT steps 3 and 4: the protocol evaluator, then the search)

(The evaluator needs the spherical mean's pipeline errors back first: roadmap item 18.)

- nerve v_ax, D_ax, compartment T2s, T1; the nerve's dispersion
- the b = 0 SNR, the shortest TE at each b, the time per slice, the number of slices, today's
  protocol
- complex or magnitude averaging; whether the scanner can export complex (or phase) images
- the registration SNR floor, and of which image
- the complex- and magnitude-averaged nerve DTI pair, to measure the floor in the lab's own data

## Decisions for the owner

- Whether `blocks-proto` (and with it `smt-proto` and `nerve-smt`) goes into `lean`, after
  testing lean.
- Delete the old remote branches `cli-proto`, `mrimages-proto`, `mrimodel-proto` and
  `newmodel-proto` on GitHub (the session's proxy refuses branch deletion).

## Discussed, not built

- **Roadmap item 17, directional diffusion:** StickZeppelin (DTI is built, 2026-11-07); `diffusionencoding(bval, bvec)`, one helper whose
  named tuple goes into any directional model's constructor (DIPY's gradient table as the
  model); the B-matrix when b-tensor data come. DKI written out (examples/dki.jl, 22 parameters): the fit
  is fine (130–170 µs per voxel, all converged); the mess is the 15 names, the indices (MK,
  AK, RK) and the constraints, and crlb, whose kernel did not finish compiling at 22
  parameters (46 s with its pieces not inlined: the fix, measured).
- **Views for the compartments** (item 17): a generic `rotinv(l)` projection (SMI's rotational
  invariants; `powder` is l = 0; one function, not l = 2 and 4 in every compartment) and a
  Watson-dispersed view.
- **A cylinder compartment (axon diameter):** first a crlb check that diameter is measurable on the
  lab's δ, Δ and maximum gradient; the units for restricted compartments to decide.
- **Other compartments:** a cylinder and a sphere (not a tensor: decided 2026-11-07) (Panagiotaki's taxonomy;
  Microstructure.jl has the cylinder and sphere, with the GPD approximation; the dot is
  `iso(b, 0)`). Not compartments,
  and so written in their models: kurtosis, EPG for multi-echo T2, the SPGR steady state,
  exchange.
- **Roadmap item 16, error propagation:** `propagate(c, g)` for derived quantities (FA, MD,
  AD, RD: tested against Monte Carlo), uncertain fixed inputs such as a B1 map (leakage's slope
  at the fit's value, plus L·Σ·Lᵀ), and the estimator's own second-order noise bias (Box 1971).
- **Roadmap items 14 and 15:** a protocol design from a number of points; Bayesian estimation
  (priors as residuals, the van Trees bound in crlb, NUTS as an extension).

## Directional models, and how they meet a tensor (2026-11-04 to 06)

examples/directional_models.jl, discussed in the session:

- **The acquisition** is each image's b and gradient direction g (bval, bvec).
- **The cosine** ξ = g·n(θ, φ) is computed in the model, because it joins the acquisition and
  the parameters; the compartments take it as a number.
- **A stick and a zeppelin are tensors**, so the standard tensor model is the general case. DTI's
  principal direction can fix the fibers' direction, as per-voxel maps, or start a fit of θ and φ.
  Measured: SNR 100, 132 images, v_ax 0.600 ± 0.004, the direction to 0.14°, 12.8 µs per voxel
  written by hand (23.1 with `@mrimodel`, whose body makes the direction for every image).
- **Decided: gx, gy, gz, built from the files by `diffusionencoding`** (2026-11-04; changed to
  one SVector `g` per image on 2026-11-05; back on 2026-11-06, the owner's preference after
  seeing both written out: design.md, "Directions per image"). `diffusionencoding(bval, bvec)`
  reads a .bval and a .bvec (or takes a vector and a matrix) and returns `(b, gx, gy, gz)`,
  checked, for any directional model: `DTI(; diffusionencoding("dwi.bval", "dwi.bvec")...)`. No
  change to EQUIP's generic code.
- **Decided with it:** `readbvec` (MRImages, a general package) keeps returning the file's 3×N
  matrix. No supertype (the owner, 2026-11-06): everything works by field name; add one when
  code must dispatch on "a directional model". Built-in models are written out by hand
  (DTI too, 2026-11-07: `@mrimodel` later, for compactness); models with fiber angles (StickZeppelin)
  `fiberdirection(θ, φ)` once per voxel and `ξ = g ⋅ n` (Julia's dot) per image, with
  g = (gx[j], gy[j], gz[j]); QMTSIR stays written in full as the reference, with the T2 and SMT families.
- **Built (branch `directions-proto`, 2026-11-06):** `diffusionencoding` (b-values kept as
  given; the threshold only says where a direction must be a unit vector), `loaddata` reading a
  .bvec beside the .bval, `readbvec` exported, `fiberdirection` and `⋅`, `nobs` refusing a field of
  3-vectors (naming `diffusionencoding`), "fiber" spelling, the directional and DKI examples on
  `diffusionencoding`, tests. The built-in DTI followed (2026-11-07). **Next:** StickZeppelin
  (item 17).
- **Planned (roadmap item 17):** b and g until b-tensor data come, then the B-matrix B = b·ggᵀ
  as six component vectors (planar and spherical encoding need them). No `tensor` compartment
  (2026-11-07).
