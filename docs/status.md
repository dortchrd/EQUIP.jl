# Where we are (2026-11-05)

A note to come back to. The decisions and measurements behind each item are in docs/design.md;
this is the map. Everything below is committed and pushed.

## Branches

Each branch builds on the one above it.

| branch | what it adds | state |
|---|---|---|
| `lean` | the CLI, MRImages, `@mrimodel`, `newmodel`, examples, roadmap items 13–14 | the owner is testing it; nothing merged into it since |
| `smt-proto` | SMT, SMTFreeWater, SMTT2 | not merged |
| `nerve-smt` | Rician noise (`ricianmean`, `RicianMean`, `riciannoise`, `ricianbound`), orientational sampling (`directionset`, `orientationcovariance`, `sphericalmeannoise`), the floor-correction studies (bench/floor_*.jl), the real-data and SMT-or-SMI notes | not merged |
| `blocks-proto` | `EQUIP.Blocks` (`decay`, `ball`, `stick`, `zeppelin`, `powder`); the T2 and SMT families as one equation each with parts switched on and off; examples/building_blocks.jl, examples/directional_models.jl; roadmap item 15 | not merged; includes lean's item 14. Last full runs: EQUIP 2079, CLI 124, MRImages 103 |

## Done recently

- **Building blocks.** `decay(x, rate)` (rate form: `decay(TE, 1/T2)`; its docstring warns that
  `decay(TE, T2)` is wrong and silent), `ball(b, D)`, and `stick` and `zeppelin` with an explicit
  view: `powder` (the spherical mean) or a cosine t (one direction).
- **One equation per family.** The T2 decays (offset and short component switched on in the
  call) and SMT (free water, compartment T2s, the extra-axonal diffusivities as ties). T2 fits are
  bit-identical to before; SMT agrees to rounding. The blocks cost nothing measurable; the
  mono-exponentials pay about 13% for the parts they carry switched off.
- **Floor corrections for nerve SMT** (simulated):
  - Modelling the floor per direction is best when the fibres are coherent.
  - The exact per-direction inversion (`corrRicianBias` on each direction's image) is the safe
    choice when dispersion is unknown or above about ODI 0.1.
  - Phase-corrected real data removes the problem at its source.

## Waiting on the lab (nerve SMT steps 3 and 4: the protocol evaluator, then the search)

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

- **Roadmap item 17, directional diffusion:** a built-in DTI (written out, the six elements,
  a log-linear start) and StickZeppelin; `diffusionencoding(bval, bvec)`, one helper whose named tuple
  goes into any directional model's constructor (DIPY's gradient table as the model); the
  B-matrix when b-tensor data come. DKI written out (examples/dki.jl, 22 parameters): the fit
  is fine (130–170 µs per voxel, all converged); the mess is the 15 names, the indices (MK,
  AK, RK) and the constraints, and crlb, whose kernel did not finish compiling at 22
  parameters (46 s with its pieces not inlined: the fix, measured).
- **Views for the blocks** (item 17): a generic `rotinv(l)` projection (SMI's rotational
  invariants; `powder` is l = 0; one function, not l = 2 and 4 in every block) and a
  Watson-dispersed view.
- **A cylinder block (axon diameter):** first a crlb check that diameter is measurable on the
  lab's δ, Δ and maximum gradient; the units for restricted compartments to decide.
- **Other blocks:** a tensor, kurtosis, EPG for multi-echo T2, the SPGR steady state, exchange; the
  T1 family as building blocks; ADC as the T2 equation in b.
- **Roadmap item 16, error propagation:** `propagate(c, g)` for derived quantities (FA, MD,
  AD, RD: tested against Monte Carlo), uncertain fixed inputs such as a B1 map (leakage's slope
  at the fit's value, plus L·Σ·Lᵀ), and the estimator's own second-order noise bias (Box 1971).
- **Roadmap items 14 and 15:** a protocol design from a number of points; Bayesian estimation
  (priors as residuals, the van Trees bound in crlb, NUTS as an extension).

## Directional models, and how they meet a tensor (2026-11-04 and 05)

examples/directional_models.jl, discussed in the session:

- **The acquisition** is each image's b and gradient direction g (bval, bvec).
- **The cosine** t = g·n(θ, φ) is computed in the model, because it joins the acquisition and the
  parameters; the blocks take it as a number.
- **A stick and a zeppelin are tensors**, so the standard tensor model is the general case. DTI's
  principal direction can fix the fibres' direction, as per-voxel maps, or start a fit of θ and φ.
  Measured: SNR 100, 132 images, v_ax 0.600 ± 0.004, the direction to 0.14°, 20 µs per voxel.
- **Decided (2026-11-04), revised (2026-11-05): a direction is a vector, so `g` holds one
  `SVector{3}` per image,** beside `b` (design.md, "Directions per image"). It gains no speed:
  one field instead of three, a direction as one value, `g = directionset(30)` directly, and a
  matrix per image for b-tensors later. EQUIP needs two small methods (the Float32 conversion and
  the signal's number type; tested in a scratch script). An nobs×3 matrix works by hand but not
  in `@mrimodel`; gx, gy, gz works and was the first rule.
- **Decided with it:** `readbvec` (MRImages, a general package) keeps returning the file's 3×N
  matrix; EQUIP's `diffusionencoding(bval, bvec)` returns `(b, g)`, checked, and `loaddata` maps
  a .bvec to `g`. A supertype `DirectionalDiffusion` carries what directional models share; each
  keeps its own four-line struct (no field inheritance in Julia). New single-equation built-in
  models are written with `@mrimodel`; QMTSIR stays written in full as the reference, with the
  T2 and SMT families.
- **To build (branch `directions-proto`):** the two methods and a clear error for other fields;
  `diffusionencoding`; `DirectionalDiffusion`; a supertype option for `@mrimodel`; tests
  (`signal`, `simulate`, Float32 fits, crlb, ProtocolDesign, `@mrimodel` with `g`); the
  directional and DKI examples switched to `g`. Then the built-in DTI and StickZeppelin (item 17).
- **Planned (roadmap item 17):** b and g until b-tensor data come, then the B-matrix B = b·ggᵀ
  as one static matrix per image (planar and spherical encoding need it); a `tensor` block with it.
