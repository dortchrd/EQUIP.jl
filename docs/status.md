# Where we are (2026-11-04)

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

- **Views for the blocks:** a generic `rotinv(l)` projection (SMI's rotational invariants;
  `powder` is l = 0; one function, not l = 2 and 4 in every block) and a Watson-dispersed view.
- **A cylinder block (axon diameter):** first a crlb check that diameter is measurable on the
  lab's δ, Δ and maximum gradient; the units for restricted compartments to decide.
- **Other blocks:** a tensor, kurtosis, EPG for multi-echo T2, the SPGR steady state, exchange; the
  T1 family as building blocks; ADC as the T2 equation in b.
- **Roadmap item 16, error propagation:** `propagate(c, g)` for derived quantities (FA, MD,
  AD, RD: tested against Monte Carlo), uncertain fixed inputs such as a B1 map (leakage's slope
  at the fit's value, plus L·Σ·Lᵀ), and the estimator's own second-order noise bias (Box 1971).
- **Roadmap items 14 and 15:** a protocol design from a number of points; Bayesian estimation
  (priors as residuals, the van Trees bound in crlb, NUTS as an extension).

## Open question (2026-11-04): directional models, and how they meet a tensor

examples/directional_models.jl, discussed in the session:

- **The acquisition** is each image's b and gradient direction g (bval, bvec), for now as three
  vectors gx, gy, gz.
- **The cosine** t = g·n(θ, φ) is computed in the model, because it joins the acquisition and the
  parameters; the blocks take it as a number.
- **A stick and a zeppelin are tensors**, so the standard tensor model is the general case. DTI's
  principal direction can fix the fibres' direction, as per-voxel maps, or start a fit of θ and φ.
  Measured: SNR 100, 132 images, v_ax 0.600 ± 0.004, the direction to 0.14°, 20 µs per voxel.
- **Decided (2026-11-04): directions go in as three vectors of numbers,** `gx, gy, gz =
  eachrow(readbvec("dwi.bvec"))`, views of the .bvec's rows. The rule, as for TD and TR (filled
  to one value per observation in their constructors): every per-observation acquisition field
  is a vector of numbers. A field of 3-vectors would break `signal` and the Float32 conversion
  for every model, `@mrimodel` or hand-written (tested); the views fit correctly, in Float32
  too (the fit copies every field into a plain vector when it starts), and the lengths are
  checked. A model holds the views, so after changing a bvec (rotated for motion, a sign
  flipped), build a new model.
- **Small things to do:** a clear error for a field of 3-vectors, pointing to `eachrow`;
  `loaddata` mapping a .bvec to `gx`, `gy`, `gz`; the example switched to `eachrow`.
- **Still to decide:** b and g, or the B-matrix B = b·ggᵀ, which tensor-valued encoding (planar,
  spherical) needs (by the same rule, six component vectors); a `tensor` block.
