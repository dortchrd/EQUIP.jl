# Where the work stands (2026-11-28)

The page to read first when coming back: the branches, the work in progress, what is next, and
what waits on the owner. The code as it is: [architecture.md](architecture.md). The reasons and
measurements, the roadmap by item number, and what was tried and not adopted:
[design.md](design.md) (indexed at its top). What was merged when, and what each tier built:
[history.md](history.md).

## Branches

| branch | at | what it is |
|---|---|---|
| `main` | 18974d4 | the released work (EQUIP 0.3.0 and since): every built-in model but `QMTSPGRPortnoy`; CI passed |
| `mtr` | 500951f | MTR, unmerged: `mtr`, `MTRResult`, `writemaps` for MTR maps, `propagate(m, θ, g)` and `montecarlo(m, θ, g)` (src/errors/images.jl), `design(f, m)` (src/design/objective.jl), examples/mtr_design.jl; CI passed |
| `mamt-proto` | 05815e3 | `mtr` plus `QMTSPGRPortnoy` (MAMT), unmerged |
| `claude/peaceful-lovelace-lybjbl` | (newest) | `mamt-proto` plus QMTSPGRPortnoy's tests (43a1ef6) and the documentation review of 2026-11-28. **Start a session here**, or on `mamt-proto` once the owner fast-forwards it to this |
| `bestof-proto` | 0e2e123 | the dictionary start, on hold (roadmap item 4) |
| `experiments` | 02e5f22 | the archive of prototypes not adopted; its README says how to get each back (design.md, "Tried and not adopted") |
| `lean` | fae57ba | where `main` started on 2026-11-08; holds nothing `main` lacks. Whether to delete it: the owner |
| `claude-oneshot` | bd6f278 | the first version (VoxelFit, 2026-10-14) |

The owner holds every merge (2026-11-27: "Hold all merges until I say so"). Sessions cannot
delete branches on GitHub: the owner does (`git push origin --delete NAME`). The remote moved
from dortchrd/voxelfit.jl to dortchrd/EQUIP.jl: old clones need `git remote set-url origin
https://github.com/dortchrd/EQUIP.jl.git`.

## In progress: MT protocol optimization

**The goal** (the owner): optimize conventional MT acquisitions (TR, excitation flip angle, MT
power, offset, perhaps the MT pulse's duration) from tissue properties: predict the MT-weighted
and reference signals, MTR and its noise, with EQUIP's simulation, derivatives and design.
Objectives: MTR's SNR, its sensitivity to PSR (also per noise), tissue CNR, robustness to B1 and
R1, with scan time and SAR. **Keep it simple** (the owner, 2026-11-27: "i fear you are going
into overengineering territory. remember goals and keep it simple"): no MTR-specific framework;
the objectives as recipes on what exists.

**Done.** MTR and its errors from the images, and `design(f, m)` (branch `mtr`; design.md item
5). `QMTSPGRPortnoy` (branch `mamt-proto`; design.md item 6, "The exact reference"): the
two-pool Bloch equations through the shaped MT pulse, the reference Yarnykh's model was missing
(Yarnykh's is +20% and +79% above it at 1 kHz for 4 and 8 µT, +9% and +5% at 2 kHz). Step 1,
its tests (2026-11-28; design.md item 6, "Tested"): RK4, the CW and far-off-resonance limits,
the Jacobian, both precisions; `mtshape` given with `mtpulse` now refused. The models file
passes (1561), and roles and starts_solver (292); the full `Pkg.test` not yet run.

**Next, in order** (each small; the owner reviews each before the next):

0. The owner's decision on the audit below (it changes what steps 3 and 4 build on).
1. ~~Tests for `QMTSPGRPortnoy`~~ (done, above).
2. Its speed against Yarnykh's (a block: one 5×5 exponential an image; a shape: one a segment),
   by adding it to bench/qmt/speed.jl.
3. examples/mtr_design.jl again with `QMTSPGRPortnoy`: do the 1-2 kHz optima found with
   Yarnykh's model hold? (They are where Yarnykh's is biased.)
4. The owner's objectives as recipes on what exists (no new API): MTR's SNR, ∂MTR/∂PSR, per
   noise, CNR, B1 and R1 robustness, scan time, SAR (B1e²·tm/TR as a constraint); `tm` searched
   too.
5. Merge `mtr`, then `mamt-proto`: the full `Pkg.test`, then CI (the precompile workload runs
   `design`). Only when the owner says.

**Open, the owner's to decide:** the new model's name (`QMTSPGRPortnoy`, Claude's choice by the
convention of naming after the paper's first author; not yet confirmed); the excitation's
saturation of the bound pool (neglected now; matters at high flip angles and short TR); the
sequence's structure (an MT pulse every TR, as now, or every N excitations, or MT-prepared TSE);
the lab's MT pulse shapes (as samples); MTsat, a model or not; which B1 and B0 methods the lab
acquires. Testing on the owner's computer is still blocked (no Remote Control folder).

## The audit of 2026-11-28 (for the owner)

The owner asked for one ("a little drift lately ... some overengineering in the MT stuff"). No
code was changed; the documentation was.

**Drift in the documentation, fixed in this review.** CLAUDE.md's counts and times (about 2360
tests: about 3550 at the last full run; the unit group "about a minute": 5 minutes measured in a
cloud container), its file map (moved to architecture.md; it lacked src/fit/rician.jl and
src/sim/tissues.jl), the precompile workload's coverage (not DKI, the SMT models or
`QMTSPGRPortnoy`). This file: four branches listed "to delete" that are gone; the history mixed
with the current state (moved to history.md). design.md: headings saying "proposed" or "the
plan" for what is built (now indexed with a status each); item 6's plan bullets for MAMT ("as I
recall") after it was built; "Scope so far" from 2026-10-31. The guide: `mtr`'s list of models
without `QMTSPGRPortnoy`, and the MTR design's 1-2 kHz optima stated without saying they come
from Yarnykh's model where it is biased (both now said).

**Drift in the code, not fixed** (no code changes were asked for): src/EQUIP.jl's export
comment, `diffusionencoding # .bval + .bvec -> b, gx, gy, gz` (it returns b and g since
2026-11-25); `mtr`'s docstring names three models, not `QMTSPGRPortnoy`; `QMTSPGRPortnoy` keeps a
helper used once, `_mtamplitude`, against the self-contained rule.

**A latent bug, on `mtr`**: `design(f, m)` builds its candidate protocols with the model's
positional constructor (`_protocol`, src/design/objective.jl), which skips the keyword
constructor's checks. A candidate with TR shorter than tm + ts gives finite signals, not an
error (checked: TR 8 ms against tm + ts 14 ms), so a search whose bounds allow it would rank
impossible protocols. examples/mtr_design.jl's bounds avoid it. To fix before merging: build
candidates through the keyword constructor, or rank such a candidate Inf.

**Overengineering in the MT work**, almost all on `mtr` (unmerged): about 800 lines added to
src/ for MTR, whose formula is one line, and its design. What the goal needs against what is
there:

- **mtr.jl** (230 lines): references named three ways (`:maxoffset` with an exception rule,
  indices, a Bool per image), a pairing rule with a fallback, `MTRResult` with a `getproperty`
  alias (`r.B1e` for its `power` field) and its own display, and `writemaps` for it (a 4-D file
  with a sidecar entry per volume). A design objective needs `1 - S[j] / S[ref]` on `signal(p,
  θ)`; maps from data need `mtr(S, Sref)`, which exists.
- **images.jl** (260 lines): `propagate(m, θ, g)` and `montecarlo(m, θ, g)`, two result types
  of 13 and 16 fields, `g` returning three shapes, correlated uncertain inputs, the second-order
  bias, display tables. Both names already mean something else (`propagate(c, g)` takes crlb's
  result; `montecarlo(m, θ)` fits). For the goal: MTR's SD is a closed form, (σ/S_ref)·√(1 +
  (S/S_ref)²); ∂MTR/∂PSR and ∂MTR/∂B1 are one ForwardDiff call each.
- **objective.jl** (144 lines, and rounding by field in search.jl): bounds per image with
  numbers to keep, `shared`, `round` by field, `constraints`, protocols rebuilt from the search's
  values (the bug above). The search over a plain vector within bounds, with the user's `f(x)`
  building the protocol by the keyword constructor (which checks it), would do the same.
- **QMTSPGRPortnoy's pulse** (`mamt-proto`), given three ways: a block from `B1e` and `tm`;
  `mtshape`, scaled to `B1e` and stretched to `tm` (its own durations ignored, a trap); or
  `mtpulse` and `mtflip`. Two would do: the block, and `mtpulse` with `mtflip`.
- **On `main`**: `_MT_NAMEHINTS` (mtnames.jl), 35 other conventions' names mapped to hints. Small
  and harmless, but breadth beyond any need met so far.

The core is sound and tested: `QMTSPGRPortnoy`'s signal (RK4, two physical limits), the qMT
models on `main`, `mtr(S, Sref)`, reusing the protocol search for any objective. Claude's
recommendation: before steps 3 and 4, a branch that trims `mtr`'s additions to what the goal
uses (`mtr(S, Sref)`, MTR from an acquisition by indices, one function for an image quantity's
value, SD and gradient, the search over a plain vector), with results checked unchanged and the
owner's review; or merge as is. The owner's choice.

## The priorities, and what is left in each

The order agreed with the owner on 2026-11-08 is in design.md's Roadmap ("Priorities"); what
each built is in history.md. Left:

1. **The prerelease**: done (0.3.0, 2026-11-08).
2. **Bias from noise at low SNR** (items 13, 16, 20): built but estimating σ (step 3, on hold:
   the owner, 2026-11-12, "let's hold off on different ways to estimate noise sd"); DKI in CSF
   with the Rician fit (item 20); `:wmean`'s second-order bias; propagation by draws (item 16);
   a protocol design, `leakage`, `weights` and averages for the Rician fits.
3. **Protocol design** (items 3, 14, 19): a protocol from a number of points (item 14); an
   objective on a derived quantity (MK's SD, not W1111's); the search's crlb and leakage from
   one kernel; the worst case inside a box.
4. **Derived quantities** (item 19): standard-error maps (`FA_SE`); `compare` for them; an
   angular error for V1; `vary` over them in `scenarios()`; T1 from R1, SMT's D⊥.
5. **Models** (items 5, 6, 7, 17, 25, 31): MTR (in progress, above), MTsat, B1 and B0 maps;
   pulsed qMT (MAMT, above); free-water DTI; DKI's constraints imposed; SAGE (prototyped: rates
   or times, R2* ≥ R2, the DSC layout to decide); joint models (SIR with a two-pool VFA first).
6. **Starts across all models** (item 4, after the models): plausible ranges per model, every
   start tested over them in both precisions, the dictionary start (`bestof-proto`), an example
   `datastart`.
7. **Nerve SMT**, when the lab sends its values (item 18; below).

MATLAB (item 10): built for a first user's test with SIR data (2026-11-14); tested in Octave,
not in MATLAB or on Windows; the user's report not yet recorded.

**Recorded, not ranked** (design.md has each): the tests consolidated area by area when work
touches an area, and a benchmark framework with a stored baseline (item 21); DKI's RTK, KFA,
Kmax and DIPY's numerical MK, AK, RK (19); Bayesian estimation (15); VarPro, log coordinates,
solver backends, GPU (11, 12); later (the owner, 2026-11-23): DKI's and DTI's Jacobian written
out, stopping tolerances learned per model, the branch-free `exp` (26), Bingham–NODDI faster;
the other start changes (ADCKurtosis and T2DecayOffset on `:loglinear`, `BestOf(fits = k)`,
neighbours; item 4); the B-matrix, `rotinv`, cylinders, spheres, N sticks (17); phantoms (2);
the LsqFit bridge and a "coming from" page (8); protocol files, formats, MRImages in its own
repository with Julia 1.10 (9); `--init`, a server, Python, an installer (10); preprocessing
(22); a units switch (23); the names review (24); myelin water, DCE, the sinc gradient echo
(12); `equip mtr` (with CI: it touches the command line).

## The owner's own

- Open (2026-11-13): Dortch et al. 2018, Fig. 1: did it show the 5-point SIR (kmf fitted), and
  at which SNR? At 10,000 copies crlb predicts the 4-point at SNR 200 within Monte Carlo error,
  but not the 5-point: PSR's SD is 1.28 of crlb's at SNR 200, 1.04 at 400, within error from
  800 (design.md, tier 2, "Step 6, built and measured").
- Later: whether to delete `lean`; whether to adopt one working branch at a time (proposed:
  made from `main`, merged when the owner has seen it, then deleted).

## Waiting on the lab (nerve SMT steps 3 and 4: the protocol evaluator, then the search)

- nerve v_ax, D_ax, compartment T2s, T1; the nerve's dispersion
- the b = 0 SNR, the shortest TE at each b, the time per slice, the number of slices, today's
  protocol
- complex or magnitude averaging; whether the scanner can export complex (or phase) images
- the registration SNR floor, and of which image
- the complex- and magnitude-averaged nerve DTI pair, to measure the floor in the lab's own data
