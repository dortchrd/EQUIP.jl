# Variable projection (VarPro) prototype

Not part of EQUIP: a test of whether solving the linear parameters exactly inside every LM
evaluation (variable projection; Golub & Pereyra 1973) is worth adding. Kept on the
`varpro-proto` branch only.

## What it does

`levmar` gets everything from `normal_equations(prob, θ, buf)`. `vp.jl` wraps the model in `VP`
and adds a more specific `normal_equations` method, so the unchanged `levmar` solves the reduced
problem: LM sees only the nonlinear parameters, and the amplitudes c = (SᵀS)⁻¹Sᵀy are computed
in dual arithmetic at every evaluation, which gives the exact Jacobian of the reduced problem.
The columns S of the linear part come either from the model's own `signal!` with the linear
parameters at unit values (generic: S0 = 1), or from functions the model supplies (only each
column's own terms). Amplitudes that must be ≥ 0 use a small active set (2 columns).

## Files

- `vp.jl`: the method. `sanity.jl`: noise-free check (SSE ≈ 0 at the truth, the truth recovered
  from EQUIP's starts, in all five models).
- `compare_lib.jl`: the comparison (same starts for both methods, from EQUIP's own start
  pipeline; single thread, solver time only, best of 3; "reached": final SSE within 1e-6 of the
  lowest SSE any method found for that voxel). `compare.jl`: all cases, generic columns;
  output in `results_generic_columns.txt`. `compare2.jl`: the two-amplitude models with columns
  supplied by the model. `qmtsir_bad_start.jl`: where QMTSIR ends from a far start.
- Run from the repository root: `julia --project=. prototypes/varpro/compare.jl`.

## Results (2000 voxels per case, one seed, SNR 50; biexponential SNR 100)

From EQUIP's own starts (full → VarPro; biexponential and offset with model-supplied columns):

| case | reached | median iterations | µs per voxel |
|---|---|---|---|
| T2Decay | 100 → 100% | 4 → 4 | 1.57 → 1.61 |
| T1IR, T1 = 0.1 s | 100 → 100% | 5 → 5 | 1.26 → 1.00 |
| T1IR, T1 = 0.9 s | 100 → 100% | 5 → 5 | 1.18 → 0.92 |
| QMTSIR, 5-point | 100 → 100% | 6 → 6 | 2.86 → 2.68 |
| biexp, 15% at 20 ms + 85% at 80 ms | 97.9 → 99.2% (max iterations 43 → 18) | 24 → 20 | 23.8 → 19.3 |
| biexp, 40% at 15 ms + 60% at 100 ms | 96.6 → 100% | 18 → 15 | 19.5 → 14.5 |
| T2DecayOffset | 100 → 100% | 10 → 11 | 5.2 → 5.0 |

With generic columns (the whole model evaluated per column) the two-amplitude models were
slower than the full fit: biexponential 23.1 → 27.2 µs, offset 5.2 → 6.5 µs.

From bad starts:

- Coupling, which VarPro removes: T2Decay from T2 = 2 ms, 34 → 8 iterations (13.9 → 3.8 µs);
  T2DecayOffset from T2 = 1 s, 19 → 15 iterations (9.2 → 6.8 µs).
- Other minima, which it does not: T1IR (T1 = 0.1 s) from one start, 0% (full) and 0.7%
  (VarPro) reached; T1IR (T1 = 0.9 s) from 4 s, 100% and 97.5%; QMTSIR from PSR = 0.4, R1f = 3,
  0% for both.
- VarPro with a 2-D grid of 64 T2 pairs for the biexponential: 7–12 iterations, but ~50 µs per
  voxel and no more voxels reaching the minimum than VarPro from the default starts.

QMTSIR side finding: from PSR = 0.4, R1f = 3, every fit (full and VarPro) ends with PSR at its
bound 0, Sf at its bound −1.2 and R1f ≈ 4, at about 340× the SSE of the fit from the default
starts, all reported as converged.

## Reading

- Two linear amplitudes (biexponential, offset): worth adopting, with columns supplied by the
  model and a mapping back to the reported parameters (S0 = c₁ + c₂, fshort = c₁/(c₁ + c₂)).
- One linear parameter: 0–22% faster, robustness unchanged from EQUIP's starts (the `:scale`
  start already captures most of the gain); could be generic.
- Not a replacement for `BestOf`: it removes the coupling between amplitudes and the other
  parameters, not multiple minima.
- Open: how a model declares its linear parameters and columns; standard errors from the full
  Jacobian at the end; a wider study (several truths and SNRs) before adopting.

## Log coordinates (`logt.jl`, `logt_compare.jl`, output in `results_logt.txt`)

The solver works in φ = log θ for chosen parameters; the model sees θ = exp(φ) (the dual numbers
apply the chain rule). `LogT` wraps a model or a VarPro wrapper. `levmar_pp` is EQUIP's `levmar`
with one change: the step test is per parameter, |Δφ_j| ≤ rtol, i.e. with every parameter in
log coordinates, "every parameter changes by less than a relative rtol". Same starts for all
(EQUIP's), 2000 voxels, one seed, solver time. "off>1SD": voxels whose estimates differ from the
same method's tight fit by more than one noise SD (stopped too early). At loose tolerances,
"reached" (SSE within 1e-6 of the best) drops even when nothing is wrong; read "off>1SD" there.

Biexponential (SNR 100), the two truths:

| method, tolerance | reached | off>1SD | mean it | max-iteration stops | µs/voxel |
|---|---|---|---|---|---|
| full, default (today) | 98.3% / 96.6% | 0 / 1 | 27.2 / 21.1 | 43 / 0 | 19.5 / 15.1 |
| full, loose (1e-4, 1e-6) | 96.5% / 95.8% | 19 / 2 | 21.5 / 17.2 | 21 / 0 | 15.7 / 11.9 |
| full + log T2s, default | 98.8% / 100% | 12 / 0 | 27.9 / 16.2 | 29 / 0 | 24.3 / 12.0 |
| VarPro, default | 99.8% / 100% | 0 / 0 | 22.8 / 15.7 | 18 / 0 | 17.3 / 12.2 |
| VarPro + log T2s, default | 99.5% / 100% | 23 / 0 | 19.3 / 11.8 | 0 / 0 | 15.8 / 9.8 |
| VarPro + log T2s, per-parameter step 1e-4 | 99.5% / 100% | 0 / 0 | 16.7 / 9.5 | 0 / 0 | 11.6 / 6.3 |
| VarPro + log T2s, per-parameter step 1e-3 | 97.7% / 90.3% | 0 / 0 | 14.9 / 8.5 | 0 / 0 | 10.2 / 5.3 |

T2Decay (T2 = 2 s, TE ≤ 320 ms, SNR 50, from its log-linear start) and T1IR (T1 = 0.1 and 0.9 s,
from `BestOf`): log T2 / log T1 change nothing measurable (same iterations, no early stops); the
good starts already do the job. (The `LogT` wrapper adds some time there, e.g. T2Decay 0.83 →
1.17 µs at default tolerances; the prototype's path is not optimized.)

Reading:

- Log coordinates help the hard model: with VarPro, max-iteration stops 21 → 1 (tight
  tolerances, first truth) and 15–25% fewer iterations; for the full fit, 96.6 → 100% reaching
  the minimum on the second truth.
- But the current joint step test misbehaves in log coordinates (12–23 early stops at default
  tolerances): log coordinates need the per-parameter step test.
- VarPro + log T2s + per-parameter relative step 1e-4 is the best combination tested for the
  biexponential: 40–59% faster than today's full fit (19.5 → 11.6 µs, 15.1 → 6.3 µs), more
  fits reaching the minimum (98.3 → 99.5%, 96.6 → 100%), no early stops, no max-iteration
  stops.
- The per-parameter relative test answers "stop when every parameter changes by less than
  0.01%", but only for positive parameters (log coordinates); Sf, PSR, fshort or an offset need
  another rule (an absolute tolerance, or the standard-error rule).

## On the benchmark's biexponential cases (`biexp_bench.jl`)

`bench/cases.jl`'s :biexp (SNR 200) and :biexp_snr100: 1000 voxels each, T2short 10-30 ms, T2long
60-120 ms, fshort 0.05-0.30, every voxel different; the same starts and stopping rule as
`bench/solvers.jl`; "not best": a clearly worse fit than the best any method found. Output in
`results_biexp_bench.txt`.

| case | EQUIP full | EQUIP VarPro | best reference |
|---|---|---|---|
| SNR 200, no bounds | 180 not best, 34.5 steps | 8, 27.4 steps | TrustRegion 15 |
| SNR 200, bounds | 4, 31.1 steps | 4, 25.6 steps | LsqFit 6 |
| SNR 100, no bounds | 213, 80.1 steps | 16, 62.3 steps | TrustRegion 42 |
| SNR 100, bounds | 13, 86.3 steps | 11, 67.0 steps | LsqFit 26 |

VarPro removes EQUIP's unbounded biexponential problem (the full fit's path into the region
where the two components merge): its misses drop to below NonlinearSolve's TrustRegion, and its
errors to the best any method reached (SNR 100, median fshort error 43.7% → 28.0%). With bounds
it is as good or better, with ~20% fewer steps. Speed: about the same; two runs gave different
orders (µs per voxel, full → VarPro, 4 threads, 1000 voxels: SNR 200 bounds 8.0 → 5.6 and
7.9 → 9.0; SNR 100 no bounds 14.0 → 12.3 and 18.8 → 11.9), so at this size the timings are
within run-to-run noise. The answers and step counts are identical between runs.
