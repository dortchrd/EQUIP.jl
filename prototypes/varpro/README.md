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
