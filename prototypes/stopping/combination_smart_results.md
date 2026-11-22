`combination_smart.jl` on branch combined-proto, 2026-11-22: the owner's scan at full size
(512×512×80, 6.5M voxels in the mask), warm, 4 threads. Run twice; the times varied by about
10% between the runs (the first: 7.7 s fit for the second row, 6.0 for the third).

| configuration | read s | fit s | write s | total s | iterations | stalled | clean voxels: largest |ΔPSR| | SSE > σ² above the baseline | > σ² below |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline: QMTSIR, default stopping, gzipped maps | 1.2 | 11.3 | 7.9 | 20.4 | 13.9 | 0.0% | 0.0e+00 | 0.00% | 0.00% |
| QMTSIREqualR1, atol, stall 1%, uncompressed | 1.1 | 8.5 | 0.7 | 10.2 | 10.9 | 4.7% | 1.1e-03 | 0.73% | 0.00% |
| + BestOf R1f 8 (0.3–3), dictionary | 1.1 | 6.4 | 0.8 | 8.3 | 7.8 | 3.9% | 4.9e-01 | 0.57% | 5.70% |
| + PSR × R1f 6×8, dictionary | 1.1 | 7.1 | 0.9 | 9.2 | 8.1 | 3.7% | 5.9e-01 | 0.99% | 6.37% |
| + PSR × R1f 6×10 (R1f 0.1–3), dictionary | 1.1 | 7.0 | 0.9 | 8.9 | 8.0 | 3.7% | 5.9e-01 | 0.92% | 6.37% |
|   PSR × R1f 6×8, dictionary, default stopping | 1.1 | 8.0 | 0.8 | 9.9 | 11.1 | 0.0% | 5.9e-01 | 0.02% | 6.37% |

(σ = 26.3; clean voxels, the baseline's fits with no parameter on a bound and SSE < 10σ²: 69% of
the mask.)

The PSR changes of 0.5 in clean voxels (`smart_ambiguous.jl`, the 5 slices, tight stopping): 4
points, 4 parameters, and in about 1% of the clean voxels two exact fits (SSE 0 both: e.g. PSR
0.275 and 0.108, the same R1f, Sf −0.44 and −0.41); the start decides which. Not the starts'
error, the protocol's: nothing in the data tells them apart. They are poorly inverted voxels (Sf
median −0.5: scalp, skull, neck); of the well-inverted clean voxels (Sf < −0.8, 190,217), 73
(BestOf R1f) or 46 (PSR × R1f) change PSR by more than 0.01.
