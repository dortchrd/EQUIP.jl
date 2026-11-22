The owner's scan at full size (512×512×80, 4 volumes, 6.5M voxels in the mask; its 5 slices
repeated), warm, 4 threads, 2026-11-22 (`combination.jl`):

| configuration | read s | fit s | write s | total s | iterations | stalled | clean voxels: largest |ΔPSR| |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline: QMTSIR, default stopping, gzipped maps | 2.3 | 11.1 | 8.1 | 21.5 | 13.9 | 0.0% | 0.0e+00 |
| QMTSIREqualR1 | 1.1 | 8.5 | 8.0 | 17.6 | 13.9 | 0.0% | 1.5e-14 |
| + calibrated absolute tolerances | 1.6 | 8.0 | 8.1 | 17.6 | 11.2 | 0.0% | 1.1e-03 |
| + stall rule 0.1% | 1.1 | 7.6 | 8.1 | 16.7 | 11.2 | 3.7% | 1.1e-03 |
| + stall rule 1% | 1.3 | 7.3 | 7.8 | 16.4 | 10.9 | 4.7% | 1.1e-03 |
| + uncompressed maps | 1.1 | 7.5 | 0.7 | 9.3 | 10.9 | 4.7% | 1.1e-03 |
|   the same with relative tolerances | 1.1 | 8.1 | 0.7 | 9.9 | 10.9 | 4.7% | 1.3e-02 |
|   the same with absolute, and R1f ≥ 0.2 s⁻¹ | 1.0 | 6.8 | 0.7 | 8.6 | 9.6 | 6.1% | 1.0e-01 |

(σ = 26.3; clean voxels, where the baseline has no parameter on a bound and SSE < 10σ²: 69% of
the mask; tolerances PSR 0.0011, R1f 0.004, Sf 0.0028, M0f 190)
