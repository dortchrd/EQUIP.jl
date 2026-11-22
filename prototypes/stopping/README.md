# The statistical stop: the prototype and its measurements

Branch `stopping-proto` (2026-11-21). Two rules. `LM(atol = (PSR = 5e-4, ...))` stops when the
undamped Gauss–Newton step predicts every listed parameter will change by less than its
tolerance (`_gnstep_small`), and `stoptolerances(m, data)` sets those tolerances from a sample of
voxels (c × the median standard error, σ pooled over the sample). And `LM(ctol = c)` stops a fit when it is within c standard
errors of the minimum: the Gauss–Newton decrement gᵀ(JᵀWJ)⁻¹g, the SSE a full Gauss–Newton
step would still gain, below c²·SSE/(N − P) (`_decrement_small`, src/fit/levmar.jl). Only with
more observations than free parameters, and not while a parameter is held on a bound. Also the stall rule,
`LM(stall = true)` (2026-11-22): a fit whose set of parameters on a bound has been the same for
`stallsteps` (3) accepted steps, the SSE falling less than `stalltol` (0.1%; 1% measured) of
itself over them, stops with status `StalledOnBound`. No σ needed. And `writemaps(...; compress
= false)` (`--compress no`), .nii maps for .nii.gz. Results
and the decision: docs/design.md, Performance notes, "Stopping tolerances against the noise".

| script | what it measures | run |
|---|---|---|
| `statistical.jl` | the stop at c = 0.01, 0.03, 0.1 against the defaults, uniformly looser tolerances and a tight reference, six models, SNR 50 and 200, 10,000 voxels each (about 15 minutes); `statistical_results.md` is its output of 2026-11-21 | `julia -t 1 --project=. prototypes/stopping/statistical.jl` |
| `per_parameter.jl` | the per-parameter tolerances (calibrated; PSR at 5e-4 with the others calibrated; PSR alone) and the statistical stop, eight models with the four SIR schemes, SNR 50 and 200; `per_parameter_results.md` is its output of 2026-11-21 | `julia -t 1 --project=. prototypes/stopping/per_parameter.jl` |
| `relative_se.jl` | the calibrated absolute tolerances against fractions of each voxel's values (`LM(rtol)`, calibrated) and of its own standard errors (`LM(setol, sigma)`, σ pooled by `stopsigma` or given), nine models (with T2 + offset); `relative_se_results.md` is its output of 2026-11-21 | `julia -t 1 --project=. prototypes/stopping/relative_se.jl` |
| `degenerate_voxel.jl` | the one 16-point voxel (SNR 50) where every looser rule leaves PSR 0.95 from the reference: a degenerate minimum on PSR's bound | `julia -t 4 --project=. prototypes/stopping/degenerate_voxel.jl` |
| `own_se.jl` | the deviations in each voxel's own standard error (what the stop promises), kurtosis and the bi-exponential, and kurtosis's worst voxels | `julia -t 4 --project=. prototypes/stopping/own_se.jl` |
| `combination.jl` | the owner's scan at full size (its 5 slices repeated to 80), read, fit and write, every combination: QMTSIREqualR1, calibrated atol or rtol, the stall rule at 0.1% and 1%, R1f ≥ 0.2, uncompressed maps; the data are not in the repository (the owner's) | `julia -t 4 --project=cli prototypes/stopping/combination.jl <folder with SIR.nii.gz, MASK.nii.gz>` |
| `robustness.jl` | default, atol, atol + stall 1%: nine models, SNR 50 and 200, and a poor mask (Rician; tissue, weak signal, noise only) for four; `robustness_results.md` is its output of 2026-11-22 (about 10 minutes) | `julia -t 1 --project=. prototypes/stopping/robustness.jl` |
