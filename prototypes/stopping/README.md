# The statistical stop: the prototype and its measurements

Branch `stopping-proto` (2026-11-21). Two rules. `LM(atol = (PSR = 5e-4, ...))` stops when the
undamped Gauss–Newton step predicts every listed parameter will change by less than its
tolerance (`_gnstep_small`), and `stoptolerances(m, data)` sets those tolerances from a sample of
voxels (c × the median standard error, σ pooled over the sample). And `LM(ctol = c)` stops a fit when it is within c standard
errors of the minimum: the Gauss–Newton decrement gᵀ(JᵀWJ)⁻¹g, the SSE a full Gauss–Newton
step would still gain, below c²·SSE/(N − P) (`_decrement_small`, src/fit/levmar.jl). Only with
more observations than free parameters, and not while a parameter is held on a bound. Results
and the decision: docs/design.md, Performance notes, "Stopping tolerances against the noise".

| script | what it measures | run |
|---|---|---|
| `statistical.jl` | the stop at c = 0.01, 0.03, 0.1 against the defaults, uniformly looser tolerances and a tight reference, six models, SNR 50 and 200, 10,000 voxels each (about 15 minutes); `statistical_results.md` is its output of 2026-11-21 | `julia -t 1 --project=. prototypes/stopping/statistical.jl` |
| `per_parameter.jl` | the per-parameter tolerances (calibrated; PSR at 5e-4 with the others calibrated; PSR alone) and the statistical stop, eight models with the four SIR schemes, SNR 50 and 200; `per_parameter_results.md` is its output of 2026-11-21 | `julia -t 1 --project=. prototypes/stopping/per_parameter.jl` |
| `degenerate_voxel.jl` | the one 16-point voxel (SNR 50) where every looser rule leaves PSR 0.95 from the reference: a degenerate minimum on PSR's bound | `julia -t 4 --project=. prototypes/stopping/degenerate_voxel.jl` |
| `own_se.jl` | the deviations in each voxel's own standard error (what the stop promises), kurtosis and the bi-exponential, and kurtosis's worst voxels | `julia -t 4 --project=. prototypes/stopping/own_se.jl` |
