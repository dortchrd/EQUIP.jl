# Geodesic acceleration: the prototype and its measurements

Branch `geodesic-proto` (2026-11-21). `LM(geodesic = true)` adds Transtrum & Sethna's geodesic
acceleration (arXiv:1201.5885, 2012) to each Levenberg–Marquardt step: the second derivative
of the model along the step, exact, from nested dual numbers (`_geodesic_rhs`,
src/fit/levmar.jl), and the step v + a/2, accepted when 2‖a‖/‖v‖ ≤ `alpha`; beyond it the step
is rejected (`georeject = true`, Transtrum's rule) or taken without the correction (`false`).
The result: slower in every model measured, and not more robust; not adopted (docs/design.md,
Performance notes, "Geodesic acceleration, prototyped").

| script | what it measures | run |
|---|---|---|
| `study.jl` | plain LM against geodesic acceleration at α 0.5, 0.75, 1.0, six models, SNR 50 and 200, 10,000 voxels each; `study_results.md` is its output of 2026-11-21 | `julia -t 1 --project=. prototypes/geodesic/study.jl` |
| `fallback.jl` | the variant without rejections, on the three sloppiest models | `julia -t 1 --project=. prototypes/geodesic/fallback.jl` |
