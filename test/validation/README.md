# Validation tests

Expensive statistical and numerical validation, run apart from the unit and integration groups:
`Pkg.test(test_args = ["validation"])`, or `julia -t auto --project=test test/runtests.jl
validation` (about 5 minutes), or one file by name (`... runtests.jl agreement`). CI runs them
nightly and on demand (.github/workflows/CI.yml, the `validation` job).

- agreement.jl: crlb against montecarlo at 10,000 copies, in Monte Carlo standard errors: T2
  decay with Gaussian noise; Dortch et al. 2018, Fig. 1 (the 4-point SIR, kmf assumed, four
  tissues, SNR 200); and where first order ends as the SNR falls (the 4-point below SNR 100, the
  5-point with kmf fitted below 800). Was bench/agreement.jl.
- rician_fits.jl: the Rician fits (`likelihood = :rician`) per model against least squares on the
  same data: the floor's bias removed in tissue, the SD at `ricianbound`, the fits without
  information ending on their caps, a σ map.
- secondorder.jl: crlb's second-order noise bias against montecarlo at 20,000 copies (DTI's
  derived quantities at SNR 20; least squares on Rician magnitude data), and its coefficient
  exactly, with antithetic pairs.

Still to come here (docs/design.md, "Tests and benchmarks", item 21): the protocol search to
Table 2's optima (Dortch et al., MRM 2018) and convergence with SNR. A file here is listed in
test/runtests.jl's `GROUPS.validation`; the measurements behind each file's thresholds are in
design.md (tier 2, "Step 6, built and measured" and "Step 8, built and measured").
