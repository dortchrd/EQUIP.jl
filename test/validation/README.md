# Validation tests

Expensive statistical and numerical validation, run apart from the unit and integration groups:
`Pkg.test(test_args = ["validation"])`, or `julia -t auto --project=test test/runtests.jl
validation`. CI is to run them nightly, on demand, and before merging into `main`.

None yet. To come here (docs/design.md, "Tests and benchmarks"): crlb against Monte Carlo at
n = 10 000 (bench/agreement.jl), the protocol search to Table 2's optima (Dortch et al., MRM
2018), convergence with SNR, and the statistical checks Rician fitting adds (the roadmap's tier 2).
A file here is listed in test/runtests.jl's `GROUPS.validation`.
