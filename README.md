# EQUIP.jl

*EQUIP: Estimation of QUantitative Imaging Parameters: fit, predict (CRLB), measure (Monte
Carlo) and design MRI protocols.*

MRI signal models and fast voxel-wise fitting, for quantitative maps (T1, T2, qMT, diffusion,
...). Built for millions of small fits: fewer than 10 parameters and fewer than 100 observations
per voxel. A custom Levenberg–Marquardt solver with automatic derivatives fits one voxel in a few
microseconds without allocating memory, and threads spread the voxels over your CPU cores.

The signal model is the center of the package, and four parts build on it:

- **Fit**: `fit` a voxel or a dataset, with the fit's statistics (`stderror`, AIC/BIC, ...).
- **Predict**: `crlb`, the bias and SD of each fitted parameter from derivatives, before
  scanning: the Cramér–Rao bound, biased when the fit assumes a wrong value.
- **Measure**: `montecarlo`, the same measured by simulating and fitting noisy copies, with
  crlb's prediction alongside.
- **Design**: `design`, the protocol that minimizes the predicted error over a set of
  `scenarios` (tissues, ranges of true values, conditions); its objective, `ProtocolDesign`, can
  be made once and reused.

The same four as a command, `equip`, for NIfTI (or MATLAB) data without writing Julia.

**The [guide](docs/guide.md) has every part in full**: the models, free and fixed parameters,
outputs and statistics, writing your own model, simulation, crlb and montecarlo, protocol
design, data files, the command line, and how the fitter works.

This is a prerelease (0.3.0): the interface may still change.

## Install and test

Requires Julia 1.11 or later.

```julia
] dev https://github.com/dortchrd/EQUIP.jl        # or the path of a clone: ] dev /path/to/EQUIP.jl
```

```sh
julia -t auto --project=. -e 'using Pkg; Pkg.test()'                # EQUIP (unit + integration)
julia -t auto --project=test test/runtests.jl unit                  # one group or file, directly
julia --project=MRImages -e 'using Pkg; Pkg.test()'                 # MRImages (image files)
julia --project=cli -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the command line
```

Start Julia with threads (`julia -t auto`) so fitting a dataset uses all cores.

The first `using EQUIP` after installing (or after a change to its code) takes about a minute:
EQUIP compiles the common cases (a fit of every built-in model, crlb, montecarlo, a protocol
design) and saves them, so that in every new session they run at once. Other combinations
compile on first use, a second or two (DKI's fit about 6 s, left out on purpose:
docs/design.md, "The prerelease"). When developing EQUIP, switch that step off with a
`LocalPreferences.toml` next to `Project.toml` (not committed) holding `[EQUIP]` and
`precompile_workload = false`: about 3 s per change.

## In short

```julia
using EQUIP

# Fit: a dataset from files to maps
data, info = readdata("sub-01_MESE.nii.gz")              # TE from the BIDS sidecar
mask = readmap("brain.nii.gz", info) .> 0                # checked against the data's geometry
res = fit(T2Decay(TE = info.found[:TE]), data; mask)    # threads; maps of T2 and S0
writemaps("results/sub-01_", res, info; stderror = true)  # T2map.nii.gz, ..., with sidecars

# Predict: SIR qMT (Dortch et al., MRM 2018), the fit assuming kmf = 12.5 in white matter
m = QMTSIR(; SIR_4POINT...)
c = crlb(m, nominal(QMTSIR, :SIR_WM_I); snr = 200, kmf = 12.5)
c.bias.PSR, c.sd.PSR                                     # −0.0282, 0.0116

# Measure: 10 000 noisy copies, Rician noise, crlb alongside
mc = montecarlo(m, Scenario(QMTSIR, nominal(QMTSIR, :SIR_WM_I); kmf = 12.5); noise = :rician, snr = 200)

# Design: the 4-point scheme over four tissues (Eq. 5)
ss = scenarios(QMTSIR, [:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion]; kmf = 12.5)
d = ProtocolDesign(QMTSIR(; SIR_4POINT..., magnitude = false), ss; snr = 200,
                   acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),
                   constraints = m -> m.TI .+ m.TD .- 0.5, scantime = m -> sum(m.TI .+ m.TD .+ 0.3))
r = design(d)                                 # Eq. 5: 0.2433, the paper's optimum
```

```bash
equip fit T2Decay t2.nii.gz                    # TE from the BIDS sidecar
equip fit DTI dwi.nii.gz                       # b and the directions from dwi.bval, dwi.bvec
equip crlb QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I,SIR_lesion --snr 200 --kmf 12.5
equip fit QMTSIR --help                        # a model's flags
```

## Models

Relaxometry (inversion and saturation recovery, variable flip angle, mono- and bi-exponential
T2, with offsets), diffusion (ADC, kurtosis, DTI, DKI with MK, AK, RK and MKT as DIPY's,
stick–zeppelin and ball–stick along each voxel's fiber, IVIM, the spherical mean technique with
free water and compartment T2s) and qMT (selective inversion
recovery), each in the time-constant and the rate form where both exist: the
[guide's table](docs/guide.md#built-in-models). Your own model: one equation with
`@mrimodel`, from diffusion compartments (`iso`, `stick`, `zeppelin`), or a struct and
`signal!` ([Writing a model](docs/guide.md#writing-a-model)).

## Examples

Each runs from the repository folder, `julia -t auto --project=. examples/NAME.jl`, in under a
minute:

| part | script | what it shows |
|---|---|---|
| fit | `fit_maps.jl` | a multi-echo T2 dataset from NIfTI files to maps, with standard errors and sidecars |
| predict | `crlb.jl` | SIR qMT's bias from a wrong kmf (Dortch 2018), over SNRs; DTI's FA and MD |
| measure | `montecarlo.jl` | montecarlo against crlb, in standard errors; T2's Rician bias at low SNR |
| design | `protocol_design.jl` | Dortch 2018's 4-point SIR optimization (Table 2) |
| models | `compartments.jl`, `one_free_parameter.jl`, `directional_models.jl`, `dki.jl` | models written with `@mrimodel` and compartments; one parameter free; stick–zeppelin with DTI's direction; DKI's indices |
| tests | `sir_wm_4point.jl`, `sir_4point_range.jl` | Monte Carlo datasets of the 4-point SIR scheme at SNR 200, as CSV |

`bench/solvers.jl` compares EQUIP with LsqFit, NonlinearSolve and FastLevenbergMarquardt over
test cases from easy to hard (its own environment: `julia --project=bench -e 'using Pkg;
Pkg.instantiate()'` once).

## More

- [docs/guide.md](docs/guide.md): the guide, every part in full.
- [docs/design.md](docs/design.md): the decision log, with the measurements behind each choice,
  and the roadmap.
- [docs/status.md](docs/status.md): where the work stands.
- [MRImages](MRImages/): MRI images in and out of files (NIfTI, BIDS sidecars, `.bval`/`.bvec`,
  MATLAB), a package of its own in this repository.

## Roadmap

In order of priority (agreed 2026-11-08; docs/design.md has each item):

1. **The prerelease** (this version).
2. **Bias from noise at low SNR**: Rician fitting for every model (magnitude data sit on a
   floor of about 1.25σ where the signal approaches the noise, and least squares reads it as
   signal: DTI's MD is 11% low at SNR 10); crlb's second-order noise bias (the fit's own, Box
   1971, and a derived quantity's), opt-in and checked against montecarlo; error propagation
   for any function of the parameters, and from uncertain inputs such as a measured B1 map.
3. **Protocol design**: a protocol from a number of points; objectives on derived quantities
   (a DKI protocol for MK); the worst case inside a box of true values.
4. **Derived quantities, finished**: standard-error maps (`FA_SE`), `compare` for derived
   quantities, an angular error for the principal direction, derived quantities for the
   other models.
5. **Models**: stick–zeppelin, ball–stick and IVIM (built, 2026-11-23); then MTR, MTsat, B1 and B0 maps,
   pulsed qMT (Yarnykh, Ramani), NODDI, free-water DTI, DKI with its constraints imposed.
6. **Starts, tested across all models**: plausible ranges per model, every start tested over
   them, a dictionary start (`BestOf` over several parameters, the signals made once per fit),
   and an example `datastart`.
7. **Peripheral nerve SMT**: a protocol evaluator and search, when the lab's values arrive.

Recorded for later: DKI's RTK, KFA and Kmax; Bayesian estimation; variable projection and log
coordinates; solver backends and GPU; digital phantoms; formats and MRImages as a package of its
own; Python and MATLAB front ends.

## License

MIT ([LICENSE](LICENSE)).
