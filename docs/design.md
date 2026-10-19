# EQUIP design (0.3)

The design decisions, recorded so the code and the discussion share one reference.

## Inputs: where each kind of information lives

| input | varies across voxels? | Julia form | example |
|---|---|---|---|
| **Model**: physics, acquisition and options | no | a `@kwdef` struct `<: MRIModel` | `QMTSIR(TI = [...], TD = 2.5, magnitude = true)` |
| **Parameters**: every quantity in the signal equation | each one can be free, a constant, a per-voxel map or a tie | values: a named tuple `p`; default roles: `parameters(m)` | PSR, R1f, kmf, B1, Sm |
| **Fit configuration**: which parameters are estimated in this fit | no | keywords named after the parameters | `kmf = 15.0`, `kmf = Free()` |
| **Data** | yes | N-d array, observations last | X×Y×Z×M |

Decisions behind this:

- **The acquisition lives in the model struct** (TI, TD, flip angles, ...), next to options
  such as `magnitude`.
  - The model alone then defines the forward model f(θ): `signal(m, p)`, `init(m)` and
    `fit(m, data)` need no extra argument.
  - The struct documents which acquisition a model needs, and a missing field fails at
    construction.
  - Protocol optimization just builds a new model per candidate, which costs nothing.
  - Front ends can build any model from named fields through the `@kwdef` keyword
    constructor.
- **The model never holds parameter values.** One model object is reused for millions of
  voxels, while `p` changes at every LM step.
- **A parameter keeps its place in the model whether it is fixed or estimated.** Its role
  (`Free(init; lower, upper)` / `Fixed(value)`) sits next to it, so the starting value, the
  bounds and the default value live in one place. A fit changes roles through keywords, so
  sensitivity analyses (e.g. fix kmf at different values) never edit the model.
- **Model structs use `@kwdef`**, with one type parameter per vector field and no inner
  constructors (no Parameters.jl):
  - The only invariant, equal lengths of the per-observation vectors, is checked generically
    by `nobs(m)` when `init` and `signal` run.
  - A field that may be one number for all observations (TD, TR) gets a one-line method that
    fills it to a vector. So `signal!` always indexes vectors (`TD[j]`), and any other code
    reading the protocol (scan time, plots) sees vectors too.
- **`signal!` unpacks by name**: `(; TI, TD, magnitude) = m` and `(; PSR, ...) = p`. Never by
  position, because the order of `p` depends on which parameters are free.
- **Multiplication is written out**: `2 * mean(TE)`, not `2mean(TE)`. Julia allows a number
  directly before a name, but readers coming from MATLAB or Python may not know it, and it binds
  tighter than `/` (`1 / 2x` is `1 / (2x)`).
- **The supertype is `MRIModel`**, not `AbstractModel`. The generic name is used by other
  packages (e.g. AbstractMCMC.jl), and exporting the same name from two packages forces users
  to qualify it.
- **One function, `fit`, chosen by the shape of the data.** `fit(m, y::AbstractVector)` fits one
  voxel and returns plain numbers; `fit(m, data::AbstractArray)` fits every voxel (observations
  last, any number of spatial dimensions) and returns maps. Both return a `FitResult` with
  the same fields (`params`, `sse`, `iterations`, `status`, `model`, `settings`), numbers or
  maps. `init` + `fit!` remain for user-written loops, and return a `FitResult` too.
- **Fit the quantity you will report; separate models for each form.** Mono-exponential T2 decay
  comes as `T2Decay` (T2) and `R2Decay` (R2), plus offset versions. Each form has its own
  struct, `parameters` and docstring; the equation is written once, in the rate form's
  `signal!` (a short loop that reads like the equation, no conditionals). The time form converts
  and calls it, e.g. `signal!(out, m::T1IR, p) = signal!(out, R1IR(m.TI, m.magnitude),
  (; p.M0, p.Sf, R1 = 1 / p.T1))`.
  - The conversion costs almost nothing: the rate model wraps the same acquisition vectors (no
    copy), and ForwardDiff carries the derivatives through 1/T. The fits are the same, bit for
    bit, and within 2% of the speed (Performance notes).
  - Always time → rate, never rate → time: the rate's bound R = 0 (a flat curve) is a valid
    value, but T = 1/0 is not (and its derivative would be NaN); T stays ≥ `_TMIN`, so 1/T is
    always safe.
  - Offsets stay separate models with their own loop. Measured alternatives (500 voxels, SNR
    100, 8 and 32 echoes, identical results in every case):
    - C as a parameter of `T2Decay`/`R2Decay`, fixed at 0 unless freed: fits without an offset,
      the common case, 5–11% slower (adding C to every echo), to save two models. The offset is
      also an approximation, better chosen by name than offered as an option of every decay.
    - The offset model calling the decay model, then adding C in a second pass: offset fits
      3–10% slower, to save one loop, and the equation S0·exp(−R2·TE) + C split in two.
  - Why both forms: standard errors, CRLB and Monte Carlo then come out directly for the
    reported quantity. Each voxel's estimate and its standard error do convert exactly between
    forms (least squares gives R̂ = 1/T̂ when both reach the same minimum; SE(T̂) = SE(R̂)/R̂²),
    but averages over voxels do not (mean(T̂) ≠ 1/mean(R̂)), and the API is simplest without
    conversions.
  - The rate form is more robust (measured: 500 noisy voxels per case, a fit counted bad if its
    SSE exceeds the SSE at the true parameters). On nearly flat curves the time form runs off
    to huge T and stops at max iterations, whatever its start (8 echoes, TE 10–80 ms, SNR 20:
    152 of 500 voxels at T2 = 2 s, 245 at 10 s, from mean(TE) or from the log-linear start);
    the rate form settles near 0 and converges in all of them. With a start of mean(x), the
    time form also used to fail silently when the true decay was much faster than the start,
    stepping to a tiny T where the signal at every x > 0 vanishes and so does the gradient (all
    500 voxels for a time constant 1/4.5 of the start, SNR 50). The dead-zone guard
    (Performance notes) removed those failures: 0 of 500 now, from mean(TE) and from T2 = 10 ms.
  - The T1 models come in both forms too (`T1IR`/`R1IR`, `T1IRTD`/`R1IRTD`, `T1SR`/`R1SR`,
    `T1MFA`/`R1MFA`). From the default starts (T1 = 1 s, R1 = 1/s), 200 voxels at SNR 100:
    - Short T1: the T1 forms failed silently in most or all voxels (IR at 0.1 s; IRTD at
      0.1–0.3 s; SR at ≤ 0.2 s; MFA at 0.05 and 0.2 s); the R1 forms only where no form can work.
      The dead-zone guard in the solver (below) fixed SR and MFA completely; what remained for IR
      and IRTD was magnitude inversion recovery's local minima (polarity), not the dead zone,
      and the grid-search start (below) fixed that.
    - Long T1, low SNR: for inversion recovery with T1 = 4 s at SNR 20 (TIs up to 3.2 s), the
      two forms ended at different answers in 371 of 2000 voxels from a single start, each in a
      different local minimum. With the grid-search start they agree except in 15 voxels whose
      noisy data are best fitted by T1 → ∞ (an almost straight line; SSE below that at the true
      parameters); both forms stop there at max iterations. MFA's R1 form needs 1.5–2× more
      iterations for long T1.
    - Why the time forms fail: the rate form's bound, R = 0, is a flat curve whose gradient is
      not zero, so a fit can come back from it. The time form's bound, T → 0, is a dead zone:
      every sample has fully decayed or recovered, the gradient is zero, and a step that
      overshoots there stays. The solver now rejects such steps (next item).
  - Offsets are separate models (`T2DecayOffset`, `R2DecayOffset`), with C free by
    default. A constant offset is a common approximation for the magnitude noise floor, not the
    correct model (the Rician bias depends on the signal level); Rician-likelihood fitting is
    on the roadmap.
  - Diffusion has its own model (`ADCDecay`, b and ADC): the same equation with
    diffusion's names, repeated on purpose. Later the models may be grouped by area
    (relaxometry, diffusion, MT).
  - Harder models, added to test the fitter on hard cases as the roadmap goes on:
    - `T2BiexpDecay`/`R2BiexpDecay`: two T2 components, both forms, no offset version (hard
      enough already). Four strongly correlated parameters with real local minima, and the
      two components can swap labels: (fshort, T2short, T2long) and (1 − fshort, T2long,
      T2short) give the same signal. From the default starts the T2 form swapped them in 196
      of 200 voxels at SNR 100 (even on noise-free data), the R2 form in none, so each model
      reorders after the fit (`relabel`, a model hook called once per voxel; the shorter T2
      is reported as T2short, only when all three parameters are free). With that, both forms
      report the same answer where they reach the same minimum (96 of 100 noisy voxels; 1
      converged to different local minima in the two forms, 3 hit max iterations). On
      mono-exponential data the fit is degenerate but exact (SSE ≈ 0): the T2 form put both
      T2s at the true value, the R2 form put fshort ≈ 1; on noisy mono-exponential data 43 of
      200 voxels hit max iterations. The dead-zone guard did not stop fshort from reaching its
      bound.
    - `ADCKurtosis`: diffusion kurtosis along one direction or of direction-averaged data, a
      rate form only (ADC is a rate). K in [0, 3]. The signal rises again at high b once
      b·ADC·K > 3; that limit involves two parameters, so it is not a bound, and is not
      enforced.
  - Considered and rejected: one shared loop with an optional offset (a conditional in the
    loop to skip adding C = 0); generic parameter names (T, R, x); a parameter for both T2 and
    R2 tied by a function (outputs only include free parameters); the name as a type parameter
    (`EchoDecay{:T2star}`); models generated with `@eval`.
- **Model names put the quantity first, then the method**: `T1IR`, `T1SR`, `T1MFA`, `T2Decay`,
  `R2Decay`, `ADCDecay`, `QMTSIR` (as qMRLab does with `vfa_t1`, `mono_t2`, `qmt_sirfse`). Tab
  completion then groups models by quantity, rate versions read the same way (`R1IR`, `R2Decay`), and new methods fit in (`QMTSPGR`).
  Acronyms stay upper case, as in Base (`QRCompactWY`). Files are grouped by area:
  `src/models/relaxometry/`, `diffusion/`, `qmt/`.
- **Standard function names are shared, not redefined.** `fit`, `fit!` and `nobs` are StatsAPI's
  functions (the interface behind StatsBase, GLM, Distributions) and `init` is CommonSolve's
  (SciML). We add methods for our types, so loading those packages alongside causes no clash.

## Writing a model

```julia
@kwdef struct QMTSIR{A<:AbstractVector,B<:AbstractVector} <: MRIModel   # 1. acquisition + options
    TI::A
    TD::B
    magnitude::Bool = true
end
QMTSIR(TI, TD::Real, magnitude) = QMTSIR(TI, fill(TD, length(TI)), magnitude)   # one TD for all

parameters(::QMTSIR) = (                          # 2. every parameter with its default role
    PSR = Free(0.1; lower = 0, upper = 1),
    R1f = Free(1.0; lower = 0.05, upper = 10),
    Sf  = Free(-0.95; lower = -1.2, upper = 0),
    M0f = Free(:scale; lower = 0),             # starts at the least-squares signal scale
    kmf = Fixed(12.5),
    R1m = Fixed(:R1f),                         # tie: always equal to R1f
    Sm  = Fixed(0.83),
    Mzf_td0 = Fixed(0.0),                      # Mz at the start of TD (fraction of equilibrium):
    Mzm_td0 = Fixed(0.0),                      # 0 = saturated by the readout
)

function signal!(out, m::QMTSIR, p)               # 3. one voxel's signal at every observation
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf, R1m, Sm, Mzf_td0, Mzm_td0) = p
    @inbounds for j in eachindex(out, TI, TD)  #    one length check, then TI[j], TD[j]
        ...
    end
end
```

Rules for `signal!`: no allocation, no state, generic in the number type of `p` (dual numbers
pass through it). Loop with `eachindex(out, <every per-observation field>)`: one length check
per call (no measurable cost), after which `@inbounds` is safe whoever calls `signal!` (the
fitter, `signal`, or future CRLB / protocol code with its own buffers). Measured: `@inbounds`
gains ~4–7% in a `signal!` loop and up to 1.16–1.41× over whole fits (mostly the solver's own
loops); `@simd` gains nothing real here (`exp` prevents vectorization). `signal(m, p)`
(provided) allocates the output and fills in fixed defaults, for simulation.

Optional hooks: `datastart(m, y, p) = (T2 = value,)` computes starting values from each voxel's
data, for the parameters declared `:data`, e.g. the log-linear fit of `T2Decay` (Performance
notes); `p` holds the voxel's fixed values (e.g. B1). `relabel(m, θ)` chooses the labels to report
when they can swap without changing the signal (the two components of `T2BiexpDecay`).

## Parameter roles

- `Free(start; lower, upper)`: estimated. The start says where each voxel's starting value
  comes from:

  | `start =` | per voxel | given by |
  |---|---|---|
  | a number or a map | start there | model default (a number) or the call |
  | `:scale` | least squares given the other starts, a = (s·y)/(s·s) | the model only |
  | `:data` | the model's estimate from this voxel's data (`datastart`) | model default or the call |
  | `BestOf(values)` | try each value (`:scale` solved for each), start from the lowest SSE | model default or the call |

  Models declare theirs in `parameters(m)`, so a reader sees where each start comes from:
  ```julia
  parameters(m::T1IR) = (M0 = Free(:scale; lower = 0), Sf = Free(-0.95; ...),
                         T1 = Free(BestOf(_t1grid(m.TI)); lower = _TMIN))
  parameters(m::T2Decay) = (S0 = Free(:scale; lower = 0), T2 = Free(:data; lower = _TMIN))
  ```
  A fit changes them like any start: `T1 = Free(start = 0.8)` (one start, no search),
  `T1 = Free(start = BestOf(logrange(0.05, 5, 50)))` (other values), `T2 = Free(start = 0.05)`
  (no data start). The model's defaults are the automatic choice, so there is no `:auto`.
  - `:scale`: s is the signal at scale 1 given the other starts. This one generic rule
    replaces data normalization. Only the model declares it, since the rule assumes the signal
    is proportional to the parameter: `setup` refuses `T1 = Free(start = :scale)` (before this
    check it ran, giving T1 a meaningless start). A fit can turn it off,
    `M0 = Free(start = 1000.0)`, but not move it.
  - `:data`: only for parameters the model's `datastart` estimates; `setup` asks it once, on
    placeholder data, which ones it covers, and refuses the others. `datastart` returns its own
    fallback (T2Decay: mean(TE) when the data do not decay), since a `:data` parameter has no
    number to fall back to; a NaN makes that voxel's fit fail (status `Failed`), not patched.
  - `BestOf(values)`: values outside the bounds are dropped, and `setup` refuses a `BestOf`
    with none left. The parameter holds the middle value as its placeholder, the start if no
    value can be scored. One `BestOf` parameter per fit for now (several: roadmap item 4).
  - **Decision: the name `BestOf`.** It was planned as `Grid`, but the values can be anything
    (`[0.3, 0.8, 1.5]`), and they form a grid only when several parameters are combined. The
    name states the rule: "start at the best of these". Other candidates: `Candidates` (names
    the values, not what happens to them), `OneOf`/`Try` (vague), `Grid` (suggests even
    spacing; a generic name other packages use). A wrapper at all because a plain vector start
    already means a per-voxel map. The internal helpers that do make log-spaced grids keep
    their names (`_t1grid`, `_loggrid`).
  - **Decision: starts declared in `parameters(m)`**, replacing the `startgrid(m)` hook and a
    `userstart` list in `setup` (which tracked which parameters the call had given a start, so
    that the data start and the grid would leave them alone): a start given in the call now
    simply replaces the model's `:data` or `BestOf`. Also replaces a planned per-fit `starts`
    option, so no fourth reserved option name. Results unchanged (fingerprint of starts and
    fits for every model, both precisions, with and without overrides: identical).
- `Fixed(value)`: not estimated. `value` is a number, a per-voxel map, or a tie, resolved after
  the free values are merged in, so derivatives flow through it. A tie may not refer to another
  tie.
  - **By name**, `Fixed(:R1f)`: always equal to R1f. The common case. It is checked in `setup`
    (the name must exist and must not be a tie), can be written in config files and on a command
    line later, and becomes `_Tie{:R1f}()`, so resolving it is a field read whose name the
    compiler knows.
  - **By function**, `Fixed(p -> 0.8 * p.R1f)`: for real relations; a named function for long
    ones. Cannot be checked before it runs.
  - An R1 map as a constraint on SIR needs no extra feature: with R1m = R1f the slow two-pool
    rate is exactly R1f, so `R1f = r1map` does it. (A formula R1f(R1obs, …) would
    only be needed if R1m ≠ R1f; that would require auxiliary per-voxel inputs. Not planned.)

Fit-time keywords, one per parameter, in `fit`, `init` and `refit`: a value fixes the parameter
(`kmf = 15.0`, `Sm = sm_map`, `R1m = :R1f`); `Free(...)` estimates it, and whatever it leaves
out keeps the model's default (`kmf = Free()`, `R1f = Free(start = r1map)`,
`Sf = Free(lower = -1.5)`). A parameter fixed by default and freed starts at its fixed value,
with no bounds unless given; one fixed by default to a tie or a map needs a start. A
non-finite value in the data, a fixed map or a start map marks that voxel invalid (status
`InvalidData`).

Three keyword names are options, not parameters: `mask`, `precision` and `solver`
(`solver = LM(maxiter, xtol, ftol, gtol, damping)`). `setup` refuses a model with a parameter
of one of those names, and a keyword that is neither a parameter nor an option.

**Decision: per-parameter keywords** (replacing the grouped `fixed = (kmf = 15.0,)`,
`free = (:kmf,)`, `start = (R1f = 0.8,)`).
- Gained: one keyword per parameter, read as the paper writes it; no one-element named tuples
  with their trailing comma; one place for a parameter's start and bounds (`Free`), instead of
  `start` and `free` both setting starts; no way to name a parameter both fixed and free in one
  call; `refit` and settings merge per parameter with plain `merge`.
- Lost: parameters and options share one namespace. Kept safe by reserving few names, grouping
  the solver's settings into one `solver = LM(...)` value (so `maxiter`, `damping`, tolerances
  never clash with a parameter), and checking the names in `setup`. A future option takes a
  name no model uses, and joins `_OPTIONS`.
- It needs no per-model code: `setup` loops over the keywords and checks each against
  `parameters(m)`.
- From the Julia fitting packages: `refit` follows SciML's `remake` (a new problem from an old
  one, with changes); `solver = LM()` follows SciML's `solve(prob, Alg())`, an algorithm as a
  value with its own settings. EQUIP keeps its one-call `fit`, which already runs `init` and
  `fit!` (SciML's `init`/`solve!`). From LsqFit.jl: follow StatsAPI's names for the statistics
  of roadmap item 3 (StatsAPI's `dof` is the number of parameters and `dof_residual` is n − p;
  LsqFit's `dof` means n − p), and separate weights with a known σ from an estimated σ.

## Fitting pipeline

1. `setup` merges the fit's keywords into `parameters(m)`, splits free from fixed, and returns
   a `FitSpec{F,P,T}`. The free names `F` and their count `P` are type parameters, so the
   compiler generates code specialized to them. Without that, building
   `NamedTuple{F}(θ)` from a run-time list of names would be type-unstable and slow.
2. `init` checks the acquisition lengths (`nobs`) and rebuilds the model with its numeric
   vectors converted to the working precision. Together with solver options and buffers this
   gives a `FitCache`, made once per fit call; a dataset fit copies its buffers for each chunk
   of voxels (one task each).
3. `fit!` handles one voxel (`fit(m, y)` with a vector is `fit!(init(m), y)`): pick its map
   values, build θ0 (numbers and maps as given; `:data` parameters from `datastart`; then a
   `BestOf` parameter at its best value, else the `:scale` start), run LM.
   Each LM evaluation does `p = resolve_ties(merge(NamedTuple{F}(seeded duals), fixed))`, then
   `signal!(buf, m, p)`, then sums SSE, JᵀJ and Jᵀr from `buf`. The m×P Jacobian is never
   stored and nothing is allocated.
4. `fit(m, data)` reshapes the data to voxels × observations (a view) and turns the mask into an
   index list. It splits the list into about 64 chunks per thread (at least 16 voxels each) and
   fits each chunk in a task with its own cache. Many small chunks keep threads busy when slow
   voxels are clustered; the floor keeps the ~1 µs per-task overhead negligible. It returns a
   `FitResult` of maps; inside, the loop stores a lean per-voxel result (`_VoxelFit`) into the
   maps, and `fit!` wraps the same per-voxel result with the model and settings.

LM details: Marquardt scaling with a monotone diagonal (MINPACK), Nielsen damping updates,
bounds by projection with an active set, Cholesky without exceptions.

## Status codes

`FitStatus` names, stored in one byte (codes in brackets): `NotFitted` (0, outside the mask),
`ConvergedX` / `ConvergedF` / `ConvergedG` (1–3: step, cost, gradient), `MaxIterations` (4),
`Failed` (5, non-finite model at the start), `InvalidData` (6). A dataset's status map is a
`FitStatus` array (1 byte per voxel, prints names); `UInt8.(res.status)` gives the codes.
The names are not exported (`Failed` is too generic): `EQUIP.MaxIterations`, or `isconverged`.

## FitResult

One type for every fit (`src/fit/result.jl`): `params`, `sse`, `iterations`, `status`, plus
`model` (as fitted: its acquisition in the working precision) and `settings` (the call's
keywords, by reference: the parameter keywords, `precision`, `solver`, and `mask` for a
dataset). Numbers for one voxel, maps for a dataset; `_onevoxel(res)` (is the status a single
`FitStatus`?) tells them apart at compile time.
- `fit!` returns it too and still allocates nothing: the model and settings are references in
  an immutable struct. Measured, best of 3 alternating runs: `fit!` T2Decay 1841 → 1852 ns,
  QMTSIR 2951 → 2948 ns; a 20k-voxel dataset 39.9 → 40.2 ms (it now also fills an iterations
  map).
- `fitted(res)` / `fitted(res, i...)` rebuild a curve with `signal`, from the model, the fitted
  parameters and the fixed values in `settings` (numbers, roles, ties, or that voxel's value of
  a map); `residuals(res, y)` / `residuals(res, data, i...)` subtract it from the data. Both are
  StatsAPI's names. A one-voxel result fitted with fixed maps (`fit!` with `voxel`) cannot
  rebuild its curve: the voxel index is not stored.
- Refit: `fit(res.model, data; res.settings...)` repeats the fit with the same choices.
  `refit(res, data; changes...)` merges the changes into the settings and starts every free
  parameter not changed from its previous estimate (`name = Free(res.params.name, lower,
  upper)`, keeping the bounds of the first fit), e.g. `refit(res, data; kmf = Free())`. A voxel
  whose previous estimate is NaN becomes `InvalidData`, as for any non-finite start map.
- Display: one line in `print` and inside collections; at the REPL a summary (one voxel: status,
  iterations, SSE, values, fixed parameters; a dataset: voxels fitted, % converged, counts per
  status, median and interquartile range of each parameter over converged voxels, fixed
  parameters).

## Simulation

`src/sim/` (uses models and fit): `simulate`, the tissue presets, phantoms and `compare`.

- **SNR is relative to the scale parameter** (the one declared `Free(:scale)`: M0, S0, M0f),
  σ = scale/snr, voxel by voxel: Dortch et al. 2018's "SNR = 200 at thermal equilibrium". It
  does not depend on the protocol, as σ on a scanner, so protocols compare at equal noise. The
  measured signal can be far below the scale (MFA at low flip angles, SIR schemes that never
  reach full recovery), so image SNR and this SNR differ. `σ` gives the noise directly (e.g.
  from an image background); an SNR relative to the measured signal can be added if needed.
- **Rician noise by default** (|S + σ·(n₁ + i·n₂)|), Gaussian for models fitted to signed data
  (`magnitude = false`). For magnitude inversion recovery, adding complex noise to the signed
  signal and taking the magnitude depends only on |S|, so Rician noise on the model's |S| is
  exact.
- **Tissue presets are published values with their source**: Stanisz et al., MRM 2005;54:507,
  Table 1 (T1 and T2 at 3 T, 37 °C, in vitro: animal tissues and human blood; white matter is
  bovine corpus callosum, gray matter cortex, so in vivo values differ); Dortch et al., MRM
  2018, Table 1 (the four SIR classes, with Sm = 0.83, Sf = −1.0). CSF (Stanisz has none): T1 =
  4.124 s, the mean of the three in vivo inversion recovery studies in the review of Bojorquez
  et al., MRI 2017;35:69, Table 1c (4391, 3817, 4163 ms; the VFA values spread from 4.2 to
  6.9 s); T2 = 1.75 s, the middle of the usual 1.5–2 s at 3 T, a placeholder (the review has no
  CSF T2). Users give their own values as a named tuple, or change one with `merge`. `truth(m, tissue)` converts T1/T2 to rates by name, sets
  the scale parameter to 1, and refuses to guess free parameters that are not tissue properties
  (the inversion efficiency Sf of the inversion recovery models, B1): they are given as keywords.
- **Phantoms**: `phantom(labels, tissues)` fills one map per parameter from label => truth
  pairs, NaN elsewhere, so the background gives NaN data and is left out with
  `mask = labels .> 0`. `labelshapes(dims)`: nested ellipses (labels 1, 2) and a small disc
  (3), 2-D or 3-D.
- **`compare`** uses every fitted voxel, converged or not (failures are part of the real
  performance; % converged is reported beside them), overall and per label. Rows are named
  tuples, so `c.rows` is a table for DataFrames or CSV. No timing: speed is measured in
  `bench/`.
- **Check against Dortch et al. 2018, Fig. 1** (RRMS lesion, Rician noise, SNR 200, 10,000
  realizations): PSR 5.09% (5-point, kmf free) and 4.74–4.75% (4-point, kmf fixed at 12.5, the
  bias the paper shows), R1f SD 0.015 and 0.017. The 5-point PSR distribution has a heavy upper
  tail (99.9th percentile up to 11%), so its SD varies with the seed (0.77–0.90% over five
  seeds) while a robust spread (IQR/1.349) stays at 0.72–0.75%: worth knowing when comparing
  Monte Carlo SDs with the CRLB (item 3). `test/` runs a 2000-realization version.

## Package structure and name

Julia 1.11 or later (`julia = "1.11"` in Project.toml), raised from 1.10 (the long-term-support
release) so examples and code can use `logrange`, as in `BestOf(logrange(0.05, 5, 50))`. 1.11
also brings the `public` keyword, for names that are part of the API without being exported
(e.g. the `FitStatus` values, now reached as `EQUIP.MaxIterations`); not used yet.

**One package, in layers.** Each layer only uses the layers before it:

```
src/
  EQUIP.jl             the module: dependencies, exports, includes
  models/              interface (MRIModel, Free, Fixed, parameters, signal!, signal, nobs);
                       builtin.jl (includes), then one folder per area:
    relaxometry/       T1IR, T1SR, T1MFA, T2Decay (+ R1/R2 and offset versions), T2BiexpDecay
    diffusion/         ADCDecay, ADCKurtosis
    qmt/               QMTSIR
  fit/                 levmar.jl (solver), setup.jl (roles -> FitSpec), result.jl
                       (FitResult), starts.jl (starting values: :data, BestOf, :scale),
                       cache.jl (init, fit!), fit.jl (fit: one voxel or a dataset, threaded)
  sim/                 simulate (noise), TISSUES / tissue / truth (presets), phantom and
                       labelshapes, compare (fit vs truth)
  analysis/            (later) Monte Carlo, CRLB, sensitivity
  design/              (later) protocol objectives (Eq. 5, Eq. 6), scan time
  io/                  (later) loaddata, protocol files, units, savemaps, PAR/REC reader
ext/                   (later) package extensions: OptimExt (optimize_protocol), and one per file
                       format package (NIfTI, MAT, NPZ, DICOM, JSON3, MRIFiles)
test/                  correctness and deterministic performance checks (no allocation, type
                       stability); no timings
bench/                 (later) own Project.toml (BenchmarkTools, LsqFit): speed suite + accuracy
                       studies, results.md
```

- Why not several packages: CRLB (biased, Eq. 3), Monte Carlo and sensitivity analysis all
  need the model, its Jacobian and the fitter. Separate packages would mean keeping versions
  in step while the API is still changing, and none of these features adds a heavy dependency.
- Heavy optional dependencies (an optimizer, file formats) go in package extensions (Julia ≥
  1.9): their code loads only when the user also loads that package.
- The protocol objectives stay in the core: the signal can already be differentiated with
  respect to TI/TD (dual numbers pass through the model's vectors), so any optimizer can use
  them.
- Python and MATLAB wrappers will be separate, thin packages (PyPI, MATLAB toolbox).
- When to split later: if another project wants only the models (e.g. model-based
  reconstruction), or loading time becomes a problem. The layering makes that a move of
  folders, following the one-repository-several-packages pattern of KomaMRI (KomaMRIBase,
  KomaMRICore, ...) and MRIReco (MRIBase, ...).

**Name.** `EQUIP`: **E**stimation of **QU**antitative **I**maging **P**arameters (the expansion
can grow to "…Parameters and Protocols" as protocol design becomes central). A pronounceable
word, specific about what the package does; ".jl" already says "in Julia" (Julia's naming
guidelines advise against "Julia"/"Ju" in names). All-caps acronyms are fine in Julia (DECAES,
DICOM). Earlier names: VoxelFit (only fitting), MRISignalModels (descriptive but long).
Alternatives considered: Quanta, Saguaro, Larmor, DESQ, QUIJOTE; avoided QUIT (an existing C++
qMRI toolbox), QUEST (an MRS method), QED (quantum-electrodynamics packages), and the
institute's name (implies endorsement). Checked against the General registry (Oct 2026):
unregistered. `QuantitativeMRI` is taken (aTrotier/QuantitativeMRI.jl).

## Performance notes

- **QMTSIR `signal!` follows Li et al. 2010 (Eqs. 3–6) step by step**: recovery over TD from the
  start values, the inversion pulse, recovery over TI. It uses Li's names (R1⁺, R1⁻, bf⁺, bf⁻,
  and Mz as a fraction of equilibrium), so each line can be checked against the paper. Li's
  Eq. 4 already allows any starting values, so `Mzf_td0`/`Mzm_td0` (Mz at the start of TD) cost
  nothing: whole fits measured the same with and without them. Verified against the matrix form
  (Dortch 2018, Eq. 1, extended to nonzero start values; ~1e-14), a numerical (RK4) solution of
  Li's Eq. 2 (~3e-13), and finite-difference Jacobians (~6e-7, the finite differences' error).
- **What made it fast, measured with `@belapsed`** (5-point scheme / 16-point scheme, whole fits):
  - Naming each `exp` once (`E_td⁺`, …) matters: writing them out where used is 1.7–2× slower,
    because the compiler does not merge repeated `exp` calls.
  - Naming plain arithmetic does not matter: the compiler shares repeated products itself, so
    intermediate names (the old `bf1`, `bf2`) are for readability only.
  - Moving parameter-only work out of the loop gains nothing measurable (the compiler does it).
  - bf⁻ from bf⁺ + bf⁻ = Mf(0) − 1 (Li Eq. 3 at t = 0) instead of Li's full expression: ~7%
    faster (2.2 vs 2.37 µs; 4.1 vs 4.4 µs). Multiplying by 1/(R1⁺ − R1⁻) instead of dividing:
    ~3%.
  - A rearranged version (all parameter-only work folded into six coefficients, so
    `s = M0f + A·exp(-R1⁺·TI) + B·exp(-R1⁻·TI)` with A, B depending only on TD, reused when TD
    repeats) is 1.05× faster on the paper's 5-point scheme and 1.6× with one TD. Not used:
    harder to check against the paper. Revisit if Monte Carlo or protocol optimization on
    constant-TD schemes becomes speed-limited.
- **The matrix form written literally is slower**: Base `exp` on 2×2 matrices is ~180× slower
  than the closed form and cannot take dual numbers; 2×2 StaticArrays with `exp(A·t)` in closed
  form (eigen-decomposition) is 1.3–1.4× slower and works with duals. The tests use Base `exp`
  as an independent reference. A reasonable route for models without a convenient closed form.
- **Time forms call the rate forms**, e.g. `signal!(out, m::T2Decay, p) = signal!(out,
  R2Decay(m.TE), (; p.S0, R2 = 1 / p.T2))`. So R2 = 1/T2 is computed once per call and the loop
  multiplies, instead of dividing every TE by T2: 5–16% faster whole fits than dividing
  (minimum of 10 alternating `@belapsed` runs; this machine varies 15–25% between single runs).
  Calling the rate form instead of repeating its loop: 500 noisy voxels gave bit-identical
  results for all six time models (T1IR, T1IRTD, T1SR, T1MFA, T2Decay, T2DecayOffset), and one
  voxel's fit was 0.4–2% slower (about 10 ns; T1IR 1.71 → 1.74 µs, T1MFA 0.70 → 0.71 µs, minimum
  of 3 alternating runs), at the level of the noise (the unchanged R2Decay varied by 8% in the
  same runs). `@inline` on the one-liners is not needed: Julia already inlines them (the
  compiled `normal_equations` calls the rate model's `signal!` directly, checked with
  `code_typed`), and adding it changed one voxel's fit by −1.6% to +0.9%, the same as for the
  unchanged rate models in the same runs. Where it does matter: `_score` in `src/fit/starts.jl`
  (the scale solve and SSE of a candidate start), too large to be inlined automatically and
  called 30 times per voxel by the grid start; without `@inline`, T1IR fits were 8% slower. In practice the whole question is
  small: a 128×128×90 dataset with 32 echoes and 25% of voxels in the mask (368k voxels) fits
  in 0.66–0.76 s on 1 thread and 0.16–0.19 s on 4, whichever variant (SIR 5-point on the same
  mask: 0.93 s and 0.24 s).
- **The solver rejects steps into a dead zone** (`_lost_influence` in `src/fit/levmar.jl`): a
  trial step after which a free parameter no longer affects the signal (its ‖Jⱼ‖², the diagonal
  of JᵀJ, collapses by more than a factor 1/eps) is rejected like any bad step. The typical case
  is a time constant projected onto its lower bound (1e-6 s), where every sample has decayed or
  recovered: the derivative is exactly zero, the solver can never move it again, and yet the
  step lowered the SSE (the other parameters adapt), so it used to be accepted.
  - Measured (200 voxels, SNR 100, default starts): T1SR at T1 = 0.05/0.1/0.2 s and T1MFA at
    0.05/0.2 s went from 200/200 bad fits to 0; T1IRTD at 0.3 s from 199 to 79. Fits that end on
    a bound (R2 = 0 for flat curves, PSR = 0 without MT) are identical; typical fits of all
    models give identical answers in the same iterations; a 368k-voxel dataset fits in the same
    time (T2Decay 0.36–0.38 s, QMTSIR 0.31 s on 4 threads). Per voxel, the check is written branch-free over the
    diagonal (`any(...)` on static vectors): a loop with early return made QMTSIR fits 7%
    slower (2.46 → 2.66 µs), the branch-free form costs nothing measurable (≤ 2%, within noise).
  - Precedent: none that we know of for exactly this rule, so it is EQUIP's own guard. Related:
    Transtrum & Sethna's "parameter evaporation" (fits drifting to where a parameter stops
    mattering) and their acceptance test on geodesic acceleration; SciPy's `least_squares`
    trust-region-reflective method, which keeps iterates strictly inside the bounds (step-back
    θ = max(0.995, 1 − ‖g‖), reflection, Coleman–Li scaling); lmfit/MINUIT, which remove bounds
    by a change of variables. LsqFit.jl projects onto the bounds as we did, and has the same
    weakness. The standard acceptance threshold (actual/predicted decrease > η) does not catch
    it: the jump lowered the SSE by 39%, and was accepted with η up to 0.25.
  - Rejected alternative: moving only part of the way to a bound (a naive strictly-feasible
    step, without Coleman–Li scaling or reflection). It fixed SR and MFA too, but fits that
    should end on a bound stalled: QMTSIR with PSR = 0 hit max iterations in 401 of 1000 voxels,
    some with a 4.4× worse SSE.
  - Not a solver issue: whether the protocol samples the curve well. Short T2 failed only when the
    first echo came at ≥ 2× T2 at low SNR (an earlier first echo removed every failure); long T2
    or T1 is a question of precision (T2 = 2 s from TE ≤ 320 ms: 0.6–2.9× the truth at SNR 10),
    for CRLB and protocol design.
- **Grid-search start for magnitude inversion recovery** (`T1 = Free(BestOf(_t1grid(m.TI)))`;
  `_best_start` in `src/fit/starts.jl`; it was the `startgrid` hook when measured). |S| has local minima: from one start a fit can settle with the signal
  null in the wrong place. Before LM, each voxel evaluates the signal at 30 values of T1
  (log-spaced from 1/5 of the shortest TI to 5× the longest; the R1 forms use the same values
  as rates), the `:scale` parameter at its least-squares value for each, and starts from the
  lowest SSE. Other parameters keep their starts (Sf = −0.95).
  - Measured (200 voxels, bad fit = SSE above that at the true parameters, T1 form / R1 form):

    | T1 (s) | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1 | 2 | 4 |
    |---|---|---|---|---|---|---|---|---|
    | IR, SNR 100, start T1 = 1 s | 200/200 | 200/0 | 0/0 | 3/0 | 0/0 | 0/0 | 0/0 | 200/200 |
    | IR, SNR 100, grid | 0/0 | 0/0 | 0/0 | 2/1 | 0/0 | 0/0 | 0/0 | 0/0 |
    | IR, SNR 20, start T1 = 1 s | 200/200 | 148/36 | 11/0 | 0/0 | 0/25 | 0/0 | 53/54 | 152/188 |
    | IR, SNR 20, grid | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/1 |
    | IRTD, SNR 100, start T1 = 1 s | 200/200 | 176/200 | 200/0 | 103/0 | 0/0 | 0/0 | 0/0 | 91/89 |
    | IRTD, SNR 100, grid | 0/0 | 0/0 | 0/0 | 0/0 | 1/1 | 0/0 | 0/0 | 1/1 |
    | IRTD, SNR 20, start T1 = 1 s | 200/200 | 105/182 | 152/5 | 112/14 | 0/0 | 0/0 | 0/0 | 13/12 |
    | IRTD, SNR 20, grid | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

    T1IR with TIs 0.05–3.2 s (7, doubling), true B = 1.95, i.e. Sf = −0.95; T1IRTD with the 6
    TI/TD pairs of the tests, Sf = −0.95. Both with the dead-zone guard. Noise was added to the
    signed signal before taking the magnitude, the same noise for both forms. (Measured when
    T1IR was parametrized with B = 1 − Sf, starting at B = 2. It now uses Sf, starting at −0.95
    like T1IRTD; rechecked with the same data, IR rows with the grid: SNR 100, 2/0 at 0.3 s, else
    0; SNR 20, 2/2 at 4 s, else 0.)
  - Cost per voxel (one voxel, `@belapsed`): T1IR 0.72 → 1.80 µs, R1IR 0.90 → 1.62 µs, T1IRTD
    0.89 → 2.37 µs; models without a grid unchanged (T1SR 0.48 → 0.45 µs). About 1 µs for 30
    signal evaluations, which run on the dual-number buffer; a plain-number buffer would make
    them cheaper (not done: more code for ~1 µs).
  - Skipped when the parameter is fixed or given a start in the call, so a user's start always
    wins.
  - **Polarity restoration: tested, not adopted** (Barral et al., MRM 2010, as in qMRLab). As
    tested: take the point with the smallest |y|, flip the signs of the points before it (as a
    second candidate, of that point too), fit the signed model to each from the default start,
    keep the lower SSE. "Before" means TI order for T1IR; for T1IRTD, a lower null T1 (the T1 at
    which that point's signal is zero, from its TI and TD and the Sf start). Same data as the
    table above (200 voxels, T1 0.05–4 s, SNR 100 and 20), bad fits:
    - T1IR: 0 everywhere (grid start ≤ 2).
    - R1IR at T1 = 0.05 s: 200/200 at SNR 100, 197/200 at SNR 20 (grid start 0). The signs were
      right; the signed fit itself failed, from R1 = 1/s on an almost fully recovered curve.
      Choosing the signs does not fix a start far from the answer.
    - T1IRTD and R1IRTD at SNR 20: 4, 28 and 27 at T1 = 1, 2 and 4 s (grid start 0). With TD
      varying per TI, the smallest |y| does not mark the null, so the wrong points get
      flipped. Trying every split would fix that, at n + 1 signed fits per voxel.
    - Time per voxel (one voxel, a slower machine than the table above): about the same as the
      grid start at T1 = 0.9 s (T1IR 2.65 vs 2.75 µs), 2–3× slower at 0.1 s (8.1 vs 2.8 µs),
      since both signed fits start far away; T1IRTD 4.5–7.5 vs 3.8 µs.
    - Better in one case: with a TI right at the null (T1IR, T1 = 0.3 s, TI = 0.2 s) it found a
      lower SSE than the grid start in 17–19% of 2000 voxels (two nearly equal minima, set by
      the sign of the point at the null), but T1 was not more accurate: RMS error 1.9% for both
      at SNR 100; 9.8% (polarity restoration) vs 9.3% (grid start) at SNR 20.
    - Barral's method also searches over T1 (as we recall the paper and qMRLab's code; not
      checked here), and that search is what fixes the far-start failures. The grid start does
      that search without the sign choice, and works for any parameter declared `BestOf`.
- **Log-linear start for T2Decay** (`datastart`; `_loglinear_rate` in `src/models/builtin.jl`,
  `_data_start` in `src/fit/starts.jl`). Each voxel's T2 starts from a straight-line fit to
  log(y) against TE, weighted by y² (the noise in log y is about σ/y), skipping points with
  y ≤ 0; mean(TE) when the data do not decay. Exact on noise-free data.
  - Measured (500 voxels of magnitude data per case, time per voxel of a dataset fit on 1
    thread, best of 3 alternating runs; absolute times on a slower machine than some notes
    above): faster in almost every case, most when T2 is far from mean(TE). 32 echoes (TE
    10–320 ms), SNR 50: T2 = 40 ms 5.8 → 3.7 µs, 80 ms 5.1 → 3.1 µs, 2 s 8.0 → 2.8 µs, 10 s
    12.1 → 3.0 µs (4.1×). 8 echoes (TE 10–80 ms), SNR 50: 0.5 s 4.2 → 1.3 µs, 2 s 6.2 → 2.3 µs.
    Never slower beyond the noise (worst: 8 echoes, T2 = 40 ms, 1.69 → 1.80 µs).
  - Bad fits and max-iteration counts unchanged (to within 1 voxel in 24 cases): a speed-up,
    not a robustness fix; the dead-zone guard had already removed the time form's failures.
  - Models without a `datastart` pay nothing (the check is decided when compiling): R2Decay
    1.68 → 1.66 µs, T1IR 2.78 → 2.68 µs, QMTSIR 2.80 → 2.80 µs (one voxel's fit).
  - Not (yet) for R2Decay: it is already robust and fast from 1/mean(TE); in a prototype the
    same start gained it up to about 30% (single runs). ADCDecay could use the same helper.
- **Starting values in one place** (`src/fit/starts.jl`): the data start, the grid and the scale
  start, which used to be in `cache.jl`. One helper, `_score`, solves the scale and the SSE of a
  candidate start; the grid scores each grid value with it, the scale start scores one. Before,
  the grid repeated the scale calculation (and its fall-back to 1, unclamped there). Moving the
  code changed no result: starting points and final fits identical, bit for bit, for 18 model
  setups in Float64 and Float32. The helper is the core of the candidate starts of roadmap item 4.
- **Solver size limits** (`levmar`, static-array method): no allocation and faster than LsqFit
  up to at least 28 free parameters (DKI-like test, 3 observations per parameter: P = 7 2.4 µs,
  P = 15 25 µs, P = 22 64 µs, P = 28 74 µs per fit; LsqFit 3.5–19× slower). Compile time grows
  (≈1 s → ≈8 s, once per model and set of free parameters). Beyond ~30–40 free parameters, add
  a `levmar` method for ordinary arrays (`mul!`, `cholesky!`); the model interface does
  not change.
- **Threads**: chunks of voxels as tasks (`Threads.@spawn`), about 64 per thread with at least 16
  voxels each; plain `Threads.@threads` was 2× slower when slow voxels cluster (it splits the
  work into one block per thread). See the comment in `src/fit/fit.jl`.

## Scope so far

In: the interface above (with `:scale`, `:data` and `BestOf` starts, `datastart` and `relabel`), seventeen models: relaxometry (T1IR, R1IR, T1IRTD,
R1IRTD, T1SR, R1SR, T1MFA, R1MFA, T2Decay, R2Decay, T2DecayOffset, R2DecayOffset, T2BiexpDecay, R2BiexpDecay), diffusion (ADCDecay, ADCKurtosis), qMT (QMTSIR), `fit` for one voxel or a dataset returning a
`FitResult` (with `fitted`, `residuals` and a summary display), Float64/Float32, bounds, masks, per-voxel maps, ties,
threads, tests.

Out for now; earlier implementations are on the `api-v0.2` and `main` branches:
CLI / Python / MATLAB, GPU, other model-specific and neighbour starting values, standard errors,
weights, other data layouts, precompilation, more models.

## Roadmap

1. **One result type for `fit`**: done (`FitResult`, see the section above): `params`, `sse`,
   `iterations`, `status` as `FitStatus`, `model`, `settings`, `fitted` / `residuals`, and a
   summary display. Still to add, with the uncertainty work (3): JᵀJ for a single voxel,
   optional standard-error maps for datasets, a flag for parameters that ended at a bound, and
   StatsAPI queries (`coef`, `coefnames`, `rss`, `stderror`, `vcov`, `informationmatrix`). The
   internal `SolverResult` stays as it is; the solver already computes JᵀJ at the solution.
2. **Simulation, digital phantoms and benchmarks**: one simulation layer (`sim/`) shared by
   Monte Carlo, benchmarks, accuracy studies and tests, and useful to users directly.
   - Done (2a, see the Simulation section): `simulate` (Rician or Gaussian noise, `snr`
     relative to the scale parameter or `σ`, `n` copies or maps, a seeded `rng`); `TISSUES`
     (Stanisz et al. 2005, Table 1, 3 T; Dortch et al. 2018, Table 1) with `tissue` and
     `truth`; `phantom(labels, tissues)` and `labelshapes(dims)`; `compare` (bias, SD, RMSE,
     relative errors, % converged, overall and per label; no timing: speed belongs to
     `bench/`); `SIR_5POINT` / `SIR_4POINT` (Dortch et al. 2018, Table 2) and `SIR_16POINT`
     (log-spaced TIs from 10 ms to 2 s rounded to the ms, then 10 s; TD 2.5 s); a seeded end-to-end
     test and the Fig. 1 check in `test/`; `@inferred` checks of `fit!`.
   - Still to do in `sim/`: phantoms with smooth variation, partial volume, a B1 field and
     lesions; brain-like phantoms from tissue segmentations (BrainWeb, or the brain phantom
     KomaMRI uses, read from file) and real data (qMRLab's OSF datasets); a measured CSF T2;
     in vivo WM/GM values (e.g. from the studies in Bojorquez et al. 2017).
   - Monte Carlo vs phantoms: Monte Carlo (independent noisy copies of chosen tissues) gives
     bias/SD/RMSE against the CRLB and accuracy vs SNR; phantoms are needed for everything that
     uses neighbours or geometry (neighbour starts, outlier detection, masks) and for realistic
     whole-volume speed, where hard voxels cluster.
   - Where things live: `test/` gets a tiny seeded phantom (e.g. 8×8×2) for an end-to-end check
     (simulate → fit → compare) with loose accuracy limits, plus `@inferred` checks next to the
     allocation checks; timings never go in tests. `bench/` (own Project.toml) gets the speed
     suite as `const SUITE = BenchmarkGroup()` (the convention PkgBenchmark.jl and
     AirspeedVelocity.jl use to compare commits), timed with BenchmarkTools (`@btime` /
     `@belapsed` with `$` interpolation; `evals = 1, samples = 5` for whole-dataset passes): one
     fit per model, `signal!` alone, dataset throughput on threads, bounds vs no bounds, scaling
     with P, LsqFit comparison. Accuracy studies are scripts there too (starting-value
     strategies, bounds, accuracy vs SNR, protocol comparisons), each writing a table to
     `bench/results.md` with machine, Julia version and thread count.
3. **Uncertainty** (formulation from Dortch et al., Magn Reson Med 2018;80:1824):
   - Fisher information for Gaussian noise (Eq. 4): F = JᵀJ/σ², with J the derivatives of
     the signal at the true parameters with respect to the free ones. This is the JᵀJ the
     solver already accumulates.
   - Unbiased estimator (e.g. SIR with kmf free): MSE(θ̂ⱼ) = [F⁻¹]ⱼⱼ.
   - Biased estimator (e.g. SIR with kmf fixed at an assumed value) (Eq. 3):
     MSE(θ̂ⱼ) = bⱼ² + [(I + ∇b) F⁻¹ (I + ∇b)ᵀ]ⱼⱼ, with bⱼ = E[θ̂ⱼ] − θⱼ and ∇b = ∂b/∂θ. As in
     the paper, b and ∇b come from fits to noise-free data with the constrained parameter at
     its assumed value, repeated with each true parameter perturbed (finite differences).
   - Protocol design helpers: the minimax objective (Eq. 5) and sensitivity efficiency
     (Eq. 6). The acquisition can be differentiated too, since `signal` accepts dual numbers
     in the model's vectors.
   - Monte Carlo with Gaussian/Rician noise (bias, SD, RMSE, SD/CRLB).
   - Sensitivity to model assumptions: the bias from fitting with an assumed value that is
     wrong, the same machinery as Eq. 3's b. First case: QMTSIR's `Mzf_td0 = Mzm_td0 = 0` (both
     pools saturated at the start of TD). Simulate with nonzero values, fit with the default.
     A first check (5-point scheme, noise-free, 5% of M0 left): PSR +2%, R1f +4%, M0f −5%.
   - Validation targets: Table 1 tissue classes, Table 2 schemes, Fig. 2B.
   - Fit statistics on `FitResult`, added together as one set under StatsAPI's names, with one
     convention for the number of parameters (`dof`; packages differ on whether σ counts):
     `rss` (the SSE), `nobs`, `dof`, `dof_residual` (n − p), the residual SD
     σ̂ = √(SSE/(n − p)) per voxel (also a noise estimate, and the σ in the standard errors),
     `stderror`, `vcov` and the parameter correlations from JᵀJ, the "ended at a bound" flag,
     and `aic` / `bic` for choosing between models fitted to the same data (mono vs
     bi-exponential, ADCDecay vs ADCKurtosis). No new fields where a value follows from what is
     stored (σ̂, AIC, BIC come from `sse`, n and p). All assume Gaussian noise: biased for
     low-SNR magnitude data, documented once, next to the Rician-likelihood work.
4. **Starting values and outliers**:
   - Done (the baseline, see Parameter roles): start options per parameter in `Free` (a number
     or map, `:scale`, `:data`, `BestOf(values)`), declared by models in `parameters(m)`;
     `datastart(m, y, p)` with the voxel's fixed values and its own fallback; the `startgrid`
     hook, the `userstart` list and the planned per-fit `starts` option removed.
   - **`BestOf` on several parameters tries every combination** (`setup` allows one today).
     The cost multiplies. Measured per signal evaluation, about the cost of one candidate:
     T1IR (6 TIs) 0.04 µs, T2BiexpDecay (32 echoes) 0.55 µs; one biexponential fit takes
     16.6 µs. So for the biexponential, 10 × 10 = 100 candidates ≈ 55 µs per voxel (3× the
     fit), 30 × 30 ≈ 0.5 ms, 30³ ≈ 15 ms: about 4 s, 30 s and 15 min per million voxels on 16
     threads. No cap: the cost is predictable, and documented. To keep it affordable:
     - **Precomputed candidates (a dictionary).** When the other starts and the fixed values
       are the same in every voxel, the candidate signals s (and s·s) do not depend on the
       voxel: compute them once per fit. Each voxel then needs one dot product s·y per
       candidate (32 multiply-adds instead of 64 exponentials; estimated 10–20 µs for 900, not
       measured), as in MR fingerprinting's dictionary matching. A per-voxel map (B1) or a data
       start for another parameter rules it out; candidates are then evaluated per voxel, as now.
     - **Disjoint ranges for parameters that can swap** (T2short's values below T2long's):
       (20, 80) and (80, 20) ms fit identically and `relabel` swaps them anyway, so overlapping
       ranges waste up to half the candidates.
     - Fewer values: a start only needs the right basin (the biexponential study tests whether
       10 × 10 is enough).
     - One parameter at a time (n₁ + n₂ candidates) is cheaper but can miss the right pair when
       the parameters are coupled (T2short, T2long). Not the default.
   - **One fit from the best candidate, or several**: `BestOf(values; fits = k)`, default 1.
     Today `BestOf` evaluates the model at each value and runs one LM fit from the lowest SSE
     (N signals + 1 fit). That is enough for T1IR: at most 2 of 200 voxels failed, against 200
     of 200 from one start. It can fail when the other parameters' starts are off: a
     candidate's SSE is computed with them at their starts, not optimized, so the ranking can
     put the wrong basin first (Sf at −1 when the truth is −0.9; fshort at 0.2 when it is 0.4).
     The more coupled the parameters, the likelier; the biexponential is the risk (untested).
     `fits = k` runs full fits from the k best *local minima of the SSE over the values*,
     candidates lower than all their neighbours, so each is a different basin (the k lowest
     values are usually neighbours in one basin, and k fits would find the same minimum k
     times); it keeps the lowest final SSE. Cost: k fits.
   - **Reference for the studies: a full fit from every candidate** (`fits = :all`; T1IR 30
     fits, biexponential 10 × 10 ≈ 1.7 ms per voxel). The start-strategy and biexponential
     studies (item 2) measure, per strategy, the % of voxels reaching the lowest final SSE found
     by any strategy, and the time, to decide from data whether `fits = k` is worth having and
     for which models.
   - **More data starts.** History: VoxelFit 0.1 had a per-fit `init = :auto`, and 0.2 a
     per-parameter `Estimate(:auto)` from `initial_guess(m, acq, y, p)`, with a guess for every
     model. The lean rewrite (9a6e173) removed data starts and kept `Free(:scale)`. The 0.2
     guesses are in git (99a0367) to port as `datastart` methods: log-linear
     (mono-exponential), peeling (bi-exponential: slow component from the late echoes, fast
     from the residual of the early ones), the null point for inversion recovery
     (T1 = TI_null / ln 2), log(S0 − y) for saturation recovery, DESPOT1 with B1 for
     SPGR/T1MFA, max |y| for M0f. Candidates: R2Decay and ADCDecay (the same log-linear fit;
     R2Decay gained up to ~30% in a prototype; on hold), ADCKurtosis (quadratic in b of
     log y), T1SR/R1SR, the offset models, the biexponential models, T1MFA/R1MFA (now possible:
     `datastart` gets B1 in `p`). None for the IR family or QMTSIR, where `BestOf` is the tool.
     Test the IR null point against `BestOf` in the start-strategy study (never compared)
     rather than assume.
   - **Starts as candidates** (the implementation underneath): every start option proposes
     starting points, EQUIP computes the SSE at each (with the `:scale` parameter solved for
     each) and runs one fit from the best (or a few, above). A combination is the union of
     candidates, at about one signal evaluation each, so the data start and `BestOf` share one
     code path. Candidates could be evaluated on a plain-number buffer (`BestOf` now uses the
     dual buffer: ~1 µs per voxel for 30).
   - Later, if needed: several `:scale` parameters (biexponential amplitudes, an offset) solved
     together by linear least squares.
   - **Neighbours as a second pass**, not a first-pass start: voxels are fitted in parallel in
     no fixed order, so neighbours' results are not available during the fit. After a dataset
     fit, detect outliers from what the fit already reports: status (hit `maxiter`, failed,
     invalid), a parameter that ended at a bound (the `FitResult` flag of item 1), and a large
     deviation from the local neighbourhood (e.g. distance from the median of the surrounding
     voxels, in units of a robust spread such as the MAD).
   - Then, per outlier, either refit with its neighbours' median as the start, a map start
     over the outliers only: `refit(res, data; mask = bad, T1 = Free(start = neighbour_median))`
     (keep the refit if it converges, within bounds, with a lower SSE), or replace it with a
     neighbourhood average. Mark replaced voxels in the status map, so they
     are never mistaken for fitted values.
   - Bounds stay on by default: measured on SIR-FSE (5-point scheme, SNR 100), they cost
     nothing for good voxels (2.98 vs 3.00 µs, same iterations and results) and make bad ones
     faster and contained (noise-only voxels: 1 vs 154 hitting `maxiter`; 0 vs 874/1000 PSR
     values outside [0, 1]). Unbounded fits remain possible per fit:
     `PSR = Free(lower = -Inf, upper = Inf)`.
5. **Data I/O**: make getting data in and maps out as easy as possible. An `io/` layer
   (dispatch on file type, protocol handling, the PAR/REC reader) plus one package extension per
   file-format package, so the core never depends on file formats.
   - Reading: `data, info = loaddata(path_or_paths; ...)` returns the data with observations
     last, plus `info`: the geometry needed to write maps back, and any acquisition values found
     (TE, TR, FA, TI, ...).

     | format | via | notes |
     |---|---|---|
     | NIfTI (.nii, .nii.gz) | NIfTI.jl | apply `scl_slope`/`scl_inter`; memory-map uncompressed files |
     | BIDS JSON sidecars | JSON3.jl | EchoTime, RepetitionTimeExcitation, FlipAngle, InversionTime (s, deg); qMRI file collections (`_echo-`, `_flip-`, `_inv-`, `_mt-` entities) stacked into one array |
     | MATLAB .mat | MAT.jl | pick the variable by name |
     | NumPy .npy / .npz | NPZ.jl | C order: reverse dimensions, or observations first |
     | DICOM | DICOM.jl | one series; TE/TR/FA/TI tags (ms → s); sort by instance/echo number |
     | Philips PAR/REC, XML/REC | own reader (text/XML + raw binary; DECAES has one to learn from, check its license) | per-slice scaling; TE/TR/FA/TI in the header |
     | ISMRMRD | MRIFiles.jl / HDF5.jl | image datasets; raw k-space must be reconstructed first (e.g. MRIReco.jl) |

   - Several files (one per echo, flip angle or inversion time): a list or pattern, stacked
     along the observation dimension and sorted by the acquisition value or BIDS entity index.
   - **Acquisition values that are not in the headers** (TI and TD for SIR rarely are):
     - Priority: values given in the call > a protocol file > values found in headers/sidecars.
     - Protocol file: TOML/JSON with named vectors, or CSV with one row per observation:
       ```toml
       model = "QMTSIR"
       TI = [0.010, 0.050, 0.056, 0.277, 0.843]
       TD = [3.270, 4.489, 1.652, 2.922, 0.010]
       ```
     - The model is built with its keyword constructor, e.g. `QMTSIR(; TI, TD)`, so this works
       for every model, user-defined ones included. Missing fields raise an error that names
       them and says where they were looked for.
     - Units: converted to seconds and degrees on reading (DICOM and PAR/REC store ms, BIDS
       uses s). Warn when values look like the wrong unit (e.g. TE > 10 s). The number of
       observations must match the data's last dimension.
   - Writing: `savemaps(prefix, res; like = info)` writes one file per map in the input's format
     and geometry (intensity scaling reset), with BIDS-derivative names where they exist
     (T1map, R1map, T2map, T2starmap, M0map, S0map, ...), plus status and SSE maps and a JSON
     sidecar recording the model, acquisition, fixed/free/start settings and package version.
6. **Front ends**, built on the `io/` layer: command line (data files + protocol file in, map
   files out), Python, MATLAB.
7. **`levmar` for ordinary arrays** (only when a model needs more than ~30–40 free parameters):
   a second method of the same function, chosen by dispatch on the argument types:
   `levmar(prob, θ0::Vector, lo::Vector, hi::Vector, opts, work::LevmarWorkspace)`, with a
   workspace allocated once per thread (JᵀJ, Jᵀr, step, Cholesky factor) and in-place
   `mul!`/`cholesky!`/`ldiv!`. `setup` picks static or ordinary arrays from the number of free
   parameters, once per fit (a threshold around 32), so every voxel dispatches to the right
   method at no run-time cost. Derivatives may need chunked dual numbers
   (`ForwardDiff.JacobianConfig`) at that size. The model interface does not change.
8. **Later**: GPU, Rician-likelihood fitting, more models:
   - Gradient-echo decay with the through-slice sinc term for macroscopic field gradients,
     S0·exp(−TE/T2*)·|sinc(Δf·TE)| (Fernández-Seara & Wehrli, MRM 2000), as a separate model.
     Δf, the frequency spread across the slice (Hz), is fixed from a B0 field map or fitted
     (default 0: no correction). Julia's `sinc` is the normalized sinc, matching the slice
     integral. Later: in-plane gradients (a product of sincs), non-rectangular slice profiles.
   - Polarity restoration for magnitude inversion recovery: tested against the grid start and
     not adopted (Performance notes, grid-search start).
   - Bounds in the solver: if boundary problems remain that the dead-zone guard does not handle,
     compare against a trust-region-reflective step (SciPy's `trf`) or a change of variables.
   - Variable projection for the `:scale` parameter: solve M0/S0 by least squares at every
     iteration, not only at the start, so LM works on one parameter fewer (Golub & Pereyra;
     Barral's RD-NLS does this for inversion recovery). To measure: speed and robustness.
   - Diffusion tensor and kurtosis, IVIM, multi-exponential T2 (myelin water), DCE.
