# EQUIP design (0.3)

The design decisions, recorded so the code and the discussion share one reference.

## Inputs: where each kind of information lives

| input | varies across voxels? | Julia form | example |
|---|---|---|---|
| **Model**: physics, acquisition and options | no | a `@kwdef` struct `<: MRIModel` | `QMTSIR(TI = [...], TD = 2.5, magnitude = true)` |
| **Parameters**: every quantity in the signal equation | each one can be free, a constant, a per-voxel map or a tie | values: a named tuple `p`; default roles: `parameters(m)` | PSR, R1f, kmf, B1, Sm |
| **Fit configuration**: which parameters are estimated in this fit | no | keywords named after the parameters | `kmf = 15.0`, `kmf = Free()` |
| **Data** | yes | N-d array, observations last | X×Y×Z×M |

**Naming rule**, in code, docstrings and docs: "parameters" always means the quantities in the
signal equation (T1, PSR, M0, B1, ...), estimated or fixed; the protocol's values (TI, TE, flip
angles, ...) are always the "acquisition", never "parameters" (MRI speech also says "scan
parameters"; EQUIP does not). A model is the signal equation (its type) plus its acquisition
and options (its fields).
- Kept: `MRIModel` (`SignalModel` would be slightly more exact but more generic, so more likely
  to clash with another package's names) and `parameters(m)` (the word users and model authors
  expect; ModelingToolkit exports a `parameters` with a different meaning, so a session using
  both writes `EQUIP.parameters`).

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
  (`Free(start = ..., lower = ..., upper = ...)` / `Fixed(value)`) sits next to it, so the starting value, the
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
    PSR = Free(start = 0.1, lower = 0, upper = 1),
    R1f = Free(start = 1.0, lower = 0.05, upper = 10),
    Sf  = Free(start = -0.95, lower = -1.2, upper = 0),
    M0f = Free(start = :scale, lower = 0),     # starts at the least-squares signal scale
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
(provided) allocates the output and resolves ties, for simulation; `p` gives every parameter
(see Simulation: no silent defaults).

Optional hooks: `datastart(m, y, p) = (T2 = value,)` computes starting values from each voxel's
data, for the parameters declared `:data`, e.g. the log-linear fit of `T2Decay` (Performance
notes); `p` holds the voxel's fixed values (e.g. B1). `relabel(m, θ)` chooses the labels to report
when they can swap without changing the signal (the two components of `T2BiexpDecay`).

## Parameter roles

- `Free(; start, lower, upper)`: estimated. The start says where each voxel's starting value
  comes from:

  | `start =` | per voxel | given by |
  |---|---|---|
  | a number or a map | start there | model default (a number) or the call |
  | `:scale` | least squares given the other starts, a = (s·y)/(s·s) | the model only |
  | `:data` | the model's estimate from this voxel's data (`datastart`) | model default or the call |
  | `BestOf(values)` | try each value (`:scale` solved for each), start from the lowest SSE | model default or the call |

  Models declare theirs in `parameters(m)`, so a reader sees where each start comes from:
  ```julia
  parameters(m::T1IR) = (M0 = Free(start = :scale, lower = 0), Sf = Free(start = -0.95, ...),
                         T1 = Free(start = BestOf(_t1grid(m.TI)), lower = _TMIN))
  parameters(m::T2Decay) = (S0 = Free(start = :scale, lower = 0),
                            T2 = Free(start = :data, lower = _TMIN))
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
  - **Decision: the start is a keyword, the only form: `Free(start = 0.1, lower = 0)`.**
    `Free(0.1; lower = 0)` used to work too, so the same role was written two ways (the models
    positionally, fits by keyword), and a bare value beside named bounds did not say what it
    was. With every part named, a model's `parameters` reads like a fit's keywords, and a later
    setting such as `varpro = true` (see the VarPro design) sits beside the others the same way.
    `Free` has one constructor, by keyword (an inner constructor, so Julia's positional
    `Free(start, lower, upper)` is gone too); a positional call is an error that shows the call
    written by name (`Free(0.8)` → "write Free(start = 0.8)"). Results unchanged: every model's
    roles as printed, and the starts and fits of every bench case (bounded and unbounded, two
    solver settings, both precisions) are identical bit for bit; dataset fit times equal within
    noise (10 000 voxels: T2 4.2–5.3 ms, SIR 4-point 5.0–5.6 ms, biexponential 42–44 ms).
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
    rate is exactly R1f, so `R1f = r1map` does it. A formula R1f(R1obs, …) is needed when
    R1m ≠ R1f, and in the qMT-SPGR models (roadmap item 6). It needs no new feature either: a
    tie by function, with the observed R1 map as a fixed parameter (a tie can use fixed maps).

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

- **Roles on the model's type** (`parameters(::Type{<:QMTSIR})`), with the fallback
  `parameters(m::MRIModel)` asking the type, as Base's `eltype(x) = eltype(typeof(x))`. Why: the
  roles are the same for every acquisition, and making a model with a throwaway protocol just to
  ask `nominal(QMTSIR(TI = ..., TD = ...), :SIR_WM_I)` was clumsy; now `nominal(QMTSIR, :SIR_WM_I)`,
  `parameters(QMTSIR)` and `defaults(QMTSIR)`. `Type{<:M}`, not `Type{M}`: the models are
  parametric, so a model's type (`QMTSIR{Vector{Float64}, Vector{Float64}}`) does not match
  `Type{QMTSIR}` (types are invariant). Starts that depend on the acquisition (ADC's
  1/mean(b), the R2 and bi-exponential models' from mean(TE), the inversion-recovery models'
  `BestOf` grid from the TIs) became functions of the model, `Free(start = m -> 1 / mean(m.b))`,
  computed once per fit in `setup` (after the fit's keywords, so a function start given in a fit
  works too), and filled in when `parameters` is asked of a model. Rejected: moving them to
  `datastart` (`:data`): it runs per voxel, on the model already converted to the working
  precision, so Float32 starts could change by a rounding step; the function start is computed
  from the model as given, as before. A model can still define `parameters(m::MyModel)` on the
  instance (more specific than the fallback); its type then cannot be asked (an error says so).
  Checked: every built-in model's starts and fits (400 noisy voxels each, Float64 and Float32,
  `maxiter = 0` and default) and `nominal(m)`, 108 entries, and every bench case (starts and
  fits, bounded and not, three solver settings, both precisions; the roles as printed), 137
  entries: all bit-identical before and after. Speed: unchanged (bi-exponential bench case,
  alternating builds, three rounds: 1 thread 127–150 ms before, 128–130 after; 4 threads
  38.6–40.4 ms before, 37.1–39.4 after).

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

One type for every fit (`src/fit/result.jl`): `params`, `fixed`, `sse`, `iterations`, `status`,
plus `model` (as fitted: its acquisition in the working precision) and `settings` (the call's
keywords, by reference: the parameter keywords, `precision`, `solver`, and `mask` for a
dataset). Numbers for one voxel, maps for a dataset; `_onevoxel(res)` (is the status a single
`FitStatus`?) tells them apart at compile time.
- `fit!` returns it too and still allocates nothing: the model and settings are references in
  an immutable struct. Measured, best of 3 alternating runs: `fit!` T2Decay 1841 → 1852 ns,
  QMTSIR 2951 → 2948 ns; a 20k-voxel dataset 39.9 → 40.2 ms (it now also fills an iterations
  map).
- `fixed`: the fixed values the fit used, the model's defaults and the call's together: numbers
  and ties (`R1m = :R1f`), and for a dataset maps (the user's arrays, not copied). **Decision:**
  a one-voxel result stores that voxel's values (a map's value at the voxel), so it is
  self-contained: before, a one-voxel result fitted with fixed maps (`fit!` with `voxel`) could
  not rebuild its curve (the voxel index was not stored), and `fitted` rebuilt the fixed values
  from `settings` and `defaults(m)`. Results and speed unchanged (fingerprint of every bench case
  identical bit for bit; `fit!` T2Decay 1.6, QMTSIR 2.5, T1MFA 1.1 µs, before and after; still
  no allocation).
- `resolve(res)` / `resolve(res, i...)`: every parameter's value, fitted and fixed, ties
  evaluated, in model order (a method of `resolve(m, p)`). `fitted(res)` / `fitted(res, i...)`
  are `signal(res.model, resolve(...))`; `residuals(res, y)` / `residuals(res, data, i...)`
  subtract them from the data. `fitted` and `residuals` are StatsAPI's names.
- Refit: `fit(res.model, data; res.settings...)` repeats the fit with the same choices.
  `refit(res, data; changes...)` merges the changes into the settings. A free parameter starts
  from a new start given by a change, else its previous estimate, else the model's default; a
  change of bounds only (`Free(upper = 0.2)`) keeps the previous estimate and the stored bounds
  it does not mention. **Decision** (earlier, a changed parameter restarted from the model's
  default start, dropping the previous estimate). A voxel whose previous estimate is NaN becomes
  `InvalidData`, as for any non-finite start map.
- Maps (fixed values, starts) and the mask must have the data's spatial size, not just its number
  of voxels; trailing length-1 dimensions are ignored. **Decision:** a transposed B1 map was
  accepted (same voxel count) and gave T1 errors up to 92%, silently. The cache interface
  (`fit!(c, y; voxel = v)`) reads maps by linear index.
- Display: one line in `print` and inside collections; at the REPL a summary (one voxel: status,
  iterations, SSE, values, fixed parameters; a dataset: voxels fitted, % converged, counts per
  status, median and interquartile range of each parameter over converged voxels, fixed
  parameters).

## Simulation

`src/sim/` (uses models only): the tissue presets, phantoms, the noise model and `simulate`;
from true values to data. (`compare`, which judges a fit, moved to `src/errors/`: see
"Folders by what they produce" below.)

- **Folders by what they produce** (`org-proto`): `sim/` makes data from true values (the
  tissue presets, phantoms, the noise model, `simulate`) and fits nothing; `errors/` (was
  `analysis/`) fits and judges against a truth, the bias and SD of the estimates, predicted
  (`crlb` and its pieces) or measured (`montecarlo` for one voxel, `compare` for a dataset such
  as a phantom), with the diagnostics that say where the errors come from (`cond`, `cor`;
  leakage and sensitivity maps next) and the sweeps over both. Why: by method, `montecarlo` and
  `compare` would sit with `simulate`, since they simulate; by product they sit with `crlb`,
  the other answer to the same question, which they are checked against and shown beside
  (every Monte Carlo result carries crlb's prediction, and a sweep takes either). The include
  order says the same: `montecarlo` calls `simulate`, `fit` and `crlb` and keeps a
  `CRLBResult`, so it comes after all three; in `sim/` it would make `sim/` need `errors/`,
  which needs `sim/`'s noise model. The name: "errors" as in error analysis, the systematic
  (bias) and the random (SD), RMSE both; "accuracy" alone reads as bias, "uncertainty" (the
  GUM's) leaves out the bias that is often the point, "performance" would be read as speed
  here. Also: the noise model (`snr`, `σ` or `Σ`, and their checks) was written twice, in
  `simulate` and in `noisecovariance`; now once, sim/noise.jl, for both (a `σ` neither a number
  nor a vector is now rejected by `simulate` too, with the same message). `_scalename`, the
  model's `Free(start = :scale)` parameter, moved to models/interface.jl, so `sim/` uses
  `models/` alone. No exported name changed. Results identical: `simulate`, `compare` and
  `noisecovariance` (67 cases, the 18 error messages), the 688 crlb cases with either default,
  the 8 sweeps, the design objectives; 1541 tests. Costs unchanged (three alternated runs
  against `lean`): `noisecovariance` 0.39–0.41 µs, a sweep setting 5.4–5.5 µs, `crlb!` and
  `simulate` the same. Found on the way: the first version called `nobs(m)` twice per noise
  check, and the generic `nobs` allocates (it collects the acquisition's lengths in a vector:
  0.06 µs and 144 bytes); N is now passed in.
- **`compare`'s relative errors are relative to |truth|**, as `crlb`'s and `montecarlo`'s: the
  relative bias has the sign of the bias. Before, it divided by the truth itself, so for a
  negative true value (Sf = −1) a bias of −0.009 showed as a relative bias of +0.9%, the
  opposite of `montecarlo`'s for the same estimates. Found when `compare` moved beside
  `montecarlo`. Only negative truths change, by an exact sign flip (the phantom snapshot: the
  Sf rows, nothing else); `relrmse` and `medrel` were unaffected (squares, absolute values).
  Tested: `compare(mc.result, θ)` gives `montecarlo`'s bias, relative bias, SD and RMSE within
  2e-12 (4-point WM I, kmf assumed 12.5, 4000 copies).
- **`nobs` allocates nothing** (`nobs-proto`): the generic method collected the lengths of the
  model's vector fields in an array, 35–65 ns and 80–240 bytes a call, on every noise level
  crlb is given (`noisecovariance`), every `simulate` and every `signal`. Now a `map` over the
  tuple of fields (each field's type known, so the lengths form a tuple: no array), 2–3 ns and
  0 bytes (constant-folded for static vectors). `crlb!` allocates 144 bytes less (4-point 3696 →
  3552, 16-point 4560 → 4416), its time unchanged within noise; one-voxel `simulate` 0.2 µs
  faster (8.3 → 8.1 µs). The 21 built-in models' values and the error messages identical; the
  phantom snapshot and the 688 crlb cases bit-identical; 1560 tests (an allocation test added).

- **Every value is given; nothing is filled in silently.** `signal(m, p)` and `simulate(m, p)`
  need a value for every parameter of the model (a number, a map, or a tie such as
  `R1m = :R1f`); a missing one is an error that lists every missing name and points to
  `nominal(m)`. `defaults(m)` is the named tuple of the values `parameters(m)` fixes
  (QMTSIR: `kmf = 12.5, R1m = :R1f, Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0`).
  `resolve(m, p)` (formerly the internal `_allparams`) checks a set and resolves its ties: what
  `signal` does before `signal!`, public for loops calling `signal!`. Why every value: the fixed
  defaults are fitting assumptions (kmf = 12.5 is the 4-point scheme's assumption, not a tissue value), and a
  simulation that used them without saying so hides part of its truth; e.g. a Monte Carlo of
  "kmf fixed wrongly" needs the true kmf written in the script. Before this, a left-out fixed
  parameter took its default silently. `fitted(res)` uses the fixed values the result stores
  (`res.fixed`), since a fit's fixed values are its own settings. Checked: the bench cases and
  both examples give the same data and fits as before (byte-identical CSVs, equal hashes).
- **`nominal` gives every value: one function, the tissue optional.** `nominal(m; changes...)` and
  `nominal(m, tissue; changes...)`: each value from a keyword, else the tissue (by name, or
  converted between time and rate), else the model (a fixed value, 1 for the scale and, without
  a tissue, a free parameter's start); ties kept so they follow later changes; nothing for a model
  author to write. `nominal(m)` (the model's values) was first added for models no tissue
  preset covers, next to `truth(m, tissue)`, then the two were merged under the name `nominal`:
  both give a simulation's ground truth, from the model, the literature or the user, and one
  name is one thing to learn. One rule keeps the two sources apart: with a tissue, a free
  parameter must come from the tissue or a keyword, never from a start. A fallback to the starts would turn a tissue that does not fit the model into
  the model's guesses without a word (`nominal(QMTSIR(...), :WM)`: WM has T1 and T2, no PSR,
  R1f or Sf), and values that are not tissue properties (an inversion efficiency Sf) are the
  user's choice. Without a tissue, a free parameter with no number to take (`:data`, `BestOf`)
  must be given. Three methods, one per kind of second argument (none, a preset's `Symbol`, a
  `NamedTuple` of values), around one shared body: first written as one method with an optional
  argument `t = nothing` and `t === nothing` / `t isa Symbol` branches, changed to dispatch so
  "no tissue" is not a sentinel value, a wrong type gets Julia's MethodError listing the forms,
  and another source of values (a fitted voxel of a `FitResult`, for data-driven design) is one
  more method. Checked: 292 calls (8 models × presets, own values, keywords, errors) give the
  same values and the same error messages before and after.
  What to give is shown, not asked for: `nominal`'s error lists every parameter with the value it
  takes (`required` for the missing ones), and a model typed at the REPL shows the same table
  with its acquisition (`show(io, ::MIME"text/plain", m)`, one method for every model). A
  `nominal(m, :help)` option was tried and dropped: a symbol argument that switches a function
  to printing is not how Julia packages do it (`?truth` is the docstring, `show` the object).
- **SNR is relative to the scale parameter** (the one declared `Free(start = :scale)`: M0, S0, M0f),
  σ = scale/snr, voxel by voxel: Dortch et al. 2018's "SNR = 200 at thermal equilibrium". It
  does not depend on the protocol, as σ on a scanner, so protocols compare at equal noise. The
  measured signal can be far below the scale (MFA at low flip angles, SIR schemes that never
  reach full recovery), so image SNR and this SNR differ. `σ` gives the noise directly (e.g.
  from an image background); an SNR relative to the measured signal can be added if needed.
- **The noise's distribution is the user's choice, with no default** (`noise = :rician`,
  |S + σ·(n₁ + i·n₂)|, or `:gaussian`, S + σ·n), in `simulate` and `montecarlo` (one scenario
  or a set), checked before anything runs. It was Rician unless the model had
  `magnitude = false`, but that inference was wrong twice over: `magnitude` says how the signal
  is computed, not how the images were reconstructed, and most models (T2 decay, VFA, ...) have
  no such option, so they got Rician noise silently whatever the data. How the images were made
  is something only the user knows, so the user says it, and every script shows it. (Changed
  2026-10; calls that relied on the default now pass `noise = :rician`, and give the same data
  draw for draw.) It is given only where data are made: `fit` (least squares), `crlb` (first
  order in the noise), its pieces, `ProtocolDesign` and `Scenario` all assume or need no
  distribution, and refuse a `noise` keyword with an error that says why, rather than a
  generic "not a parameter". For magnitude inversion recovery, adding complex noise to the
  signed signal and taking the magnitude depends only on |S|, so Rician noise on the model's
  |S| is exact.
- **Averages: `averages = nex`, and with Rician noise `averaging = :complex` or `:magnitude`**
  (2026-10). With `averages` (one per observation, or one number for all; whole numbers ≥ 1),
  `snr` and `σ` are one average's noise and observation j's is σ/√NEXⱼ, in `simulate`,
  `montecarlo`, `crlb` (every form), `crlb!`, `noisecovariance`, `informationmatrix` and
  `ProtocolDesign`; results record them (`c.averages`, `mc.averages`, `mc.averaging`) and the
  displays say "σ = 0.05 per average, 1 to 8 averages". Why a keyword, when σ per observation
  could already express it: with `snr` the σ is relative to each voxel's M0, so it cannot be
  given as numbers, and the script should say how the data were acquired. How the averages
  were combined changes Rician data, so it is the user's choice as the distribution is, with
  no default:
  - `:complex`: the complex images averaged, then the magnitude: Rician with σ/√NEX, the same
    data, draw for draw, as σ/√NEX given per observation (tested);
  - `:magnitude`: the magnitude images averaged (most diffusion: shot-to-shot motion phase
    rules out complex averaging): the mean of NEX Rician values with one average's σ. Its SD
    falls with √NEX but its floor stays at one average's. Measured (ADC, b 0–3000 s/mm², NEX
    1, 1, 2, 2, 4, 4, 8, SNR 20 per average at b = 0, 40 000 copies): at b = 3000 (1.8σ for one
    average) the mean is 0.106 (one average's floor: 0.106), complex averaging's 0.093, the
    signal 0.091. Fitted ADC (montecarlo, 20 000 copies): bias −1.96% ± 0.04% and SD/crlb
    0.987, against −0.33% and 1.002 averaged in complex.
  Refused, each with a message that says what to do: `averaging` without `averages`, or with
  Gaussian noise (real-valued data average the same either way); `averages` with `Σ` (Σ is
  the covariance of the data as fitted, after averaging and any processing, which is where
  correlation between observations mostly comes from, e.g. denoising; per-average correlation
  with different NEX per observation has no single meaning); `averaging` to crlb, its pieces
  and ProtocolDesign (first order, σ/√NEX however combined; the floor is montecarlo's to
  measure); `averages` or `averaging` to `fit` (unweighted least squares; weights next) or a
  `Scenario` (the noise belongs to the data). `ProtocolDesign` now takes σ² per point too
  (averages that differ, or σ per point): a design group holds one variance or one vector per
  scenario, the kernel's per-observation path; the objective still allocates nothing.
  Correlation between voxels (parallel imaging, k-space filtering) is not modelled: each voxel
  is fitted alone, so it changes no voxel's bias or SD, only ROI averages.
- **`simulate` allocates only the data** (2026-10). It called `signal(m, p)` for every voxel,
  which checked the whole truth again (`resolve`: names, missing values, roles, ties) and made
  a new vector: 2.8 KB and 8.7 µs a copy for the 4-point SIR, more than nine-tenths of a
  montecarlo's time (10 000 copies on 4 threads: 91 ms, of which simulate 86 ms; now 7.5 ms, of
  which simulate 2.0 ms). Now the truth is checked once (`_complete`, `resolve`'s checks) and kept in the
  model's order with ties as `_Tie`s; each voxel reads its numbers and follows its ties by
  `map`s over named tuples (unrolled by the compiler, no allocation), and `signal!` writes into
  one buffer of `signal`'s element type (`_signaltype`, shared). Correlated noise: `randn!` into
  one buffer and `mul!` into two, in place of `L * randn(rng, M)`. The loop is behind a function
  barrier (`_simulate!`), so it is compiled for the truth's and the options' types. The random
  numbers are drawn in the same order, so the data are the same bit for bit: checked on 85 cases
  (ten models, each with snr Rician and Gaussian, σ per observation, Σ Rician and Gaussian,
  averages combined as magnitudes and in complex, one voxel; maps with a NaN background, a tie
  by name and by function, Σ on maps; a montecarlo's estimates). Measured, a copy, one thread:
  4-point SIR 8.71 → 0.21 µs (2800 → 0 bytes beyond the data), T2 decay with 32 echoes 1.41 →
  1.29 µs (384 → 0), the same with a full Σ 2.13 → 1.17 µs (1664 → 0), ADC with magnitude
  averaging over 7 b-values 1.07 → 0.48 µs (368 → 0). (The T2 cases are the random numbers and
  exponentials themselves.) Still one thread, so that the data stay the same draw for draw.
- **Precision: a choice where the data are large, Float64 where the numbers are delicate**
  (2026-11-01). `fit`, `init`, `refit` and montecarlo's fits take `precision = Float32`; every
  input (data, maps of fixed values, starts, weights, the acquisition) is converted once, as it
  enters the cache, so the solver runs in one type (checked: the cache's buffers, duals and
  weights all Float32 with Float64 inputs; the same estimates from Float64 data and from the
  same data rounded to Float32; 0 bytes per voxel). `simulate` takes `precision` for its output
  only (computed in Float64 and rounded, so the same seed gives the same data, rounded), and the
  data reader will too, for memory: a 256×256×100 dataset with 32 echoes is 1.7 GB in Float64.
  crlb, leakage, the design's objective and the fit statistics stay in Float64: one voxel each,
  second derivatives and inverses whose error grows with the condition number (large for the
  biexponential and qMT), and microseconds each, so Float32 would buy nothing.
  Measured (4 threads; one voxel's `fit!`, then 100 000 voxels): Float32 is faster for most
  models, VFA 1.53× and 1.45×, 4-point SIR 1.34× and 1.54×, 16-point SIR 1.34× and 1.24×, T1IR
  1.12× and 1.26×, the biexponential 0.95× and 1.25×; T2 and ADC 1.28–1.39× per voxel but the
  same on a dataset. (The first README's "about 1.6×" was for the code before Phase 1.)
- **Float32 dataset fits flush subnormal numbers to zero** (2026-11-01). Some SIR fits were
  slower in Float32 than in Float64: 5-point with kmf free in the lesion 0.70×, 16-point 0.76×
  (50k voxels, 4 threads). Not more solver work (fewer evaluations in Float32: 8.1 against 9.9 a
  voxel in the 5-point case), but slower evaluations, all in the observations with a long TD:
  one observation at a time, TI = 0.050 s, TD = 4.489 s took 181.5 ns in Float32 against 77.6 ns
  in Float64, every other one 86–96 ns. There exp(-R1⁺·TD) is 6.8e-33 (still normal), but its
  products with the amplitudes' derivatives are 4.3e-40, below Float32's smallest normal number
  (1.2e-38): subnormal, which the CPU handles on a slow path, tens of times slower per
  operation. With the CPU's flush-to-zero mode (`set_zero_subnormals(true)`) that observation
  took 86.6 ns, and every Float32 fit was at least as fast as the Float64 one. So the dataset
  loop sets the mode for each chunk of a Float32 (or Float16) fit and restores it after
  (`_fit_chunk!`, fit/dataset.jl): once per chunk, because switching costs about 15 ns, and
  around each single `fit!` it made a Float32 loop 9% slower (so `fit!` is left as it is).
  Float64 fits are untouched (their smallest normal is 2.2e-308). Measured, 50k voxels, 4
  threads, Float32 before → after (and against Float64 after): 5-point lesion 136.9 → 76.8 ms
  (1.21×), 16-point lesion 359.9 → 224.4 ms (1.10×), 5-point WM I with kmf 12.5 60.7 → 45.8 ms
  (1.19×), 4-point 39.8 → 29.4 ms (1.03×), T2 1.43×, biexponential 1.12×. Results: the 91-entry
  snapshot (Float32 fits of every bench case and model included) identical, bit for bit; the
  flushed values are about 1e-31 of the terms they are added to.
- **Tissue presets are published values with their source**- **Tissue presets are published values with their source**: Stanisz et al., MRM 2005;54:507,
  Table 1 (T1 and T2 at 3 T, 37 °C, in vitro: animal tissues and human blood; white matter is
  bovine corpus callosum, gray matter cortex, so in vivo values differ); Dortch et al., MRM
  2018, Table 1 (the four SIR classes, with Sm = 0.83, Sf = −1.0). CSF (Stanisz has none): T1 =
  4.124 s, the mean of the three in vivo inversion recovery studies in the review of Bojorquez
  et al., MRI 2017;35:69, Table 1c (4391, 3817, 4163 ms; the VFA values spread from 4.2 to
  6.9 s); T2 = 1.75 s, the middle of the usual 1.5–2 s at 3 T, a placeholder (the review has no
  CSF T2). Users give their own values as a named tuple, or change one with `merge`. `nominal(m, tissue)` converts T1/T2 to rates by name, sets
  the scale parameter to 1, and refuses to guess free parameters that are not tissue properties
  (the inversion efficiency Sf of the inversion recovery models): they are given as keywords.
  A parameter the model fixes by default takes that default (B1 = 1, QMTSIR's kmf, R1m = :R1f),
  in the result.
- **Phantoms**: `phantom(labels, tissues)` fills one map per parameter from label => truth
  pairs (a tie, the same in every tissue, stays a tie), NaN elsewhere, so the background gives NaN data and is left out with
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
                       cache.jl (init: the cache), voxel.jl (one voxel: fit!, and fit for a
                       vector), dataset.jl (fit for a dataset, threaded; refit)
  sim/                 TISSUES / tissue / nominal (presets), phantom and labelshapes, the noise
                       model (noise.jl), simulate: from true values to data
  errors/              a fit's errors against the truth: Scenario, crlb and its pieces,
                       montecarlo, compare (a dataset fit vs its truth), diagnostics (cond,
                       cor; leakage and sensitivity maps later), sweeps (see "Simulation, CRLB
                       and protocol design")
  design/              (later) protocol objective (Eq. 5 and options), scan time, constraints,
                       tissue sets
  io/                  (later) loaddata, protocol files, units, savemaps, PAR/REC reader
ext/                   (later) package extensions: OptimExt (optimize_protocol), and one per file
                       format package (NIfTI, MAT, NPZ, DICOM, JSON3, MRIFiles)
test/                  correctness and deterministic performance checks (no allocation, type
                       stability); no timings
bench/                 own Project.toml (BenchmarkTools, LsqFit, NonlinearSolve, FastLM);
                       solvers.jl: EQUIP vs the other solvers over the test cases in cases.jl.
                       (later) speed suite + accuracy studies, results.md
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

- **Precompile workload** (`src/precompile.jl`, PrecompileTools.jl): while EQUIP precompiles, it
  fits every built-in model with its default roles (one voxel; a 2-voxel matrix with and without
  a mask; 4-D data), runs `simulate` and the REPL summary, plus QMTSIR with kmf free and T1MFA
  with a B1 map. Since Julia 1.9 the compiled code is saved with the package. Measured in fresh
  processes, one thread, 10 000 voxels:

  | | before | after |
  |---|---|---|
  | `using EQUIP` | 0.3 s | 0.4 s |
  | first QMTSIR fit (compiles before) | 3.8–4.1 s | 0.03–0.05 s (as fast as a compiled fit) |
  | first QMTSIR fit with kmf free | 1.6 s | 0.05–0.06 s |
  | first T1IR fit | 1.3–1.4 s | 0.04 s |
  | T1IR with Sf fixed (not in the workload) | 0.9 s | 0.6–0.7 s |
  | T1IR in Float32 (not in the workload) | 2.3–2.5 s | 2.4–2.5 s |
  | whole process (all of the above, with `simulate`) | 12–13 s | 4.6–4.9 s |

  The price: precompiling EQUIP takes about 41 s instead of about 3 s, once per install or code
  change. For development, the preference `precompile_workload = false` (in
  `LocalPreferences.toml`, not committed) switches the workload off: about 3 s per change;
  Julia keeps both compiled versions, so switching back costs nothing. Float32 is left out
  (it would roughly double the precompile time; Float32 users pay ~2.5 s once per session).
  The test suite also runs faster (45 s vs ~60 s). Still to do for the command line (item 10): a
  system image or a standalone app (PackageCompiler.jl), or a Julia process kept running.

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
- **Grid-search start for magnitude inversion recovery**
  (`T1 = Free(start = BestOf(_t1grid(m.TI)))`; `_best_start` in `src/fit/starts.jl`; it was the `startgrid` hook when measured). |S| has local minima: from one start a fit can settle with the signal
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
  work into one block per thread). See the comment in `src/fit/dataset.jl`.
- **Against LsqFit and NonlinearSolve** (`bench/solvers.jl`, formerly `sir_4point_vs.jl`; first on
  `examples/sir_wm_4point.jl`'s data, every voxel white matter at SNR 200). Everything shared: EQUIP's `signal!`, the starts, no bounds, ForwardDiff,
  the threads, the number of runs, and the stopping rule: a relative parameter step of 1e-8,
  every other test off. That rule cannot be made identical: EQUIP scales the step by
  Marquardt's D^½, LsqFit does not, and NonlinearSolve's least-squares default tests the
  residual, which never becomes small with noisy data, so the script gives it a step rule
  (`RelNormSafeBestTerminationMode`, one step, relative to the start). Measured, 500 voxels,
  4 threads, µs per voxel: EQUIP 0.94 (0.66 on 10 000 voxels, 200 runs), LsqFit 18, NonlinearSolve
  TrustRegion 3.5, NonlinearSolve LevenbergMarquardt 155; mean steps per voxel 5.0, —, 3.0, 8.8;
  all converged on every voxel, estimates within 5e-8 of EQUIP's. LsqFit and NonlinearSolve's
  LevenbergMarquardt do not speed up with threads (LsqFit 17 µs on 1 thread, SciML's LM 58 µs
  on 1 thread vs 155 on 4): they allocate on every voxel, and memory cleanup stops all threads.
  TrustRegion is the closest; it takes fewer, Gauss–Newton-like steps. LsqFit's README still
  describes central finite differences as its default; version 0.16 defaults to ForwardDiff.
  - On an Apple M2 Max (10 000 voxels, 20 runs, ms per run, 1/2/4/8 threads): EQUIP 14.3/7.6/
    4.3/2.3, TrustRegion 38.1/20.4/11.1/5.8, LsqFit 94.3/77.2/97.3/165, SciML LM 284/405/625/729.
    Per core EQUIP is 2.7× faster than TrustRegion, 6.6× than LsqFit, 20× than SciML's LM.
  - Why TrustRegion takes fewer steps: by default it is Moré's trust region (MINPACK `lmpar`),
    i.e. Levenberg–Marquardt with the damping chosen to fit a trust radius, and zero (a pure
    Gauss–Newton step) whenever the Gauss–Newton step fits. EQUIP's Nielsen update always keeps
    some damping. With 4 observations and 4 free parameters every fit is exact (median SSE
    ~1e-23), which favours Gauss–Newton steps. A lower initial damping (1e-5) took EQUIP from 5
    to 4 steps, 11% faster, same answers. To try next: Moré-style damping in `levmar`
    (prototype branch), then possibly a `TrustRegion` solver option (item 11).
  - Bounds cost nothing measurable here (no voxel reaches one): EQUIP 1.90–1.93 µs/voxel without,
    1.93–2.00 with (3 rounds of 50 runs, alternating), LsqFit 18.0 vs 17.6 µs (one thread).
  - EQUIP allocates nothing per voxel (`fit!` with a cache: 0 bytes); a dataset call allocates its
    result maps (48 bytes per voxel) and ~2600 small objects (setup, per-task caches, tasks),
    whatever the number of voxels. LsqFit allocates ~8.7 KB per voxel.
  - On `examples/sir_4point_range.jl`'s data (PSR, kmf, R1f uniform over the range of the four
    SIR classes, Sf -1 to -0.8, M0f 0.8–1.2, SNR 200; 4 threads, 3 runs, cloud machine): µs per
    voxel EQUIP 0.72, TrustRegion 1.69, LsqFit 18.3, SciML LM 152; steps 5.3, 3.6, —, 8.7; all
    converged everywhere and reached EQUIP's minimum, except 3 voxels where TrustRegion stopped
    at a clearly higher SSE.
  - SimpleNonlinearSolve (SciML's low-overhead solvers for small, static problems) is the real
    competitor: set up its fast way (an out-of-place residual returning an SVector, EQUIP's
    `signal!` inlined into it with `@inline` so its MVector stays off the heap, the
    NonlinearFunction built once, a new problem per voxel: ~300 bytes per voxel, 0 inside the
    solve) SimpleGaussNewton runs at EQUIP's speed: range data, 5 runs, µs per voxel on 1/4
    threads EQUIP 2.39/0.83, SimpleGaussNewton 2.54/0.95, SimpleTrustRegion 3.12/1.02 (3 voxels
    at a clearly higher SSE). Set up naively (a new problem from a plain function, no
    `@inline`) the same solve took 15 µs. SimpleGaussNewton is pure Gauss–Newton (no damping, no
    safeguard), and its default stop tests the residual norm, which only works here because
    every fit is exact (4 points, 4 parameters); with more points than parameters it would run
    to maxiters. So EQUIP's speed lead on this test is over typical usage (LsqFit, NonlinearSolve's
    general solvers), not over the best hand-tuned SciML setup; the harder tests (more points
    than parameters, poor starts) and the damping prototype decide the rest.
  - FastLevenbergMarquardt.jl (Kames; a small Levenberg–Marquardt for static arrays, with
    bounds, depending only on StaticArrays and the standard library, `using` 0.43 s; v0.1.0, one
    maintainer; needs a Jacobian function, here ForwardDiff on the same static residual): range
    data, one thread, µs per voxel EQUIP 2.08, FastLM 2.12 (steps 5.3 vs 5.1; same answers on
    every voxel, nothing allocated per voxel); 4 threads, 5 runs: FastLM 0.66, SimpleGaussNewton
    0.76, EQUIP 0.80, SimpleTrustRegion 0.92. Its defaults: initial damping 1e-6, damping ÷13
    after a good step, ×3 after a rejected one; per-parameter step test; returns the reason it
    stopped (ftol, xtol, gtol, iteration limit) and the iteration count.
  - Conclusion so far: three independent static implementations (EQUIP, SimpleGaussNewton,
    FastLevenbergMarquardt) land at the same speed. On small problems the time is the model and
    its dual-number derivatives times the number of steps; the solver library hardly matters
    once its overhead is gone. The choice between them is robustness, bounds, stopping and status
    codes, load time and maintenance, so the harder tests decide; EQUIP's threaded dataset
    driver (0.80 vs FastLM's plain threaded loop at 0.66 on 4 threads) is worth a look.
  - **The harder cases** (`bench/cases.jl`; 1000 voxels per case, 1 run, 4 threads, cloud
    machine). "Not best": voxels where the method's fit is clearly worse (SSE above the best by
    more than 1e-8 Σy²) than the best any method found (among methods solving the same problem:
    with bounds or without), or failed. µs per voxel (not best):

    | method | sir4_range | sir4_poorstart | sir16_kmf | sir16_kmf_snr50 | sir4_bounds | biexp |
    |---|---|---|---|---|---|---|
    | EQUIP | 1.18 (0) | 4.11 (26) | 4.58 (5) | 9.78 (20) | 1.32 (0) | 20.5 (218) |
    | LsqFit | 16.7 (0) | 20.7 (2) | 34.0 (3) | 115 (30) | 153 (214) | 316 (122) |
    | NonlinearSolve LM | 178 (0) | 607 (9) | 2965 (4) | 5250 (36) | 182 (0)* | 3758 (16) |
    | NonlinearSolve TrustRegion | 2.85 (0) | 11.9 (18) | 109 (59) | 153 (71) | 4.48 (0)* | 92 (38) |
    | SimpleGaussNewton | 2.16 (0) | 4.49 (138) | 184 (281) | 298 (430) | 2.41 (0)* | 2948 (951) |
    | SimpleTrustRegion | 5.15 (0) | 7.06 (11) | 1053 (427) | 1394 (561) | 3.43 (1)* | 2265 (995) |
    | FastLevenbergMarquardt | 5.63 (0) | 5.25 (33) | 8.10 (10) | 23.9 (24) | 14.0 (211) | 140 (151) |

    \* without bounds (they take none), compared among themselves.

    The everyday relaxometry models, added after (same settings): µs per voxel (not best)

    | method | t2 | t1mfa | t1ir |
    |---|---|---|---|
    | EQUIP | 0.79 (0) | 0.74 (0) | 0.69 (9) |
    | LsqFit | 14.5 (0) | 15.4 (0) | 15.2 (8) |
    | NonlinearSolve LM | 716 (0) | 965 (0) | 1354 (7) |
    | NonlinearSolve TrustRegion | 11.8 (0) | 11.6 (0) | 12.0 (10) |
    | SimpleGaussNewton | 91.6 (0) | 51.9 (0) | 48.5 (10) |
    | SimpleTrustRegion | 172 (39) | 83.5 (0) | 94.1 (11) |
    | FastLevenbergMarquardt | 2.17 (0) | 6.86 (0) | 1.26 (10) |

    On T2 decay and variable flip angle every method finds the same fits; on magnitude
    inversion recovery every method misses the best fit in ~1% of voxels (false minima around
    the null point), EQUIP no more often than the others.
  - **Diagnosis of EQUIP's misses** (1000 voxels, one thread, settings only, no code change):
    - Biexponential without bounds: 202 voxels not best. 199 of them stopped on the step test
      (a stationary point: restarting there does not move it), at SSE ~2.1× the best, with the
      two components merged (T2short ≈ T2long ≈ 63-65 ms) and fshort outside 0-1 (median -0.06;
      some T2s absurd, -2.3e4 and 4e3 s). LsqFit from the same starts finds the biexponential
      (fshort 0.31, T2s 27 and 80 ms). Not early stopping: EQUIP's normal tolerances, 10× the
      iterations, xtol 1e-12, damping 1e-6 or a restart change nothing (202-226). EQUIP's path
      runs into the degenerate region, the other solvers' paths less often.
    - With the model's bounds (normal use) the problem disappears: not best 4 (EQUIP, benchmark
      settings), 22 (EQUIP's defaults; LsqFit bounded 22, FastLM bounded 243). No final fit sits
      on a bound: the bounds keep the path physical, the answers are interior.
    - EQUIP's defaults on the bounded biexponential: ~6% of voxels crawl in a narrow valley
      (MaxIterations at 100: 65 voxels; at 1000 still 60). Raising maxiter barely helps (worse
      than a very thorough fit: 18 at 100, 16 at 300, 13 at 1000 iterations, 21.6 → 58.2 µs per
      voxel), so the default stays. The crawl is the target for step control (below).
    - Poor starts without bounds: 23 of EQUIP's 24 misses ran away (R1f → 0, M0f → ~4e5) to the
      iteration limit; bounds prevent that (R1f ≥ 0.05). With bounds and the same poor starts,
      every bounded method misses the best fit in ~37-42% of voxels (EQUIP 383, LsqFit 415,
      FastLM 367): the 4-point scheme's several exact fits, not a solver problem.
    - The benchmark now runs every case twice: without bounds (every method) and with the
      models' bounds (EQUIP, LsqFit, FastLM). With bounds (1000 voxels, 1 run, 4 threads), µs per
      voxel (not best among the three):

      | method | t2 | t1mfa | t1ir | sir4_range | sir4_poorstart | sir16_kmf | sir16_kmf_snr50 | sir4_lowpsr | biexp |
      |---|---|---|---|---|---|---|---|---|---|
      | EQUIP | 1.86 (0) | 0.68 (0) | 0.84 (9) | 1.26 (0) | 2.60 (26) | 4.83 (3) | 5.42 (11) | 1.19 (0) | 17.3 (4) |
      | LsqFit | 13.9 (0) | 14.8 (0) | 16.8 (8) | 19.3 (0) | 610 (409) | 27.2 (1) | 95.8 (13) | 139 (214) | 230 (22) |
      | FastLM | 1.48 (0) | 2.04 (0) | 1.45 (10) | 1.92 (0) | 27.5 (357) | 11.2 (7) | 22.8 (24) | 14.3 (211) | 31.6 (243) |

      With bounds EQUIP finds the best fit as often as or more often than the others on every
      case, and is fastest except T2 decay (1.86 vs FastLM 1.48, one run; without bounds EQUIP
      0.80). The unbounded table is in the earlier bullet (`sir4_bounds` there is now
      `sir4_lowpsr`, and its rows there were mixed: bounded EQUIP/LsqFit/FastLM, unbounded SciML).
    - On an Apple M3, 8 threads, Julia 1.13.2 (1000 voxels, 3 runs; the "not best" counts are the
      same as on the cloud machine, since the data come from fixed seeds), µs per voxel (not best):

      | without bounds | t2 | t1mfa | t1ir | sir4_range | sir4_poorstart | sir16_kmf | sir16_kmf_snr50 | sir4_lowpsr | biexp |
      |---|---|---|---|---|---|---|---|---|---|
      | EQUIP | 0.30 (0) | 0.29 (0) | 0.30 (9) | 0.44 (0) | 1.74 (26) | 2.09 (5) | 5.41 (20) | 0.38 (0) | 9.67 (218) |
      | LsqFit | 11.9 (0) | 13.4 (0) | 14.4 (8) | 13.9 (0) | 18.1 (2) | 25.8 (3) | 86.1 (30) | 14.2 (0) | 254 (122) |
      | NonlinearSolve LM | 590 (0) | 559 (0) | 776 (7) | 86.7 (0) | 285 (9) | 1509 (4) | 2551 (36) | 89.5 (0) | 2626 (16) |
      | NonlinearSolve TrustRegion | 4.19 (0) | 3.26 (0) | 4.08 (10) | 1.16 (0) | 6.02 (18) | 51.4 (59) | 54.9 (71) | 1.28 (0) | 30.3 (38) |
      | SimpleGaussNewton | 46.9 (0) | 61.2 (0) | 62.9 (10) | 0.76 (0) | 4.05 (138) | 164 (281) | 256 (430) | 0.75 (0) | 2569 (951) |
      | SimpleTrustRegion | 139 (39) | 94.8 (0) | 98.6 (11) | 0.85 (0) | 3.59 (11) | 988 (426) | 1288 (557) | 0.96 (1) | 2085 (995) |
      | FastLM | 0.75 (0) | 0.55 (0) | 0.51 (10) | 0.60 (0) | 2.09 (33) | 4.10 (10) | 9.60 (24) | 0.61 (0) | 17.0 (151) |

      | with bounds | t2 | t1mfa | t1ir | sir4_range | sir4_poorstart | sir16_kmf | sir16_kmf_snr50 | sir4_lowpsr | biexp |
      |---|---|---|---|---|---|---|---|---|---|
      | EQUIP | 0.31 (0) | 0.29 (0) | 0.30 (9) | 0.39 (0) | 0.79 (26) | 1.43 (3) | 2.79 (11) | 0.41 (0) | 9.90 (4) |
      | LsqFit | 11.7 (0) | 13.1 (0) | 14.0 (8) | 14.3 (0) | 547 (409) | 26.1 (1) | 84.7 (13) | 126 (214) | 207 (22) |
      | FastLM | 0.66 (0) | 0.53 (0) | 0.48 (10) | 0.68 (0) | 13.2 (357) | 3.89 (7) | 8.70 (24) | 6.99 (211) | 12.9 (243) |

      EQUIP is the fastest on every case in both runs: 1.2-2.5× FastLM without bounds (1.3-17×
      with), 10-50× LsqFit, 160-2500× NonlinearSolve's LM. With bounds it misses the best fit
      about as rarely as the best of the others everywhere (t1ir 9 vs 8, sir16_kmf 3 vs 1) and far
      more rarely on poor starts, small PSR and the biexponential.
    - "Not best" is not "less accurate" where the fit is not unique: on sir4_poorstart with bounds
      EQUIP finds the lowest SSE (26 vs 409 and 357) yet its median PSR error is 230% against
      FastLM's 73%. Several exact fits exist; FastLM's stuck, higher-SSE answers happen to lie
      nearer the truth. Judge that case by the truth error, not the SSE.
    - **Step-control prototype** (branch `step-proto`: LM options `scaling`, `update`, `gnfirst`,
      `guard`, `trust`; `bench/steps.jl`, results in `bench/steps_results.txt`). Every case, 1000
      voxels, with and without bounds; "not best" against the best of all variants and LsqFit,
      FastLM and NonlinearSolve's TrustRegion. Totals over the nine cases (without / with
      bounds): today's solver 262 / 71; Marquardt update 260 / 92; damping 1e-6 283 / 70;
      Gauss–Newton step first 300 / 73; dead-zone guard off 264 / 220; scaling recomputed each
      step (as LsqFit) 432 / 347; recomputed for the damping only, growing for the step test
      411 / 348; a MINPACK-style trust radius 272 / 75; LsqFit 160 / 674, FastLM 220 / 859.
      - Decision: keep today's step control. With bounds (normal use) it is already the best;
        the growing Marquardt scaling and the dead-zone guard both protect the path (biexponential
        with bounds: 6 misses; recomputed scaling 280, guard off 158). Recomputing the scaling
        halves the biexponential steps (86 → 44) but lands on worse fits.
      - Without bounds no variant closes the biexponential gap (today 209, best variant 199,
        NonlinearSolve's TrustRegion 41): the fits stop in the degenerate region where the two
        components merge. Better handled by the model than by the solver: keep bounds (the
        default), parametrize the biexponential so the components cannot swap or merge (e.g.
        T2long = T2short + ΔT2, ΔT2 ≥ 0), or variable projection (`varpro-proto`: more
        biexponential fits reached the minimum). The trust radius helps poor starts (24 → 10)
        and 16-point kmf (4 → 0) without bounds only; not adopted.
    - The biexponential and SNR (1000 voxels, the benchmark's T2 ranges and starts; EQUIP vs
      LsqFit, and NonlinearSolve's TrustRegion without bounds). The conditioning depends on the
      T2 separation, the amplitude ratio, the TE sampling and the SNR; in practice it needs SNR
      above about 100-200. Median fshort error with bounds, EQUIP / LsqFit: SNR 100 28% / 28%,
      200 11% / 11%, 500 4% / 4%: at SNR 100 fshort is barely determined, for every method.
      EQUIP not best with bounds: 4 at every SNR (LsqFit 22, 6, 0). Without bounds EQUIP's misses
      barely fall with SNR (202, 178, 152; TrustRegion 30, 13, 17), so they are EQUIP's path from
      the default starts, not the conditioning; at SNR 100 they cluster at small T2 ratios
      (T2long/T2short 2-3: 40% of voxels; 3-5: 22%; 5-12: 11%) and, unexpectedly, at large fshort
      (0.05-0.13: 5%; 0.22-0.30: 37%).
    - The benchmark's `biexp` case is now at SNR 200 (the realistic minimum); the SNR 100 case
      the tables above call `biexp` is now `biexp_snr100`, kept as a stress case. 1000 voxels,
      1 run, 4 threads, µs per voxel (not best), biexp / biexp_snr100: without bounds EQUIP
      8.67 (180) / 14.7 (218), LsqFit 161 (133) / 294 (122), NonlinearSolve LM 2413 (2) / 3544
      (16), TrustRegion 40.6 (16) / 67.6 (38), FastLM 32.0 (118) / 117 (151); with bounds EQUIP
      9.40 (4) / 22.4 (4), LsqFit 128 (6) / 228 (22), FastLM 9.71 (282) / 31.2 (243).
    - **VarPro on these biexponential cases** (branch `varpro-proto`,
      `prototypes/varpro/biexp_bench.jl`; LM on the two T2s, the two amplitudes solved exactly,
      ≥ 0 with bounds). Not best out of 1000, EQUIP full → VarPro (best reference): SNR 200 no
      bounds 180 → 8 (TrustRegion 15); with bounds 4 → 4 (LsqFit 6); SNR 100 no bounds 213 → 16
      (TrustRegion 42); with bounds 13 → 11 (LsqFit 26). Steps ~20% fewer (34.5 → 27.4, 31.1 →
      25.6, 80.1 → 62.3, 86.3 → 67.0). Without bounds its errors drop to the best any method
      reached (SNR 100 median fshort error 43.7% → 28.0%). Speed about the same (two runs
      disagreed on which was faster). VarPro removes the full fit's unbounded path problem: the
      amplitudes can no longer wander, so the two components do not merge.
      - Recommendation: add VarPro to EQUIP for models with linear amplitudes, starting with the
        biexponential (since designed: "Variable projection (VarPro): proposed design"). The
        questions were: how a model declares its linear parameters and columns (a
        method like `linearparts(m)`, or columns from `signal!` at unit amplitudes); mapping
        amplitudes back to the model's parameters (S0, fshort); bounds (amplitudes ≥ 0 by the
        active set; other bounds on amplitudes?); Float32; statuses and `FitResult`; and when it
        is used (automatically for such models, or `solver = LM(varpro = true)`). Since answered
        there: `varpro = true` per parameter, `LM(varpro = false)` to turn it off.
    - Next: VarPro in EQUIP (see "Variable projection (VarPro): proposed design"); and,
      later, errors judged against the Cramér–Rao bound (roadmap item 3), so methods are compared
      where the data determine the parameters. Candidates: Marquardt scaling recomputed
      each step (LsqFit) instead of only growing (`d = max.(d, diag(A))`), the damping update,
      a Gauss–Newton step first. Targets: the unbounded biexponential path (202), the crawling
      voxels, steps on every case; nothing may get worse with bounds. The benchmark should also
      report the bounded configuration, EQUIP's normal use, next to the unbounded one.
    - The SciML solvers stop by their own least-squares default here: a shared step rule makes
      them stop early on hard fits, because their stopping machinery sees a rejected step as no
      movement (biexp, 2000 voxels, NonlinearSolve TrustRegion: worse than the best in 1934
      voxels with a one-step rule, 639 with ten steps, 70 with its default).
    - EQUIP is the fastest on every case, and the most reliable with bounds (LsqFit and FastLM
      stall at the bound: ~210 worse fits). Its weak spot is the biexponential: 218 of 1000 voxels
      not best (SciML's LM 16, TrustRegion 38, LsqFit 122), ~88 steps per voxel, 94.7% converged,
      larger fshort and T2short errors. From poor starts it is also slightly worse than LsqFit
      (26 vs 2). Both point at EQUIP's damping and stopping, the next prototype.
    - Pure Gauss–Newton (SimpleGaussNewton) fails once the fits are not exact or the starts are
      poor; SimpleTrustRegion converges on few voxels of the 16-point and biexponential cases.
    - The 4-point scheme from poor starts: every method converges, but the median PSR error is
      350–450% for all of them. With 4 observations and 4 free parameters there can be more than
      one exact fit, and far starts find the wrong one: the scheme depends on sensible starts.

## Variable projection (VarPro): proposed design

Status: proposed, not implemented. Prototype on branch `varpro-proto` (`prototypes/varpro/`,
with its README); measurements in the Performance notes. On the benchmark's biexponential
(`bench/cases.jl`, 1000 voxels, every voxel different): voxels not best, EQUIP's full fit →
VarPro, SNR 200 without bounds 180 → 8 (NonlinearSolve's TrustRegion 15), with bounds 4 → 4;
SNR 100 without bounds 213 → 16 (TrustRegion 42), with bounds 13 → 11; ~20% fewer steps
everywhere; speed about the same.

### What it is

Many signals are linear in some of their parameters: S = s₀(θ) + Σₖ aₖ·sₖ(θ), with K amplitudes
aₖ and columns sₖ that depend only on the nonlinear parameters θ (T2s, T1, ...), plus an affine
part s₀ when fixed parameters add to the signal. For any θ the best amplitudes are a small
linear least-squares solve, a(θ) = (SᵀS)⁻¹Sᵀ(y − s₀) with S = [s₁ … s_K] (M × K). Variable
projection (Golub & Pereyra 1973) lets LM fit θ only, with the amplitudes solved at every
evaluation: it minimizes ‖s₀(θ) + S(θ)·a(θ) − y‖². LM then has fewer, less correlated
parameters, and the amplitudes can never wander. The biexponential's misses without bounds were
exactly that: its full fit drifts through unphysical amplitudes until the two components merge.

In EQUIP the amplitudes are solved in dual arithmetic, so the reduced problem's derivatives are
exact (no Kaufman approximation), and `levmar` does not change: the reduced problem supplies its
own `normal_equations`, as in the prototype. Least squares only: not for Rician-likelihood
fitting (roadmap), and with magnitude data only amplitudes that keep the signal's sign are
linear (|M0·shape| = M0·|shape| for M0 ≥ 0; a sum of decays with amplitudes ≥ 0 is positive).

### Declaring it: `varpro = true`, a setting of its own

A parameter to be solved by projection says so with its own keyword on `Free`, next to its start
and bounds, in the model's `parameters` or in a fit:

```julia
parameters(m::T2DecayOffset) = (
    S0 = Free(start = :scale, lower = 0, varpro = true),   # a start and solved: both
    T2 = Free(start = mean(m.TE), lower = _TMIN),
    C = Free(varpro = true),                              # offset: no start needed
)
```

- **Decision: `varpro` is its own setting, not a start.** A start says where the iteration
  begins; `varpro = true` says how the parameter is determined (solved exactly at every
  evaluation, never iterated). They are different questions, and a scale parameter has an
  answer to both: `S0 = Free(start = :scale, lower = 0, varpro = true)`. This replaces the
  start `Free(:varpro)` proposed first (751d249), under which a scale parameter could not be
  projected without giving up `:scale`, and a per-fit `Free(start = 0.2)` silently turned
  projection off. The keyword-only `Free` (8be779a) makes the new setting read like the others.
- **Linear parameters other than the scale need no start**: `fshort = Free(lower = 0,
  upper = 1, varpro = true)`. When the parameter is projected, it is solved and a start would go
  unused. When it is fitted the ordinary way instead (`LM(varpro = false)`, a per-fit
  `varpro = false`, or a voxel's fallback, below), it starts at its least-squares value given
  the nonlinear parameters' starts (`_score`, generalized from one amplitude to K), so the data
  still need no normalization. A start given anyway is used only in that ordinary fit.
- **`:scale` stays a model fact**: "the signal is proportional to this parameter". That gives
  its start (the least-squares scale), the parameter `simulate`'s `snr` is relative to
  (σ = S0/snr), and its linearity, so `S0 = Free(varpro = true)` in a fit needs nothing else.
  A scale parameter without `varpro = true` is fitted the ordinary way, as today.
- **`varpro = true` in a model's `parameters` also declares the parameter linear** (checked at
  setup, below), which is what gives the ordinary fit its least-squares start.
- **Per fit**, like the start and bounds (whatever is left out keeps the model's default):
  `S0 = Free(varpro = true)` projects a scale parameter the model leaves to LM;
  `C = Free(varpro = false)` fits the offset the ordinary way; on a custom model's parameter,
  `varpro = true` opts in with no code to write (the check decides). `solver = LM(varpro =
  false)` turns projection off for the whole fit (for comparisons, and as an escape hatch).
- **Built-in models set `varpro = true` only where it was measured**: the biexponential
  (robustness, above) and the decay with offset (the prototype's other two-amplitude model).
  The scale parameter alone in the other models (M0, S0, M0f) waits for a measurement: the
  prototype found 0–22% faster and robustness unchanged there, and it changes every fit's path
  (same minima within the tolerance, not bit for bit).
- **Simulation does not change**: `varpro` is part of a role, and `signal` and `simulate` take
  values (passing roles is an error), so it never reaches them.
- Plain amplitudes need nothing else: M0, S0, an offset C, the components of a multi-compartment
  model written as amplitudes. EQUIP builds the columns from the model's own `signal!` (below).

### What EQUIP does with them

For each fit, `setup` takes the free parameters with `varpro = true` (the "linear part", K
parameters; a `varpro` parameter fixed or tied in this fit is simply fixed), and:

1. **Builds the columns from `signal!`, as differences around a base point.** With the
   nonlinear parameters at θ and the amplitudes at a base point b: column k is
   sₖ(θ) = signal(b + eₖ) − signal(b), and the affine part s₀(θ) = signal(b) − Σₖ bₖ·sₖ(θ).
   For plain amplitudes b = 0 works (then s₀ = signal(0), skipped when setup finds it zero).
   With `fromamplitudes` (below), b has every amplitude non-zero, b = (1, …, 1): unit vectors
   eₖ themselves can map to invalid parameters (for SIR, Sf = a₂/a₁ = 1/0). So K + 1 model
   evaluations, in dual numbers, per LM evaluation (K when b = 0 and s₀ = 0).
2. **Checks that the signal really is linear** in those parameters, once per fit call: at a few
   random amplitudes (within the parameters' bounds) and the start's nonlinear values, the
   signal must equal s₀ + Σₖ aₖ·sₖ to within rounding. A wrong declaration is an error that
   names the parameter ("the signal of T2DecayOffset is not linear in T2"). The check also
   catches magnitude data: |M0·shape| = M0·|shape| holds for M0 ≥ 0, so a magnitude model's M0
   passes with its bound M0 ≥ 0 and fails without it, and magnitude SIR fails for Sf (below).
3. **Sorts the bounds.** Bounds that say "amplitudes ≥ 0" are kept inside the projection by an
   active set (below): a plain amplitude with `lower = 0` and no upper bound, or a model's
   default bounds when its `fromamplitudes` says they mean a ≥ 0 (the biexponential's S0 ≥ 0
   and 0 ≤ fshort ≤ 1). No bounds: unconstrained. **Any other bound** (S0 ≥ 100, C ≤ 5, SIR's
   −1.2 ≤ Sf ≤ 0, i.e. −1.2·a₁ ≤ a₂ ≤ 0, a cone rather than a ≥ 0): the projection solves
   without it, and a voxel whose answer leaves the bounds is refitted with the full fit,
   starting from the VarPro answer moved into the bounds. When the bound does not bind (most
   voxels, for sensible bounds), VarPro's answer is the bounded answer; only the voxels where
   it binds pay for a second fit. (The first design fell back for the whole fit call.)

### Parameters that are not themselves amplitudes: one optional mapping

The biexponential, S = S0·[fshort·exp(−TE/T2short) + (1 − fshort)·exp(−TE/T2long)], is linear in
the amplitudes a₁ = S0·fshort and a₂ = S0·(1 − fshort), but not in S0 and fshort (they multiply
each other), so declaring S0 and fshort alone fails the check. Such a model keeps its
parameters (myelin water users want fshort) and adds one method, the parameters from the
amplitudes:

```julia
parameters(m::T2BiexpDecay) = (
    S0 = Free(start = :scale, lower = 0, varpro = true),
    fshort = Free(lower = 0, upper = 1, varpro = true),  # fraction of S0 in the short component
    T2short = Free(start = mean(m.TE) / 4, lower = _TMIN),
    T2long = Free(start = 2 * mean(m.TE), lower = _TMIN),
)
fromamplitudes(::T2BiexpDecay, a) = (S0 = a[1] + a[2], fshort = a[1] / (a[1] + a[2]))
```

- With it, EQUIP builds the columns from the signal at `fromamplitudes(b)` and
  `fromamplitudes(b + eₖ)` (the base point above: b = (1, 1) is S0 = 2, fshort = 0.5), checks
  linearity in the amplitudes the same way, and reports the linear parameters through
  `fromamplitudes` after the fit. Setup checks that `fromamplitudes(b)` is finite.
- Bounds: the biexponential's default bounds (S0 ≥ 0, 0 ≤ fshort ≤ 1) are exactly a ≥ 0, which
  the model states (how: an open decision); bounds changed in a fit (fshort ≤ 0.5) take the
  per-voxel fallback above.
- Not all of them free or projected in this fit (e.g. `fshort = 0.2` fixed, or `fshort =
  Free(varpro = false)`): the mapping does not apply; the remaining `varpro` parameters are
  treated as plain amplitudes, and the check decides (here S0 alone passes: the signal is
  linear in S0 at a given fshort, so K = 1).
- With projection off, fshort now starts at its least-squares value rather than today's 0.2:
  a change to the ordinary biexponential fit, measured before the declaration is adopted.
- **A second example, SIR on signed data.** The free-pool signal of QMTSIR (Dortch et al., MRM
  2018, Eq. 1) is Mzf = M0f·[Sf·g(θ) + h(θ)], with θ = (PSR, R1f, kmf, R1m, Sm, ...): Sf enters
  only through the inversion of the free pool, and M0f scales everything. So with
  `magnitude = false` (phase-corrected data) the signal is linear in the amplitudes a₁ = M0f
  and a₂ = M0f·Sf, with `fromamplitudes(::QMTSIR, a) = (M0f = a[1], Sf = a[2] / a[1])`, and
  LM would fit PSR and R1f only (kmf fixed). The unit vector e₂ maps to M0f = 0, Sf = 1/0,
  which is why the columns come from a base point with every amplitude non-zero. Magnitude
  data (the default) is not linear in Sf: |M0f·(Sf·g + h)| changes form where the signal
  crosses zero, so the check refuses Sf there (M0f alone, with M0f ≥ 0, still passes).
  Checked numerically (4-point scheme, PSR 0.12, R1f 1, Mzf_td0 0.05, columns from the base
  point (1, 1), 200 random M0f 0.5–1.5 and Sf −1.2–0): signed, the signal equals a₁·s₁ + a₂·s₂
  to 1e-15·M0f, with no affine part; magnitude, it misses by up to 2.3·M0f. Its
  bounds, M0f ≥ 0 and −1.2 ≤ Sf ≤ 0, are a cone in (a₁, a₂), not a ≥ 0: the per-voxel
  fallback. Not proposed for QMTSIR's defaults until measured (a signed SIR case in the
  benchmark first).
- The alternative, parametrizing the model by its amplitudes (Sshort, Slong) and reporting S0
  and fshort as derived values, is cleaner for the solver but changes what the model fits and
  reports, and needs derived outputs (roadmap); not proposed.

### An optional speed hook: the columns

Columns built from `signal!` cost one full model evaluation per column (plus one for the base
point), in dual numbers, at every LM evaluation. For the biexponential the prototype measured
that route slower than the full fit (23.1 → 27.2 µs per voxel), against 19.3 µs with columns
computed directly. So a model may also give its columns, each computing only its own terms:

```julia
function columns!(cols, m::T2BiexpDecay, p)       # p: the parameters (may hold dual numbers)
    cshort, clong = cols
    @inbounds for j in eachindex(m.TE)
        cshort[j] = exp(-m.TE[j] / p.T2short)
        clong[j] = exp(-m.TE[j] / p.T2long)
    end
    return cols
end
```

Not needed for correctness: a test checks it against the columns built from `signal!`.

### When it is used

- Per parameter: a free parameter with `varpro = true` (the model's default or the fit's) is
  projected; the others are fitted by LM as today. A fit with no such parameter fits exactly
  as today, so models that declare none are unchanged, bit for bit.
- Per fit: `solver = LM(varpro = false)` fits every parameter the ordinary way (the `varpro`
  ones from their least-squares starts). The default, `LM(varpro = true)`, follows the
  parameters' settings.
- Per voxel: the full fit for a voxel whose projected answer leaves a bound the projection
  cannot hold (above).
- `res.settings` and the REPL summary say which parameters were projected, and how many voxels
  fell back to the full fit.

### Inside the fit

- `setup`: splits the free names F into the nonlinear names Fn and the linear ones (K); Fn, K,
  the amplitude constraints and whether there is an affine part go into the type, as F does now,
  so everything is specialized and nothing is looked up at run time. The full problem is
  compiled too, for the per-voxel fallback (more compile time for such fits).
- `init`: K + 1 column buffers of dual numbers per cache, so one set per task, as the model
  buffer today.
- Per voxel (`_fit_voxel`): the nonlinear parameters' starts as today (numbers, maps, `:data`,
  `BestOf`). `BestOf` scores each value with the K amplitudes solved (`_score` generalized from
  K = 1). Then `levmar` on the reduced problem; then the amplitudes at the solution → the linear
  parameters (themselves, or `fromamplitudes`) → the bound check (the fallback if outside) →
  every parameter → `relabel` as today (the biexponential's components in order). In the
  ordinary fit, `_score`'s amplitudes are the linear parameters' starts, and `levmar` runs on
  the full problem as today.
- `normal_equations` of the reduced problem: fill the columns at θ (duals), solve the K × K
  system for the amplitudes in dual arithmetic, then residuals rᵢ = s₀ᵢ + Σₖ aₖ·sₖᵢ − yᵢ with
  their partials → JᵀJ, Jᵀr and SSE as now. K = 1: a = (s·(y − s₀))/(s·s); K = 2: closed form;
  larger K: the small Cholesky already in `levmar.jl`.
- Amplitudes ≥ 0: an active set. An amplitude that comes out negative is set to 0 and the others
  re-solved (K = 2: closed form, as the prototype; larger K: Lawson–Hanson NNLS). The reduced
  problem has a kink where the active set switches; LM's acceptance test copes (prototype).
- Collinear columns (the two T2s equal): the 2 × 2 system is singular. Detect it (determinant
  ≤ ε·g₁₁·g₂₂) and keep the better single-column fit, so the result is a monoexponential
  (fshort 0 or 1), not NaN.
- Float32: the K × K normal equations square the columns' condition number, and nearly collinear
  columns lose the amplitudes in Float32. Proposed: accumulate the K × K Gram matrix and Sᵀy in
  Float64 (cheap for K ≤ 3); measure against the full fit in Float32 before deciding.

### What the user sees

- The same `fit` call and the same `FitResult`: every parameter (S0, fshort, T2short, T2long),
  the SSE, the status, and the iterations (of the reduced problem, so fewer; of the full fit
  for a voxel that fell back).
- Statuses as today; a voxel whose amplitudes cannot be computed (zero or NaN columns) is
  `Failed`.
- Standard errors (roadmap item 3) from the full Jacobian at the solution, in the model's own
  parameters, so they do not depend on how the fit got there.
- `parameters(m)` shows the setting: `fshort = Free(lower = 0.0, upper = 1.0, varpro = true)`.

### Tests and acceptance

- The declarations: the linearity check passes for every built-in `varpro = true` and fails,
  with a clear message, for a wrong one (`T2 = Free(varpro = true)`), for a magnitude model's
  amplitude without its bound ≥ 0, and for magnitude SIR's Sf; signed SIR passes with its
  mapping. Fixed and tied `varpro` parameters; `Free(varpro = false)` and `LM(varpro = false)`
  give the ordinary fit with least-squares starts; a start given with `varpro = true` is used
  only there.
- The hooks: columns from `columns!` equal the columns built from `signal!`; `fromamplitudes`
  round-trips; the base point gives the same columns as b = 0 for plain amplitudes.
- Noise-free recovery from the default starts; full fit and VarPro agree within the tolerance on
  well-conditioned data; the least-squares starts.
- Bounds: amplitudes held at 0 (fshort exactly 0 or 1); a bound the projection cannot hold
  (S0 ≥ 100 on data with S0 = 50; SIR's Sf ≤ 0 on data that want Sf > 0) → those voxels, and
  only those, refitted, their answers on the bound; no bounds → no constraint; the affine part
  (a fixed offset).
- No allocation per voxel; Float32; statuses; `refit`; maps of fixed values on nonlinear
  parameters. Simulation unchanged (`signal`, `simulate` and `nominal` never see `varpro`).
- The benchmark: an "EQUIP VarPro" row in `bench/solvers.jl` until the built-in models declare
  `varpro = true`. The biexponential cases at least as good as the prototype (not best 8 / 4 /
  16 / 11); every other case bit-identical to today (nothing else declared: the fingerprint
  check); speed no worse beyond noise (10 000 voxels, at least 10 runs).

### Order of work

1. The `varpro` keyword on `Free` (and its printing), setup's grouping, the linearity check,
   columns from `signal!` at a base point, the reduced problem with amplitudes ≥ 0, the
   per-voxel fallback for other bounds, the least-squares starts, `LM(varpro = false)`: first
   on the decay with offset (plain amplitudes S0 and C), which exercises the generic route with
   no hook.
2. The biexponential (T2 and R2 forms): `fromamplitudes` and `columns!`; tests; the benchmark
   acceptance above.
3. The scale parameter alone in the other models (`varpro = true` on M0, S0, M0f?): measure
   every case first, then decide.
4. Signed SIR (`magnitude = false`): a benchmark case, then M0f and Sf through
   `fromamplitudes` if it helps.
5. Separately, later: log coordinates for positive nonlinear parameters with the per-parameter
   step test (with VarPro on the biexponential: 40–59% faster, more fits reaching the minimum;
   see the roadmap).

### Open decisions

- A bound the projection cannot hold: refit the voxels where it binds (proposed), fall back
  for the whole fit call (the first design), or an error.
- How a model with `fromamplitudes` says its default bounds mean a ≥ 0: a method next to it
  (e.g. `nonnegative(m) = true`, proposed), or found by sampling at setup. Either way the bound
  check at the answer catches a wrong statement, through the fallback.
- A start given with `varpro = true`: used only in the ordinary fit (proposed), or an error.
- Float32: Float64 accumulation for the K × K solve (proposed), or the full fit in Float32.
- Where the whole-fit switch lives: in `LM` (proposed: it is how the solver treats the
  problem), or a separate solver type through the solver hook (roadmap item 11), e.g.
  `VarPro(LM())`.
- Settled: the name and place of the declaration (`varpro = true` on `Free`, its own setting),
  and what a projected parameter's start is for (nothing, unless it is fitted the ordinary way).

## Simulation, CRLB and protocol design: proposed design

Status: `crlb`, `montecarlo` and sweeps implemented (on `lean` since bb52890, from the
`crlb-proto` branch); fit settings side by side, data from another model, general noise and
the fast pieces (`CRLBCache`, derivatives.jl) on `lean` since 007e196, from `analysis-proto`;
fit settings as keywords (`kwargs-proto`), the `Scenario` and the faster sweeps
(`scenario-proto`) on `lean` since 6c7cc50; `ProtocolDesign` and `optimize_protocol`
(`design-proto`) on `lean` since e269647; the rest proposed. Prototyped in scratch scripts;
every number below was checked against the MATLAB code of Dortch et al., MRM 2018;80:1824
(`signalSIR.m`, `CRLB_SIR.m`, `biasedCRLB_SIR_fixedkmf.m`, `calcVmax*.m`,
`generateManuscriptResults.m`) and against the published values of Li et al., MRM 2010;64:491.

Goal: (1) predict the bias and precision of a fit before scanning (CRLB, biased and unbiased;
Monte Carlo); (2) choose protocols by those predictions, as Dortch 2018 did (Eq. 5); (3) do the
same for a subject or a population, from fitted maps (data-driven design). What is optimized is
the RMS error of the fitted parameters per unit scan time: fixing a parameter (kmf) adds bias and
gains precision, and both count.

The pieces, each using the ones before it:

1. `crlb`: bias, SD and RMSE predicted for a protocol and a scenario (a truth and the fit).
2. `montecarlo`: the same by simulating and fitting; checks `crlb` and covers what it cannot.
3. Diagnostics: sensitivity maps (Eq. 6), leakage of fixed parameters into fitted ones, cond(J).
4. The objective: Eq. 5 by default, with weights, time and aggregation options, over tissues and
   uncertain conditions.
5. The optimizer: global search and a local polish over the protocol (a package extension).
6. Data-driven input: tissues and noise from `FitResult`s.

### `crlb`: one function, biased or not by the fit settings

Status: implemented (src/errors/crlb.jl); merged into `lean` from `crlb-proto`, and its
pieces (src/errors/derivatives.jl), general noise and `CRLBCache` from `analysis-proto`; the
`Scenario` (src/errors/scenario.jl) and plain keywords from `scenario-proto` (6c7cc50).
Checked: Fopt4 = 0.243395 and Fopt5 = 1.602062 (Eq. 5 of the Table 2 schemes, both
covariances), the exact-fit identity, ∇b against central differences of noise-free fits
(1e-5), the classical CRLB when nothing fixed is wrong, and Monte Carlo (below).

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)
c = crlb(m, θ; snr = 200)                                 # kmf held at the truth's 5: unbiased
c = crlb(m, θ; snr = 200, kmf = 12.5)                     # kmf fixed at 12.5, truth 5: biased
c = crlb(m, Scenario(QMTSIR, θ; kmf = 12.5); snr = 200)   # the same, the scenario made once
c = crlb(m, θ; snr = 200, kmf = Free())                   # kmf fitted: unbiased
c = crlb(m, θ; σ = [σ1, σ2, σ3, σ4], cov = :biased)       # σ per observation; Eq. 3
c.bias.PSR, c.sd.PSR, c.rmse.PSR             # absolute, by name (as res.params.PSR)
c.relbias.PSR, c.relsd.PSR, c.relrmse.PSR    # relative to the truth (Eq. 5 sums relrmse²)
c.fit, c.cov                                 # the noise-free fit's values; the covariance matrix
```

- `truth` gives every parameter (as for `simulate`; `nominal(m, tissue)`); keywords named after
  the parameters change the fit, as for `fit(m, y; kmf = 12.5)`; the fit holds the other fixed
  parameters at the truth's values (a `Scenario`, below; until `scenario-proto`, at the model's
  own). The estimate is biased exactly when a value the fit fixes differs from the truth
  (kmf fixed at 12.5, data with kmf = 5). One function where the MATLAB code has two
  (`CRLB_SIR`, `biasedCRLB_SIR_fixedkmf`). No separate "unbiased" option: with nothing changed
  the fit is unbiased, the classical CRLB.
- **The fit's settings as plain keywords** (`kwargs-proto`; changed from one keyword holding a
  named tuple, `fit = (kmf = 5.0,)`): `crlb(m, θ; snr = 200, kmf = 5.0)`, the same for
  `montecarlo` (where `fit`'s options `solver` and `precision` pass through too), `CRLBCache`
  and the one-call pieces (`noisefree`, `bias`, `jacobian`, `hessian`, `informationmatrix`,
  `biasgradient`). Why: one rule everywhere, the same as fitting (θ positional is the truth,
  parameter keywords are the fit), and no one-entry named tuple with its trailing comma. The
  wrapper was there to avoid clashes with `crlb`'s own keywords; handled instead as `setup`
  handles `fit`'s options: a model parameter named like an option of the function (`snr`, `σ`,
  `Σ`, `cov`, `vary`, `fits`, `simulate_with`; `montecarlo` adds `n`, `noise`, `rng`) is an
  error, as is a keyword that is neither (the message lists both), and the old `fit = (...)`
  form gets a message showing the new one. None of the built-in models has such a parameter.
  `vary` takes pairs: `vary = :kmf => 5:2.5:20`, or several, `vary = [:kmf => 5:5:20,
  :PSR => [0.1, 0.2]]` (a named tuple still works). `fits` (several settings side by side)
  stays a named tuple of named tuples, written `free = (; kmf = Free())`. Results unchanged
  (the 688 cases identical).
- **Roles made by a call, passed whole** (`kwargs-proto`): `parameters(m; changes...)` returns
  every parameter's role with the changes applied, by `setup`'s own rules (`_apply_roles`, shared
  with `setup`), as `nominal(m; changes...)` returns every value. Every function that fits or
  predicts a fit takes it as the argument after the data or the truth, with no splat:
  `fit(m, y, roles)`, `init(m, roles)`, `crlb(m, θ, roles; snr)`, `montecarlo(m, θ, roles;
  snr)`, `CRLBCache(m, roles)`, and the one-call pieces (`jacobian(m, θ, roles)`, ...); keywords
  still change it further (a keyword wins over the roles' entry). Why: one set of assumptions
  made once and kept, for the prediction (`crlb`), the check (`montecarlo`) and the real fit,
  without writing named tuples. A model with its own `parameters(m::MyModel)` method still gets
  the changes (a call with keywords goes to the general method, which asks the model's own for
  the roles). Positional rather than a keyword (`fit = roles`): shorter, and the order reads
  as what it is (model, data or truth, what the fit assumes). Arguments swapped (roles where the
  truth goes) get the "is a role, not a value" message, the truth being checked first. Checked:
  every built-in model's starts and fits (both precisions) bit-identical after moving `setup`'s
  step 2 into `_apply_roles` (108 fingerprints); each function gives the keyword form's result
  exactly. For crlb, montecarlo, `CRLBCache` and the pieces, replaced by the scenario (next).
- **The scenario: a truth and the fit judged against it** (`scenario-proto`; replaces the
  roles argument of crlb and montecarlo). `Scenario(M, θ; starts = :truth, changes...)`, from a
  model's type (or a model: its type) and the truth θ, holds `truth`, `roles` (every
  parameter's role in the fit), `starts` and `wrong` (the fixed values that differ from the
  truth's, each `(fit = ..., truth = ...)`), and no acquisition: `crlb(m, s; snr)` and
  `montecarlo(m, s; snr)` judge it under protocol `m`, so one scenario serves every candidate
  protocol. The fit is built from the truth: the model's free parameters estimated (the model's
  bounds), the others held at their TRUE values; then the changes by `fit`'s rules (a value
  fixes, `Free(...)` estimates, a start or bound given wins). The keyword forms,
  `crlb(m, θ; kmf = 12.5)`, `montecarlo(m, θ; starts, changes...)`, `CRLBCache(m, θ; ...)` and
  the one-call pieces, make one inside. Why: passing a full set of roles to crlb meant passing
  starts and bounds that the noise-free fit does not use, and the fixed values came from the
  model, not the tissue; the scenario says what the comparison is (this truth, this fit, these
  assumptions wrong), shows it at the REPL as a table (true value, role, start or value,
  bounds, `← wrong`), and is the same object for the prediction and the check.
  - **The default became unbiased.** `crlb(m, θ)` holds the fixed parameters at θ's values
    (kmf 5 for WM I), not the model's defaults (12.5): a wrong assumption is now asked for
    (`kmf = 12.5`) and named in `s.wrong`. The old default is `crlb(m, θ; defaults(m)...)`.
  - **Starts:** `starts = :truth` (default; every estimated parameter starts at its true value)
    or `:model` (the model's rules, `:scale`, `BestOf`, `:data`, as `fit` starts a real fit);
    a start given as a keyword wins. A parameter fixed in the base and freed by a change starts
    at its true value (`:truth`) or where `fit` would start it (`:model`: the model's value;
    a tie gets fit's "give it a start" error). crlb's noise-free fit always starts at the truth
    (the pseudo-true value nearest it), whatever `starts` says; montecarlo's fits start where
    the scenario says. A start that depends on the protocol (T1IR's BestOf grid from its TIs)
    stays a function, computed for each protocol the scenario meets (shown as "from the
    protocol"). For 4-point SIR the starts make no difference (crlb and montecarlo identical
    from the truth and from the model's rules, SNR 200 Gaussian and SNR 50 Rician).
  - **wrong** is judged at the truth: a tie is followed (R1m = :R1f is 1.5 in WM I, wrong for a
    tissue with R1m = 1.0), and a number equal to the truth's is not wrong. The result displays
    use the same rule (`R1m = :R1f = 1.5 (true 1.0)`; the old display compared numbers only).
  - **Sweeps:** `vary` changes the truth, never the fit: the fixed values stay the scenario's
    (with `kmf = 12.5` and the true kmf swept, Fig. 2B), and in montecarlo the starts taken from
    the truth follow each setting's truth (`s.attruth`, the parameters that start there; a
    start given does not move). `CRLBCache(m, s)` holds the scenario's fit, so `crlb!` with
    another truth judges that same fit against it.
  - **Side by side:** `crlb(m, (fixed = s1, free = s2); snr)` (and montecarlo, a paired
    comparison on the same noise), replacing the `fits` keyword: one more sweep axis, `fit`, as
    before. Kept, rather than only broadcasting over scenarios, for the Sweep's table and
    montecarlo's common noise. The old keywords `fits` and `starts` (in crlb) get messages.
  - `fit(m, data, roles)` and `init(m, roles)` stay (a fit of real data has no truth), and
    `fit(m, data, s.roles)` fits data with a scenario's fit. `parameters(m; changes...)` stays.
  - Considered and dropped: a model wrapper `assume(m; kmf = 12.5)` that is itself an
    `MRIModel` (it mixed the protocol with the assumptions, and crlb still needed the truth
    apart); a scenario holding only free names and fixed values (montecarlo then needed starts
    and bounds from elsewhere); a scenario with the acquisition (the same assumptions under many
    protocols is the protocol search).
  - Checked: the 688 cases with the old default written out (`merge(defaults(m), settings)`)
    identical, bit for bit; every built-in model's starts and fits (both precisions, 108
    fingerprints) bit-identical after splitting `setup` (steps 3–5 into `_spec`, which takes a
    scenario's roles whole) and `resolve` (its checks into `_complete`). Speed (one thread,
    fastest of 200, two alternated runs, 4- to 16-point SIR and T2 with a full Σ): the kernel
    unchanged (1.0–15.6 µs); `crlb(m, s)` 26–42 µs, 11 µs less than the old `crlb(m, θ)`
    (37–53 µs), as the roles go to the spec whole instead of merged one keyword at a time;
    `crlb(m, θ; kmf = 12.5)` 41–57 µs (2–5 µs more than before; T2 5 µs less). The scenario
    itself takes 12–13 µs (QMTSIR, 9 parameters; 2 µs for T2): 25 µs at first, halved by a
    function barrier after the changes (the roles' type is unknown there; the checks and
    `wrong` then compile to straight-line `map`s over the named tuples).
- **Sweeps, profiled** (`scenario-proto`): a crlb sweep cost 37–40 µs a setting (4-point
  SIR; 50 µs 16-point) where `crlb!` alone took 11 µs. The profile: 61% building the table
  (`_rows` resolved the whole truth again for every row, setting × parameter: 42% on its own),
  34% in `crlb!`, of which a third was checking the truth (`_crlb_truth` resolved it only to
  check it, then reordered it again) and a third `_condition` (`c.cond`: LAPACK's symmetric
  eigenvalues). Changed: each setting's rows built in a function of their own (a function
  barrier, the setting's type known there) from the free parameters' true values read once
  (`_freetruth`: directly when numbers, through `resolve` only when the truth ties a free
  parameter); `_crlb_truth` one pass of `resolve`'s checks (`_complete`) instead of `resolve`
  plus a reorder; `_reject_roles` (a quick `map` first) and `_check_ties` (a `map` over names
  and values, one `_check_tie` method per kind of value) without the `for` loop over a named
  tuple of mixed types, which looked each value's type up at run time (0.7 and 0.8 µs for
  QMTSIR, now 0.03 and 0.1). Now 9.2–9.7 µs a setting (4-point) and 19–21 µs (16-point): 4×
  and 2.5× faster; the rest is `crlb!` (6.6 µs for 4-point: the kernel 2.7, `_condition` 2.0,
  the result). The same checks made every call faster: `crlb!` 6.6 / 19–20 µs (was 10.7 /
  23.7), `crlb(m, s)` 21–36 µs, `crlb(m, θ; kmf = 12.5)` 32–49 µs (below the 37–53 before the
  scenario), `Scenario` 9.8 µs; the kernel unchanged. Results unchanged: 8 sweeps (378 rows,
  every result field and the displays; crlb and montecarlo, scenarios side by side,
  `simulate_with`) identical, the 688 cases and the 108 fingerprints bit-identical. Left:
  `_condition` (2 µs a call, a third of a 4-point `crlb!`); a pure-Julia eigenvalue solver for
  the small static matrix would remove most of it, at the price of `cond` changing at rounding
  level, so not done.
- **The result pays only for what is read** (`result-proto`): of a 4-point `crlb!` (5.7 µs),
  the kernel was 2.1 µs, checking the truth 1.4, `cond` 1.45 (LAPACK's symmetric eigenvalues),
  everything else in the `CRLBResult` (named tuples, matrix copies, 7 KB) about 0.3. So the
  struct itself was not the cost; a value computed for every result and read by few was. Fits
  were measured the same way first and need nothing: `fit!`'s `FitResult` costs what the
  per-voxel core costs (0 bytes), `fitted` and `residuals` are computed when called, and a
  dataset's maps, tasks and result add at most 0.1 µs a voxel (≤ 5%; one-off `fit(m, y)` is
  mostly `init`, 7–13 µs). Changed: `c.cond` is computed when read (`getproperty`, from J₀ᵀJ₀
  and the true values the kernel left, kept as small static values), and a sweep whose varied
  values are numbers (data from the fitted model) puts each setting's truth in the kernel's
  form without checking it again (`_crlb_varied!`; the scenario's truth was checked when it was
  made). `crlb!` 6.3–6.6 → 4.7–4.8 µs (4-point, 7.1 → 4.6 KB) and 15.7–16.0 → 13.5–14.4 µs
  (16-point); a sweep setting 8.7–9.1 → 5.9–6.1 µs (4-point) and 18.9–20.8 → 16.2–17.4 µs
  (16-point); `crlb(m, s)` 19–34 µs, `crlb(m, θ; kmf = 12.5)` 30–47 µs (three alternated runs
  against `lean`). Results identical, `cond` included: the 688 cases with either default, the 8
  sweeps, the design objectives.
- **Diagnostics are functions of a result, not fields** (`diag-proto`): `cond(c)` (and
  `cor(c)`, `cor(mc)`, new) are called when wanted, as `fitted(res)` and `residuals(res, y)` are
  for a fit, rather than a property computed on read (`c.cond` through a `getproperty` method,
  the step before). Why: a field reads as stored, so a property that runs an eigenvalue problem
  hides a cost and makes `CRLBResult` a struct with a special case; a function says it computes,
  composes with broadcasting over a sweep (`cond.(s.results)`), and needs nothing kept but what
  it is computed from. The names are LinearAlgebra's `cond` and Statistics' `cor`, with methods
  for the results and re-exported (the same function as theirs, so `using LinearAlgebra,
  Statistics` beside EQUIP clashes with nothing). What they read: `cond`, J₀ᵀJ₀ (`c.gram`, a
  static P×P matrix, the one field added in place of the lazy property's tuple) and the free
  true values, the parameters on a bound and the number of observations, already in the result
  (`_freetruth`, `onbound`, `σ`); `cor(c)`, the covariance, Cᵢⱼ/(SDᵢSDⱼ) with 1 on the diagonal
  and `NaN` beside a parameter held on a bound (SD 0) or undetermined (SD Inf), as Statistics'
  `cor` treats a constant column; `cor(mc)`, Statistics' `cor` of the estimates `vcov(mc)` uses.
  Costs unchanged (three alternated runs against `lean`): `crlb!` 4.62–4.72 µs (4-point; was
  4.73–4.84, 96 bytes less: one static matrix kept instead of a tuple) and 13.6–13.8 µs
  (16-point), a sweep setting 5.9–6.0 µs and 16.0–17.2 µs, `cond(c)` 1.5 µs as `c.cond` was.
  Results identical, `cond` included: the 688 cases with either default, the 8 sweeps (rows
  and displays), the design objectives; 1541 tests (10 new). On the 4-point WM I fit
  with kmf assumed 12.5, the predicted correlations are within 0.023 of those measured by 4000
  Monte Carlo fits (about 1.5 standard errors, (1 − r²)/√n).
- Named `crlb`, the name the qMRI papers and the MATLAB code use; the docstring also says CRB,
  the signal-processing spelling, so a search for either finds it.
- Output, indexed by quantity then parameter (`c.sd.PSR`, like `res.params.PSR`): the
  noise-free fit's values θ̂ (the "pseudo-true" values), bias, SD and RMSE, absolute and
  relative; the covariance matrix; cond(J); which parameters ended on a bound. The REPL shows a
  table with the columns of `compare`, so `crlb`, `montecarlo` and `compare` line up. Several
  tissues: broadcast, `crlb.(Ref(m), truths; snr = 200)`.
- Noise: `snr` relative to the scale parameter (as `simulate`), `σ` (a number, or one SD per
  observation), or `Σ`, the full covariance matrix of the noise between observations
  One SD per observation is needed for models that combine sequences with
  different bandwidths (mcDESPOT: σ_SSFP = √3·σ_SPGR, Lankford & Does, MRM 2013;69:127) or
  numbers of averages; a full Σ for correlated noise (after denoising, a joint
  reconstruction, interpolation across volumes). Chosen by type, so the usual case, Σ = σ²I
  (independent thermal noise with the same variance at every observation: separate
  acquisitions, the same readout), is "pulled out", JᵀΣ⁻¹J = JᵀJ/σ², at no cost. A full Σ is
  factored once (Σ = LLᵀ, checked symmetric and positive definite) and whitens the sums:
  Fisher J_tᵀΣ⁻¹J_t = Σₖ vₖvₖᵀ with vₖ the rows of L⁻¹J_t, sandwich H⁻¹J₀ᵀΣJ₀H⁻¹ = Σₖ aₖaₖᵀ
  with aₖ = H⁻¹(row k of LᵀJ₀); sums of squares, allocation-free, O(N²P) (N ≤ 100). Each row is
  made where it is used, from L (the sandwich) or from L⁻¹, inverted once with the factor
  (Fisher: vₖ = Σᵢ≤ₖ (L⁻¹)ₖᵢJᵢ; first by forward substitution, which kept the rows in a work
  buffer inside the noise, so one noise could not be shared between tasks; changed on
  `noise-proto`: read only now, shared by 8 tasks on 4 threads with the same covariances as
  one task; L⁻¹ stored transposed so the loop reads down columns; Fisher and observed
  information within 5e-16 of before, Eq. 3 SDs within 6e-15, the sandwich identical; the Eq. 3
  kernel 0.1–0.3 µs slower at 32 echoes, the noise's setup a few µs, a one-off crlb unchanged). The pattern is LsqFit.jl's for weighted fits (a Cholesky factor of the
  inverse covariance folded into the residuals). With unequal or correlated noise EQUIP's
  unweighted fit is not efficient: the sandwich is its SD, Eq. 3 (`:biased`) the bound a
  weighted fit would reach. Checked: σ²I, a diagonal Σ and the number or vector give the same
  SDs (1e-12); with AR(1) noise (ρ = 0.5, σ = 10, T2Decay, 32 echoes) the bound is
  (JᵀΣ⁻¹J)⁻¹ and the sandwich (JᵀJ)⁻¹JᵀΣJ(JᵀJ)⁻¹ exactly (S0 SD 11.34 and 12.05), and Monte
  Carlo with that correlated noise (20 000 copies) follows the sandwich, SD/crlb 1.005 ±
  0.005. Refactoring check: the 688 cases' fits and biases identical, SDs to 1e-15; the kernel
  still allocates nothing (full Σ too), 3.5–4.3 µs for the 4-point scheme either way.
- **StatsAPI's names for the results:** `informationmatrix(c; expected = true)` (the Fisher
  information J₀ᵀΣ⁻¹J₀ at θ̂), `informationmatrix(c; expected = false)` (the observed
  information, the Hessian of the negative log-likelihood of the noise-free data at θ̂:
  J₀ᵀΣ⁻¹J₀ + Σᵢ wᵢ∇²gᵢ, w = Σ⁻¹r; H/σ² with one σ), `vcov(c)`, `stderror(c)`, and
  `informationmatrix(m, θ; noise, fit)` at any values θ (below); `vcov(mc)` and `stderror(mc)` for a
  Monte Carlo (the estimates' covariance and SDs). StatsAPI defines these functions without
  methods (as for `fit` and `nobs`), so EQUIP's are its own computations under the names that
  GLM, LsqFit and CurveFit users know; StatsAPI's `expected` keyword is exactly the
  Fisher/observed distinction, and with one σ the sandwich is `inv(I_obs) * I * inv(I_obs)`
  (tested). Why not an existing implementation: a search of the General registry (all 14 627
  packages, Oct 2026) found no package for Fisher information or the CRLB of a model at a
  hypothetical truth; LsqFit's and CurveFit's `vcov`/`stderror` are for a fitted result (the
  unbiased case, after a fit with their own solver), CovarianceMatrices.jl's sandwiches are
  estimated from residuals, Kirstine.jl (optimal design for nonlinear regression),
  CorleoneOED and DynamicOED (dynamic systems) have no bias from wrong fixed values. The
  reusable part, the derivatives, is ForwardDiff's; F = JᵀΣ⁻¹J itself is one line.
- **One implementation, fast: a cache and allocation-free pieces** (src/errors/derivatives.jl;
  the kernel `_crlb!` in crlb.jl). Each step of `crlb` is a function
  of its own, one formula each, and the kernel is their composition:

  | Piece | Formula |
  |---|---|
  | `CRLBCache(m; cov, settings...)` | once per fit setting: free names in the type, bounds, fixed values as written, buffers (what `init` is for fitting) |
  | `noisefree!(c, m, t)` | θ̂: `levmar` from the truth's free values, the fit's fixed values and bounds |
  | `hessian!(c, m, θ̂, exact)` | H = Σᵢ JᵢJᵢᵀ + rᵢ∇²gᵢ, and G = Σᵢ JᵢJᵢᵀ |
  | `meat(rows, σ², u)` | J₀ᵀΣJ₀ |
  | `information(rows, σ²)` | JᵀΣ⁻¹J (Eq. 4) |
  | `crossgram(rows₀, rowsₜ, u)` | J₀ᵀJ_t |
  | `curvature(rows, w)` | Σᵢ wᵢ∇²gᵢ |
  | `sandwich(H, M)`, `biasgradient(H, K)`, `biasedcrlb(∇b, F)` | H⁻¹MH⁻¹; H⁻¹K − I; (I + ∇b)F⁻¹(I + ∇b)ᵀ (Eq. 3) |

  The sums are accumulated one row of J at a time from dual numbers into static P×P matrices,
  as the solver's `normal_equations` sums JᵀJ: one evaluation of the model gives every gᵢ with
  its gradient (with nested duals, ∇²gᵢ too), J is never stored, nothing is allocated. The
  formulas also have methods for ordinary matrices (`sandwich(H, J, Σ)`, `biasedcrlb(∇b, F)`),
  and one-call functions from a model and values (`noisecovariance`, `noisefree`, `bias`,
  `jacobian`, `hessian`, `informationmatrix(m, θ)`, `biasgradient`) make a cache and use the
  same pieces. `crlb!(c, θ; noise)` is the kernel plus checks and the result; `crlb(m, θ; ...)`
  makes the cache first; sweeps make one cache per fit setting. The textbook versions
  (ForwardDiff.jacobian and ForwardDiff.hessian on `signal`, plain matrices, `\`, a tightly
  converged fit) live in the tests, as the reference `crlb` must match (bias to 1e-8 of θ̂, SD
  to 1e-6, both covariances, SIR schemes, T2, bi-exponential T2 and IR, σ per observation and
  correlated noise).
  - **How we got here** (crlb, one-off call, 4-point / 16-point scheme):

    | Version | Time | What it showed |
    |---|---|---|
    | First `crlb` (f79e0f2): ForwardDiff on `signal`, plain matrices | 307–374 µs | correct; allocation and run-time types |
    | Allocation-free kernel (c0a28e1) | kernel 5–7 / 23–28 µs, `crlb` 82–138 µs | the maths is fast; the rest was setup and packaging |
    | General Σ, information matrices (fbab4f9) | 57–74 / 95–295 µs | the same split |
    | Readable pieces, plain matrices (c35d796) | 324 / 348 µs | readable, but every piece re-ran `setup` (24–43 µs) and closures typed at run time |
    | This: cache + allocation-free pieces | 37–55 µs one-off; `crlb!` 10–24 µs; kernel 1.0–15.6 µs | readable by naming the steps, fast by summing static matrices once the types are known |

    Readability came from naming the steps, not from forming the matrices; speed from three
    things: setup once with the types known, static P×P sums over the observations, nothing
    allocated per call.
  - **Where the time went** (one call, 4-point / 16-point, before this change): `setup` 24–43 µs
    (once per fit setting now); the noise-free fit 2.2 µs (6 iterations) / 6–14 µs (7–16);
    J₀ and ∇²g with nested duals 2.0 / 6.0 µs, J alone 0.5 / 1.1 µs; cond(J) by SVD 4–5 µs;
    `resolve` (which checks every value) 4.4 µs, run four times per call (now once).
  - **The noise-free fit: `levmar`, xtol 1e-10** (was 1e-13). Over the 12 cases of the 4-, 5-
    and 16-point schemes × 4 tissues (kmf fixed at 12.5): xtol 1e-10 takes 5–19 iterations
    (1.8–8.1 µs) instead of 6–21 (2.1–14.3 µs), and moves θ̂ by at most 3.1e-9 (relative, over
    all 688 cases); the minimum is flat enough that the damping alone moves θ̂ by 1.6e-9 at
    xtol 1e-13, so the extra digits were not real. The fit's own defaults (xtol 1e-8) are faster
    still (4–7 iterations) but leave θ̂ to 1.2e-7, a visible fraction of a tiny bias (R1f in WM
    I: 5e-6). Plain Gauss–Newton from the truth was no faster per iteration (the same
    evaluation) and needs a case-dependent number of steps: 3 for the 4-point scheme (4e-12),
    about 10 for 5-point WM I (2.4e-4 after 3 steps); it has no bounds. Lower starting damping
    (1e-8): no consistent gain. The fit uses the model's roles (`parameters(m)` merged with
    `fit`, the same `FitSpec` as `fit`): the fixed values as written, the bounds; only the start
    is the truth's, written into nothing (the free values go to `levmar` directly). The model's
    own starts reach the same θ̂ in all 12 cases (to 1.5e-9), but the truth is the right
    definition (the minimum nearest the truth is where noisy fits around it end up) and skips
    the start search.
  - **Second derivatives only when needed.** When the noise-free fit is exact (‖r‖ ≤ 1e-12‖y‖:
    nothing fixed wrong, or as many observations as free parameters, e.g. 4-point) the
    curvature term vanishes, H = JᵀJ, and first-order duals suffice: 304 of 695 test cases; SDs
    change by ≤ 1.1e-11.
  - **cond(J)** from the scaled P×P Gram, √(λmax/λmin), instead of an SVD of J (no J to
    decompose); agrees to 2.7e-10 (2.1e-12 at the old tolerance).
  - **Checks (695 cases: the 688 below, 4 with a full Σ, 3 with data from another model),
    against the previous `crlb`:** the prototype with the old settings reproduced it (fits and
    biases identical, SDs to 2.1e-12); with the new settings, θ̂ and b to 3.1e-9, SDs to
    1.3e-9, information matrices to 1.6e-9, the same undetermined and on-bound parameters. The
    built code equals the prototype (SDs to 1.1e-14). Kernel: 0 bytes in every case.
  - **Speed** (one thread, minimum; kernel / `crlb!` with a cache / one-off `crlb`): 4-point WM I
    2.5–2.9 / 10–11 / 37–39 µs; 5-point WM I kmf fixed 10.0–10.5 / 18–19 / 49–55 µs, kmf free
    1.7–2.8 / 10–20 / 39–42 µs; 16-point WM I 15.6 / 24 / 51–54 µs, GM 11.9–12.8 / 21 / 49–50 µs;
    T2Decay, 32 echoes, full Σ 1.1–2.1 / 34–50 / 53–54 µs (Σ checked and factored per call). A
    sweep of 21 settings (16-point, 3 SNR × 7 true kmf): 1.0 ms, 11 ms before. The previous
    kernel: 4.4–4.7, 10.9–11.6, 4.3–5.1, 21.6–22.6, 12.9–14.5, 1.4–1.8 µs.
  - Conventions kept from the readable version: θ is values, `fit` the fit settings; what runs a
    fit uses all of the fit's settings; the derivatives are taken at θ with the settings
    choosing the columns (`jacobian(m, θ)` is J_t; the fit's J₀ is `jacobian(noisefree(m, θ))`,
    from the
    `FitResult` methods `jacobian(r)`, `hessian(r, y)`, `bias(r, θ)`). `informationmatrix(m, θ)`
    is at θ (Eq. 3's F at the truth). `biasgradient(m, θ; method = :unrolled)` (ForwardDiff
    through Gauss–Newton steps; 2e-11 from the implicit one, 4-point) stays as a cross-check and
    the pattern for derivatives with respect to the protocol. `jacobian` and `hessian` are also
    exported by Zygote and DifferentiationInterface: with either loaded, `EQUIP.jacobian`.
- Covariance, `cov = :sandwich` (default) or `cov = :biased` (Dortch 2018 Eq. 3); see below.
  The sandwich is the default because it is what EQUIP's least-squares fit achieves, so an
  optimizer cannot credit a protocol with precision the fit will not deliver; Eq. 3 stays for
  comparison with the paper. Reproducing the paper does not depend on the choice: its
  optimized 5-point scheme fits kmf (no bias) and its 4-point scheme fits exactly, where the
  two are equal.

The methods, and where each lives:

| Method | Gives | Valid when | Where |
|---|---|---|---|
| Unbiased CRLB, σ²(JᵀJ)⁻¹ | SD | no fixed value is wrong | `crlb`, automatically |
| Sandwich, σ²H⁻¹J₀ᵀJ₀H⁻¹ | b; the SD least squares achieves | any assumption error; first order in σ | `crlb` (default) |
| Eq. 3, (I+∇b)F⁻¹(I+∇b)ᵀ | b; a bound for any estimator with that ∇b | the same | `crlb(...; cov = :biased)` |
| Monte Carlo | everything: Rician floor, bounds, nonlinearity | always (sampling error) | `montecarlo` |
| Linear leakage, −(JᵀJ)⁻¹Jᵀ∂g/∂c | first-order bias per unit error | small assumption errors | diagnostics, not `crlb` |

The steps:

```
 truth θ ──────▶ g_true = signal(m, θ)                 the noise-free data
 fit settings ─▶ free and fixed parameters, bounds     as fit() uses them
 ① noise-free fit  θ̂ = fit of g_true, LM from the truth;  b = θ̂ − θ;  r₀ = g(θ̂) − g_true
 ② derivatives     J₀ = ∂g/∂θ at θ̂ (ForwardDiff, ties followed);  H = J₀ᵀJ₀ + Σᵢ r₀ᵢ∇²gᵢ
 ③ covariance      :sandwich  H⁻¹J₀ᵀΣJ₀H⁻¹
                   :biased    (I+∇b)F⁻¹(I+∇b)ᵀ,  I+∇b = H⁻¹J₀ᵀJ_t,  F = J_tᵀΣ⁻¹J_t
 ④ report          SD = √diag, RMSE = √(b² + SD²), absolute and relative to the truth
```

Edge cases:

- A parameter that ends on a bound in the noise-free fit is held there: SD 0, its row of ∇b 0,
  the others computed without it; the result says which.
- Singular or ill-conditioned: SD Inf (a failed Cholesky factorization), and cond(J) reported,
  rather than the MATLAB's `abs()` of a negative variance.
- σ per observation: EQUIP's fits are unweighted, so the sandwich is the SD of the unweighted
  fit (H⁻¹J₀ᵀΣJ₀H⁻¹) and Eq. 3 the bound (F = J_tᵀΣ⁻¹J_t), which a weighted fit would reach;
  equal when every σ is the same. Weighted fits: later, if needed.
- Far assumption errors: the noise-free fit can stop in a local minimum. A multi-start check:
  later.
- Identifiability is tested before inverting: fewer observations than free parameters, or a
  Cholesky pivot of J₀ᵀJ₀ with unit diagonal below 1e-12 (the squared sine of the angle between
  a column of J₀ and the span of the others: an angle below 1e-6 rad, cond(J₀) ≳ 1e6), gives
  Inf. Found in the first implementation: with 4 observations and 5 free parameters (kmf freed
  on the 4-point scheme), rounding made the singular H look positive definite, and the sandwich
  then gave finite SDs (a pseudo-inverse in effect), not Inf. (The first version tested
  cond(J₀) ≥ 1e8, columns scaled by the true values; the pivot test needs no SVD, so the kernel
  can run it without allocating. Same Inf cases over the 688 cases below; the largest cond
  among the determined ones is 2.5e3.) `cond(c)` still reports the value-scaled cond(J₀).
- **An allocation-free kernel for the optimizer** (`_crlb_plan`, `_crlb_truth`, `_crlb!`; the
  plan is now `CRLBCache` and the kernel a composition of named pieces, see "One
  implementation, fast" above):
  the plan, made once, holds the roles, bounds, solver settings and buffers; the kernel takes a
  protocol (a model with as many observations; static vectors work), a prepared truth and σ²,
  and returns θ̂, b and the covariance as static arrays. The noise-free fit is the package's own
  `levmar` (allocation-free; a new `VoxelProblem` per call carries the protocol), J₀ and H come
  from one evaluation of the model with nested dual numbers (gᵢ, ∇gᵢ and ∇²gᵢ together), and
  the covariances from small Cholesky factorizations. `crlb` is plan + kernel + packaging, so
  its tests check the kernel; the tests also check zero allocation, and one plan reused for
  another protocol. Refactoring check, 688 cases (4 SIR schemes × 7 fit settings × 4 tissues ×
  3 truths × both covariances, plus T2, bi-exponential T2, MFA and IR with σ a number and per
  observation): noise-free fits and biases bit-identical, SDs to 1.6e-12, the same Inf and
  on-bound cases. Speed (one thread): the kernel 5.1–6.9 µs (4-point) and 23–28 µs (16-point;
  the noise-free fit needs more iterations when the residual is not 0), 0 bytes; `crlb` itself
  82–138 µs and 43–47 KiB, down from 307–374 µs and 133–158 KiB (the rest is setup and
  packaging). Not yet: dual numbers through the acquisition (the buffers are Float64); for
  gradients with respect to TI and TD, the implicit-function step again (∂θ̂/∂TI = −H⁻¹∂G/∂TI).
- Not in the first version: `crlb(res; σ)` from a `FitResult` (maps of predicted SD, for
  data-driven design), `vary = :kmf => 5:2.5:20` sweeps, derivatives with respect to TI and TD
  (for a gradient-based optimizer), a Rician Fisher information.

How, with what was measured:

- **Derivatives by ForwardDiff**, exact, no step size. J = ∂g/∂θ at the truth over the fitted
  parameters; ties are differentiated through (R1m = R1f moves with R1f, as the MATLAB's
  `R1 + eps` on both). Checked: `signalSIR.m` equals QMTSIR (signed) to 8e-15·M0f over 1000
  random voxels; `CRLB_SIR.m` (central differences, step 1e-3 of each value) and ForwardDiff
  agree to ≤ 1e-6 relative, the differences' own error.
- **The bias b from the noise-free fit, by Gauss–Newton from the truth.** Two or three steps
  reach the full fit (4-point scheme, kmf 12.5, Table 1 tissues: Eq. 5 off by 5e-5 after 2
  steps, 3e-11 after 3). The first-order formula b ≈ −(JᵀJ)⁻¹Jᵀ(∂g/∂kmf)·δkmf, no iteration, is
  not enough: 13× off for WM I (kmf 5 against 12.5). Optional multi-start check for assumptions
  far from the truth, where the nearest minimum may not be the global one (Lankford & Does refit
  each noise-free signal from several starts).
- **∇b without fits**, by the implicit function theorem at θ̂: the fit's gradient
  G(θ̂, θ) = J₀ᵀ[g(θ̂; assumed) − g(θ; true)] is 0, so ∂θ̂/∂θ = −H⁻¹·∂G/∂θ, with H the full
  Hessian of ½‖r‖² (the residual is not 0 when an assumption is wrong). A parameter on a bound:
  its row is 0. Checked against `biasedCRLB_SIR_fixedkmf.m` (9 fits per tissue, central
  differences): bias to 3e-15, SD to 4e-7, ∇b to 1e-9. Where derivatives with respect to the
  protocol are needed, ForwardDiff through a fixed number of Gauss–Newton steps instead (3 steps:
  exact to 1e-10 here).
- **Eq. 3 and the sandwich.** With J₀ the fit Jacobian at θ̂ and J_t the true one, Eq. 3's
  covariance (I + ∇b)F⁻¹(I + ∇b)ᵀ = σ²H⁻¹J₀ᵀP_tJ₀H⁻¹, P_t = J_t(J_tᵀJ_t)⁻¹J_tᵀ; the first-order
  covariance of the least-squares fit itself (the "sandwich" of the misspecified-CRB literature)
  is σ²H⁻¹J₀ᵀJ₀H⁻¹. They are equal when the fit is exact (as many points as fitted parameters:
  the 4-point scheme, to every digit), and then simply σ²(J₀ᵀJ₀)⁻¹ at θ̂: no ∇b at all. With more
  points Eq. 3 ≤ sandwich (16-point, kmf fixed: WM I PSR SD 2.72% vs 2.88%): Eq. 3 bounds any
  estimator with that bias gradient, the sandwich is what least squares achieves. Both offered;
  the sandwich the default (above). Monte Carlo confirms it (16-point, WM I, kmf
  fixed at 12.5, SNR 200, Gaussian noise, 20 000 fits, all converged): SD of PSR 0.00580 ±
  0.00003, sandwich 0.00576, Eq. 3 0.00544 (12 standard errors away); R1f 0.01441 ± 0.00007,
  0.01439, 0.01392; and the fits center on θ̂ (PSR 0.11361 vs 0.11367 ± 0.00004), not on the
  truth (0.20).
- **A singular F gives Inf.** The MATLAB takes `abs(V)` "for cases where sigma is imaginary": a
  nearly singular F can give negative diagonal elements of F⁻¹ by rounding, which `abs` hides. A
  failed Cholesky factorization gives Inf instead, so an optimizer avoids protocols that cannot
  determine the parameters.
- The noise-free fit uses the fit's own bounds (`fit` keywords). The MATLAB's biased fits used
  Sf ≥ −1.05, R1f ≤ 3, M0f ≤ 10: those for reproducing the paper, the user's in the package.
- Speed: one Eq. 5 evaluation (4 tissues, 4-point scheme, Gauss–Newton then σ²(J₀ᵀJ₀)⁻¹, static
  arrays) takes 17.6 µs; the MATLAB needs an estimated 0.4 s for the same (36 `lsqnonlin`
  fits). That difference is why the paper's genetic-algorithm runs took hours to days.

### `montecarlo`: sweeps, and simulating with another model

Status: one case and sweeps implemented, merged into `lean` (src/errors/montecarlo.jl,
sweep.jl); fit settings side by side and `simulate_with` too (from `analysis-proto`); a
`Scenario`, its starts, and scenarios side by side (from `scenario-proto`).

```julia
mc = montecarlo(m, nominal(m, :SIR_WM_I); snr = 200, kmf = 12.5)   # n = 10 000 noisy copies
mc.bias.PSR, mc.sd.PSR, mc.se_sd.PSR, mc.sd.PSR / mc.crlb.sd.PSR
montecarlo(m, θ; snr = 200, kmf = Free(), noise = :gaussian, rng = MersenneTwister(1))
```

What the first version does, and why:

- `simulate` (n copies of one voxel) → one threaded dataset `fit` → statistics, with `crlb` at
  the same truth, noise and fit settings attached (`mc.crlb`) and shown alongside. The fits start
  as `fit` starts them (`:scale`, `:data`, `BestOf`), not at the truth: local minima and the
  starting rules are part of what is measured. (Since `scenario-proto` the scenario's starts:
  the truth by default, `starts = :model` for these.)
- The same names as `crlb`, by quantity then parameter (`mc.sd.PSR`, `mc.relbias.PSR`), so a
  measurement and its prediction line up; plus standard errors (`se_mean` = SD/√n, `se_sd` =
  SD/√(2(n − 1))), so "agrees with crlb" has a scale. Statistics over every fitted copy,
  converged or not, as `compare`; `mc.converged` gives the fraction; `mc.result` (the
  `FitResult`) keeps every estimate, so no `keep` keyword.
- The fit's settings as keywords, as in `crlb`, but options pass through too (`precision`, `solver`: they change
  the fit being measured); crlb gets the roles alone. `noise` and `rng` as `simulate`: `noise`
  always given (`:rician` for magnitude images, `:gaussian` for real-valued data);
  `MersenneTwister(seed)` for repeatable data. The noise as
  `crlb`: `snr`, `σ` a number or one per observation, or a full `Σ`;
  `simulate` draws correlated noise as L·z (Σ = LLᵀ), for Rician noise in each channel; one σ
  given as a vector gives the same data, draw for draw (tested). `mc.σ` is now one SD per
  observation (it was a number).
- Measured (4 threads): 10 000 copies of the 4-point scheme in 0.23 s (7.5 ms since
  `simulate` allocates only the data, 2026-10). Against crlb, WM I with
  kmf fixed at 12.5, Rician noise, SNR 200, 10 000 copies: SD/crlb 1.004, 0.994, 0.996, 0.997
  (PSR, R1f, Sf, M0f; ± 0.007), PSR bias −14.2% measured, −14.1% predicted. T2Decay, Gaussian:
  0.995 and 0.990 (± 0.007). One caution, seen in testing: within one sample the parameters'
  ratios move together (the same noise drives every estimate), so four ratios 3 standard errors
  low in one 2 000-copy sample (seed 1, kmf at its true value) were one fluctuation, not four:
  other seeds 0.98–1.03, and 20 000 copies 0.993–1.006, in Float32 and Float64 alike.

Sweeps (`crlb` and `montecarlo` alike):

```julia
s = crlb(m, nominal(m, :SIR_WM_II); snr = 200, vary = :kmf => 5:2.5:20)    # Fig. 2B
s = montecarlo(m, θ; snr = [100, 200], vary = [:kmf => [5.0, 20.0], :Sm => [0.8, 0.83]])
s[1], s.settings[1], s.rows                  # a result, its setting, the table
```

- `snr = [...]` and `vary = :name => values` (or several pairs); every combination, the first keyword varying
  slowest. `vary` changes the truth, never the fit (the assumption-bias study); varying a
  fitted parameter gives the accuracy over a tissue range. σ is not a sweep axis: in `crlb` a
  vector σ already means one value per observation.
- The result is a `Sweep{R}`: `settings`, `results` (`CRLBResult`s or `MonteCarloResult`s;
  indexing and iteration go to them) and `rows`, one named tuple per setting and parameter
  (montecarlo: measured, then `crlb_...`), for `DataFrame(s.rows)`. The REPL shows the
  setting's values on its first row only.
- Noise: one seed drawn from `rng` per call, and every setting's noise from that seed
  (`Xoshiro(seed)`): common random numbers. Differences between settings are then not noise,
  adding a setting does not change the others' results (tested), and a setting equals the
  one-case run with the same `rng` (tested; the one-case path draws its seed the same way). The
  price: all settings share one sampling error. Seen in testing (4-point, WM II, kmf swept
  5–20, 5 000 copies, Rician): the PSR bias was 0.25 points below crlb's in every setting
  (−0.46% vs −0.22% at kmf 10), and SD/crlb 0.985 in every setting. With 20 000 copies and three
  seeds: Gaussian −0.28, −0.27, −0.16% (± 0.05) against −0.22% predicted; Rician −0.34, −0.34,
  −0.21%. So mostly the shared sampling error, with at most ~0.06 points from Rician noise here.

Scenarios side by side and data from another model (implemented), for `crlb` and
`montecarlo` alike:

```julia
fs = (fixed = Scenario(m, θ; kmf = 12.5), free = Scenario(m, θ; kmf = Free()))
crlb(m, fs; snr = 200)                                     # a Sweep, axis `fit`
montecarlo(m, fs; snr = 200)
crlb(T2Decay(TE = te), (S0 = 1.0, T2 = 0.08); snr = 100,
     simulate_with = T2BiexpDecay(TE = te) => (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08))
```

- Scenarios by name, `(name = scenario, ...)` (until `scenario-proto`, `fits = (name =
  settings, ...)`): one more sweep axis, `fit`, the fastest, so a table shows the fits of one
  case next to each other. In montecarlo the same noisy copies are
  fitted each way (the common seed), a paired comparison; each setting equals the run with that
  `fit` and the same generator (tested). The display gives each fit's fixed values on a line of
  its own. The crlb table's column for the noise-free fit's value is `noisefree` (it was `fit`,
  which collided with the setting's `fit` column: caught by a test).
- `simulate_with = m2 => θ2`: the data come from another model (as many observations), the
  estimates are judged against `truth`, the reference values for the fitted model's parameters
  (e.g. the long T2 of a bi-exponential tissue fitted mono-exponentially; the user's choice:
  there is no unique "true" T2 then). `snr` is relative to `m2`'s scale parameter; `vary`
  changes θ2, the data's truth; montecarlo's noise defaults to `m2`'s (Rician for magnitude
  data). crlb: the noise-free fit of `m` to `m2`'s signal (started at the reference values)
  gives the bias, the sandwich the SD (it never needed the true model's form); Eq. 3 is refused,
  as it needs the true model's derivatives with respect to `m`'s parameters. The kernel takes
  the data model as two more arguments and still allocates nothing (tested).

What montecarlo showed against crlb with these (10 000 copies each):

- **The CRLB is first order, and SNR 200 is not always high enough.** 5-point scheme
  (Dortch 2018 Table 2), WM I, kmf fitted: PSR SD 1.25× crlb's with Gaussian noise (± 0.009),
  1.29× with Rician, and a PSR bias of +3.7% (+3.5%), where crlb predicts none; kmf SD 1.03×.
  At SNR 1000 both agree (SD/crlb 0.99–1.00, bias 0.1%). So the nonlinearity of a 5-parameter
  fit, not the noise distribution; WM I (the slowest exchange, kmf 5) is the hard case. For
  protocol design: Eq. 5 from the CRLB is optimistic for fits like this one at the paper's SNR;
  a Monte Carlo check of a chosen protocol (or a design objective from Monte Carlo near the
  optimum) is worth adding. With kmf fixed (4-point, or 5-point fitted with kmf fixed), the
  agreement was within standard errors (above).
- **The Rician floor.** A mono-exponential fit to bi-exponential data (fshort 0.2, T2 20 and
  80 ms, 32 echoes to 320 ms, SNR 100): mean T2 0.07295 with Rician noise, 0.07268 with
  Gaussian, predicted 0.07267 (± 0.00001): the late echoes (about 1.5σ) are raised by the
  magnitude floor, lengthening T2 by 0.4%, which crlb (Gaussian) cannot show.

The plan for the rest:

```julia
montecarlo(m, t; snr = 200, n = 10_000)                         # one case
montecarlo(m, t; snr = [50, 100, 200, 400])                     # accuracy against SNR
montecarlo(m, t; kmf = 12.5, vary = :kmf => 5:2.5:20)          # the fit keeps kmf = 12.5
montecarlo(m, (fixed = s1, free = s2))                          # scenarios side by side
montecarlo(m, t; simulate_with = m2)                            # data from another model
```

- Glue: `simulate` → `fit` → `compare` for each setting; each setting's n copies are one threaded
  dataset fit; one seeded noise stream per setting, so adding a setting does not change the
  others' data.
- `vary` changes the truth, not the fit: varying a fixed parameter is the assumption-bias study
  (Dortch 2018 Fig. 2B; the Mzf_td0 study in its scripts). `crlb` takes the same `vary`. This
  replaces the planned `sensitivity(...)` (qMRLab's SimVary, roadmap item 3).
- `simulate_with`: data from one model, fitted with another (the 2-component, myelin and
  non-myelin, SIR simulations in the paper's scripts). Designed in from the start.
- Output: one table, the setting's columns then `compare`'s, plus SD/CRLB (from `crlb` at the
  same truth); `keep = true` keeps the estimates (histograms, Fig. 1).
- Why both tools: the CRLB bounds any estimator with the given bias gradient under Gaussian
  noise; Monte Carlo shows the actual estimator: Rician noise (the Gaussian CRLB has no noise
  floor), bounds, local minima, nonlinearity at low SNR. Lankford & Does found CRLB and Monte
  Carlo within 8%, but needed very high SNR for the 7-parameter mcDESPOT fit; Dortch 2018 Fig. 1:
  Rician Monte Carlo matches the CRLB at SNR 200.

### Diagnostics: why a protocol works

- **Sensitivity efficiency** (Dortch 2018 Eq. 6, Fig. 2A): S_eff = (∂S/∂θ)/√t, t the
  observation's time (TI + TD + t_FSE for SIR), for each parameter at each observation.
  **Not a function: it is `jacobian` divided by √t** (`jacobian(m, θ; kmf = Free()) ./
  sqrt.(t)`, kmf freed to get its column; ×θⱼ/M0f for the paper's maps), shown in `jacobian`'s
  docstring with Fig. 2A's grid made as one protocol (`Iterators.product`). Two versions were
  built on `sens-proto` and removed: `sensitivity` (a grid of every combination of the vectors
  given, `TI = ti, TD = td`, with 85%-of-maximum regions) and then `sensitivityefficiency` (at
  any protocol's observations, no grid or regions). Why removed: the quantity is a Jacobian
  divided by √t (equal to 1.7e-16, and `jacobian` is 5× faster: 0.036 vs 0.17 s for 100,000
  observations); a TI × TD grid assumes two acquisition variables to choose; the 85% is the
  figure's contour choice (the paper's text says "85th percentile", its script contours
  0.85 × the maximum), not a property of the protocol. What has value is the figure: a plotting
  function, on the roadmap (item 3, "Sensitivity-efficiency plots").
  - **Measured on the way** (the plotting function's acceptance tests): the paper's script
    (generateManuscriptResults.m, Fig 02) uses PSR 0.12, R1f = R1m = 1 s⁻¹ (moved together:
    the tie R1m = :R1f), Sf −1, Sm 0.83, kmf 12.5 s⁻¹, M0f 1, t_FSE 0.3 s, the magnitude
    signal, log grids of 1000 TIs and 100 TDs from 10 ms to 5 s (not Table 1's white matter,
    as the caption says). Its central differences (ε = 1e-6) agree within 1e-7 of each map's
    largest value. The paper's reading of Fig. 2A holds at the schemes' own points against
    that grid's best: the 5-point scheme reaches 0.87 (PSR), 0.98 (R1f), 0.90 (Sf), 0.92
    (M0f) and 0.99 (kmf); the 4-point scheme 0.88, 0.93, 0.86, 0.95 and at most 0.31 for
    kmf. Largest values (θ·∂S/∂θ, % of M0f per √s): PSR 7.65 at TI 166 ms, TD 1.62 s; R1f
    34.5 at 670 ms, 10 ms; Sf 56.4 at 10 ms, 1.52 s; M0f 57.6 at 1.56 s, 10 ms; kmf 3.83 at
    65 ms, 1.62 s. Analytic: T2·∂S/∂T2 = S0·(TE/T2)·e^(−TE/T2), largest S0/e at TE = T2.
- **Leakage**: the first-order bias of each fitted parameter per unit error in each fixed one
  (kmf, Sm, Sf, Mzf_td0, B1, a tie). The slope of Fig. 2B; covers roadmap item 3's
  "sensitivity to model assumptions". Status: implemented (`leakage-proto`, errors/
  diagnostics.jl), as planned below except where "As built" says otherwise.

  ```julia
  s = Scenario(QMTSIR, nominal(QMTSIR, :SIR_WM_I))      # or with kmf = 12.5: at a biased point
  l = leakage(QMTSIR(; SIR_5POINT...), s)
  l.absolute.kmf.PSR     # ∂PSR̂/∂kmf, in PSR's units per s⁻¹ of error in the assumed kmf
  l.relative.kmf.PSR     # the same relative to the true PSR: −0.119 (−11.9% per s⁻¹)
  l.elasticity.kmf.PSR   # % of PSR per % of kmf: comparable across fixed parameters
  ```

  - **What:** for each fixed parameter c of the scenario's fit (fixed numbers in the first
    version), dθ̂/dc, how the noise-free fit's answer moves when the value it assumes for c
    changes, at that fit (the truth when nothing fixed is wrong). Times an error Δc, it is the
    first-order bias: b ≈ (dθ̂/dc)·Δc (crlb's noise-free fit gives the exact one), for every
    fixed parameter at once and without a fit per assumed value.
  - **How:** the implicit function theorem, as for ∇b. The fit's gradient G(θ̂; c) = J₀ᵀr is 0
    for every assumed c, so dθ̂/dc = −H_θθ⁻¹H_θc, both blocks of the cost's Hessian over (θ, c)
    at (θ̂, c): H_θc = J₀ᵀ∂g/∂c + Σᵢ rᵢ∂²gᵢ/∂θ∂c. At the truth (r = 0) the second term is 0 and
    this is −(JᵀJ)⁻¹Jᵀ∂g/∂c. One evaluation with nested dual numbers over the P free and K fixed
    parameters (the fixed ones seeded too, as `hessian!` seeds the free ones), in the kernel's
    style: a cache made once per fit, allocation-free, for use in the protocol search.
  - **Why the full form** (measured, scratch; PSR, % of the true PSR per s⁻¹ of kmf error):

    | | 4-point | 5-point | 16-point |
    |---|---|---|---|
    | at the truth, WM I: −(JᵀJ)⁻¹Jᵀ∂g/∂c | −7.85 | −11.90 | −15.42 |
    | at the truth, WM II | −0.62 | −2.30 | −2.98 |
    | WM I, kmf assumed 12.5: the Gauss–Newton form alone | | −2.58 | −1.55 |
    | the same with the curvature term (full form) | | −3.25 | −2.09 |
    | central differences of noise-free fits (both cases) | equal to the line above | | |

    At the truth the simple form is exact (and equals the numbers above, from the earlier
    prototype). Away from it the curvature term is 21–26% of the answer: needed.
  - **Output:** by fixed parameter, then free parameter (`l.kmf.PSR`), as crlb's results; the
    absolute value, relative to the true value (per unit of c, the numbers above) and as an
    elasticity, (c/θⱼ)·dθ̂ⱼ/dc (% per %), to compare fixed parameters with different units
    (kmf in s⁻¹, Sm and B1 without). At the REPL a table: one row per fixed parameter, one
    column per free one.
  - **Uses:** which fixed parameters matter for a protocol and tissue, and by how much; the
    first-order bias of an error without a scenario per value; a robust design term later
    (leakage × the plausible error of each fixed parameter, a cheap alternative to a grid of
    tissues × conditions in `ProtocolDesign`).
  - **Accepted when:** the numbers above (at the truth, the three schemes, WM I and WM II, and
    the biased points); central differences of noise-free fits within 1e-6 relative, at the
    truth and at a biased point; the simple form at the truth; the first-order bias against
    crlb's for a small error (kmf 5 → 5.25: within the second-order term); the kernel form
    allocates nothing.
  - **Where:** src/errors/diagnostics.jl, beside `cond` and `cor` (it uses derivatives.jl's
    cache, `hessian!` with the fixed parameters seeded).
  - **Open:** ties (R1m = :R1f: leakage of a deviation from the tie, R1m = R1f + δ; later);
    maps of fixed values (B1 maps: per voxel, later).
  - **As built:**
    - *Both blocks from one evaluation, by a second cache:* `LeakageCache` holds crlb's cache
      (the noise-free fit θ̂) and a "joint" one, made from the same roles with the fixed numbers
      made free. Nothing is fitted with it: `hessian!` at (θ̂, the assumed values), with the same
      data, seeds every one of them, so H over (θ, c) comes out whole and H_θθ, H_θc are its
      blocks. No new derivative code; the nested duals cover P + K parameters (QMTSIR: 4 + 4).
    - *Fields, not `l.kmf`:* `l.absolute.kmf.PSR` rather than the plan's `l.kmf.PSR`, which would
      need a `getproperty` method; the result is a plain struct, as `CRLBResult` became.
    - *Elasticity* (|c|/|θⱼ|)·dθ̂ⱼ/dc, magnitudes on both sides, as `relbias` divides by |truth|:
      a fixed value of 0 (Mzf_td0, Mzm_td0) has elasticity 0, its relative slope still says how
      much an error leaks. Shown as a ratio (0.4 = 0.4% per 1%), the relative slopes in %.
    - *Reported:* the fixed parameters held at numbers; ties follow the parameter they name (an
      untied `R1m = 1.0` is reported). A fit with no fixed number is an error. A free parameter
      held on a bound has slope 0 (its row masked, as in crlb); undetermined parameters NaN.
    - *Exported:* `leakage` and `LeakageResult`; the cache and kernel (`_LeakageCache`,
      `_leakage!`) internal until the protocol search uses them.
  - **Measured:** the table's numbers, all eight (−7.85, −11.90, −15.42; −0.62, −2.30, −2.98;
    −3.25, −2.09). The simple form at the truth within 1e-15 (4-, 5-, 16-point). Central
    differences: of crlb's noise-free fits, 1e-5 to 1e-2 off, but erratic in h (the fits stop
    at the solver's 1e-10, and the difference divides that by h); of fits converged to rounding
    (Newton on the exact Hessian), 9e-9 to 6e-7 per slope (5- and 16-point, at the truth and at
    kmf 12.5), the error falling as h² (Mzf_td0: 1.8e-6, 1.8e-8, 1.6e-10 at h = 1e-3, 1e-4,
    1e-5); 4-point's worst, 1e-5, is an R1f slope of 1e-6 (1e-8 of its column's largest). The
    first-order bias against crlb's for kmf 5 → 5.25: 4% off, a quarter of that at 5.125 (the
    second-order term). The kernel allocates nothing: 1.1 µs at the truth (exact fit, first-order
    duals), 16 µs (5-point) and 32 µs (16-point) at kmf 12.5 (nested duals over 8 parameters);
    `leakage(m, s)` 66–71 µs, most of it the two caches. 30 tests.
- **cond(J)**, an identifiability flag (Lankford & Does).
- What they showed for SIR, behind the design rule "sample where ∂S/∂kmf is small and does not
  overlap the fitted parameters' sensitivities": kmf enters the signal as a rate, PSR as an
  amplitude. At TD = 2.5 s the kmf sensitivity peaks at TI ≈ 1/R1⁺ (R1⁺ ≈ kmf(1 + PSR) + R1f:
  47 ms in the lesion to 133 ms in WM I), PSR's later (170–250 ms), once the fast exchange is
  over: the signal then records how much exchange happened, not how fast. kmf/PSR sensitivity
  ratio at TI = 278 ms: 0.02–0.43; at 1007 ms ≈ 0; at 50 ms 0.58–0.88. The 4-point scheme avoids
  the 30–130 ms window that the 5-point scheme and Ke's sample to measure kmf; WM I (slowest
  exchange) is the limit and carries the largest bias (−14%). PSR leakage, % per s⁻¹ of kmf error
  (4-point / 5-point / 16-point / Ke's 5-point): WM I −7.9 / −11.9 / −15.4 / −11.1, WM II −0.6 /
  −2.3 / −3.0 / −1.9; under 1 in GM and lesion for all. The paper states it: "Fortuitously, the
  SIR signal exhibits sensitivity to the other model parameters outside of the regime of kmf
  sensitivity".

### The objective

Default, Eq. 5 of Dortch 2018 (after Li et al. 2010, Eqs. 7–10, and Cercignani & Alexander, MRM
2006;56:803): v = maxₗ Σⱼ Wⱼ·MSE(θ̂ⱼₗ)/θⱼₗ² · Σₙ(TIₙ + TDₙ + t_FSE), over tissues l, equal
weights. Options:

- **Weights per parameter.** Default equal: optimizing for PSR alone was unstable in earlier
  work, and R1f matters too. A parameter with weight 0 still enters through F⁻¹, since
  it is fitted.
- **Time: × T (Eq. 5) or a fixed budget B**: Σ Wⱼ(bⱼ² + σⱼ²·T/B)/θⱼ². Bias does not shrink with
  averaging, so multiplying it by T treats it like variance; with no bias both give the same
  protocol. With bias, the best protocol, and even whether to fix a parameter, depends on the SNR
  and time available. Table 2 schemes, PSR RMSE at equal scan time: fixing kmf (4-point) wins at
  the paper's SNR in every tissue (WM I 14.9 vs 17.6%, WM II 5.6 vs 7.7, GM 7.3 vs 9.0, lesion
  12.4 vs 14.7); fitting kmf (5-point) wins beyond 1.5× that time in WM I (bias floor −14%), 3.5×
  in the lesion, 20× in GM, ~570× in WM II.
- **Aggregation over tissues**: the max (minimax: the paper, Li's Vmax, Cercignani & Alexander), a
  weighted mean (a Bayesian design; weights e.g. prevalence), a percentile; later the
  standardized minimax (each tissue relative to its own best protocol; Dette, J R Stat Soc B
  1997;59:97). At a minimax optimum the worst tissues tie: Table 2 4-point, lesion 0.2434 vs WM I
  0.2428; 5-point, lesion 1.602 vs WM I 1.586. Minimax serves the worst tissue, rare or not (the
  paper's MATLAB notes that optimizing kmf too moved it to the worst tissue's value and raised the
  WM and GM bias); for data-driven sets the max would be set by the single worst voxel, so mean
  or a percentile there.
- **Tissue sets**: a list (the MATLAB `'multiple'`), or every combination of ranges plus the
  midpoint (`'ranges'`; Li: 2⁵ + 1 = 33 sets). The worst case need not be on a corner; with fast
  CRLBs a denser grid, or an inner maximization, is affordable.

### Conditions: fixed values that are uncertain

Besides the tissues, the true values of parameters the fit fixes can vary over conditions the
user gives: B1, B0, T2m, mapped to Sf and Sm by a simulation of the inversion pulse (free pool:
rotations per pulse segment, no relaxation; bound pool: Sm = exp(−Σ π·ω₁²·g(0)·t), g(0) =
T2m/√(2π), as `calcSm_blockPulse.m`; the 1-ms block gives Sm = 0.831 at T2m = 15 µs, the paper's
0.83). The worst case (or mean) then runs over tissues × conditions: a robust design.

- **Decision: optimize robustly when a fixed parameter is uncertain.** A 3-point SIR scheme
  (90x-180y-90x composite inversion; Sf = −1, Sm = 0.69, kmf = 12.5 fixed) optimized for the
  nominal case put all its PSR information into one inversion point: nominally 10–15% lower PSR
  error than the Table 2 4-point at equal time, but with B1 ±10%, B0 ±50 Hz and T2m 10–20 µs its
  worst PSR error rose to about 50% (bias +25 to +49%), against 20% for the 4-point. Optimized
  over those conditions (objective: worst PSR RMSE at equal time), its worst case was 18.3%.
- Composite against block (2 ms against 1 ms, same amplitude): Sf at B1 0.9/1.1: −0.999 against
  −0.951; at 0.8/1.2: −0.982 against −0.809; off resonance alike (±100 Hz: −0.931 against
  −0.921). Sm 0.69 against 0.83 at T2m 15 µs (0.61–0.78 against 0.78–0.88 over 10–20 µs): twice
  the inversion's RF energy (−ln Sm ∝ ∫B1²dt). PSR leakage at the 4-point's TIs: 0.7–2.8% per
  0.01 of Sf error, about 0.6% per 0.01 of Sm. Fixing Sm already moves the Table 2 4-point's PSR
  by about ±3% over T2m 10–20 µs (the robust 3-point with the composite: about ±6%).
- **Result, with Sm fixed and correct** (the simple case), Eq. 5 with equal weights, robust over
  B1 0.9–1.1 × Δf 0/50 Hz: the 4-point re-optimization returned the Table 2 scheme (TI 10, 10,
  275, 1004 ms; TD 4242, 733, 2754, 10 ms; Eq. 5 0.2610 against Table 2's 0.2625): Table 2 is
  already robust, since it fits Sf. The best 3-point (TI 10, 17, 262 ms; TD 607, 4960, 2734 ms;
  9.5 s per repetition, 6% shorter) is 2% better in Eq. 5 over PSR, R1f and M0f (0.252 against
  0.258) but worse for R1f in WM I (2.4 against 1.7% at equal time) and less robust (worst PSR
  12% against 7–9% in WM II and GM at B1 ±10%; 19–21% at ±20%). Not adopted. An earlier estimate
  of "2.5× shorter" came from optimizing PSR precision alone with kmf known; with R1f weighted
  and the kmf bias counted, it disappears.

### The protocol space and constraints

- The acquisition fields that vary (TI, TD), their bounds (10 ms to 10 s in the prototype; the
  paper: ≥ 10 ms, no upper bound), linear constraints (TI + TD ≥ 0.5 s, the paper's minimum TR of
  0.8 s for SAR at 3 T with t_FSE = 0.3 s), one shared TD (the MATLAB `tdType = 'fixed'`),
  rounding to the ms at the end, and a canonical order (sorted by TI: reordering the points gives
  the same scheme, which a search should not count twice).
- Scan time per model, `scantime(m)`; for SIR Σ(TI + TD + t_FSE). t_FSE is a property of the
  sequence that QMTSIR does not hold yet: a keyword of the design call, or a field.
- The number of points is scanned (Li: precision efficiency nearly flat in N, 2–4% lower at small
  N; 5 points enough for qMT, 3 for T1). Optimal designs need few distinct points, repeated (Li's
  optimized 35-point scheme falls into 7 groups); optimizing a few distinct points and their
  repeat counts (or weights) is a smaller search than N free points (later).

### The optimizer

- Global search, then a local polish, from several starts on threads. Prototype: adaptive
  differential evolution (BlackBoxOptim.jl) and Nelder–Mead (Optim.jl), 12–24 restarts, which
  agreed; 3 and 4 points with 6–48 conditions per tissue: 14–43 s on 4 threads. The paper used a
  genetic algorithm then `fmincon` (SQP), 50 runs; Li and Cercignani & Alexander, simulated
  annealing from several starts.
- Gradient-based later: the objective is differentiable in TI and TD (dual numbers pass through
  the acquisition and through a fixed number of Gauss–Newton steps). The max over tissues has
  corners, so either "minimize t with every Vₗ ≤ t" (SQP) or a smooth maximum.
- In a package extension (optimizer packages are heavy); the objective stays in the core.
  (Revised in the plan below: a small search of our own, an extension later.)

### The plan: `ProtocolDesign`, an objective made once, and the search

Status: the first version implemented (src/design/: `ProtocolDesign`, `optimize_protocol`;
on `lean` since e269647, from `design-proto`), results at the end of this section; the
scenario grid, conditions and Li's targets still to do. What the sections above leave open, made concrete
with the pieces that now exist: the `Scenario` (a truth and the fit judged against it), one
`CRLBCache` per fit, and the allocation-free kernel, which takes any protocol with as many
observations. First goal: reproduce Dortch 2018 Table 2 by optimization (Fopt4 0.2433, Fopt5
1.6020), then robust designs over conditions.

```julia
tissues = [nominal(QMTSIR, t) for t in (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)]
ss = [Scenario(QMTSIR, θ; kmf = 12.5) for θ in tissues]       # what the 4-point fit assumes
d = ProtocolDesign(QMTSIR(; SIR_4POINT..., magnitude = false), ss; snr = 200,
                   acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),   # bounds, s
                   constraints = m -> m.TI .+ m.TD .- 0.5,          # TR ≥ 0.8 s with t_FSE 0.3 s
                   scantime = m -> sum(m.TI .+ m.TD .+ 0.3))        # Σ(TI + TD + t_FSE)
d(QMTSIR(; SIR_4POINT..., magnitude = false))   # Eq. 5 at a protocol: 0.243395
evaluate(d, m)                    # a table: each scenario's relative bias, SD and RMSE, the time
r = optimize_protocol(d; starts = 24, rng = MersenneTwister(1))     # search, then polish
r.protocol, r.value               # the best protocol (sorted by TI) and its Eq. 5
r.rounded, r.rounded_value        # the same rounded to the ms, evaluated again
```

- **An objective made once, called many times** (`ProtocolDesign`, as `CRLBCache` and `init`):
  it holds one `CRLBCache` per distinct fit among the scenarios (scenarios that differ only in
  the truth share one: the four tissues above, all with kmf = 12.5, use a single cache), each
  scenario's truth in the kernel's form and its σ² (σ = M0/snr from the scenario's truth), the
  weights and a buffer for the per-scenario values. `d(x)`, x the varied acquisition values as
  one vector, builds the protocol with static vectors and calls the kernel once per scenario.
  Measured (scratch prototype): Eq. 5 at the Table 2 4-point scheme 0.243395 (crlb's), 11.5 µs
  for the four tissues, 0 bytes allocated, the protocol's construction included. Not
  thread-safe (the caches' buffers): one copy per task, as `fit` copies its cache per chunk.
  `d(m)` at a model, for checking a published scheme.
- **Scenarios carry the tissues and the wrong assumptions.** A vector of `Scenario`s: each says
  what is true and what the fit assumes, so a wrong assumption is counted as the bias it causes
  (Dortch's biased CRLB inside Eq. 5) without a separate mechanism. Conditions (B1, B0, T2m,
  mapped to Sf and Sm) are more scenarios, the truth varied and the fit kept:
  `Scenario(QMTSIR, merge(θ, (Sm = 0.78,)); kmf = 12.5, Sm = 0.83)`, and `scenarios` makes the
  sets (done: "Sets of scenarios" below). A design optimizes one fit's protocol in
  the usual case; comparing fits (kmf fixed or fitted) is comparing their designs' optima, as
  Table 2 does.
- **Sets of scenarios: `scenarios(M, truths; vary, midpoint = false, fit...)`** (`scen-proto`),
  the modes of the paper's `calcVmax.m`: `'single'` (one truth), `'multiple'` (discrete
  tissues: what the 4-point paper used, for its computing time) and `'ranges'` (every
  combination of each parameter's values, plus the midpoint: for two values each, a box's
  corners and center, Li 2010's 2⁵ + 1 = 33 sets), and the two combined: each tissue × each
  combination (tissues × conditions, for a robust design; calcVmax holds Sm common).
  - *`vary` is crlb's:* the same forms (a pair, pairs, a named tuple), values a vector or a
    range, every combination with the first parameter slowest, the truth changed and the fit
    never (one parser, `_varyaxes`, and one ordering, `_sweep_settings`). A first proposal
    wrote boxes as tuples, `PSR = (0.05, 0.2)`, which crlb's `vary` rejects: one syntax for both.
  - *The result is a `Sweep` whose results are the scenarios:* `settings` says which tissue
    (`tissue`: its name, key or place; slowest) and values each has, `rows` a table (the
    setting and every true value), and the display a row per scenario with what its fit gets
    wrong. `ProtocolDesign` takes it unchanged (it takes any collection of scenarios); after an
    evaluation `d.values` holds each one's value, so `s.settings[argmax(d.values)]` is the worst.
  - *`midpoint = true` adds the box's center* (calcVmax's `'ranges'` always includes it); put
    first, as calcVmax evaluates it first; halfway between each parameter's smallest and
    largest value, rounded to 12 significant digits (found by the tests: (0.78 + 0.88)/2 is
    0.8300000000000001, which a fit assuming 0.83 counted as wrong, and which duplicated a given
    0.83); not added when a combination is within 1e-12 of it. Numbers only (a tie has no
    midpoint). Off by default (`names-proto`; first built on by default): `vary` then gives
    exactly the combinations of the values given, as for `crlb` and `montecarlo`, where the
    same `vary = :Sm => [0.78, 0.88]` had given 2 cases and 3 here.
  - *A fixed parameter the fit does not set follows each scenario's truth* (`Scenario`'s rule):
    varying Sm without `Sm = 0.83` gives unbiased fits, one per Sm, so one cache each (3 groups
    for one tissue × 3 values, tested).
  - *Many scenarios are cheap to hold:* a `Scenario` is 472 bytes; `ProtocolDesign` keeps, per
    fit, one cache and a typed vector of truths (the tuple is over fits, not scenarios), so
    evaluation is a loop and the compile time does not grow with their number. Measured (one
    fit, Table 2's 4-point): 4 tissues 17.5 µs an evaluation, a 2⁵ + 1 box 144 µs, a 3⁵ grid
    (243) 783 µs: 3–4 µs a scenario; the design made in 0.5, 1.1 and 7.5 ms. The search is then
    the cost (243 scenarios: about 30 s per start of differential evolution, 40,000
    evaluations); if it matters, an active set for minimax (search over the currently worst
    scenarios, check all at the end), not another container. At Table 2's scheme the box and
    the 3⁵ grid give the same value (0.8526): at that protocol, a grid's interior points found
    nothing worse than the corners; that is not a guarantee (see "What a set is, and is not",
    below).
  - *Checked for efficiency* (warm, best of many runs): `scenarios` costs about 15 µs a
    scenario, once (243 in 3.7 ms: 4–5 evaluations' worth); a design made from its `Sweep`
    costs what one from a vector does; an evaluation allocates nothing, and 93–96% of it is
    crlb's kernel (the design adds 0.2–0.4 µs a scenario). No regression: the test's 4-point
    design takes 17 µs an evaluation on this machine on the commit that recorded 11.4 µs
    (a3ac634), on `lean` and here alike (the machine is slower than then); the 5-point design
    5.2 µs (6.6–9.7 at a3ac634: the unbiased rows reused since). Where more speed would come
    from, if a search needs it: warm starts (each scenario's noise-free fit from its answer at
    the previous protocol: a search's successive protocols are close, and the solver's
    iterations are most of a biased scenario's kernel), and the active set above.
  - *What a set is, and is not* (`set-proto`, after review): stated in `scenarios`' docstring.
    A set of points, not the box: corners and the midpoint need not contain the worst case of a
    nonlinear model, so a set's minimax is an approximation, not an exhaustive robust optimum
    (the check is on the roadmap: an inner maximization over the box). Equal weights: with
    `aggregate = mean` the center counts as much as each corner, and a midpoint added to two
    values gets a third of the weight; a discrete mean, not one over the continuous box (for
    that: a regular grid or quadrature points with `midpoint = false`, or weights). The scale
    parameter need not be varied: with `snr` relative to it the relative errors do not depend
    on it (Eq. 5's terms at M0f 0.8, 1.0, 1.2 agree to 6e-14), so SIR's box over PSR, kmf,
    R1f and Sf is 2⁴ + 1 = 17 scenarios, not calcVmax's 33 with M0f. Free against fixed:
    varying a free parameter moves the operating point; varying a fixed one with the fit's
    value given is a misspecification (each scenario's `wrong`).
  - *crlb and montecarlo take a set* (`set-proto`): one result per scenario, a `Sweep` with the
    set's settings, after the noise levels when `snr` is a vector; one cache per scenario,
    reused across noise levels; montecarlo's from one seed (common random numbers). `vary` and
    `simulate_with` are refused: the set's truths are given. Each result equals the
    single-scenario call (tested).
  - *Checked:* the tissue set gives the hand-built list's design value to the bit (0.243395,
    Table 2); the box's 33 settings in calcVmax's order; the robust set's 12 (0.3274 at Table 2,
    against 0.2434 over the tissues; worst WM I at Sm = 0.78), equal to the hand-built 12.
    31 tests.
- **Eq. 5 by default, every choice a keyword**:
  - `weights`: per parameter, by name (default 1 for each free parameter, as Eq. 5). A
    parameter with weight 0 still costs precision through the fit (it is estimated).
  - Time: `× T` (default, Eq. 5: precision per unit time, Σⱼ Wⱼ(bⱼ² + σⱼ²)/θⱼ² · T), or
    `budget = B` (Σⱼ Wⱼ(bⱼ² + σⱼ²·T/B)/θⱼ²: averaging shrinks the variance, never the bias).
    With bias the two can choose different protocols, and whether to fix kmf at all depends on
    the time available (above, "The objective").
  - `aggregate`: over scenarios, `maximum` (default: minimax, Eq. 5), a weighted mean (scenario
    weights, e.g. prevalence), or any function of the vector of per-scenario values (a
    quantile).
  - Relative to the truth: a free parameter with true value 0 cannot be; an error at
    construction, unless its weight is 0. Parameters the data cannot determine (SD `Inf`) make
    the value `Inf`, which the search avoids.
- **The protocol space**: `acquisition` names the fields to optimize with their bounds (named
  `vary` until `names-proto`, renamed because `vary` elsewhere, in `crlb`, `montecarlo` and
  `scenarios`, changes true parameter values: two meanings for one keyword, and a parameter
  name for the acquisition, against the naming rule); the other fields
  stay as in the model given (its number of points, `magnitude`, TR of an MFA). One value shared
  by every point (the MATLAB's `tdType = 'fixed'`) is an option (syntax open). Constraints are
  functions of the protocol whose values must be ≥ 0 (a number or a vector): a penalty in the
  search (its size the violation, so the search is led back), exact in the result. The scan
  time is a function of the protocol (`scantime`): t_FSE is a property of the readout, which
  QMTSIR does not hold. The order of the points does not change the CRLB (sums over
  observations), so no canonical order in the search; the result is sorted by TI. Rounded to
  the ms at the end and evaluated again (both reported).
- **The search**: differential evolution (Storn & Price, J Glob Optim 1997;11:341) within the
  bounds, then Nelder–Mead from its best point, from `starts` random populations on threads
  (one `ProtocolDesign` copy each), the model given as one of the starts; `rng` for repeatable
  runs. The earlier prototype (BlackBoxOptim's adaptive DE, Optim's Nelder–Mead) found the two
  enough and in agreement. Proposed: both written here, short and readable (about 60 and 40
  lines), rather than an extension on BlackBoxOptim and Optim: no dependencies, the inner loop
  stays allocation-free, the same `rng` convention as `simulate`. `d(x)` is a plain function,
  so any other optimizer works on it today, and an extension (Optimization.jl) can come later.
  Budget: DE with a population of 10 × dim (80 for 4 points) over 500 generations, 40 000
  evaluations: about 0.5 s a start for the four tissues (11.5 µs each), 24 starts in about 3 s
  on 4 threads; 48 scenarios (4 tissues × 12 conditions) about 12× that.
- **The result** (`DesignResult`): the protocol (a model), its value, the rounded protocol and
  its value, `evaluate`'s table at it, and every start's best (value and protocol), so the
  spread across starts shows whether the search is reliable.
- **Where**: `src/design/`: design.jl (`ProtocolDesign`, `evaluate`, the scenario grid),
  search.jl (DE, Nelder–Mead, `optimize_protocol`, `DesignResult`). Diagnostics (sensitivity
  maps, leakage; "Diagnostics" above) are independent of this and can follow.
- **Accepted when**:
  - `d` at the Table 2 schemes gives 0.243395 (4-point, kmf 12.5) and 1.602062 (5-point, kmf
    fitted), both covariances, as crlb; `d(x)` allocates nothing.
  - `optimize_protocol` reaches Fopt4 ≤ 0.2433 and Fopt5 ≤ 1.6020 (the paper's GA optima
    before rounding), with most of 24 starts within 1%, and the rounded schemes near Table 2's
    (any order of the points).
  - Li 2010's T1 targets (62.4 s: 10 points, one TD of 1 s; 44.0 s: TI and TD free; 49.8 s:
    3 points), with his conventions (σ = 1; the T1 experiment's t_FSE is uncertain, "Tests
    and acceptance" below).
  - Robust over B1 0.9–1.1 × Δf 0/50 Hz with Sm fixed and correct: the 4-point re-optimization
    returns the Table 2 scheme (the prototype's: Eq. 5 0.2610 against Table 2's 0.2625).
- **Order**: (1) `ProtocolDesign`, `d(x)`, `evaluate`, tested at the Table 2 schemes; (2) the
  search, Fopt4 and Fopt5; (3) the scenario grid and conditions (Sf and Sm from a pulse model:
  open, below), robust designs; (4) Li's targets; then diagnostics and data-driven design.
- **Open, for review**:
  - The search written here, or an extension on existing packages (proposed: here).
  - `scantime` as a function keyword (proposed), or a readout time in the model (QMTSIR has
    none).
  - The default time handling: × T (proposed, to reproduce the paper) or a budget.
  - Names: `ProtocolDesign`, `evaluate`, `optimize_protocol`, `DesignResult`, the scenario
    grid's helper; the syntax for a value shared by every point.
  - Distinct points with repeat counts (Li's optimized 35-point scheme falls into 7 groups):
    later.

- **First results** (`design-proto`; 4 threads; the MATLAB code's settings: signed signal,
  t_FSE 0.3 s, SNR 200, its fits' bounds for the 4-point scheme, TI and TD in 10 ms–10 s,
  TI + TD ≥ 0.5 s):
  - The objective at the Table 2 schemes: 0.243395 (4-point) and 1.602062 (5-point), crlb's,
    both covariances; 11.4 µs and 4.9 µs an evaluation over the four tissues, no allocation.
  - The search, 8 starts: 4-point 0.243255 (every start; the paper's GA optimum before
    rounding, Fopt4 0.2433), rounded to the ms 0.243312, below Table 2's rounded scheme
    (0.243395), in 4 s; 5-point 1.602025 (7 of 8 starts; one at 1.6045), Fopt5 1.6020, rounded
    1.60206, Table 2's own value, in 1.6 s. The 5-point optimum rounded to the nearest ms is
    Table 2's scheme exactly (TI 10, 50, 56, 277, 843 ms; TD 3270, 4489, 1652, 2922, 10 ms).
  - **Rounding to the ms is a search of its own.** At the 4-point optimum WM I and the lesion tie
    (0.023991 each, × T); rounding each value to the nearest ms raised WM I to 0.024073 and the
    maximum by 0.3%, to 0.244072, worse than Table 2's rounded scheme. So the result is the best
    of the ways to round each value down or up (2^D of them up to 12 variables: 256 for 4
    points, 3 ms), then single ms steps while one improves: 0.243312.
  - **DE, simulated annealing or a GA** (the question for the global stage): each written for
    the comparison (SA on log Eq. 5, one coordinate at a time, steps adapted to 44% acceptance,
    geometric cooling; a real-coded GA with binary tournaments, simulated binary crossover and
    polynomial mutation, Deb & Agrawal, Complex Syst 1995;9:115, two best kept), 24 starts each,
    the same budget per start, each then polished by the same Nelder–Mead. Starts within 0.01%
    of the best known after the polish:

    | budget (evaluations a start) | | Fopt4 (4-point) | Fopt5 (5-point) |
    |---|---|---|---|
    | 5 000 × D | DE | 24/24 | 20/24 (all within 0.14%) |
    | | SA | 23/24 (1 at 0.292) | 11/24 (worst 1.726) |
    | | GA | 16/24 (8 at 0.292) | 24/24 |
    | 10 000 × D, another seed | DE | 24/24 (converged at 67 000) | 22/24 (worst 0.015% off) |
    | | GA | 22/24 (2 at 0.292) | 24/24 |

    Before the polish, the GA was the closest on Fopt5 (all 24 within 0.01%) and DE on Fopt4.
    Decision: DE (with Nelder–Mead), the default `generations = 1000` being the second budget:
    no start of DE was ever more than 0.14% from the optimum, where a GA start could stop in a
    local minimum 20% worse (0.292 against 0.243) a third of the time. SA was the weakest on
    both. Taking the best of the starts, DE and the GA both found the optimum of each problem.
- **The unbiased case without the extra work** (`design-proto`): when nothing fixed is wrong,
  the CRLB is σ²(JᵀJ)⁻¹ at the truth, one evaluation of the model and an inverse; the kernel took
  about twice that (4-point 0.60 µs against 0.28 µs for the bare formula; 16-point 1.52 against
  0.63), since `hessian!` evaluated the model again at the point the noise-free fit had just
  evaluated (the fit itself stops at its start: residual exactly 0, 0 iterations). Now, when the
  fit stopped at its start with a residual of exactly 0, its rows are copied (`_startrows!`):
  0.51, 0.57 and 1.11 µs (4-, 5-, 16-point), 1.6–1.8× the bare formula; the rest is what makes it
  general (the fit's check at the start, the test that the data determine the parameters, the
  sandwich), kept rather than a second, unbiased path to keep consistent. A separate `cov =
  :unbiased` was considered and not added: "unbiased" means either the fit with the fixed
  values at the truth (`Scenario(M, θ)`, the default) or the naive CRLB that takes the
  assumption as the truth (`merge(θ, (kmf = 12.5,))`), both already a scenario, and side by side
  in one call: `crlb(m, (assumed = s, unbiased = Scenario(M, θ)); snr)`. Results identical (the
  688 cases with either default, the 8 sweeps, the design objectives); a protocol search with
  unbiased scenarios (5-point, kmf fitted) 21% faster an evaluation (3.24 → 2.56 µs); biased
  ones unchanged.
### Data-driven design

- Tissues from a `FitResult`: the maps in `res.params`, masked to converged voxels and without
  CSF or partial volume, reduced to representative sets (clusters, quantiles, segmentation
  labels) or used as a weighted sample. `res.settings` is the analysis model. σ from the image
  background or a noise scan: an exact fit (the 4-point scheme) leaves zero residuals.
- Aggregation: mean or a percentile (a Bayesian design with an empirical prior), not the max,
  which one bad voxel would set.
- Better input from a richer pilot: a 4-point fit has no kmf map and a biased PSR; a pilot with
  kmf free (5- or 16-point) gives better tissues. The paper's scripts did this by hand
  ("Investigation of variable td schemes": ROI fits with the 16-point scheme, other schemes
  evaluated at those values).
- Adding points: Fisher information adds over points, so "the best k points to add to what was
  acquired" is the same calculation: pilot, then an optimized scan.
- One subject or a study: several `FitResult`s pooled (one subject's optimum is tuned to that
  subject).

### Where it lives

`src/errors/`: `crlb`, `montecarlo`, sensitivity maps, leakage. `src/design/`: the objective,
`scantime`, constraints, tissue sets (presets, ranges, or from `FitResult`s). `ext/`: the
optimizer (`optimize_protocol`). Names and keywords to settle with the first implementation.

### Related work

No Julia package does general qMRI protocol design with a biased CRLB and a worst case over
tissues (search, Oct 2026). Closest: MRIgeneralizedBloch.jl (Assländer et al.; qMT with the
generalized Bloch model, CRB and gradient-based optimal control of hybrid-state sequences; its
notebook `CRB_SIR.ipynb` computes the unbiased CRB of the 5-point SIR scheme, a cross-check) and
Kirstine.jl (Bayesian and locally optimal designs for nonlinear regression, D and A criteria).
CRLB protocol design in qMRI: Cercignani & Alexander 2006 (qMT), Li 2010 and Dortch 2018 (SIR),
Zhao et al., IEEE TMI 2019 (MRF), Lee et al., MRM 2019;82:1438 (automatic differentiation of
Bloch simulations). Bias from a wrong fixed parameter: the misspecified CRB (Fortunati et al.,
IEEE Signal Process Mag 2017), applied to qMRI fixed parameters in an ASL study (arXiv
2602.10336, 2026), and Lankford & Does 2013 (analysis, not design). Putting that bias into the
design objective, as Dortch 2018 did, was not found elsewhere (a limited search). Bias from noise
(second order): Box, J R Stat Soc B 1971;33:171; Cook, Tsai & Wei, Biometrika 1986;73:615.
Optimal design theory: Fedorov & Leonov 2013; Chaloner & Verdinelli, Stat Sci 1995. Implicit
differentiation through solvers: ImplicitDifferentiation.jl.

### Tests and acceptance

- Dortch 2018: Fopt5 = 1.6020 and Fopt4 = 0.2433 (the GA optima before rounding; the Table 2
  schemes as rounded give 1.602062 and 0.243395, matched); the optimizer reaches them or better
  (any order of the points). Fig. 1 (lesion: 4-point mean PSR 4.75%; Rician Monte Carlo against
  the CRLB), Fig. 2A (Eq. 6 maps on the scripts' grids), Fig. 2B (bias and RMSE against the true
  kmf).
- Li 2010, Table 1 (σ = 1; t_FSE = 0.155 s, from the table's time costs): V of the typical set,
  schemes 2 and 3 within 1% (1.389e4 vs 1.39e4; 1.50e4 vs 1.51e4), scheme 1 +5% (3.31e4 vs
  3.14e4); T1, conventional 10-point scheme: 274.9 vs 286.1 s (−4%; t_FSE of the T1 experiment
  uncertain). Vmax over the 33 sets is 1.53× ours for all three schemes: a convention difference
  in the sets (open). T1 optimizer targets: 62.4 s (10 points, one TD of 1 s), 44.0 s (10 points,
  TI and TD free), 49.8 s (3 points).
- Exact derivatives against finite differences; the exact-fit identity (Eq. 3 = σ²(J₀ᵀJ₀)⁻¹);
  the multi-start check finds the global noise-free fit; σ per observation; Monte Carlo SD/CRLB
  ≈ 1 at high SNR.

### Order of work

1. `crlb`: unbiased and biased, Eq. 3 and the sandwich, σ per observation; tests: the paper's
   numbers and Li's Table 1.
2. `montecarlo`, one case first (done), sweeps over `snr` and `vary` (done, with
   `crlb`'s), fit settings side by side and `simulate_with` (done); Figs. 1
   and 2B. Moved ahead of the diagnostics: it is the check on `crlb`, and it
   settles the sandwich against Eq. 3 with data (16-point, WM I, kmf fixed: 2.88% vs 2.72%).
3. Diagnostics: `cond(c)`, `cor(c)` and `leakage(m, s)` done ("Diagnostics" above); Eq. 6 is
   `jacobian` ./ √t; its plot (Fig. 2A) on the roadmap, item 3.
4. The objective (Eq. 5 and its options; tissue sets; conditions), `scantime`, the protocol
   space: `ProtocolDesign` (done; the tissue × condition grid next).
5. The optimizer: reproduce Fopt5 and Fopt4 (done, `optimize_protocol`), and Li's T1 targets.
6. Data-driven: tissues and σ from `FitResult`s; adding points to a protocol.

### Open decisions

- Li's Vmax factor (1.53×): how his 33 sets were built (Ke's code?).
- The pulse model for Sf and Sm (rotations and saturation, no relaxation, for now), and where it
  lives: a QMTSIR helper, or user functions Sf(B1, B0) and Sm(B1, T2m).
- Default ranges for conditions (B1 ±10% or ±20% at 3 T; B0 ±50 Hz).
- Gaussian Fisher information only (Rician by Monte Carlo), or a Rician one for low-SNR
  protocols.
- The default time handling: × T (the paper) or a fixed budget.
- t_FSE: a QMTSIR field or a keyword of the design call.
- Designs as distinct points with repeat counts (later).

## Weighted fits: the plan (2026-10, branch `weights-proto`)

Decided with the owner: the keyword is `weights` for the fit and for `ProtocolDesign`'s
per-parameter weights alike (a vector by observation for the fit, a named tuple by parameter
for the design: told apart by type); diagonal weights first, one per observation, the same in
every voxel. A full weight matrix (GLS, W = Σ⁻¹ for correlated noise) and per-voxel weight
maps are later.

- **What the fit minimizes**: ½ Σᵢ wᵢ(gᵢ(θ) − yᵢ)², wᵢ ≥ 0 (0 drops an observation), at least
  one positive. Only the ratios matter for the estimates. W = Σ⁻¹ (wᵢ = 1/σᵢ²; with averages
  wᵢ = NEXⱼ) is the efficient choice for Gaussian noise: the fit then reaches the bound
  (JᵀΣ⁻¹J)⁻¹ that crlb's `:biased` gives for an unbiased fit, where the unweighted fit's
  sandwich is above it.
- **Where the weights live**: a keyword of `fit`, `init` and `refit` (`weights = nex`), checked
  once (length N, finite, ≥ 0, one positive) and kept in the `FitSpec` (`weights`: `nothing`
  or a vector in the working precision), so the solver, the statistics and crlb's cache read
  them from one place. `VoxelProblem` carries them to the solver; `nothing` means unweighted,
  and every sum multiplies by `_wt(w, i)`, which is `one(T)` then: x·1 is exact, so the
  unweighted results stay the same bit for bit (checked against a snapshot: fits of every
  model in both precisions, crlb, leakage, the design, a montecarlo), at the same speed.
- **The solver**: JᵀWJ, JᵀWr and Σwr² row by row (`normal_equations`); the starts' scale
  (`_score`) as the weighted slope a = Σwsy/Σws², the SSE weighted. The model's `datastart`
  (log-linear and the like) stays unweighted: it only starts the fit.
- **Statistics**: `res.sse` is the weighted SSE. `residualsd` = √(Σwr²/(N − P)), the SD of an
  observation of weight 1 (so with wᵢ = 1/σᵢ², about 1; with wᵢ = NEXᵢ, one average's σ); `vcov`
  = σ̂²(JᵀWJ)⁻¹, right when W ∝ Σ⁻¹; `stderror(res; σ)` takes σ as that unit-weight SD.
  N counts the observations with positive weight. `loglikelihood` for variances s²/wᵢ:
  −N/2(log(2πŝ²) + 1) + ½Σ log wᵢ, ŝ² = Σwr²/N (unchanged when every w = 1, and the same for
  any scaling of w).
- **Scenario**: the weights are part of the fit judged (`Scenario(m, θ; weights = nex)`,
  `scenarios(...; weights)`, and through `crlb(m, θ; weights)`, `montecarlo`, `leakage`), checked
  against the protocol's N where it meets one. Shown in the display.
- **crlb's kernel** (the cache takes the spec's weights): the noise-free fit weighted;
  H = Σ wᵢ(JᵢJᵢᵀ + rᵢ∇²gᵢ), G = Σ wᵢJᵢJᵢᵀ; the meat J₀ᵀWΣWJ₀ (Σ wᵢ²σᵢ²JᵢJᵢᵀ; with a full Σ,
  rows of LᵀWJ₀), so the sandwich is H⁻¹J₀ᵀWΣWJ₀H⁻¹, = (JᵀΣ⁻¹J)⁻¹ when W = Σ⁻¹ and the fit is
  unbiased; K = J₀ᵀWJ_t and ∇b = H⁻¹K − I for Eq. 3. The Fisher information F = J_tᵀΣ⁻¹J_t,
  `c.information` and `c.observed` are the likelihood's, not the fit's: unchanged. The exactness
  test compares the weighted SSE with Σwy².
- **leakage**: the joint cache weighted the same way (dθ̂/dc = −H_θθ⁻¹H_θc with the weighted H).
- **montecarlo** fits with the scenario's weights; **ProtocolDesign** groups scenarios by their
  weights too (part of the fit's key).
- **Not changed**: `jacobian` (the signal's), `informationmatrix` (the likelihood's).
  `hessian(m, s, y; Σ)` is the likelihood's Hessian (with Σ given), as now.
- **Tests**: weights all 1 give the unweighted fit exactly; weights scaled give the same
  estimates; a zero weight equals dropping the observation; a weighted fit equals an unweighted
  one of √w-scaled data and model (T2 with S0 scaled); crlb's weighted sandwich against the
  closed form (JᵀWJ)⁻¹JᵀWΣWJ(JᵀWJ)⁻¹ and, with W = Σ⁻¹, against (JᵀΣ⁻¹J)⁻¹; montecarlo of a
  weighted fit against crlb; stderror against the closed form; errors (length, negative, all
  zero, NaN, a parameter named `weights`).
- **Built as planned** (2026-10), with one addition: the results say a fit was weighted
  (`c.weights`, the crlb and montecarlo displays: "weighted fit: weights 1.0 to 16.0 (5
  observations)"; the scenario's too). The unrolled bias gradient takes Gauss–Newton steps with
  the weights; unweighted it keeps (JᵀJ)⁻¹Jᵀr itself (J'·I·J rounds differently from J'J).
  Measured: unweighted results bit-identical (the snapshot: 91 entries, every bench case and
  the other built-in models in both precisions, statistics, crlb in every noise form and
  covariance, sweeps, sets, the pieces, leakage, the design, montecarlo); speed unchanged
  within run-to-run noise, alternating with the commit before (`fit!` 1.6–20 µs by model, a
  100k-voxel dataset, `crlb!` 5.2 µs and the 5-point Eq. 3 13.3 µs, the design 11.6 µs,
  leakage 67 µs); the weighted fit, crlb kernel and design allocate nothing.
  Weighted against unweighted (500 voxels, SNR 100): the sums cost the same alone (16-point SIR
  702 and 711 ns), but inlined into `levmar` (twice) the weighted loop made the solver slower for
  the bigger models with the same iterations (weights all 1: 16-point SIR with kmf free 8.3 →
  11.5 µs, biexponential 15.9 → 18.8 µs, T2 1.55 → 1.61 µs). Not the size of the compiled
  solver: inlined, the weighted `levmar` had about as many lines of LLVM IR as the fast
  unweighted one (16-point SIR 4040 against 3962), so how LLVM optimizes the merged loop, not
  yet pinned down. So `normal_equations` is one function for both (each term times `_wt`, 1
  unweighted) and not inlined: unweighted 8.3, 15.7 and 1.56 µs (as inlined, within noise; the
  same numbers bit for bit), weights all 1 8.7, 15.9 and 1.55 µs. (Two methods, the unweighted
  inlined, were 1–3% faster unweighted; one function was chosen to keep one sum to read.) Weights that differ change the
  path, not the cost of a step: with weights 1 to 8 the biexponential takes 28.7 iterations
  against 22.0 (24.0 → 30.7 evaluations), so its fits take longer; T2, T1IR and ADC the same. T2 decay, σ 5 → 40
  over 32 echoes, wᵢ = 1/σᵢ²: SD of T2 1.15e-3 over 2000 copies against the bound 1.12e-3
  (unweighted 1.75e-3 against its sandwich 1.70e-3: 1.5×); median stderror 1.12e-3, σ̂ ≈ 1;
  montecarlo (20 000 copies) SD/crlb 0.99 ± 0.005. Leakage against central differences of the
  weighted noise-free fit: 3e-11; ∇b implicit against unrolled: 5e-10. 50 tests.

## The command line and the data reader: the plan (2026-11-01)

For the prerelease: one command per part of the package, and a data reader, so that someone
who does not write Julia can fit maps, predict and measure errors, and design a protocol.
Replaces items 9 and 10 of the roadmap for the release's scope; what they hold beyond it (more
formats, the `--init` prep step, a server, Python and MATLAB) stays there, for later.

### Decided with the owner

- **One command, `equip`, with a subcommand per part**: `equip fit`, `equip crlb`,
  `equip montecarlo`, `equip design`.
- **Flags first.** Every setting is a flag, so a quick command needs no file. (A file per run,
  first proposed, was too much for a quick command.)
- **A config file only when wanted, and it is the command line itself**: `--config run.cnf`
  reads a plain text file with one command-line argument per line and `#` comments, exactly
  what one would type (FSL's `--config=` files, e.g. topup's `.cnf`); flags given alongside
  override it. `--save run.cnf` writes the file a command amounts to, so a quick command becomes
  a record to rerun or edit. One syntax to learn, typed or saved.
- **Why not TOML, JSON, YAML or XML** (first proposed: TOML). What the neuroimaging tools do:
  FSL, MRtrix3 and ANTs take everything as flags; FSL's config files are the flags, one per
  line; MRtrix3's one config file holds global preferences as `Key: value` lines; fMRIPrep
  writes its settings to TOML as a record of a run, not as something users write; BIDS keeps
  metadata in JSON sidecars and tables in TSV. JSON has no comments and is strict about quotes
  and commas (good for files programs write, poor for files people annotate); XML is too
  verbose to write by hand; TOML and YAML are one more syntax, unfamiliar in imaging.
- **JSON and TSV where imaging tools read them**: acquisition values read from BIDS JSON
  sidecars; a JSON sidecar written next to each map (the model, acquisition, settings, package
  version: how the map was made); crlb and montecarlo tables as TSV (Excel, MATLAB, Python, R).

### The commands

```bash
equip fit T2Decay t2.nii.gz                                       # TE from the BIDS sidecar
equip fit QMTSIR sir.nii.gz --TI 0.01,0.05,0.056,0.277,0.843 --TD 3.27,4.489,1.652,2.922,0.01 --kmf 12.5
equip fit QMTSIR sir.nii.gz --protocol SIR_5POINT --mask brain.nii.gz --Sm sm_map.nii.gz
equip fit ADCDecay dwi.nii.gz --b @dwi.bval --weights @nex.txt
equip crlb QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I --snr 200 --kmf 12.5
equip montecarlo QMTSIR --protocol SIR_4POINT --tissue SIR_WM_I --snr 200 --noise rician --kmf 12.5
equip design QMTSIR --protocol SIR_4POINT --tissues SIR_WM_I,SIR_GM,SIR_lesion --snr 200 \
    --kmf 12.5 --acquisition TI=0.01:10,TD=0.01:10 --readout 0.3
equip fit QMTSIR --help                                           # this model's fields and parameters
```

- **The model**: the first argument, a built-in model's name (a user's own model is used from
  Julia). `equip <part> <model> --help` lists its acquisition fields and parameters with their
  defaults, generated from the model (`parameters(m)`, its fields), so it never goes stale.
- **Acquisition fields**: a flag named after the field (`--TI`, `--TD`, `--TE`, `--b`, `--FA`),
  its values as a comma list, a range `start:step:stop` (`--TE 0.01:0.01:0.32`), or `@file`
  (one value per line, or whitespace-separated: an FSL `.bval` file as it is); or a named
  protocol (`--protocol SIR_4POINT`). Priority: flags, then a config file, then sidecars.
- **Parameters**: a flag named after the parameter, by `fit`'s rules: `--kmf 12.5` fixes it,
  `--Sm map.nii.gz` fixes it to a map, `--R1m R1f` ties it, `--kmf free` estimates it,
  `--R1f free,start=0.8,lower=0` with a start or bounds.
- **Options**: `--mask`, `--precision`, `--weights`, `--out` (a prefix), `--stderror` (also
  write standard-error maps); threads by Julia's `-t` (or `JULIA_NUM_THREADS`).
- **crlb, montecarlo, design**: the truth by `--tissue`/`--tissues` (the presets) or
  `--truth PSR=0.1,R1f=1.0,...`; the noise by `--snr` or `--sigma`, `--averages`; montecarlo's
  `--noise` (required, as in Julia), `--averaging`, `-n`, `--seed`; `--vary Sm=0.78,0.88` (true
  values); `--cov`.
- **design**: the acquisition to optimize with its bounds (`--acquisition`), the scan time by a
  rule (`--readout 0.3`: Σ(TI + TD + readout), the SIR-FSE case) or a Julia expression of the
  protocol's fields (`--scantime "sum(TI .+ TD .+ 0.3)"`, run as given: a local tool, the
  user's own command), `--budget`, `--weights PSR=1,R1f=0`, `--aggregate`.

### What each writes

- **fit**: one NIfTI map per parameter in the input's geometry (intensity scaling reset;
  BIDS-derivative names where they exist: `T1map`, `R1map`, `T2map`, `M0map`, `S0map`, ...),
  status and SSE maps, standard errors with `--stderror`; a JSON sidecar beside each.
- **crlb, montecarlo**: the table (`rows`) as TSV, and the summary printed as at the REPL.
- **design**: the optimized protocol printed and written as a config file (its acquisition
  flags), so `equip fit --config optimized.cnf` fits with it; its Eq. 5 value against the
  starting protocol's.

### The data reader

`data, info = loaddata(paths)`: the data with the observations last, and `info` (the
geometry, to write maps back, and the acquisition values found). For the release:

- **NIfTI** (`.nii`, `.nii.gz`; NIfTI.jl): one 4-D file, or several 3-D files (one per echo,
  flip angle or inversion time) stacked along the last dimension in the order given; the
  scaling (`scl_slope`, `scl_inter`) applied.
- **BIDS JSON sidecars**: `EchoTime`, `RepetitionTime`, `FlipAngle`, `InversionTime` (s,
  degrees) into the model's fields, for files from `dcm2niix` (so DICOM from any scanner,
  converted first). Checked: the number of values against the data's last dimension; values
  given as flags win, and a disagreement with a sidecar is reported.
- **MATLAB `.mat`** (MAT.jl; decided 2026-11-01): the data as a variable picked by name
  (`data.mat:S`, or the only array in the file), observations last as for NIfTI; acquisition
  values from a variable too (`--TI @protocol.mat:TI`). No geometry in a `.mat` file, so maps
  are written to a `.mat` file (one variable per map), or to NIfTI with `--like ref.nii.gz`.
  (Waiting for the session's network to allow Julia's package server, which MAT.jl comes
  from; NIfTI.jl and a JSON package were installed already.)
- **Text files** for values: `@file` (above).
- **Writing**: `savemaps(prefix, res; like = info)`.
- **Later** (item 9): Philips PAR/REC, DICOM read directly, NumPy, ISMRMRD.

### Where it lives, and start-up

- **A companion package in the same repository** (e.g. `cli/`, with its own Project.toml as
  `bench/` has): EQUIP plus NIfTI.jl, JSON.jl and MAT.jl. The flags are parsed by its own
  small parser, not ArgParse.jl: they depend on the model (its fields and parameters), which a
  parser declared in advance does not fit, and one dependency less to load at each call. The core keeps no file-format dependencies. Installed as a Julia app, an
  `equip` command (Pkg apps, Julia ≥ 1.12; the core's compat stays 1.11), or run as
  `julia --project=cli -m EQUIPCLI fit ...`.
- **Start-up**: a call is a new Julia process: about 1 s to start Julia and load EQUIP (item
  10, measured), then the work, as long as nothing compiles. So the precompile workload covers
  each subcommand with each built-in model and the options the flags offer, and a test runs
  each in a fresh process with `--trace-compile` and fails if anything compiled.

### Tests

Each subcommand on small synthetic NIfTI files written by the tests, against the Julia API's
results (the same numbers); the flags' parsing (lists, ranges, `@file`, roles); `--save` then
`--config` gives the same run; sidecar values read, and overridden by flags; errors that name
the model's fields and parameters (an unknown flag, a wrong count of values, a map of the wrong
size).

### Order of work

The reader and writer; `equip fit`; `equip crlb`; `equip montecarlo`; `equip design`;
`--config` and `--save`; the workload and the no-compile test; the README's command-line
section.

### Progress

- **The reader and writer** (`cli/src/io.jl`, 2026-11-01): `loaddata` (one 4-D file or one per
  observation; scaling applied; size and geometry checked; BIDS sidecars' EchoTime,
  InversionTime, RepetitionTime, FlipAngle; an FSL `.bval` beside the data; `precision =
  Float32` for half the memory), `loadmap`, `readvalues`, `savemaps` (the data's geometry, BIDS
  names, status, SSE, standard errors, a JSON sidecar per map). MATLAB `.mat` waits for MAT.jl.
- **`equip fit`** (`cli/src/args.jl`, `models.jl`, `fit.jl`): the flags as planned; `--model` and
  `--data` as flags too, so a config file can hold a whole command; `--save` writes the
  acquisition actually used (from flags, a protocol or the sidecars), so the file reruns the
  fit without the sidecars; a flag that differs from a sidecar's value wins, with a warning;
  `--help` lists the model's fields and parameters (generated from it), its examples in the
  model's own names; mistakes end with a message naming what to give, and exit code 1. Run as
  `cli/bin/equip` (`julia -t auto --project=cli -m EQUIPCLI`; the `@main` entry point). 59 tests,
  each fit against the Julia API's (the same maps, bit for bit or to Float32 rounding).
- **Start-up**: without a workload a command spent 17 s compiling (T2 decay, 60 voxels: 173
  methods), against 0.8 s to start Julia and load the package. EQUIPCLI's workload
  (`cli/src/precompile.jl`) runs `main` once per built-in model on a 2×2×1 dataset, plus a
  sidecar, a mask, `--stderror`, `--save`/`--config`, `--help` and a mistake: 2.7 s a command.
  The commands write into IOBuffers that `main` copies to `stdout` and `stderr` at the end, so
  they are compiled once, for IOBuffer, whatever `stdout` is (a terminal, a file or a pipe each
  compiled the command again): 1.7 s a command. Then three more causes, found with
  `--trace-compile-timing`: (1) `loaddata`'s `info` was a named tuple whose names were the
  acquisition values found, so a dataset with a sidecar was a new type and `savemaps` compiled
  again (549 ms): now a struct, `DataInfo`, with the values in a Dict; (2) `savemaps` and the
  commands' runner are `@nospecialize`d on the result and the streams (a `FitResult`'s type
  records the flags given, a mask or not: each combination compiled them again); (3) `julia -m`
  calls `main(ARGS)`, which dispatched to the keyword method meant for Julia callers, compiled
  for the real streams' types: the entry point is now `main` alone, and Julia callers use
  `equip(args; io, err)`. Now 0.95 s a command: 0.2 s to start Julia, about 0.65 s to load
  EQUIP, JSON, NIfTI and EQUIPCLI (their precompiled code), about 40 ms of small Base methods
  still compiled, and the work itself 2 ms (60 voxels; in a running session the whole command
  is 2.1 ms). Building EQUIPCLI takes about 35–45 s, once per change. (Warnings print when the
  command ends, with the rest.) Below 0.95 s needs the loading gone: a system image
  (PackageCompiler), or a server for many calls in a row (roadmap item 10). Still to do: the
  no-compile test.
- **`equip crlb`, `equip montecarlo`, `equip design`** (`cli/src/errors.jl`, `design.jl`): the
  truth by `--tissue` (several: `scenarios`, one per tissue) and `--truth NAME=value` (with
  several tissues, a one-value axis of `vary`: the same change in each, and a column in the
  table); the fit's flags as for `equip fit` (no maps); `--snr` (a list: a sweep) or `--sigma`,
  `--averages`, `--vary NAME=values` (repeated for several), `--weights`; tables as TSV
  (`--out`, the sweep's `rows`, full precision). montecarlo: `--noise` required, `--averaging`,
  `--n`, `--seed` (an `Xoshiro`, the same numbers as the Julia API with that seed), `--starts`,
  `--precision`. design: `--acquisition NAME=lower..upper` (decided: `..`, so a bound is not
  read as a `start:step:stop` range; Julia's interval notation), `--shared`, the scan time by
  `--readout` (Σ(TI + TD + readout)) or `--scantime`, a Julia expression of the model's fields,
  made a function at run time (`invokelatest`), `--constraint` likewise; Eq. 5's weights as
  `--weights PSR=1,R1f=0` (numbers are the fit's, per observation: told apart by the form, as in
  Julia by the type); `--out` the rounded protocol as a config file of its acquisition flags
  (`equip fit --config sir4.cnf data.nii.gz` fits with it; tested). On the 4-point scheme over
  the four tissues the search returns the paper's protocol (Eq. 5 0.243314 rounded against
  0.243395, 0.03% lower). The tissues go to `scenarios` as a list of names, not a named tuple
  of truths by tissue (whose names would be its type: each set of tissues compiled again).
  The workload covers the SIR schemes and T2 decay for these three (another model's crlb
  compiles its second derivatives at its first use); EQUIPCLI's package image grew to 47 MB
  and loading to about 1.2 s, so a command takes 1.4–2 s; a new combination of sweep axes still
  compiles EQUIP's table code once (0.5 s). 90 tests.

### Still to decide

1. (Decided 2026-11-01: NIfTI with BIDS sidecars, and MATLAB `.mat`; PAR/REC and the rest
   later.)
2. (Decided 2026-11-01: seconds only, as in the package; sidecars converted on reading.)
3. The bounds' syntax for `design` (`TI=0.01:10` reads as a range with two numbers).
4. The companion package's name, and how it is installed.
5. Python and MATLAB: after the release, as thin wrappers (proposed).

## Scope so far

In: the interface above (with `:scale`, `:data` and `BestOf` starts, `datastart` and `relabel`), seventeen models: relaxometry (T1IR, R1IR, T1IRTD,
R1IRTD, T1SR, R1SR, T1MFA, R1MFA, T2Decay, R2Decay, T2DecayOffset, R2DecayOffset, T2BiexpDecay, R2BiexpDecay), diffusion (ADCDecay, ADCKurtosis), qMT (QMTSIR), `fit` for one voxel or a dataset returning a
`FitResult` (with `fitted`, `residuals` and a summary display), Float64/Float32, bounds, masks, per-voxel maps, ties,
threads, tests.

Out for now; earlier implementations are on the `api-v0.2` and `main` branches:
CLI / Python / MATLAB, GPU, other model-specific and neighbour starting values, standard errors,
weights, other data layouts, more models. (Precompilation is back: a workload, see Performance
notes.)

## The prerelease

The first version to share is a prerelease, not a 1.0: its interface may still change. Its
scope (decided 2026-10-31):

- **In:** fitting (the model interface, the solver, datasets on threads, `refit`); `crlb` (the
  sandwich and Eq. 3, sweeps, fits side by side, general noise); `montecarlo`, with crlb's
  prediction alongside; protocol design (`scenarios`, `ProtocolDesign`, `optimize_protocol`);
  the simulation they need (`simulate`, the noise model, `TISSUES` and `nominal`, `compare`); the
  diagnostics already built (`cond`, `cor`, `leakage`).
- **Trimmed, for later:** digital phantoms (`phantom(labels, tissues)`, `labelshapes(dims)`):
  set aside, in git at d901143 (src/sim/phantom.jl and its test), to come back with the
  phantom work of item 2 (smooth variation, partial volume, B1, brain-like label images).
  `compare` stays: it judges any dataset fit, and its `labels` is any integer map.
- **To add before it:** fit statistics on `FitResult` (item 3; done, `stats-proto`); models: NODDI and SMT (item 7),
  and the pulsed qMT models of Yarnykh and of Ramani (item 6), which bring the bound pool's
  saturation (the pulse models, item 6) with them; a command line for each of the four parts
  and a data reader (decided 2026-11-01: "The command line and the data reader: the plan",
  below).
- **Then the release basics:** a license, CI running the tests, a version number, the
  repository's name (the package is `EQUIP`, the repository `voxelfit.jl`; registration needs
  `EQUIP.jl`), the precompile workload extended to crlb, montecarlo and a design (a first
  `ProtocolDesign` compiled for 8.8 s), an example script per part, a shorter README.
- **Kept exported:** crlb's pieces (`noisefree`, `bias`, `hessian`, `biasgradient`,
  `sandwich`, `biasedcrlb`), asked for at the REPL; unexporting them is open.

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
     `truth` (now `nominal`); `phantom(labels, tissues)` and `labelshapes(dims)`; `compare` (bias, SD, RMSE,
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
     MSE(θ̂ⱼ) = bⱼ² + [(I + ∇b) F⁻¹ (I + ∇b)ᵀ]ⱼⱼ, with bⱼ = E[θ̂ⱼ] − θⱼ and ∇b = ∂b/∂θ. b from
     the noise-free fit with the assumed value (Gauss–Newton from the truth), ∇b by the implicit
     function theorem, not by finite differences of fits.
   - Designed, with protocol optimization (Eq. 5 and its options), Monte Carlo, the diagnostics
     (Eq. 6 maps, leakage) and data-driven design: see "Simulation, CRLB and protocol design:
     proposed design".
   - Sensitivity to model assumptions: the bias from fitting with an assumed value that is
     wrong, the same machinery as Eq. 3's b. First case: QMTSIR's `Mzf_td0 = Mzm_td0 = 0` (both
     pools saturated at the start of TD). Simulate with nonzero values, fit with the default.
     A first check (5-point scheme, noise-free, 5% of M0 left): PSR +2%, R1f +4%, M0f −5%.
   - A sensitivity sweep in the shape of qMRLab's `SimVary` (vary one parameter's truth over a
     range): now the `vary` keyword of `montecarlo` and `crlb` (see the design section).
   - Validation targets: in the design section (Fopt5, Fopt4, Figs. 1 and 2, Li et al. 2010,
     Table 1).
   - Fit statistics on `FitResult`, added together as one set under StatsAPI's names, with one
     convention for the number of parameters (`dof`; packages differ on whether σ counts):
     `rss` (the SSE), `nobs`, `dof`, `dof_residual` (n − p), the residual SD
     σ̂ = √(SSE/(n − p)) per voxel (also a noise estimate, and the σ in the standard errors),
     `stderror`, `vcov` and the parameter correlations from JᵀJ, the "ended at a bound" flag,
     and `aic` / `bic` for choosing between models fitted to the same data (mono vs
     bi-exponential, ADCDecay vs ADCKurtosis). No new fields where a value follows from what is
     stored (σ̂, AIC, BIC come from `sse`, n and p). All assume Gaussian noise: biased for
     low-SNR magnitude data, documented once, next to the Rician-likelihood work.
     **Done** (`stats-proto`, src/fit/statistics.jl), as built:
     - *The values' shape:* numbers for one voxel; for a dataset, maps shaped like `r.params`
       (`stderror(res).T2` a map; `vcov` and `cor` an array of static P×P matrices, one per
       voxel). Not StatsAPI's vector for `stderror`: a named tuple by parameter, as `r.params`
       (crlb's `stderror` stays a vector: it has `c.sd` by name already).
     - *`dof` = P + 1* (the free parameters and σ, GLM's convention for a linear model), used
       by AIC/BIC; `dof_residual` = N − P; ℓ with σ at its maximum-likelihood value, rss/N.
     - *`σ`, the noise, given or estimated:* σ̂ = √(rss/(N − P)) per voxel by default; `σ = ...`
       (a number or a map) for a known noise, and the only way when N = P (4-point SIR: four
       observations, four free parameters, σ̂ undefined: NaN).
     - *A parameter on a bound* (`onbound`, the solver's 1e-9 rule, `_near`): held, as crlb
       holds it (its row of JᵀJ masked), SD NaN (not 0: the fit did not estimate it freely),
       the others as if it were fixed (tested against JᵀJ without its column). Undetermined
       (dependent columns, `_determined`): Inf. Unfitted voxels: NaN.
     - *How:* the fit's own setup again (`init` with the result's settings, maps included, in
       Float64 whatever the fit's precision), JᵀJ from the solver's `normal_equations` at each
       voxel's answer with that voxel's fixed values (`r.fixed`: for one voxel from a map, the
       value it used), σ²(JᵀJ)⁻¹ by Cholesky. 2000 voxels (T2) in 0.6 ms, serial. `_near` and
       `_determined` moved from errors/derivatives.jl to here (crlb uses them too; the fit
       layer comes first).
     - *Checked:* σ̂²(JᵀJ)⁻¹ with `jacobian(r)`; with σ given, crlb's SD at the fit's values
       and assumptions (T2; 4-point SIR at N = P); 2000 noisy copies, the median SE against
       the estimates' SD (within 1%) and the median σ̂ against σ; each voxel of a dataset
       (B1 map, mask) as its own one-voxel fit; a Float32 fit's statistics as the Float64
       fit's; AIC and BIC choose the bi-exponential for bi-exponential data and the mono for
       mono. 34 tests.
     - *Fixed after review* (`aicc-fix`): AICc's correction 2k(k + 1)/(N − k − 1) was used for
       any N, so at N < k + 1 it was negative (the 4-point SIR, N = 4, k = 5: −30, rewarding
       the parameters it should penalize); and at N = P the log-likelihood took the log of a
       rounding-level SSE, making AIC, AICc and BIC huge and meaningless. Now NaN: all four
       when N ≤ P (as `residualsd`), AICc also when N ≤ k + 1. Tested at the boundaries (T2
       with 3, 4 and 5 echoes; the 4-point SIR; a dataset). 43 tests.
   - **Weighted fits** (diagonal W built, 2026-10: "Weighted fits: the plan" above; what
     remains is a full W and per-voxel weight maps): minimize ½(g − y)ᵀW(g − y) with W = Σ⁻¹, from the noise as crlb takes
     it (`σ` per observation: a weight per row; a full `Σ`: rows whitened by L⁻¹, as
     `_FullNoise` does for crlb). Why: with unequal σ (different averages or bandwidths, σ
     rising over the echoes) the unweighted fit is not efficient; its SD is the sandwich
     H⁻¹J₀ᵀΣJ₀H⁻¹, above the bound (JᵀΣ⁻¹J)⁻¹ a weighted fit reaches (crlb's tests: σ 5 → 20
     over 32 echoes, T2). How: the solver sums JᵀWJ and JᵀWr row by row (one multiply per row
     for a diagonal W); the weights are part of the fit, so of a `Scenario`, and crlb, Eq. 3 and
     leakage follow: H = Σᵢ wᵢ(JᵢJᵢᵀ + rᵢ∇²gᵢ), meat J₀ᵀWΣWJ₀ (= J₀ᵀWJ₀ when W = Σ⁻¹, so the
     sandwich at the truth is F⁻¹), ∇b = H⁻¹J₀ᵀWJ_t − I, dθ̂/dc = −H_θθ⁻¹H_θc with the weighted
     H. Per-voxel σ maps later. With `averages` (2026-10), weights from them (wⱼ = NEXⱼ, so
     W = Σ⁻¹ up to a scale); the magnitude-averaging floor biases most the points weighted
     most, so montecarlo with both shows the trade of SD against bias.
   - **Iteratively reweighted least squares (IRWLS)**: the weights recomputed from the current
     fit, then fitted again, until they settle (a few rounds, an outer loop around the weighted
     fit of each voxel). Uses: robust fits against outliers (Huber or Tukey's bisquare weights
     from the residuals, scaled by a robust σ such as the MAD), noise whose variance depends on
     the signal (e.g. weights ∝ S² for log-linear fits, as diffusion's WLLS/IWLLS, Veraart et
     al., NeuroImage 2013), and one route to the Rician-likelihood fitting of item 12. Built on
     weighted fits. To settle with it: the weight-function interface, the stopping rule, and
     crlb with the converged weights (the weights' own dependence on θ ignored, or kept, for
     M-estimators, through the sandwich's ψ′ terms).
   - **Leakage and crlb, faster for the protocol search**: when profiling shows the search
     needs it (e.g. once a robust-design term uses leakage × the plausible error of each fixed
     parameter). Two changes, in this order of value:
     - *The 4×8 nested duals*: the outer layer seeded with the P free parameters only, the
       inner with all P + K, so one evaluation gives ∂/∂θ of the full gradient: the H_θθ and
       H_θc blocks exactly, without the K×K block of the fixed parameters or the symmetric
       repeats (32 of 64 second derivatives for QMTSIR, 4 + 4). Estimated to halve the
       biased-point leakage kernel (16 → about 8 µs, 5-point).
     - *One combined kernel*: crlb and leakage each run their own noise-free fit and Hessian,
       but leakage's joint Hessian holds crlb's H as its θθ block and its buffer J₀'s rows. One
       noise-free fit, one joint Hessian, one Cholesky factor of H_θθ and one evaluation at the
       truth (J_t) give the covariance (sandwich or Eq. 3), ∇b and the leakage together, about
       half the cost of the two. The bias gradient needs nothing new: crlb's kernel already
       computes ∇b = H⁻¹J₀ᵀJ_t − I (Eq. 3; `biasgradient(m, s)`).
   - **An inner maximization: the worst case inside the box, not only at its points.** A set's
     minimax is over its points (corners, midpoint, a grid), and a nonlinear model's worst
     case can lie between them. After a search, at the protocol found: maximize the
     scenario's value over the box of true values (a bounded local search from the worst
     point of the set, or several), and if it finds a worse one, add it as a scenario and
     search again, until nothing worse is found (the exchange method of semi-infinite
     minimax problems; with the active set below, the same loop). The check alone, without
     the re-search, already says whether the set's answer is the box's.
   - **An active set for minimax designs** (when a design has thousands of scenarios, or a
     model's kernel is much slower than SIR's): with `aggregate = maximum` only the scenarios
     near the worst set the value, so evaluate those (the set A, e.g. within 50% of the worst)
     for most candidate protocols. The maximum over A is a lower bound on the true one; every
     scenario is checked for the candidates the search accepts, every few generations and for
     the final answer, and one that has risen near the top joins A (the exchange method of
     minimax and semi-infinite problems, as in Remez's algorithm). Measured on the 3⁵ grid of
     true values (243 scenarios, kmf assumed 12.5): within 10% of the worst, 6 at Table 2's
     4-point scheme, 3 at the scheme moved slightly, 6 at one far from it; within 50%, 45, 27
     and 63. So about 30–60 kernel calls an evaluation instead of 243, 4–8× fewer. The full
     checks are essential: the worst scenario changed between those protocols (kmf 5, then
     20, then R1f 0.5), and a set fixed once would let the search prefer a protocol that is
     good on A and bad elsewhere. For `maximum` only (a mean needs every scenario). Not needed
     now: 243 scenarios cost about 30 s a start.
   - **Sensitivity-efficiency plots** (Dortch 2018, Fig. 2A): the figure is what has value
     (the numbers are `jacobian` ./ √t: "Diagnostics", Sensitivity efficiency). A function,
     through a plotting package extension (Makie or Plots), so that EQUIP itself does not
     depend on a plotting package: for a model, a truth, two acquisition variables over ranges
     (log axes; the others held) and the time per observation, the map of |θ·∂S/∂θ|/√t of each
     parameter, each contoured at a fraction of its largest value (the paper's script: 85%),
     with schemes' observations overlaid (`SIR_4POINT`, `SIR_5POINT`, an optimizer's result);
     and the one-variable view (efficiency against TI at one TD, the script's figure 4). To
     settle with it: the name; how the two variables and the held ones are given (any model,
     not only TI × TD); one panel per parameter or the contours overlaid, as the paper. Its
     acceptance tests: the measurements recorded under "Sensitivity efficiency" (the script's
     central differences, the schemes against the grid's best, T2's analytic maximum).
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
     model. The lean rewrite (9a6e173) removed data starts and kept `Free(start = :scale)`. The 0.2
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
   - Several linear parameters solved together (biexponential amplitudes, an offset): see
     variable projection, item 12 (prototype measured).
   - **A far start can end in a corner, reported as converged.** QMTSIR (5-point scheme, SNR 50,
     2000 voxels) from PSR = 0.4, R1f = 3: every fit ends with PSR at its bound 0 and Sf at its
     bound −1.2, R1f ≈ 4, at about 340× the SSE of the fit from the default starts (median),
     with both the full fit and variable projection. The default starts avoid it. Argues for
     the "ended at a bound" flag (item 3), and perhaps a `BestOf` start for QMTSIR (on PSR or
     R1f), to test in the start-strategy study.
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
5. **MT ratio, MTsat, B1 and B0 maps**: mostly closed-form estimators (a formula on as many
   images as unknowns), brought in as ordinary models.
   - **Approach: models whose data start is the exact answer.** Each method is a normal model
     (acquisition struct, `parameters`, `signal!`) whose `datastart` returns the closed-form
     solution. With as many images as parameters the start already fits exactly, and LM stops after
     0–1 iterations (the cost, about one signal evaluation, to measure); with extra images
     (repeats, more MT offsets) the same model becomes a least-squares fit started from the closed
     form. Masks, threads, maps, `FitResult`, B1 as a fixed map, and (because `signal!` exists)
     `simulate`, Monte Carlo and CRLB come for free. A solver-free path only if measurement shows
     it matters. Names follow the convention, quantity first then method (e.g. `B1DAM`, `B1AFI`).
   - **MTR**: images with and without the MT pulse (e.g. `MTR(mt = [false, true])`). Parameters M0
     (`:scale`) and MTR; signal M0 for the reference images, M0·(1 − MTR) for the MT ones; data
     start (S_ref − S_MT)/S_ref. qMRLab exports MTR only when the MT- and PD-weighted TRs match;
     the model can check that.
   - **MTsat** (Helms et al., MRM 2008), with the inputs of qMRLab's `mt_sat`: MT-, T1- and
     PD-weighted SPGR images, a flip angle and TR per image, a B1 map and a mask. Parameters the
     apparent amplitude A (`:scale`), R1 (or T1) and MTsat; signal the SPGR equation with the MT
     image's extra saturation; data start Helms' closed form.
     - B1 as a fixed map (`B1 = b1map`) scales every flip angle, as in T1MFA, which corrects A and
       R1. MTsat keeps a residual B1 dependence, which qMRLab corrects empirically after the fit
       (a factor of 0.4 by default; Weiskopf et al., Front Neurosci 2013; Helms, ISMRM 2015,
       abstract 3360): not part of the signal model, so a derived output (below).
     - Helms' formulas use the small-flip-angle approximation; `signal!` could use it or the exact
       SPGR equation, with the closed form as the start. To decide.
   - **B1 maps**: the double-angle method (DAM: α and 2α, long TR) and actual flip-angle imaging
     (AFI: two TRs) are closed form from magnitude images, the same pattern. A multi-flip-angle B1
     fit is an ordinary model. Bloch–Siegert needs phase data: later.
   - **B0 maps**: need phase or complex data (EQUIP takes real-valued arrays now). A dual-echo phase
     difference, ΔB0 = Δφ/(2π·ΔTE), is closed form per voxel when ΔTE is short enough that the
     phase does not wrap. Multi-echo B0 needs phase unwrapping across neighbouring voxels, which is
     not voxel-wise: use an existing Julia package (ROMEO.jl, MriResearchTools.jl, as I recall)
     through a package extension, with the I/O layer (item 9).
   - **Using the maps downstream**: a map is a fixed parameter, and a result's map passes straight
     in: `b1 = fit(B1DAM(FA = 60), b1data; mask).params.B1`, then
     `fit(MTsat(...), data; mask, B1 = b1)`. B0 later the same way (the effective MT offset in
     qMT-SPGR, bSSFP, the through-slice sinc model of item 12). Registering or resampling a map to
     the data's grid stays outside EQUIP; the I/O layer can check that sizes match.
   - **Needed in EQUIP**:
     1. **Derived outputs**: a hook computing extra maps per voxel after the fit, from the fitted
        and fixed values: MTsat's empirical B1 correction, MTR from the MTsat images, R1 alongside
        T1, QMTSIR's bound pool fraction BPF = PSR/(1 + PSR). (`relabel` sees only the fitted
        values, so it cannot.)
     2. The statistics of item 3 for as many images as parameters: zero residual degrees of
        freedom, so σ̂ and the standard errors are NaN, not nonsense.
     3. Complex or phase data, for B0 and Bloch–Siegert.
   - Order: MTR and MTsat, with derived outputs; then the B1 maps (DAM, AFI, the multi-angle fit);
     then B0, with complex data and the I/O layer.
6. **More qMT models, from approximate to exact.** Here the cost per voxel is in the signal
   (lineshape integrals, small matrix exponentials, steady states), so EQUIP's design (no
   allocation, static arrays, exact derivatives from dual numbers, threads) matters most; the exact
   models are where its speed may help most.
   - **Pulse models: the conditions behind the fixed parameters.** For SIR, Sf and Sm are what
     the inversion pulse does to each pool, so B1, B0 and T2m (the conditions a robust design
     varies) map to them: the free pool rotated segment by segment (B1 scaling the flip angle,
     the B0 offset; relaxation neglected over the pulse) for Sf; the bound pool saturated,
     Sm = exp(−Σₖ π·ω₁,ₖ²·g(Δ)·tₖ), with a lineshape g: Gaussian at resonance,
     g(0) = T2m/√(2π), as `calcSm_blockPulse.m` and `calcSm_compPulse.m` (Dortch 2018; the
     1-ms block gives Sm = 0.831 at T2m = 15 µs, the paper's 0.83), or super-Lorentzian for
     tissue (it diverges on resonance: a cutoff). Pulses: block, the composite 90x-180y-90x,
     shaped ones given as segments (`calcSm_compPulse.m`'s 64 flip angles and phases). The
     same bound-pool saturation is the core of the pulsed qMT models below (the MT pulse's
     saturation in SPGR and bSSFP), so built with them, generic, not as an SIR special case.
     Then designs robust to B1 × B0 × T2m are scenarios as any other (`Scenario`s whose true
     Sf and Sm come from the conditions). Measured in an earlier prototype ("Conditions" in the
     protocol-design section): a 3-point scheme optimized for the nominal case had a worst PSR
     error of about 50% over B1 ±10%, B0 ±50 Hz and T2m 10–20 µs, against 20% for Table 2's
     4-point; optimized over those conditions, 18.3%.
   - **Henkelman's continuous-wave steady state** (Henkelman et al., MRM 1993): the two-pool
     Bloch–McConnell equations under constant off-resonance saturation, in closed form. The
     reference model, and the building block of the pulsed approximations.
   - **Pulsed MT with SPGR**, the models of qMRLab's `qmt_spgr` (inputs: the MT data, R1, B1 and
     B0 maps, a mask):
     - Sled & Pike (J Magn Reson 2000): rectangular-pulse (`SledPikeRP`) and continuous-wave
       (`SledPikeCW`) approximations;
     - Ramani et al. (Magn Reson Imaging 2002): the continuous-wave power equivalent of the MT
       pulse;
     - Yarnykh (MRM 2002; Yarnykh & Yuan, NeuroImage 2004), and the single-point macromolecular
       proton fraction (Yarnykh, MRM 2012): one MT image plus R1, B1 and B0 maps, the other
       parameters constrained, so one parameter per voxel (the "exact data start" pattern of
       item 5).
     - Also in qMRLab: qMT with bSSFP (Gloor et al.), a later candidate.
   - **MAMT** (Stanisz group): the MT pulse cut into pieces of constant RF amplitude, each solved
     exactly (the constant-amplitude solution) and propagated through the sequence to the steady
     state. (Reference to confirm; as I recall, Portnoy & Stanisz, MRM 2007, compared models of
     pulsed MT.)
   - **The full coupled equations, without approximations**: the two-pool Bloch–McConnell
     equations through the actual sequence (shaped MT pulses, excitations, spoiling) to the steady
     state, by piecewise-constant propagators (static matrix exponentials) or an ODE solver. The
     steady state needs no simulation over many TRs: it is the fixed point of one TR's affine map,
     M = (I − A)⁻¹ b. Hundreds of small matrix exponentials per MT offset and evaluation, with
     dual numbers through them for exact derivatives: a per-voxel fit then costs milliseconds,
     so speed decides whether whole-brain maps are practical (measure: per-voxel cost, threads,
     and the precompute/dictionary ideas of item 4). Related, as I recall: Assländer's
     generalized Bloch model for the bound pool under short pulses (MRM 2022) and his Julia
     package MRIgeneralizedBloch.jl, a reference to compare against or build on.
   - **Shared pieces**, written once: bound-pool lineshapes (Gaussian, Lorentzian,
     super-Lorentzian; the super-Lorentzian's integral needs care near the magic angle, and a
     fixed quadrature keeps it differentiable when the bound pool's T2 is fitted); MT pulse
     shapes (Gaussian, sinc, Fermi, hard) and their power integrals; two-pool steady-state
     solvers on static matrices.
   - **Per-voxel inputs are fixed maps**, already supported: B1 (scales the MT and excitation
     pulses), B0 (shifts the MT offsets), an observed R1. The usual constraint of R1f from the
     observed R1 (qMRLab's "use R1map to constrain R1f", from Sled & Pike) is a tie by function,
     with the R1 map as a fixed parameter the user gives: `R1f = p -> f(p.R1obs, ...)` (see
     Parameter roles).
   - **Validation**: against qMRLab's `qmt_spgr` on its example data (a `bench/` study), and the
     exact model as the reference for the approximations' bias (short TR, high MT power, short
     pulses).
7. **Diffusion microstructure: NODDI and SMT**, many small fits on multi-shell data.
   - **SMT**, the spherical mean technique (Kaden et al., MRM 2016; NeuroImage 2016): averaging
     the signal over directions within each shell removes the fibre orientation distribution;
     the microscopic model (per-axon parallel and perpendicular diffusivities; multi-compartment
     SMT with an intra-axonal stick and an extra-axonal compartment) is fitted to a few shell
     averages per voxel. Small, fast fits: a natural fit for EQUIP. It needs `erf` of dual
     numbers (SpecialFunctions with ForwardDiff provides it). The averaging per shell is a
     preprocessing step (needs the gradient directions, from the I/O layer), or the user gives
     shell averages.
   - **NODDI** (Zhang et al., NeuroImage 2012): intra-neurite sticks with Watson orientation
     dispersion, an anisotropic extra-neurite compartment with tortuosity, isotropic CSF.
     - Needs the gradient direction of each observation: an acquisition field of 3-vectors
       (`nobs` already handles any vector field; converting to the working precision needs a
       method for vectors of static vectors), and the special functions of the Watson integrals,
       working with dual numbers.
     - Known difficulties: local minima (`BestOf` on several parameters, item 4) and diffusivities
       usually fixed (d∥, d_iso: fixed parameters, already supported).
     - The volume fractions are nearly linear, so variable projection applies in part
       (tortuosity ties the extra-neurite signal to the intra-neurite fraction); AMICO's
       linearization (Daducci et al., NeuroImage 2015) is the dictionary idea of item 4.
   - DTI, DKI with directions and IVIM stay under item 12.
8. **Custom models, and coming from other fitters.** EQUIP's model interface (a struct,
   `parameters`, `signal!`) is short, but heavier than LsqFit's `model(x, p)` or MATLAB's
   `lsqcurvefit`, and it makes side-by-side comparisons awkward.
   - **`@mrimodel`**: an ordinary Julia function of one observation, with the macro in front,
     becomes a model:
     ```julia
     @mrimodel MyT2(TE; S0, T2, C = 0.0) = S0 * exp(-TE / T2) + C

     @mrimodel function MySIR(TI, TD, magnitude = true;
                              PSR = Free(start = 0.1, lower = 0, upper = 1), ...,
                              kmf = 12.5, R1m = Fixed(:R1f))
         ...                                   # a multi-line body; the last line is the signal
     end
     fit(MySIR(TI = ti, TD = td), data; mask, kmf = Free())
     ```
     - The function's name becomes the model type; its positional arguments the acquisition
       (in the body, one observation's values); positional arguments with a default, options;
       its keywords the parameters, in order, their defaults the default roles: `Free(...)`;
       a plain number or `Fixed(...)` for fixed; no default for free with no start, given in
       the call (LsqFit's way, `fit(m, y, p0)`, or `T2 = Free(start = ...)`). A docstring before
       the macro goes on the type.
     - A default may use the acquisition (`T2 = Free(start = mean(TE))`, `TE` then the whole
       vector), as Julia's own keyword defaults may use earlier arguments.
     - It generates the struct, `parameters`, and a `signal!` that calls the body as a plain
       positional function for each observation: as fast as hand-written code, no allocation;
       `@macroexpand` shows the result.
     - Not in a first version: a whole-curve body (work shared across observations, like
       `QMTSIR`'s eigenvalues: write `signal!` by hand) and hooks (`datastart`, `relabel`:
       ordinary methods on the generated type, added afterwards). Not named `@model`, which
       Turing.jl exports.
     - Tests: macro-made versions give the same fits as `T2DecayOffset` and `QMTSIR`;
       docstrings; zero allocations per fit.
   - **`newmodel(...)`**, a template generator: writes a commented model file (struct,
     `parameters`, `signal!` with a TODO, optional hooks, a test snippet) to edit, for models
     that will grow, and as the route from a custom model to a built-in one.
   - **The LsqFit bridge**: `fit(m, y, p0::AbstractVector)` (starts in parameter order, as in
     `curve_fit(model, x, y, p0)`) and `coef(res)`, StatsAPI's name; with item 3's `stderror`,
     `vcov`, `dof` and `rss`, comparison code is identical for LsqFit's and EQUIP's results. A
     bare `fit(f::Function, x, y, p0)` would be type piracy on StatsAPI's `fit` (neither the
     function nor the argument types are EQUIP's), so the function is always wrapped in an EQUIP
     type first.
   - Optionally, **`CurveModel`**: a model wrapping an existing function `f(x, p)` (LsqFit's
     form; dotted, element-wise code works per observation unchanged), declared with keywords or
     with qMRLab's and lsqcurvefit's vectors (`names`, `start`, `lower`, `upper`, `fixed` for
     `xnames`, `st`, `lb`, `ub`, `fx`). The function gets the full parameters in declared order
     (`p[1]` or `p.S0`), as qMRLab's `equation(obj, x)` does after `addfix`. Probably covered by
     `@mrimodel`: decide after it.
   - **Docs**: "Coming from LsqFit, lsqcurvefit or qMRLab", the same models side by side,
     including a qMRLab model ported as-is (`Model.fx(3) = 1; Model.st(3) = 15` →
     `fit(m, data; kmf = 15.0)`). **Decided**: not to rework the core to mirror qMRLab (five
     parallel vectors kept in sync by position; fit settings changed in place on the model
     object, unsafe across threads and easy to forget; maps and ties special-cased per model),
     but to mirror its vocabulary and tools at the surface.
   - **Closing over values**, to document: a value captured in the function is a hidden constant
     (fine if it never changes; it cannot be freed, mapped, or simulated with another value); a
     declared parameter can. Measured (a million evaluations): captured inside a function, a
     `const` global, or through `let`: 4.9–6.3 ms, no allocation; a plain global, or a variable
     reassigned after the closure captures it (Julia "boxes" it): 51–60 ms and 46 MB, 10× slower.
   - **Small usability fixes**: (done: roles print as written, by name,
     `Free(start = 0.1, lower = 0.0, upper = 1.0)` and `Fixed(12.5)`); (done) a REPL summary of a
     model (its acquisition, and a table of its parameters: free or fixed, start, bounds, ties,
     the value `nominal(m)` takes);
     a clear error when `signal(m, p)` gets a map ("one voxel: give numbers; for maps, use
     `simulate`"; today a `MethodError` on `float`), and a docstring note that values are not
     checked against bounds; a public name for `_allparams` (done: `resolve`), for loops calling
     `signal!`; in "Writing a model", that a model whose
     type parameters cannot be rebuilt from its fields needs a precision-conversion method.
9. **Data I/O** (the release's part: "The command line and the data reader: the plan", above):
   make getting data in and maps out as easy as possible. An `io/` layer
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
10. **Front ends: command line, Python, MATLAB**, built on the `io/` layer (item 9): data files
   and a protocol file in, map files out. (The release's command line: flags first, a config
   file of flags; "The command line and the data reader: the plan", above. The TOML protocol
   file below was superseded by it.)
   - **Why start-up needs a design.** Every command-line call is a new Julia process, and so is
     every new Python or MATLAB session, so whatever is not already compiled compiles again
     each time. Measured (4-point SIR, default roles, 4 threads), a fresh process that loads
     EQUIP and fits 1 M voxels: 0.9 s to start Julia and load EQUIP, then 1.1 s to fit, and
     nothing compiles, because the case is in the precompile workload. So a call costs about
     1 s on top of the fit, and the speed advantage stays, as long as (a) a call fits a whole
     dataset (not a voxel or a slice per call), and (b) everything the call runs is
     precompiled.
   - **The front ends are the easy case.** They offer the built-in models and a fixed menu of
     options (which parameters are free or fixed, numbers or maps, precision, mask, file
     formats), so the cases they reach are known in advance and can all be precompiled. Only
     custom models compile at run time, and the people writing them work in Julia.
   - **1. The workload covers every front-end call**: each built-in model with each option the
     front ends offer, and file reading and writing. Also the simulation path, which today
     compiles about 3.5 s in `examples/sir_wm_4point.jl` (measured, first calls in a fresh
     process): `compare` 2.1 s (of which `median` alone 0.93 s: Julia's sorting code, not
     shipped compiled for `Vector{Float64}` by Statistics), printing its table 0.73 s,
     `simulate` 0.26 s (the preset's full parameter set has 6 names where the workload's had 4,
     and a `MersenneTwister` is a different type from the default generator), `truth` (now `nominal`) 0.21 s,
     and a fixed parameter given as a number (`kmf = 10.0`) 0.47 s. Code from Statistics and
     Base called by the workload is saved with EQUIP too.
   - **2. A test that nothing compiles**: a fresh process runs each front-end call with
     `--trace-compile` and fails if anything compiled, so a change that silently breaks
     precompilation is caught by the tests, not by users.
   - **3. A prep step for cases outside the workload: `equip --init <the call's arguments>`.**
     It runs the real call's code on 2 stand-in voxels of the same shape and types and saves the
     compiled code on disk; the same call without `--init` then starts compiled, in any later
     process, across reboots. How: a local "prep" package in the user's depot (PrecompileTools'
     "startup package" pattern) whose workload runs the cases listed in a `cases.toml`;
     `include_dependency` on that file makes Julia rebuild the package when a case is added;
     `--init` adds the case and loads the package (which rebuilds it); every call loads it
     with EQUIP. The core of the prototype:

     ```julia
     module EQUIPWarm
     using EQUIP, PrecompileTools, TOML
     const CASES = joinpath(@__DIR__, "..", "cases.toml")   # must exist, even empty
     include_dependency(CASES)       # editing cases.toml makes Julia rebuild this package
     @setup_workload begin
         cases = get(TOML.parsefile(CASES), "case", [])
         @compile_workload for c in cases
             # build the model and options from c, then fit 2 stand-in voxels:
             # fit(m, rand(2, nobs(m)); precision = ..., fixed...)
         end
     end
     end
     ```

     Measured with the prototype (T1IR, 10 000 voxels, the first fit in a fresh process):

     | case (not in EQUIP's workload) | before prep | after prep |
     |---|---|---|
     | `precision = Float32` | 3.2–3.8 s | 0.06 s |
     | `Sf = -0.95` fixed | 0.6–0.7 s | 0.1 s |

     The prep step took 9.9 s for both cases; loading stayed at 0.6 s. Points for the design:
     - A case is a set of types, not values: the model, which parameters are free, numbers or
       maps, precision, data dimensions, mask or not, file format. Other TIs or another fixed
       value are the same case; a B1 map instead of a number is a new one. So `--init` must
       run the real call's own code on stand-in data, never a hand-written copy of it.
     - Adding a case rebuilds the whole prep package (seconds per case). An EQUIP update makes
       Julia rebuild it once, at the next call, automatically (it depends on EQUIP).
     - A lab can keep its cases in a file (`equip --init cases.toml`). Python and MATLAB load
       the same prep package (`equip.prepare(...)`).
   - **4. A server, for many calls in a row** (a script calling the command line per subject,
     MATLAB): a Julia process that stays running; each call sends its arguments and file paths,
     so it also skips the 0.9 s of start-up and loading, and whatever compiled once stays
     compiled while the server runs (lost when it stops; the prep package survives). Existing
     tools: DaemonMode.jl (command line), MATDaemon.jl (`jlcall` from MATLAB, used by DECAES.jl
     for its MATLAB interface).
   - **Per front end:**
     - Command line: one call per dataset or per list of subjects; `--init` for other cases;
       the server for many calls; a system image or app (PackageCompiler.jl) if start-up still
       matters (not measured).
     - Python: juliacall (PythonCall.jl) runs Julia inside the Python process. It starts on
       `import`, and later calls cost only the fit, so a notebook stays fast all session.
     - MATLAB: running Julia inside MATLAB is fragile, so a server; data passed in files. qMRLab
       could call EQUIP as its fitting engine this way.
   - **Order**: workload coverage and the no-compile test; the command line with `--init`;
     Python; MATLAB.
   - **Faster start-up for the release's command line, later (decided 2026-11-01: a lean
     command line first).** Measured with `equip fit` (T2, 60 voxels): 0.95 s a command, of
     which 0.2 s starts Julia, about 0.65 s loads EQUIP, JSON, NIfTI and EQUIPCLI (their
     precompiled code), about 40 ms compiles small Base methods, and 2 ms is the work (the whole
     command in a running session: 2.1 ms). Below 0.95 s the loading has to go:
     - *A system image* (PackageCompiler.jl, `create_sysimage` with EQUIPCLI and its workload):
       expected about 0.2–0.3 s a command, no process left running; a build of several minutes,
       a few hundred MB, rebuilt after each update. To measure first.
     - *A service* (a server and a thin client): one Julia process, started by the first command
       (or `equip server start`), keeps EQUIP loaded and waits on a local socket; `equip`
       becomes a small client (bash's `/dev/tcp`, no Julia) that sends the arguments and the
       working directory, prints what comes back and exits with its code. Milliseconds a call,
       and what compiles (a case outside the workload) compiles once for the server's life.
       Requests one at a time (a fit uses every thread); paths resolved against the client's
       directory; the server stops after an idle time and restarts when EQUIP is updated (the
       client checks the version); the client runs the command itself if the server is down. A
       server that runs commands must be reachable by its user alone: a socket file only they
       can open, or a random token kept in a private file. DaemonMode.jl does this for Julia
       scripts, but its client is Julia (0.2–0.3 s a call). The way for scripts calling `equip`
       per subject, and for MATLAB and Python (item 10's point 4).
     - *`juliac`* (Julia ≥ 1.12, experimental): a small binary, tens of ms to start, but it needs
       code without dynamic dispatch (the command line picks the model by name at run time).
     - *`--init`* (point 3 above): for cases outside the workload, which today compile at the
       first call of each process.
   - **Calling EQUIP from Python and MATLAB (2026-11-01, after the command line).**
     - *Python, in the same process*: `juliacall` (PythonCall.jl) starts Julia inside Python
       (`from juliacall import Main as jl; jl.seval("using EQUIP")`): 1–2 s once a session,
       then each call costs only its work, so a notebook stays fast. NumPy arrays pass to Julia
       without a copy and Julia's come back as NumPy arrays, but the two store arrays in
       opposite orders (row- and column-major): a (x, y, z, observations) image arrives with its
       dimensions reversed, so a thin wrapper puts the observations last, with one copy at most
       (`np.asfortranarray`). Julia's threads are set before the import (an environment
       variable). Not measured here yet (needs `juliacall` and its Julia packages installed).
     - *MATLAB, through a Julia server*: loading Julia into MATLAB's process (a MEX file and
       Julia's C embedding) is fragile: both bring their own LLVM, BLAS and C++ runtime, and
       their own handling of signals and threads. So a Julia process started once a MATLAB
       session keeps EQUIP loaded, and each call passes the arguments and the data (a `.mat`
       file in a temporary folder) and reads the results back: milliseconds plus the transfer,
       small next to a fit. MATDaemon.jl (`jlcall`) does this, and DECAES.jl's MATLAB interface
       uses it; it is how qMRLab could use EQUIP as its fitting engine. The same server as the
       command line's service (above), so one piece of work serves both. Possible but untested:
       MATLAB → Python (MATLAB runs Python in-process) → `juliacall`, with more copies and one
       more layer to go wrong.
   - **A one-line installer, for people without Julia (2026-11-01).** In order of effort:
     1. *A script that installs Julia and then EQUIP*: `curl -fsSL
        https://github.com/dortchrd/EQUIP.jl/releases/latest/download/install.sh | sh` (and a
        PowerShell one-liner for Windows): installs Julia with juliaup if it is missing
        (non-interactively), installs EQUIPCLI as a Julia app (`equip` on the PATH, Julia ≥ 1.12),
        runs `equip --help`. About 40 lines; the first install precompiles for a few minutes,
        Julia is about 500 MB, and a command starts in 1–2 s, as now. Possible for the first
        release.
     2. *Prebuilt bundles, no Julia needed*: PackageCompiler's `create_app` (an `equip`
        executable with its own Julia runtime and EQUIP compiled in), built by CI for each
        platform (Linux x86_64, macOS Intel and Apple silicon, Windows) at each release and
        attached to it; the same `curl … | sh` picks the archive for the system and unpacks it to
        `~/.equip`. No Julia to install and nothing to precompile, and it is the system image's
        start-up too (about 0.2–0.3 s a command, expected). Archives of about 250–400 MB; no
        custom models (those are written in Julia).
     3. *A container* (Docker, and Apptainer/Singularity on clusters), as fMRIPrep, QSIPrep and
        MRtrix offer: straightforward once the bundle exists.
     First, either way: where the packages come from (registered in General, EQUIPCLI from the
     repository's `cli/` folder, or installed from GitHub directly; the CLI's `[sources]` path
     to EQUIP then points at the repository), and a short, readable script served over HTTPS
     with checksums for what it downloads. Proposed: (1) for the first release, then (2), with
     the container alongside.
11. **Solver backends** (LsqFit, SciML), **and problems too large for static arrays.**
   - **Why.** EQUIP's own solver is narrow on purpose: static arrays and dual numbers, no
     allocation, for millions of tiny problems (3.5–19× faster than LsqFit on the same problems,
     see `src/fit/levmar.jl`). That leaves gaps: models that cannot take dual numbers (external
     code, lookup tables, some ODE solvers), many free parameters (more than ~30–40), other
     algorithms or objectives, and checking EQUIP against a reference solver.
   - **The hook.** `solver` is already a value (`solver = LM()`), so the per-voxel solve can
     dispatch on its type: `_fit_voxel` calls `solve_voxel(solver, prob, θ0, lo, hi, cache)`,
     which for `LM` is today's `levmar`. Backends live in package extensions, so EQUIP depends on
     none of them: e.g. `solver = LsqFitLM()` (LsqFit) or `solver = SciMLLeastSquares(alg)`
     (NonlinearSolve). Models, roles, starts, masks, maps, threads, `FitResult` and simulation
     do not change. `init` builds each solver's own state (for `LM`, its options and the dual
     buffer), reused across voxels where the backend allows.
   - **Jacobians come with the backends**, so EQUIP's own solver keeps dual numbers only: LsqFit
     offers finite differences, forward AD or a user-supplied Jacobian; SciML chooses through
     ADTypes (`AutoFiniteDiff()`, `AutoForwardDiff()`, ...). Measured, one signal evaluation:
     with dual numbers (the value and all P derivatives in one pass) it costs 1.24× a plain
     evaluation for T2Decay, 1.30× T1MFA, 1.63× T1IR, 1.92× the biexponential, 1.93–2.07×
     QMTSIR (5- and 16-point). Forward finite differences need P + 1 plain evaluations, 2–2.6×
     the dual cost, and are less accurate. A perfect analytical Jacobian could save at most
     the gap between dual and plain (about 30–45% of the solver's time for the biexponential and
     QMTSIR, 10–20% for T2Decay and T1MFA), at the price of hand-derived derivatives for every
     model and form, right for every combination of free, fixed and tied parameters; variable
     projection (fewer partials) and better stopping rules gain as much more cheaply. The
     exception to measure: the exact qMT models (item 6), where dual numbers through many small
     matrix exponentials may cost far more, and the derivative of a matrix exponential has a
     closed form (Van Loan's block-matrix trick).
   - **First uses**: LsqFit as the reference solver in the benchmark comparison (item 2); models
     that cannot take dual numbers; models with many parameters (a backend may make the
     ordinary-array method below unnecessary).
   - **To check**: bounds (LsqFit's Levenberg–Marquardt takes them; SciML's NonlinearSolve
     least-squares solvers mostly do not, as I recall, and Optimization.jl may serve there);
     mapping each backend's outcome to `FitStatus`; backends allocate and are slower, which is
     fine for these uses.
   - **Ordinary arrays in EQUIP's own solver**, if no backend serves (only when a model needs more
     than ~30–40 free parameters): a second method of the same function, chosen by dispatch on
     the argument types: `levmar(prob, θ0::Vector, lo::Vector, hi::Vector, opts,
     work::LevmarWorkspace)`, with a workspace allocated once per thread (JᵀJ, Jᵀr, step,
     Cholesky factor) and in-place `mul!`/`cholesky!`/`ldiv!`. `setup` picks static or ordinary
     arrays from the number of free parameters, once per fit (a threshold around 32), so every
     voxel dispatches to the right method at no run-time cost. Derivatives may need chunked
     dual numbers (`ForwardDiff.JacobianConfig`) at that size. The model interface does not
     change.
12. **Later**: GPU, Rician-likelihood fitting, more models:
   - Gradient-echo decay with the through-slice sinc term for macroscopic field gradients,
     S0·exp(−TE/T2*)·|sinc(Δf·TE)| (Fernández-Seara & Wehrli, MRM 2000), as a separate model.
     Δf, the frequency spread across the slice (Hz), is fixed from a B0 field map or fitted
     (default 0: no correction). Julia's `sinc` is the normalized sinc, matching the slice
     integral. Later: in-plane gradients (a product of sincs), non-rectangular slice profiles.
   - Polarity restoration for magnitude inversion recovery: tested against the grid start and
     not adopted (Performance notes, grid-search start).
   - Bounds in the solver: if boundary problems remain that the dead-zone guard does not handle,
     compare against a trust-region-reflective step (SciPy's `trf`) or a change of variables.
   - **Variable projection (VarPro)**: solve the parameters the signal is linear in (M0/S0, the
     biexponential amplitudes, an offset) by least squares inside every LM evaluation, so LM
     works on the nonlinear ones only (Golub & Pereyra 1973; Barral's RD-NLS does this for
     inversion recovery). Alternating between the linear solve and LM steps would zig-zag when
     the two are coupled (S0 and T2 are); VarPro re-solves the linear part at every evaluation.
     Prototype measured (branch `varpro-proto`, `prototypes/varpro/`, with its README):
     - How: a model wrapper with its own `normal_equations` method, so the unchanged `levmar`
       solves the reduced problem. The amplitudes are computed in dual arithmetic, which gives
       the exact Jacobian of the reduced problem (no Kaufman approximation). The columns of the
       linear part come from the model's `signal!` with the linear parameters at unit values
       (generic), or from functions the model supplies (only each column's own terms).
     - From EQUIP's own starts (2000 voxels per case, one seed, SNR 50, biexponential SNR 100;
       single thread, solver time, best of 3; "reached": final SSE within 1e-6 of the lowest
       any method found for that voxel; full → VarPro):

       | case | reached | median iterations | µs per voxel |
       |---|---|---|---|
       | T2Decay | 100 → 100% | 4 → 4 | 1.57 → 1.61 |
       | T1IR, T1 = 0.1 s | 100 → 100% | 5 → 5 | 1.26 → 1.00 |
       | T1IR, T1 = 0.9 s | 100 → 100% | 5 → 5 | 1.18 → 0.92 |
       | QMTSIR, 5-point | 100 → 100% | 6 → 6 | 2.86 → 2.68 |
       | biexp, 15% at 20 ms + 85% at 80 ms | 97.9 → 99.2% | 24 → 20 | 23.8 → 19.3 |
       | biexp, 40% at 15 ms + 60% at 100 ms | 96.6 → 100% | 18 → 15 | 19.5 → 14.5 |
       | T2DecayOffset | 100 → 100% | 10 → 11 | 5.2 → 5.0 |

       (biexponential and offset with model-supplied columns; biexponential fits stopping at
       max iterations: 43 → 18 of 2000.)
     - From bad starts: it helps where the trouble is coupling (T2Decay from T2 = 2 ms: 34 → 8
       iterations, 13.9 → 3.8 µs; offset from T2 = 1 s: 19 → 15 iterations), not with other
       minima (T1IR at T1 = 0.1 s from one start: 0% and 0.7% reached; at 0.9 s from 4 s: 100%
       and 97.5%; QMTSIR from a far start: 0% for both). Not a replacement for `BestOf`.
     - Generic columns cost too much with two amplitudes (the whole model evaluated per column:
       biexponential 23.1 → 27.2 µs, offset 5.2 → 6.5 µs), so those models need their own
       columns. A 2-D grid of 64 T2 pairs with VarPro (biexponential): 7–12 iterations, but
       ~50 µs per voxel and no more fits reaching the minimum.
     - Reading: worth adopting for the two-amplitude models (biexponential, offset), with
       model-supplied columns and the map back to the reported parameters (S0 = c₁ + c₂,
       fshort = c₁/(c₁ + c₂)). With one `:scale` parameter: 0–22% faster, robustness unchanged
       from EQUIP's starts (the `:scale` start already captures most of the gain), so optional;
       it can be generic there (the column is the signal at scale 1, as in `_score`).
     - Before adopting: answered in "Variable projection (VarPro): proposed design" above
       (`varpro = true`, a setting of its own on `Free`, checked at setup; columns from
       `signal!` at a base point, or a model's `columns!`; `fromamplitudes` for parameters that
       are not amplitudes, e.g. the biexponential and signed SIR; bounds, the per-voxel
       fallback, Float32, tests, order of work, open decisions).
   - **Log coordinates and a per-parameter stopping test** (prototype measured, branch
     `varpro-proto`, `prototypes/varpro/logt.jl`, results in `results_logt.txt`): the solver works
     in φ = log θ for chosen positive parameters and the model sees θ = exp(φ); the dual numbers
     apply the chain rule, so no derivative code changes.
     - Why: the step test (`xtol`) is one joint, relative test, ‖D^½ step‖ ≤ xtol·‖D^½ θ‖ (D the
       diagonal of JᵀJ, so the units cancel). M0's term is about the size of the signal, so in
       effect it says "the step changes the signal by less than xtol × the signal", not "every
       parameter has settled": a weakly determined parameter can still be moving. Measured
       (2000 voxels, SNR 50; biexponential SNR 100): the defaults (1e-8, 1e-12) stop about 10⁶
       times finer than the noise (estimates within ~5 × 10⁻⁷ noise SDs of a very tight fit);
       (1e-4, 1e-6) is 20–35% faster with changes under 1% of a noise SD for QMTSIR, T1IR and
       T2Decay, but stops 20 of 2000 biexponential voxels more than one noise SD early, in
       flat valleys where the SSE barely changes (Float32's default tolerances, 1e-5 and 1e-6, do
       the same). A per-parameter test, "stop when every parameter changes by less than a
       fraction of itself", is |Δ log θ| ≤ rtol in log coordinates.
     - Results, biexponential (two truths: 15% at 20 ms + 85% at 80 ms / 40% at 15 ms + 60% at
       100 ms; same starts; "early": voxels more than one noise SD from the same method's tight
       fit):

       | method | reached the minimum | early | µs per voxel |
       |---|---|---|---|
       | full fit, default tolerances (today) | 98.3% / 96.6% | 0 / 1 | 19.5 / 15.1 |
       | full fit + log T2s | 98.8% / 100% | 12 / 0 | 24.3 / 12.0 |
       | VarPro | 99.8% / 100% | 0 / 0 | 17.3 / 12.2 |
       | VarPro + log T2s | 99.5% / 100% | 23 / 0 | 15.8 / 9.8 |
       | VarPro + log T2s + per-parameter step 1e-4 | 99.5% / 100% | 0 / 0 | 11.6 / 6.3 |

     - Log coordinates help the hard fit converge (with VarPro, max-iteration stops 21 → 1 at
       tight tolerances and 15–25% fewer iterations; the full fit reached the minimum in 100%
       instead of 96.6% of voxels on the second truth), but the joint step test misbehaves in
       them (12–23 early stops at default tolerances: the "size" it compares the step with is a
       vector of logs). With the per-parameter test there were none, and VarPro + log T2s + a
       per-parameter relative step of 1e-4 was the best combination tested: 40–59% faster than
       today's fit, more voxels reaching the minimum. (A step of 1e-3: faster still, still no
       early stops.)
     - T2Decay (T2 = 2 s, TE ≤ 320 ms, SNR 50, from its log-linear start) and T1IR (T1 = 0.1 and
       0.9 s, from `BestOf`): log T2 / log T1 change nothing measurable; the starts already do the
       job. (The prototype's wrapper adds some time there, e.g. T2Decay 0.83 → 1.17 µs; that path
       is not optimized.)
     - Proposal: an option to fit positive parameters in log coordinates, with the
       per-parameter relative step test for them and the joint test for the rest; first for the
       biexponential, together with VarPro. The relative test cannot serve parameters that can
       be 0 or negative (Sf, PSR, fshort, an offset): they need an absolute tolerance or the
       standard-error rule (stop when every step is a small fraction of the parameter's
       standard error, the idea behind Bates & Watts' relative-offset criterion, which R's `nls`
       uses as I recall; it adapts to the SNR, and would come with item 3's statistics). Loosening
       the default tolerances alone: (1e-6, 1e-8) is the safe step (11–19% faster on the simple
       models, 1 of 2000 biexponential voxels affected); check the hard models before changing.
     - Limits: one seed, two biexponential truths, one SNR, solver time only.
   - Diffusion tensor and kurtosis, IVIM, multi-exponential T2 (myelin water), DCE.
