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
- **Log-linear start for T2Decay** (superseded 2026-11-08: T2Decay now starts from the shared
  `:loglinear` start on 1/T2, and this `datastart` is gone; "T2Decay on the shared log-linear
  start", below) (`datastart`; `_loglinear_rate` in `src/models/builtin.jl`,
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
- **Later** (item 9): NumPy, ISMRMRD. (DICOM and Philips PAR/REC: converted with dcm2niix,
  decided 2026-11-02: "DICOM through dcm2niix", below.)

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
- **The reader moved into EQUIP, on MRImages** (2026-11-02, "MRImages: MRI images in and out of
  files", below): `cli/src/io.jl` became EQUIP's `src/io/` on a package of its own for the
  files; MATLAB `.mat` data, maps and values (`file.mat:NAME`), MAT loaded only by a command
  that names one; `--like` for NIfTI maps of MATLAB data. 105 tests.
- **`equip info FILE... [--find PATTERN]`** (2026-11-02, `cli/src/info.jl`): a NIfTI file's
  header (readable: size, voxel size, axes, affine, sform/qform, storage and scaling; the header
  alone is read), its sidecar's entries, the `.bval`/`.bvec` beside it, and what EQUIP reads
  for the models' fields (or why it cannot: a count that does not fit the volumes); `--find`
  lists only the entries whose names match; a MATLAB file's variables with their sizes. 1.2 s,
  as any command (in the workload). 124 tests.

### Still to decide

1. (Decided 2026-11-01: NIfTI with BIDS sidecars, and MATLAB `.mat`; PAR/REC and the rest
   later.)
2. (Decided 2026-11-01: seconds only, as in the package; sidecars converted on reading.)
3. The bounds' syntax for `design` (`TI=0.01:10` reads as a range with two numbers).
4. The companion package's name, and how it is installed. (The data reader is no longer in it:
   EQUIP's `src/io/`, on MRImages, 2026-11-02.)
5. Python and MATLAB: after the release, as thin wrappers (proposed).

## MRImages: MRI images in and out of files (2026-11-02, branch `mrimages-proto`)

### Why a package of its own

Julia has pieces and nothing unified like Python's nibabel: NIfTI.jl (NIfTI only), DICOM.jl,
MAT.jl, MRIFiles.jl (raw k-space formats, not images; the name is taken), MriResearchTools.jl
(NIfTI helpers for phase and QSM), and readers inside analysis packages (DECAES has its own
PAR/REC reader). So every analysis package writes the same reading code again. nibabel's value
is that one library owns the image model (the values, the geometry, the header, the metadata)
and the formats, and nilearn, dipy and fMRIPrep build on it. Proposed by the owner while the
reader was being moved into EQUIP ("part of me thinks we need to build a separate i/o package
for mri..like nibabel for julia"); agreed.

**The split.** The reader written for the command line was two layers:

- **Generic, in MRImages** (knows nothing about EQUIP): read an image (its values with the
  scaling applied, its geometry, its sidecar); write values in another image's geometry; BIDS
  sidecars; `.bval`, `.bvec` and text files of values; MATLAB files through an extension.
- **EQUIP's, in `src/io/`**: stacking files into observations, sidecar keys to the models'
  field names (`EchoTime` → `TE`), a fit's maps written with BIDS names and the fit's provenance,
  tables as TSV.

Small for the release (NIfTI, BIDS sidecars, `.bval`/`.bvec`, MATLAB); a nibabel-scale package
(many formats, orientation handling, lazy loading) is a project in its own right and does not
block EQUIP. In this repository for now (as `cli/` is: `MRImages/`, its own Project.toml and
tests), moved to its own repository later (perhaps under JuliaNeuroscience). Since the boundary
is drawn now, moving it will not change EQUIP's interface.

**The name** (decided 2026-11-02, the owner's choice of the candidates): `MRImages`. The owner's
first, a joke, was MENT (EQUIP + MENT): General's AutoMerge refuses names shorter than 5
characters (`_AUTOMERGE_NEW_PACKAGE_MINIMUM_NAME_LENGTH = 5`, read in AutoMerge.jl's source),
so it would need a manual merge. MRImages reads plainly outside the field (the naming
guidelines' first rule), is plural for the type it provides (`MRImage`, as `DataFrames`
provides `DataFrame`), is not limited to the brain (spinal cord, muscle), and is distinct from
`MRIFiles` (raw data). Checked free in the General registry (the local copy of 2026-10-14),
with `NeuroImageIO`, `MRIFormats` and `MRIImages`. `NeuroImages` was avoided: `Neuroimaging`
(an EEG package) is taken, and the similarity check could refuse it. EQUIP itself (5
characters, all capitals) passes AutoMerge's length and capitalization checks: the latter is
`^[A-Z][A-Za-z0-9]*?$`, which allows capitals only (its message says otherwise).

### Decisions

- **EQUIP depends on MRImages; MAT is an extension** (decided 2026-11-02 with the owner, the
  option "EQUIP; MAT as extension"): `using EQUIP` reads and writes NIfTI with sidecars
  (NIfTI.jl and JSON.jl, measured +0.18 s of loading); `.mat` works after `using MAT`, so only
  those who read MATLAB files pay its 0.43 s (it brings HDF5). This replaces the earlier rule
  "the core never depends on file formats" (item 9): fitting data is the main use, and without
  a reader EQUIP could not do it. EQUIP re-exports `MRImage`, `readimage`, `writeimage` and
  `readvalues`, so one `using EQUIP` is enough to fit files.
- **`MRImage` is a struct, not an AbstractArray**: `data` (an Array, the scaling applied, in a
  floating type: `precision`), `header` (the NIfTI header, or `nothing` for a MATLAB array),
  `meta` (the sidecar as a `Dict{String,Any}`), `path`. As nibabel's images are not arrays:
  what an operation returns is never in doubt (an array, without the geometry), and showing
  one prints a summary. The header stays NIfTI's (DICOM and PAR/REC come as NIfTI, through
  dcm2niix: below; a format-neutral geometry type only if a format with geometry is ever read
  directly).
- **Names**: `readimage`/`writeimage` (not `load`/`save`, which are FileIO's and would clash).
- **Writing** (`writeimage`): the values as they are (no intensity scaling: the scaling was
  applied on reading), in their element type, true/false as UInt8 (NIfTI's 1-bit type is read
  by few tools); the image's `meta` as the sidecar. `MRImage(values, like)` takes `like`'s
  geometry and resets what describes the values (scaling, display range, intent; `descrip`).
- **MATLAB data** (`file.mat:NAME`, or `file.mat` holding one variable): observations along the
  last dimension, after up to three spatial ones (`nx×ny×nz×N`, `nx×ny×N`, `nvoxels×N`);
  `loaddata` reshapes to 4-D (`nvoxels×1×1×N`), so fits and maps are the same code (and the
  same compiled code) as for NIfTI, and `savemaps` writes the maps back in the array's own
  shape, one `.mat` file (`PREFIX_maps.mat`, a variable per map) with a JSON sidecar; NIfTI
  maps in another image's geometry with `like`. MATLAB has no geometry, so a NIfTI map given
  with MATLAB data is checked for its size only. Written as v7.3 (MAT.jl writes v4 or v7.3;
  MATLAB's `load` reads both). Acquisition values from a variable: `readvalues("p.mat:TI")`.
- **The extension** adds three methods (`_matnames`, `_matread`, `_matwrite`); MRImages' public
  functions check that it is loaded (`Base.get_extension`) and say how otherwise ("`using
  MAT`"), and call it through `invokelatest`, so it works when MAT is loaded while a program
  runs. Not loaded automatically by MRImages (as FileIO would): `using MAT` is the user's
  choice, as decided.
- **The command line loads MAT only when a command names a `.mat` file** (any word, after
  `--config` files are expanded: `Base.require` of MAT's package id, MAT in EQUIPCLI's
  dependencies). EQUIPCLI no longer depends on NIfTI.jl or JSON.jl directly.
- **`savemaps` records how the maps were made by itself**: the model, its acquisition (the
  model's fields), each parameter's role as written (`Free(start = 0.07)`, a number, "a map"),
  the precision, the weights, EQUIP's version; the caller's `sidecar` entries are added and win
  (the command line adds the command, the data files, the flags as typed).

### Measured

Three runs each, nothing else running:

| | before (`cli-proto`) | after |
|---|---|---|
| Julia's start | 0.19–0.24 s | 0.17–0.19 s |
| `using EQUIP` (its workload compiled) | 0.69–0.78 s | 0.76–0.78 s |
| `using MRImages` alone | | 0.43 s |
| `using EQUIPCLI` | 1.15–1.22 s | 1.16–1.24 s |
| `equip fit T2Decay t2.nii.gz` (60 voxels, TE from the sidecar) | 1.20–1.36 s | 1.26–1.37 s |
| `equip crlb QMTSIR ... --tissue SIR_WM_I` | 1.23–1.40 s | 1.16–1.28 s |
| `equip models` | 1.16–1.40 s | 1.17–1.28 s |
| `equip fit T2Decay t2.mat:S --TE @t2.mat:TE` | | 2.0–2.06 s |

So EQUIP loads about 0.04 s slower (NIfTI.jl and JSON.jl), and the commands are unchanged
within run-to-run noise.

**MATLAB commands, 11 s → 2.0 s.** At first a `.mat` command took 10.5–11 s: MAT.jl compiles
its reading code on first use (the first `matread` of a session: 6.9 s). A precompile workload
in MRImages' extension (writing and reading a small file of each kind of variable) brought it to
3.8–4.2 s, and no further: the extension's code was compiled again ("recompile" in
`--trace-compile`: 1.7 s for `_matnames` alone), because StaticArrays invalidates it. The first
`readmat` with nothing loaded first: 7 ms; with Statistics and LinearAlgebra, CommonSolve or
StatsAPI loaded first: 6–10 ms; with StaticArrays: 1.72 s; ForwardDiff (which loads it): 2.18 s;
EQUIP: 2.33 s. StaticArrays is loaded in every EQUIP session, so EQUIP has an extension of its
own, `EQUIPMATExt` (triggers MAT and MRImages), holding only a precompile workload (loaddata,
loadmap, a fit, savemaps and readvalues on a small `.mat` file), compiled with StaticArrays
loaded: its code stays valid. Its triggers include MRImages because Julia lets an extension load
another while it is built only if the other's triggers are a strict subset of its own
(Base.loading, `loadable_exts`), and the workload reads through MRImages' extension. Now: 2.0 s,
which is 1.2 s as any command, 0.66 s loading MAT (HDF5) and the two extensions, and 0.16 s
still compiled (small Base methods). In a Julia session (`using EQUIP, MAT`), the first `readmat`
takes 6 ms, the first `loaddata` of a `.mat` file 24 ms, the first `savemaps` to `.mat` 0.25 s.

**Results unchanged**: no code of fitting, crlb, montecarlo or design changed (`src/EQUIP.jl`
gained includes and exports only), and the command line's tests compare its maps and tables
with the Julia API's (exactly, or within the rounding of maps written as Float32). Tests:
MRImages 86; EQUIP 1840 (46 new: `io`); the command line 105 (15 new: MATLAB data, values and
maps, MAT loaded only when named, `--like`).

### DICOM through dcm2niix (decided 2026-11-02)

MRImages reads no DICOM and no PAR/REC: they are converted first with dcm2niix (Rorden), and
the NIfTI file and its BIDS JSON sidecar are the input. Raised by the owner: reading DICOM
"just seems like an enormous task ... all the rules", against converting with dcm2niix or
offering NIfTI only. Why:

- **The rules are the work, and dcm2niix has them.** Siemens keeps values in private CSA
  headers and packs slices in mosaics; Philips has its own intensity scaling; each vendor
  stores diffusion directions its own way; enhanced (multi-frame) DICOM differs from classic.
  dcm2niix has met these for years, across scanners and software versions, and the BIDS tools
  (HeuDiConv, dcm2bids) are built on it. A reader of our own would repeat that, and lag it. It
  also reads Philips PAR/REC and XML/REC, so that reader goes too.
- **Its sidecar is the data dictionary.** What the DICOM headers hold, in BIDS names and units
  (EchoTime and RepetitionTime in s, FlipAngle in degrees, InversionTime, Manufacturer,
  SeriesDescription, ...), the vendors' differences resolved. MRImages keeps every entry as it
  is (`img.meta`, nested objects too) and every NIfTI header field (`img.header`), so nothing
  is lost; `findmeta(img, "echo")` searches the entries by name, `headerinfo` shows the header
  readable, and `equip info FILE [--find PATTERN]` does both from the command line (with the
  files beside it and what EQUIP reads from them). EQUIP's own rules stay five: EchoTime →
  `TE`, InversionTime → `TI`, RepetitionTime → `TR`, FlipAngle → `FA`, a `.bval` → `b`.
- **No reader removes the flags.** Values a sequence keeps in no standard field (SIR's TD, MT
  pulses, a custom sequence's own settings) come from flags or a config file, whatever reads
  the headers.
- **Later, perhaps**: `equip convert dicomdir/ out/`, running dcm2niix if it is installed (pip,
  conda, Homebrew; FSL and MRIcroGL include it), with a message saying how to get it if not.
  There is no Julia package of dcm2niix in the General registry (checked, 2026-11-02); one built
  with BinaryBuilder (Yggdrasil; dcm2niix is BSD-licensed C++) would install it with EQUIP.
  Not for the release: the documentation says to convert with dcm2niix first.

### Next

- MRImages in a repository of its own, registered before EQUIP (EQUIP's registration needs
  it).
- **Julia 1.10 (the long-term-support release), with it** (the owner, 2026-11-08: "it's fine
  for now", on the roadmap). Today the minimum is 1.11 for one reason: the `[sources]` entries
  (EQUIP's Project.toml finds MRImages in this repository, the command line's EQUIP and
  MRImages), which Julia reads from 1.11 on. Once MRImages is registered, they go, and 1.10 is
  `julia = "1.10"` in the three [compat] sections, 1.10 in CI's matrix, and whatever its run
  finds (checked 2026-11-08: no other 1.11-only feature in the code; `logrange`, from 1.11, only
  in `BestOf`'s docstring example). CI tests 1.11 and the latest release; on 1.11.9 its first
  run found only the allocation test measuring from top-level code (fixed).
- DICOM and PAR/REC through dcm2niix (above): perhaps `equip convert`, or dcm2niix as a Julia
  binary package. Other formats when asked for (MGH, NumPy); orientation handling (to a
  standard orientation and back), memory-mapped reading of large files.
- The command line's no-compile test (still to do from the plan), now with a `.mat` command.

## Peripheral nerve SMT: the lab's protocol questions (2026-11-03, branch `nerve-smt`)

A research branch, kept off `lean` until the owner has tested `lean` and the work here is
accepted. It starts from `smt-proto` (SMT, SMTFreeWater, SMTT2, sphericalmean).

### The questions (the owner's)

The lab images peripheral nerves with diffusion MRI, today with about 10 averages over several
gradient directions.

1. **Directions, averages and b-values**: what is the best use of them for a nerve SMT protocol
   under 10 minutes? With TR and the number of slices in the scan time, and a floor on the SNR of
   the raw images (each direction's image, after the scanner's averages), below which they cannot
   be registered for motion and distortion correction.
2. **An echo-time dimension**: does repeating SMT at two echo times (`SMTT2`), using the
   different T2s of the intra- and extra-axonal compartments, separate their signals better? And
   does it fix the T2 weighting of the compartments in nerve, where single-TE SMT reports a
   T2-weighted fraction, not the volume fraction (2026-11-03: v_ax 0.45, T2 90 and 50 ms, TE 80 ms
   gives 0.66)?

### How EQUIP answers them

The quantity judged is the error of the fitted volume fraction v_ax (its bias and SD, so its
RMSE), worst case over a set of nerve scenarios (healthy and pathological values; the T2s). Each
source of error is a perturbation of the shell means, carried to v_ax by the fit's sensitivity
(crlb's sandwich, and its bias from data made by another model):

- **Noise**: σ²/(N_dirs·N_avg) per shell, from the per-image SNR (crlb's `averages`).
- **The Rician floor**: the scanner's averages are (probably) complex, and shrink the floor; the
  spherical mean averages magnitudes, and does not. At high b, where a nerve's signal is low, the
  floor biases the shell means upward however many directions are averaged: this is where
  averages can beat directions. Needs the Rician mean E|y| (closed form) for the bias, and the
  Rician information (Devan 2020, Appendix) for crlb.
- **Orientational sampling**: a finite direction set does not average the anisotropic signal
  exactly. A nerve's fibers are one aligned population, the worst case. Closed form for a
  direction set, by a Legendre expansion of one fiber's signal,
  Var = Σ_{l≥2} a_l²/(2l+1) · (1/N²) Σ_ij P_l(g_i·g_j), its covariance across shells by the same
  sum over the two shells' directions (the same fiber in every shell). Checked against
  simulation (2026-11-03: white matter, b = 3000 s/mm², a Fibonacci set: 6 directions 17.2%
  against 17.2%, 20: 3.41% against 3.42%, 64: 0.96% against 0.96%; 5% at about 15 directions, as
  Schilling 2022's SNR = ∞ curve).
- **T2 weighting**: single-TE SMT fitted to data with different compartment T2s (the bias, from
  `simulate_with = SMTT2(...) => θ`), against SMTT2's own error at two TEs: bias against
  precision, so RMSE, at the same scan time.
- **Scan time**: T = Σ_shells N_dirs·N_avg·TR (+ the b = 0 images) for each TE, with TR set by the
  number of slices and the time per slice (which depends on TE and the readout), and T1
  recovery if TR is short. Under 10 minutes: the designer's `budget`, or a constraint.
- **The registration floor**: for each shell, the per-image SNR (after the scanner's averages)
  of its lowest-signal direction (parallel to the nerve) or of its mean, at least a threshold:
  a designer constraint.
- **Design variables**: each shell's b (continuous), N_dirs and N_avg (integers: enumerated),
  the echo times (continuous; the minimum TE depends on b through the gradients), the number of
  slices given. Validated by a full Monte Carlo of the pipeline: directional signals, the fibers'
  orientation, noise with the scanner's averaging, the spherical mean, the fit.

### To build here, in order

1. The Rician mean and the Rician information (the floor's bias; crlb under Rician noise).
   **Done** (2026-11-03, src/sim/rician.jl): `ricianmean(A, σ)` (σ√(π/2)·L₁/₂(−A²/2σ²), through
   scaled Bessel functions, differentiable), `ricianvar`, `ricianinformation` (R(A/σ)/σ², by
   quadrature, the expansion 1 − 1/(2s²) − 1/(4s⁴) above s = 40); `RicianMean(m; σ, averages,
   averaging)`, a model whose signal is the magnitudes' mean; `riciannoise(m, θ; snr, averages,
   averaging)`, crlb's keywords for a least-squares fit of magnitude data (`simulate_with` the
   mean: the floor's bias; `Σ` the Rician variance: the sandwich's SD); `ricianbound`, the
   Cramér–Rao bound of Rician data (crlb with σⱼ/√(NEXⱼRⱼ) and the matching weights). Built from
   crlb's own keywords: its kernel is unchanged.
   - Checked: the mean and variance against simulated magnitudes; R against Monte Carlo of the
     score's variance (s = 1: 0.5214 against 0.5217; s = 3: 0.940 against 0.941).
   - **The floor's bias, predicted**: SMT, black holes, protocol A, SNR 20 per direction, the
     spherical mean a magnitude average: crlb predicts a bias of +0.071 in v_ax (truth 0.241),
     montecarlo measures +0.069; SD 0.0243 against 0.0244. The floor's bias is three times the
     SD there: what directions cannot average away and the scanner's complex averages can.
     Fitting `RicianMean(SMT; σ)` (σ known) removes it.
   - **Devan 2020, Fig. 1, with the Rician bound**: the v_ax precision efficiency of the five
     protocols at SNR 20 is their figure within 5% everywhere but protocol D (3–8% low); black
     holes A 0.90 against 0.90, B 0.90 against 0.905, C 0.83 against 0.825, E 0.68 against 0.68,
     where the Gaussian bound was up to 13% high (A, B). So their Fig. 1 is the Rician CRLB at
     SNR 20 per direction, each direction its own magnitude image.
2. The orientational covariance for a direction set (closed form above), and direction sets
   (electrostatic repulsion, Jones et al., Magn Reson Med 1999), tested against simulation.
   **Done** (2026-11-03, src/models/diffusion/orientation.jl): `directionset(N)` (repulsion
   between the directions and their opposites, from a Fibonacci spiral: deterministic; z ≥ 0;
   second moment I/3 to 1e-9); `orientationcovariance(m, θ, sets; σ)` for the SMT models (each
   model's directional signal `_profile`, its Legendre coefficients by 120-node Gauss–Legendre,
   orders to 60); `RicianSphericalMean(m; σ)` and `sphericalmeannoise(m, θ; snr, directions,
   averages, sets)`, crlb's keywords for the whole spherical-mean pipeline.
   - Checked: the covariance against 20,000–50,000 random fiber orientations, cross-shell terms
     too (1.135 against 1.163 ×10⁻⁶; diagonal within 1%), mean 0. Schilling 2022's minimal
     directions at SNR = ∞ (v 0.6, λ 2 μm²/ms, 5%): 4, 7, 10, 16, 23, 28 at b = 1, 2, 3, 5, 8, 10
     ms/μm², as their Fig. 4.
   - **Found: the floor acts direction by direction.** A first comparison with the whole
     pipeline simulated (one fiber population in a random orientation per voxel, a Rician image
     per direction, the spherical mean, the SMT fit) matched the SD but not the bias: v_ax 0.156
     simulated against 0.033 from the floor of the shell's mean signal (`riciannoise`). For one
     aligned population the directions near the fibers have almost no signal at high b and sit
     on the floor, and E|y| is convex in the signal, so the mean of magnitudes is above the
     floor of the mean. With the floor applied to each direction's signal and averaged over the
     angle (`RicianSphericalMean`), and the noise and orientation terms on the same footing
     (`sphericalmeannoise`): bias 0.159 against 0.156 (30 directions, SNR 20), 0.172 against
     0.158 (16–24 directions), 0.045 against 0.049 (8 directions, SNR 50); SD within 10–25%
     (crlb is linear; simulated fits that reach D_ax's upper bound narrow the spread). With 6
     directions the orientational SD (about 0.2) leaves the linear regime: the bias half the
     simulated one. Candidates near that edge are for the Monte Carlo.
   - **A first look at directions against averages** (SMT, b = 0, 1000, 2500, 4000 s/mm²; v 0.6,
     D 2e-3; 120 acquisitions per diffusion shell; SNR 20 per acquisition; the scanner's
     averages complex): RMSE of v_ax 0.162 for 120 directions × 1 average (the floor: bias 0.16),
     0.105 (60 × 2), 0.069 (30 × 4), 0.058 (20 × 6), 0.074 (12 × 10: the orientational SD grows),
     0.218 (6 × 20). Placeholder values; the lab's own come next (steps 3 and 4).
   - **Floor corrections compared** (2026-11-03; the pipeline above: SMT, b = 0, 1000, 2500,
     4000 s/mm², 4/30/30/30 directions, v 0.6, D 2e-3, one fiber population in a random
     orientation, 2000 voxels, σ known). Bias, SD and RMSE of v_ax at SNR 20 per image, then 10:
     | Method | SNR 20 | SNR 10 |
     |---|---|---|
     | least squares, no correction | +0.154, 0.056, 0.164 | +0.321, 0.079, 0.330 |
     | Karlsen inversion of each shell mean, then least squares | +0.125, 0.058, 0.138 | +0.227, 0.097, 0.247 |
     | Karlsen inversion of each direction, then the mean (Kaden's `--rician`) | +0.045, 0.060, 0.075 | +0.118, 0.105, 0.158 |
     | √max(M² − σ², 0) of each direction, then the mean (NYU's SMI.m) | +0.071, 0.060, 0.093 | +0.167, 0.101, 0.195 |
     | √\|M² − σ²\| of each direction, then the mean (NYU's DESIGNER v1 and v2) | +0.111, 0.057, 0.125 | +0.247, 0.088, 0.262 |
     | E\|y\| of the shell mean as the model (`RicianMean`) | +0.124, 0.058, 0.137 | +0.223, 0.097, 0.243 |
     | E\|y\| of each direction as the model (`RicianSphericalMean`) | −0.001, 0.044, 0.044 | −0.000, 0.080, 0.080 |

     The Karlsen inversion (Magn Reson Med 1999;41:614, Eq. 4: the A whose E|y| is the measured
     value, 0 below the floor σ√(π/2); the owner's `corrRicianBias.m`, and Kaden's
     `ricedebias`, a Chebyshev fit of the same inverse, applied to every volume before the
     spherical mean, Kaden et al., Magn Reson Med 2016;75:1752) is exact for the mean of an ROI
     whose pixels share one A. A shell's spherical mean is not one: its directions have
     different A, and E|y| is convex in A, so inverting the mean removes only the floor of the
     mean (the same as fitting `RicianMean`: 0.125 against 0.124). Inverting each direction is
     better but is the inverse of a single noisy value, biased at low SNR (below the floor it
     clips to 0, above it is not). Modelling E|y| direction by direction inside the fit
     (`RicianSphericalMean`) has no bias and the smallest SD, but assumes the directional
     profile, one fiber population: right for a coherent nerve, not invariant to dispersion or
     crossings as SMT itself is. Devan 2020 fitted their Rician simulations with Kaden's code
     and does not say it used `--rician`.

     NYU's correction (SMI.m's `NoiseBias = 'Rician'`, DESIGNER's `-rician`, σ from MP-PCA, each
     direction's image before the spherical harmonics) is the high-SNR form of Gudbjartsson &
     Patz (Magn Reson Med 1995), E|y| ≈ √(A² + σ²): right far above the floor, short of the
     exact inversion near it; DESIGNER's absolute value also turns a value below σ into
     √(σ² − M²), up to σ, where the floor should give 0. DESIGNER-v2 points to the other cure,
     denoising complex data (`-phase`, not to be combined with `-rician`), which removes the
     floor at its source. The owner's view (2026-11-03), which these numbers bear out: the floor
     is a property of each direction's image; after the spherical mean the noise is close to
     Gaussian (many directions), so least squares suits it, but its mean carries the average of
     the directions' floors. A correction of each value works from one noisy value and cannot
     be unbiased near the floor; the model of the mean (`RicianSphericalMean`) can.
   - **Robustness of the corrections** (2026-11-03, bench/floor_robustness.jl, the same pipeline;
     v_ax bias / RMSE, truth 0.6). σ misestimated (each correction given f × the true σ, the same
     data), one fiber direction:
     | Method | SNR 20, f 0.8 | f 1.0 | f 1.2 | SNR 10, f 0.8 | f 1.0 | f 1.2 |
     |---|---|---|---|---|---|---|
     | no correction | +0.156 / 0.165 | +0.156 / 0.165 | +0.156 / 0.165 | +0.321 / 0.330 | +0.321 / 0.330 | +0.321 / 0.330 |
     | exact inversion per direction | +0.082 / 0.101 | +0.047 / 0.076 | +0.013 / 0.059 | +0.184 / 0.209 | +0.119 / 0.159 | +0.055 / 0.119 |
     | SMI.m | +0.099 / 0.115 | +0.072 / 0.094 | +0.044 / 0.074 | +0.217 / 0.237 | +0.168 / 0.195 | +0.116 / 0.155 |
     | DESIGNER | +0.121 / 0.134 | +0.112 / 0.126 | +0.109 / 0.123 | +0.259 / 0.273 | +0.246 / 0.261 | +0.243 / 0.257 |
     | E\|y\| per direction in the model | +0.035 / 0.058 | +0.000 / 0.044 | −0.034 / 0.053 | +0.075 / 0.112 | +0.001 / 0.080 | −0.076 / 0.106 |

     Fibers dispersed about the voxel's orientation (100 fibers from a Watson distribution; ODI
     0.04, 0.16, 0.50 for κ = 16, 4, 1), σ right:
     | Method | SNR 20, ODI 0.04 | 0.16 | 0.50 | SNR 10, ODI 0.04 | 0.16 | 0.50 |
     |---|---|---|---|---|---|---|
     | no correction | +0.128 / 0.141 | +0.057 / 0.084 | +0.035 / 0.069 | +0.288 / 0.301 | +0.187 / 0.212 | +0.138 / 0.172 |
     | exact inversion per direction | +0.031 / 0.067 | −0.008 / 0.062 | −0.005 / 0.060 | +0.086 / 0.139 | −0.013 / 0.118 | −0.059 / 0.131 |
     | SMI.m | +0.053 / 0.079 | +0.005 / 0.062 | −0.001 / 0.059 | +0.135 / 0.170 | +0.035 / 0.119 | −0.010 / 0.115 |
     | DESIGNER | +0.085 / 0.103 | +0.016 / 0.063 | −0.000 / 0.059 | +0.207 / 0.227 | +0.088 / 0.138 | +0.022 / 0.111 |
     | E\|y\| per direction in the model | −0.023 / 0.049 | −0.079 / 0.091 | −0.096 / 0.105 | −0.036 / 0.087 | −0.125 / 0.147 | −0.166 / 0.181 |

     - σ: every correction's bias moves in proportion to σ's error; the model's by ∓0.017 for
       ±10% at SNR 20 (∓0.037 at SNR 10), and it keeps the lowest RMSE across ±20%. The
       per-direction corrections fall short at the right σ, so a σ too high helps them.
     - Dispersion lowers the floor's bias for every method (no correction: 0.157 at ODI 0 to
       0.035 at 0.50, SNR 20): with the fibers spread, fewer directions sit near zero signal.
       The model, which assumes one fiber direction, then over-corrects (−0.023 at ODI 0.04,
       −0.079 at 0.16), while the per-direction corrections become about right at ODI 0.16. The
       crossover is between ODI 0.04 and 0.16: below it the model is best, above it the exact
       inversion per direction or SMI.m. The floor is worst where the fibers are coherent, which
       is where the model is right.
     - Worst RMSE over all ten conditions: SNR 20, no correction 0.166, exact inversion 0.101,
       SMI.m 0.115, DESIGNER 0.134, the model 0.105; SNR 10, 0.332, 0.209, 0.237, 0.273, 0.181.
     - Open: the nerve's dispersion (an input from the lab). The model could carry it, a Watson
       ODF in `_profile` with κ fixed from elsewhere (the spherical mean itself holds almost no
       information about κ; the l = 2 rotational invariant, NODDI or the literature do).
   - **The spherical mean from a Rician spherical-harmonic fit** (2026-11-03, bench/floor_sh.jl,
     the owner's question: the l = 0 coefficient is the spherical mean, so fit each shell's
     directions with E|y| of an SH series and need no fiber ODF). v_ax bias / RMSE, σ right:
     | Method | SNR 20, ODI 0 | 0.04 | 0.16 | 0.50 | SNR 10, ODI 0 | 0.04 | 0.16 | 0.50 |
     |---|---|---|---|---|---|---|---|---|
     | no correction | +0.157 / 0.166 | +0.128 / 0.141 | +0.057 / 0.084 | +0.035 / 0.069 | +0.323 / 0.332 | +0.288 / 0.301 | +0.187 / 0.212 | +0.138 / 0.172 |
     | exact inversion per direction | +0.048 / 0.076 | +0.031 / 0.067 | −0.008 / 0.062 | −0.005 / 0.060 | +0.120 / 0.158 | +0.086 / 0.139 | −0.013 / 0.118 | −0.059 / 0.131 |
     | E\|y\| per direction, one fiber direction | +0.001 / 0.044 | −0.023 / 0.049 | −0.079 / 0.091 | −0.096 / 0.105 | +0.002 / 0.080 | −0.036 / 0.087 | −0.125 / 0.147 | −0.166 / 0.181 |
     | SH L = 2, E\|y\| | +0.015 / 0.074 | −0.037 / 0.069 | −0.075 / 0.102 | +0.000 / 0.059 | +0.051 / 0.121 | −0.016 / 0.116 | −0.113 / 0.177 | −0.043 / 0.162 |
     | SH L = 4, E\|y\| | −0.094 / 0.109 | −0.070 / 0.094 | −0.032 / 0.077 | −0.002 / 0.061 | −0.069 / 0.139 | −0.055 / 0.139 | −0.115 / 0.181 | −0.147 / 0.222 |
     | SH L = 6, E\|y\| | +0.021 / 0.065 | +0.014 / 0.062 | −0.019 / 0.066 | −0.006 / 0.061 | +0.090 / 0.141 | +0.060 / 0.127 | −0.042 / 0.128 | −0.088 / 0.147 |

     - L = 6 has the smallest worst-case RMSE of every method (SNR 20: 0.066, against 0.076 for
       the exact inversion per direction and 0.105 for the one-direction model; SNR 10: 0.147,
       0.158, 0.181), and no ODF assumption. But it is no cure: L = 2 and 4 are biased by
       truncation (a coherent bundle at b = 2500–4000 s/mm² is sharp on the sphere: L = 4 gives
       −0.094 with no dispersion), L = 6 needs at least 28 directions per shell, its 28
       coefficients from 30 values pool little across directions (SD about 0.06 against 0.044
       for the one-direction model at SNR 20), and at SNR 10 its bias reaches ±0.09.
     - Why no correction of the spherical mean can be free of the ODF: SMT's independence of the
       ODF holds for the noise-free signal, the average of the directions' signals. The floor
       acts on each direction through the nonlinear E|y|, so the mean of magnitudes depends on
       how the signal is spread over directions, which is the ODF. A correction must assume an
       ODF (the one-direction model), describe the directional signal freely (the SH fit, at the
       cost of many coefficients), or work one noisy value at a time (the inversions). Averaging
       complex data before the magnitude (the scanner's averages, or phase-corrected real data)
       is the only route that keeps SMT's ODF independence at low SNR.
     - The owner's follow-up: if the directional signal is fitted anyway, why the spherical mean?
       A directional model, the SMT kernel convolved with a Watson ODF (orientation, κ, v_ax,
       D_ax, S0: 6 parameters for the 94 images) and E|y| per direction, pools across directions
       far more than 28 SH coefficients per shell and models the floor exactly; it gives up SMT's
       robustness to the ODF's form (crossings, fascicles at angles), which matters less in a
       single coherent nerve. Next to test, on the same pipeline.
3. A protocol evaluator: shells (b, TE, N_dirs, N_avg), TR and slices in; scan time, the
   registration check, and v_ax's bias, SD and RMSE out, from crlb with the terms above;
   checked against the full Monte Carlo. For each kind of data: magnitude images (the
   scanner's averages complex or magnitude; the floor correction the pipeline uses) or
   phase-corrected real images ("Real-valued data", below).
4. The search: N_dirs and N_avg enumerated, b and TE optimized (`optimize_protocol`), under the
   budget and the SNR floor, worst case over the nerve scenarios; one TE (SMT) against two
   (SMTT2), by the RMSE of the volume fraction.

### Real-valued data: removing the floor at its source (2026-11-03)

Every correction of magnitude data in step 2 pays somewhere: an assumed ODF, many SH
coefficients, or one noisy value at a time. The floor itself goes away if each image is
phase-corrected and its real part kept (Eichner et al., NeuroImage 2015): Gaussian noise with
zero mean, negative values allowed. Then every average is linear (the scanner's repeats, the
spherical mean, SH coefficients), least squares is unbiased with no σ needed for a correction,
and SMT's independence of the ODF holds at any SNR. The other limits stay: the model's own
assumptions (tortuosity, D_ax = De∥, exchange, the degeneracy without them). The lab has nerve
DTI averaged on the scanner both as complex data and as magnitudes (repeats, not directions).

- **Directions are no harder than repeats.** Each image's phase is removed from that image before
  any average, so phases need not agree between directions or between repeats. The phase from
  eddy currents (smooth: linear terms and a B0 offset that change with the gradient direction)
  and from motion during the diffusion encoding (random from shot to shot) is absorbed by a
  smooth phase estimated from each image; no calibration per direction is needed, and motion
  phase could not be calibrated anyway. The geometric part of eddy currents (shifts, shears)
  needs registration, as for magnitude data.
- **What it needs**:
  1. A complex coil combination (SENSE-type, or adaptive with coil sensitivities), exported as
     complex or magnitude and phase images, or raw data with an own reconstruction. A sum of
     squares discards the phase (and its noise is noncentral χ, not Rician).
  2. A phase estimate per image, a low-pass filtered phase. Its error δφ lowers the real part to
     A·cos δφ, about A·exp(−σ_φ²/2) for small Gaussian errors: a small downward bias, largest
     where the phase changes fast (edges, vessels, pulsation) and at low SNR, where the
     smoothed phase is itself noisy. The filter's width trades the two. Partial Fourier
     reconstruction handles the phase too, and has to be taken into account.
  3. A pipeline that keeps negative values: the phase correction in each image's own space, the
     real part, then distortion and motion correction (linear interpolation keeps the noise
     Gaussian, correlated in space). Any step that clips negatives brings the floor back.
- **A check, and σ, from the imaginary part**: after the correction it should be noise alone
  (mean 0, SD σ). Structure in it is signal lost from the real part; its SD is σ, without
  MP-PCA or a background region.
- **The lab's data measure the floor**: complex averaging already removed the shot-to-shot phase
  of the repeats (the floor lowered by about √NEX, 3 for 10 averages), but each direction is
  still a magnitude image. The complex and the magnitude averages of the same nerve data,
  compared at the highest b in the directions along the nerve (the lowest signal), give the
  floor's bias in the lab's own data: whether all of this matters there.
- **What it changes in the design**: without the floor, a number of images gives the same noise
  in the spherical mean however it is split between directions and averages; more directions
  then only lower the orientational sampling error. The averages are set by the registration
  floor (each image registrable, step 3), no longer by the Rician floor: 120 directions × 1
  average, the worst of step 2's split with magnitude data, becomes the best if its images can
  be registered.
- **In EQUIP**: Gaussian noise on each direction, with `sphericalmeannoise`'s noise and
  orientation terms and no floor (a real-data option to add: today it applies the floor
  direction by direction); the phase errors, if they matter, a multiplicative bias per
  direction, exp(−σ_φ²/2), through `simulate_with`, with σ_φ from the filter and the image's SNR.

### SMT or SMI in disorganized nerve (2026-11-03, discussed with the owner)

The question: in a neuroma, with axons in every direction, can SMI (Standard Model Imaging:
Novikov et al., NeuroImage 2018; Coelho et al., NeuroImage 2022) give v_ax, or is SMT needed?

- Both fit the same Standard Model (sticks, a zeppelin, optionally free water, convolved with
  the ODF). SMT uses only the l = 0 rotational invariant (the spherical mean, independent of the
  ODF) and makes v_ax identifiable from it by fixing the extra-axonal diffusivities (De∥ = D_ax,
  tortuosity). SMI adds the l = 2 (and l = 4) invariants, S_l(b) = p_l·K_l(b), which give the
  ODF's anisotropy p₂ and the equations that let D_a, De∥ and De⊥ be fitted separately (with
  free water and, at several TEs, compartment T2s); its estimator is a regression trained on a
  prior.
- With an isotropic ODF, p₂ = 0: the l = 2 invariant is zero for any kernel, every direction's
  signal is the spherical mean, and SMI has only the l = 0 data. Without SMT's constraints the
  l = 0 data with linear encoding have two families of solutions (Jelescu et al., NeuroImage
  2016); SMI's answer then comes from its prior. In a study of nerve against neuroma, SMI's
  v_ax would be pinned partly by l = 2 data in one group and by the prior in the other: a group
  difference could come from the information, not the tissue. SMT uses the same information in
  both, valid as far as its constraints hold in both (fibrosis and disorganization are where
  they may not; exchange breaks both).
- What determines v_ax without l = 2: SMT's constraints; a second echo time (SMTT2: the
  compartments' T2 contrast does not depend on orientation, so it adds l = 0 information where
  l = 2 fails, a further reason for the TE question); planar or spherical b-tensor encoding,
  if the scanner has it; a prior, knowingly.
- The floor is mild in isotropic tissue (every direction at the spherical mean: the floor of the
  mean is the mean of the floors), as the dispersion runs in step 2 show.
- Tentative: v_ax from SMT or SMTT2, which treat both tissues alike; SMI's p₂ (from l = 2, no
  kernel diffusivities needed) as the measure of disorganization. EQUIP can quantify it without
  simulation: crlb of the l = 0 and l = 2 invariants as p₂ goes from nerve to neuroma (offered).

### Inputs needed from the lab

- Nerve tissue values: v_ax and D_ax (healthy, and the pathology to detect), the intra- and
  extra-axonal T2s, T1 (for TR).
- The scanner and sequence: the SNR of one b = 0 image at the shortest TE; the shortest TE for
  each b (gradient strength, δ, Δ, readout); the time per slice; the number of slices; today's
  protocol (b-values, directions, 10 averages) as the baseline.
- How the scanner combines averages (complex or magnitude).
- Whether the scanner can export complex (or magnitude and phase) images for diffusion EPI, and
  its coil combination; the existing complex- and magnitude-averaged nerve DTI (to measure the
  floor in the lab's data).
- The nerve's fiber dispersion (ODI or Watson κ: NODDI, the l = 2 invariant, or published
  values), which decides the floor correction (step 2, "Robustness of the corrections").
- The registration floor: the SNR needed, and of which image (the weakest direction, or the
  shell's mean).

## Building blocks: one signal per family, parts switched on and off (2026-11-03, branch `blocks-proto`)

(The families and `decay` were undone on 2026-11-06: "Reversed", at the end of this section. The
blocks remain, as EQUIP's exported diffusion compartments.)

The owner's request, after reading Microstructure.jl (roadmap item 7): a model family as one
function made of building blocks, where parts are turned on and off, rather than a type per
variant, "unless you're paying a huge computational price: it's cleaner from a maintenance and
code perspective". Tried on two families: the T2 decays (six models) and SMT (three).

### Decisions

1. **Blocks** (src/models/blocks.jl; public names since, `decay`, `ball`, `stick`, `zeppelin`,
   `powder`, below): `_relax(t, R)` = exp(−t·R) (R = 0: none); the diffusion
   compartments `_stick`, `_zeppelin`, `_ball`, each in two views chosen by dispatch on the last
   argument, a cosine t (one gradient direction) or `_SphericalMean()` (the average over
   directions); the zeppelin is a stick with D∥ − D⊥ times exp(−b·D⊥), so both views follow from
   the stick's. `_sphmean` moved there from smt.jl; `_optobs` reads an acquisition field a model
   may not have (SMT's TE: `false`, Julia's strong zero, without it).
2. **A part switched off is a role**: `Fixed(value; free = Free(...))`, fixed at a value that
   removes it (an offset 0, a fraction 0, a T2 of Inf), with the role it takes when a call
   switches it on: `fit(m, y; C = Free())` takes the template's start and bounds (the call's own
   win). A complete set of values does not need it (`simulate`, `signal`, `nominal`, a
   scenario's truth fill it in at its value); a real parameter such as T1MFA's B1 is still
   required, as decided for strict values. Two predicates (models/interface.jl): `_switchable`,
   a Fixed with such a role (filled in), and `_isoff`, switchable at a value that removes the
   part, 0 or Inf: shown as `off` in the model's display, and left out of `leakage` (a relative
   error of 0 or Inf means nothing; fix one in the call, `C = 0.0`, to include it). SMTFreeWater's
   `Diso` = 3e-3 is switchable but not off: leakage keeps it. A scenario keeps the role when freed
   (`base(r::Fixed, ...)` in errors/scenario.jl). A direct `signal!` call needs complete values,
   the parts off included (`resolve(m, θ)` first), as documented; the CLI lists the parts off
   among a model's flags (`--C`, `--fshort`, ...).
3. **One family, small named types.** `AbstractT2Decay` (T2Decay, T2DecayOffset, T2BiexpDecay),
   `AbstractR2Decay` (the rate forms) and `AbstractSMT` (SMT, SMTFreeWater, SMTT2): one
   `signal!` per family (T2's time form converts to rates and calls the rate kernel, as before),
   one table of parameters (`_t2_parts`, `_r2_parts`, `_smt_parts`), and each named model a
   struct whose `parameters` change a few roles. Kept as types, rather than one type with a
   preset type parameter or presets as named tuples of roles: the generic code that rebuilds a
   model from its fields (`_with_precision`, `ProtocolDesign`), `Scenario`, `parameters(T)` and
   the display names then work unchanged, and `T2DecayOffset(TE = ...)` stays what users write.
4. **Order kept**: each table lists the parameters so that every named model's free parameters
   keep the order they had (S0, fshort, T2short, T2, C), so its fits are unchanged to the bit.
5. **Names**: the bi-exponential's long component is `T2` (`R2`), as the mono-exponential's
   (it was `T2long`, `R2long`); 50 uses in tests, bench, precompile workloads, README.
6. **Infinite fixed numbers are allowed** (a T2 of Inf). The per-voxel check now depends on where
   a fixed value comes from: a map's value must be finite (NaN or Inf marks a bad voxel, as
   before), a number must not be NaN.
7. **Off parts are skipped**: a part whose weight is a plain number at exactly zero (`_skip(x::
   AbstractFloat)`) is not computed, and a relaxation with rate 0 is 1 without an exp. A dual
   number is never skipped, at zero too: it is being differentiated (a free parameter, or crlb's
   and leakage's seeded fixed values). Tested: ∂S/∂fshort at fshort = 0 is S0·(e_short − e_main).
8. **SMT's parts**: free water `fiso` (with `Diso`), a T2 per compartment (`T2ax`, `T2ex`,
   `T2iso`, Inf: no T2 weighting; `TE` an optional field of every SMT model), and the
   extra-axonal diffusivities as ties, `Dexpar = :Dax` and `Dexperp = p -> (1 − p.vax)·p.Dax`
   (Kaden's assumptions), released in a call (`Dexperp = Free()`: SMT without the tortuosity
   limit, as Microstructure.jl's SMT guide fits it). orientation.jl's three `_profile`s are one,
   the same tissue function in the directional view; `Union{SMT,SMTFreeWater,SMTT2}` is
   `AbstractSMT`. New combinations come free: two T2 components with an offset; SMT with free
   water and compartment T2s together.

### Measured

- **Results** (bench snapshot of every built-in model, both precisions, 300 voxels at SNR 100,
  starts and fits; scratch scripts snapshot.jl and compare.jl): the T2 family **bit-identical**
  to before in Float64 and Float32 (starts, values, SSE, iterations, statuses), the models not
  refactored too. The SMT family differs at rounding level, from one source: the zeppelin's
  anisotropy is now D∥ex − D⊥ex = D_ax − (1 − v)·D_ax, not v·D_ax. Float64: at most 4e-12
  (relative) for SMT and SMTFreeWater, 2.7e-4 for SMTT2, whose fit at SNR 100 is poorly
  determined (S0's SD 4800) and amplifies rounding; Float32: single voxels move (SMTT2 v_ax by up
  to 0.13), with the same convergence counts and the same means and SDs to every digit shown.
- **Speed** (4 threads, 20,000 voxels, best of 3 runs alternated between the old code and the
  new, µs per voxel):

  | model | before | after | ratio |
  |---|---|---|---|
  | T2Decay | 0.40 | 0.44 | 1.10 |
  | R2Decay | 0.49 | 0.56 | 1.14 |
  | T2DecayOffset | 1.04 | 1.13 | 1.09 |
  | R2DecayOffset | 0.81 | 0.95 | 1.17 |
  | T2BiexpDecay | 4.79 | 4.98 | 1.04 |
  | R2BiexpDecay | 6.34 | 6.42 | 1.01 |
  | SMT | 0.61 | 0.59 | 0.97 |
  | SMTFreeWater | 1.20 | 1.08 | 0.90 |
  | SMTT2 | 5.06 | 4.61 | 0.91 |

  The mono-exponentials pay 4–17% (0.04–0.14 µs per voxel, 0.1 s for a million voxels): the
  extra fixed parameters carried through the fit. Before skipping (decision 7), the off parts
  cost more: T2Decay's signal 2.4× and its fits +45%, SMT's signal 2× and fits +12%. The signal
  timings alone vary ±30% between runs on this machine; only the alternated fit timings are
  comparable.

- **Blocks against written-out equations** (bench/blocks_cost.jl, the owner's question: is
  building by parts slower than writing the equation out?). Each equation three ways in one
  process, interleaved, best of 7, 4 threads:

  | equation | how | signal! ns | fit µs/voxel | first fit s |
  |---|---|---|---|---|
  | T2 decay, 32 TEs | written | 93.0 | 0.445 | 0.57 |
  | | blocks | 107.7 | 0.474 | 0.44 |
  | | family (R2Decay, parts off) | 116.6 | 0.501 | 0.71 |
  | ADC, 7 b | written (ADCDecay) | 22.6 | 0.234 | 0.60 |
  | | blocks (`_relax(b, ADC)`, now `decay`) | 21.0 | 0.184 | 0.37 |
  | SMT, 4 shells | written (before) | 60.7 | 0.583 | 1.02 |
  | | blocks | 59.8 | 0.555 | 0.52 |
  | | family (SMT, parts off) | 70.3 | 0.556 | 0.87 |

  The blocks cost nothing that can be measured: they are inlined, and the ADC built from them
  came out faster than the identical equation written out (0.18 against 0.23 µs), which says
  that differences below about 20% are this machine's noise. What the T2 family pays, about 13%
  of a 0.45 µs fit (0.06 µs per voxel), is the switched-off parts carried as fixed parameters
  and their skip tests, consistent with the alternated timings above; SMT pays nothing, its cost
  being in erf and exp. Diffusion hides the overhead, T2 shows it: one exp per echo is so cheap
  that anything else is a visible fraction. Compilation (the first fit) stays under a second
  either way. In a plain-number call (the signal! column) `_relax` tests its rate for zero; in a
  fit the rate is a dual number and the test is not compiled at all.
- **One exponential unit**: `decay(x, rate)` = exp(−x·rate) is it (named `_relax` at first),
  for a T2 over TE (rate 1/T2), an ADC over b (rate D) and any decay; `ball(b, D)` is the same unit, and the stick's directional
  view the unit at b·t². With the stick's spherical mean (√π·erf(x)/(2x)) and the rules for
  combining (fractions that sum, products for a compartment's T2 weighting times its
  diffusion), these are the pieces T2 and SMT are made of.

- **Public names: `decay`, `ball`, and an explicit view** (2026-11-04, decided with the owner).
  The blocks are a module, `EQUIP.Blocks`, exporting `decay(x, rate)` = exp(−x·rate), `ball`,
  `stick`, `zeppelin` and `powder`:
  - Plain names, since the leading underscore is Julia's mark of an internal name and these are
    for writing models; a submodule rather than exports from EQUIP, so that `stick` and `ball`
    stay out of user code that does not ask for them (`using EQUIP.Blocks` brings them in).
    Internals (`_skip`, `_sphmean`, `_optobs`) keep the underscore.
  - One exponential unit, in rate form, named `decay`: "relax" reads wrong for b ("for diffusion
    we have an exponential decay with b"). A time constant is given as its rate, `decay(TE,
    1/T2)`; the model's parameter, T2, is still what is fitted. The risk, written into its
    docstring: `decay(TE, T2)` is exp(−TE·T2) and nothing reports it. `ball(b, D)` stays as its
    own name (it is `decay(b, D)`), so diffusion models read like the compartment taxonomy.
    (Tried first, the same day: `relax(t, T)` in the T2 form beside `relaxrate(t, R)`; replaced.)
  - The view explicit: `stick(b, D, powder)` for the spherical mean (NMR's powder average),
    `stick(b, D, ξ)` along one direction; `stick(b, D)` without a view is an error that says
    which to write. (Tried first: the spherical mean as the two-argument method; the owner
    preferred it explicit.) `powder` is the one value of a type, `Powder`, so the view is chosen
    by dispatch, when compiling.
  - Measured: a model written `S0 * decay(TE, 1/T2)` gives T2Decay's fits bit for bit (tested,
    from the same start), and is no slower (examples/building_blocks.jl, 20,000 voxels, 32
    echoes, 4 threads: 0.95 µs per voxel against T2Decay's 1.05): 1/T2 does not change between
    echoes, so the compiler hoists it out of the loop. Every built-in model's fits are
    bit-identical across the renames (the snapshot, both precisions).

- **Directions per image: three vectors of numbers** (2026-11-04, decided with the owner). A
  directional model (examples/directional_models.jl) takes each image's b and gradient direction.
  Every generic routine assumes a per-observation acquisition field is a vector of numbers: the
  signal's number type (`_signaltype`), the conversion to the fit's precision, the length checks,
  the protocol designer's static vectors. A field of 3-vectors broke `signal` (`float(Any)`) and
  stayed Float64 in a Float32 fit, for an `@mrimodel` and a hand-written model alike (tested).
  The rule that TD and TR already follow (filled to one value per observation in their
  constructors) is kept: the components go in as `gx, gy, gz = eachrow(bvec)`, views of a .bvec's
  rows, with no change to EQUIP. Tested: `signal`, Float64 and Float32 fits correct (the fit
  copies every field into a plain vector when it starts, so the views cost nothing there), the
  lengths checked. A model holds the views: after changing a bvec, build a new model. A b-tensor
  per image would follow the same rule, as six component vectors.
- **Revised (2026-11-05, with the owner): one direction per image, as a vector, `g`.** The
  owner's reason: a gradient direction is a vector, so it is passed as one. `g` holds one
  `SVector{3}` per image (`Vector{SVector{3,T}}`), beside `b`; a model reads `x, y, z = g[j]` or
  `dot(g[j], n)`. It gains no speed (the same three numbers are loaded) and no capability: what it
  gives is one field instead of three, a direction as one value, `g = directionset(30)` directly
  (which already returns SVectors), and, for b-tensor encoding later, one matrix per image instead
  of six component vectors.
  - **EQUIP needs two methods** (src/models/interface.jl): the conversion to the fit's precision
    for a vector of static vectors (`Vector{SVector{N,T}}`), and `_signaltype` taking the number
    type inside each entry. Tested from outside in a scratch script (dti_g.jl: DTI, 6 + 30
    images, SNR 50, 2000 voxels): without them `signal` fails (`float(Any)`); with them `signal`,
    `simulate`, Float64 and Float32 fits (`precision = Float32`; the fit's model then holds
    `SVector{3,Float32}`) and crlb all work, every voxel converged, Dxx 1.6996e-3 ± 3.19e-5 in
    both precisions, crlb 3.17e-5. `nobs` and `@mrimodel` already handle it (a vector, one entry
    per observation; the macro's body gets `g[j]`, one direction). Also to add: a clear error for
    a field that is neither numbers nor static vectors.
  - **The alternatives, measured or read.** gx, gy, gz (the 2026-11-04 rule) works and needs
    nothing, but takes three fields. An nobs×3 matrix with `@views gx, gy, gz = g[:, 1], g[:, 2],
    g[:, 3]` in `signal!` works by hand (scratch dti_matrix.jl: the same fits, 3.1 µs per voxel),
    but a Float32 fit kept it in Float64, `nobs` does not check its rows (only `eachindex` in the
    model's own loop caught a short one), and `@mrimodel` passes a matrix whole, so the body
    cannot index it; teaching the macro "rows are observations" would set that rule for every
    model's matrix fields. A named tuple of three vectors needs four methods and the same macro
    change.
  - **Files.** `readbvec` stays in MRImages as it is, the file's 3×N matrix: MRImages is a
    general package (like nibabel), without StaticArrays (which also invalidates MRImages'
    compiled code, "MRImages" above), and a matrix is the easier form for rotating or flipping
    directions. EQUIP converts: `diffusionencoding(bval, bvec) -> (b = ..., g = ...)`, from files
    or arrays, splatted into any directional model (`DTI(; diffusionencoding("dwi.bval",
    "dwi.bvec")...)`), with the checks (lengths, unit vectors, small b to 0), and `loaddata`
    mapping a .bvec to `g`. Named with the owner: not `diffusion` (too broad), and not `diff…`,
    which in Julia and MATLAB reads as "differences".
  - **One supertype, a struct per model.** `abstract type DirectionalDiffusion <: MRIModel` (models
    with fields `b` and `g`) carries what is shared, written once: the construction from files,
    the checks, a summary display (shells and counts, not every vector), a default S0 start, the
    command line. Julia types cannot inherit fields, so each model keeps its own four-line
    struct. Rejected: one struct with a tag (`Diffusion{:DTI}`): dispatch works, but EQUIP
    rebuilds models from the bare type (`Base.typename(typeof(m)).wrapper(...)`, in
    `_with_precision` and ProtocolDesign), which would drop the tag, and every model would print
    as "Diffusion" (the reason the T2 and SMT families kept named types). A macro that writes
    the struct (`@directional DTI`; tested, the Float32 rebuild included) waits until there are
    more than about five such models or their shared fields change together.
  - **Built-in models written with `@mrimodel`** (the owner's suggestion): new single-equation
    models (DTI, StickZeppelin, IVIM, perhaps DKI) with the macro, as compact as their equations;
    by hand, QMTSIR (the full layout as the reference), the T2 and SMT families (one equation for
    several named types, which the macro cannot express), and signals that do not split by
    observation (EPG). The macro's measurements allow it (roadmap item 8: T2DecayOffset, T1MFA
    and QMTSIR bit for bit and as fast), and `newmodel(M)` writes any of them out in full. The
    macro needs a supertype option (`... <: DirectionalDiffusion`) and its own precision
    conversion checked with `g`. The existing simple models (the T1 models, ADCDecay,
    ADCKurtosis) move one at a time, each with the refactoring check.
- **Revised again (2026-11-06, the owner): back to gx, gy, gz, converted once from the files.**
  Having seen both written out by hand (scratch options/optionA.jl and optionB.jl, the same
  fits), the owner prefers the three vectors, with one function doing the conversion:
  `diffusionencoding(bval, bvec) -> (b = ..., gx = ..., gy = ..., gz = ...)` (file names, or a
  vector and a 3×N or N×3 matrix; the checks as above; the rows copied, so a model does not
  hold views of the user's matrix), splatted into any directional model, `DTI(;
  diffusionencoding("dwi.bval", "dwi.bvec")...)`. It needs no change to EQUIP's generic code (the
  two methods for static-vector fields are not needed), and every routine keeps its rule: a
  per-observation field is a vector of numbers. The rest of the 2026-11-05 decisions stand:
  `readbvec` unchanged in MRImages; the name `diffusionencoding`; `DirectionalDiffusion`, now
  with fields b, gx, gy, gz, a struct per model; new single-equation built-ins with
  `@mrimodel` (its supertype option still to add). Still to add: a clear error for a field of
  3-vectors or a matrix, pointing to `diffusionencoding`. A b-tensor per image, when it comes,
  as six component vectors (the 2026-11-04 rule).
- **Built (2026-11-06, branch `directions-proto`), with these decisions:**
  - **`diffusionencoding`** (src/io/encoding.jl, exported): from file names or arrays (a 3×N
    .bvec, or N×3; a 3×3 matrix read as a .bvec), returning plain `Vector{Float64}`s, copied. The
    b-values are **kept as given**: the threshold (50 s/mm²) only says where a direction must be
    a unit vector (within 1e-2; FSL writes 0 0 0 at b = 0). Proposed first: set b below it to 0,
    as a cleaning step; dropped, because a b of 5 s/mm² is a real b-value and the model is more
    accurate with it (DIPY's `b0_threshold` likewise marks the b = 0 images and changes no
    value). Errors say what is wrong: a direction per b-value, the matrix's shape, which image
    is not a unit vector and its length. `readbvec` is exported from EQUIP too.
  - **`loaddata`** reads a `.bvec` beside the `.bval` into `found[:gx]`, `[:gy]`, `[:gz]`
    through `diffusionencoding` (its errors given with the file's name), so the command line's
    model building, which takes every found value whose name is a field of the model, gives a
    directional model its directions with no flag; `equip info` labels them as from the .bvec.
  - **No supertype** (the owner, 2026-11-06: "do we need a supertype? since we are using gx, gy,
    gz"). Everything planned for `DirectionalDiffusion` works by field name: the keyword
    constructors take `diffusionencoding`'s result, `loaddata` and the command line map by
    name, the checks are in `diffusionencoding`, and the generic display already summarizes a
    long vector (`b = 131 values from 0.0 to 2600.0`, checked). Add one when code must dispatch
    on "a directional model" (a spherical mean of any of them, a shared log-linear start):
    changing `<: MRIModel` to `<: DirectionalDiffusion` breaks nothing. The macro's supertype
    option is not needed either.
  - **Two blocks**, the owner's names (the second since replaced by Julia's `⋅`, below):
    `fiberdirection(θ, φ)`, the unit vector as a tuple, and `fibercosine(g, n)`, the cosine
    between one image's gradient direction and the
    fiber (the dot product; for unit vectors also the projection; the view `ξ` of `stick` and
    `zeppelin`). Plain tuples: no StaticArrays needed (measured: 0 bytes, the same speed as
    the cosine written out or as `SVector(gx[j], gy[j], gz[j]) ⋅ n`, which differs by 8e-16).
    Tried and dropped on the way: a `gdotn(g, n, j)` on `g = (gx, gy, gz)` (the owner disliked
    passing j; all images at once allocated 608 bytes a call and fitted 70% slower); one
    function taking the angles, `fibercosine(gx[j], gy[j], gz[j], θ, φ)`, which the owner
    preferred to read but which recomputes the sines and cosines for every image: the
    compiler does not move them out of the loop (stick–zeppelin, 66 images, 4000 voxels: fit
    10.4 µs per voxel against 5.4 with n once per voxel; 9.7 with `sincos`); and our own `dot`
    or `⋅` (extending LinearAlgebra's `dot` to tuples is type piracy, changing it for every
    package; a `⋅` of our own would clash with LinearAlgebra's for anyone using both). The
    spelling is US throughout now: "fibre" became "fiber" in every comment, docstring and doc
    (the owner's choice, with the names). The cosine itself is **ξ**, not t (the owner,
    2026-11-06: t reads as time in MRI): SMI's notation, ξ = ĝ·n̂ in its kernel K(b, ξ)
    (Novikov et al., NeuroImage 2018), and the Legendre forms P_l(ξ) a later rotational-invariant
    view would use. Renamed in the blocks (`stick(b, D, ξ)`), their docstrings, `_profile` and
    the Legendre and Rician sums in orientation.jl, the examples, the tests, the Watson samplers
    in bench/; the nerve example's `c` too. Only names changed: the view is passed by position.
    And `fibercosine` takes two triples, `fibercosine(g, n)` with g = (gx[j], gy[j], gz[j]) and n
    from `fiberdirection` (the owner, 2026-11-06: three numbers for g beside a tuple for n "feels
    off"), so the line reads as the equation, ξ = g·n; symmetric in its arguments, tuples or
    SVectors alike, and allocation-free (a tuple of three numbers is the three loads). Then
    replaced by Julia's own dot (the owner: "or we could even do g⋅n!"): LinearAlgebra's `⋅`
    takes any iterable containers, tuples included, so `ξ = g ⋅ n` needs no block of ours.
    Measured against `fibercosine(g, n)` (stick–zeppelin, 66 images, 4000 voxels): signals and
    fits bit-identical in both precisions, nothing allocated, the same speed within noise (10.6
    against 11.4 µs per voxel, timed while the test suite ran). `EQUIP.Blocks` passes `⋅` on
    (`using LinearAlgebra: ⋅; export ⋅`), so `using EQUIP.Blocks` is enough; it is the same
    binding as LinearAlgebra's, so `using LinearAlgebra` beside it does not clash (checked).
  - **Angle models by hand, not with `@mrimodel`** (revising "built-in models written with
    `@mrimodel`" above for them): a macro body is one image's signal, so a direction made from
    θ and φ in it is made for every image. Measured in examples/directional_models.jl (132
    images, 300 voxels, 4 threads): by hand 12.8 µs per voxel, the same model with the macro
    23.1, the same fits. DTI, with no angles, still suits the macro. Giving the macro a "once
    per voxel" section would be new design work.
  - **One value per observation, checked**: `nobs` (where every fit, signal and crlb starts)
    refuses a vector field whose entries are not numbers, naming the field and the remedy
    (`diffusionencoding`), where before a field of 3-vectors failed later as `float(Any)` or
    stayed Float64 in a Float32 fit. The test is on the fields' types, settled when compiling:
    `nobs` takes 2–3 ns with or without it (T2Decay, QMTSIR), no allocation. A matrix is not
    checked: it is not a per-observation field (a lookup table, say).
  - Tests: diffusionencoding (arrays, files, the layouts, copies, b kept, every error), a
    .bvec in `loaddata` (and its errors), the blocks, a hand-written stick–zeppelin on them
    (its signal against the equation written out, no allocation, fits in both precisions),
    and the refusal of 3-vector fields by `signal`, `fit` and `nobs`. 27 tests; full runs: EQUIP 2106, CLI 124, MRImages 103.

- **Reversed (2026-11-06, the owner): no families, no `decay`; every model in the standard
  format, the blocks EQUIP's own diffusion functions.** In the owner's words: "outside of the
  decay block the rest is diffusion ... maybe we remove that and just use blocks for diffusion,
  consistent with Microstructure.jl"; "I don't want it to be its own module. These are just
  functions that are used to build models more easily"; "I want decay removed. I do not like how
  unreadable that code is, switching things on and off ... same for SMT"; "we want to maintain
  standard EQUIP format throughout". So the family refactor above (decisions 1–8) is undone, by
  `git revert` of its commit (8cc4bc7) with the later work kept:
  - **Every model written out, its own struct, `parameters` and `signal!`**, as before the
    refactor and as QMTSIR is: T2Decay, R2Decay, T2DecayOffset, R2DecayOffset, T2BiexpDecay,
    R2BiexpDecay (in their own file again, `T2long`/`R2long` as before, kept by the owner: they
    pair with `T2short`/`R2short`), SMT, SMTFreeWater,
    SMTT2, each its own equation, with nothing switched on or off. Gone with the families: the
    switchable roles (`Fixed(value; free = Free(...))`, `_switchable`, `_isoff`, the `off`
    display, leakage's leaving them out), the skip of a part at zero (`_skip`), infinite fixed
    numbers, SMT's optional TE and its ties for the extra-axonal diffusivities, and the
    combinations no named model had (two T2 components with an offset; SMT with free water and
    compartment T2s): write one as its own model.
  - **The SMT models are written with the diffusion blocks**, `stick(b, D_ax, powder)` and
    `zeppelin(b, D_ax, D⊥, powder)` (and `ball` for free water), each model its own `signal!`;
    the three directional profiles in orientation.jl likewise, one per model, in the ξ view.
  - **The blocks are plain functions of EQUIP, exported** (no module `EQUIP.Blocks`): `ball`,
    `stick`, `zeppelin`, `powder`, `fiberdirection` and `⋅`. `using EQUIP` is all a model
    needs; nothing clashed (no other name in EQUIP or its tests). They are the diffusion
    compartments only (Microstructure.jl's carry a T2 each, as the paper of 2026 below
    shows; here a compartment's T2 is written in the model); relaxation is written as its equation,
    `exp(-TE / T2)`, which cannot be mistaken (`decay(TE, T2)` was exp(−TE·T2), silently wrong:
    its docstring had to warn of it). No `decay`, public or internal.
  - **Measured, the refactoring check** (every built-in model, 300 voxels at SNR 100, starts,
    values, SSE, iterations, statuses, both precisions): against the code before this change
    (directions-proto as committed), **bit-identical for every model**: the standard-format code
    computes exactly what the family code did. Against the code before the family refactor: the
    T2 models and all others bit-identical; the SMT models differ at rounding level, from the
    one source recorded then (the zeppelin's anisotropy D_ax − D⊥, not v_ax·D_ax): Float64 at
    most 6e-13 (SMT), 3e-12 (SMTFreeWater), 2.7e-4 for SMTT2 (poorly determined at SNR 100);
    Float32, single voxels move. A model written `S0 * exp(-TE / T2)` gives T2Decay's fit to
    rounding (dividing by T2 rounds differently from multiplying by 1/T2, as T2Decay does).
  - bench/blocks_cost.jl is removed: it compared written-out equations, the blocks and the
    families (its numbers are in "Measured" above); the blocks' cost was nothing measurable.
  - Found on the way: `stick`'s docstring gave its spherical mean at b = 1000 s/mm², D = 2e-3
    mm²/s, as 0.606; it is 0.598 (√π·erf(√2)/(2√2)), now tested.
  - Tests: the family's 41 replaced by 18 on the blocks and the SMT models written with them
    (each signal against Kaden's equation, each profile's mean against the signal); full runs
    EQUIP 2084, CLI 124, MRImages 103.

- **"Compartments", not "blocks"** (2026-11-06, the owner). The literature's word: the ball,
  stick, zeppelin, tensor, cylinder, sphere and dot are the compartments of Panagiotaki et al.'s
  taxonomy (NeuroImage 2012), and Microstructure.jl calls them so; a model of them is a
  multi-compartment model ("kernel", in the standard-model and SMI papers, is one fiber
  bundle's signal, its compartments together). src/models/blocks.jl is
  src/models/compartments.jl, examples/building_blocks.jl is examples/compartments.jl, the
  README's section is "Compartments". What is not a compartment is placed by what it is:
  `powder` stays with them (the view that picks a compartment's spherical-mean signal, the
  owner: it belongs with the compartments); `fiberdirection` moves to orientation.jl, beside
  `directionset`, as the geometry that gives a directional compartment its view ξ = g ⋅ n. The
  functions keep their names; only files, comments and docs change.

- **`iso`, not `ball`** (2026-11-06, the owner, from Gong & Yendiki, "Microstructure.jl: A Julia
  package for probabilistic microstructure model fitting with diffusion MRI", Imaging
  Neuroscience 2026;4, doi:10.1162/IMAG.a.1102). Their compartments are Cylinder, Stick,
  Zeppelin, Sphere and Iso; Iso (section 2.2.1.5, Eq. 5) is isotropic Gaussian diffusion, used
  for CSF, an extra-cellular space, or with D = 0 for immobile water in ex vivo tissue, the
  "dot" (their SANDIdot). So `iso(b, D)` is Panagiotaki's ball and, at D = 0, the dot: no dot of
  its own is needed. Read from the paper, two more things: their Eqs. 6–8 write the cosine as
  ξ = g·n, the notation chosen here; and each of their compartments carries a T2, a factor
  e^(−t/T2) included only with multi-TE data (sections 2.2, 2.2.1), so EQUIP's compartments,
  diffusion only with a compartment's T2 written in the model, differ from theirs there. (This
  log had said "diffusion only, as Microstructure.jl's are"; corrected.)

- **The zeppelin as its own equation** (2026-11-06, the owner: writing it through a stick,
  `exp(−b·D⊥)·stick(b, D∥ − D⊥, view)`, "might be true mathematically, but I have never seen
  this"). Each view is now its own method, written as Gong & Yendiki write it: along one
  direction exp(−b·D∥·ξ² − b·D⊥·(1 − ξ²)) (their Eq. 7), and the spherical mean
  exp(−b·D⊥)·√π·erf(√(b(D∥ − D⊥)))/(2√(b(D∥ − D⊥))) (their Eq. 1), through the same helper
  `_sphmean` the stick's spherical mean uses (for b = 0 and the fit's dual numbers). Measured:
  the spherical-mean form computes exactly what it did, so SMT, SMTFreeWater and SMTT2 give
  bit-identical starts and fits, both precisions; the directional form rounds differently, at
  most 2.3e-15 relative (b to 5000 s/mm², every ξ), with the same fits in
  examples/directional_models.jl. EQUIP 2085 tests pass. Then written with the exponent split
  as the spherical mean is (the owner): S(ξ) = exp(−b·D⊥)·exp(−b·(D∥ − D⊥)·ξ²), every direction
  seeing D⊥ and only the excess depending on the angle, as one exp of the two exponents (no
  second exp per image); the paper's form stays in the comment.

- **D⊥ > D∥, and the spherical means' helper** (2026-11-06, discussed with the owner). The
  zeppelin's directional form is right for any D∥ and D⊥, but its spherical mean took √(b·(D∥ −
  D⊥)) and threw a DomainError once b·(D∥ − D⊥) < −0.05 (tested: D∥ 0.5e-3, D⊥ 1e-3 mm²/s at b =
  1000); a fit that strayed there would have stopped. Two levels, both decided:
  - **The compartment is defined for both signs.** For D⊥ > D∥ (an oblate tensor) the mean over
    directions of exp(+|a|·ξ²) is √π·erfi(√|a|)/(2√|a|): checked against the integral computed
    numerically (0.4396004 both), and `erfi` takes the fit's dual numbers. The helper has three
    branches: the series near 0 (either sign), erf for a > 0, erfi for a < 0.
  - **What is physical is the model's to say**, by its parametrization, as Microstructure.jl
    does ("to constrain D⊥ to be smaller than D∥, D⊥ is represented as D∥ multiplied by a
    fraction between [0, 1]", Gong & Yendiki 2026, section 2.2.1.1): a model that fits D⊥ fits a
    fraction `fperp` with bounds 0 and 1 and computes D⊥ = fperp·D∥; the struct (the
    acquisition) does not change. SMT's tortuosity limit is the case fperp = 1 − v_ax; no
    built-in model frees D⊥ today.
  - **The helper renamed for what it computes**: `_sphmean` was not the spherical mean but one
    factor of it (the owner found it confusing). It is `_meanexp(a)` = ∫₀¹ exp(−a·ξ²) dξ, the
    mean over directions of exp(−a·ξ²), from which every axially symmetric compartment's
    spherical mean is made (stick: _meanexp(b·D); zeppelin: exp(−b·D⊥)·_meanexp(b·(D∥ − D⊥));
    a cylinder's would be the zeppelin's with its restricted D⊥, Gong & Yendiki Eq. 2; a sphere
    is isotropic and needs none; a full tensor's has no erf form). Each method carries its
    full equation in a comment above it, with the paper's equation number. Writing the closed
    form out in every method instead was considered: each would repeat the three-way branch
    (0/0 at b = 0 or D∥ = D⊥, the infinite derivative of √a at 0, the erfi case).
  - Measured: every built-in model's starts and fits bit-identical to before (the positive
    branch is the same expression); the SMT, compartment and direction test sets pass (134).

- **The compartments live with the diffusion models** (2026-11-06, the owner):
  src/models/diffusion/compartments.jl, included first in the diffusion group.

- **DTI built in, written out by hand** (2026-11-07, the owner: "just build standard DTI like we
  showed way back when ... We have the @mrimodel macro we can use later on to make more compact
  code"). src/models/diffusion/dti.jl, in the standard format: the struct (`b`, `gx`, `gy`,
  `gz`), `S0` and the six elements, and gᵀDg written out in `signal!`, as the papers write it.
  This revises "built-in models written with `@mrimodel`" above for DTI; the macro is for making
  the code compact later.
  - **No `tensor` compartment, and no multi-tensor models** (discussed the same day). A tensor
    has only the directional view: its spherical mean is a one-dimensional integral,
    ∫₀¹ exp(−b·λ3·z² − b·(1 − z²)·(λ1 + λ2)/2)·I₀(b·(1 − z²)·(λ1 − λ2)/2) dz, the zeppelin's erf
    form when λ1 = λ2, with no closed form otherwise and an eigendecomposition per call. And a
    sum of full tensors is hard to fit: swapping two tensors (f ↔ 1 − f) gives the same signal,
    identical starts cannot separate them, and parallel fibers are nearly degenerate. A model
    with several fiber populations uses zeppelins with directions, which the compartments
    already give. Also considered for gᵀDg (scratch gdg.jl: 132 images, duals for 7 parameters,
    µs per signal): written out 2.11; `dot(g, D, g)` with D an SMatrix (Julia's three-argument
    dot is gᵀDg) 2.25; `g' * D * g` 5.92, all allocation-free and equal to rounding. Written out
    was chosen: the fastest, and g stays a tuple as in the other directional models. The names
    `diffusivity` and `adc` for a gᵀDg helper were dropped with it (`adc` reads as ADCDecay's).
  - **The start, `datastart`** (replaced on `loglinear-proto` by the generic `:loglinear`,
    "The log-linear start" below): the weighted log-linear fit, ln y = ln S0 − b·gᵀDg, linear in ln
    S0 and the six elements (the B-matrix design matrix), weights y² (Basser et al., J Magn
    Reson B 1994). The 7×7 normal equations are summed image by image in static arrays (no
    allocation), in Float64 at least (Float32 data too), and scaled to a unit diagonal before a
    Cholesky factorization with `check = false`, which reports a failure rather than throwing.
    Found on the way: a system the data cannot determine (one shell with no b = 0) is singular,
    but rounding can leave it a tiny pivot, so the factorization succeeds and gives a nonsense
    start. So a pivot below eps^(1/4) (1.2e-4) counts as undetermined. Measured (scratch
    pivots.jl, 50 rotated tensors per case at SNR 30: white matter, grey matter, CSF and
    2.5/0.1/0.1e-3; Float64 and Float32 data): the smallest pivot is 0.02 or more for every
    determined design (one b = 0 and six directions; 6 b = 0 and 30; two shells with or without
    b = 0; 4 + 64 + 64; directions within 3° of a plane), and 3e-7 or less for the undetermined
    ones (one shell with no b = 0; five directions; all directions in one plane, or on one cone,
    which cannot separate Dzz from Dxx + Dyy). When undetermined, the start is isotropic,
    1/mean(b), as ADCDecay's.
  - **Measured** (scratch check.jl: 6 b = 0 and 30 directions at b = 1000, a rotated tensor with
    eigenvalues 1.7, 0.4, 0.3e-3 mm²/s, SNR 50, 2000 voxels, 4 threads). From noise-free data
    the start is the tensor itself (5e-16 of λ1 in Float64, 7e-8 in Float32). All voxels
    converged in both precisions. The data start took 4.0 iterations on average (Float64),
    against 6.0 from a constant start (D = 1e-3·I), and gave the same fits (to 3e-8 of λ1;
    Float32, 7e-5). Time per voxel: 2.4–2.6 µs, against 2.9 from the constant start.
    examples/directional_models.jl now uses the built-in DTI, with the same results (its
    direction 0.2° from the fibers; v_ax 0.600 ± 0.004).
  - Not yet: FA, MD, AD, RD and the principal direction as maps (roadmap item 16's `propagate`
    for their errors); DTI in the command line (`equip fit DTI`, which `loaddata`'s .bvec
    reading already allows) and in the precompile workload (SMT is in neither).
  - Tests: 37 (the signal against the matrix form, no allocation; the start exact on noise-free
    data in both precisions, no allocation, and isotropic for the three undetermined designs;
    fits in both precisions against the truth, crlb's SD and a constant start's fits); full runs
    EQUIP 2119, CLI 124 (MRImages unchanged, 103).

- **The log-linear start, `Free(start = :loglinear)`, from the model's own signal** (2026-11-07,
  branch `loglinear-proto`; the owner: "we should be able to feed it signal instead of
  recalculating and writing a custom function for both it and DKI"). DTI's `datastart` restated
  the signal equation in a design matrix, and DKI would have needed a third copy. The log-linear
  fit is Gauss–Newton on ln y, so EQUIP computes it from what the solver already has: `signal!`
  evaluated in dual numbers gives S and ∂S/∂θ, hence the log residuals ln S − ln y and their
  Jacobian (∂S/∂θ)/S; weights y² (times the fit's own weights). Where ln S is linear in the
  parameters, the first step lands on the weighted log-linear fit from any point; where it is
  nearly linear, a few steps reach it. The code is in src/fit/starts.jl (`_loglinear_start`), one
  step of the voxel's start sequence, after `:data` and before `BestOf` and `:scale`; DTI's
  parameters say `Free(start = :loglinear)` and its `datastart` is gone.
  - **What moves:** the parameters at `:loglinear`, from 0 (their placeholder), and the `:scale`
    parameter, the fit's intercept (without it the others would absorb the signal's level); the
    rest stay at their starts. The scale is then set again by least squares on y, as for every
    start. Found on the way: ln S is linear in ln S0, not in S0, so stepped in S0 the start crept
    from S0 = 1 towards 1000 over all ten steps (48 µs for one voxel of 132 images instead of 9).
    Since `:scale` multiplies the signal by definition, its step is taken on ln S0,
    S0·exp(ΔS0/S0), which keeps DTI's first step exact.
  - **Kept from DTI's start:** Float64 sums and solve (Float32 fits too); the system scaled to a
    unit diagonal; the solver's own `_cholesky`, which reports failure rather than throwing; the
    pivot threshold eps^(1/4) for a design that cannot determine the parameters ("DTI built in",
    pivots.jl). New: a parameter the data do not measure at the current point (a zero column:
    DKI's W while D = 0; Dzz when every direction lies in the xy plane) does not move that step;
    a step that raises the weighted log-SSE is halved; it stops when the predicted decrease is
    below √eps of the log-SSE, or after 10 steps. A changed behaviour: where the design cannot
    give the parameters (one shell with no b = 0), they stay at 0 rather than at the isotropic
    1/mean(b); the in-plane case now gives the in-plane elements exactly, the others 0.
  - **For which parameters:** those on which ln S depends linearly or nearly, and for which 0 is
    a valid value: diffusivities, rates, kurtosis. Not time constants (T2 = 0 is no signal):
    T2Decay keeps its `datastart`, or a model uses the rate.
  - **Measured, DTI** (scratch loglin_dti.jl: a rotated tensor 1.7, 0.4, 0.3e-3 mm²/s, Rician SNR
    50, 2000 voxels, 4 threads), against DTI's former start (directions-proto):
    | protocol | starts differ (of λ1) | fits differ | iterations | µs per voxel, fit (start alone) |
    |---|---|---|---|---|
    | 6 b = 0 + 30 at 1000 | 1.7e-14 | 1.3e-16 | 4.044, the same | 2.31 against 2.23 (1.24 against 1.05) |
    | 4 + 64 at 1000 + 64 at 2500 | 4.1e-14 | 1.3e-16 | 4.447 against 4.448 | 7.21 against 6.20 (3.81 against 2.65) |
    Float32: starts within 7e-8 of λ1, fits within 9e-5 (Float32's tolerance), iterations the
    same to 0.01. No allocation. The cost, 4% and 16% of the fit, is two evaluations of the signal
    in dual numbers (the step, then the check that nothing is left) where the hand-written start
    summed the design matrix once.
  - **Measured, DKI** (scratch loglin_dki.jl: examples/dki.jl's model and data, 22 parameters, 66
    images, SNR 100, 2000 voxels, 4 threads), against its constant start (D = 1e-3·I, W = 0): all
    converged from both; 4.6 iterations on average (at most 7) against 15.7 (at most 17); the fit
    74.7 µs per voxel against 84.3 (the start itself 41 µs); the same fits (W within 5e-7). The
    first call, compiling, 8.0 s, as before. The start's W1111 has mean 1.05 and SD 0.25, the fit's
    0.99 and 0.25 (truth 1): the log fit's weighting is not the fit's, as expected.
  - **Why not a library's least squares** (the owner's question, the same day): it was measured
    on DTI's start (scratch wls_library.jl, 36 images, 1000 voxels): Julia's `\` (pivoted QR,
    LAPACK) with the rows weighted, `(y .* A) \ (y .* log.(y))`, gives the same answer (to 9e-18
    mm²/s) in 5.8 µs and 25 KB per voxel with the design matrix made once, 7.6 µs and 32 KB made
    per voxel, against 1.5 µs and nothing for the static normal equations: a million voxels
    would make about 30 GB of garbage, and collecting it stops every thread. (It also needs the
    same care for y ≤ 0: the test crashed on negative data at b = 2500.) The solve itself is a
    library's here too, the factorization of the small static system.
  - examples/dki.jl uses it: 69 µs per voxel, 4.6 iterations on average, all 2000 converged,
    the same means as before (Dxx 1.698e-3, W1111 0.988, W1122 0.332, MKT 0.997);
    examples/directional_models.jl, on the built-in DTI, the same results as before.
  - Tests: DTI's start checked through the fit (`LM(maxiter = 0)` returns the start): exact on
    noise-free data in both precisions, the in-plane and one-shell designs, no positive data, no
    allocation; and 6 for `:loglinear` itself (ADCDecay's start against `_loglinear_rate`, with
    and without weights; ADCKurtosis's against the weighted polynomial fit; `Free` accepting it).
    Full runs: EQUIP 2125, CLI 124.
  - **Against DIPY** (2026-11-07, the owner's question; DIPY 1.12.1, NumPy 2.5, SciPy 1.18, cvxpy for
    its DKI NLS; scratch dipycmp/equip_side.jl and dipy_side.py): the same simulated data, Rician,
    a white-matter tensor (1.7, 0.4, 0.3e-3 mm²/s) in a random orientation per voxel, 20 000
    voxels (DIPY's per-voxel loops on 2000, its DKI NLS on 500); DTI at SNR 50, DKI (isotropic
    K = 1) at SNR 100. µs per voxel; DIPY's times were the same with 1 and 4 BLAS threads (its
    fits loop over voxels in Python, or solve many small systems), and its parallel engines
    (joblib) are not reachable for DKI in 1.12 (the keyword reaches `ls_fit_dki` and fails).

    | | EQUIP, 1 thread | EQUIP, 4 threads | DIPY |
    |---|---|---|---|
    | DTI 36 images: weighted linear (EQUIP's `:loglinear` start; DIPY's WLS, its default) | 4.4 | 1.2 | 17.1 |
    | DTI 36 images: nonlinear (EQUIP's fit; DIPY's NLLS, MINPACK) | 8.6 | 2.3 | 102 |
    | DTI 36 images: DIPY's OLS (one pseudo-inverse for every voxel) | | | 1.4 |
    | DTI 132 images: weighted linear | 14.8 | 4.0 | 33.2 |
    | DTI 132 images: nonlinear | 28.3 | 7.5 | 158 |
    | DKI 66 images: weighted linear | 168 | 42 | 312–338 |
    | DKI 66 images: nonlinear (DIPY's NLS) | 320 | 81 | 208 |

    The nonlinear fits find the same minimum: DIPY's NLLS against EQUIP's fit, the D elements
    within 3e-7 of λ1 at most (median 7e-9; DKI 3e-7, W within 6e-6). The linear fits differ by
    their weights: DIPY's WLS weights each image by the square of the signal OLS predicts
    (Chung et al. 2006, the two-pass method), EQUIP's start by y²; the D elements differ by a median
    of 1e-3 (36 images) to 5e-3 (132) of λ1. That matters at high b: RMSE of the D elements against
    the truth, in units of λ1, 132 images: EQUIP's start 0.0112, DIPY's WLS 0.0062, DIPY's OLS
    0.0240, the nonlinear fits 0.0065 (EQUIP) and 0.0066 (DIPY); at 36 images all 0.011–0.013.
    Since the fit follows the start, EQUIP's estimates are the nonlinear ones either way.
    - DTI: EQUIP is 5–12× faster per core than DIPY's NLLS, 20–45× on 4 threads, and 2–4× faster
      per core than DIPY's linear WLS while also running the nonlinear fit after it.
    - DKI: DIPY's NLS is 1.5× faster per core than EQUIP (208 against 320 µs); EQUIP is faster
      only on threads (81). Two reasons, both EQUIP's model rather than its solver: DIPY
      parameterizes DKI so that ln S is linear (D with V = MD²·W, W recovered after), so its
      Jacobian is S times the design matrix, cheap, and its OLS start is already the log-linear
      fit; EQUIP's DKI (examples/dki.jl) multiplies W by MD², so its `:loglinear` start takes
      about five Gauss–Newton steps (168 µs, half the fit), and every evaluation carries 22
      dual partials through MD² and the 15 monomials. To try: the DKI model in DIPY's linear
      parameterization (the start then exact in one step), and weights from the predicted signal
      in `:loglinear` (DIPY's WLS: a second step with S(θ)² as the weights).
  - **EQUIP's fit against DIPY's WLS, the estimates users look at** (the owner: "also compare
    with WLS"; scratch dipycmp/accuracy.py, the same data, the same voxels for every method:
    2000 for DTI, 500 for DKI). Bias (mean − truth) and SD, relative to the truth (MD 0.8e-3
    mm²/s, FA 0.763, MKT 1):

    | | MD bias | MD SD | FA bias | FA SD |
    |---|---|---|---|---|
    | DTI 36: EQUIP's fit = DIPY's NLLS | −0.12% | 1.55% | −0.08% | 1.40% |
    | DTI 36: DIPY's WLS | −0.03% | 1.55% | −0.06% | 1.41% |
    | DTI 36: DIPY's OLS | 0.00% | 1.58% | −0.01% | 1.72% |
    | DTI 36: EQUIP's start | −0.47% | 1.55% | −0.26% | 1.41% |
    | DTI 132: EQUIP's fit = DIPY's NLLS | −0.53% | 0.98% | −0.17% | 0.67% |
    | DTI 132: DIPY's WLS | −0.20% | 0.98% | −0.09% | 0.68% |
    | DTI 132: DIPY's OLS | −2.72% | 2.85% | −1.92% | 1.73% |
    | DTI 132: EQUIP's start | −1.98% | 1.00% | −0.64% | 0.69% |

    DKI 66 (MD, then MKT): EQUIP's fit = DIPY's NLS +0.14%, 1.35%; +0.39%, 3.93%. DIPY's WLS
    +0.19%, 1.35%; +0.58%, 3.92%. DIPY's OLS +0.13%, 1.41%; +0.10%, 4.40%. EQUIP's start
    +0.20%, 1.35%; +1.37%, 3.91%.
    So the nonlinear fit and DIPY's WLS are equally precise (the same SD everywhere); their biases
    differ by a few tenths of a percent, either way (WLS less in DTI's MD at b = 2500, the
    nonlinear fit less in MKT), far below the SD. Both fit a Gaussian model to Rician data, so
    the floor at high b pulls MD down; EQUIP's `RicianMean` would model it (not tried on DTI).
    EQUIP's start, weighted by y², is the most biased of the weighted fits at b = 2500 (MD −2.0%):
    another reason for predicted-signal weights in `:loglinear`. Time, µs per voxel, EQUIP's
    nonlinear fit against DIPY's WLS: DTI 36 images 8.6 (1 thread) and 2.3 (4) against 17.1; 132
    images 28.3 and 7.5 against 33.2; DKI 320 and 81 against 312–338.
  - **Predicted-signal weights in `:loglinear`** (2026-11-07, the owner: "test predicted-signal
    weights"; a prototype switch, `EQUIP._LOGLINEAR_WEIGHTS[]`, default `:data`; scratch
    dipycmp/weights_equip.jl and accuracy_weights.py, the DIPY comparison's data). The weights
    of the log fit, four ways: `:data`, y² (as built); `:twopass`, one unweighted step, then one
    weighted by the predicted signal² and stop (DIPY's WLS; DTI only, since DKI's W waits a
    step); `:predicted`, the predicted signal² at every step until the predicted decrease is
    below √eps (iteratively reweighted; the first step is unweighted by itself, the signal being
    flat at the placeholder start); `:predictedstop`, the same, taking a step and stopping when
    it gains under 1% of the weighted log-SSE. Bias in % of the truth (SD the same for all
    within 0.02%), and µs per voxel on 4 threads (the start; the whole fit, its LM iterations):

    | | `:data` | `:twopass` | `:predicted` | `:predictedstop` | DIPY's WLS | EQUIP's fit |
    |---|---|---|---|---|---|---|
    | DTI 36: MD bias | −0.48 | −0.04 | −0.01 | −0.01 | −0.04 | −0.13 |
    | DTI 36: time (fit, iterations) | 1.27 (2.31, 4.04) | 1.18 (2.29, 3.81) | 2.29 (3.38, 3.91) | 1.56 (2.61, 3.91) | | |
    | DTI 132: MD bias | −1.97 | −0.20 | −0.08 | −0.08 | −0.20 | −0.53 |
    | DTI 132: time (fit, iterations) | 3.85 (7.32, 4.48) | 3.91 (6.89, 4.00) | 7.98 (10.9, 4.04) | 5.27 (8.52, 4.04) | | |
    | DKI 66: MKT bias | +1.08 | | −0.25 | −0.21 | +0.29 | +0.10 |
    | DKI 66: time (fit, iterations) | 41.8 (81.5, 4.69) | | 42.7 (80.0, 4.35) | 31.9 (67.7, 4.32) | | |

    `:twopass` is DIPY's WLS exactly (within 1e-13 of λ1), which checks the implementation, and
    costs nothing: the same two evaluations. Weighting by the prediction removes the start's
    bias (y²'s noisy weights caused it); iterating the weights removes a little more, even
    below the nonlinear fit's own Rician bias. `:predictedstop` is the one rule that does well
    on both: the least biased start, DKI 17% faster than now (fewer start steps and LM
    iterations), DTI 13–16% slower (it usually takes a third step). The fit's estimates are the
    nonlinear fit's whichever start is used (all converged in every voxel).
    - **Decided (the owner, the same day): predicted-signal weights, reweighted at every step,
      stopping after a step that gains under a quarter of the weighted log-SSE**; the switch is
      gone, and the y² weights with it. The cutoff, measured (scratch cutoff_equip.jl, µs per
      voxel on 4 threads, the whole fit; every cutoff converged in every voxel):

      | cutoff | DTI 36 | DTI 132 | DKI 66 | start's MD bias, DTI 132 |
      |---|---|---|---|---|
      | 1% | 2.79 | 8.57 | 66.0 | −0.08% |
      | 5% | 2.44 | 8.74 | 64.9 | −0.09% |
      | 10% | 2.34 | 8.62 | 64.6 | −0.09% |
      | 25% | 2.26 | 7.24 | 65.7 | −0.17% |

      At 25% the one-shell DTI start is DIPY's WLS exactly (the second step gains little); every
      cutoff's start is within 0.04 SD of the start reweighted to convergence, so the loosest,
      fastest one costs nothing that matters for a start. Final, the same data (scratch
      equip_side.jl, accuracy.py; µs per voxel, 1 and 4 threads, the whole fit, with y²
      weights before): DTI 36 images 8.70 and 2.39 (8.57, 2.30); 132 images 27.5 and 6.96
      (28.3, 7.47); DKI 263 and 64.7 (320, 81.4): DKI 18–21% faster, DTI the same within noise.
      The start's bias, now beside DIPY's WLS: DTI 132 MD −0.18% (DIPY −0.20%), FA −0.08%
      (−0.09%); DKI MKT +0.09% (+0.58%). Against DIPY's NLS per core, DKI is now 263 against 208
      µs (1.26× slower, from 1.5×). Tests: the start exact on noise-free data (ADCDecay,
      ADCKurtosis, DTI) and within 0.05 SD of the fit reweighted to convergence, written out in
      the test (measured 0.006 SD for ADCDecay, 0.001 for ADCKurtosis).
      Full runs: EQUIP 2128, CLI 124; examples/dki.jl gives the same estimates (MKT 0.997, 4.2
      iterations).
  - **DKI in DIPY's parameterization** (2026-11-07, the owner: "try it and time against dipy";
    scratch dipycmp/dkiv.jl; examples/dki.jl now written this way). DIPY fits V = MD²·W in place
    of W (`dki_design_matrix`, then `result[6:21] / MD_square`), so ln S = ln S0 − b·gᵀDg +
    b²·ΣgᵢgⱼgₖgₗVᵢⱼₖₗ/6 is linear in all 22 parameters; W = V/MD² after the fit. In EQUIP the
    signal loses its MD² factor and the 15 kurtosis parameters become V (in (mm²/s)²). The
    comparison's DKI data (20 000 voxels), µs per voxel:

    | | 1 thread: start | 1 thread: fit | 4 threads: start | 4 threads: fit | iterations |
    |---|---|---|---|---|---|
    | W form | 119 | 249 | 31.8 | 64.0 | 4.31 |
    | V form (DIPY's) | 60.8 | 138 | 15.5 | 34.7 | 4.35 |
    | DIPY NLS (serial) | | 208–232 | | | |
    | DIPY WLS, its default (serial) | | 312–348 | | | |

    The same fits: W from V within 7e-7 of the W form's (median 4e-9), D within 3e-8 of λ1,
    the same RMSE against the truth (0.0701). The start halves (ln S now linear: the
    reweighted steps stop early) and so does each LM iteration (the kurtosis terms no longer
    carry D's derivatives through MD²). So EQUIP's DKI is now 1.5–1.7× faster per core than
    DIPY's NLS and 6–7× on 4 threads; 2.3–2.5× and 9–10× than DIPY's default WLS. The example
    (2000 voxels, 4 threads): 47 µs per voxel, 4.3 iterations, the same estimates as before to
    every printed digit (MKT 0.9972, W1111 0.9878); K ≥ 0 checked as gᵀVg/(gᵀDg)², the same
    sign as in W. The cost: V is what the fit returns, W is computed after it (one line in the
    example; roadmap item 16's `propagate` would give its errors).
  - **Reviewed, and revised (2026-11-07): exactly two passes.** The owner brought a review of
    the fitting (from a chat), its issues ranked: (1) parameter scaling and conditioning, high;
    (2) bounds, fixed parameters and AD consistency, high; (3) the low-SNR noise model, high for
    MRI accuracy; (4) the log start's safeguards, moderate; (5) the reweighting stopping rule,
    low to moderate; (6) the extra WLS passes, low. On 5 and 6: the 25% rule compared each
    step's predicted gain with the log-misfit at the same point and weights, but that misfit is
    mostly noise, so the rule depends on the SNR, and in parameter terms it is loose (a
    Mahalanobis step of up to √((n − p)/2) standard errors, about 4 for 36 images); it worked
    because each reweighting moves the estimates little, not by design. Decided: a fixed two
    passes, OLS then WLS weighted by OLS's predicted signal² (Chung et al., NeuroImage 2006;
    DIPY's WLS), for models whose ln S is linear in their parameters; for others two Gauss–
    Newton steps, an approximate start that LM refines (rewrite such a model to be linear, as
    DKI in V, or use `:data`). The first pass is unweighted by rule: with a parameter started
    from a map the signal at the start is not flat, and weighting by it would make the first
    pass something other than OLS. Measured (the DIPY comparison's data, µs per voxel on 1 and 4
    threads, the whole fit; with the 25% rule before): DTI 36 images 8.13 and 2.23 (8.70, 2.39);
    132 images 24.3 and 6.40 (27.5, 6.96); DKI in V 119 and 34.8 (138, 34.7); DKI in W 279 and
    70.5 (263, 64.7, 6.1 iterations instead of 4.3: two passes are not enough where ln S is not
    linear, and the W form is no longer used). On 1 and 2, step 1 of the response, tested:
    - **Float32 at 22 parameters** (DKI in V, 200 voxels, SNR 100): every voxel converged in
      both precisions, Float32 within 0.005 SD of Float64 (measured on 500 voxels; tested at
      0.01 SD). Scaling: LM's Marquardt scaling makes its damping and tests unit-free, and a
      Cholesky factorization's accuracy depends on the equilibrated matrix's condition, not on
      the raw scales (S0 ~ 1000, D ~ 1e-3, V ~ 2e-7); the start equilibrates explicitly.
    - **The start against the two passes written out** in the tests, to 1e-12 or better: DTI
      (measured 1.3e-15 of λ1); DTI with the off-diagonal elements fixed (the diagonal's
      columns); with a tie, Dzz = Dyy (the two columns merged); with Dxx started from a map
      (held at the map's values, the others fitted to ln y − Xₓₓ·Dxx with weights from the whole
      prediction); DKI in V on DIPY's `dki_design_matrix`; ADCDecay. A bound reached in the
      start holds (an upper bound below the truth: the start and the fit stop there, converged;
      a signal rising with b: ADC starts at its lower bound 0).
    - Full runs: EQUIP 2147 (19 new), CLI 124.
    - **Issues 3 and 4, measured** (2026-11-07, the owner: "look at both"; scratch
      lowsnr/lowsnr.jl, A.txt and B.txt): DTI (4 b = 0, 64 directions at 1000 and at 2500) and
      DKI in V (6 b = 0, 30 at 1000 and 2000), Rician noise, SNR (at b = 0) 10–50 for DTI and
      20–100 for DKI, 2000 voxels a case, random orientations; WM (1.7, 0.4, 0.3e-3, DKI K = 1),
      GM (0.8e-3 isotropic), CSF (3.0e-3 isotropic, K = 0). Three estimators: the two-pass WLS
      (the start, DIPY's default), the Gaussian NLLS from it (EQUIP's fit), and least squares on
      the Rician mean, `RicianMean(m; σ)` with the true σ, started from the Gaussian fit (free,
      and with D's diagonal bounded to [0, 4e-3] mm²/s). Bias and SD in % of the truth (FA
      and MKT of isotropic tissue: their mean):

      | | SNR | WLS: MD bias | Gaussian NLLS: MD bias, SD | Rician: MD bias, SD | the other index: WLS, Gaussian, Rician |
      |---|---|---|---|---|---|
      | DTI WM | 10 | −10.6 | −11.4, 4.7 | +0.4, 5.3 | FA −3.3, −3.3, +0.1 |
      | DTI WM | 20 | −2.0 | −3.0, 2.4 | +0.2, 2.5 | FA −0.8, −0.9, 0.0 |
      | DTI WM | 30 | −0.8 | −1.4, 1.6 | 0.0, 1.7 | FA −0.3, −0.5, 0.0 |
      | DTI WM | 50 | −0.2 | −0.6, 1.0 | 0.0, 1.0 | FA −0.1, −0.2, 0.0 |
      | DTI GM | 10 | −6.8 | −11.7, 4.6 | +0.6, 5.6 | FA 0.114, 0.110, 0.123 |
      | DTI GM | 20 | −0.6 | −2.7, 2.4 | +0.1, 2.5 | FA 0.058, 0.057, 0.058 |
      | DTI CSF | 10 | −88.8 | −37.3, 3.4 | bounded: +14.0, 12.1 | FA 0.42, 0.15, bounded 0.49 |
      | DTI CSF | 20 | −78.1 | −15.9, 2.4 | bounded: +9.1, 9.9 | FA 0.26, 0.11, bounded 0.31 |
      | DTI CSF | 30 | −65.0 | −8.1, 2.1 | bounded: +2.4, 5.1 | FA 0.19, 0.09, bounded 0.16 |
      | DTI CSF | 50 | −39.3 | −3.0, 1.5 | +0.4, 1.8 | FA 0.11, 0.06, 0.07 |
      | DKI WM | 20 | +2.1 | +0.9, 7.2 | −0.4, 7.6 | MKT +10.3, +6.6 (SD 18.8), −8.2 (SD 26.7) |
      | DKI WM | 30 | +1.0 | +0.4, 4.7 | −0.1, 4.9 | MKT +4.7, +2.7 (13.1), −3.1 (15.1) |
      | DKI WM | 50 | +0.4 | +0.1, 2.9 | 0.0, 2.9 | MKT +1.9, +1.1 (7.8), −0.8 (8.1) |
      | DKI WM | 100 | +0.1 | 0.0, 1.4 | 0.0, 1.4 | MKT +0.5, +0.3 (3.9), −0.2 (4.0) |
      | DKI CSF | 20–100 | +27 to +38 | +24 to +32, 3–7 | fails (below) | MKT 0.33–0.48, 0.32–0.49 (truth 0) |

      - **The noise model (3).** In tissue the Gaussian fits are biased low in MD by the floor
        (−11% at SNR 10, −3% at 20, −1.4% at 30, −0.5% at 50, WM), and least squares on the
        Rician mean removes it (≤ 0.6% at every SNR) at the cost of a slightly larger SD at SNR
        10 (5.3% against 4.7%) and none from 20 up. The two-pass WLS is a little less biased
        than the Gaussian NLLS in tissue (its weights, from a prediction that sits below the
        floor-raised data, give the high-b images less say). The Rician fit is not a cure-all:
        (a) where the high-b images are all floor (CSF at b = 2500 is 0.05% of S0), any large D
        fits them, so unbounded it runs away (MD +10 000%, 20% of voxels not converged at SNR
        10); with D's diagonal bounded to [0, 4e-3] it converges (98–100%) but stays biased and
        noisy at low SNR (+14%, SD 12% at SNR 10; spurious FA 0.49); (b) for DKI's MKT it
        overcorrects at low SNR (−8.2% at SNR 20) with a larger SD, so its RMSE is worse than
        the Gaussian fit's (27.9% against 19.9% at SNR 20; 15.4 against 13.4 at 30); (c) DKI in
        CSF fails with or without bounds (6–92% converged): at b = 2000 CSF is 0.25% of S0,
        below the floor, and kurtosis and diffusivity trade off freely. The Gaussian DKI fit in
        CSF invents kurtosis (MKT 0.32–0.49, truth 0) and overestimates MD by 24–32% at every
        SNR up to 100: a property of the protocol and model, not the estimator (DKI in CSF is
        usually masked out). Cost: the Rician fit is 10–40× slower (150–2000 µs per voxel on 4
        threads against 7–120), the Bessel functions in dual numbers. σ must be known (the
        simulation's here; in data, from the background or a noise scan; EQUIP has no estimator).
      - **The start at the floor (4).** In WM and GM the two-pass start is within 1 SD of the fit
        (99th percentile under 2 SD) and LM converges in 4–7 iterations everywhere. In CSF the
        OLS pass is dragged by the floor at b = 2500: the start's MD is 39–89% low, 9–12 SD from
        the fit, and LM needs 8–15 iterations (all 2000 converge). Weighting the first pass by
        y² instead (variant B; the floor images then count little) brings CSF's start to 0.8–4
        SD and saves 1–2 iterations there, but moves WM and GM's starts slightly further (GM at
        SNR 20: 0.85 against 0.67 SD) and adds 0.1–0.3 iterations: no failures either way, so
        no change. In DKI CSF 3–12 voxels in 2000 reach the 100-iteration limit in both variants:
        the fit's ill-posedness, not the start's.
      - Options, for the owner: the Rician fit as a documented choice for DTI in tissue at
        SNR ≤ 30 (with D bounded, and `:loglinear` through `RicianMean` delegated to the wrapped
        model, which is not yet so: here it was started from the Gaussian fit); a faster
        `ricianmean` if it is used routinely; for DKI, the constraints (K ≥ 0, the signal not
        rising with b; DIPY's CWLS) and masking CSF matter more than the noise model.
  - Not yet: ADCDecay and ADCKurtosis (their starts are numbers
    from the acquisition) and T2Decay's rate model R2Decay, each with the refactoring check; a
    linearization point other than 0, if a model needs one. (Measured 2026-11-08, next entry:
    R2Decay and ADCDecay keep their constant starts.)
- **T2Decay on the shared log-linear start; time constants at `:loglinear`; the first pass made
  once per fit; the voxel fit called as a function above 8 parameters** (2026-11-08, branch
  `loglinear-time`; the owner: "I don't like having a one-off datastart just for that"; scratch
  llt/).
  - **Why T2Decay needs a data start and the rate forms do not** (the owner asked: "it makes zero
    sense"). The solver steps in the parameter it fits. ln S is linear in R2, so a step from any
    constant lands near the answer; in T2 a step goes through R2 = 1/T2, ΔT2 = −T2²·ΔR2, which for
    an exact rate step gives T2_new = T2·(2 − T2/T2*): at most a doubling towards a long T2, an
    overshoot below zero from beyond 2·T2*. Traced on a noise-free curve (32 echoes, T2 = 10 s,
    from mean(TE) = 0.165 s): 0.165, 0.165, 0.165, 0.33, 0.46, 0.62, 0.72, 0.87 s, …, 33
    iterations; R2Decay from 1/mean(TE): R2 = 0, then 0.1/s at the third, 6 iterations.
  - **Time constants at `:loglinear`.** `setup` finds a `:loglinear` parameter whose 0 gives no
    signal (0, or NaN where TE = 0), and whose large value (`_LONG`, 1/√eps: 6.7e7 in Float64,
    2896 in Float32) gives a valid one, and marks it `reciprocal`: it starts there, and the two
    passes run on u = 1/θ (column −θ²·∂ln S/∂θ, step θ = 1/(1/θ + Δu), no step where u would be
    ≤ 0). Only where the signal is missing at 0, not negative: a real-valued inversion recovery's
    rate made large looks valid (fully recovered) and was at first mistaken for a time constant,
    caught by its test. T2Decay's T2 starts at `:loglinear`; its `datastart` and
    `_loglinear_rate` (one pass weighted by the data's y²) are gone, and `datastart`'s
    documented example is now a model of one's own (T2 from two echoes). The hook stays (custom
    models; item 5's closed forms).
  - **R2Decay and ADCDecay keep their constant starts.** On `:loglinear` they were 28–68% and
    29–42% slower (alternated runs against lean, minimum of two; QMTSIR, T1IR and ADCKurtosis,
    unchanged, within ±20%): their steps land in a few iterations from 1/mean(TE) or 1/mean(b),
    and they converged everywhere, so the start cost more than it saved.
  - **The first pass made once per fit** (`_LogPass`, `c.logpass`): when no start or fixed value
    is a map and nothing starts at `:data`, every voxel's first pass starts from the same point;
    its signal and ∂ln S/∂θ are made when the cache is, and each voxel only sums its data against
    them. The same arithmetic, so bit-identical starts (T2Decay, R2Decay at :loglinear, ADCDecay
    at :loglinear, DTI, DKI, both precisions). One session: the start 20–32% faster (T2Decay at
    32 echoes 2.92 → 2.35 µs, DTI 4.38 → 3.70, DKI 56.6 → 38.4), the whole fit 5–8%. The rest
    cannot be shared: each voxel's logarithms, in both passes, and the second pass at its own
    point.
  - **The voxel fit called as a function above 8 free parameters** (`_FIT_INLINE_MAXP`, voxel.jl;
    the dataset loop and the log start). Found on the way: with the start's code grown, DKI's
    dataset fit was 41–83% slower than lean's, while a loop of `fit!` over the same voxels and
    `levmar` alone were not: compiled into the dataset loop at P = 22, the voxel fit made slow
    code, as crlb's kernel did ("crlb at many parameters"). Called as a function for every
    model, the small models were 4–23% slower, so the rule is crlb's: above 8.
  - **Measured** (alternated against lean, Float64, one thread): T2Decay +14–28% at typical T2
    (32 echoes 40 ms 2.19 → 2.49 µs), −47% where the data barely decay (8 echoes, T2 = 2 s: 1.40
    → 0.74 µs), and every voxel converged where lean's start left 7 of 2000 (32 echoes, 10 s) and
    114 of 2000 (8 echoes, 2 s) at the iteration limit; DTI −16%, DKI −29% (141 → 100 µs).
    Snapshot (scratch llt/snapshot.jl, every built-in model, 300 voxels, both precisions): every
    model's starts and fits bit-identical to lean's except T2Decay's, which agree to 2e-8
    (Float64) and 6.5e-5 (Float32), in fewer iterations (1159 → 1058, 694 → 603).
  - **A consequence to know**: in voxels that do not decay (R2Decay fits R2 = 0), T2Decay now
    reports convergence at an enormous T2 (about 6e10 s; its sum of squares equal to R2Decay's at
    R2 = 0, the least-squares answer), where lean's start ran to the iteration limit at about
    5000 s. Both mean "no measurable decay"; the docstring says to prefer R2Decay for such data.
  - Not done: the second pass from the linear prediction (no model evaluation) for models whose
    ln S is exactly linear (a once-per-fit check needed); T2DecayOffset on `:loglinear` (40–80
    ms: 5 iterations instead of 10–14; the owner: not now); the offset models' long-T2 limit (S0,
    C and T2 not separable on a nearly straight decay, every form 40–50% unconverged at 2–10 s:
    the acquisition's or the model's to fix, not a start's).
  - Full runs: EQUIP 2341, CLI 124, MRImages 103.

### Not done yet

- The other families: the T1 models (inversion and saturation recovery, MFA) share relaxation
  blocks and could be one family with switchable inversion efficiency and delay terms; ADCDecay
  is T2Decay's equation in b (the same unit, `decay(b, ADC)`); QMTSIR stays as it is.
- A third T2 component (e.g. CSF, for myelin water imaging's three pools): one more part.
- From the same SMT tissue function: a Watson-dispersed directional model and SMI's rotational
  invariants (roadmap item 7, Microstructure.jl notes).
- `newmodel` writes roles as text; an off part is written as its value without the template.

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
- **Where it stands (2026-11-08):** in, as planned: fit statistics, SMT, the command line for
  the four parts and the data reader; also DTI and DKI (item 17) and derived quantities (item
  19). The repository is now github.com/dortchrd/EQUIP.jl. Not built, on purpose (the owner,
  2026-11-08: "ship without them"): NODDI and the pulsed qMT models, the first models after
  (tier 5). The rest, done on branch `prerelease` (2026-11-08), below.

### The prerelease: what was done (2026-11-08, branch `prerelease`)

The owner, 2026-11-08: "ok lets do tier1". Each item, with what was measured:

- **T2Decay's T2 bounded at 2.5 s** (the owner: "T2s max should be 2.5 sec. CSF is around
  2s"; `_T2MAX`, src/models/builtin.jl; T2Decay only, the T2 forms with an offset or two
  components reach the iteration limit instead, already flagged; tier 6). Data that do not
  decay are best fitted by T2 → ∞: since the log-linear start (item 4) T2Decay reported
  convergence there at about 1e10 s (up to 5e11 s). Now they end on the bound, where `onbound`
  flags them, as R2Decay's R2 = 0 is. Widened in a call: `T2 = Free(upper = 10)` (the start
  stays `:loglinear`). Measured (2000 voxels per case, Rician noise, against no bound):
  - 32 echoes 10–320 ms: no decay, every voxel on the bound (SNR 50 and 200); CSF-like T2 = 2 s
    unchanged in 1985 of 2000 at SNR 50 (14 on the bound), all 2000 at SNR 100 and 200; tissue
    (40, 80 ms) and T2 = 1.5 s bit-identical, the same iterations.
  - 8 echoes 10–80 ms, where T2 = 2 s is barely measurable: 26–36% of CSF-like voxels end on the
    bound at SNR 50–100 (without it, the same voxels came out above 2.5 s); the median 1.98–1.99
    s either way.
  - **Extended to every T2 model, both forms** (the owner, 2026-11-09: "apply cap to biexp and
    1/cap for R2 models"; branch `t2-bounds`): T2 ≤ 2.5 s in T2DecayOffset and T2BiexpDecay
    (both components), R2 ≥ 0.4 s⁻¹ (`_R2MIN = 1/_T2MAX`) in R2Decay, R2DecayOffset and
    R2BiexpDecay (both). T2DecayOffset was not named; it is R2DecayOffset in its other form, and
    the two forms must agree on the same data (a test checks it). Measured (32 echoes to 320 ms,
    SNR 100, Rician, 2000 voxels, the new bounds given in the call against today's, both
    precisions): tissue values (T2 80 ms; biexponential 20 and 80 ms) bit-identical where no
    voxel nears the bound (the biexponentials: 30 voxels whose long T2 ran off now end on it);
    R2 = 0.5 s⁻¹, no voxel on the bound, differences at most 7e-8 relative (the solver's path near
    a bound, within its tolerance). Long T2 (2 s): every fit converged, against 1397, 1301 and
    1652 of 2000 for T2DecayOffset, R2DecayOffset and T2BiexpDecay (603 at the iteration limit
    for T2DecayOffset before, none after). Data with no decay: every voxel on the bound and
    flagged; R2BiexpDecay's other parameters then wander in 256 of 2000 (the iteration limit,
    also a flag; before, R2long = 0 "converged"); T2BiexpDecay at the limit in 64% in Float64
    (55% before) and 49% in Float32 (86% before).
  - **Found on the way, `onbound` in Float32**: its test was "within 1e-9 of the bound", whatever
    the fit's precision, and a Float32 fit's bound 0.4 is 0.4f0, 6e-9 from 0.4: a Float32 fit was
    never reported on a bound that Float32 cannot hold exactly (0.4; QMTSIR's R1f ≥ 0.05 and
    Sf ≥ −1.2), only on 0, 1, 2.5. Now within a few units in the last place of the fit's own
    precision too (src/fit/statistics.jl, `_near`); a test guards it.
- **`equip fit DTI` and `equip fit DKI`.** In the command line's menu (cli/src/models.jl); the
  gradient directions from the FSL `.bval` and `.bvec` beside the data (`loaddata` already read
  them into b, gx, gy, gz), or `--b --gx --gy --gz`; the derived maps written (MD, FA, AD, RD,
  V1x–z; MK, AK, RK, MKT). A missing `.bvec` gets a message naming the files; `--help` says
  where the directions come from, and describes `:loglinear` as "the log-linear fit to the
  data". Tested against the Julia API's fit (cli/test).
- **The precompile workload** (src/precompile.jl) extended: DTI (fits, derived quantities),
  crlb, montecarlo and a protocol design as the README calls them (the SIR schemes with kmf
  assumed, a sweep over SNR, T2 decay; `ProtocolDesign`, an evaluation, a two-generation
  search). Measured (fresh processes, one thread; before = `main` at 84b6d54):
  - first calls: DTI fit (1000 voxels) 2.0 → 0.01 s; crlb QMTSIR 3.0 → 0.02 s; crlb T2Decay
    0.9–1.2 → 0.00 s; montecarlo QMTSIR (n = 1000) 1.7 → 0.02 s; ProtocolDesign + d(x) 1.2–1.4
    → 0.2–0.3 s; derived(DTI fit) 0.7–0.9 → 0.01 s.
  - the cost: precompiling 46 → 65 s (once per install or code change); `using EQUIP` 0.56 →
    0.74 s.
  - **DKI left out**: from the package image its fit ran 20–70% slower than compiled at first
    use (2000 voxels, one thread, min of 5, alternated: 116–129 against 75–96 µs per voxel;
    with DKI out, 78). Not the CPU target (`native`); DTI, T2Decay, T1IR, crlb!, the design's
    d(x) and montecarlo unchanged within noise. A session pays DKI's compile once (about 6–7 s)
    and fits at full speed. **Open:** QMTSIR, in the workload since before, measured 2.6–3.0
    against 2.3 µs per voxel in three alternated pairs (+12–27%): the same effect, smaller, and
    older than this change; the cause to find (inference or codegen of package images at large
    static sizes?).
  - The command line's workload (cli/src/precompile.jl) does include DTI and DKI, written as
    files with a `.bval` and `.bvec`: each `equip` call is a new process, so without it every
    call would compile DKI. Measured (2000 voxels, one thread, in a process that loads EQUIPCLI
    against one that loads EQUIP only, two alternated pairs): the first call 0.41–0.43 s against
    6.7–7.1 s, then 112–128 against 74–98 µs per voxel: the precompiled fit comes out ahead up to
    about 160 000 voxels on one thread, more with threads. The install, precompiling all three
    packages (`main` → `prerelease`): EQUIP 42 → 66 s, its MAT extension 9 → 8 s, EQUIPCLI 58 →
    52 s (EQUIP's image now holds the crlb, montecarlo and design code the CLI's workload calls),
    109 → 125 s in all.
- **License: MIT** (the owner); LICENSE.
- **Versions**: EQUIP 0.3.0 (the owner: continuing VoxelFit's 0.1 and 0.2), EQUIPCLI and
  MRImages 0.1.0.
- **CI**: .github/workflows/CI.yml, the three suites on Julia 1.11 (the [compat] minimum) and
  the latest release, ubuntu, on every push and pull request; Manifest.toml not committed, so
  each job resolves first. Its first full run (2026-11-08): MRImages and the CLI passed on both
  (Julia 1.11.9 and 1.13.3); EQUIP failed on both, for reasons older than this branch, never
  run before on another machine:
  - 1.13.3: the flush test's probe recorded under a lock, which can make a task yield and resume
    on another thread (the FTZ mode is the thread's); reproduced here too (1–2 of 50 Float32
    checks, 13–14 of 50 Float64 ones, at 4 and 16 threads). Fixed: the probe counts with
    atomics, which never yield (0 of 100). dataset.jl's comment now says a model whose
    `signal!` yields breaks the mode's restore; the built-in models do not.
  - 1.11.9: 48 zero-allocation checks failed (34 of "fit! does not allocate", every model in
    both precisions, and 14 others: fit! weighted and from `@mrimodel`, crlb's kernel, the
    design's d(x), leakage's kernel; 96 to 6560 bytes). The cause: `Pkg.test` on Julia 1.11
    starts the tests with `--check-bounds=yes` (Pkg's Operations.jl; 1.13's no longer does),
    which turns off the code's `@inbounds`, and the allocation-free paths then allocate; with
    the flag, 1.13 allocates too (fit! for QMTSIR 576 bytes, R1IRTD in Float32 192). Not a
    property of 1.11: run as users run it, fit! allocates nothing on 1.11.9 (every model, both
    precisions, measured). The owner: "this needs to work on older versions"; Julia's download
    host allowed in the environment (julialang-s3.julialang.org), Julia 1.11.9 installed in the
    session to find it. (First taken for top-level measuring, which on 1.11 adds 128 bytes but
    is not the cause: corrected the same day.) Fixed in the tests: every zero-allocation check
    is `@test_noalloc`, a `@test` skipped when bounds checks are forced (`BOUNDS_FORCED`), so on
    1.11 the suite runs with bounds checks forced and the allocations go unchecked, and on 1.13
    the allocations are checked; `fit!` measured inside a function (`fitallocs`).
- **An example per part**: examples/fit_maps.jl (files to maps, standard errors, `onbound`),
  crlb.jl (Dortch 2018's kmf bias over SNRs; DTI's FA and MD), montecarlo.jl (against crlb in
  standard errors; T2's Rician bias at low SNR), protocol_design.jl (Table 2: 0.243395 at the
  4-point scheme, the optimum 0.243255, rounded 0.243312). The montecarlo example shows tier 2's
  point: PSR's measured bias −0.02846 against crlb's first-order −0.02818, 2.5 standard errors
  apart.
- **A shorter README**: the front page (152 lines: what it is, install and test, the four parts
  in a few lines each, the models, the examples, the roadmap, the license); everything it held
  before moved to docs/guide.md (the same text, with DTI and DKI in the command line and
  T2Decay's bound added).
- `python/` (on the list as "still named voxelfit"): not in the repository; untracked leftovers
  in one clone (the Python wrapper was removed in 9a6e173).
- Full runs: EQUIP 2347, CLI 139, MRImages 103 (the first EQUIP run found the one test that
  printed T2Decay's old bound, Inf, in `nominal`'s table; fixed).

## Roadmap

**Priorities** (agreed with the owner, 2026-11-08: "converge and tighten this into a priority
list"; it keeps the earlier order, the prerelease, then Rician fitting, then a protocol from a
number of points, and places everything else in it). The items keep their numbers; the numbers
are not an order. The same list, with each tier's loose ends, is in docs/status.md.

1. **Finish the prerelease** ("The prerelease", above): done on branch `prerelease`
   (2026-11-08; "The prerelease: what was done"): the T2Decay guard, `equip fit DTI` and `DKI`,
   the precompile workload, MIT, 0.3.0, CI, an example per part, a shorter README (the full text
   in docs/guide.md). Shipped without pulsed qMT (item 6) and NODDI (item 7) (the owner).
2. **Bias from noise at low SNR** (the owner, 2026-11-08: "put it in tier 2"): Rician fitting
   for every model (items 13 and 20, one item now); crlb's second-order bias, Box's bias of the
   parameters and the derived quantities' ½ tr(∇²g·C), opt-in and tested against montecarlo;
   `propagate(c, g)` for any function of the parameters; uncertain inputs (a measured B1 map's
   SD added to the covariance). Item 16. With it, from the test audit (item 21): the missing
   independent checks, and the validation group filled with its statistical checks, run nightly.
3. **Protocol design**: a protocol from a number of points (item 14); an objective on a derived
   quantity (MK's SD, not W1111's; item 19); the search's crlb and leakage from one kernel, and
   the worst case inside a box (item 3).
4. **Finish derived quantities** (item 19): standard-error maps (`FA_SE`); `compare` for
   derived quantities; an angular error for V1; `vary` over derived quantities in `scenarios()`;
   derived quantities for the other models (qMT's kfm = kmf·PSR, T1 from R1, SMT's D⊥).
5. **Models**: stick–zeppelin, ball–stick and IVIM first (the owner, 2026-11-08; item 17), then
   MTR, MTsat, B1 and B0 maps (item 5), pulsed qMT (item 6), NODDI (item 7), free-water DTI,
   DKI's constraints imposed (item 17).
6. **Starts, across all models** (the owner, 2026-11-08: "I want them to be tested and smart
   across all models, but let's build all models first"): plausible ranges per model; every
   model's start tested over them, both precisions, and safe failure outside them; the
   dictionary start (`BestOf` over several parameters, the signals made once per fit); an
   example `datastart`. Item 4.
7. **Nerve SMT, when the lab sends its values**: the protocol evaluator and the search (steps 3
   and 4), with the spherical mean's pipeline errors restored (item 18).

Recorded, not ranked: DKI's RTK, KFA, Kmax and DIPY's numerical MK, AK, RK (item 19); Bayesian
estimation (15); VarPro, log coordinates, solver backends, GPU (11, 12); the other start
changes (ADCKurtosis and T2DecayOffset on `:loglinear`, the second pass from the linear
prediction, `BestOf(fits = k)`, neighbours; item 4); the B-matrix, `rotinv` and Watson views,
cylinder and sphere compartments, N sticks (item 17); phantoms (2); the LsqFit bridge and a
"coming from" page (8); formats, protocol files, MRImages in its own repository and with it
Julia 1.10 (9); `--init`,
a server, the Python and MATLAB front ends, an installer (10); IRWLS and sensitivity–efficiency
plots (3); myelin water, DCE, the sinc gradient echo (12); from the test audit (21): the tests
consolidated area by area (when work touches an area), and the benchmark framework (before
performance work, e.g. VarPro).

History: the first order, decided with the owner before 2026-11-03, was the prerelease, then
**Rician-likelihood fitting** (item 13; raised from "later", 2026-11-02, because magnitude data
at low SNR are the common case where least squares is wrong, and every part of EQUIP assumed
Gaussian noise when it fitted), then **a protocol design from a number of points** (item 14,
2026-11-03); Bayesian estimation (item 15) recorded, not a priority.

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
   - **Starts across all models, after the models are built** (the owner, 2026-11-08: "I want
     them to be tested and smart across all models, but let's build all models first"; tier 6
     of the priorities). Where the starts stand: the IR family uses `BestOf` over 30 T1 values
     spread over the TIs (`_t1grid`); T1SR, T1MFA, the SMT models, QMTSIR, the biexponentials
     and ADCKurtosis start from constants; DTI, DKI and T2Decay from `:loglinear`; R2Decay and
     ADCDecay from 1/mean(acquisition). The owner's concern: "we are testing unreasonable values
     at times". The plan:
     - **Plausible ranges per model**, from the tissue presets and the literature, one source
       for the start tests, the dictionary's grid and sweeps.
     - **Every model's start tested over its plausible range**, both precisions: the fraction
       reaching the lowest SSE found from any start, and the time. Outside the range (no-decay
       voxels, a T2 of seconds), tested only for failing safely: flagged on a bound or not
       converged, never "converged" at nonsense. The first case: **T2Decay reports "converged"
       at a huge T2 in voxels with no decay**; guarded since 2026-11-08 (T2 ≤ 2.5 s, `onbound`;
       "The prerelease: what was done").
     - **The dictionary start**: `BestOf` over several parameters, over the plausible ranges,
       with the candidate signals made once per fit ("Precomputed candidates", above). The
       owner asked (2026-11-08) for a generic design matrix from automatic differentiation and
       the noise-free signal, made once: the shared first pass is that (`_LogPass`,
       src/fit/starts.jl: ∂ln S/∂θ by dual numbers at the start values, once per fit; each voxel
       solves P×P normal equations), exact for models linear in ln S (DTI, DKI, ADC, T2 on 1/T2:
       DIPY's design matrix, derived instead of written). For the others (IR, MFA, qMT, the
       biexponentials, SMT, IVIM) one matrix is a linearization about the nominal point, one
       Gauss–Newton step, poor far from it; the dictionary is the generic object made once for
       them. Estimated (not measured): 300–1000 entries × 30 observations, 2–10 µs per voxel.
       Directional models: the direction from DTI's, the rest from the dictionary.
     - **An example `datastart`** (examples/datastart.jl): DESPOT1's linearization for a
       variable-flip-angle T1 model (S/sin α against S/tan α, slope E1 = exp(−TR/T1)) and IVIM's
       segmented fit, with when to write a `datastart` and when `:loglinear` or `BestOf` is
       enough.
     - Also open, not ranked: ADCKurtosis on `:loglinear` (ln S = ln S0 − b·ADC +
       b²·ADC²·K/6 is linear in ln S0, ADC and ADC²·K, not in K itself, so the first pass is a
       linearization in K; to measure);
       T2DecayOffset on `:loglinear` (the owner: not now); the second pass from the linear
       prediction.
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
     the signal over directions within each shell removes the fiber orientation distribution;
     the microscopic model (per-axon parallel and perpendicular diffusivities; multi-compartment
     SMT with an intra-axonal stick and an extra-axonal compartment) is fitted to a few shell
     averages per voxel. Small, fast fits: a natural fit for EQUIP. It needs `erf` of dual
     numbers (SpecialFunctions with ForwardDiff provides it). The averaging per shell is a
     preprocessing step (needs the gradient directions, from the I/O layer), or the user gives
     shell averages.
     - **Built** (2026-11-03, branch `smt-proto`, src/models/diffusion/smt.jl), in the form of
       Devan et al., Magn Reson Imaging 2020;74:56 (the owner's group: SMT for multiple
       sclerosis): `SMT(b = ...)`, two compartments, sticks (D⊥,in = 0) and a zeppelin with the
       same D_ax and the tortuosity limit D⊥,ex = (1 − v_ax)·D_ax; parameters `S0` (`:scale`),
       `vax` (0 to 1), `Dax` (mm²/s, 0 to 3e-3), b in s/mm² as for `ADCDecay`.
       `sphericalmean(data, b; tol = 100)` averages each shell (b-values within 100 s/mm²
       merged, as in Devan) and returns the shells' b and numbers of directions, which are
       crlb's and montecarlo's `averages` (a shell's mean has σ/√N) and the fit's `weights`.
       `SMT_PROTOCOLS` holds Devan's Table 2 (A–E, b and directions), `TISSUES` its Table 1
       (`SMT_NWM`, `SMT_NAWM`, `SMT_T2lesion`, `SMT_cBH`, from Bagnato et al. 2019).
       - **Numerics**: the stick's spherical mean √π·erf(x)/(2x) is computed from x² = b·D,
         by its Taylor series below x² = 0.05 (6 terms, error < 2e-12; no 0/0 at b = 0 and no
         square root, whose derivative is infinite at 0), `erf` above; the two agree in value
         and derivative at the switch, and both against the direction average they stand for
         (∫₀¹ exp(−x²u²) du, quadrature, 1e-9). Integer coefficients, π converted to the
         argument's type: a Float32 fit stays in Float32. SpecialFunctions was loaded by
         ForwardDiff already: no new loading cost.
       - **Against Devan 2020, Fig. 1** (v_ax precision efficiency 1/√(N·CRLB/v_ax²) of the five
         protocols in the four tissues): crlb on spherical means, with averages and weights
         = directions, at SNR 20 per direction gives their figure: normal white matter within
         1–5% for every protocol (A 0.90 against 0.90, B 1.00 against 1.00, C 1.10 against
         1.10, D 0.86 against 0.865), NAWM and T2 lesions within 6%, black holes up to 13%
         higher (protocols A and B). Their CRLB is of the directional signals, the worst case
         over fiber arrangements (isotropic, where each direction's signal is the spherical
         mean: the same Fisher information as ours), with the Rician factor R(S, σ), which lowers
         the information most where the signal is lowest: the high-b shells of A and B, in the
         tissue with the lowest v_ax. So the figure's SNR (not stated) is 20, the SNR of their
         simulations, and the remaining difference is Rician (item 13).
       - **Fits** from the default starts (Gaussian, SNR 20, 5000 voxels per tissue, protocols
         A and D): all converged, mean v_ax and D_ax within 0.006 and 0.05e-3 of the truth,
         SDs within 5% of crlb's. 51 tests.
       - **Variants** (2026-11-03, asked by the owner: free water, and compartment T2s, an area of
         the lab's research; since "Building blocks", below, parts of one SMT equation, switched
         on in a call): `SMTFreeWater(b)`, a third, isotropic compartment,
         S0·[(1 − fiso)·SMT + fiso·exp(−b·Diso)] (NODDI's; Diso fixed at 3e-3 mm²/s), `vax` the
         fraction of the tissue without the free water (Devan's "relative" fraction); and
         `SMTT2(b, TE)`, a T2 per compartment, S0·[v_ax·e^(−TE/T2ax)·S̄_ax + (1 − v_ax)·
         e^(−TE/T2ex)·S̄_ex], for shells at several echo times (Veraart et al., NeuroImage 2018,
         TEdDI; McKinnon & Jensen, Magn Reson Med 2019), `vax` the volume fraction (and the one
         the tortuosity limit uses). `sphericalmean(data, b; TE)` groups by echo time, then
         b-value, and returns a named tuple `(means, b, directions, TE)`.
         - At one TE, SMT's fraction is T2-weighted, and not exactly the T2-weighted signal
           fraction (the tortuosity limit uses it too): v_ax 0.45, T2 90 ms in and 50 ms out, TE
           80 ms: SMT 0.664, the T2-weighted fraction 0.625.
         - Both need protocols made for them. Free water (v_ax 0.45, D_ax 1.8e-3, f_iso 0.2;
           b 0–4500 s/mm², 6 shells; SNR 50 per shell mean): crlb SD 0.10 in v_ax and 0.29 in
           f_iso; at SNR 500, 0.010 and 0.029 (b = 100–300 added: f_iso 0.21, 0.021). T2s (T2
           90 and 50 ms; 3 TEs, 70–130 ms, × 4 b-values; SNR 50): SD 0.48 in v_ax and 47% in
           T2ax, and without an upper bound 14% of the fits wandered to T2s of hundreds of
           seconds (now bounded at 1 s); 4 TEs, 60–200 ms: 0.31 and 27%; at SNR 500: 0.03–0.05,
           3–5%. At SNR 500 both fit without bias (at most 10% of the SD, the nonlinear bias
           from the noise itself, measured with 20,000 copies) and with crlb's SDs (within 2%).
           A per-shell-mean SNR is the per-direction SNR × √directions: 20 × √30 ≈ 110.
       - Not yet: the command line (`equip fit SMT` needs the spherical mean as a step, from the
         `.bval`), Rician noise in the fits and in crlb (item 13; Devan's Appendix gives the
         Rician Fisher information, the factor R(S, σ), which would close the black-hole gap
         above), the single-compartment microscopic tensor (Kaden et al., MRM 2016), and the
         number of directions a shell needs (Schilling et al., Magn Reson Imaging 2022: the
         orientational sampling error of a finite direction set, besides the noise).
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
   - **Design ideas from Microstructure.jl** (Gong & Yendiki, arXiv:2407.06379, 2024; v0.1.8,
     github.com/Tinggong/Microstructure.jl; read 2026-11-03, the owner's suggestion). A Julia
     package for diffusion compartment models, estimated by MCMC or by neural networks trained
     on simulated data. How it is built:
     - Compartments are types (`Stick`, `Zeppelin`, `Iso`, `Cylinder` (Van Gelderen 1994),
       `Sphere` (Neuman 1974), `Tensor`), each with its tissue values and a `t2` field, and a
       signal for a `Protocol` (b, TE, Δ, δ, G, directions, and the SH order of each
       measurement, all in SI units, converted from the text files' s/mm² and ms). A model is a
       struct of compartments and fractions (`SANDI`, `ExCaliber`, `MTE_SMT`, `SMI`, ...);
       `model_signals` sums fraction × compartment, then divides by the first (b = 0) signal.
     - `t2 = 0` means no T2 weighting: the same compartments make the single-TE model and the
       multi-TE one (MTE-SMT, MTE-NODDI: Gong et al., NeuroImage 2020), where the fractions
       are no longer T2-weighted.
     - Links between parameters are pairs of field paths (`"extra.dpara" => "axon.dpara"`);
       a constrained diffusivity is a fraction of another (`dperp_frac` in [0, 1] for De⊥/De∥,
       `p4_frac` for p₄/p₂), which keeps every bound a simple box.
     - SMI's kernels (`Stick_kernel`, ...) give the l-th rotational invariant by projecting
       the directional signal on P_l (200-node Gauss–Legendre on [0, 1]); the fODF enters as
       p₂ (and p₄). `get_sh_order` picks a shell's SH order from its number of directions
       (> 28 for order 6) and its b (order 4 up to b = 2500 s/mm², 6 up to 7000).
       `correct_rician_bias` is √(S² − σ²) above σ, 0 below (NYU's SMI.m form). Spherical
       means group the measurements by (b, TE, Δ, δ).
     - The likelihood is a `Noisemodel` with a `logpdf`, Gaussian or Rician (Distributions'
       `Rician`), with σ sampled as one more parameter in the MCMC.
     - What EQUIP can take:
       1. **Compartments as building blocks.** Small functions of one observation and the
          cosine t between fiber and gradient (stick, zeppelin, ball, later cylinder and
          sphere), each with its directional signal f(t), its spherical mean (closed form where
          there is one), and its l-th Legendre projection. Today SMT, SMTFreeWater and SMTT2 are
          three types with three `signal!`s and three `_profile`s, and orientation.jl takes
          their `Union`; a model composed of compartments gives `_profile`, the rotational
          invariants (an SMI model on observations (b, l)) and a Watson-dispersed directional
          model (the nerve question, step 2) from one definition.
       2. **Variants as roles, not types**, EQUIP's own idiom: one SMT with `T2ax = T2ex =
          Fixed(Inf)` (exp(−TE/∞) = 1: no T2 weighting, and no sentinel 0), `fiso = Fixed(0)`,
          De∥ tied to D_ax and De⊥ tied by the tortuosity formula. Freeing the T2s gives SMTT2,
          freeing `fiso` SMTFreeWater, freeing De⊥ SMT without tortuosity (as Microstructure.jl's
          SMT guide fits it), each in the call. To check: the speed and compile time of a model
          with nine parameters, most of them fixed, against the three small types.
       3. **Fractions for constrained diffusivities** (De⊥ = r·De∥, r in [0, 1]): box bounds,
          which EQUIP's solver takes, instead of an inequality between two parameters.
     - What not to take: mutable structs that hold the values, and links by field-path strings
         (neither type-stable nor dual-number friendly: EQUIP keeps values in named tuples and
         roles outside the model); dividing by the b = 0 signal (it puts that image's noise in
         every value and drops it from the likelihood, which `logp_rician` does: EQUIP fits S0);
         allocating vector code (the package needs no derivatives; EQUIP's fits do).
     - Two cautions from our own runs: their SH order rule gives order 4 at b = 2500 s/mm², which
       our coherent-fiber runs found biased (bench/floor_sh.jl); and an l = 2 rotational
       invariant is a norm of 5 SH coefficients, so with noise it has a floor of its own (a
       noncentral χ with 5 degrees of freedom): an SMI fit by least squares has to model it.
     - Restricted compartments (`Cylinder`, `Sphere`) need Δ, δ and G per observation and a
       decision on units (EQUIP: b in s/mm², D in mm²/s; radii and the gyromagnetic ratio favour
       SI inside). Peripheral nerve axons are larger than the brain's, so axon diameter is a
       candidate for crlb (their sensitivity criterion, z_α/(SNR·√n), is a rough version of it).
   - DTI and DKI are built (item 17); IVIM moved to item 17 (2026-11-08), with ball–stick and
     stick–zeppelin; free-water DTI is on item 17 too.
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
     - **Built** (2026-11-02, branch `mrimodel-proto`, src/models/macro.jl), as planned, with
       these decisions:
       - **Ties written as Julia writes keyword defaults**: a default that names another
         parameter is a tie (`R1m = R1f` → `Fixed(:R1f)`), a formula of others a tie by function
         (`R1m = 0.8 * R1f` → `Fixed(p -> 0.8 * p.R1f)`), since Julia's keyword defaults may
         use earlier keywords already. `Fixed(...)` written out passes through.
       - **An acquisition argument may be one number for every observation** (a single TR, as
         T1MFA and QMTSIR allow by a method of their own): the generated loop reads a vector's
         element or the number (`_obs`), and checks each vector's length against the output
         once (`_checkobs`, then `@inbounds`). The conversion to the fit's precision converts
         such numbers too (`_with_precision` for the type; the default converts vectors only).
       - **The body becomes a method of `EQUIP._obsignal`**, dispatched on the model, its
         arguments the acquisition, the options and the parameters by name, `@inline`d into
         the loop; `signal!` reads the parameters once by name (`p.S0`). No name is added to
         the user's module but the type. Escaped as a whole, EQUIP's own functions written as
         `EQUIP.f` (the module object itself), so it works whatever the caller imported.
       - **Mistakes are said while the macro expands**, with what to write instead: no
         acquisition or no parameters, a name used twice, types on arguments (they come from
         the values), options before the acquisition, a start or bound that depends on another
         parameter, a bound or fixed value that depends on the acquisition (only a start can),
         a tie to itself, `where` clauses. A free parameter with no start fails at the fit,
         now for any model with "give one in the call, `k = Free(start = ...)`".
       - **Measured**: `T2DecayOffset`, `T1MFA` (one TR) and `QMTSIR` written with the macro
         give the same fits as the built-ins, bit for bit, in Float64 and Float32, with the
         same iterations; zero allocations per `fit!`. Speed (100,000 voxels, 4 threads, best
         of 5): T2 with offset 128 ms against 125 ms (Float64), 112 against 105 ms (Float32);
         SIR 5-point 135 against 148 ms (Float64), 129 against 128 ms (Float32): the same within
         run-to-run noise. The SIR body recomputes the exchange rates at each observation, yet
         its `signal!` takes 129 ns against the built-in's 134 ns: inlined, the rates do not
         depend on the observation, and the compiler moves them out of the loop. So the
         planned limit ("a whole-curve body: write `signal!` by hand") matters less than
         expected; still, a body that loops or allocates per observation would not be helped.
         35 tests.
   - **`newmodel(...)`**, a template generator: writes a commented model file (struct,
     `parameters`, `signal!` with a TODO, optional hooks, a test snippet) to edit, for models
     that will grow, and as the route from a custom model to a built-in one.
     - **Built** (2026-11-02, branch `newmodel-proto`, src/models/template.jl), with the
       `@mrimodel` variant proposed with it: `newmodel(M)` for a model made with the macro
       writes the code the macro generates as plain, readable Julia, so a model can start as
       one line and grow. The macro keeps its definition for that (`_mrimodel_source(::Type)`,
       the expression itself), and the file is written from it: the roles as written
       (`m -> mean(m.TE)`, `Fixed(:R1f)`, `Fixed(p -> 0.8 * p.R1f)`, as text: Julia prints
       anonymous functions as `(m->begin ... end)`), the body as a function of one observation
       of its own (`_name(...)`, called in the loop: a body may `return`, so it cannot be
       pasted into the loop), `c ? a : b` and `a, b = x, y` written back as typed (Julia
       prints them as an if-block and with parentheses), the docstring kept. Blank:
       `newmodel(:Name; acquisition, parameters, options)`, parameters as names or roles, the
       signal a TODO returning zero. Every file has the house layout (header block,
       docstring, struct, `parameters`, signal of one observation, `signal!`, the hooks and a
       first check commented out). An existing file is kept unless `force = true`.
       - The written file uses vector fields, as the built-ins (`@kwdef`, `eachindex(out,
         ...)`); a model whose acquisition was one number for every observation gets, as a
         comment, the one-line method that allows it (T1MFA's).
       - Tests: the written `MacroSIR` and `MacroT2Offset` files, `include`d into a module of
         their own, give the same fits as the built-ins, bit for bit; a blank file loads and
         runs; 19 tests.
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
9. **Data I/O** (the release's part: "The command line and the data reader: the plan", above;
   started 2026-11-02: "MRImages: MRI images in and out of files", above, which replaced "the
   core never depends on file formats" with EQUIP depending on MRImages and MAT as an
   extension; DICOM and PAR/REC through dcm2niix, decided 2026-11-02: "DICOM through
   dcm2niix", above): make getting data in and maps out as easy as possible. An `io/` layer
   (dispatch on file type, protocol handling) plus one package extension per file-format
   package.
   - Reading: `data, info = loaddata(path_or_paths; ...)` returns the data with observations
     last, plus `info`: the geometry needed to write maps back, and any acquisition values found
     (TE, TR, FA, TI, ...).

     | format | via | notes |
     |---|---|---|
     | NIfTI (.nii, .nii.gz) | NIfTI.jl | apply `scl_slope`/`scl_inter`; memory-map uncompressed files |
     | BIDS JSON sidecars | JSON3.jl | EchoTime, RepetitionTimeExcitation, FlipAngle, InversionTime (s, deg); qMRI file collections (`_echo-`, `_flip-`, `_inv-`, `_mt-` entities) stacked into one array |
     | MATLAB .mat | MAT.jl | pick the variable by name |
     | NumPy .npy / .npz | NPZ.jl | C order: reverse dimensions, or observations first |
     | DICOM | dcm2niix (converted first, to NIfTI with a sidecar) | decided 2026-11-02: no DICOM reader of our own |
     | Philips PAR/REC, XML/REC | dcm2niix (converted first) | as DICOM |
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
     - Units: seconds and degrees, as BIDS sidecars hold them (dcm2niix converts DICOM's and
       PAR/REC's ms). Warn when values look like the wrong unit (e.g. TE > 10 s). The number of
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
   - **Compilation is the first limit, before ~30–40** (measured 2026-11-05 on a 22-parameter
     DKI, item 17): the fit compiles in 8–9 s, but crlb's kernel, with all its pieces inlined,
     took 118 s at 13 parameters and did not finish in 23 minutes at 22. Its pieces not inlined:
     18 s and 46 s. To do: don't inline above some P (decided at compile time, from the type
     parameter), after measuring the small models' kernels with and without.
12. **Later**: GPU, more models (Rician-likelihood fitting moved to item 13, a priority):
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
   - Multi-exponential T2 (myelin water), DCE. (Diffusion tensor and kurtosis: built, item 17;
     IVIM: moved to item 17, 2026-11-08.)

13. **Rician-likelihood fitting** (a priority after the prerelease, decided 2026-11-02). Started
   on branch `nerve-smt` (2026-11-03): step 1 below, the expected magnitude as the model
   (`RicianMean`), with the Rician bound (`ricianbound`) and crlb of least squares on magnitude
   data (`riciannoise`); see "Peripheral nerve SMT".
   Magnitude images have Rician noise (Gudbjartsson & Patz, Magn Reson Med 1995): at low SNR
   the magnitude of a small signal sits on a floor of about σ√(π/2), and least squares, which
   assumes Gaussian noise, fits that floor as signal.
   - **Measured** (2026-11-02, `ADCDecay`, true ADC 1.5×10⁻³ mm²/s, b 0–3000 s/mm², SNR 15,
     Rician, 20,000 voxels): least squares gives a median ADC of 1.36×10⁻³ (−9%); a constant
     offset fitted with it (`S0·exp(−b·ADC) + C`, made with `@mrimodel`) over-corrects, 1.70×10⁻³
     (+13%), with an SD 1.6× larger (2.9 against 1.9×10⁻⁴, predicted by crlb and measured; ADC
     and C correlated at 0.75): the floor depends on the signal (σ√(π/2) where it is near zero,
     about 0 where it is high), so no constant removes it. `T2DecayOffset`'s docstring says the
     same.
   - **What it changes**: `fit` minimizes the negative log-likelihood instead of the sum of
     squares, per voxel, Σⱼ [ (yⱼ² + Aⱼ²)/(2σⱼ²) − log I₀(yⱼAⱼ/σⱼ²) ] + terms without θ, where
     Aⱼ = signal(θ) ≥ 0 (the magnitude model) and I₀ the modified Bessel function; computed as
     log I₀(z) = log besselix(0, z) + z (the scaled function: no overflow at high SNR, where the
     likelihood tends to the Gaussian one). Maximum likelihood for Rician data: Sijbers et al.,
     IEEE Trans Med Imaging 1998; for T1 and T2 fitting, Karlsen et al., Magn Reson Med 1999.
   - **σ** must be known (a noise region, the background, or `montecarlo`'s own), or fitted as
     one more parameter per voxel (often poorly determined with few observations; for later).
     Per observation, σ/√NEX with averages (complex averaging; magnitude averaging is not Rician,
     item 3's averages). Several coils with sum of squares give a noncentral χ distribution
     (Constantinides et al., Magn Reson Med 1997): the same structure, for later.
   - **Two steps, the cheap one first**:
     1. *The expected magnitude* as the model: fit E[|y|] = σ√(π/2)·L₁/₂(−A²/(2σ²)) (the Rician
        mean, a Laguerre function, through Bessel functions) by least squares, a model wrapper
        around any model. It keeps the LM solver, crlb, montecarlo and design unchanged, and
        removes most of the bias in the mean (the expected-value correction; compare Koay &
        Basser, J Magn Reson 2006, for the signal-correction approach). Needs the Bessel
        functions of dual numbers (SpecialFunctions.jl with ForwardDiff's rules: check).
     2. *Maximum likelihood*: the likelihood above, minimized by a Newton or trust-region step
        with ForwardDiff's Hessian (P is small: the Hessian is cheap), or by Fisher scoring
        (the Rician Fisher information per observation, a function of A/σ alone: tabulated
        once), which keeps an LM-like structure. Item 3's iteratively reweighted least squares
        is a third route, on weighted fits.
   - **crlb and the statistics**: the Fisher information for Rician noise, Σⱼ JⱼJⱼᵀ·F(Aⱼ/σ)/σ²
     with F → 1 at high SNR (Gaussian) and smaller at low SNR, so crlb predicts the precision
     Rician data allow (today: "not modelled by crlb"); `stderror`, `loglikelihood`, AIC/BIC
     from the Rician likelihood for such fits; `montecarlo` measures it as for any fit.
   - **Interface**, to decide: a keyword of `fit` naming the likelihood (`fit(m, y; σ,
     likelihood = :rician)`), with σ a number or a map. `fit` refuses `noise` today (decided:
     the noise's distribution is the user's choice in `simulate` and `montecarlo`, and least
     squares has no other); with this item the fit gets a choice of its own, kept separate from
     `simulate`'s `noise` (data simulated one way can be fitted another, as `simulate_with`
     allows).
   - **Tests**: montecarlo at SNR 5–50 for T2, ADC and SIR (bias against least squares, SD
     against the Rician crlb); the likelihood against Gaussian least squares at high SNR (the
     same fits); the expected magnitude against a simulated mean.
14. **A protocol design from a number of points** (after the prerelease and Rician-likelihood
   fitting, decided 2026-11-03). `ProtocolDesign(m, scenarios; ...)` takes a protocol, a model
   with its acquisition, and uses it four ways: the number of points N (a type parameter: every
   protocol's varied fields are `SVector{N}`); the fields that are not varied (`magnitude =
   false`, a TR not searched), which every protocol evaluated keeps; a model to build the fits'
   caches from; and a start, `x0 = _x(d, d.model)`, one member of the first start's initial
   population (the search keeps its best member, so it ends no worse than x0) and evaluated once
   at construction (`d(m)`: a mistake in `scantime` or `constraints` shows there). Only the
   start needs real values, and asking for one is odd when the search covers the whole box of
   bounds. In the Table 2 example, `QMTSIR(; SIR_4POINT..., magnitude = false)`, the protocol
   given is the paper's answer, 1 member in 640 (8 starts, a population of 80), and 7 of the 8
   starts reached the optimum within 1% from random populations without it.
   - **Interface** (proposed; the keyword names to decide): the model's type and a number of
     points.
     ```julia
     d = ProtocolDesign(QMTSIR, ss; points = 4,
                        acquisition = (TI = (0.01, 10.0), TD = (0.01, 10.0)),
                        others = (magnitude = false,),         # the fields not varied
                        snr = 200, scantime = m -> sum(m.TI .+ m.TD .+ 0.3),
                        constraints = m -> m.TI .+ m.TD .- 0.5)
     optimize_protocol(d)                                       # every start random
     optimize_protocol(d; start = QMTSIR(; SIR_4POINT..., magnitude = false))  # never worse than this
     ```
     The template is built inside: the varied fields at their bounds' midpoints, the others from
     `others` or the model's defaults; a field with neither is an error that names it. The
     starting protocol moves to `optimize_protocol`, because it belongs to the search, not to
     the objective. `ProtocolDesign(m, ss; ...)` stays: it is how a given protocol is scored
     (`d(m)`, e.g. a published scheme).
   - **The number of points as a question of its own**: with `points` a number, `points = 3:8`
     compares designs of different lengths. Eq. 5 is a precision per unit scan time (× T), so
     extra points pay for their time; with a `budget`, the designs are compared at one scan time.
   - **To check**: models whose constructors are not keyword ones (positional arguments,
     `@mrimodel` options), and protocols that are not one vector per point: in a multi-shell
     diffusion design the points are shells, with b and TE per shell continuous and the
     directions and averages per shell integers (the nerve SMT search, branch `nerve-smt`, step
     4, which this would serve).
   - **Tests**: the same optimum as from a given protocol (Table 2, the four tissues); the
     template's errors; the result from `start = m` no worse than `d(m)`; `ProtocolDesign(m, ...)`
     unchanged (the same objective values and search results, seeded).
15. **Bayesian estimation: priors, their bound, sampling** (discussed 2026-11-03, after reading
   Microstructure.jl's estimators; not a priority yet). Microstructure.jl fits by random-walk
   Metropolis–Hastings per voxel (box priors, Gaussian or Rician likelihood, σ sampled too; the
   posterior mean after burn-in, 0.24 s per voxel on 8 threads, about three days for a million
   voxels, against 1–20 µs per voxel for EQUIP's fits), or by a network trained on signals
   simulated from priors (uncertainty from dropout at test time). At high SNR the posterior is
   close to Gaussian with covariance (JᵀJ/σ²)⁻¹, what `stderror` and crlb give; a sampler adds
   information only where it is not: low SNR, bounds, degenerate models (the bi-exponential,
   SMI's two branches, NODDI), where it shows the branches instead of landing on one. In order:
   1. **Priors in the solver (MAP)**: a Gaussian prior is one more residual per parameter,
      (θ − μ)/τ, appended to the least-squares problem: the LM solver and its speed unchanged.
   2. **The prior in crlb**: its precision added to the Fisher information, the Bayesian
      (van Trees) bound. Then crlb says, before any fit, how much of an estimate the data
      determine and how much the prior does: the question for SMI in disorganized nerve (the
      nerve SMT section), which a sampler's or a network's user cannot easily see.
   3. **Sampling as an extension**: NUTS (AdvancedHMC.jl or Turing.jl, a weak dependency) with
      EQUIP's exact ForwardDiff gradients, far more efficient than a random walk; for regions of
      interest and validation, not whole volumes.
   Networks trained on simulations stay out of scope (`simulate` and `scenarios` make the
   training sets for anyone who wants one).
16. **Error propagation: to derived quantities, from uncertain inputs, and the estimator's own
   noise bias** (discussed 2026-11-05; not built). Three pieces, each the delta method with a
   different Jacobian:
   - **Outputs: `propagate(c, g)`.** A quantity g of the fitted parameters (FA, MD, AD, RD from
     DTI's six elements; any index or ratio): SD √(J Σ Jᵀ) with Σ = `c.cov`, bias g(fit) − g(truth)
     (the assumptions', exact) + ½ tr(H Σ) (the noise's, second order), a covariance when g gives
     several. Tested in a scratch script (DTI, 6 + 30 images, eigenvalues 1.7, 0.4, 0.3e-3, SNR 50
     and 20, against 20,000 Monte Carlo fits): every SD within 1%, the FA–MD correlation (−0.12),
     FA's noise bias +1.16e-3 against +1.37e-3. Eigenvalues differentiate through StaticArrays'
     closed-form 3×3 symmetric solver (`eigvals(Symmetric(SMatrix))`): gradients equal
     Hellmann–Feynman's; Hessians finite unless the eigenvalue in question is repeated. FA from the
     six elements as √(3/2)·‖D − MD·I‖/‖D‖ (Basser & Pierpaoli 1996; Ennis & Kindlmann 2006),
     checked equal to the eigenvalue forms to 7e-16.
   - **Inputs: a fixed parameter that was measured** (a B1 map, a T1 for MTsat). The fit's
     sensitivity is leakage's dθ̂/dc. Its bias: crlb with the fit's fixed value at the map's (exact,
     already in EQUIP); its random part, L·Σ_c·Lᵀ added to the covariance (independent scans; to
     add, e.g. `crlb(m, θ; snr, uncertain = (B1 = 0.03,))`). Measured (T1MFA, FA 2–30°, T1 1 s, a
     B1 map 5% high with a 3% SD, SNR 1000, 20,000 voxels): bias −0.0945 s predicted, −0.0919
     measured; SD 0.0572 predicted, 0.0576 measured, with the slope taken where the fit runs (B1 =
     1.05: dT1/dB1 = −1.76; at the truth, −2.03, it over-predicts the SD by 14%: VFA's T1 goes as
     1/B1²). Errors in the data themselves (the spherical mean's Rician floor, noise and
     orientational sampling) are the sandwich's and `simulate_with`'s: `sphericalmeannoise`.
   - **The estimator's own noise bias** (Box, J R Stat Soc B 1971): a nonlinear least-squares fit
     is biased at second order in the noise, b ≈ −½ (JᵀWJ)⁻¹JᵀW·d, dᵢ = tr(Cov·Hᵢ), Hᵢ the Hessian
     of observation i's signal. An addition to crlb, not part of the sandwich (the sandwich is the
     first-order covariance). Measured, the same T1MFA at SNR 100 with B1 right: +0.026 s against
     crlb's 0; with the biased B1 map the two add, −0.0945 + 0.026 = −0.069 against −0.0686
     measured. crlb would report it beside `bias`, and `propagate` would use ∇g·b + ½ tr(H Σ).
   - **Where it stands (2026-11-08; tier 2 of the priorities, beside Rician fitting, the owner:
     "put it in tier 2").** `propagate`'s first-order half is built, as the `derived` hook in
     crlb and montecarlo (item 19: the delta method's SD, the bias g(θ̂) − g(θ) of what the fit
     assumes). Still to build, together:
     - **`propagate(c, g)` for any function** of the parameters, not only a model's `derived`:
       the same code with g in place of `derived`.
     - **The second-order noise bias, both terms**: Box's bias of the parameters, −½
       (JᵀWJ)⁻¹JᵀW·d, and the derived quantity's own ½ tr(∇²g·C); for a derived quantity, ∇g·b
       + ½ tr(∇²g·C). Cheap where crlb already is: its misspecified Hessian evaluates the model
       with nested duals (`hbuf`, src/errors/derivatives.jl), which gives every ∇²gᵢ; Box's
       bias is then one more sum over observations and one solve; ∇²g is nested duals through
       `derived`. Opt-in (e.g. `crlb(m, θ; snr, secondorder = true)`), so the protocol search
       keeps its first-order kernel unless it asks. Tested against montecarlo, as
       bench/agreement.jl does for the first order. To check: whether Box's bias of DTI's six
       elements closes FA's gap (+1.16e-3 predicted from ½ tr(∇²g·C) alone, +1.37e-3 measured,
       SNR 20).
     - **Where the expansion fails**, montecarlo the reference: near equal eigenvalues (FA at
       low anisotropy, GM and CSF: FA behaves like a norm near isotropy, not smooth), near a
       bound, and at low SNR generally.
     - **Uncertain inputs**: L·Σ_c·Lᵀ added to the covariance (e.g. `uncertain = (B1 =
       0.03,)`), measured above.
     - Not the same as the Rician floor, which shifts the data's mean and which crlb already
       includes under Rician noise (`riciannoise`, `ricianbound`; item 20). At low SNR both
       matter, and they add.
17. **Directional diffusion: DTI, stick–zeppelin, DKI, and the acquisition they share**
   (discussed 2026-11-04 and 05; examples/directional_models.jl, examples/dki.jl and scratch
   scripts; not built).
   The directions' rule is decided ("Directions per image", in the building-blocks section,
   revised 2026-11-05 and back on 2026-11-06): `b`, `gx`, `gy`, `gz`, each a vector of numbers,
   built from the files by `diffusionencoding` (no supertype: everything works by field name,
   decided 2026-11-06).
   What follows is what to build on it, in order.
   - **A built-in DTI, written out: built 2026-11-07** (by hand, in the standard format, not with
     `@mrimodel`: "DTI built in", in the compartments section). S = S0·exp(−b·gᵀDg),
     gᵀDg = gx²Dxx + gy²Dyy + gz²Dzz + 2gxgyDxy + 2gxgzDxz + 2gygzDyz, the six elements fitted.
     The owner kept this form over eigenvalues and angles (2026-11-05): the eigen form is degenerate where two eigenvalues are
     equal (the angles of a prolate tensor's minor axes are undetermined), needs a relabel, and
     its angles wrap; FA, MD, AD and RD and their errors come from `propagate` (item 16), not from
     reparametrizing. The one place an eigen form would pay: nerve DTI with λ2 = λ3 tied (an
     axially symmetric tensor, 4 numbers and a direction), when the fit itself should carry the
     symmetry. The start: the log-linear fit, linear in ln S0 and the six elements (the B-matrix
     design matrix, as DIPY's `design_matrix`), so no start needs tuning; since 2026-11-07 the
     generic `:loglinear` start, from the signal ("The log-linear start"). Measured (scratch
     dtiforms.jl, 6 + 30 images, Float64): written out, as an SMatrix product gᵀDg, and in
     B-matrix form, the same speed within noise (2.5–2.8 µs per voxel): written out, as the papers
     write it.
   - **A built-in StickZeppelin**: one fiber, the stick's diffusivity also the zeppelin's parallel
     one, D⊥ = (1 − v_ax)·D_ax by default (released in a call, as SMT's). The cosine ξ = g·n(θ, φ)
     is computed in the signal, since it joins the acquisition and the parameters; the blocks take
     a number. n as an SVector or three scalars (no allocation either way; a `[...]` vector
     allocates 208 bytes per call: static arrays inside `signal!` are only notation, and must stay
     static). n and −n are one fiber: a relabel (θ → π − θ, φ → φ + π) and φ wrapped. θ and φ from
     DTI's principal direction, as the fit's start or fixed as per-voxel maps. Measured (the
     example, SNR 100, 132 images): v_ax 0.600 ± 0.004, the direction to 0.14°, Float64 and
     Float32. Written by hand with `fiberdirection` once per voxel and `ξ = g ⋅ n` per image
     (12.8 µs per voxel; with `@mrimodel` 23.1: "Directions per image", built 2026-11-06).
   - **Stick–zeppelin, ball–stick and IVIM, first among the models** (the owner, 2026-11-08:
     "want IVIM, ball stick, and stick zeppelin as well"; tier 5 of the priorities, in this
     order). Not built.
     - **Stick–zeppelin**: the example's code moved into src/, with the relabel and a start for
       the direction (below), tests and a derived D⊥.
     - **Ball–stick** (Behrens et al., Magn Reson Med 2003): S0·[(1 − f)·iso(b, d) + f·stick(b,
       d, ξ)], one diffusivity d for both; the stick–zeppelin with the zeppelin isotropic, so the
       same direction code with one parameter fewer. One stick now; N sticks later (FSL's
       bedpostx, for crossings), which brings sticks that swap (a relabel ordered by f), a
       fraction that should go to zero where there is no second fiber (bedpostx's automatic
       relevance determination), and a flatter likelihood.
     - **IVIM** (Le Bihan et al., Radiology 1988): S0·[f·exp(−b·D*) + (1 − f)·exp(−b·D)], D*
       about 10× D. Poorly conditioned (D* very uncertain and correlated with f at typical SNR),
       which crlb and montecarlo show directly; b-value design for it is published (Lemke et
       al., Magn Reson Imaging 2011), so it also shows EQUIP's design part. A relabel reports
       the faster component as D*, as T2BiexpDecay orders T2short and T2long. Its start, the
       segmented fit (D from b ≥ 200 s/mm², log-linear; f from the intercept; then D*), as its
       `datastart`: the published method, not a fix for a poor start (the reason T2Decay's was
       removed), to measure against constant starts and `BestOf`. The simplified IVIM (D*
       fixed) already works with `Fixed`.
     - **The direction's start, for both directional models**: DTI's principal direction,
       computed in the voxel by DTI's log-linear tensor fit (a few µs against a 12.8 µs fit),
       one helper for both; θ and φ as start maps from a DTI fit remain possible, as today.
     - **Free-water DTI** (Pasternak et al., Magn Reson Med 2009): DTI plus an isotropic
       compartment at the free-water diffusivity; after these three.
   - **The acquisition, shared.** Every directional model takes b, gx, gy, gz; one helper builds
     them from the files, `diffusionencoding(bval, bvec) -> (b = ..., gx = ..., gy = ..., gz =
     ...)` (tested in scratch handmodels.jl, as `diffusion`, and as option A of 2026-11-05),
     splatted into any model's keyword constructor: `DTI(; diffusionencoding("dwi.bval",
     "dwi.bvec")...)`. It checks the lengths,
     transposes an N×3 table, checks unit vectors (atol 1e-2, as DIPY's gradient table) where b
     is above a threshold (50 s/mm², DIPY's `b0_threshold`), and keeps b as given (decided
     2026-11-06). DIPY's `GradientTable` is the
     model: one object per acquisition, shared by every model, with b-tensors as `btens` (linear,
     planar, spherical, cigar). Tried and dropped: an `Encoding` struct as a field of each model.
     It breaks the per-observation rule for nothing (the cosine form costs the same, measured).
   - **Constructors.** An outer constructor (`DTI(bval, bvec)`) is safe; an inner one removes the
     positional constructor that `_with_precision` and `ProtocolDesign` rebuild models with; a
     second keyword-only constructor overwrites `@mrimodel`'s (an error at precompile time in a
     package). So: `diffusionencoding(...)` and the generated keyword constructors, and the
     supertype `DirectionalDiffusion` for what the models share (decided 2026-11-05, dropped
     2026-11-06).
   - **The small things decided** (status.md): a clear error for a field of 3-vectors or a
     matrix, pointing to `diffusionencoding`; `loaddata` mapping a .bvec to gx, gy, gz;
     `readbvec` exported from EQUIP (now only in EQUIP.MRImages); the directional example
     switched to `diffusionencoding`.
   - **The B-matrix, when b-tensor data come.** B = γ²∫F(t)F(t)ᵀdt with F the gradient's
     integral, S = S0·exp(−B:D); linear encoding is B = b·ggᵀ, so b and g say everything until
     then. Planar and spherical encoding need all six elements: by the same rule, six component
     vectors (Bxx, Byy, Bzz, Bxy, Bxz, Byz). The blocks in B form: ball exp(−D·tr B), stick
     exp(−D·nᵀBn), zeppelin exp(−D⊥·tr B − (D∥ − D⊥)·nᵀBn), a tensor exp(−B:D) (no `tensor`
     compartment: decided 2026-11-07, "DTI built in").
   - **More views and blocks** (discussed, not built; status.md): a generic `rotinv(l)` view for
     SMI's rotational invariants (`powder` is l = 0; one function, not l = 2 and 4 in every
     block); a Watson-dispersed view; a cylinder of finite radius (axon diameter: first a crlb
     check that the lab's δ, Δ and maximum gradient can measure it); kurtosis, EPG for multi-echo
     T2, the SPGR steady state, exchange.
   - **DKI, written out** (the owner's question, 2026-11-05: "I feel like that would be a mess";
     examples/dki.jl). S = S0·exp(−b·gᵀDg + b²·MD²·Σ gᵢgⱼgₖgₗWᵢⱼₖₗ/6) (Jensen et
     al., MRM 2005; ADCKurtosis is the same along one direction), 22 parameters: S0, D's six
     elements, W's 15 distinct ones. The signal itself is short: Σ gᵢgⱼgₖgₗWᵢⱼₖₗ is the dot product
     of the 15 elements with their monomials times their multiplicities (1, 4, 6, 12), one helper
     function. Measured: 6 b = 0 and 30 directions at b = 1000 and 2000 (66 images), D's
     eigenvalues 1.7, 0.4, 0.3e-3 mm²/s, an isotropic kurtosis K = 1, SNR 100, 2000 voxels, 4
     threads:
     - the fit: compiled in 8–9 s (under a second for the small models), 130–170 µs per voxel
       in two runs (DTI's 2.5), all 2000 converged from W = 0 without a datastart; means right
       (Dxx 1.698e-3, W1111 0.988, W1122 0.332, W1123 −0.0005 for 0); W1111's SD 0.25 at SNR
       100, MKT's 0.040 (mean 0.997 for 1): the index is far better determined than the
       elements it is made of. No voxel had K < 0 along a measured direction.
     - crlb: did not finish compiling in 23 minutes (stopped). The cause, measured on the same
       model with W's elements partly fixed (scratch dki_scaling.jl, dki_pieces.jl,
       dki_noinline.jl): crlb's first call took 7 s at P = 7, 20 s at P = 10, 118 s at P = 13,
       while the fit's stayed at 4–6 s. Each piece of the kernel compiles in under 3 s at P = 13
       (the noise-free fit 2.2, `hessian!` 2.4, the rest under 1; about 7 s in all), and the
       signal in nested dual numbers in 2 s, so neither the nested duals nor any one piece is it.
       The kernel `_crlb!` alone took 43 s: every piece is `@inline`, so the kernel is one
       function holding all their unrolled P×P static-array code, and the compiler's time grows
       much faster than that code's size. With the kernel's calls marked `@noinline` (call-site
       annotations, the pieces unchanged): the first crlb 18 s at P = 13 and 46 s at P = 22, then
       0.5 ms a call (`crlb!` on its cache 0.1 ms); W1111's SD 0.242 predicted, 0.250 in the fits.
       **Done, "crlb at many parameters" (2026-11-07, branch `noinline-proto`):** the kernel's
       calls go through `@_piece` (src/errors/crlb.jl), which marks them `@noinline` when P >
       `_CRLB_INLINE_MAXP` = 8 and leaves them inlined otherwise; P is a type parameter, so the
       `? :` is folded when compiling. Measured (scratch noinline/kernel_speed.jl, one thread,
       first crlb call in a fresh session, then `crlb!` on its cache; DKI in V with some of V's
       elements fixed at the truth's for P = 10 and 13):

       | model (P) | first call, all inlined | none inlined | threshold 8 | `crlb!`, all inlined | none inlined | threshold 8 |
       |---|---|---|---|---|---|---|
       | T2Decay (2) | 4.9 s | 4.4 s | 4.9 s | 1.02 µs | 1.15 µs | 1.04 µs |
       | T1IR (3) | 3.0 | 2.9 | 3.0 | 3.42 | 3.91 | 3.37 |
       | QMTSIR 5-point (4) | 4.0 | 3.9 | 3.8 | 2.95 | 3.07 | 2.88 |
       | QMTSIR 16-point, kmf free (5) | 4.4 | 3.7 | 4.4 | 4.53 | 4.60 | 4.33 |
       | SMT (3) | 2.4 | 1.9 | 2.4 | 1.14 | 1.17 | 1.12–1.65 |
       | DTI (7) | 6.0 | 3.8 | 6.2 | 6.45 | 7.15 | 6.47 |
       | DKI (10) | 17.0 | 8.4 | 8.3 | 36.4 | 35.7 | 36.1 |
       | DKI (13) | 76.1 | 10.8 | 10.2 | 48.3 | 46.6 | 48.6 |
       | DKI (22) | over 23 min | | 39.0 | | | 96.3 |

       Not inlining costs the small models 2–14% of the kernel's time (T2Decay, T1IR, DTI the
       most), and nothing from P = 10 up; so they keep it, and the threshold sits at 8, where the
       inlined compile is still seconds (P = 7: 6 s; P = 10: 17 s). The values are bit-identical
       in every case, inlined, not inlined or at the threshold (the same arithmetic, only how it
       is compiled differs). Single timings vary by up to 40% on the cloud machine (QMTSIR
       5-point once 5.07 µs, rerun 2.88): the table gives the reruns. Tested: crlb at P = 10 in
       the DKI test set, its SD within 15% of 400 fits'. The fit's own `levmar` compiles in about
       8 s at P = 22; it may meet the same cliff at more parameters. Full runs: EQUIP 2205 (the
       CLI suite not rerun: the change is inside crlb's kernel, and the command line's crlb
       models are below the threshold; 124 at the previous change).
     Where it is a mess, and what to do about each:
     - the names: 15 Wᵢⱼₖₗ keywords and their multiplicities, best generated, not typed.
     - the indices: the mean kurtosis tensor, MKT = (W1111 + W2222 + W3333 + 2(W1122 + W1133 +
       W2233))/5 (Hansen et al., MRM 2013), is linear in W and simple (in the example); MK, AK
       and RK need W rotated into D's eigenframe and, in Tabesh's closed forms (MRM 2011),
       Carlson's elliptic integrals. All of them are functions of the fit, so the `derived`
       hook in crlb and montecarlo gives their errors (item 19; `propagate`, item 16, was the
       plan); AK and RK inherit the eigenvectors' degeneracy where λ2 = λ3.
     - the constraints: K(g) ≥ 0 in every direction, and the signal must not rise with b (b·D(g)·
       K(g) < 3 at b_max), are conditions on many parameters at once, which box bounds cannot
       state (ADCKurtosis notes the same for one direction). DIPY's constrained DKI solves a
       convex problem for it (Dela Haije et al., NeuroImage 2020). Here: check them on the result,
       or penalize them as residuals (as item 15's priors).
     - the start: the log-linear fit is linear in 21 unknowns (ln S0, D, and MD²·W, from which W),
       a datastart as DTI's; it was not needed at SNR 100.
     - the size: 22 parameters fit in static arrays (item 11's limit is about 30–40), and the fit
       is fast enough (a million voxels in about 3 minutes on 4 threads); crlb's compilation is
       the limit, as above, and it is EQUIP's, not DKI's.
18. **The spherical mean's pipeline errors, back with the nerve protocol work** (taken out of
   `directions-proto` on 2026-11-06, the owner: "remove this section ... for now, I don't fully
   understand it ... leave it on the roadmap while we add other models"). Built on `nerve-smt`
   (2026-11-03; "Peripheral nerve SMT" above). The code, its tests and the studies are in
   `lean`'s history at commit a9c7be4 (`nerve-smt`'s last commit; it survives deleting the
   branch): e.g. `git show a9c7be4:src/models/diffusion/orientation.jl`, or `git checkout a9c7be4
   -- src/models/diffusion/orientation.jl bench/floor_sh.jl` to bring files back. What it holds:
   - `orientationcovariance(m, θ, sets)`: the covariance across shells of the error a finite
     direction set makes in a spherical mean, for fibers in a random orientation, exact from the
     Legendre coefficients of the model's directional signal (the addition theorem); added to
     the noise's covariance in crlb, the fitted parameters' spread over orientations.
   - `RicianSphericalMean(m; σ)`: the expected spherical mean of magnitude images, the Rician
     floor acting direction by direction before the average (what the data are; the bias).
   - `sphericalmeannoise(m, θ; snr, directions, averages)`: both, and each shell's Rician noise
     over its directions, as crlb's keywords for the whole pipeline.
   - bench/floor_corrections.jl, floor_robustness.jl, floor_sh.jl: the floor-correction studies
     (their results above, "Peripheral nerve SMT").
   What stays on `directions-proto`: `directionset` and `fiberdirection` (src/models/diffusion/
   directions.jl), the Rician noise tools (`ricianmean`, `RicianMean`, `riciannoise`,
   `ricianbound`), and `sphericalmean`. Two changes when it comes back, discussed with the owner:
   - **Any fiber orientation distribution, not one population.** The directional signal of a
     voxel is one bundle's kernel convolved with the ODF; averaged over random rotations, each
     Legendre order l of the error takes a factor p_l², the ODF's normalized rotational invariant
     (SMI's p_l): one population p_l = 1 (today's formula, the worst case, a nerve), isotropic
     p_l = 0 for l ≥ 2, dispersion and crossings between (a Watson κ gives them).
   - **Each SMT model's equation written once, over the view**, `_smtsignal(m, p, j, view)`:
     `signal!` passes `powder`, the error analysis a cosine ξ, instead of a hand-written
     directional profile per model (the owner disliked the duplication). The same per-model
     function is the "kernel" of the literature, which a directional fit and SMI's rotational
     invariants (a `rotinv(l)` view) could share.
   Needed by the nerve SMT protocol evaluator and search (steps 3 and 4, waiting on the lab).
19. **Derived quantities: the `derived` hook** (proposed 2026-11-07, the owner: "we have to
   derive their radial, axial kurtosis, etc. anyway ... no one cares about the W terms
   directly"; not built). What a user reads off a DTI or DKI fit is not its parameters (six
   tensor elements, fifteen of V = MD²·W) but quantities computed from them: MD, FA, AD, RD; MK,
   AK, RK, MKT. Each is a function of one voxel's parameters, so a model declares them once, and
   every part of EQUIP that reports parameters reports them too: the fit's maps, `savemaps`, the
   fit's standard errors, crlb, montecarlo, and a protocol design's objective. This is the
   vehicle for item 16's outputs (`propagate`): the model, not the caller, knows its g.
   - **The hook**, optional, as `datastart` and `relabel` are:

     ```julia
     derived(m, p) -> nothing or (name = value, ...)
     ```

     `p` is one voxel's every parameter by name, fitted and fixed, ties evaluated (what
     `resolve` gives). The rules are `signal!`'s: no allocation (it runs once per voxel, and
     inside crlb's kernel), generic in the number type (crlb and the fit's standard errors
     differentiate it with dual numbers, the delta method), the same names in every voxel, NaN
     where a quantity is undefined (FA of a zero tensor). The default, `nothing`, means none.
     Written for DTI (`src/models/diffusion/dti.jl`, after its `signal!`):

     ```julia
     function derived(m::DTI, p)
         D = SMatrix{3,3}(p.Dxx, p.Dxy, p.Dxz, p.Dxy, p.Dyy, p.Dyz, p.Dxz, p.Dyz, p.Dzz)
         λ = eigvals(Symmetric(D))                     # ascending; StaticArrays' closed form
         MD = tr(D) / 3
         FA = sqrt(3 / 2) * norm(D - MD * I) / norm(D) # Basser & Pierpaoli 1996, no eigenvalues
         return (MD = MD, FA = FA, AD = λ[3], RD = (λ[1] + λ[2]) / 2)
     end
     ```

     and for DKI in V (with D, λ and the eigenvectors e₁, e₂, e₃ as above; K(n) = V(n)/(nᵀDn)²,
     the apparent kurtosis along n, V(n) = Σ nᵢnⱼnₖnₗVᵢⱼₖₗ): MKT = (V1111 + V2222 + V3333 +
     2(V1122 + V1133 + V2233))/(5·MD²) (Hansen et al., MRM 2013); AK = K(e₁); RK = the mean of
     K over the circle ⊥ e₁; MK = the mean of K over the sphere.
   - **Where it shows** (each a consequence of the hook, nothing per model):
     - `derived(res)`: the maps of a dataset fit (numbers for one voxel), computed per voxel
       from `resolve(res, i)` on threads, NaN where the voxel was not fitted, in the fit's
       precision. Computed when asked, like `stderror(res)`, not stored in `FitResult`: the fit
       pays nothing, and the result's type does not change.
     - `savemaps` writes them beside the parameters' maps (`FA.nii.gz`, ...), and with
       `stderror = true` their standard errors (`FA_SE`), by the delta method from the fit's
       own covariance: √(∇g Σ ∇gᵀ), ∇g from dual numbers through `derived`.
     - `crlb` and `montecarlo`: the derived quantities in the same `bias`, `sd` and `rmse` named
       tuples as the parameters (`c.sd.FA`, `mc.bias.MK`), and in the displayed table below
       them. crlb's: the SD by the delta method; the bias, ∇g·(the parameters' bias) plus the
       noise's own second-order term ½ tr(H Σ), H by nested duals (item 16: FA's noise bias,
       +1.16e-3 predicted against +1.37e-3 measured at SNR 20). montecarlo's: measured from
       each fit's derived values, so the two can be compared, as for the parameters.
     - `ProtocolDesign` and `optimize_protocol`: an objective on a derived quantity, e.g. the
       SD of MK, since they read crlb's named tuples (a DKI protocol designed for MK, not for
       W1111).
   - **Measured before (item 16)**: FA and MD's SDs by the delta method within 1% of 20 000
     Monte Carlo fits (DTI, 6 + 30 images, SNR 50 and 20), the FA–MD correlation (−0.12) too;
     eigenvalues differentiate through StaticArrays' 3×3 symmetric solver (gradients exact,
     Hessians finite unless the eigenvalue is repeated); FA from the six elements checked
     equal to the eigenvalue form to 7e-16.
   - **Open questions, for the owner:**
     - Names: MD, FA, AD, RD (DIPY's and most papers'), or FSL's L1, L2, L3 beside them; the
       principal direction as three maps (V1x, V1y, V1z, FSL's V1) or not derived at all (its
       sign is arbitrary, and at λ1 = λ2 it is undefined: no derivative for crlb).
     - MK and RK: averages over a fixed direction set (simple, smooth for the delta method,
       accurate to about 1e-3 with a few hundred directions: to measure against DIPY's analytic
       values) or Tabesh et al.'s closed forms (MRM 2011, Carlson's elliptic integrals; DIPY's
       default; special cases where eigenvalues are equal). Numerical first, I would suggest.
     - DIPY clips K to [−3/7, 10] in MK, AK and RK by default; EQUIP would report the values
       as computed and leave the K ≥ 0 check to the user (examples/dki.jl), unless the owner
       prefers DIPY's convention.
     - Also derived from a fit's parameters elsewhere: SMT's extra-axonal D⊥ (tortuosity),
       qMT's exchange rate kfm = kmf·PSR, T1 from R1; a model can expose any of them.
   - **Order:** the hook, `derived(res)` and `savemaps` with DTI's four; then crlb and
     montecarlo (the delta method and the second-order bias, tested against Monte Carlo as in
     item 16); then DKI built in (V form) with MK, AK, RK and MKT, checked against DIPY on the
     comparison's data; then the standard-error maps and the protocol objective.
   - **Step 1 built (2026-11-07, branch `loglinear-proto`):** the hook (`derived(::MRIModel, p)
     = nothing`, exported, documented with `datastart` and `relabel`, in `newmodel`'s template
     and the README); `derived(res)` (src/fit/result.jl: numbers for one voxel; for a dataset a
     map per quantity in the fit's precision, on threads, from each voxel's `resolve(res, i)`, NaN
     where a voxel's parameters are not finite; the names from one fitted voxel); `savemaps`
     writing them after the parameters' maps (an error if a derived name is a parameter map's);
     DTI's MD, FA, AD and RD (FA from the six elements by Frobenius norms, AD and RD from
     StaticArrays' closed-form eigenvalues). Names as proposed (MD, FA, AD, RD); the principal
     direction not derived yet (its sign, and λ1 = λ2). Tests: the four against the eigenvalues
     (1e-12), no allocation, FA's gradient through dual numbers against finite differences
     (1e-5), `derived(res)` for one voxel, a masked dataset and Float32, `nothing` for T2Decay,
     and `savemaps` on a DTI fit read back from NIfTI. Full runs: EQUIP 2164, CLI 124.
   - **The principal eigenvector added** (the owner, the same day): V1x, V1y, V1z (FSL's V1, as
     three maps), from StaticArrays' closed-form `eigen` (no allocation; its derivatives with
     dual numbers equal finite differences to 1e-9), put in the upper hemisphere (z ≥ 0, then
     y ≥ 0, then x ≥ 0) so that neighbouring voxels agree, as `directionset`'s. Undefined where
     λ1 = λ2 (oblate or isotropic tensors): the values there are arbitrary, and so are any
     errors crlb would give them. Tested: D·e1 = λ1·e1, unit length, the hemisphere rule, and the
     three maps written by `savemaps`. Full run: EQUIP 2168 (the CLI suite not rerun: the
     command line has no DTI; 124 at the previous change).
   - **Step 3 built: DKI, with MK, AK, RK and MKT as DIPY's** (2026-11-07, the owner: "build dki
     with mk, ak, rk ... I would mirror dipy defaults"). src/models/diffusion/dki.jl, exported:
     the model in V = MD²·W (examples/dki.jl's, now built in), every element at `:loglinear`;
     `derived` gives DTI's quantities (MD, FA, AD, RD, V1) and the indices as DIPY 1.12's fit
     methods compute them by default (`fit.mk()`, `ak()`, `rk()`, `mkt()`): MK and RK in Tabesh
     et al.'s closed forms (Magn Reson Med 2011) with DIPY's functions `_F1m`, `_F2m`, `_G1m`,
     `_G2m` and their limits where eigenvalues are nearly equal (2.5% for F; 1e5·eps absolute for
     G), AK = Ŵ₁₁₁₁·MD²/λ1², MKT = (W1111 + W2222 + W3333 + 2(W1122 + W1133 + W2233))/5 (Hansen
     et al. 2013); each clipped to [−3/7, 10] (DIPY's fit methods' default; its fast path clips
     MK, its slow Python path does not); MK, AK and RK 0 where an eigenvalue is below 2e-7 mm²/s.
     W in D's eigenframe by polarization of the quartic form, W(u,u,v,v) = [W₄(u+v) + W₄(u−v) −
     2W₄(u) − 2W₄(v)]/12, so no 81-element tensor is built. No allocation (2.9 µs per voxel for
     all eleven quantities), differentiable (MK's gradient through dual numbers against finite
     differences, 1e-5).
     - **Against DIPY** (its own values, scratch dkiref/dipy_reference.py, for tensors reaching
       every branch: a general tensor, a prolate one, an oblate one, isotropic D, clipped above
       and below, a negative eigenvalue): AK, RK and MKT to 1e-14 where the principal direction
       is defined; MK to DIPY's own error. Where λ1 = λ2 (oblate, isotropic D) e1 is undefined,
       and AK and RK depend on which vector of the plane each eigensolver returns: they differ
       from DIPY's (AK 0.4527 against 0.4536, oblate), as they would between any two programs;
       MK and MKT agree.
     - **DIPY's MK is 4e-5 off, from its Carlson tolerances**: R_F at errtol 3e-4 is about 1e-6
       off, and MK's formula amplifies it. For the general tensor DIPY gives 1.985127; the sphere
       average of K(n) by quadrature (64 Gauss–Legendre nodes in z × 256 in φ, converged to 12
       digits) is 1.985204, and EQUIP's closed form with errtol 1e-12 gives it to 12 digits (at
       1e-8, R_F was still 3e-10 off and MK 9e-9). So EQUIP keeps DIPY's definitions and clipping
       but not its tolerances; its MK differs from DIPY's by at most a few 1e-5, far below any
       noise. Also checked independently of DIPY: RK the circle average ⊥ e1 (1e-9), AK = K(e1)
       (1e-12), and isotropic D and W giving K in all four.
     - examples/dki.jl uses the built-in model (37 µs per voxel on 4 threads, 4.2 iterations,
       every voxel converged; MK 2.032 against 2.038, AK 0.219 against 0.222, RK 5.38 against
       5.39, MKT 0.997 against 1: for isotropic W and anisotropic D, MKT is K but MK is not).
       The test model for DKI in V (test/runtests.jl) is replaced by the built-in one. Full
       runs: EQUIP 2199, CLI 124.
   - **Also on the way, a guard for `:loglinear`** (the owner asked what happens when it is
     chosen for a model it does not suit; scratch dti/ll_nonlinear.jl): R2Decay (log-linear)
     works; R2BiexpDecay (not log-linear) gives an approximate start, here stuck at 0 (the rates'
     columns are equal there), and the fit reaches the same answer as from its own start; but
     T2Decay with T2 at `:loglinear` started at T2 = 0, no signal, and stayed at its lower bound,
     where the fit stopped and reported convergence: a silent failure. Now `setup` checks that
     the signal where the log fit starts is positive and finite in every observation and
     otherwise refuses, naming the parameters and what to use (a rate, `:data`, a number); a
     real-valued inversion recovery (a signal that changes sign) is refused the same way.
   - **Step 2 built: crlb and montecarlo** (2026-11-07, the owner: "now add derived quantities
     to crlb and montecarlo"; branch `derived-errors`). As planned, the derived quantities follow
     the parameters in every named tuple of `CRLBResult` and `MonteCarloResult` (`c.sd.FA`,
     `mc.bias.MK`), in their tables (below a line "derived from the parameters") and in a
     sweep's rows; `cov`, `vcov`, `stderror`, `cor`, `cond` and `informationmatrix` stay the
     parameters' (StatsAPI's coefficients; `_freenames(c)`, the first P names). crlb's, in
     `_crlbresult` after the kernel (the kernel and the protocol optimizer unchanged): g(θ̂) at
     the noise-free fit (θ̂ with the fit's fixed values, ties followed), the bias g(θ̂) − g(θ),
     exact for what the fit assumes, and the SD by the delta method, √diag(∇g·C·∇gᵀ), ∇g by
     `ForwardDiff.jacobian` through `derived` over the free parameters at θ̂ (static, P
     partials); first order, as the parameters': the noise's own ½tr(∇²g·C) is not added, nor
     Box's bias of the parameters (item 16; both for later, together). SD Inf when C is
     (parameters not determined); on a bound, C's zero rows add nothing. montecarlo's: from
     `derived(res)` of the copies' fits, against `derived(m, θ)`, every statistic as for the
     parameters. A derived name equal to a parameter's is refused. Measured (scratch derr/):
     - DTI, 6 b = 0 + 30 directions, eigenvalues 1.7, 0.4, 0.3e-3 along x, y, z, SNR 50, 20 000
       fits: the SD ratios measured/predicted MD 0.995, FA 0.990, AD 0.993, RD 0.992 (± 0.005),
       the parameters' 0.993–1.004. S0 fixed 5% low: FA biased +0.031 (+4.0%), MD −6.4%.
     - DKI, 22 parameters, 6 + 30 + 30 images, the example's truth, SNR 100, 4000 fits: MD 0.992,
       FA 0.987, AD 0.989, RD 0.987, MK 1.011, AK 0.986, RK 1.010, MKT 0.990 (± 0.011). MK's
       measured bias −0.0039 ± 0.0005 (−0.19%) is the noise's own, which crlb leaves out.
     - Speed: models with no derived quantities unchanged (QMTSIR 4-point `crlb!` 4.57 → 4.56
       µs, a 400-setting sweep 2.06 → 2.05 ms; the branch is known when compiling). A first
       version read the free names as `keys(c.fit)[1:P]` with P from the matrix's size, which the
       compiler cannot type: the sweep 60% slower, until the names came from the result's type.
       DTI's `crlb!` 6.1 → 12.2 µs (seven quantities and their 7×7 Jacobian through the
       eigensolver); DKI's first crlb at P = 22 50 s (39 s before), 0.67 ms a call after.
     - Eigen derivatives: dual numbers through StaticArrays' closed form are right at a diagonal
       tensor too (∂V1y/∂Dxy = 1/(λ1 − λ2) = 769, perturbation theory's; finite differences
       there flip the eigenvector's sign). Where two eigenvalues are equal they are not: the
       eigenvectors of the pair have no derivative, so MK and RK come out NaN for a prolate D
       (λ2 = λ3), V1 where λ1 = λ2; MD, FA, AD, RD, AK, MKT stay finite. Near it (relative gap r
       between λ2 and λ3, W isotropic and anisotropic in the plane): MK's SD right down to r =
       1e-7 (F's limits, 2.5% wide, are smooth); RK's right from r = 1e-4 (r = 1e-5: 12% off for
       the anisotropic W; 1e-6: 100×), because G1 and G2 (DIPY's) switch to their limits only at
       an absolute 1e5·eps, and their general forms lose digits before that; RK's value itself
       is 3% off at r = 1e-7, as DIPY's would be. Not fixed: a dual-number rule for the
       eigenvectors (perturbation theory, the pair's coupling dropped) would give finite
       numbers there, but right ones only when W is symmetric in the pair's plane, and Tabesh's
       limit forms in G use b, not (b + c)/2: silently wrong rather than NaN. Documented
       instead: a truth with eigenvalues at least 0.01% apart, or montecarlo. To decide later:
       G's threshold (relative, wider) at the price of DIPY parity in RK's last digits.
       (Both fixed in the next step, "Truths from derived quantities": the eigenvector rule with
       a guard that keeps NaN where it would be wrong, and G1, G2 without their cancellation.)
     - V1 in montecarlo: each fit's is folded into the upper hemisphere, so for fibers near z = 0
       (here along x) its components' statistics are the folded vectors' (V1x's mean 0, SD 1),
       not crlb's (the unfolded vector's derivative). An angular error (the cone of
       uncertainty) would be the quantity to report for directions; not built.
     - Tests: R2 = 1/T2 on a test model (SD(T2)/T2² exactly; the bias with S0 assumed wrong;
       the parameters' numbers, `vcov`, `cond`, `cor` as without it; a sweep's rows; montecarlo's
       statistics of 1/T2; a clashing name refused); DTI's seven against the delta method
       written out (1e-10) and MD, FA, AD, RD against 3000 fits (4 standard errors); DKI's MD,
       FA, MK, AK, RK, MKT at P = 10 against 400 fits (15%). Full runs: EQUIP 2239, CLI 124.
   - **Truths from derived quantities, and derivatives where eigenvalues are equal** (2026-11-07,
     the owner asked how to simulate with derived quantities prescribed instead of Dxx and the
     rest, and said yes to this; branch `derived-truths`).
     - **The hook, `fromderived(M, q)`**, optional, exported, the reverse of `derived`: the
       parameters that give the derived quantities `q`. `nominal` hands it the call's keywords
       that are not parameters (src/sim/tissues.jl, `_fromderived`), merges what it makes with
       the keywords that are (one given and made: refused), and checks the rest as before; a
       model without it: the names are unknown parameters, as before. Derived quantities seldom
       fix the parameters (MD and FA are two numbers, a tensor six), so the model says which sets
       it takes and what it assumes, and refuses the rest naming what it takes.
     - **DTI's**: an axially symmetric tensor about `direction` (three numbers, any length),
       D = RD·I + (AD − RD)·nnᵀ, from (AD, RD) or (MD, FA) with δ = MD·FA/√(3 − 2FA²), λ1 = MD +
       2δ, λ2 = λ3 = MD − δ (from FA² = 3δ²/(MD² + 2δ²); RD > 0 for FA < 1). Refused: one of a
       pair, both pairs, no direction, FA ≥ 1, RD > AD (oblate: the six elements). A general
       tensor: its elements.
     - **DKI's**: D as DTI's; W from MKT alone (isotropic, W4 = MKT·|g|⁴) or (MK, AK, RK) (W
       axially symmetric about the same axis: W∥ξ⁴ + 6W∥⊥ξ²ρ² + W⊥ρ⁴, three numbers; MK, AK and
       RK are linear in W for a fixed D, so a 3×3 solve on `_kurtosisindices`, the unclipped
       indices, now split from `derived`); V = MD²·W, its fifteen elements by least squares from
       30 directions (exact to rounding). Or neither, and V's elements from the call. Values
       outside DIPY's [−3/7, 10] refused. `derived` gives back MK, AK, RK to 1e-10, MD, FA, AD,
       RD to 1e-12.
     - **The eigensolver's derivatives** (`_symeigen`, dti.jl): for dual numbers, the values'
       decomposition (StaticArrays' closed form) and each partial by first-order perturbation
       theory, dλᵢ = uᵢᵀdDuᵢ, duᵢ = Σⱼ uⱼ(uⱼᵀdDuᵢ)/(λᵢ − λⱼ), the term of an equal pair dropped
       (equal to 1e-6 relative; the closed form splits an exact pair by up to 1.5e-8, measured
       over 10 000 rotations). Differentiating the closed form instead gave NaN for every
       quantity of a prolate tensor. Checked against central differences: DTI's seven for a
       rotated tensor with distinct eigenvalues (2e-9) and an exactly prolate one (3e-10); AD's
       Hessian through nested duals (3e-9). Values unchanged (numbers take the closed form).
     - **The guard** (`_nopartials`: the value kept, every derivative NaN): dropping the pair's
       term is right for what does not depend on the basis chosen in the pair's plane. Where
       λ1 = λ2, e1 is not defined: AD, RD, V1 (DTI), AK and RK (DKI) get NaN derivatives. Where
       a pair is equal, MK and RK are right when W is symmetric about the third axis in that
       plane (its quartic form constant around the plane's circle at 5 angles, W(k, k, n, n) at
       3: `_symmetricin`, checked only where a pair is equal) and NaN otherwise. Why that is the
       condition: near an equal pair (gap δ) the eigenvectors turn within the plane at a rate
       1/δ, and MK changes with that turn at a rate proportional to δ times W's asymmetry in the
       plane; the product has a finite limit, which the dropped term misses, unless W is
       symmetric there.
     - **G1 and G2 without their cancellation**: with s = √(c/b), G1 = (a + b + c)²(s + 2)/
       (18b²(1 + s)²) and G2 = (a + b + c)²/(3√(bc)(√b + √c)²), exact rewritings of DIPY's (the
       factor (b − c)² cancels), no branch, their limits at b = c. DIPY's agree to rounding
       where accurate (1e-12; the DIPY reference tests unchanged); near b = c theirs lost digits
       (RK 3% off at a gap of 1e-7, its derivative 12% at 1e-5). F1 and F2 keep DIPY's limits
       (2.5% wide, means of the pair: smooth).
     - **Measured** (scratch derr/): the gap scan again (W isotropic, and W asymmetric about the
       axis): isotropic, RK 4.3403 ± 0.0792 and MK 1.7919 ± 0.0266 at every gap from 0 to 1e-5
       (before: NaN at 0, RK ± 9.1 at 1e-6, ± 1.6e5 at 1e-7); asymmetric, NaN at gaps ≤ 1e-6 (the
       guard), RK ± 0.1152 from 1e-5 (before: 0.1287 at 1e-5, wrong). At the axially symmetric
       truth (AD 1.7e-3, RD 0.35e-3, MK 1, AK 0.8, RK 1.3, axis θ = 0.3, φ = 1.2; 6 + 30 + 30
       images; 22 parameters): MK, AK, RK, AD, RD's gradients against central differences in
       random directions, 3e-10; against 4000 fits, SD ratios at SNR 100 MD 0.998, FA 1.008, AD
       1.011, RD 1.000, MK 1.030, AK 1.026, RK 1.046, MKT 1.018 (± 0.011), and at SNR 400, the
       same noise, MK 1.008, RK 1.015, the others as at 100: RK's relative SD is 16% at SNR 100,
       so its 4.6% there is the delta method's first order (it shrinks with the noise), while the
       nearly linear ones keep one sample's fluctuation (common random numbers).
     - **Speed**: unchanged. `derived` 0.79 µs (DTI), 2.53 µs (DKI), as before; DTI's `crlb!`
       14.3 µs both; DKI's 0.166 → 0.147 ms, its first crlb 47.5 → 44.6 s.
     - Not done: tissue presets by derived quantities (`nominal(DTI, :WM_DTI; direction)`), and a
       sweep over a derived quantity (`vary = :FA => 0.2:0.1:0.9`), which would go through the
       same hook. Full runs: EQUIP 2283, CLI 124.
   - **Tissue presets and sweeps by derived quantities** (2026-11-07, the owner: "do both";
     branch `derived-sweeps`, the first off `lean` after the merge).
     - **`fromderived(M)`**, the type alone: the names the hook takes (DTI: MD, FA, AD, RD,
       direction; DKI: those and MK, AK, RK, MKT; the default `()`). `_axialtensor` checks
       against it, and refuses a derived quantity that is not a number.
     - **A tissue's derived quantities**: `nominal` hands the hook the tissue's values of the
       names it takes, under the keywords (a keyword wins), and leaves the rest of the tissue
       alone (a preset's T1 for DTI, its MK for DTI).
     - **Presets, `DKI_...`** (src/sim/tissues.jl): Lätt et al., J Magn Reson Imaging
       2013;37:610, Table 1 (36 healthy adults, 19–64 years; 3 T, Philips Achieva; 15
       directions, b = 0, 500, 1000, 2500, 2750 s/mm²; the DKI model by nonlinear least squares
       in Matlab; regional means), ten regions: CC splenium, genu, body, PLIC, centrum
       semiovale, corona radiata, frontal subcortical WM, thalamus, putamen, caudate head; MD
       (the table's μm²/ms, as mm²/s), FA, MK, RK. Not the globus pallidus (its MK, the authors
       say, may be biased). The table gives RD too (headed "mm²/s", a misprint: its values are
       μm²/ms, as MD's), and the axially symmetric tensor from MD and FA reproduces it within 4%
       in every region (thalamus 3.6%, frontal WM 3.7%, the rest under 2%): checked in the tests.
       No AK in the paper (its authors: biased by the non-Gaussian noise), so DKI's (MK, AK, RK)
       needs AK in the call; its error says so. Read from PMC (pmc.ncbi.nlm.nih.gov, added to the
       environment's allowed hosts for it). A second source with AK, and CSF and cortical GM
       values, are still to choose (the owner's call: which study, which protocol).
     - **The truth by derived quantities, everywhere a truth goes**: `Scenario` (so `crlb(m, θ)`
       and `montecarlo(m, θ)`), `crlb!`, `simulate`, and `simulate_with`'s data truth convert
       through `_fromderived`; the scenario keeps the truth as given in a new field, `given`
       (its parameters in `truth`, as before). Bit-identical to giving the parameters (tested
       for simulate and crlb).
     - **Sweeps over a derived quantity**: `vary` may name what the model's hook takes; each
       setting's truth is made again from the truth as given with the varied value in place
       (`_truth_of`), so what stays fixed is what was given: with MD and FA given, varying FA
       holds MD (and the direction). Varying a derived quantity of a truth given by its
       parameters is refused (what would stay fixed?); so is `scenarios`' `vary` over one (its
       truths are completed by `nominal` first; crlb and montecarlo take the spec). Such
       settings take crlb's checked path, not the numbers-only fast one.
     - **Speed**: unchanged once the names come from the cache's types (`_parameternames(c)`):
       QMTSIR 4-point `crlb!` 4.54–4.59 µs (lean 4.51–4.54), a 400-setting sweep 2.01–2.06 ms
       (2.04), DTI's `crlb!` 12.5 µs (12.2–12.4); with `keys(parameters(c.model))` in `crlb!`,
       which builds the model's roles each call, 3–6% slower.
     - Measured on the way: DTI fitted to DKI's signal (K = 1 isotropic, b to 2000 s/mm²; MD
       0.8e-3, FA 0.7) by `simulate_with`: MD biased −0.20e-3 (−25%) and FA +0.15, the kurtosis
       the tensor leaves out (why DTI is fitted on b ≤ 1000).
     - Full runs: EQUIP 2331, CLI 124. (The first run caught a regression the new check made:
       `scenarios(QMTSIR, ...; vary = :T1 => ...)` said "takes parameters" about derived
       quantities QMTSIR does not have; now only a name the model's hook takes is refused that
       way, the rest as before, "not a parameter".)
   - **On the roadmap: DKI's other indices, and DIPY's numerical variants** (the owner,
     2026-11-08: "analytic is enough for now, put on roadmap, along with the other cheap
     indices"). From DIPY 1.12's source (dipy/reconst/dki.py), what it has beyond EQUIP's MK, AK,
     RK and MKT:
     - **More derived quantities, cheap** (functions of the 21 elements, no fit): RTK, the
       rescaled radial tensor kurtosis (Hansen et al., Magn Reson Med 2013, with MKT;
       `radial_tensor_kurtosis`); KFA, the kurtosis fractional anisotropy (Glenn et al., NMR
       Biomed 2015; `kurtosis_fractional_anisotropy`); Kmax, the largest K(n) over a sphere
       (`kurtosis_maximum`, a search from a 100-direction sphere, refined). Each would be another
       name in DKI's `derived`, so it comes into maps, `savemaps`, crlb and montecarlo with no
       other code; to check against DIPY's values as MK, AK and RK were.
     - **The numerical variants** (`analytical=False`): MK as the mean of K(n) over a
       45-direction spherical t-design, RK over 10 directions perpendicular to e1, AK the
       apparent kurtosis along e1. Not more accurate: K(n) is a ratio of polynomials, so 45
       directions do not average it exactly (MK 0.4% low for D = 1.7, 0.4, 0.3e-3 and K = 1:
       2.030 against 2.038; RK and AK the same either way). What differs is the clipping: each
       direction's K is clipped to [−3/7, 10] before the average, the analytic forms clip only the
       result, so a few implausible directions cannot dominate (a nearly zero diffusivity, 1.7,
       0.4, 0.05e-3: MK 5.52 against 7.19), robust maps in CSF, partial volume and noisy voxels,
       but a different definition of MK. Also smooth (no eigenvectors for MK, no special cases),
       so differentiable everywhere for crlb. In DIPY 340× slower (Python loops: 581 against 1.7
       µs per voxel for MK and RK); in EQUIP 45 directions a voxel would cost well under a µs.
       As an option of the model, the analytic forms staying the default (DIPY's, and what papers
       report).
     - Also noted: DIPY's standalone `mean_kurtosis()` clips at 3 by default, its fit method
       `mk()` at 10; EQUIP follows the fit methods. Its other DKI fits: OLS and WLS (the
       default), CLS and CWLS (K ≥ 0 imposed), NLS.
   - **Still open (2026-11-08; tier 4 of the priorities, except the protocol objective, tier
     3):**
     - Standard-error maps from the fit's own covariance (`savemaps` writing `FA_SE`).
     - `compare` (a dataset fit against truth maps) for derived quantities.
     - An angular error for the principal direction (the cone of uncertainty) in place of V1's
       components, whose statistics break near the hemisphere's fold.
     - `vary` over derived quantities in `scenarios()` (sweeps take them since 2026-11-07;
       `scenarios()` refuses them).
     - Derived quantities for the other models: qMT's kfm = kmf·PSR, T1 from R1 (the rate
       forms), SMT's D⊥.
     - A protocol objective on a derived quantity (`ProtocolDesign` for MK's SD).
     - The second-order terms: item 16, tier 2.
20. **Rician fitting, for every model** (the owner, 2026-11-07: "put Rician fitting on the
   roadmap for everything"; not built). Magnitude data are Rician: where the true signal
   approaches the noise (high b; long TE in a T2 decay; an inversion recovery near its null; a
   SIR's early TIs), the measured signal sits on a floor of about 1.25σ, and every fit that
   assumes Gaussian noise reads the floor as signal. Measured for DTI and DKI ("Issues 3 and 4,
   measured", item 17's review): MD 11% low at SNR 10 in white matter, 3% at 20, 1.4% at 30,
   0.5% at 50; least squares on the Rician mean removes it in tissue (≤ 0.6%).
   - **What exists:** `RicianMean(m; σ)` (src/sim/rician.jl), a model whose signal is the
     expected magnitude of `m`'s, for any model, already fittable: `fit(RicianMean(m; σ),
     data)`; and the Rician noise for crlb (`riciannoise`, `ricianbound`). The study fitted it
     started from the Gaussian fit's maps, by hand.
   - **What to build:**
     - **One way to ask for it, for any model**, e.g. `fit(m, data; noise = :rician, σ)`:
       the Gaussian fit first (its starts, `:loglinear`, `datastart`, `:scale`, which assume
       the signal itself), then the Rician fit from it. Today the starts go through
       `RicianMean`'s signal, where `:scale` is wrong (the Rician mean is not proportional to
       S0) and the log start is poorly determined near the floor.
     - **σ as a map:** one σ per voxel (noise varies across the image with coil sensitivity
       and parallel imaging), not only one per observation, and so σ held like a fixed
       parameter; and where σ comes from: the image background (Rayleigh), a noise scan, or
       MP-PCA (Veraart et al., NeuroImage 2016). EQUIP estimates none of them yet.
     - **Bounds where the signal can vanish below the floor:** there any large decay rate
       fits the floor equally, and an unbounded fit runs away (DTI in CSF at b = 2500: MD
       +10 000%, 20% of voxels not converged at SNR 10; with D's diagonal at most 4e-3 mm²/s,
       98–100% converged). Each model's defaults to say which bounds the Rician fit needs.
     - **Speed:** the fit was 10–40× slower than the Gaussian one (150–2000 µs per voxel on 4
       threads against 7–120), from the Bessel functions in dual numbers. A cheaper
       `ricianmean` (a rational or tabulated approximation with its derivative, accurate to
       well below the noise) would make it routine.
     - **The objective:** least squares on the Rician mean (as now) is consistent but not
       efficient at low SNR; the Rician likelihood itself (maximum likelihood) is, and is not a
       sum of squares: Fisher scoring or iteratively reweighted least squares around LM, a
       solver option. Measured, the least-squares version overcorrects DKI's MKT at SNR 20
       (−8%, its RMSE above the Gaussian fit's): the likelihood may not, to test.
     - **Coils:** a sum-of-squares reconstruction from L coils is noncentral χ, not Rician
       (Constantinides et al., MRM 1997); `ricianmean` generalizes with L (and an effective
       σ and L for parallel imaging).
     - **crlb with it:** the Rician fit's own errors (the sandwich with the Rician variance),
       beside the Gaussian fit's floor bias that `riciannoise` already gives: so a protocol
       can be designed for the estimator that will be used.
   - **For every model:** DTI and DKI at high b; T2Decay and T2BiexpDecay at long TE (myelin
     water's short component and the floor are confounded); T1IR, R1IR and QMTSIR with
     magnitude data near the inversion's null; ADCDecay, ADCKurtosis, the SMT models (their
     spherical means already have `averaging = :magnitude` in `RicianMean`). Tests: the floor
     bias recovered against simulations for each, the no-information cases with bounds, σ maps.
   - **Not the remedy for everything:** where the data have no information (CSF at b = 2000 in
     DKI: 0.25% of S0, below the floor) no noise model recovers it; mask, or constrain.
21. **Tests and benchmarks** (the owner, 2026-11-08: "a systematic look at our testing and
   benchmarking infrastructure"; an audit first, then "do 1 and 6 and add other items to
   roadmap").
   - **The audit (2026-11-08, Julia 1.13, 4 threads).** test/runtests.jl, one file of 3480 lines:
     56 top-level test sets, 1263 assertions written, 2347 run (loops); the command line 139,
     MRImages 103. About 225 assertions check errors (178 messages, 45 `@test_throws`), 22 what
     is printed, 20 allocations, about 67 internals. Timed per test set (each run cold, then warm,
     in one session; compile time from `Base.cumulative_compile_time_ns`): 307 s cold in all, of
     which nearly all is compiling; the numerical work itself is about 6 s. The costliest: truths
     from derived quantities 25 s, weighted fits 19, leakage 18, DKI 16 (22 parameters), crlb
     against the textbook formulas 14, DTI 14, @mrimodel 14, noise-free recovery 13 (its fits
     0.09 s). The compiling is the configurations outside the precompile workload (Float32, about
     1.8 s a model; roles, weights, each model's crlb with its second derivatives, DKI); loading the
     test packages does not invalidate EQUIP's precompiled code (checked). Pkg.test's own setup
     adds 0.5 to 8 minutes after a change. So: fewer tests or moving the Monte Carlo runs would
     save little; running only the group being changed, and one configuration per behavior, save
     most.
   - Found: few expected values produced by EQUIP itself (SMT's v_ax 0.6642, untraced, where a
     simple estimate of the T2-weighted fraction gives 0.625; PSR's relative bias −0.1409 and
     −0.02818, EQUIP's own crlb, which montecarlo agrees with elsewhere); many independent
     references (SIR's matrix form, Table 2, the textbook crlb formulas, Gauss–Legendre spherical
     means, DIPY's design matrix, central differences for the bias gradient, leakage and the eigen
     derivatives). Assertions multiplied by loops over random voxels (the time and rate forms:
     180 run for two properties); error messages matched as prose, spread over 30 test sets;
     implementation choices tested as such (`setup`'s fields, `_FIT_INLINE_MAXP`, the shared first
     pass's bit-identity); the zero-allocation check of fit! in five test sets (different
     configurations); statistical checks (Monte Carlo up to 4000 copies, simulations up to 40 000
     voxels) inside the main suite.
   - **Done (branch `test-structure`, 2026-11-08; merged into `main`, the owner: "looks good"):** the one file split into test/setup.jl and
     groups (unit: models, roles, starts_solver; integration: fitting, simulation, crlb,
     montecarlo_sweeps, noise_weights, scenarios_leakage, design, io, diffusion, derived, macro;
     validation, empty), every test set moved as it was (checked against the original
     mechanically: each found once, verbatim; setup.jl the original preamble); groups and files
     chosen by name (`Pkg.test(test_args = ["unit"])`, or directly `julia -t auto --project=test
     test/runtests.jl unit`, test/Project.toml naming EQUIP from the repository). The unit group:
     1010 tests in about a minute. CLAUDE.md: the commands and a section of rules for tests.
     Checked: Pkg.test on Julia 1.13, 2347 passed (5 m 54 s, as before the split); on 1.11.9,
     2294 passed and 53 allocation checks skipped (forced bounds checks); each of the 14 files
     alone, after setup.jl only, every one passing (761, 124, 125, 151, 89, 224, 120, 130, 135,
     32, 52, 230, 120 and 54 tests: 2347, so no file depends on another; 17 s to 68 s each);
     CI, all six jobs.
   - **The workload off while testing, without a file to remember** (the owner, 2026-11-08: "I'm
     going to forget that"): `[preferences.EQUIP] precompile_workload = false` in
     test/Project.toml, committed; a code change rebuilds in seconds. Pkg.test copies the test
     environment's preferences into its sandbox (Pkg's Operations.jl), so CI's EQUIP job writes a
     test/LocalPreferences.toml turning the workload back on (it takes precedence over the
     Project.toml's; both checked), and tests EQUIP as users install it. Which build ran is printed at
     the start of every test run ("EQUIP's precompile workload: on, as users install it" or
     "off, for quick rebuilds"), so a log shows it, and a CI log that says "off" means the
     override was lost (the owner, 2026-11-09: kept the committed setting over a local file; this
     line added).
     TestItemRunner.jl not taken: each `@testitem` self-contained means rewriting the file, for
     what include files already give; reconsider with VS Code's test panel.
   - **To do, in this order:**
     - **With tier 2: the missing independent checks.** Analytic values of the T1 signals
       (inversion recovery's null at TI = T1·ln 2 with full inversion, the Ernst angle for
       T1MFA, saturation recovery at TD = T1); invariances (data scaled: S0 or M0 scaled, the
       rest unchanged; observations reordered: the same fit; crlb's SD ∝ σ and ∝ 1/√NEX; the
       relative errors independent of M0); every built-in model's Jacobian against central
       differences (today only against ForwardDiff, EQUIP's own engine); DKI's MK, AK and RK
       gradients against central differences (FA, AD, RD and V1's are checked); edge cases (an
       all-false mask, empty data, Inf, as many observations as parameters).
     - **With tier 2: the validation group** (test/validation/): crlb against Monte Carlo at
       n = 10 000 (bench/agreement.jl moved), the search to Table 2's optima, convergence with
       SNR, the Rician fits' statistical checks; and the large Monte Carlo checks of the main
       suite moved there if they grow; CI runs it nightly, on demand, and before merging into
       `main`.
     - **When work touches an area: consolidation**, area by area, keeping a list of the
       behaviors each test covers: loops of per-voxel assertions into dataset fits; one error
       check per error path, matching the name the message is about; regression values traced
       or replaced by the property (SMT's v_ax); implementation-detail checks into behavior; one
       configuration per behavior. The compile time saved: to measure (a guess, 20–40% of the
       full suite).
     - **Before performance work: the benchmark framework.** bench/ renamed benchmark/, its
       own environment: `benchmarks.jl` (a BenchmarkTools `SUITE` by area, tagged quick or not:
       models' signals and Jacobians, fit! per model in both precisions, a dataset on threads,
       crlb! and a sweep, leakage, the design's d(x) and a short search), `run.jl` (quick or
       full; JSON with the commit and the machine; compared with a baseline by `judge`, the
       minimum, tolerances for this VM's ±20–50%; an older commit through a git worktree),
       `firstcall.jl` (fresh sessions: load, first calls, the precompile), `baselines/`, and
       `comparison/` (solvers.jl and cases.jl as they are); speed.jl folded into the suite;
       agreement.jl to the validation group. No timing assertions in tests.
