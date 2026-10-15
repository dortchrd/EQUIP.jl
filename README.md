# VoxelFit.jl

Fast, memory-efficient nonlinear least-squares fitting of MRI signal models, voxel by voxel, for
building parametric maps (T1, T2, T2\*, ADC, kurtosis, IVIM, SIR qMT, ...). It is written in Julia
and can be used from Julia, the terminal, Python and MATLAB.

- **Fast**: a custom Levenberg–Marquardt solver that allocates no memory per voxel. About 1.4 µs
  per voxel for a 2-parameter, 32-echo T2 fit on one core, and about 2.6 million voxels/s on
  4 threads in `Float32`. That is 8–17× faster than LsqFit.jl and 17–400× faster than
  NonlinearSolve.jl here (see [Benchmarks](#benchmarks)).
- **Flexible parameters**: every parameter in a signal equation can be estimated, fixed to a
  constant, given as a per-voxel map (e.g. B1), or tied to another parameter, chosen for each fit
  without changing the model.
- **Easy to extend**: a model is a struct, a list of its parameters and one `signal!` function.
- **Callable from everywhere**: Julia API, a command-line tool (NIfTI/NumPy in and out), a
  Python package (in-process, zero-copy NumPy) and a MATLAB function.

```julia
using VoxelFit
te  = 10.0:10.0:320.0                                           # echo times (ms)
res = fit_maps(MonoExp(), te, data; mask = brain, precision = Float32)   # data: X×Y×Z×nTE
res[:T2]                                                        # X×Y×Z T2 map

acq = (TI = ti, TD = td)                                        # SIR-FSE protocol (s)
res = fit_maps(SIRFSE(), acq, data; mask = brain, fixed = (kmf = 12.5,))
```

The design and its reasoning are in [`docs/design.md`](docs/design.md).

## Contents

- [Concepts](#concepts) · [Writing a model](#writing-a-model) · [Fitting](#fitting) · [Built-in models](#built-in-models)
- [Command line](#command-line) · [Python](#python) · [MATLAB](#matlab)
- [How it works](#how-it-works) · [Benchmarks](#benchmarks) · [Tests](#tests) · [Roadmap](#roadmap)

## Concepts

A fit combines four kinds of input:

| | varies across voxels? | examples | given as |
|---|---|---|---|
| **Model** | no | magnitude or signed data | the model struct: `SIRFSE(magnitude = true)` |
| **Protocol** | no | TE, TI, TD, flip angles, TR | a named tuple `acq`: vectors are per observation, scalars constant. A plain vector works for one-setting models. |
| **Parameters** | each one: estimated, constant, per-voxel map, or tied | PSR, kmf, R1f, B1, Sm | the model's defaults, overridden with `free`, `fixed` and `start` |
| **Data** | yes | the images | an N-d array, observations along one dimension |

## Writing a model

A model is three pieces. Here is the SIR-FSE model that ships with the package (abridged):

```julia
struct SIRFSE <: AbstractModel                 # 1. settings, the same for every voxel
    magnitude::Bool
end

VoxelFit.parameters(::SIRFSE) = (              # 2. every parameter, with its default role
    pmf = Estimate(0.1;   lower = 0,    upper = 1),
    R1f = Estimate(1.0;   lower = 0.05, upper = 10),
    Sf  = Estimate(-0.95; lower = -1.2, upper = 0),
    M0f = Estimate(:auto; lower = 0),          # :auto -> initial_guess
    kmf = Fixed(12.5),
    R1m = Fixed(p -> p.R1f),                   # tied to R1f
    Sm  = Fixed(0.83),
)
VoxelFit.protocol(::SIRFSE) = (:TI, :TD)       # settings it reads (checked before fitting)

function VoxelFit.signal!(out, m::SIRFSE, acq, p)  # 3. one voxel's signal at every observation
    (; pmf, R1f, Sf, M0f, kmf, R1m, Sm) = p        # parameters by name, whatever their role
    ...
    for j in eachindex(out)
        (; TI, TD) = obs(acq, j)                    # observation j's settings
        ...
        out[j] = m.magnitude ? abs(s) : s
    end
    return out
end

VoxelFit.initial_guess(::SIRFSE, acq, y, p) = (M0f = maximum(abs, y),)   # optional
```

Rules for `signal!`: compute the noise-free signal for every observation, read parameters by
name, and write it generically so dual numbers can pass through (derivatives are automatic). It
must not contain noise or state.

For quick experiments, `FunctionModel(f!, roles)` wraps a plain function, and `expr_model`
builds a model from a string:

```julia
m = expr_model("S0 * exp(-TE / T2) + C"; params = (:S0, :T2, :C), protocol = (:TE,),
               p0 = (1000, 50, 0), lower = (0, 0, 0))
```

## Fitting

```julia
res = fit_maps(model, acq, data;
    obsdim    = ndims(data),    # dimension holding the observations (last or first: no copy)
    mask      = nothing,        # spatial array; zero = skip
    fixed     = (;),            # e.g. (kmf = 12.5, Sm = sm_map, R1m = p -> p.R1f)
    free      = (),             # e.g. (:kmf,) or (:kmf => Estimate(20.0; lower = 5, upper = 50),)
    start     = (;),            # starting values or maps, e.g. (pmf = 0.1,), or a previous MapResult
    weights   = nothing,        # per-observation weights
    precision = Float64,        # or Float32
    stderr    = false,          # also return standard-error maps
    maxiter = 100, xtol, ftol, gtol, damping,     # see ?LMOptions
    ntasks = Threads.nthreads(), chunksize = 128)

res[:pmf]                       # map of an estimated parameter (a view)
res.names, res.params           # (spatial..., P)
res.sse, res.iterations, res.status, res.stderr
```

- **Fixed maps:** a per-voxel map must have the data's spatial size. A NaN or Inf in a fixed map
  marks that voxel invalid (status 6), as does non-finite data.
- **Single voxels:** a reusable cache fits one voxel at a time without allocating:

  ```julia
  cache = init(SIRFSE(), acq; fixed = (kmf = 12.5,))
  r = fit!(cache, y)            # r.params.pmf, r.sse, r.status, standard_errors(r, length(y))
  ```

- **Simulation:** `signal(model, acq, p)` gives the noise-free signal for any full parameter set.

Status codes: 0 outside mask, 1 converged (step), 2 converged (cost), 3 converged (gradient),
4 max iterations, 5 failed (non-finite model at the start), 6 invalid (non-finite data or
fixed-map value). Start Julia with threads (`julia -t auto`) for parallel fitting.

## Built-in models

| model | signal | protocol | estimated by default | fixed by default |
|---|---|---|---|---|
| `MonoExp()` | S0·exp(−TE/T2) | `TE` | S0, T2 | |
| `MonoExpOffset()` | S0·exp(−TE/T2) + C | `TE` | S0, T2, C | |
| `BiExp()` | A1·exp(−TE/T2a) + A2·exp(−TE/T2b) | `TE` | A1, T2a, A2, T2b | |
| `ADC()` | S0·exp(−bD) | `b` | S0, D | |
| `Kurtosis()` | S0·exp(−bD + b²D²K/6) | `b` | S0, D, K | |
| `IVIM(bthresh=200)` | S0·(f·exp(−bDp) + (1−f)·exp(−bD)) | `b` | S0, f, Dp, D | |
| `InversionRecovery(magnitude=true)` | \|S0·(1 − B·exp(−TI/T1))\| | `TI` | S0, B, T1 | |
| `SaturationRecovery()` | S0·(1 − exp(−t/T1)) | `t` | S0, T1 | |
| `SPGR()` | M0·sin α·(1−E1)/(1−cos α·E1), α = B1·FA | `FA` (deg), `TR` | M0, T1 | B1 = 1 |
| `SIRFSE(magnitude=true)` | two-pool SIR-FSE (Eq. 1 of Dortch et al., MRM 2018) | `TI`, `TD` (s) | pmf, R1f, Sf, M0f | kmf = 12.5, R1m = R1f, Sm = 0.83 |

Units pass through: time constants come out in the units of the protocol's times.

## Command line

```bash
bin/voxelfit fit --model MonoExp --x-file te.txt --data echoes.nii.gz --mask brain.nii.gz --out maps/sub01
bin/voxelfit fit --model SIRFSE --acq TI=0.01,0.05,0.056,0.277,0.843 --acq TD=3.27,4.489,1.652,2.922,0.01 \
                 --data sir.nii.gz --fixed kmf=12.5 --out maps/sir
bin/voxelfit fit --model SPGR --acq FA=3,10,20 --acq TR=15 --fixed B1=b1.nii.gz --data vfa.nii.gz --out maps/t1
bin/voxelfit fit --model SIRFSE --acq-file protocol.txt --free kmf --bounds kmf=1,50 --data sir.nii.gz --out maps/sir5
bin/voxelfit models          # built-in models with their protocol and parameters
bin/voxelfit help
```

- **Protocol:** `--x`/`--x-file` for one setting, `--acq NAME=VALUES` (repeatable), or
  `--acq-file` (a text table whose header names the columns).
- **Parameters:** `--fixed NAME=VALUE` or `NAME=map.nii.gz`, `--free NAMES`,
  `--start NAME=VALUE|FILE`, `--bounds NAME=LO,HI`.
- **Input:** `.nii`, `.nii.gz` or `.npy`, observations along the last axis (`--obs-axis first`
  otherwise). `.nii` and `.npy` inputs are memory-mapped.
- **Output:** `PREFIX_<param>`, `PREFIX_sse`, `PREFIX_iterations`, `PREFIX_status` (and
  `PREFIX_stderr_<param>`) in the input's format. NIfTI outputs copy the input header.
- **Start-up:** about 2.5 s end to end for a typical slab. Build a system image once to cut that
  to about 1 s: `julia --project=build build/build_sysimage.jl`, then
  `export VOXELFIT_SYSIMAGE=$PWD/build/voxelfit.so`.

## Python

```bash
pip install -e /path/to/VoxelFit.jl/python   # pulls juliacall; Julia is installed if missing
```

```python
import numpy as np, voxelfit

res = voxelfit.fit("MonoExp", te, data, mask=mask, precision="float32")   # data: (..., n_echoes)
res = voxelfit.fit("SIRFSE", {"TI": ti, "TD": td}, data, fixed={"kmf": 12.5, "Sm": sm_map})
res = voxelfit.fit("SIRFSE", {"TI": ti, "TD": td}, data, free={"kmf": voxelfit.Estimate(20, 5, 50)})
res = voxelfit.fit("SPGR", {"FA": fa, "TR": 15.0}, vfa, fixed={"B1": b1}, start=previous_result)
res["pmf"], res.status, res.converged
voxelfit.models()
```

Julia runs inside the Python process; data, masks and maps are wrapped, not copied, and Julia
writes results directly into NumPy arrays. Threads come from `PYTHON_JULIACALL_THREADS`
(default `auto`). To use an existing Julia, set `PYTHON_JULIAPKG_EXE=/path/to/julia`. Tests:
`python python/tests/test_voxelfit.py`.

If Julia fails to start with an OpenSSL/libcurl symbol-version error (`OPENSSL_3.x not found`),
Python has already loaded a system OpenSSL older than the one Julia's libcurl needs. Preload
Julia's copies, e.g. `LD_PRELOAD=$JULIA_DIR/lib/libssl.so.3:$JULIA_DIR/lib/libcrypto.so.3`.

## MATLAB

```matlab
addpath /path/to/VoxelFit.jl/matlab
res = voxelfit('MonoExp', te, img, 'Mask', brain);                        % img: X×Y×Z×nTE
acq = struct('TI', ti, 'TD', td);
res = voxelfit('SIRFSE', acq, img, 'Fixed', struct('kmf', 12.5, 'Sm', smMap), 'Mask', brain);
res = voxelfit('SIRFSE', acq, img, 'Free', {'kmf'}, 'Bounds', struct('kmf', [1 50]));
res = voxelfit('SPGR', struct('FA', fa, 'TR', 15), vfa, 'Fixed', struct('B1', b1map));
```

The wrapper writes Fortran-ordered `.npy` files (MATLAB's own memory layout), runs the
command-line tool and reads the maps back. No toolboxes are needed, and it also runs in GNU
Octave (`matlab/test_voxelfit.m` passes in Octave 10). Each call starts Julia: about 3 s, or
0.7 s with `'Sysimage', '/path/to/voxelfit.so'`. So fit whole volumes per call.

## How it works

- **Solver** (`src/lm.jl`): for each iteration, the voxel's full parameter set is assembled
  (estimates as dual numbers, fixed values, resolved ties) and `signal!` is called into a
  per-thread buffer. SSE, JᵀJ and Jᵀr are accumulated from that buffer into static arrays, so the
  m×P Jacobian is never stored. The step is Levenberg–Marquardt with Marquardt scaling (MINPACK's
  monotone diagonal), Nielsen's damping update, scale-free stopping tests, and box constraints
  by projection with an active set. A hand-written Cholesky returns a flag instead of throwing.
- **Parameter roles** (`src/cache.jl`): `init` resolves the model's defaults plus `free`,
  `fixed` and `start` once. The names of estimated parameters become a compile-time constant, so
  assembling a voxel's parameters costs nothing at run time (measured).
- **Driver** (`src/driver.jl`): data stay in their N-d shape. Each thread copies a block of 128
  voxels into a small buffer while reading the source contiguously (a blocked transpose); data
  with observations first are fitted in place. A fixed pool of tasks, each with its own cache,
  pulls chunks of the masked voxel list from an atomic counter.
- **Starting values**: model guesses come from closed-form linearizations (log-linear fits,
  DESPOT1, null-point, segmented IVIM), computed in one pass without allocation.
- **Precompiled**: a workload covers every built-in model in both precisions and both layouts,
  so a fresh process fits immediately.

## Benchmarks

`julia -t auto --project=bench bench/benchmarks.jl` (Julia 1.13, 4 vCPU Xeon @ 2.8 GHz). Each
solver fits the same 2000 noisy voxels from the same starting points.

| per voxel, one thread | MonoExp (2 params, 32 echoes) | IVIM (4 params, 16 b-values) |
|---|---:|---:|
| **VoxelFit** (no heap allocation) | **1.35 µs** | **4.95 µs** |
| LsqFit, analytic Jacobian | 15.7 µs (12×), 7.9 KiB | 40.0 µs (8×), 11.6 KiB |
| LsqFit, ForwardDiff Jacobian | 23–96 µs across runs, 10 KiB | 42.4 µs (9×), 14 KiB |
| NonlinearSolve TrustRegion (cached) | 37.2 µs (28×) | 83.9 µs (17×) |
| NonlinearSolve LevenbergMarquardt (cached) | 553 µs (410×), 105 KiB | 1215 µs (245×), 136 KiB |

All solvers agree with VoxelFit to a median relative difference of about 1e-9 (MonoExp) and 1e-7
(IVIM). The one IVIM voxel where some differ has identical SSE for both answers: its
pseudo-diffusion term has decayed to zero by the first nonzero b-value, so Dp is not
identifiable. NonlinearSolve's LM is slow here because its stopping rule waits for the residual
to reach zero, which never happens with noisy data.

Whole map, 128×128×64 = 1,048,576 voxels, 32 echoes: 2.12 s (Float64) and 1.67 s (Float32) on
one thread; 0.68 s and **0.41 s (2.6 million voxels/s)** on 4 threads. LsqFit on 4 threads
manages about 47,000 voxels/s.

## Tests

```bash
julia -t 4 --project=. -e 'using Pkg; Pkg.test()'       # 175 tests
python python/tests/test_voxelfit.py                     # Python front end
octave-cli matlab/test_voxelfit.m                        # MATLAB front end (or run in MATLAB)
```

The tests cover:
- noise-free recovery for every model in both precisions, with no memory allocation;
- SIRFSE against Eq. 1 of the 2018 paper (matrix exponential), for magnitude and signed data;
- parameter roles: defaults, `fixed`, `free`, ties, error messages;
- per-voxel maps, including invalid map values, and B1 correction;
- data layouts and thread counts, masks, NaN/Inf and integer data;
- warm starts, active bounds, weights, and calibrated standard errors;
- custom, expression and multi-setting models;
- `.npy` I/O and the command-line tool (NIfTI and NumPy, `--acq`, `--fixed` maps, `--free`,
  `--bounds`).

## Roadmap

See [`docs/design.md`](docs/design.md). Next:
1. **CRLB for unbiased and biased estimators**, following Dortch et al., MRM 2018: Fisher
   information, MSE = b² + [(I + ∇b)F⁻¹(I + ∇b)ᵀ] for constrained models such as fixed kmf, the
   protocol-design objective and sensitivity efficiency. Plus `simulate` with Gaussian or Rician
   noise and a Monte Carlo summary.
2. **Starting values from neighbours** (candidate starts and a second refit pass) and linearized
   sensitivity maps (e.g. ∂PSR/∂kmf).
3. **Later:** GPU fitting (removed in 0.2; the 0.1 KernelAbstractions version is in the git
   history), coarse-to-fine starts, Rician-likelihood fitting.
