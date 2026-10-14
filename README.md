# VoxelFit.jl

Fast, memory-efficient nonlinear least-squares fitting of MRI signal models, voxel by voxel, for
building parametric maps (T1, T2, T2\*, ADC, kurtosis, IVIM, ...). It is written in Julia and can be
used from Julia, the terminal, Python and MATLAB.

- **Fast**: a custom Levenberg–Marquardt that allocates nothing. It takes about 1.5 µs per voxel for
  a 2-parameter, 32-echo T2 fit and ~9 µs for 4-parameter IVIM on one core, 4–15× faster than
  LsqFit.jl and 11–400× faster than NonlinearSolve.jl here (see [Benchmarks](#benchmarks)).
  It fits about 1.9 M voxels/s on 4 threads in `Float32`.
- **Memory-efficient**: no per-voxel heap allocation, no transposed copy of the dataset, inputs
  of any numeric type read in place (memory-mapped for `.nii`/`.npy`), and masks skip background.
- **Parallel**: threaded CPU fitting with dynamic load balancing. GPU fitting goes through
  KernelAbstractions (CUDA, AMDGPU, Metal, oneAPI) and uses the *same* solver code.
- **Callable from everywhere**: Julia API, a CLI (NIfTI/NumPy in and out), a Python package
  (in-process via juliacall, zero-copy NumPy) and a MATLAB function.

```julia
using VoxelFit
te  = 10.0:10.0:320.0                                       # echo times (ms)
res = fit_maps(MonoExp(), te, data; mask=brain, precision=Float32)   # data: X×Y×Z×nTE
res[:T2]                                                    # X×Y×Z T2 map
```

## Contents

- [Design](#design)
- [Installation](#installation)
- [Julia API](#julia-api) · [Built-in models](#built-in-models) · [Custom models](#custom-models)
- [Command line](#command-line) · [Python](#python) · [MATLAB](#matlab) · [GPU](#gpu)
- [Benchmarks](#benchmarks) · [Tests](#tests) · [Limitations and next steps](#limitations-and-next-steps)

## Design

These problems have fewer than 10 parameters and fewer than ~100 observations, repeated millions
of times. That changes what matters compared with a general-purpose least-squares library.

**Why a custom LM instead of LsqFit.jl or NonlinearSolve.jl.** Both are general solvers built
around heap-allocated residual vectors and Jacobian matrices, convergence objects and options
structs. For a 2-parameter fit the bookkeeping costs more than the arithmetic. LsqFit allocates
8–14 KiB per fit. NonlinearSolve can reuse a cache (`init`/`reinit!`), but its least-squares
stopping rules test the residual norm, which never reaches zero with noisy data, so its LM takes
~135 iterations per voxel until stall detection fires. Its `SimpleNonlinearSolve` static-array
solvers need the residual as an `SVector{m}`, which is impractical for m = 32–100. Neither runs
inside a GPU kernel for this problem shape. Both remain useful references: the benchmark script
checks that all three reach the same estimates (median relative difference ~1e-9).

**Solver** (`src/lm.jl`):

1. *Normal equations streamed per observation.* The model is evaluated one observation at a
   time, `f(x_i, θ)`. Each value and its gradient (from `ForwardDiff.Dual` numbers seeded on an
   `SVector`, or an analytic override) are accumulated straight into `JᵀWJ` (an `SMatrix{P,P}`),
   `JᵀWr` and the SSE. The m×P Jacobian is never stored, so all state lives in registers,
   whatever the number of observations.
2. *One pass per iteration.* Every trial point is evaluated with derivatives. Most LM steps are
   accepted, and an accepted step then already has its normal equations.
3. *Scale invariance.* Marquardt scaling with MINPACK's monotone diagonal `D = max(D, diag(JᵀJ))`,
   scaled step/gradient stopping tests, and Nielsen's damping update. Parameters of very different
   magnitudes (S0 ~ 10³, D ~ 10⁻³) need no rescaling.
4. *Box constraints.* Steps are projected onto the bounds. Parameters pinned at a bound by the
   gradient leave the linear solve (projected-gradient active set), which avoids the stalls of
   plain clamping.
5. *No exceptions, no allocation, no dynamic dispatch.* A hand-written Cholesky returns a
   success flag instead of throwing, so the same function compiles for GPU kernels (verified by
   compiling it for an NVIDIA PTX target and by running it through KernelAbstractions' POCL
   device backend in the test suite).

**Initial guesses** (`src/builtin_models.jl`) are closed-form linearizations, e.g. log-linear
weighted regression for exponentials, segmented fits for IVIM, the null point for inversion
recovery and the DESPOT1 regression for VFA. They are computed in one pass without allocation, and
usually start LM within a few iterations of the solution.

**Driver** (`src/driver.jl`):

- *Layouts without copies.* NIfTI/MATLAB data are X×Y×Z×M, so a voxel's curve is strided by the
  number of voxels. Each task copies a block of voxels into a small M×128 buffer while reading
  the source contiguously (a blocked transpose) and fits from the buffer. Observation-first data
  (C-ordered NumPy arrays as seen from Julia) are fitted in place.
- *Load balancing.* A fixed pool of tasks pulls 128-voxel chunks from an atomic counter. Each task
  allocates its scratch once, and only masked voxels are queued.
- *Precision.* `Float64` or `Float32` throughout (outputs too). `Float32` is about 1.6× faster on
  CPU, halves memory, and is the natural choice on GPUs.
- *Precompiled.* A PrecompileTools workload covers every built-in model in both precisions and
  both layouts, so a fresh process fits immediately (first-call overhead ≈ 0.2 s).

**GPU** (`ext/VoxelFitKernelAbstractionsExt.jl`): one work item per voxel runs the same
`lm_fit`. Data are staged in voxel-major batches, so neighbouring work items read neighbouring
addresses (coalesced). Batching (`batchsize`, default 2¹⁸ voxels) bounds device memory for any
dataset size.

## Installation

Julia ≥ 1.10. Until the package is registered:

```julia
using Pkg
Pkg.develop(path="/path/to/VoxelFit.jl")     # or Pkg.add(url="<git url>")
```

Start Julia with threads for parallel CPU fitting: `julia -t auto`.

## Julia API

```julia
res = fit_maps(model, x, data;
    obsdim    = ndims(data),   # dimension holding the observations (last or first: no copy)
    mask      = nothing,       # spatial array; zero = skip
    init      = :auto,         # :auto (per-voxel guess), P values, or a (spatial..., P) array
    lower     = nothing,       # P bounds (±Inf for none); defaults from the model
    upper     = nothing,
    covariates= nothing,       # per-voxel known maps, spatial or (spatial..., K), e.g. B1
    weights   = nothing,       # per-observation weights
    precision = Float64,       # or Float32
    stderr    = false,         # also return standard-error maps
    maxiter = 100, xtol, ftol, gtol, damping,     # see ?LMOptions
    ntasks  = Threads.nthreads(), chunksize = 128,
    backend = nothing, batchsize = 2^18)          # GPU: backend = CUDABackend() etc.

res[:T2]          # parameter map (a view)
res.params        # (spatial..., P);  res.names
res.sse           # weighted residual sum of squares
res.iterations, res.status   # status: see ?FitStatus (1–3 converged)
res.stderr        # (spatial..., P) or nothing

fit_maps!(res, model, x, data; ...)    # into preallocated/memory-mapped outputs
r = fit_voxel(model, x, y; ...)        # one curve -> LMResult (θ, sse, JtJ, iterations, status)
```

`x` holds one acquisition setting per observation (TE, b, TI, flip angle). Use an
`nobs × k` matrix when each observation has several settings (e.g. TE and TR).

Status codes: 0 outside mask, 1 converged (step), 2 converged (cost), 3 converged
(gradient), 4 max iterations, 5 failed (non-finite model at the start), 6 non-finite data.

Standard errors are √diag(σ̂²(JᵀWJ)⁻¹) with σ̂² = SSE/(m−P). On synthetic data their z-scores have
unit spread (checked in the tests).

## Built-in models

| model | signal | parameters | `x` |
|---|---|---|---|
| `MonoExp()` | S0·exp(−t/T2) | S0, T2 | echo time |
| `MonoExpOffset()` | S0·exp(−t/T2) + C | S0, T2, C | echo time |
| `BiExp()` | A1·exp(−t/T2a) + A2·exp(−t/T2b) | A1, T2a, A2, T2b | echo time |
| `ADC()` | S0·exp(−bD) | S0, D | b-value |
| `Kurtosis()` | S0·exp(−bD + b²D²K/6) | S0, D, K | b-value |
| `IVIM(bthresh=200)` | S0·(f·exp(−bDp) + (1−f)·exp(−bD)) | S0, f, Dp, D | b-value |
| `InversionRecovery(magnitude=true)` | \|S0·(1 − B·exp(−TI/T1))\| | S0, B, T1 | inversion time |
| `SaturationRecovery()` | S0·(1 − exp(−t/T1)) | S0, T1 | recovery time |
| `SPGR(TR=…)` | M0·sin α·(1−E1)/(1−cos α·E1), α = B1·FA | M0, T1 | flip angle (deg); optional B1 covariate |

Units pass through, so time constants come out in the units of `x`. `MonoExp` and `ADC` have
analytic gradients; the rest use automatic differentiation.

## Custom models

Any of these work in the threaded CPU path; the first three also run on GPUs.

```julia
# 1. A function (θ is an SVector; write it generically so dual numbers pass through)
m = PointModel((t, θ) -> θ[1] * exp(-(t / θ[2])^θ[3]), (:S0, :tau, :beta);
               p0 = (1000, 60, 1), lower = (0, 1e-3, 0.1), upper = (Inf, Inf, 2))

# 2. A type, with data-driven start, bounds and optionally an analytic gradient
struct T2Star <: AbstractModel{2} end
(::T2Star)(t, θ) = θ[1] * exp(-t / θ[2])
VoxelFit.paramnames(::T2Star) = (:S0, :T2s)
VoxelFit.lower_bounds(::T2Star) = (0.0, 1e-6)
VoxelFit.initial_guess(::T2Star, t, y) = SVector(maximum(y), 20.0)
# VoxelFit.value_gradient(::T2Star, t, θ) = (f, SVector(∂f/∂S0, ∂f/∂T2s))   # optional

# 3. A string (what the CLI, Python and MATLAB front ends use)
m = expr_model("S0 * exp(-TE / T2) * (1 - exp(-TR / T1))"; params = (:S0, :T2, :T1),
               x = (:TE, :TR), p0 = (1000, 60, 1000))          # x: nobs × 2 matrix

# 4. Per-voxel covariates: define the 3-argument form; pass covariates = map
(m::MyModel)(x, θ, c) = ...           # c::SVector{K}

# 5. Whole-curve models (e.g. EPG, DCE convolution): CPU only
struct MyCurve <: AbstractCurveModel{3} ... end
VoxelFit.curve!(out, m::MyCurve, x, θ) = ...   # fill out[i] for all observations
```

`examples/dce_tofts_curve_model.jl` implements the extended Tofts model as a curve model: an
O(m) recursive convolution instead of O(m²) pointwise.

## Command line

```bash
bin/voxelfit fit --model MonoExp --x-file te.txt --data echoes.nii.gz --mask brain.nii.gz \
                 --out maps/sub01 [--stderr] [--precision double] [--backend cuda]
bin/voxelfit fit --model 'SPGR(TR=15)' --x 3,10,20 --data vfa.nii.gz --covariate b1.nii.gz --out maps/t1
bin/voxelfit fit --expr 'S0*exp(-TE/T2)+C' --params S0,T2,C --xname TE --p0 1000,50,0 ...
bin/voxelfit models          # list built-in models
bin/voxelfit help
```

- Input: `.nii`, `.nii.gz` or `.npy`; observations along the last axis (`--obs-axis first`
  otherwise). `.nii` and `.npy` inputs are memory-mapped.
- Output: `PREFIX_<param>`, `PREFIX_sse`, `PREFIX_iterations`, `PREFIX_status` (and
  `PREFIX_stderr_<param>`) in the input's format (`--format` to change). NIfTI outputs copy the
  input header (affine, voxel size). `.npy` outputs keep the input's memory order and add
  `PREFIX_params.txt`.
- `bin/voxelfit` runs `julia -t auto --project=<package>`; with Julia ≥ 1.12 you can also run
  `julia -t auto --project=/path/to/VoxelFit.jl -m VoxelFit fit ...`.
- Fitting a 96×96×20×32-echo NIfTI (129k masked voxels) takes 2.6 s end to end, about 2 s of
  which is Julia start-up and package loading. With a system image it takes 1.2 s:
  `julia --project=build build/build_sysimage.jl` once, then
  `export VOXELFIT_SYSIMAGE=$PWD/build/voxelfit.so` (rebuild after updating VoxelFit or Julia).

## Python

```bash
pip install -e /path/to/VoxelFit.jl/python   # pulls juliacall; Julia is installed if missing
```

```python
import numpy as np, voxelfit

res = voxelfit.fit("MonoExp", te, data, mask=mask, precision="float32")   # data: (..., n_echoes)
res["T2"], res.status, res.converged
res = voxelfit.fit("SPGR(TR=15)", fa, vfa, covariates=b1)
res = voxelfit.fit(voxelfit.Expr("S0*exp(-TE/T2)+C", params=["S0", "T2", "C"], x="TE",
                                 p0=[1000, 50, 0], lower=[0, 0, 0]), te, data)
voxelfit.models()
```

Julia runs inside the Python process. The data array is wrapped, not copied (after at most one
conversion to C-contiguous with observations last), and Julia writes results directly into NumPy
arrays. Threads come from `PYTHON_JULIACALL_THREADS` (defaults to `auto`). To use an existing
Julia, set `PYTHON_JULIAPKG_EXE=/path/to/julia`. Tests: `python python/tests/test_voxelfit.py`.

If Julia fails to start with an OpenSSL/libcurl symbol-version error (`OPENSSL_3.x not found`),
Python has already loaded a system OpenSSL older than the one Julia's libcurl needs. Preload
Julia's copies, e.g. `LD_PRELOAD=$JULIA_DIR/lib/libssl.so.3:$JULIA_DIR/lib/libcrypto.so.3`
(this was needed with a conda-forge Julia on the development machine).

## MATLAB

```matlab
addpath /path/to/VoxelFit.jl/matlab
res = voxelfit('MonoExp', te, img, 'Mask', brain);           % img: X×Y×Z×nTE
imagesc(res.T2(:,:,10), [0 150]); axis image; colorbar
res = voxelfit('SPGR(TR=15)', fa, vfa, 'Covariates', b1map, 'Precision', 'double');
m = struct('expr', 'S0*exp(-TE/T2)+C', 'params', {{'S0','T2','C'}}, 'xname', 'TE');
res = voxelfit(m, te, img, 'P0', [1000 50 0], 'StdErr', true);
```

The wrapper writes Fortran-ordered `.npy` files (MATLAB's own memory layout, so nothing is
permuted), runs the CLI, and reads the maps back. No toolboxes are needed and it also runs in
GNU Octave (`matlab/test_voxelfit.m` passes in Octave 10). Each call starts Julia: about 3 s
of overhead, 0.7 s with `'Sysimage', '/path/to/voxelfit.so'`. So fit whole volumes per call. Options: `'Threads'`, `'Julia'`
(or env `VOXELFIT_JULIA`), `'Backend'`, `'MaxIter'`, `'XTol'`, `'Lower'`, `'Upper'`, `'Weights'`.

## GPU

```julia
using VoxelFit, CUDA          # or AMDGPU (ROCBackend()), Metal (MetalBackend()), oneAPI
res = fit_maps(MonoExp(), te, data; mask, precision=Float32, backend=CUDABackend())
```

The KernelAbstractions extension loads automatically with any GPU package. Pointwise models
(built-in, `PointModel`, `expr_model`, your own `isbits` types) are supported; curve models are
CPU-only. Use `Float32` on consumer GPUs. Data stay on the host and stream through in batches.

**Validation status.** The kernel is tested through KernelAbstractions' POCL backend
(OpenCL/SPIR-V device compilation, including argument conversion and buffer reuse across
batches), and the solver compiles for the NVIDIA PTX target with no allocations or runtime calls
for every built-in model. It has **not** yet been run on CUDA/ROCm hardware: do that before
relying on it, and benchmark `batchsize` and work-group size there.

## Benchmarks

`julia -t auto --project=bench bench/benchmarks.jl` (Julia 1.13, 4 vCPU Xeon @ 2.8 GHz). Each
solver fits the same 2000 noisy voxels from the same starting points.

**MonoExp (2 params, 32 echoes)**, one thread:

| solver | µs/voxel | heap/voxel | VoxelFit speed-up | rel. diff vs VoxelFit |
|---|---:|---:|---:|---|
| **VoxelFit** | **1.47** | **0** | 1× | — |
| LsqFit, ForwardDiff Jacobian | 22.6 | 10.0 KiB | 15× | 6e-10 median |
| LsqFit, analytic Jacobian | 16.9 | 7.9 KiB | 12× | 6e-10 median |
| NonlinearSolve LevenbergMarquardt (cached) | 606 | 105 KiB | 412× | 6e-10 median |
| NonlinearSolve TrustRegion (cached) | 42.7 | 80 B | 29× | 5e-10 median |

**IVIM (4 params, 16 b-values)**, one thread:

| solver | µs/voxel | heap/voxel | VoxelFit speed-up | rel. diff vs VoxelFit |
|---|---:|---:|---:|---|
| **VoxelFit** | **8.70** | **0** | 1× | — |
| LsqFit, ForwardDiff Jacobian | 38.9 | 13.9 KiB | 4× | 6e-8 median |
| LsqFit, analytic Jacobian | 31.7 | 11.5 KiB | 4× | 6e-8 median |
| NonlinearSolve LevenbergMarquardt (cached) | 1206 | 133 KiB | 139× | 8e-8 median (worst 2e-2) |
| NonlinearSolve TrustRegion (cached) | 96.7 | 96 B | 11× | 8e-8 median (worst 0.2) |

(NonlinearSolve has no bound constraints, so on a few IVIM voxels it settles in a different
local minimum.)

**Whole map**, 128×128×64 = 1,048,576 voxels, 32 echoes:

| configuration | time | voxels/s |
|---|---:|---:|
| VoxelFit Float64, 1 thread | 2.89 s | 3.6e5 |
| VoxelFit Float64, 4 threads | 0.82 s | 1.3e6 |
| VoxelFit Float32, 1 thread | 1.76 s | 6.0e5 |
| VoxelFit Float32, 4 threads | 0.56 s | 1.9e6 |
| LsqFit (analytic Jacobian), 4 threads | (27 s extrapolated) | 3.9e4 |

## Tests

```bash
julia -t 4 --project=. -e 'using Pkg; Pkg.test()'      # 164 tests
python python/tests/test_voxelfit.py                    # Python front end
octave-cli matlab/test_voxelfit.m                       # MATLAB front end (or run in MATLAB)
```

The tests cover noise-free recovery for every model in both precisions, zero allocation, analytic
vs AD gradients, layout equivalence (obs first/middle/last, 1 vs many threads), masks, NaN/Inf
data, integer inputs, warm starts, active bounds, weights, calibrated standard errors, B1
covariates, custom/expression/multi-setting/curve models, the KernelAbstractions backend, `.npy`
I/O and the CLI (NIfTI and NumPy).

## Limitations and next steps

- **Rician noise.** Least squares on magnitude images is biased at low SNR (e.g. late echoes).
  `MonoExpOffset` absorbs a noise floor. A Rician likelihood or noise-floor-corrected model is a
  natural addition.
- **Multi-modal models** (IVIM, bi-exponential). LM finds the local minimum nearest the start.
  Variable projection (eliminating linear amplitudes) and a coarse grid/dictionary start
  would make these more robust; both fit the current design (grid search is also GPU-friendly).
- **GPU** hardware validation and tuning (see above).
- **MATLAB start-up.** The file-based CLI costs ~1–3 s per call. A MEX/C-ABI library
  (PackageCompiler `create_library`, or `juliac --trim` on Julia ≥ 1.12) or a persistent server
  process would remove it.
- **Complex data, per-voxel `x`** (e.g. per-voxel b-vectors) and robust losses are not yet
  supported.
