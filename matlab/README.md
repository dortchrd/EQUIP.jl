# EQUIP from MATLAB

EQUIP fits quantitative MRI models voxel by voxel: qMT by selective inversion recovery (SIR),
T1, T2, diffusion and others. It is written in Julia, a programming language for numerical work;
the MATLAB functions in this folder run it for you, so you need no Julia. You install Julia once,
like any other program, and from then on everything happens in MATLAB.

The functions:

| function | what it does |
|---|---|
| `equip_setup` | gets EQUIP ready (once after installing, and after each update) |
| `equip_fit` | fits a model to your data and returns the maps as MATLAB variables |
| `equip` | runs any EQUIP command, as text (help, file information, CRLB, ...) |

## What you need

- MATLAB R2016b or newer; no toolboxes. (GNU Octave 8 works too.)
- Internet, for the installation.
- A few GB of disk for Julia and EQUIP's packages.
- macOS or Linux. Windows should work but has not been tried yet: tell Richard what happens.

## 1. Install Julia (once, about 5 minutes)

**macOS and Linux.** Open the Terminal app, paste this line and press Enter:

```bash
curl -fsSL https://install.julialang.org | sh
```

Accept what it proposes (press Enter at its questions). When it finishes, close the Terminal
window, open a new one, and type `julia --version`: it should print a version, 1.11 or newer.

**Windows.** Install "Julia" from the Microsoft Store, or in Command Prompt:

```
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```

Then, in a new Command Prompt, `julia --version` should print a version, 1.11 or newer.

If these instructions differ from Julia's own page, https://julialang.org/install, follow that
page.

## 2. Get EQUIP (once)

EQUIP's repository is private for now.

1. Accept the invitation to the repository that GitHub emails you (you need a free GitHub
   account).
2. Go to https://github.com/dortchrd/EQUIP.jl, click the green **Code** button, then
   **Download ZIP**.
3. Unzip it. You get a folder named `EQUIP.jl-main`: rename it `EQUIP.jl` and put it somewhere
   it can stay, for example in Documents. (If you move it later, change the line in step 4.)

If you use git: `git clone https://github.com/dortchrd/EQUIP.jl.git` does the same.

## 3. Set up EQUIP in MATLAB (once, 5 to 15 minutes)

In MATLAB's command window (with your own path to the folder):

```matlab
cd('/Users/you/Documents/EQUIP.jl/matlab')
equip_setup
```

It finds Julia, installs EQUIP's Julia packages (you see their progress; MATLAB is busy until it
finishes), fits a small simulated dataset to check that everything works, and ends by printing
the line for step 4, something like:

```
1. Julia 1.11.6, at /Users/you/.juliaup/bin/julia
2. Installing EQUIP's Julia packages ...
3. A test fit (simulated T2 decay)
   T2 fitted as simulated (largest error 2e-08): EQUIP works from MATLAB.
4. So that MATLAB finds EQUIP every time, put this line in startup.m ...
       addpath('/Users/you/Documents/EQUIP.jl/matlab')
```

## 4. Tell MATLAB where EQUIP is: startup.m (once)

`startup.m` is a file MATLAB runs every time it starts. To open it (MATLAB creates it if it does
not exist):

```matlab
edit(fullfile(userpath, 'startup.m'))
```

Add the line `equip_setup` printed, for example

```matlab
addpath('/Users/you/Documents/EQUIP.jl/matlab')
```

and save. From the next start of MATLAB on, `equip_fit` works from any folder (check:
`which equip_fit` shows its path).

If `equip_setup` said **Julia was not found**: find where Julia is (in a terminal, `which julia`
on macOS and Linux, `where julia` on Windows) and add this line to `startup.m` too, with that
path:

```matlab
setenv('JULIA', '/Users/you/.juliaup/bin/julia')
```

## 5. Fit your SIR data

With your data in a folder as `SIR.nii` (the four SIR volumes in one 4-D image) and `MASK.nii`:

```matlab
cd('/path/to/your/data')
maps = equip_fit('QMTSIR', 'SIR.nii', 'mask', 'MASK.nii', ...
                 'TI', [1007 10 278 10] / 1000, ...     % in seconds, in the order of the volumes
                 'TD', [10 4174 2730 684] / 1000, ...
                 'kmf', 12.5);

figure; imagesc(maps.PSR(:, :, round(end / 2))); axis image; colorbar; title('PSR')
```

This fits the two-pool SIR model (Dortch et al., Magn Reson Med 2018;80:1824) in every voxel of
the mask, with kmf, the exchange rate, fixed at 12.5 1/s as the 4-point scheme assumes.

**In MATLAB** you get a struct, one 3-D array per map, the size of one volume:

| map | what it is |
|---|---|
| `maps.PSR` | the macromolecular-to-free pool size ratio |
| `maps.R1f` | the free pool's longitudinal relaxation rate, 1/s |
| `maps.Sf` | the inversion efficiency of the free pool (about −1) |
| `maps.M0f` | the free pool's equilibrium signal (the signal's scale) |
| `maps.status` | how each voxel's fit ended: 0 not fitted (outside the mask), 1 to 3 converged, 4 ran out of iterations, 5 failed, 6 invalid data (a NaN) |
| `maps.SSE` | the sum of squared residuals: large where the model does not fit |

Voxels outside the mask are NaN.

**On disk**, beside `SIR.nii`: each map as NIfTI (`SIR_PSR.nii.gz`, `SIR_R1f.nii.gz`, ...), to
view in FSLeyes, ITK-SNAP or MRIcroGL, each with a `.json` file that records how it was made
(the model, TI, TD, kmf, the command), and all of them in `SIR_maps.mat`. A new fit of the same
file replaces them; to keep several, give each a name: `'out', 'results/run1_'`.

**Check before you trust the maps:**

- Times are in seconds: divide milliseconds by 1000 (`equip_fit` stops if times look like
  milliseconds).
- TI and TD are in the order of the volumes in `SIR.nii`: here volume 1 has TI = 1007 ms and
  TD = 10 ms, volume 2 TI = 10 ms and TD = 4174 ms, and so on. If your volumes are in another
  order, reorder TI and TD to match.
- Look at `maps.status`: inside the mask it should be 1, 2 or 3 nearly everywhere.

EQUIP also prints a summary of each fit; for a small simulated test (white matter, SNR 200):

```
FitResult: QMTSIR, 6×6×2 voxels
  fitted   32 of 72 voxels (mask), converged 100.0%
  status   ConvergedX 32
  medians over converged voxels (interquartile range):
    PSR = 0.1738   (0.165 to 0.1811)
    R1f = 1.502   (1.477 to 1.514)
    Sf  = -1.017   (-1.026 to -1.01)
    M0f = 1.003   (0.9945 to 1.01)
  fixed    kmf = 12.5, R1m = R1f (tied), Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0
```

(The simulated tissue's PSR is 0.2 and its kmf 5 1/s: fixing kmf at 12.5 makes PSR about 14%
low there, as Dortch et al. 2018 describe.)

## More

- Your data already in MATLAB: give the array instead of the file name, `[x y z 4]`, and the
  mask as an array: `maps = equip_fit('QMTSIR', S, 'mask', M, 'TI', ..., 'TD', ..., 'kmf', 12.5)`.
  The maps then come back only in MATLAB, unless you give `'out'`.
- kmf estimated rather than fixed: `'kmf', 'free'` (for the 5-point or a longer scheme).
- Standard-error maps too: `'stderror', true`.
- How the maps were made: `[maps, info] = equip_fit(...)`, then `info.command` (the command
  run), `info.output` (EQUIP's summary) and `info.sidecar`, the record EQUIP writes beside the
  maps (`SIR_maps.json`), as a struct: the model, TI and TD (`info.sidecar.Acquisition`), the
  data and mask files, EQUIP's version, and every parameter as the fit used it, the defaults
  included: `info.sidecar.Parameters.Sm` is fixed at 0.83, `.R1m` tied to R1f, `.PSR` free
  with its start and bounds.
- All options: `help equip_fit`; a model's options and parameters: `equip('fit QMTSIR --help')`;
  every model: `equip('fit --help')`; what EQUIP reads from a file: `equip('info SIR.nii')`.

Each call starts Julia, which takes a few seconds before the fit begins; the fit itself uses all
of the computer's cores.

## Updating

When Richard says there is a new version: download the ZIP again and replace the `EQUIP.jl`
folder (same place, same name), or `git pull` if you cloned it; then run `equip_setup` again (a
few minutes).

## When something goes wrong

- **"Julia was not found"**: step 4, the `setenv('JULIA', ...)` line.
- **"EQUIP needs Julia 1.11 or newer"**: update Julia; in a terminal, `juliaup update`.
- **`equip_setup` stops while installing**: usually the internet connection; run it again.
- **`equip_fit` stops**: its message is EQUIP's and usually says what is wrong (a missing TI,
  fewer TI values than volumes, a mask of another size).
- **Anything else**: send Richard the message, the command (`info.command`, or the
  `equip_fit(...)` line you ran), your MATLAB version and your computer's system (`ver` prints
  both).
