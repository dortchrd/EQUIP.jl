# EQUIP against the one-shot version

The first version of this package, VoxelFit 0.1 (branch `claude-oneshot`, 2026-10-14), against
EQUIP, on the eight models both have:

| VoxelFit | EQUIP |
|---|---|
| MonoExp | T2Decay |
| MonoExpOffset | T2DecayOffset |
| BiExp | T2BiexpDecay |
| ADC | ADCDecay |
| Kurtosis | ADCKurtosis |
| InversionRecovery | T1IR |
| SaturationRecovery | T1SR |
| SPGR | T1MFA |

`./run.sh` does every step (about 10 minutes, 15 the first time). It checks out the one-shot
version beside this repository (a git worktree in `oneshot/`, or `ONESHOT=/path` to reuse a
checkout), and the two packages run in separate processes: they share a package UUID.

| script | what it does |
|---|---|
| `cases.jl` | the cases (acquisitions in seconds, truths) and the conversions between the two packages' parameters |
| `gen.jl` | the data (EQUIP's `simulate`, 20,000 voxels a case, SNR 50 and 200, Gaussian noise) into `data/`; checks each EQUIP model gives VoxelFit's equation |
| `vf.jl`, `eq.jl` | each package's fits: `acc` (its own starts, and for EQUIP also VoxelFit's starts), `time` (1 thread and more, Float64 and Float32, best of 3) |
| `compare.jl` | convergence, bias and SD against the truth, and voxel-by-voxel agreement |
| `diagnose.jl` | where the two differ: EQUIP's bounds, and which reached the lower sum of squares |
| `iters.jl` | iterations per voxel, beside EQUIP's default starts |
| `timetable.jl` | the timings as a table |
| `first_vf.jl`, `first_eq.jl` | load time and first fit, in fresh processes |
| `smart.jl` | branch `bestof-proto` only: EQUIP's smart BestOf starts on the same data, against VoxelFit's fits (sums of squares) and EQUIP's defaults (`julia -t 1 --project=../.. smart.jl`, after `run.sh`) |

`data/`, `out/` and `oneshot/` are not committed. The results of 2026-11-20 are in
docs/design.md, "Compared with the one-shot version, VoxelFit 0.1".
