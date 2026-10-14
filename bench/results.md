# Benchmark results

Julia 1.13.0, 4 threads, Intel(R) Xeon(R) Processor @ 2.80GHz

## Single-thread time per voxel (2000 voxels, Float64, identical starting points)

**MonoExp (2 params, 32 echoes)**

| solver | µs/voxel | heap alloc/voxel | speed-up of VoxelFit | rel. difference vs VoxelFit |
|---|---:|---:|---:|---|
| VoxelFit (custom LM) | 1.47 | 0 | 1× | — |
| LsqFit, ForwardDiff Jacobian | 22.64 | 10.0 KiB | 15× | median 5.8e-10, worst 1.3e-08 |
| LsqFit, analytic Jacobian | 16.93 | 7.9 KiB | 12× | median 5.8e-10, worst 1.3e-08 |
| NonlinearSolve LevenbergMarquardt, cached | 605.64 | 104.7 KiB | 412× | median 5.5e-10, worst 1.4e-08 |
| NonlinearSolve TrustRegion, cached | 42.65 | 80 B | 29× | median 5.3e-10, worst 1.4e-08 |

**IVIM (4 params, 16 b-values)**

| solver | µs/voxel | heap alloc/voxel | speed-up of VoxelFit | rel. difference vs VoxelFit |
|---|---:|---:|---:|---|
| VoxelFit (custom LM) | 8.70 | 0 | 1× | — |
| LsqFit, ForwardDiff Jacobian | 38.85 | 13.9 KiB | 4× | median 6.4e-08, worst 1.5e-05 |
| LsqFit, analytic Jacobian | 31.70 | 11.5 KiB | 4× | median 6.4e-08, worst 1.5e-05 |
| NonlinearSolve LevenbergMarquardt, cached | 1205.72 | 133.2 KiB | 139× | median 8.0e-08, worst 1.9e-02 |
| NonlinearSolve TrustRegion, cached | 96.65 | 96 B | 11× | median 7.9e-08, worst 2.0e-01 |

## Whole map: 128×128×64 = 1048576 voxels, 32 echoes, MonoExp

| configuration | time (s) | µs/voxel | voxels/s |
|---|---:|---:|---:|
| VoxelFit Float64, 1 thread | 2.89 | 2.756 | 3.63e+05 |
| VoxelFit Float64, 4 threads | 0.82 | 0.785 | 1.27e+06 |
| VoxelFit Float32, 1 thread | 1.76 | 1.679 | 5.96e+05 |
| VoxelFit Float32, 4 threads | 0.56 | 0.531 | 1.88e+06 |
| LsqFit (analytic Jacobian), 4 threads, first 100000 voxels | 2.55 | 25.473 | 3.93e+04 |
