# Benchmark results

Julia 1.13.0, 4 threads, Intel(R) Xeon(R) Processor @ 2.80GHz

## Single-thread time per voxel (2000 voxels, Float64, identical starting points)

**MonoExp (2 params, 32 echoes)**

| solver | µs/voxel | heap alloc/voxel | speed-up of VoxelFit | rel. difference vs VoxelFit |
|---|---:|---:|---:|---|
| VoxelFit (custom LM) | 1.35 | 0 | 1× | — |
| LsqFit, ForwardDiff Jacobian | 95.83 | 10.0 KiB | 71× | median 7.1e-10, worst 1.4e-08 |
| LsqFit, analytic Jacobian | 15.67 | 7.9 KiB | 12× | median 7.1e-10, worst 1.4e-08 |
| NonlinearSolve LevenbergMarquardt, cached | 552.89 | 104.6 KiB | 410× | median 6.8e-10, worst 1.4e-08 |
| NonlinearSolve TrustRegion, cached | 37.18 | 80 B | 28× | median 6.6e-10, worst 1.4e-08 |

**IVIM (4 params, 16 b-values)**

| solver | µs/voxel | heap alloc/voxel | speed-up of VoxelFit | rel. difference vs VoxelFit |
|---|---:|---:|---:|---|
| VoxelFit (custom LM) | 4.95 | 0 | 1× | — |
| LsqFit, ForwardDiff Jacobian | 42.44 | 14.0 KiB | 9× | median 6.5e-08, worst 8.8e+00 |
| LsqFit, analytic Jacobian | 40.02 | 11.6 KiB | 8× | median 6.5e-08, worst 8.8e+00 |
| NonlinearSolve LevenbergMarquardt, cached | 1214.66 | 135.7 KiB | 245× | median 7.9e-08, worst 8.7e+00 |
| NonlinearSolve TrustRegion, cached | 83.90 | 96 B | 17× | median 7.7e-08, worst 4.5e-04 |

## Whole map: 128×128×64 = 1048576 voxels, 32 echoes, MonoExp

| configuration | time (s) | µs/voxel | voxels/s |
|---|---:|---:|---:|
| VoxelFit Float64, 1 thread | 2.12 | 2.024 | 4.94e+05 |
| VoxelFit Float64, 4 threads | 0.68 | 0.651 | 1.54e+06 |
| VoxelFit Float32, 1 thread | 1.67 | 1.593 | 6.28e+05 |
| VoxelFit Float32, 4 threads | 0.41 | 0.387 | 2.58e+06 |
| LsqFit (analytic Jacobian), 4 threads, first 100000 voxels | 2.13 | 21.331 | 4.69e+04 |
