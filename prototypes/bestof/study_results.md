| model | SNR | start | µs/voxel | start µs | iterations | converged | not best | median error |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| T2 | 50 | default (:loglinear) | 2.46 | 1.45 | 4.2 | 100.0% | 0 | 1.64% |
| T2 | 50 | BestOf 15, per voxel | 3.74 | 2.57 | 4.8 | 100.0% | 0 | 1.64% |
| T2 | 50 | BestOf 15, dictionary | 1.66 | 0.55 | 4.8 | 100.0% | 0 | 1.64% |
| T2 | 200 | default (:loglinear) | 1.68 | 1.12 | 3.3 | 100.0% | 0 | 0.41% |
| T2 | 200 | BestOf 15, per voxel | 2.89 | 2.18 | 4.5 | 100.0% | 0 | 0.41% |
| T2 | 200 | BestOf 15, dictionary | 1.32 | 0.47 | 4.5 | 100.0% | 0 | 0.41% |
| T2offset | 50 | default (fixed) | 3.99 | 0.53 | 10.1 | 100.0% | 0 | 3.38% |
| T2offset | 50 | default start, C solved (BestOf 1 value) | 3.84 | 0.42 | 9.7 | 100.0% | 0 | 3.38% |
| T2offset | 50 | BestOf T2, per voxel (C at 0) | 5.51 | 2.94 | 7.2 | 100.0% | 0 | 3.38% |
| T2offset | 50 | BestOf T2, dictionary (C at 0) | 3.02 | 0.63 | 7.2 | 100.0% | 0 | 3.38% |
| T2offset | 50 | BestOf T2, dictionary, C solved | 2.48 | 0.83 | 5.1 | 100.0% | 0 | 3.38% |
| T2offset | 200 | default (fixed) | 4.19 | 0.54 | 10.1 | 100.0% | 0 | 0.85% |
| T2offset | 200 | default start, C solved (BestOf 1 value) | 2.99 | 0.42 | 9.7 | 100.0% | 0 | 0.85% |
| T2offset | 200 | BestOf T2, per voxel (C at 0) | 4.25 | 2.23 | 7.1 | 100.0% | 0 | 0.85% |
| T2offset | 200 | BestOf T2, dictionary (C at 0) | 2.54 | 0.67 | 7.1 | 100.0% | 0 | 0.85% |
| T2offset | 200 | BestOf T2, dictionary, C solved | 2.03 | 0.70 | 4.9 | 100.0% | 0 | 0.85% |
| biexp | 50 | default (fixed) | 22.72 | 0.93 | 33.2 | 92.5% | 1026 | 4.24% |
| biexp | 50 | default starts, fshort solved (BestOf 1 value) | 25.16 | 0.69 | 36.5 | 93.1% | 973 | 4.32% |
| biexp | 50 | BestOf T2long 8, per voxel | 23.14 | 3.06 | 29.3 | 91.9% | 1052 | 4.15% |
| biexp | 50 | BestOf T2long 8, dictionary, fshort solved | 19.94 | 0.78 | 28.0 | 92.1% | 952 | 4.15% |
| biexp | 50 | T2short × T2long 8×8, per voxel | 37.02 | 16.80 | 28.6 | 89.6% | 1302 | 3.99% |
| biexp | 50 | T2short × T2long, dictionary (fshort at 0.2) | 17.17 | 1.49 | 28.6 | 89.6% | 1302 | 3.99% |
| biexp | 50 | T2short × T2long, dictionary, fshort solved | 19.15 | 2.46 | 25.8 | 91.6% | 1015 | 4.14% |
| biexp | 200 | default (fixed) | 15.75 | 0.86 | 23.1 | 99.0% | 149 | 0.99% |
| biexp | 200 | default starts, fshort solved (BestOf 1 value) | 20.17 | 0.70 | 30.1 | 99.0% | 112 | 0.98% |
| biexp | 200 | BestOf T2long 8, per voxel | 13.06 | 2.83 | 16.0 | 99.0% | 103 | 0.98% |
| biexp | 200 | BestOf T2long 8, dictionary, fshort solved | 11.78 | 0.81 | 16.5 | 99.0% | 106 | 0.98% |
| biexp | 200 | T2short × T2long 8×8, per voxel | 27.04 | 17.44 | 13.9 | 98.8% | 123 | 0.98% |
| biexp | 200 | T2short × T2long, dictionary (fshort at 0.2) | 11.24 | 1.70 | 13.9 | 98.8% | 123 | 0.98% |
| biexp | 200 | T2short × T2long, dictionary, fshort solved | 9.52 | 2.33 | 14.0 | 99.0% | 108 | 0.98% |
| ADC | 50 | default (fixed) | 0.73 | 0.15 | 5.8 | 100.0% | 0 | 3.44% |
| ADC | 50 | BestOf 12, per voxel | 0.99 | 0.55 | 4.3 | 100.0% | 0 | 3.44% |
| ADC | 50 | BestOf 12, dictionary | 0.60 | 0.18 | 4.3 | 100.0% | 0 | 3.44% |
| ADC | 200 | default (fixed) | 0.69 | 0.16 | 5.5 | 100.0% | 0 | 0.85% |
| ADC | 200 | BestOf 12, per voxel | 0.90 | 0.52 | 4.1 | 100.0% | 0 | 0.85% |
| ADC | 200 | BestOf 12, dictionary | 0.56 | 0.19 | 4.1 | 100.0% | 0 | 0.85% |
| kurtosis | 50 | default (fixed) | 3.97 | 0.34 | 15.3 | 96.5% | 351 | 3.07% |
| kurtosis | 50 | BestOf ADC 15, dictionary | 3.28 | 0.37 | 11.8 | 97.7% | 358 | 3.06% |
| kurtosis | 50 | ADC × K 15×7, per voxel | 10.90 | 9.66 | 10.1 | 98.2% | 176 | 2.90% |
| kurtosis | 50 | ADC × K, dictionary | 3.43 | 1.08 | 10.1 | 98.2% | 176 | 2.90% |
| kurtosis | 200 | default (fixed) | 3.38 | 0.27 | 15.1 | 96.5% | 353 | 0.77% |
| kurtosis | 200 | BestOf ADC 15, dictionary | 2.58 | 0.37 | 11.6 | 97.7% | 357 | 0.77% |
| kurtosis | 200 | ADC × K 15×7, per voxel | 11.49 | 8.27 | 10.0 | 98.2% | 178 | 0.73% |
| kurtosis | 200 | ADC × K, dictionary | 3.43 | 1.11 | 10.0 | 98.2% | 178 | 0.73% |
| IR | 50 | default: BestOf 30, per voxel (main) | 2.67 | 1.28 | 5.4 | 100.0% | 139 | 2.48% |
| IR | 50 | BestOf 30, dictionary | 1.12 | 0.31 | 5.4 | 100.0% | 139 | 2.48% |
| IR | 50 | BestOf 19, dictionary | 1.15 | 0.30 | 5.5 | 100.0% | 92 | 2.47% |
| IR | 200 | default: BestOf 30, per voxel (main) | 2.08 | 1.33 | 4.9 | 100.0% | 31 | 0.61% |
| IR | 200 | BestOf 30, dictionary | 1.04 | 0.35 | 4.9 | 100.0% | 31 | 0.61% |
| IR | 200 | BestOf 19, dictionary | 1.20 | 0.32 | 5.1 | 100.0% | 18 | 0.61% |
| IR signed | 50 | default: BestOf 30, per voxel (main) | 2.47 | 1.54 | 5.1 | 100.0% | 0 | 2.40% |
| IR signed | 50 | BestOf 30, dictionary (Sf at -0.95) | 1.22 | 0.35 | 5.1 | 100.0% | 0 | 2.40% |
| IR signed | 50 | BestOf 30, dictionary, Sf solved | 1.21 | 0.44 | 5.0 | 100.0% | 0 | 2.40% |
| IR signed | 50 | BestOf 19, dictionary, Sf solved | 1.14 | 0.45 | 5.2 | 100.0% | 0 | 2.40% |
| IR signed | 200 | default: BestOf 30, per voxel (main) | 1.93 | 1.22 | 4.8 | 100.0% | 0 | 0.62% |
| IR signed | 200 | BestOf 30, dictionary (Sf at -0.95) | 0.95 | 0.29 | 4.8 | 100.0% | 0 | 0.62% |
| IR signed | 200 | BestOf 30, dictionary, Sf solved | 1.22 | 0.49 | 4.8 | 100.0% | 0 | 0.62% |
| IR signed | 200 | BestOf 19, dictionary, Sf solved | 1.14 | 0.43 | 4.9 | 100.0% | 0 | 0.62% |
| SR | 50 | default (fixed) | 0.78 | 0.18 | 6.9 | 100.0% | 0 | 3.57% |
| SR | 50 | BestOf 17, per voxel | 1.47 | 0.89 | 5.1 | 100.0% | 0 | 3.57% |
| SR | 50 | BestOf 17, dictionary | 0.80 | 0.25 | 5.1 | 100.0% | 0 | 3.57% |
| SR | 200 | default (fixed) | 0.71 | 0.17 | 6.4 | 100.0% | 0 | 0.87% |
| SR | 200 | BestOf 17, per voxel | 1.36 | 0.89 | 4.6 | 100.0% | 0 | 0.87% |
| SR | 200 | BestOf 17, dictionary | 0.71 | 0.25 | 4.6 | 100.0% | 0 | 0.87% |
| SPGR | 50 | default (fixed) | 1.88 | 0.26 | 10.2 | 100.0% | 5 | 33.10% |
| SPGR | 50 | BestOf 15, per voxel | 2.51 | 1.26 | 8.6 | 100.0% | 0 | 33.11% |
| SPGR | 50 | BestOf 15, dictionary | 1.31 | 0.26 | 8.6 | 100.0% | 0 | 33.11% |
| SPGR | 200 | default (fixed) | 1.35 | 0.26 | 7.3 | 100.0% | 0 | 8.53% |
| SPGR | 200 | BestOf 15, per voxel | 1.73 | 1.24 | 5.5 | 100.0% | 0 | 8.53% |
| SPGR | 200 | BestOf 15, dictionary | 1.15 | 0.27 | 5.5 | 100.0% | 0 | 8.53% |
| QMTSIR 4-point | 50 | default (fixed) | 1.94 | 0.34 | 5.5 | 100.0% | 28 | 17.80% |
| QMTSIR 4-point | 50 | BestOf R1f 8, per voxel | 2.74 | 1.16 | 5.2 | 100.0% | 33 | 18.52% |
| QMTSIR 4-point | 50 | BestOf R1f 8, dictionary | 1.78 | 0.33 | 5.2 | 100.0% | 33 | 18.52% |
| QMTSIR 4-point | 50 | PSR × R1f 6×8, per voxel | 7.06 | 5.88 | 5.1 | 100.0% | 29 | 18.12% |
| QMTSIR 4-point | 50 | PSR × R1f 6×8, dictionary | 1.97 | 0.57 | 5.1 | 100.0% | 29 | 18.12% |
| QMTSIR 4-point | 200 | default (fixed) | 1.90 | 0.39 | 5.1 | 100.0% | 0 | 4.51% |
| QMTSIR 4-point | 200 | BestOf R1f 8, per voxel | 2.17 | 1.17 | 4.9 | 100.0% | 0 | 4.71% |
| QMTSIR 4-point | 200 | BestOf R1f 8, dictionary | 1.70 | 0.35 | 4.9 | 100.0% | 0 | 4.71% |
| QMTSIR 4-point | 200 | PSR × R1f 6×8, per voxel | 6.96 | 5.63 | 4.8 | 100.0% | 0 | 4.57% |
| QMTSIR 4-point | 200 | PSR × R1f 6×8, dictionary | 1.95 | 0.53 | 4.8 | 100.0% | 0 | 4.57% |
