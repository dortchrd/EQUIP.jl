| model | data | stopping | µs/voxel | iterations | stalled | stalled, reference off bounds | 99.9th pct |Δ|/SD | largest |Δ|/SD, stalled |
|---|---|---|---:|---:|---:|---:|---:|---:|
| QMTSIR 4-point | SNR 50 | default | 2.15 | 5.3 | 0.0% | 0.00% | 3.3e-07 | – |
| QMTSIR 4-point | SNR 50 | atol | 1.40 | 2.6 | 0.0% | 0.00% | 3.0e-02 | – |
| QMTSIR 4-point | SNR 50 | atol + stall 1% | 1.37 | 2.6 | 0.0% | 0.00% | 3.0e-02 | – |
| QMTSIR 4-point | SNR 200 | default | 1.73 | 5.1 | 0.0% | 0.00% | 1.2e-06 | – |
| QMTSIR 4-point | SNR 200 | atol | 1.38 | 2.5 | 0.0% | 0.00% | 5.1e-02 | – |
| QMTSIR 4-point | SNR 200 | atol + stall 1% | 1.74 | 2.5 | 0.0% | 0.00% | 5.1e-02 | – |
| QMTSIR 4-point | poor mask (Rician) | default | 3.66 | 11.8 | 0.0% | 0.00% | 1.9e-05 | – |
| QMTSIR 4-point | poor mask (Rician) | atol | 3.43 | 10.0 | 0.0% | 0.00% | 6.3e-02 | – |
| QMTSIR 4-point | poor mask (Rician) | atol + stall 1% | 3.24 | 9.3 | 9.9% | 0.00% | 3.4e+00 | 8.0e+00 |
| QMTSIR 5-point, kmf fixed | SNR 50 | default | 2.72 | 7.4 | 0.0% | 0.00% | 7.8e-07 | – |
| QMTSIR 5-point, kmf fixed | SNR 50 | atol | 2.17 | 3.3 | 0.0% | 0.00% | 1.1e-02 | – |
| QMTSIR 5-point, kmf fixed | SNR 50 | atol + stall 1% | 1.79 | 3.3 | 0.1% | 0.00% | 1.1e-02 | 3.3e-04 |
| QMTSIR 5-point, kmf fixed | SNR 200 | default | 2.31 | 6.0 | 0.0% | 0.00% | 1.1e-06 | – |
| QMTSIR 5-point, kmf fixed | SNR 200 | atol | 2.26 | 2.9 | 0.0% | 0.00% | 9.1e-03 | – |
| QMTSIR 5-point, kmf fixed | SNR 200 | atol + stall 1% | 1.65 | 2.9 | 0.0% | 0.00% | 9.1e-03 | – |
| QMTSIR 5-point, kmf free | SNR 50 | default | 5.72 | 14.6 | 0.0% | 0.00% | 8.7e-07 | – |
| QMTSIR 5-point, kmf free | SNR 50 | atol | 5.55 | 11.6 | 0.0% | 0.00% | 1.0e-02 | – |
| QMTSIR 5-point, kmf free | SNR 50 | atol + stall 1% | 5.50 | 11.4 | 7.1% | 0.00% | 1.2e-02 | 9.5e-02 |
| QMTSIR 5-point, kmf free | SNR 200 | default | 3.49 | 8.8 | 0.0% | 0.00% | 4.7e-07 | – |
| QMTSIR 5-point, kmf free | SNR 200 | atol | 3.05 | 5.6 | 0.0% | 0.00% | 1.3e-02 | – |
| QMTSIR 5-point, kmf free | SNR 200 | atol + stall 1% | 3.92 | 5.6 | 0.0% | 0.00% | 1.3e-02 | 7.6e-04 |
| QMTSIR 16-point, kmf free | SNR 50 | default | 12.59 | 15.4 | 0.0% | 0.00% | 3.4e-05 | – |
| QMTSIR 16-point, kmf free | SNR 50 | atol | 9.21 | 9.9 | 0.0% | 0.00% | 2.3e-02 | – |
| QMTSIR 16-point, kmf free | SNR 50 | atol + stall 1% | 9.61 | 9.9 | 0.8% | 0.00% | 5.0e-02 | 8.6e+00 |
| QMTSIR 16-point, kmf free | SNR 200 | default | 8.33 | 10.0 | 0.0% | 0.00% | 4.8e-06 | – |
| QMTSIR 16-point, kmf free | SNR 200 | atol | 6.45 | 6.2 | 0.0% | 0.00% | 9.8e-03 | – |
| QMTSIR 16-point, kmf free | SNR 200 | atol + stall 1% | 6.26 | 6.2 | 0.0% | 0.00% | 9.8e-03 | – |
| T2, 32 echoes | SNR 50 | default | 2.86 | 4.2 | 0.0% | 0.00% | 1.8e-06 | – |
| T2, 32 echoes | SNR 50 | atol | 2.34 | 1.5 | 0.0% | 0.00% | 6.4e-03 | – |
| T2, 32 echoes | SNR 50 | atol + stall 1% | 2.27 | 1.5 | 0.0% | 0.00% | 6.4e-03 | – |
| T2, 32 echoes | SNR 200 | default | 2.40 | 3.3 | 0.0% | 0.00% | 2.9e-06 | – |
| T2, 32 echoes | SNR 200 | atol | 2.00 | 1.2 | 0.0% | 0.00% | 5.9e-03 | – |
| T2, 32 echoes | SNR 200 | atol + stall 1% | 1.97 | 1.2 | 0.0% | 0.00% | 5.9e-03 | – |
| T2, 32 echoes | poor mask (Rician) | default | 3.23 | 4.8 | 0.0% | 0.00% | 1.1e-03 | – |
| T2, 32 echoes | poor mask (Rician) | atol | 2.59 | 2.6 | 0.0% | 0.00% | 1.3e-02 | – |
| T2, 32 echoes | poor mask (Rician) | atol + stall 1% | 2.62 | 2.6 | 0.9% | 0.00% | 1.3e-02 | 0.0e+00 |
| bi-exponential | SNR 50 | default | 26.43 | 32.9 | 0.0% | 0.00% | 2.7e+00 | – |
| bi-exponential | SNR 50 | atol | 25.67 | 28.7 | 0.0% | 0.00% | 2.7e+00 | – |
| bi-exponential | SNR 50 | atol + stall 1% | 25.63 | 28.7 | 0.6% | 0.00% | 2.7e+00 | 9.3e-08 |
| bi-exponential | SNR 200 | default | 19.46 | 23.5 | 0.0% | 0.00% | 4.3e-05 | – |
| bi-exponential | SNR 200 | atol | 17.74 | 19.3 | 0.0% | 0.00% | 6.2e-03 | – |
| bi-exponential | SNR 200 | atol + stall 1% | 18.08 | 19.3 | 0.0% | 0.00% | 6.2e-03 | – |
| kurtosis | SNR 50 | default | 4.85 | 15.3 | 0.0% | 0.00% | 8.6e-07 | – |
| kurtosis | SNR 50 | atol | 4.76 | 12.6 | 0.0% | 0.00% | 8.3e-03 | – |
| kurtosis | SNR 50 | atol + stall 1% | 5.07 | 12.6 | 0.2% | 0.00% | 8.3e-03 | 1.1e-03 |
| kurtosis | SNR 200 | default | 4.87 | 15.2 | 0.0% | 0.00% | 2.4e-06 | – |
| kurtosis | SNR 200 | atol | 5.27 | 12.8 | 0.0% | 0.00% | 8.4e-03 | – |
| kurtosis | SNR 200 | atol + stall 1% | 4.97 | 12.8 | 0.0% | 0.00% | 8.4e-03 | – |
| kurtosis | poor mask (Rician) | default | 5.07 | 16.1 | 0.0% | 0.00% | 4.4e-05 | – |
| kurtosis | poor mask (Rician) | atol | 5.52 | 13.9 | 0.0% | 0.00% | 1.7e-02 | – |
| kurtosis | poor mask (Rician) | atol + stall 1% | 5.28 | 13.6 | 14.2% | 0.00% | 4.1e-01 | 2.8e+00 |
| T2 + offset | SNR 50 | default | 4.70 | 10.2 | 0.0% | 0.00% | 6.2e-07 | – |
| T2 + offset | SNR 50 | atol | 4.18 | 7.5 | 0.0% | 0.00% | 2.9e-03 | – |
| T2 + offset | SNR 50 | atol + stall 1% | 4.13 | 7.5 | 0.0% | 0.00% | 2.9e-03 | – |
| T2 + offset | SNR 200 | default | 4.62 | 10.2 | 0.0% | 0.00% | 8.4e-07 | – |
| T2 + offset | SNR 200 | atol | 4.28 | 7.8 | 0.0% | 0.00% | 3.1e-03 | – |
| T2 + offset | SNR 200 | atol + stall 1% | 4.37 | 7.8 | 0.0% | 0.00% | 3.1e-03 | – |
| T1IR | SNR 50 | default | 3.07 | 5.5 | 0.0% | 0.00% | 5.6e-07 | – |
| T1IR | SNR 50 | atol | 4.13 | 2.7 | 0.0% | 0.00% | 2.7e-03 | – |
| T1IR | SNR 50 | atol + stall 1% | 4.05 | 2.7 | 0.0% | 0.00% | 2.7e-03 | 1.1e-09 |
| T1IR | SNR 200 | default | 2.92 | 4.9 | 0.0% | 0.00% | 1.4e-06 | – |
| T1IR | SNR 200 | atol | 2.68 | 2.5 | 0.0% | 0.00% | 2.9e-03 | – |
| T1IR | SNR 200 | atol + stall 1% | 2.66 | 2.5 | 0.0% | 0.00% | 2.9e-03 | – |
| T1IR | poor mask (Rician) | default | 4.69 | 11.5 | 0.0% | 0.00% | 5.7e-03 | – |
| T1IR | poor mask (Rician) | atol | 4.35 | 9.3 | 0.0% | 0.00% | 1.0e-02 | – |
| T1IR | poor mask (Rician) | atol + stall 1% | 3.39 | 5.5 | 24.7% | 0.00% | 7.2e-02 | 4.0e-01 |
