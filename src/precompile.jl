# Precompile the common paths (all built-in models, both precisions, both data layouts, the
# cache API and the CLI) into the package image, so a fresh process fits without waiting for JIT.

@setup_workload begin
    cases = (
        (MonoExp(), (TE = [10.0, 20.0, 40.0, 80.0],), (S0 = 1000.0, T2 = 50.0)),
        (MonoExpOffset(), (TE = [10.0, 20.0, 40.0, 80.0, 160.0],), (S0 = 1000.0, T2 = 50.0, C = 20.0)),
        (BiExp(), (TE = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0],), (A1 = 300.0, T2a = 10.0, A2 = 700.0, T2b = 80.0)),
        (ADC(), (b = [0.0, 500.0, 1000.0],), (S0 = 1000.0, D = 1e-3)),
        (Kurtosis(), (b = [0.0, 500.0, 1000.0, 2000.0],), (S0 = 1000.0, D = 1e-3, K = 1.0)),
        (IVIM(), (b = [0.0, 20.0, 50.0, 100.0, 200.0, 500.0, 800.0],), (S0 = 1000.0, f = 0.1, Dp = 0.02, D = 1e-3)),
        (InversionRecovery(), (TI = [50.0, 200.0, 800.0, 2000.0],), (S0 = 1000.0, B = 2.0, T1 = 900.0)),
        (SaturationRecovery(), (t = [100.0, 500.0, 2000.0],), (S0 = 1000.0, T1 = 1000.0)),
        (SPGR(), (FA = [3.0, 10.0, 20.0], TR = 15.0), (M0 = 5000.0, T1 = 1000.0, B1 = 1.0)),
        (SIRFSE(), (TI = [0.01, 0.05, 0.3, 1.0], TD = 2.5),
         (pmf = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, R1m = 1.0, Sm = 0.83)),
    )
    @compile_workload begin
        for (m, acq, θ) in cases
            curve = signal(m, acq, θ)
            vm = Float32.(repeat(reshape(curve, 1, 1, :), 2, 1, 1))   # 2×1×nobs, voxel-major
            om = permutedims(vm, (3, 1, 2))                             # nobs×2×1, obs-major
            for T in (Float32, Float64)
                fit_maps(m, acq, vm; precision=T, ntasks=1)
                fit_maps(m, acq, om; obsdim=1, precision=T, ntasks=1)
                fit_maps(m, acq, Float64.(om); obsdim=1, precision=T, ntasks=1)
            end
            fit!(init(m, acq), curve)
        end
        # 4-d (X×Y×Z×nobs) data with a mask: the usual NIfTI/MATLAB case.
        A4 = Float32[1000 * exp(-t / 50) for _ in 1:2, _ in 1:2, _ in 1:2, t in (10, 20, 40, 80)]
        for T in (Float32, Float64)
            fit_maps(MonoExp(), [10, 20, 40, 80], A4; precision=T, mask=trues(2, 2, 2), ntasks=1)
        end
        # Per-voxel map and freed parameter
        fit_maps(SPGR(), (FA = [3.0, 10.0, 20.0], TR = 15.0), fill(100.0f0, 2, 3);
                 fixed = (B1 = ones(Float32, 2),), precision=Float32, ntasks=1)
        sir = (TI = [0.01, 0.05, 0.3, 1.0, 2.0], TD = 2.5)
        fit!(init(SIRFSE(), sir; free = (:kmf,)),
             signal(SIRFSE(), sir, (pmf = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5,
                                    R1m = 1.0, Sm = 0.83)))
        mktempdir() do dir
            x = [10.0, 20.0, 40.0, 80.0]
            A = Float32[1000 * exp(-t / 50) for _ in 1:2, _ in 1:2, t in x]
            write_npy(joinpath(dir, "d.npy"), permutedims(A, (3, 2, 1)); reversed=true)
            redirect_stdout(devnull) do
                main(["fit", "--model", "MonoExp", "--x", "10,20,40,80", "--data", joinpath(dir, "d.npy"),
                      "--out", joinpath(dir, "o")])
                NIfTI.niwrite(joinpath(dir, "d.nii.gz"), NIVolume(A))
                main(["fit", "--model", "MonoExp", "--x", "10,20,40,80", "--data",
                      joinpath(dir, "d.nii.gz"), "--out", joinpath(dir, "o")])
            end
        end
    end
end
