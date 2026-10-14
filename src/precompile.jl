# Precompile the common paths (all built-in models, both precisions, both data layouts, the
# CLI) into the package image so that the first fit in a fresh process does not wait for JIT.

@setup_workload begin
    cases = (
        (MonoExp(), [10.0, 20.0, 40.0, 80.0], (1000.0, 50.0)),
        (MonoExpOffset(), [10.0, 20.0, 40.0, 80.0, 160.0], (1000.0, 50.0, 20.0)),
        (BiExp(), [5.0, 10.0, 20.0, 40.0, 80.0, 160.0], (300.0, 10.0, 700.0, 80.0)),
        (ADC(), [0.0, 500.0, 1000.0], (1000.0, 1e-3)),
        (Kurtosis(), [0.0, 500.0, 1000.0, 2000.0], (1000.0, 1e-3, 1.0)),
        (IVIM(), [0.0, 20.0, 50.0, 100.0, 200.0, 500.0, 800.0], (1000.0, 0.1, 0.02, 1e-3)),
        (InversionRecovery(), [50.0, 200.0, 800.0, 2000.0], (1000.0, 2.0, 900.0)),
        (SaturationRecovery(), [100.0, 500.0, 2000.0], (1000.0, 1000.0)),
        (SPGR(TR=15.0), [3.0, 10.0, 20.0], (5000.0, 1000.0)),
    )
    @compile_workload begin
        for (m, x, θ) in cases
            P = nparams(m)
            curve = [m(xi, SVector{P}(θ)) for xi in x]
            vm = Float32.(repeat(reshape(curve, 1, 1, :), 2, 1, 1))   # 2×1×nobs, voxel-major
            om = permutedims(vm, (3, 1, 2))                             # nobs×2×1, obs-major
            for T in (Float32, Float64)
                fit_maps(m, x, vm; precision=T, ntasks=1)
                fit_maps(m, x, om; obsdim=1, precision=T, ntasks=1)
                fit_maps(m, x, Float64.(om); obsdim=1, precision=T, ntasks=1)
            end
            fit_voxel(m, x, curve)
        end
        # 4-d (X×Y×Z×nobs) data with a mask: the usual NIfTI/MATLAB case.
        A4 = Float32[1000 * exp(-t / 50) for _ in 1:2, _ in 1:2, _ in 1:2, t in (10, 20, 40, 80)]
        for T in (Float32, Float64)
            fit_maps(MonoExp(), [10, 20, 40, 80], A4; precision=T, mask=trues(2, 2, 2), ntasks=1)
        end
        fit_maps(SPGR(TR=15.0), [3.0, 10.0, 20.0], fill(100.0f0, 2, 3); covariates=ones(Float32, 2),
                 precision=Float32, ntasks=1)
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
