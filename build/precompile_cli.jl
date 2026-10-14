# Exercised while building the system image: typical CLI runs on small synthetic data.
using VoxelFit
mktempdir() do dir
    x = [10.0, 20.0, 40.0, 80.0]
    A = Float32[1000 * exp(-t / 50) for _ in 1:4, _ in 1:4, _ in 1:2, t in x]
    VoxelFit.write_npy(joinpath(dir, "c.npy"), permutedims(A, (4, 3, 2, 1)); reversed=true)
    VoxelFit.write_npy(joinpath(dir, "f.npy"), A)
    VoxelFit.NIfTI.niwrite(joinpath(dir, "d.nii.gz"), VoxelFit.NIVolume(A))
    for data in ("c.npy", "f.npy", "d.nii.gz"), prec in ("single", "double")
        VoxelFit.main(["fit", "--model", "MonoExp", "--x", "10,20,40,80", "--data",
                       joinpath(dir, data), "--out", joinpath(dir, "o"), "--precision", prec,
                       "--stderr", "--quiet"])
    end
end
