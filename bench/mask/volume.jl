# The end-to-end check (docs/design.md, Performance notes): julia --project=cli bench/mask/volume.jl DIR, then time
# julia -t auto --project=cli -m EQUIPCLI fit QMTSIR DIR/SIR.nii --mask DIR/MASK_poor.nii (or MASK_tight.nii)
#     --TI 1.007,0.010,0.278,0.010 --TD 0.010,4.174,2.730,0.684 --kmf 12.5 --out DIR/poor_
# A 128×128×30 SIR volume (the owner's 4-point protocol), half of it background noise, a tenth
# CSF, the rest white and grey matter; magnitude data, σ = 10; a mask of every voxel (poor) and
# one of the head only (tight); written as NIfTI for `equip fit`
using EQUIP, MRImages, Random
TI = [1.007, 0.010, 0.278, 0.010]; TD = [0.010, 4.174, 2.730, 0.684]
m = QMTSIR(TI = TI, TD = TD)
dims = (128, 128, 30); n = prod(dims)
rng = MersenneTwister(2)
cls = rand(rng, n)                                    # < 0.5 background, < 0.6 CSF, else tissue
PSR = ifelse.(cls .< 0.6, 1e-4, 0.05 .+ 0.15 .* rand(rng, n))
R1f = ifelse.(cls .< 0.6, 0.25, 0.6 .+ 0.9 .* rand(rng, n))
M0f = ifelse.(cls .< 0.5, 0.0, 1000.0)
θ = (PSR = PSR, R1f = R1f, Sf = fill(-0.95, n), M0f = M0f, kmf = fill(12.5, n), R1m = R1f, Sm = fill(0.83, n),
     Mzf_td0 = zeros(n), Mzm_td0 = zeros(n))
Y = simulate(m, θ; σ = 10.0, noise = :rician, rng)
dir = ARGS[1]
writeimage(joinpath(dir, "SIR.nii"), MRImage(Float32.(reshape(Y, dims..., 4))))
writeimage(joinpath(dir, "MASK_poor.nii"), MRImage(UInt8.(ones(dims))))
writeimage(joinpath(dir, "MASK_tight.nii"), MRImage(UInt8.(reshape(cls .>= 0.5, dims))))
println("written: ", n, " voxels, ", count(cls .>= 0.5), " in the tight mask")
