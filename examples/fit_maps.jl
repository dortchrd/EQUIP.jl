# =================================================================================================
# Fitting: from data files to maps (the part `equip fit` runs)
#
# A multi-echo T2 dataset, written as a scanner pipeline would leave it (a 4-D NIfTI file and its
# BIDS sidecar with the echo times, as dcm2niix writes them), then read, fitted voxel by voxel on
# threads, checked, and written back as maps in the data's geometry with a sidecar recording how
# each was made.
#
#   1. a phantom: T2 from 40 ms (white matter-like) to 2 s (CSF-like) across the image, S0 1000,
#      32 echoes 10–320 ms, Rician noise at SNR 100 (relative to S0)
#   2. loaddata: the 4-D file and its sidecar (TE from EchoTime)
#   3. fit: T2Decay, a mask
#   4. the fit's own statistics: standard errors, voxels that ended on a bound (`onbound`)
#   5. savemaps: T2map.nii.gz, S0map, status, SSE, the standard errors; each with a JSON sidecar
#
# Run from the repository folder (about 20 s, most of it compiling):
#   julia -t auto --project=. examples/fit_maps.jl
# The maps go to examples/output/ (not committed).
# =================================================================================================

using EQUIP, Random, Statistics, Printf

out = joinpath(@__DIR__, "output", "fit_maps")
mkpath(out)

# --- 1. the phantom, written as files ------------------------------------------------------------
te = collect(0.01:0.01:0.32)                                   # 32 echoes, s
nx, ny, nz = 40, 30, 2
T2 = [exp(range(log(0.04), log(2.0); length = nx)[i]) for i in 1:nx, j in 1:ny, k in 1:nz]
S0 = fill(1000.0, nx, ny, nz)
Y = simulate(T2Decay(TE = te), (S0 = S0, T2 = T2); snr = 100, noise = :rician,
             rng = MersenneTwister(1))                          # nx × ny × nz × 32
datafile = joinpath(out, "sub-01_MESE.nii.gz")
writeimage(datafile, Float32.(Y); meta = Dict("EchoTime" => te))   # and its .json sidecar
writeimage(joinpath(out, "mask.nii.gz"), UInt8.(trues(nx, ny, nz)))

# --- 2. read: the data and the acquisition values beside them --------------------------------------
data, info = loaddata(datafile)                                 # TE from the sidecar's EchoTime
mask = loadmap(joinpath(out, "mask.nii.gz"), info) .> 0        # checked against the data's geometry
println("read ", join(size(data), "×"), "; TE from the sidecar: ", info.found[:TE][1], " … ",
        info.found[:TE][end], " s")

# --- 3. fit ---------------------------------------------------------------------------------------
m = T2Decay(TE = info.found[:TE])
res = fit(m, data; mask)
show(stdout, MIME"text/plain"(), res)                          # voxels, % converged, medians
println()

# --- 4. the fit's own statistics --------------------------------------------------------------------
se = stderror(res).T2                                          # σ from each voxel's residuals
@printf("\nT2 at 40 ms: %.4f s ± %.4f (SE); at 2 s: %.3f s ± %.3f\n",
        median(res.params.T2[1, :, :]), median(se[1, :, :]), median(res.params.T2[end, :, :]),
        median(se[end, :, :]))
nb = count(onbound(res).T2)
println("voxels with T2 on its bound (2.5 s, or no decay): ", nb, " of ", count(mask))

# --- 5. the maps ----------------------------------------------------------------------------------
written = savemaps(joinpath(out, "sub-01_"), res, info; stderror = true)
println("\nwrote ", join(basename.(written), ", "), "\nin ", out)
