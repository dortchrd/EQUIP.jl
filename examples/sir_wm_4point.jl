# =================================================================================================
# Monte Carlo test dataset: 4-point SIR, white matter, SNR 200
#
# The setup of Dortch et al., Magn Reson Med 2018;80:1824 (the SIR optimization paper):
#   - protocol: SIR_4POINT, the scheme optimized there for the constrained model (kmf fixed at
#     12.5 s⁻¹, QMTSIR's default), Table 2
#   - tissue:   white matter from Table 1 (WM II, the mean: PSR 0.12, kmf 10 s⁻¹, R1f 1 s⁻¹;
#               WM I is the maximum: PSR 0.20, kmf 5 s⁻¹, R1f 1.5 s⁻¹), with Sf = −1, Sm = 0.83
#   - noise:    SNR 200 at thermal equilibrium (σ = M0f / 200, M0f = 1), Rician
#
# The script makes 10 000 noisy copies of that one voxel, fits them twice, prints bias and SD
# against the truth, and saves everything as CSV files (readable in MATLAB, Python, qMRLab):
#   1. kmf fixed at 12.5 s⁻¹, as the 4-point scheme was designed for. The true kmf is 10, so part
#      of the error is the model assumption, not the noise.
#   2. kmf fixed at its true value: what the noise alone does (a reference, not possible in vivo).
#
# Run from the repository folder, with threads:
#   julia -t auto --project=. examples/sir_wm_4point.jl
# or open this file in VS Code and use "Julia: Execute active File in REPL".
# =================================================================================================

using EQUIP, Random, Statistics

# --- Settings: change these to try other cases ---------------------------------------------------
const TISSUE = :SIR_WM_II         # or :SIR_WM_I, :SIR_GM, :SIR_lesion (Dortch 2018, Table 1)
const SNR = 200
const NSIM = 10_000
const SEED = 1                    # the same seed gives the same dataset every time
const OUTDIR = joinpath(@__DIR__, "output", "sir_wm_4point")

# --- The model and the truth ---------------------------------------------------------------------
m = QMTSIR(; SIR_4POINT...)       # TI and TD of the 4-point scheme; everything else default
θ = nominal(m, TISSUE)            # PSR, R1f, Sf, M0f (= 1), kmf and Sm from the preset, and
                                  # QMTSIR's defaults for the rest: R1m = :R1f (tied), Mzf_td0
                                  # and Mzm_td0 = 0 (see `defaults(m)`)
println("Protocol: TI = ", m.TI, " s, TD = ", m.TD, " s")
println("Truth ($TISSUE): ", θ)

# --- The dataset: NSIM × 4 (one row per simulated voxel, one column per TI/TD pair) -------------
Y = simulate(m, θ; snr = SNR, n = NSIM, rng = MersenneTwister(SEED))

# --- Fit 1: as designed, kmf fixed at QMTSIR's default (12.5 s⁻¹) --------------------------------
res = fit(m, Y)
println("\nFit 1: kmf fixed at 12.5 s⁻¹ (the 4-point design)")
display(res)
display(compare(res, θ))

# --- Fit 2: kmf fixed at the true value ----------------------------------------------------------
res_true = fit(m, Y; kmf = θ.kmf)
println("\n\nFit 2: kmf fixed at its true value, $(θ.kmf) s⁻¹ (noise only)")
display(compare(res_true, θ))
println()

# --- Save as CSV ---------------------------------------------------------------------------------
# Plain text, written with Base Julia only (no packages needed). One header line, then values.
function writecsv(path, header, columns)
    open(path, "w") do io
        println(io, join(header, ","))
        for i in eachindex(first(columns))
            println(io, join((c[i] for c in columns), ","))
        end
    end
end

mkpath(OUTDIR)
writecsv(joinpath(OUTDIR, "protocol.csv"), ("TI_s", "TD_s"), (m.TI, m.TD))
untie(x) = x isa Symbol ? θ[x] : x      # a tie (R1m = :R1f): the value it names
writecsv(joinpath(OUTDIR, "truth.csv"), keys(θ), Tuple([untie(v)] for v in values(θ)))
writecsv(joinpath(OUTDIR, "data.csv"), ("y$j" for j in 1:size(Y, 2)), Tuple(eachcol(Y)))
for (name, r) in (("fit_kmf12.5.csv", res), ("fit_kmftrue.csv", res_true))
    names = keys(r.params)
    writecsv(joinpath(OUTDIR, name), (names..., "status"), (values(r.params)..., string.(r.status)))
end
println("Saved protocol.csv, truth.csv, data.csv, fit_kmf12.5.csv and fit_kmftrue.csv in ", OUTDIR)
