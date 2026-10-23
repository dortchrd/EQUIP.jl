# =================================================================================================
# Monte Carlo test dataset: 4-point SIR over a range of tissues, SNR 200
#
# Like sir_wm_4point.jl, but every voxel has its own tissue values, so the fits do not all start
# near the answer. The values come from the range of the four SIR tissue classes of Dortch et
# al., Magn Reson Med 2018;80:1824, Table 1 (`TISSUES`):
#
#   class         PSR    kmf (s⁻¹)   R1f (s⁻¹)
#   SIR_WM_I      0.20    5           1.5        maximum white matter
#   SIR_WM_II     0.12   10           1.0        mean white matter
#   SIR_GM        0.08   15           0.7        gray matter
#   SIR_lesion    0.05   20           0.5        RRMS lesion
#
# Two ways to draw a voxel's PSR, kmf and R1f (setting SAMPLING):
#   :ranges  each one independently, uniform between the smallest and largest of the four classes
#            (PSR 0.05-0.20, kmf 5-20 s⁻¹, R1f 0.5-1.5 s⁻¹): every combination, including ones
#            no class has (e.g. high PSR with slow R1f), the harder test
#   :mix     a random weighted average of the four classes (weights spread evenly over every
#            combination that sums to 1): stays near the published pattern (more macromolecules
#            go with faster R1f and slower exchange)
# And for every voxel, independently:
#   - Sf (free-pool inversion efficiency): uniform between -1.0 and -0.8
#   - M0f: uniform between 0.8 and 1.2
#   - Sm = 0.83 and R1m = R1f, as in Table 1 (QMTSIR's defaults), and no Mz left over at the
#     start of TD (Mzf_td0 = Mzm_td0 = 0)
#   - noise: SNR 200 at thermal equilibrium, voxel by voxel (σ = M0f / 200), Rician
#
# Fitted twice, as in sir_wm_4point.jl: kmf fixed at 12.5 s⁻¹ (what the 4-point scheme was
# designed for; the true kmf now varies from 5 to 20 s⁻¹), and kmf fixed at each voxel's true
# value. Everything is saved as CSV files, for MATLAB, Python or qMRLab. bench/solvers.jl uses the
# same data (its case :sir4_range, made from the same seed).
#
# Run from the repository folder, with threads:
#   julia -t auto --project=. examples/sir_4point_range.jl
# =================================================================================================

using EQUIP, Random, Statistics

# --- Settings: change these to try other cases ---------------------------------------------------
const CLASSES = (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)   # the published tissue classes
const SAMPLING = :ranges          # :ranges (independent, uniform) or :mix (weighted averages)
const SF_RANGE = (-1.0, -0.8)     # inversion efficiency, uniform in this range
const M0F_RANGE = (0.8, 1.2)      # scale, uniform in this range
const SNR = 200
const NSIM = 10_000
const SEED = 1                    # the same seed gives the same dataset every time
const OUTDIR = joinpath(@__DIR__, "output", "sir_4point_range")

m = QMTSIR(; SIR_4POINT...)       # TI and TD of the 4-point scheme; everything else default
rng = MersenneTwister(SEED)

# --- The truth: one set of tissue values per voxel -----------------------------------------------
uniform(rng, (lo, hi), n) = lo .+ (hi - lo) .* rand(rng, n)     # n values, uniform in [lo, hi)
classvalues(name) = [tissue(c)[name] for c in CLASSES]          # e.g. the four classes' PSRs

# Weights spread evenly over all combinations that sum to 1 (a flat Dirichlet distribution):
# exponential random numbers, divided by their sum. 1 - rand avoids log(0).
function mixweights(rng, k)
    w = -log.(1 .- rand(rng, k))
    return w ./ sum(w)
end

# PSR, R1f and kmf for every voxel, by the chosen SAMPLING
function tissuevalues(rng, sampling, n)
    names = (:PSR, :R1f, :kmf)
    if sampling === :ranges
        return NamedTuple{names}(map(k -> uniform(rng, extrema(classvalues(k)), n), names))
    elseif sampling === :mix
        W = reduce(vcat, [mixweights(rng, length(CLASSES))' for _ in 1:n])   # n × 4, rows sum to 1
        return NamedTuple{names}(map(k -> W * classvalues(k), names))
    end
    throw(ArgumentError("SAMPLING must be :ranges or :mix"))
end

t = tissuevalues(rng, SAMPLING, NSIM)
θ = (PSR = t.PSR,
     R1f = t.R1f,
     Sf = uniform(rng, SF_RANGE, NSIM),
     M0f = uniform(rng, M0F_RANGE, NSIM),
     kmf = t.kmf,
     R1m = :R1f,                  # tied: R1m = R1f in every voxel, as in Table 1
     Sm = 0.83,                   # the same in every voxel, as in Table 1
     Mzf_td0 = 0.0,               # Mz at the start of TD (0: fully saturated by the readout)
     Mzm_td0 = 0.0)

println("Protocol: TI = ", m.TI, " s, TD = ", m.TD, " s")
println("Truth over ", NSIM, " voxels, sampling ", SAMPLING, " (min, median, max):")
for n in (:PSR, :R1f, :Sf, :M0f, :kmf)
    v = θ[n]
    println("  ", rpad(n, 4), round.((minimum(v), median(v), maximum(v)); sigdigits = 3))
end

# --- The dataset: NSIM × 4 (one row per voxel, one column per TI/TD pair) ------------------------
Y = simulate(m, θ; snr = SNR, rng)

# --- Fit 1: as designed, kmf fixed at QMTSIR's default (12.5 s⁻¹) --------------------------------
res = fit(m, Y)
println("\nFit 1: kmf fixed at 12.5 s⁻¹ (the 4-point design; the true kmf varies)")
display(res)
display(compare(res, θ))

# --- Fit 2: kmf fixed at each voxel's true value -------------------------------------------------
res_true = fit(m, Y; kmf = θ.kmf)
println("\n\nFit 2: kmf fixed at each voxel's true value (noise only)")
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

perVoxel(x) = x isa AbstractArray ? x : fill(x, NSIM)   # a number becomes a column of NSIM values
untie(x) = x isa Symbol ? θ[x] : x                      # a tie (R1m = :R1f): the values it names
mkpath(OUTDIR)
writecsv(joinpath(OUTDIR, "protocol.csv"), ("TI_s", "TD_s"), (m.TI, m.TD))
writecsv(joinpath(OUTDIR, "truth.csv"), keys(θ), map(x -> perVoxel(untie(x)), Tuple(values(θ))))
writecsv(joinpath(OUTDIR, "data.csv"), ("y$j" for j in 1:size(Y, 2)), Tuple(eachcol(Y)))
for (name, r) in (("fit_kmf12.5.csv", res), ("fit_kmftrue.csv", res_true))
    names = keys(r.params)
    writecsv(joinpath(OUTDIR, name), (names..., "status"), (values(r.params)..., string.(r.status)))
end
println("Saved protocol.csv, truth.csv (one row per voxel), data.csv, fit_kmf12.5.csv and ",
        "fit_kmftrue.csv in ", OUTDIR)
