# =================================================================================================
# Rician floor corrections for SMT, compared on the simulated spherical-mean pipeline
#
# One fiber population in a random orientation per voxel; a Rician magnitude image per gradient
# direction; the spherical mean of each shell; the SMT fit (Kaden et al., NeuroImage 2016). The
# floor handled seven ways: none; the Karlsen inversion (Magn Reson Med 1999;41:614, Eq. 4) of each
# shell mean or of each direction (Kaden's `--rician`); NYU's sqrt(max(M² − σ², 0)) (SMI.m) and
# sqrt(|M² − σ²|) (DESIGNER) of each direction; E|y| of the shell mean (`RicianMean`) or of each
# direction (`RicianSphericalMean`) as the fitted model. σ known. Bias, SD and RMSE of v_ax at SNR
# 20 and 10 per image (docs/design.md, "Peripheral nerve SMT", step 2). About 45 s:
#   julia -t auto --project=. bench/floor_corrections.jl
# =================================================================================================

using EQUIP, Random, Statistics, LinearAlgebra, StaticArrays

# Karlsen MRM 1999 Eq. 4 inverted, as corrRicianBias.m / Kaden's ricedebias: the A with E|y|(A) = M;
# 0 below the floor σ√(π/2). E is increasing in A and E(A) ≥ A, so A is in [0, M]: bisection.
function karlsen(M, σ)
    M <= σ * sqrt(π / 2) && return 0.0
    lo, hi = 0.0, M
    for _ in 1:60
        mid = (lo + hi) / 2
        ricianmean(mid, σ) < M ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

function run(snr; nvox = 2000, seed = 1)
    rng = MersenneTwister(seed)
    m = SMT(b = [0.0, 1000, 2500, 4000])
    θ = (S0 = 1.0, vax = 0.6, Dax = 2.0e-3)
    p = EQUIP.resolve(m, θ)
    d = [4, 30, 30, 30]
    σ = 1 / snr
    sets = directionset.(d)
    ns = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:nvox]
    # each direction's magnitude image, per voxel and shell
    mags = [[abs(EQUIP._profile(m, p, j, dot(n, g)) + σ * (randn(rng) + im * randn(rng))) for g in sets[j]]
            for n in ns, j in 1:4]
    raw = mean.(mags)                                   # the spherical means of magnitudes
    out = Pair{String,Any}[]
    push!(out, "least squares, no correction" => fit(m, raw; weights = d))
    push!(out, "Karlsen on each shell mean, then LS" => fit(m, karlsen.(raw, σ); weights = d))
    push!(out, "Karlsen on each direction (Kaden), mean, LS" => fit(m, [mean(karlsen.(v, σ)) for v in mags]; weights = d))
    # NYU: each direction's image corrected before the spherical harmonics / spherical mean.
    # SMI.m (preferred, "works well above SNR ~ 1.5"): sqrt(max(M² − σ², 0)); DESIGNER v1/v2
    # and SMI's alternative ("above SNR ~ 2"): sqrt(|M² − σ²|).
    push!(out, "NYU SMI.m: sqrt(max(M²-σ², 0)) each direction" => fit(m, [mean(sqrt.(max.(v .^ 2 .- σ^2, 0))) for v in mags]; weights = d))
    push!(out, "NYU DESIGNER: sqrt(|M²-σ²|) each direction" => fit(m, [mean(sqrt.(abs.(v .^ 2 .- σ^2))) for v in mags]; weights = d))
    push!(out, "E|y| of the shell mean as model (RicianMean)" => fit(RicianMean(m; σ), raw; weights = d))
    push!(out, "E|y| per direction as model (RicianSphericalMean)" => fit(RicianSphericalMean(m; σ), raw; weights = d))
    println("SNR $snr per image (b = 0), v_ax 0.6, D_ax 2.0e-3, directions $d, $nvox voxels")
    println(rpad("", 52), "  bias v_ax    SD     RMSE   | bias D_ax (1e-3)")
    for (name, r) in out
        v = r.params.vax; D = r.params.Dax
        b = mean(v) - θ.vax; s = std(v)
        println(rpad(name, 52), "  ", lpad(round(b; digits = 3), 7), "  ", lpad(round(s; digits = 3), 6), "  ",
                lpad(round(sqrt(b^2 + s^2); digits = 3), 6), "   | ", lpad(round((mean(D) - θ.Dax) * 1e3; digits = 3), 7))
    end
    c = crlb(m, θ; sphericalmeannoise(m, θ; snr, directions = d, sets)..., weights = d)
    println(rpad("(crlb predicts for no correction)", 52), "  ", lpad(round(c.bias.vax; digits = 3), 7), "  ",
            lpad(round(c.sd.vax; digits = 3), 6))
    println()
end
t = @elapsed (run(20); run(10))
println("$(round(t; digits = 1)) s")
