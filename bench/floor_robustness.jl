# =================================================================================================
# Rician floor corrections for SMT: how they hold up when σ is wrong and the fibers are dispersed
#
# The pipeline of bench/floor_corrections.jl; each correction given f × the true σ (f = 0.8 to 1.2,
# the same data), and data from fibers dispersed about the voxel's orientation (K = 100 fibers from
# a Watson distribution, κ = 16, 4, 1: ODI = (2/π)·atan(1/κ) = 0.04, 0.16, 0.50) with σ right.
# Bias and RMSE of v_ax at SNR 20 and 10 per image (docs/design.md, "Peripheral nerve SMT", step 2).
# About 4 minutes:
#   julia -t auto --project=. bench/floor_robustness.jl
# =================================================================================================

using EQUIP, Random, Statistics, LinearAlgebra, StaticArrays, Printf

# Karlsen MRM 1999 Eq. 4 inverted (corrRicianBias.m, Kaden's ricedebias): 0 below the floor.
function karlsen(M, σ)
    M <= σ * sqrt(π / 2) && return 0.0
    lo, hi = 0.0, M
    for _ in 1:60
        mid = (lo + hi) / 2
        ricianmean(mid, σ) < M ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# A Watson-distributed unit vector about μ (density ∝ exp(κ(μ·x)²)): ξ = μ·x by rejection from
# the uniform on [0, 1], a random sign, a uniform azimuth.
function watson(rng, μ, κ)
    ξ = 0.0
    while true
        ξ = rand(rng)
        rand(rng) <= exp(κ * (ξ^2 - 1)) && break
    end
    ξ *= rand(rng, Bool) ? 1 : -1
    a = abs(μ[1]) < 0.9 ? SVector(1.0, 0, 0) : SVector(0.0, 1, 0)
    e1 = normalize(a - dot(a, μ) * μ); e2 = cross(μ, e1)
    φ = 2π * rand(rng)
    return ξ * μ + sqrt(1 - ξ^2) * (cos(φ) * e1 + sin(φ) * e2)
end
odi(κ) = 2 / π * atan(1 / κ)

const m = SMT(b = [0.0, 1000, 2500, 4000])
const θ = (S0 = 1.0, vax = 0.6, Dax = 2.0e-3)
const p = EQUIP.resolve(m, θ)
const d = [4, 30, 30, 30]
const sets = directionset.(d)

# Each voxel's magnitude images, by shell: κ = Inf for one fiber direction.
function simulate_pipeline(rng, σ, κ; nvox = 2000, K = 100)
    mags = Matrix{Vector{Float64}}(undef, nvox, 4)
    for i in 1:nvox
        n = normalize(SVector{3}(randn(rng, 3)))
        fibers = isinf(κ) ? [n] : [watson(rng, n, κ) for _ in 1:K]
        for j in 1:4
            mags[i, j] = [abs(mean(EQUIP._profile(m, p, j, dot(u, g)) for u in fibers) +
                              σ * (randn(rng) + im * randn(rng))) for g in sets[j]]
        end
    end
    return mags
end

const METHODS = ["no correction", "exact inversion/dir (Kaden)", "SMI.m sqrt(max(M²-σ²,0))",
                 "DESIGNER sqrt(|M²-σ²|)", "E|y|/dir in the model"]
function analyse(mags, σa)
    raw = mean.(mags)
    rs = (fit(m, raw; weights = d),
          fit(m, [mean(karlsen.(v, σa)) for v in mags]; weights = d),
          fit(m, [mean(sqrt.(max.(v .^ 2 .- σa^2, 0))) for v in mags]; weights = d),
          fit(m, [mean(sqrt.(abs.(v .^ 2 .- σa^2))) for v in mags]; weights = d),
          fit(RicianSphericalMean(m; σ = σa), raw; weights = d))
    return [(mean(r.params.vax) - θ.vax, sqrt(mean((r.params.vax .- θ.vax) .^ 2))) for r in rs]
end

function table(title, cols, results)
    println(title)
    print(rpad("", 30)); foreach(c -> print(lpad(c, 16)), cols); println()
    for (k, name) in enumerate(METHODS)
        print(rpad(name, 30))
        for res in results
            b, r = res[k]
            print(lpad(@sprintf("%+.3f / %.3f", b, r), 16))
        end
        println()
    end
    println()
end

# The Watson sampler: E[(μ·x)²] against its integral
for κ in (1.0, 4.0, 16.0)
    rng = MersenneTwister(5); μ = SVector(0.0, 0, 1)
    sim = mean(watson(rng, μ, κ)[3]^2 for _ in 1:200_000)
    ξs = range(0, 1; length = 100_001); w = exp.(κ .* ξs .^ 2)
    @printf("Watson κ = %4.0f: E[ξ²] %.4f simulated, %.4f exact (ODI %.2f)\n", κ, sim, sum(ξs .^ 2 .* w) / sum(w), odi(κ))
end
println()

t0 = time()
for snr in (20, 10)
    σ = 1 / snr
    # σ misestimated: one dataset, each correction given σ × f
    mags = simulate_pipeline(MersenneTwister(1), σ, Inf)
    fs = (0.8, 0.9, 1.0, 1.1, 1.2)
    table("SNR $snr, one fiber direction, σ used = f × true σ: v_ax bias / RMSE (truth 0.6)",
          ["f = $f" for f in fs], [analyse(mags, f * σ) for f in fs])
    # dispersion: data from a Watson ODF, σ right
    κs = (Inf, 16.0, 4.0, 1.0)
    table("SNR $snr, σ right, fibers dispersed (Watson κ): v_ax bias / RMSE (truth 0.6)",
          [isinf(κ) ? "none" : @sprintf("κ %g, ODI %.2f", κ, odi(κ)) for κ in κs],
          [analyse(simulate_pipeline(MersenneTwister(2), σ, κ), σ) for κ in κs])
end
println(round(time() - t0; digits = 1), " s")
