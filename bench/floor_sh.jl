# =================================================================================================
# The spherical mean from a Rician spherical-harmonic fit of each shell, against the other floor
# corrections for SMT, with and without fiber dispersion
#
# The pipeline of bench/floor_robustness.jl (one fiber population per voxel, or 100 fibers from a
# Watson distribution about it; a Rician magnitude image per direction). Each shell's direction
# magnitudes are fitted by least squares with E|y|(|Σ c_lm Y_lm(g)|, σ) as the model, the real even
# spherical harmonics to order L = 2, 4, 6 (6, 15, 28 coefficients for 30 directions); the l = 0
# coefficient is the shell's spherical mean (Y₀₀ scaled to 1), then the SMT fit. No assumption on
# the fiber ODF beyond the order L. σ known. Bias and RMSE of v_ax at SNR 20 and 10 per image
# (docs/design.md, "Peripheral nerve SMT", step 2). About 7 minutes per SNR on 4 threads:
#   julia -t auto --project=. bench/floor_sh.jl              (SNRS=20 or SNRS=10 for one of them)
# =================================================================================================

using EQUIP, Random, Statistics, LinearAlgebra, StaticArrays, Printf
using SpecialFunctions: besselix

function karlsen(M, σ)
    M <= σ * sqrt(π / 2) && return 0.0
    lo, hi = 0.0, M
    for _ in 1:60
        mid = (lo + hi) / 2
        ricianmean(mid, σ) < M ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
function watson(rng, μ, κ)
    t = 0.0
    while true
        t = rand(rng); rand(rng) <= exp(κ * (t^2 - 1)) && break
    end
    t *= rand(rng, Bool) ? 1 : -1
    a = abs(μ[1]) < 0.9 ? SVector(1.0, 0, 0) : SVector(0.0, 1, 0)
    e1 = normalize(a - dot(a, μ) * μ); e2 = cross(μ, e1); φ = 2π * rand(rng)
    return t * μ + sqrt(1 - t^2) * (cos(φ) * e1 + sin(φ) * e2)
end
odi(κ) = 2 / π * atan(1 / κ)

# dE|y|/dA = √(π/2)/2 · (A/σ) · e^(−ν/2)[I₀(ν/2) + I₁(ν/2)], ν = A²/2σ² (scaled Bessel functions)
function dricianmean(A, σ)
    A > 1000σ && return 1 - σ^2 / (2A^2)
    ν = A^2 / (2σ^2)
    return sqrt(π / 2) / 2 * (A / σ) * (besselix(0, ν / 2) + besselix(1, ν / 2))
end

# Real even spherical harmonics to order L, scaled by √(4π) so that Y₀₀ = 1 and each has mean
# square 1 over the sphere: the l = 0 coefficient is the spherical mean.
function plm(l, m, x)                       # associated Legendre P_l^m(x), m ≥ 0
    pmm = 1.0; s = sqrt(max(0.0, 1 - x^2)); f = 1.0
    for _ in 1:m; pmm *= -f * s; f += 2; end
    l == m && return pmm
    pm1 = x * (2m + 1) * pmm
    l == m + 1 && return pm1
    pll = 0.0
    for ll in m+2:l
        pll = ((2ll - 1) * x * pm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pm1 = pm1, pll
    end
    return pll
end
function shrow(g, L)
    θ = acos(clamp(g[3], -1, 1)); φ = atan(g[2], g[1]); x = cos(θ)
    row = Float64[]
    for l in 0:2:L, m in -l:l
        N = sqrt((2l + 1) * factorial(l - abs(m)) / factorial(l + abs(m)))   # l ≤ 10: Int factorials suffice
        v = m == 0 ? N * plm(l, 0, x) :
            m > 0 ? sqrt(2) * N * plm(l, m, x) * cos(m * φ) : sqrt(2) * N * plm(l, -m, x) * sin(-m * φ)
        push!(row, v)
    end
    return row
end
function shmatrix(set, L)                     # one row per direction (preallocated: no repeated vcat)
    Y = Matrix{Float64}(undef, length(set), (L + 1) * (L + 2) ÷ 2)
    for (i, g) in enumerate(set)
        Y[i, :] = shrow(g, L)
    end
    return Y
end

# Least squares of the magnitudes against E|y|(|Yc|, σ), Levenberg–Marquardt; the start from a
# linear SH fit of √max(M² − σ², 0). Returns the l = 0 coefficient, the spherical mean.
function shmean(M, Y, σ; iters = 100)
    c = Y \ sqrt.(max.(M .^ 2 .- σ^2, 0))
    A = Y * c
    r = M .- ricianmean.(abs.(A), σ)
    sse = sum(abs2, r); λ = 1e-3
    for _ in 1:iters
        J = (sign.(A) .* dricianmean.(abs.(A), σ)) .* Y
        H = J' * J; gr = J' * r
        δ = (H + λ * Diagonal(diag(H) .+ 1e-12)) \ gr
        cn = c + δ; An = Y * cn; rn = M .- ricianmean.(abs.(An), σ); ssen = sum(abs2, rn)
        if ssen < sse
            done = sse - ssen < 1e-12 * (1 + sse)
            c, A, r, sse = cn, An, rn, ssen; λ = max(λ / 10, 1e-9)
            done && break
        else
            λ *= 10; λ > 1e10 && break
        end
    end
    return c[1]
end

const m = SMT(b = [0.0, 1000, 2500, 4000])
const θ = (S0 = 1.0, vax = 0.6, Dax = 2.0e-3)
const p = EQUIP.resolve(m, θ)
const d = [4, 30, 30, 30]
const sets = directionset.(d)

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

# Checks: the SH basis orthonormal in the mean over the sphere; dE/dA against a difference
let rng = MersenneTwister(9), us = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:20_000]
    Y = shmatrix(us, 6); G = Y' * Y / length(us)
    @printf("SH basis: max |mean(YᵢYⱼ) − δᵢⱼ| = %.3f over %d functions (Monte Carlo, 20k points: ±0.02 expected)\n",
            maximum(abs.(G - I)), size(Y, 2))
    h = 1e-6
    @printf("dE/dA at A/σ = 0.5, 2, 30: %.6f %.6f %.6f (differences %.6f %.6f %.6f)\n",
            dricianmean(0.5, 1), dricianmean(2.0, 1), dricianmean(30.0, 1),
            (ricianmean(0.5 + h, 1) - ricianmean(0.5 - h, 1)) / 2h, (ricianmean(2 + h, 1) - ricianmean(2 - h, 1)) / 2h,
            (ricianmean(30 + h, 1) - ricianmean(30 - h, 1)) / 2h)
    # with no noise, the l = 0 coefficient of an L = 6 fit is the spherical mean of 30 directions
end
println(); flush(stdout)

const Ls = (2, 4, 6)
const SHY = Dict(L => [shmatrix(sets[j], L) for j in 1:4] for L in Ls)
const NAMES = ["no correction", "exact inversion per direction", "E|y|/dir in the model (1 fiber)",
               "SH L=2, E|y| per direction", "SH L=4, E|y| per direction", "SH L=6, E|y| per direction"]
function analyse(mags, σ)
    raw = mean.(mags)
    out = Any[fit(m, raw; weights = d),
              fit(m, [mean(karlsen.(v, σ)) for v in mags]; weights = d),
              fit(RicianSphericalMean(m; σ), raw; weights = d)]
    for L in Ls
        # b = 0: every image the same A, so the inversion of the mean is exact
        sm = zeros(size(mags, 1), 4)
        Threads.@threads for i in axes(mags, 1)              # voxels on threads (each fit independent)
            for j in 1:4
                sm[i, j] = j == 1 ? karlsen(mean(mags[i, 1]), σ) : shmean(mags[i, j], SHY[L][j], σ)
            end
        end
        push!(out, fit(m, sm; weights = d))
    end
    return [(mean(r.params.vax) - θ.vax, sqrt(mean((r.params.vax .- θ.vax) .^ 2))) for r in out]
end

t0 = time()
κs = (Inf, 16.0, 4.0, 1.0)
for snr in parse.(Int, split(get(ENV, "SNRS", "20,10"), ","))
    σ = 1 / snr
    res = [analyse(simulate_pipeline(MersenneTwister(2), σ, κ), σ) for κ in κs]
    println("SNR $snr, σ right: v_ax bias / RMSE (truth 0.6)")
    print(rpad("", 34)); foreach(κ -> print(lpad(isinf(κ) ? "ODI 0" : @sprintf("ODI %.2f", odi(κ)), 16)), κs); println()
    for (k, name) in enumerate(NAMES)
        print(rpad(name, 34))
        foreach(rr -> print(lpad(@sprintf("%+.3f / %.3f", rr[k]...), 16)), res); println()
    end
    println(round(time() - t0; digits = 1), " s so far\n"); flush(stdout)
end
