# =================================================================================================
# Test cases for bench/solvers.jl, from easy to hard. Each makes a model, simulated data (one row
# per voxel), the truth and EQUIP's roles, from a fixed seed, so the data are the same every time.
#
# The everyday relaxometry models, with T1 and T2 over the range of the 3 T values of Stanisz et
# al., Magn Reson Med 2005;54:507, Table 1 (`TISSUES`: T1 0.81-1.93 s, T2 27-275 ms, liver to
# blood), each uniform, and the scale (M0 or S0) 0.8-1.2:
#   :t2               T2Decay, 32 echoes 10-320 ms, SNR 100: two parameters, many observations,
#                     a data-driven start; every method should agree here
#   :t1mfa            T1MFA (variable flip angle), FA 2, 5, 10, 15, 20, 30°, TR 15 ms, B1 = 1,
#                     σ = M0/1000 (spoiled gradient-echo signals are a few % of M0: image SNR
#                     about 20-90): strongly nonlinear in T1
#   :t1ir             T1IR, TI 50 ms to 3.2 s (7 values, doubling), magnitude data, Sf -1.0 to
#                     -0.8, SNR 100: the sign is lost around the null point, so there are false
#                     minima (EQUIP's start is the best of a grid of T1 values)
#
# Selective inversion recovery (qMT) and a biexponential:
#   :sir4_range       4-point SIR (SIR_4POINT), tissue values over the range of the four SIR
#                     classes of Dortch et al., Magn Reson Med 2018;80:1824, Table 1 (PSR
#                     0.05-0.20, kmf 5-20 s⁻¹, R1f 0.5-1.5 s⁻¹, each uniform), Sf -1.0 to -0.8,
#                     M0f 0.8-1.2, Sm = 0.83, R1m = R1f, SNR 200. Fitted with kmf fixed at 12.5.
#                     4 observations and 4 free parameters, so every fit is exact. The same data as
#                     examples/sir_4point_range.jl (same seed, same order of random draws).
#   :sir4_poorstart   the same data, but every method starts from random values far from the
#                     answer: PSR 0.02-0.5, R1f 0.3-3 s⁻¹, Sf -1.2 to -0.6, and M0f's usual start
#                     (scaled to the data) times 0.5-2
#   :sir16_kmf        16-point SIR (SIR_16POINT, Dortch 2018 Methods) with kmf estimated: 16
#                     observations and 5 free parameters, so the fits are least squares, not exact;
#                     same tissue ranges, SNR 200
#   :sir16_kmf_snr50  the same at SNR 50
#   :sir4_bounds      4-point SIR with a small macromolecular pool (PSR 0.005-0.05, other values as
#                     :sir4_range) at SNR 50, fitted with QMTSIR's bounds: many voxels' best
#                     unbounded fit has PSR < 0, so the bound PSR ≥ 0 comes into play
#   :biexp            T2BiexpDecay, 32 echoes 10-320 ms, values around those of myelin water
#                     imaging: fshort 0.05-0.30, T2short 10-30 ms, T2long 60-120 ms, S0 0.8-1.2,
#                     SNR 100 (relative to S0); two exponentials, the classic ill-conditioned fit
#
# Starts: EQUIP's default starts (each model's `parameters`, with the scale parameter scaled to
# the voxel's data), the same for every method, except :sir4_poorstart.
#
# A biexponential fit can come out with its two components swapped (the "short" one with the
# longer T2) and the same SSE. Before comparing answers, `biexp_order!` swaps them back
# (T2short ≤ T2long, fshort → 1 - fshort), for every method alike.
# =================================================================================================

using EQUIP, Random

const ALLCASES = (:t2, :t1mfa, :t1ir, :sir4_range, :sir4_poorstart, :sir16_kmf, :sir16_kmf_snr50,
                  :sir4_bounds, :biexp)
const NVOX = 10_000

uniform(rng, (lo, hi), n) = lo .+ (hi - lo) .* rand(rng, n)      # n values, uniform in [lo, hi)
sirclass(name) = [tissue(c)[name] for c in (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)]
const STANISZ = (:liver, :muscle, :heart, :kidney, :cartilage_0, :cartilage_55, :WM, :GM,
                 :optic_nerve, :spinal_cord, :blood)
stanisz(name) = extrema(tissue(t)[name] for t in STANISZ)        # e.g. (0.812, 1.932) for T1

# SIR tissue values, each uniform over the range of the four classes (or a given PSR range), in the
# same order of random draws as examples/sir_4point_range.jl: PSR, R1f, kmf, then Sf, then M0f.
function sirtruth(rng, n; psr = extrema(sirclass(:PSR)))
    PSR = uniform(rng, psr, n)
    R1f = uniform(rng, extrema(sirclass(:R1f)), n)
    kmf = uniform(rng, extrema(sirclass(:kmf)), n)
    Sf = uniform(rng, (-1.0, -0.8), n)
    M0f = uniform(rng, (0.8, 1.2), n)
    return (; PSR, R1f, Sf, M0f, kmf, Sm = 0.83)   # Sm fixed; R1m stays tied to R1f (the default)
end

# Poor starts: random values far from the answer, a fresh draw for every voxel. P0 holds the
# usual starts (one column per voxel, one row per free parameter, in the order of `names`).
# Put every biexponential fit in the same order, T2short ≤ T2long (est: one column per voxel)
function biexp_order!(est, names)
    f, s, l = findfirst(==(:fshort), names), findfirst(==(:T2short), names), findfirst(==(:T2long), names)
    for v in axes(est, 2)
        if est[s, v] > est[l, v]
            est[s, v], est[l, v] = est[l, v], est[s, v]
            est[f, v] = 1 - est[f, v]
        end
    end
    return est
end

const POOR = (PSR = (0.02, 0.5), R1f = (0.3, 3.0), Sf = (-1.2, -0.6))
function poorstarts(rng, P0, names)
    P = copy(P0)
    for (j, n) in enumerate(names)
        P[j, :] .= haskey(POOR, n) ? uniform(rng, POOR[n], size(P, 2)) :
                                     P0[j, :] .* uniform(rng, (0.5, 2.0), size(P, 2))
    end
    return P
end

"""
    makecase(name; n = NVOX)

One test case: `(; name, title, m, Y, truth, roles, bounds, starts, order!)`. `roles` are the
keywords for EQUIP's `fit` (e.g. `kmf = Free()`), `bounds` whether the case is fitted with the
model's bounds, `starts` a function changing the default starts (or `nothing`), `order!` a
function putting the estimates in a standard order before comparing (or `nothing`).
"""
function makecase(name::Symbol; n = NVOX)
    rng = MersenneTwister(1)
    if name === :t2
        m = T2Decay(TE = collect(0.01:0.01:0.32))
        truth = (S0 = uniform(rng, (0.8, 1.2), n), T2 = uniform(rng, stanisz(:T2), n))
        Y = simulate(m, truth; snr = 100, rng)
        return (; name, title = "T2 decay, 32 echoes, T2 27-275 ms, SNR 100", m, Y, truth,
                roles = (;), bounds = false, starts = nothing, order! = nothing)
    elseif name === :t1mfa
        m = T1MFA(FA = [2.0, 5, 10, 15, 20, 30], TR = 0.015)
        truth = (M0 = uniform(rng, (0.8, 1.2), n), T1 = uniform(rng, stanisz(:T1), n))
        Y = simulate(m, truth; snr = 1000, rng)
        return (; name, title = "T1 variable flip angle, 6 angles, T1 0.81-1.93 s, σ = M0/1000", m, Y,
                truth, roles = (;), bounds = false, starts = nothing, order! = nothing)
    elseif name === :t1ir
        m = T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
        truth = (M0 = uniform(rng, (0.8, 1.2), n), Sf = uniform(rng, (-1.0, -0.8), n),
                 T1 = uniform(rng, stanisz(:T1), n))
        Y = simulate(m, truth; snr = 100, rng)
        return (; name, title = "T1 inversion recovery, 7 TIs, magnitude, T1 0.81-1.93 s, SNR 100", m,
                Y, truth, roles = (;), bounds = false, starts = nothing, order! = nothing)
    elseif name in (:sir4_range, :sir4_poorstart)
        m = QMTSIR(; SIR_4POINT...)
        truth = sirtruth(rng, n)
        Y = simulate(m, truth; snr = 200, rng)
        poor = name === :sir4_poorstart
        title = poor ? "4-point SIR, tissue range, SNR 200, poor random starts" :
                       "4-point SIR, tissue range, SNR 200"
        return (; name, title, m, Y, truth, roles = (;), bounds = false,
                starts = poor ? poorstarts : nothing, order! = nothing)
    elseif name in (:sir16_kmf, :sir16_kmf_snr50)
        snr = name === :sir16_kmf ? 200 : 50
        m = QMTSIR(; SIR_16POINT...)
        truth = sirtruth(rng, n)
        Y = simulate(m, truth; snr, rng)
        return (; name, title = "16-point SIR, kmf estimated, tissue range, SNR $snr", m, Y, truth,
                roles = (kmf = Free(),), bounds = false, starts = nothing, order! = nothing)
    elseif name === :sir4_bounds
        m = QMTSIR(; SIR_4POINT...)
        truth = sirtruth(rng, n; psr = (0.005, 0.05))
        Y = simulate(m, truth; snr = 50, rng)
        return (; name, title = "4-point SIR, PSR 0.005-0.05, SNR 50, with bounds", m, Y, truth,
                roles = (;), bounds = true, starts = nothing, order! = nothing)
    elseif name === :biexp
        m = T2BiexpDecay(TE = collect(0.01:0.01:0.32))
        truth = (S0 = uniform(rng, (0.8, 1.2), n), fshort = uniform(rng, (0.05, 0.30), n),
                 T2short = uniform(rng, (0.010, 0.030), n), T2long = uniform(rng, (0.060, 0.120), n))
        Y = simulate(m, truth; snr = 100, rng)
        return (; name, title = "biexponential T2, 32 echoes, myelin-water-like values, SNR 100",
                m, Y, truth, roles = (;), bounds = false, starts = nothing, order! = biexp_order!)
    end
    throw(ArgumentError("unknown case $name; the cases: $(join(ALLCASES, ", "))"))
end
