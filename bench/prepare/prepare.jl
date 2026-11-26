# Is preparing the parameter-independent work once per fit worth it? (docs/design.md, roadmap item
# 28.) Two models with their per-image work that does not depend on the fitted parameters made
# once, when the model is made, against the built-ins, which make it at every signal:
#
#   NODDI: the stick's series coefficients (2l+1)/2·kₗ(b·D∥) per image (21 numbers), D∥ fixed at
#          1.7e-3, made from the table (item 27) once; at each signal only their product with
#          the fitted wₗ(κ) and the sum. A D∥ other than the one prepared for (freed, a dual
#          number; a map; crlb's leakage seeding it) falls back to the built-in's path.
#   DKI:   the 6 products of the gradient direction in gᵀDg and the 15 in Σ gᵢgⱼgₖgₗVᵢⱼₖₗ per
#          image, which depend on the acquisition alone.
#
# Each prepared model is a model of its own here (the built-in's parameters, start and labels),
# checked against the built-in's signal, and timed: one signal (plain numbers), one voxel's fit
# (one thread), and a whole brain's fit (294,912 voxels, Rician SNR 30, 4 threads; NODDI on 63
# images, b 1000 and 2500, exact and as scanners write them; DKI on 1 + 2 × 30 images, b 1000
# and 2000).
#   julia -t 4 --project=test bench/prepare/prepare.jl
using EQUIP, Random, Statistics, Printf, LinearAlgebra, StaticArrays
import EQUIP: MRIModel, parameters, signal!, datastart, relabel, _kappa, _watsoncoeffs, _jacobisum,
              _shellcoeffs, _WATSON_βMAX, _WATSON_NL, _tensordirection, _upperdirection, _kurtosisweights,
              _kurtosisvector

# --- NODDI, the stick's coefficients prepared at D∥ ------------------------------------------------
@kwdef struct NODDIPrep{A,G,K} <: MRIModel
    b::A
    g::G
    kf::K                  # per image: (2l+1)/2·kₗ(b·D∥) (the wₗ factor left out: it is fitted)
    Dprep::Float64         # the D∥ they were made for
end
parameters(::Type{<:NODDIPrep}) = parameters(NODDI)
datastart(m::NODDIPrep, y, p) = _tensordirection(m, y)
relabel(::NODDIPrep, θ) = _upperdirection(θ)
function noddiprep(acq, D = 1.7e-3)
    ones21 = SVector{_WATSON_NL}(ntuple(_ -> 1.0, _WATSON_NL))
    kf = [0 <= bj * D <= _WATSON_βMAX ? _shellcoeffs(bj * D, ones21) : zero(ones21) for bj in acq.b]
    return NODDIPrep(acq.b, acq.g, kf, D)
end
function signal!(out, m::NODDIPrep, p)
    (; b, g, kf, Dprep) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    μ = fiberdirection(theta, phi)
    w = watson(_kappa(ODI))
    Dpare, Dperpe = watsontensor(Dpar, (1 - vic) * Dpar, w)
    prepared = Dpar isa AbstractFloat && Dpar == Dprep       # else: the built-in's path
    @inbounds for j in eachindex(out, b, g)
        ξ = g[j] ⋅ μ
        Aic = prepared && b[j] > 0 && b[j] * Dpar <= _WATSON_βMAX ? _jacobisum(kf[j] .* w.w, 2ξ^2 - 1) :
              stick(b[j], Dpar, w(ξ))
        Aec = zeppelin(b[j], Dpare, Dperpe, ξ)
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

# --- DKI, the direction's products prepared --------------------------------------------------------
@kwdef struct DKIPrep{A,G,W,K} <: MRIModel
    b::A
    g::G
    dw::W                  # per image: gx², gy², gz², 2gx·gy, 2gx·gz, 2gy·gz
    kw::K                  # per image: the 15 products gᵢgⱼgₖgₗ with their multiplicities
end
parameters(::Type{<:DKIPrep}) = parameters(DKI)
dkiprep(acq) = DKIPrep(acq.b, acq.g, [EQUIP._tensorweights(x) for x in acq.g],
                       [_kurtosisweights(x) for x in acq.g])
function signal!(out, m::DKIPrep, p)
    (; b, dw, kw) = m
    (; S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz) = p
    Dv = SVector(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
    V = _kurtosisvector(p)
    @inbounds for j in eachindex(out, b, dw, kw)
        out[j] = S0 * exp(-b[j] * (dw[j] ⋅ Dv) + b[j]^2 * (kw[j] ⋅ V) / 6)
    end
    return out
end

# --- measure -----------------------------------------------------------------------------------------
timeit(f, n) = minimum(@elapsed(for _ in 1:n; f(); end) for _ in 1:5) / n
function fit1(m, Y)                                   # one voxel's fit, one thread
    c = init(m); fit!(c, view(Y, 1, :))
    return minimum(@elapsed(for i in axes(Y, 1); fit!(c, view(Y, i, :)); end) for _ in 1:3) / size(Y, 1)
end
N = round(Int, 128 * 128 * 60 * 0.3)
g30 = directionset(30)
G = [[SVector(0.0, 0, 0) for _ in 1:3]; g30; g30]
println("| model | acquisition | version | signal, largest difference | one signal | one voxel's fit | whole brain |")
println("|---|---|---|---:|---:|---:|---:|")
# NODDI
rng = MersenneTwister(4)
dirs = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:N]
viso = 0.3 .* rand(rng, N); csf = rand(rng, N) .< 0.1; viso[csf] .= 0.9
θn = (S0 = fill(1000.0, N), vic = 0.3 .+ 0.5 .* rand(rng, N), viso = viso, ODI = 0.05 .+ 0.65 .* rand(rng, N),
      theta = acos.(getindex.(dirs, 3)), phi = atan.(getindex.(dirs, 2), getindex.(dirs, 1)), Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N))
pn = (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3e-3)
for (lab, b) in (("63 images, exact b", [fill(0.0, 3); fill(1000.0, 30); fill(2500.0, 30)]),
                 ("63 images, b as scanners write them", [fill(0.0, 3); 1000 .+ rand(MersenneTwister(1), -5:5, 30); 2500 .+ rand(MersenneTwister(2), -10:10, 30)]))
    acq = diffusionencoding(b, G)
    Y = simulate(NODDI(; acq...), θn; snr = 30, noise = :rician, rng = MersenneTwister(9))
    for (ver, m) in (("built-in", NODDI(; acq...)), ("prepared", noddiprep(acq)))
        d = maximum(abs.(signal(m, pn) .- signal(NODDI(; acq...), pn))) / 1000
        out = zeros(nobs(m)); signal!(out, m, pn)
        ts = timeit(() -> signal!(out, m, pn), 2000)
        tf = fit1(m, Y[1:2000, :])
        fit(m, Y[1:100, :]); tb = minimum(@elapsed(fit(m, Y)) for _ in 1:2)
        @printf("| NODDI | %s | %s | %.1e | %.2f µs | %.1f µs | %.1f s |\n", lab, ver, d, 1e6ts, 1e6tf, tb)
        flush(stdout)
    end
end
# DKI
v = (0.8e-3)^2
θk1 = (S0 = 1000.0, Dxx = 1.7e-3, Dyy = 0.4e-3, Dzz = 0.3e-3, Dxy = 0.1e-3, Dxz = -0.05e-3, Dyz = 0.02e-3,
       V1111 = v, V2222 = v, V3333 = v, V1112 = 0.0, V1113 = 0.0, V1222 = 0.0, V2223 = 0.0, V1333 = 0.0,
       V2333 = 0.0, V1122 = v / 3, V1133 = v / 3, V2233 = v / 3, V1123 = 0.0, V1223 = 0.0, V1233 = 0.0)
acq = diffusionencoding([0.0; fill(1000.0, 30); fill(2000.0, 30)], [[SVector(0.0, 0, 0)]; g30; g30])
r2 = MersenneTwister(6)
θk = map(x -> x isa Real ? fill(x, N) : x, merge(θk1, (Dxx = 1.7e-3 .* (0.7 .+ 0.6 .* rand(r2, N)),)))
Y = simulate(DKI(; acq...), θk; snr = 30, noise = :rician, rng = MersenneTwister(9))
for (ver, m) in (("built-in", DKI(; acq...)), ("prepared", dkiprep(acq)))
    d = maximum(abs.(signal(m, θk1) .- signal(DKI(; acq...), θk1))) / 1000
    out = zeros(nobs(m)); signal!(out, m, θk1)
    ts = timeit(() -> signal!(out, m, θk1), 2000)
    tf = fit1(m, Y[1:2000, :])
    fit(m, Y[1:100, :]); tb = minimum(@elapsed(fit(m, Y)) for _ in 1:2)
    @printf("| DKI | 61 images, b 1000 and 2000 | %s | %.1e | %.2f µs | %.1f µs | %.1f s |\n", ver, d, 1e6ts, 1e6tf, tb)
    flush(stdout)
end
