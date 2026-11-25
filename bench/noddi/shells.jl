# NODDI's dispersed stick when the b-values vary image by image (docs/design.md, roadmap item 27,
# "Shells"). The stick's Watson series has 21 coefficients per b-value,
#
#   cₗ(β) = (4l+1)/2 · kₗ(β) · wₗ(κ),   kₗ(β) = ∫₋₁¹ e^(−βv²) P₂ₗ(v) dv,   β = b·D,
#
# whose kₗ cost 40 exponentials and about 840 multiply-adds (`_shellcoeffs`); the built-in keeps
# them for up to 8 distinct b-values and reuses them only where b is equal. Scanners write b per
# direction (995, 1000, 1005), so on such data almost every image recomputes them. Two ways to
# make kₗ at each image's own b cheap, each a model of its own here (NODDI's parameters, the
# built-in's in everything but kₗ, so the fits compare directly):
#
#   A. shells and a Taylor step: images within HB s/mm² of a shell's centre (the first b met)
#      share kₗ and its first three derivatives in β, made from the same 40 exponentials
#      (dᵐ/dβᵐ e^(−βv²) = (−v²)ᵐ e^(−βv²)); each image's kₗ(β0 + Δ) ≈ k + Δk′ + Δ²k″/2 + Δ³k‴/6,
#      the error about Δ⁴/24·∫v⁸ (≤ 1e-8 for Δ = HB·D ≤ 0.035); an image at the centre itself
#      takes the shell's coefficients as the built-in does, so equal b cost what they cost now
#   B. a table in β: kₗ and k′ₗ at β = 0, h, 2h, ... up to 12.75 (h = 0.05, 258 points, 83 kB),
#      made once, and cubic Hermite interpolation per image (error ≤ h⁴/384·max|k⁗| ≈ 4e-9): no
#      shells, the same cost whatever the b-values
#
# For each: the largest difference of the signal from the built-in (which computes kₗ exactly at
# every b, by its fallback past 8 distinct values: the reference); one signal (plain numbers) and
# one voxel's fit (one thread), for NODDI and NODDIConvolved (whose zeppelin's series is at
# D = vic·D∥, a dual number in the fit), on four acquisitions: two exact shells, the same
# jittered as scanners write them, ten exact shells, and ten jittered.
#   julia -t 1 --project=test bench/noddi/shells.jl
using EQUIP, Printf, Random, Statistics, StaticArrays, LinearAlgebra
import EQUIP: MRIModel, parameters, signal!, datastart, relabel, _kappa, _watsoncos2, _watsoncoeffs,
              _jacobisum, _shellcoeffs, _watsonstick, _watsonnorm, _plain, _STICK_TAB, _STICK_V,
              _WATSON_NL, _WATSON_βMAX, _tensordirection, _upperdirection, DiffusionDirections

const NL = _WATSON_NL
const FAC = SVector{NL}(ntuple(i -> (4(i - 1) + 1) / 2, NL))

# kₗ and its first three derivatives in β, from one set of exponentials
@inline function kderivs(β)
    T = typeof(exp(-β * 1.0))
    k0 = zero(MVector{NL,T}); k1 = zero(MVector{NL,T}); k2 = zero(MVector{NL,T}); k3 = zero(MVector{NL,T})
    @inbounds for k in eachindex(_STICK_V)
        v2 = _STICK_V[k]^2
        e0 = exp(-β * v2); e1 = -v2 * e0; e2 = -v2 * e1; e3 = -v2 * e2
        for l in 1:NL
            t = _STICK_TAB[l, k]
            k0[l] += t * e0; k1[l] += t * e1; k2[l] += t * e2; k3[l] += t * e3
        end
    end
    return SVector(k0), SVector(k1), SVector(k2), SVector(k3)
end

# --- A: shells within HB of a centre, a Taylor step to each image's b -------------------------------
const HB = 20.0
struct ShellsA{B,K,C}
    n::Int
    bc::MVector{8,B}
    kd::MVector{8,NTuple{4,K}}
    c0::MVector{8,C}
end
@inline function shellsA(b, D, w)
    K = typeof(exp(-first(b) * D))
    C = typeof(first(w) * D)
    bc = MVector{8,eltype(b)}(undef); kd = MVector{8,NTuple{4,SVector{NL,K}}}(undef)
    c0 = MVector{8,SVector{NL,C}}(undef)
    n = 0
    @inbounds for bj in b
        n == 8 && break
        (iszero(bj) || abs(bj * D) > _WATSON_βMAX) && continue
        found = false
        for s in 1:n
            abs(bj - bc[s]) <= HB && (found = true; break)
        end
        found && continue
        n += 1
        d = kderivs(bj * D)
        bc[n], kd[n], c0[n] = bj, d, SVector{NL,C}(FAC .* d[1] .* w)
    end
    return ShellsA(n, bc, kd, c0)
end
@inline function stickA(sh, b, D, w, κ, ξ)
    β = b * D
    iszero(b) && return one(ξ * β * first(w))
    abs(β) > _WATSON_βMAX && return _watsonstick(β, ξ, κ) / _watsonnorm(κ)
    z = 2ξ^2 - 1
    @inbounds for s in 1:sh.n
        d = b - sh.bc[s]
        abs(d) <= HB || continue
        iszero(d) && return _jacobisum(sh.c0[s], z)
        Δ = d * D
        k0, k1, k2, k3 = sh.kd[s]
        k = k0 + Δ * (k1 + Δ / 2 * (k2 + Δ / 3 * k3))
        return _jacobisum(FAC .* k .* w, z)
    end
    return _jacobisum(_shellcoeffs(β, w, typeof(first(w) * β)), z)
end

# --- B: a table of kₗ and k′ₗ in β, cubic Hermite per image -----------------------------------------
const H = 0.05
const NG = ceil(Int, _WATSON_βMAX / H) + 2
const KT = [kderivs(i * H)[1] for i in 0:NG]
const KDT = [kderivs(i * H)[2] for i in 0:NG]
@inline function kB(β)
    x = β / H
    i = unsafe_trunc(Int, _plain(x))
    t = x - i
    t2 = t * t; t3 = t2 * t
    h00 = 2t3 - 3t2 + 1; h10 = t3 - 2t2 + t; h01 = 3t2 - 2t3; h11 = t3 - t2
    @inbounds return h00 * KT[i+1] + (H * h10) * KDT[i+1] + h01 * KT[i+2] + (H * h11) * KDT[i+2]
end
@inline function stickB(b, D, w, κ, ξ)
    β = b * D
    iszero(b) && return one(ξ * β * first(w))
    abs(β) > _WATSON_βMAX && return _watsonstick(β, ξ, κ) / _watsonnorm(κ)
    return _jacobisum(FAC .* kB(β) .* w, 2ξ^2 - 1)
end

# --- the models: NODDI and NODDIConvolved, kₗ by A or by B -------------------------------------------
for M in (:NA, :NB, :NCA, :NCB)
    @eval begin
        @kwdef struct $M{A<:AbstractVector,G<:DiffusionDirections} <: MRIModel
            b::A
            g::G
        end
        parameters(::Type{<:$M}) = parameters(NODDI)
        datastart(m::$M, y, p) = _tensordirection(m, y)
        relabel(::$M, θ) = _upperdirection(θ)
    end
end
function _noddi!(out, m, p, ::Val{A}, ::Val{conv}) where {A,conv}
    (; b, g) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    μ = fiberdirection(theta, phi)
    κ = _kappa(ODI)
    w = _watsoncoeffs(κ)
    Dperp = (1 - vic) * Dpar
    τ = _watsoncos2(κ)
    Dpare, Dperpe = Dperp + (Dpar - Dperp) * τ, Dperp + (Dpar - Dperp) * (1 - τ) / 2
    sh = A === :A ? shellsA(b, Dpar, w) : nothing
    shz = A === :A && conv ? shellsA(b, Dpar - Dperp, w) : nothing
    @inbounds for j in eachindex(out, b, g)
        ξ = g[j] ⋅ μ
        Aic = A === :A ? stickA(sh, b[j], Dpar, w, κ, ξ) : stickB(b[j], Dpar, w, κ, ξ)
        Aec = conv ? exp(-b[j] * Dperp) * (A === :A ? stickA(shz, b[j], Dpar - Dperp, w, κ, ξ) :
                                                      stickB(b[j], Dpar - Dperp, w, κ, ξ)) :
                     zeppelin(b[j], Dpare, Dperpe, ξ)
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end
signal!(out, m::NA, p) = _noddi!(out, m, p, Val(:A), Val(false))
signal!(out, m::NB, p) = _noddi!(out, m, p, Val(:B), Val(false))
signal!(out, m::NCA, p) = _noddi!(out, m, p, Val(:A), Val(true))
signal!(out, m::NCB, p) = _noddi!(out, m, p, Val(:B), Val(true))

# --- the acquisitions ---------------------------------------------------------------------------------
rng = MersenneTwister(5)
dirs30 = directionset(30); dirs6 = directionset(6)
two = [0.0; fill(1000.0, 30); fill(2500.0, 30)]
twoj = [0.0; 1000 .+ rand(rng, -5:5, 30); 2500 .+ rand(rng, -10:10, 30)]
ten = [0.0; reduce(vcat, [fill(bv, 6) for bv in 500.0:500.0:5000.0])]
tenj = [0.0; reduce(vcat, [bv .+ rand(rng, -0.005bv:0.005bv, 6) for bv in 500.0:500.0:5000.0])]
acqs = [("2 shells, exact", diffusionencoding(two, [[SVector(0.0, 0, 0)]; dirs30; dirs30])),
        ("2 shells, jittered", diffusionencoding(twoj, [[SVector(0.0, 0, 0)]; dirs30; dirs30])),
        ("10 shells, exact", diffusionencoding(ten, [[SVector(0.0, 0, 0)]; repeat(dirs6, 10)])),
        ("10 shells, jittered", diffusionencoding(tenj, [[SVector(0.0, 0, 0)]; repeat(dirs6, 10)]))]

timeit(f, n) = minimum(@elapsed(for _ in 1:n; f(); end) for _ in 1:5) / n
function fittime(m, Y)
    c = init(m); fit!(c, view(Y, 1, :))
    return minimum(@elapsed(for i in axes(Y, 1); fit!(c, view(Y, i, :)); end) for _ in 1:3) / size(Y, 1)
end

# accuracy: over ODI, orientations, both forms, against the built-in
println("Largest |signal − built-in| / S0 (ODI 0.01 to 1, orientations every 30°, S0 = 1):")
for (name, acq) in acqs, (lab, mA, mB, mref) in (("NODDI", NA(; acq...), NB(; acq...), NODDI(; acq...)),
                                                 ("NODDIConvolved", NCA(; acq...), NCB(; acq...), NODDIConvolved(; acq...)))
    eA = eB = 0.0
    for o in (EQUIP._ODI_MIN, 0.05, 0.2, 0.5, 1.0), th in 0:π/6:π, ph in 0:π/3:π
        p = (S0 = 1.0, vic = 0.6, viso = 0.1, ODI = o, theta = th, phi = ph, Dpar = 1.7e-3, Diso = 3e-3)
        r = signal(mref, p)
        eA = max(eA, maximum(abs.(signal(mA, p) .- r))); eB = max(eB, maximum(abs.(signal(mB, p) .- r)))
    end
    @printf("  %-20s %-15s  A %.1e   B %.1e\n", name, lab, eA, eB)
end

println("\nSpeed, one thread: one signal (µs, plain numbers) | one voxel's fit (µs; 1000 voxels, Rician SNR 30)")
@printf("  %-20s %-15s %26s %26s %26s\n", "", "", "built-in", "A (shells + Taylor)", "B (table)")
for (name, acq) in acqs, (lab, ms) in (("NODDI", (NODDI(; acq...), NA(; acq...), NB(; acq...))),
                                      ("NODDIConvolved", (NODDIConvolved(; acq...), NCA(; acq...), NCB(; acq...))))
    N = 1000
    r2 = MersenneTwister(7); dirs = [normalize(randn(r2, 3)) for _ in 1:N]
    θs = (S0 = fill(1000.0, N), vic = 0.3 .+ 0.5 .* rand(r2, N), viso = 0.2 .* rand(r2, N), ODI = 0.05 .+ 0.6 .* rand(r2, N),
          theta = acos.(getindex.(dirs, 3)), phi = atan.(getindex.(dirs, 2), getindex.(dirs, 1)), Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N))
    Y = simulate(ms[1], θs; snr = 30, noise = :rician, rng = r2)
    p = (S0 = 1000.0, vic = 0.6, viso = 0.1, ODI = 0.2, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3e-3)
    cells = map(ms) do m
        out = zeros(nobs(m)); signal!(out, m, p)
        @sprintf("%7.2f | %7.1f", 1e6 * timeit(() -> signal!(out, m, p), 2000), 1e6 * fittime(m, Y))
    end
    @printf("  %-20s %-15s %26s %26s %26s\n", name, lab, cells...)
    flush(stdout)
end
