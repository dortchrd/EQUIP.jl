# NODDI, two further speed-ups evaluated (docs/design.md, item 7):
#   A. the series summed more cheaply per image. P₂ₙ(x) = Pₙ^(0,−1/2)(z), z = 2x² − 1 (Legendre's
#      even polynomials are Jacobi polynomials in z), so the even series is a Jacobi series of
#      half the length, summed by Clenshaw's recurrence: about 15 steps of one multiplication by
#      z, where the Legendre recurrence takes 30 steps of two (3b in options.jl)
#   B. starting values: constant (the default), the truth (the most a start can save), BestOf over
#      ODI, and ODI from the voxel's tensor's FA (a map made once from noise-free signals)
# 20,000 voxels of bench/noddi/brain.jl's tissue mix, 63 images, Rician SNR 30, 4 threads.
#   LMAX=30 julia -t 4 --project=. bench/noddi/further.jl
include(joinpath(@__DIR__, "options.jl"))
import EQUIP: _tensorindices, _cholesky, _chol_solve

# --- A. the Jacobi series by Clenshaw ------------------------------------------------------------
const NJ = LMAX ÷ 2                                       # P₀ … P_NJ in z
# Pₙ = (Aₙz + Bₙ)·Pₙ₋₁ − Cₙ·Pₙ₋₂ for Jacobi (a, b) = (0, −1/2) (the standard three-term recurrence)
_jD(n) = 2n * (n - 0.5) * (2n - 2.5)
const JA = ntuple(n -> (2n - 1.5) * (2n - 0.5) * (2n - 2.5) / _jD(n), NJ + 2)
const JB = ntuple(n -> (2n - 1.5) * (-0.25) / _jD(n), NJ + 2)
const JC = ntuple(n -> 2 * (n - 1) * (n - 1.5) * (2n - 0.5) / _jD(n), NJ + 2)
@inline function _jacobisum(c, z)
    b1 = zero(z * c[1]); b2 = b1
    @inbounds for k in NJ:-1:1
        b1, b2 = c[k+1] + (JA[k+1] * z + JB[k+1]) * b1 - JC[k+2] * b2, b1
    end
    return c[1] + (JA[1] * z + JB[1]) * b1 - JC[2] * b2
end
for M in (:NODDI_JAC, :NODDI_FA)
    @eval begin
        @kwdef struct $M{A,G} <: MRIModel
            b::A; g::G
        end
        relabel(::$M, θ) = _upperdirection(θ)
    end
end
parameters(::Type{<:NODDI_JAC}) = parameters(NODDI)
datastart(m::NODDI_JAC, y, p) = _tensordirection(m, y)
function _noddijac!(out, m, p)
    (; b, g) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    κ = _kappa(ODI); μ = fiberdirection(theta, phi); τ = _watsoncos2(κ); Dperp = (1 - vic) * Dpar
    w0 = _coeffs(PW, UW, u -> exp(κ * (u^2 - 1)))
    w = w0 / w0[1]
    blast = -one(eltype(b))
    cw = w
    @inbounds for j in eachindex(out, b, g)
        t = g[j] ⋅ μ
        if iszero(b[j])
            Aic = one(t)
        else
            if b[j] != blast
                β = b[j] * Dpar
                k = _coeffs(PK, VK, v -> exp(-β * v^2))
                cw = SVector(ntuple(i -> (4(i - 1) + 1) / 2 * k[i] * w[i], Val(NL)))
                blast = b[j]
            end
            Aic = _jacobisum(cw, 2t^2 - 1)
        end
        Aec = exp(-b[j] * (Dperp + (Dpar - Dperp) * (τ * t^2 + (1 - τ) * (1 - t^2) / 2)))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end
signal!(out, m::NODDI_JAC, p) = _noddijac!(out, m, p)
signal!(out, m::NODDI_FA, p) = _noddijac!(out, m, p)

# --- B. ODI from the tensor's FA ----------------------------------------------------------------
# The voxel's log-linear tensor (as `_tensordirection`), its FA and direction
function _tensorfa(m, y)
    (; b, g) = m
    T = eltype(y)
    A = zero(SMatrix{7,7,T}); r = zero(SVector{7,T})
    @inbounds for j in eachindex(y, b, g)
        y[j] > 0 || continue
        x, yy, z, bj = T(g[j][1]), T(g[j][2]), T(g[j][3]), T(b[j])
        row = SVector{7,T}(1, -bj * x * x, -bj * yy * yy, -bj * z * z, -2bj * x * yy, -2bj * x * z, -2bj * yy * z)
        w = y[j]^2
        A += w * row * row'; r += (w * log(y[j])) * row
    end
    L, ok = _cholesky(A)
    ok || return (T(0.3), T(π) / 2, zero(T))
    c = _chol_solve(L, r)
    D = SMatrix{3,3,T}(c[2], c[5], c[6], c[5], c[3], c[7], c[6], c[7], c[4])
    q, λ, U = _tensorindices(D)
    n = EQUIP._upperhemisphere(U[:, 3])
    return (q.FA, acos(clamp(n[3], -one(T), one(T))), atan(n[2], n[1]))
end
# FA → ODI: noise-free signals at vic 0.55, viso 0.1 over ODI, their tensors' FA (decreasing in ODI)
const ODIGRID = collect(range(0.02, 0.95; length = 60))
const FAGRID = [first(_tensorfa(NODDI_JAC(; acq...), signal(NODDI_JAC(; acq...),
                  (S0 = 1.0, vic = 0.55, viso = 0.1, ODI = o, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3e-3)))) for o in ODIGRID]
function _odifromfa(fa)
    isfinite(fa) || return 0.3                           # no tensor (setup's placeholder data)
    fa >= FAGRID[1] && return ODIGRID[1]
    fa <= FAGRID[end] && return ODIGRID[end]
    i = findfirst(<(fa), FAGRID)                         # FAGRID decreasing
    f = (fa - FAGRID[i]) / (FAGRID[i-1] - FAGRID[i])
    return ODIGRID[i] + f * (ODIGRID[i-1] - ODIGRID[i])
end
parameters(::Type{<:NODDI_FA}) = merge(parameters(NODDI), (ODI = Free(start = :data, lower = EQUIP._ODI_MIN, upper = 1),))
function datastart(m::NODDI_FA, y, p)
    fa, θ, φ = _tensorfa(m, y)
    return (theta = θ, phi = φ, ODI = oftype(fa, _odifromfa(Float64(fa))))
end

# --- measure ------------------------------------------------------------------------------------
println("A. one signal and fits (the tabulated Legendre series against the Jacobi series by Clenshaw):")
p = merge(base, (ODI = 0.2,)); out = zeros(63)
for (label, M) in (("3b. Legendre recurrence", NODDI_SHFAST), ("3c. Jacobi, Clenshaw", NODDI_JAC))
    m = M(; acq...)
    @printf("   %s: one signal %.1f µs; largest difference from the built-in %.1e\n", label, 1e-3timeit(() -> signal!(out, m, p), 3000),
            maximum(maximum(abs.(signal(m, merge(base, (ODI = o, theta = th))) ./ signal(NODDI(; acq...), merge(base, (ODI = o, theta = th))) .- 1))
                    for o in (EQUIP._ODI_MIN, 0.1, 0.4, 1.0), th in (0.0, 0.9)))
end
N = 20_000
rngs = MersenneTwister(4)
dirsb = [normalize(SVector{3}(randn(rngs, 3))) for _ in 1:N]
viso = 0.3 .* rand(rngs, N); csf = rand(rngs, N) .< 0.1; viso[csf] .= 0.9
θs = (S0 = fill(1000.0, N), vic = 0.3 .+ 0.5 .* rand(rngs, N), viso = viso, ODI = 0.05 .+ 0.65 .* rand(rngs, N),
      theta = acos.(getindex.(dirsb, 3)), phi = atan.(getindex.(dirsb, 2), getindex.(dirsb, 1)), Dpar = fill(1.7e-3, N), Diso = fill(3e-3, N))
Ys = simulate(NODDI(; acq...), θs; snr = 30, noise = :rician, rng = rngs)
mj = NODDI_JAC(; acq...)
runs = [("built-in, constant starts", NODDI(; acq...), (;)),
        ("3b. Legendre series, constant starts", NODDI_SHFAST(; acq...), (;)),
        ("3c. Jacobi series, constant starts", mj, (;)),
        ("3c, starting at the truth (the floor)", mj, (vic = Free(start = θs.vic), viso = Free(start = θs.viso), ODI = Free(start = θs.ODI),
                                                     theta = Free(start = θs.theta), phi = Free(start = θs.phi))),
        ("3c, BestOf ODI (6 values)", mj, (ODI = Free(start = BestOf([0.03, 0.1, 0.2, 0.35, 0.55, 0.8])),)),
        ("3c, ODI from the tensor's FA", NODDI_FA(; acq...), (;))]
res = []
println("\nB. | version and start | µs a voxel | iterations | converged | not at the best minimum | median |ODI error| |")
println("|---|---:|---:|---:|---:|---:|")
for (label, m, kw) in runs
    r = fit(m, Ys; kw...)                                # compiles, and the result
    t = minimum(@elapsed(fit(m, Ys; kw...)) for _ in 1:2)
    push!(res, (label, t, r))
end
best = vec(minimum(reduce(hcat, [ifelse.(isconverged.(r.status), r.sse, Inf) for (_, _, r) in res]); dims = 2))
sy = vec(sum(abs2, Ys; dims = 2))
for (label, t, r) in res
    nb = count(i -> !isconverged(r.status[i]) || r.sse[i] > best[i] + 1e-8 * sy[i], 1:N)
    @printf("| %s | %.0f | %.1f | %.1f%% | %d | %.3f |\n", label, 1e6t / N, mean(r.iterations), 100mean(isconverged.(r.status)), nb,
            median(abs.(r.params.ODI .- θs.ODI)))
    flush(stdout)
end
