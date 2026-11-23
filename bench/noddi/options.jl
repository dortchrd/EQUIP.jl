# NODDI's speed, three options against the built-in (docs/design.md, item 7, "Where its time
# goes"), each a model of its own here with the same parameters, so the fits compare directly:
#   1. NODDI_I0: the scaled Bessel function I0 alone where no derivative is needed (plain numbers;
#      on dual numbers the derivative needs I1 too, so a fit's evaluations are unchanged)
#   2. NODDI_12: 12 nodes for the stick's integral, not 16
#   3. NODDI_SH: the Funk–Hecke series, A_ic(t) = Σₗ (2l+1)/2 · kₗ(β) · wₗ(κ)/w₀(κ) · Pₗ(t), l even:
#      wₗ = ∫₋₁¹ e^(κ(u²−1)) Pₗ(u) du (the Watson's Legendre coefficients, once per voxel), kₗ =
#      ∫₋₁¹ e^(−βv²) Pₗ(v) dv (the stick's, once per shell: recomputed when b changes from the
#      previous image), both by Gauss–Legendre on [0, 1]; then per image Pₗ(t) by the recurrence
# Accuracy: the largest relative difference of the signal from the built-in (16 nodes, 2e-8 from
# a 2-D quadrature over the sphere) over ODI and orientations; speed: one signal, plain, and
# one voxel's fit, one thread; and a dataset fit, 1000 voxels, SNR 30 (Rician), 4 threads.
#   julia -t 4 --project=. bench/noddi/options.jl
using EQUIP, Printf, Random, Statistics, StaticArrays, LinearAlgebra
import EQUIP: parameters, signal!, datastart, relabel, _kappa, _watsoncos2, _i01e, _clenshaw2,
              _I0E_LO, _I0E_HI, _gausslegendre01, _tensordirection, _upperdirection
for M in (:NODDI_I0, :NODDI_12, :NODDI_SH)
    @eval begin
        @kwdef struct $M{A,X,Y,Z} <: MRIModel
            b::A; gx::X; gy::Y; gz::Z
        end
        parameters(::Type{<:$M}) = parameters(NODDI)
        datastart(m::$M, y, p) = _tensordirection(m, y)
        relabel(::$M, θ) = _upperdirection(θ)
    end
end
# e^(−x)·I0(x) alone (plain numbers)
@inline function _i0e(x::Float64)
    if x <= 8
        a1, a2 = 0.0, 0.0; t = x / 4 - 1; t2 = 2t
        @inbounds for k in length(_I0E_LO):-1:2; a1, a2 = muladd(t2, a1, -a2) + _I0E_LO[k], a1; end
        return muladd(t, a1, -a2) + _I0E_LO[1]
    end
    a1, a2 = 0.0, 0.0; t = 16 / x - 1; t2 = 2t
    @inbounds for k in length(_I0E_HI):-1:2; a1, a2 = muladd(t2, a1, -a2) + _I0E_HI[k], a1; end
    return (muladd(t, a1, -a2) + _I0E_HI[1]) / sqrt(x)
end
@inline _i0e(x) = first(_i01e(x))            # dual numbers: I1 is needed for the derivative
# the stick integral with given nodes and Bessel function
@inline function _stick(β, t, κ, Z, W, i0)
    T = κ - β; q = sqrt(T^2 / 4 + κ * β * (1 - t^2)); a = T / 2 + q
    s = zero(a)
    @inbounds for k in eachindex(Z); r = 1 - Z[k]^2; s += W[k] * exp(r * a - κ) * i0(r * q); end
    return s
end
@inline function _norm(κ, Z, W, i0)
    s = zero(κ)
    @inbounds for k in eachindex(Z); r = 1 - Z[k]^2; s += W[k] * exp((r - 1) * κ) * i0(r * κ / 2); end
    return s
end
const Z16, W16 = _gausslegendre01(16)
const Z12, W12 = _gausslegendre01(12)
function _noddi!(out, m, p, stickfun)
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    κ = _kappa(ODI); μ = fiberdirection(theta, phi); τ = _watsoncos2(κ); Dperp = (1 - vic) * Dpar
    st = stickfun(κ)
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        t = (gx[j], gy[j], gz[j]) ⋅ μ
        Aic = iszero(b[j]) ? one(t) : st(b[j] * Dpar, t)
        Aec = exp(-b[j] * (Dperp + (Dpar - Dperp) * (τ * t^2 + (1 - τ) * (1 - t^2) / 2)))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end
signal!(out, m::NODDI_I0, p) = _noddi!(out, m, p, κ -> (d = _norm(κ, Z16, W16, _i0e); (β, t) -> _stick(β, t, κ, Z16, W16, _i0e) / d))
signal!(out, m::NODDI_12, p) = _noddi!(out, m, p, κ -> (d = _norm(κ, Z12, W12, _i0e); (β, t) -> _stick(β, t, κ, Z12, W12, _i0e) / d))

# --- 3. the Funk–Hecke series -----------------------------------------------------------------------
const LMAX = parse(Int, get(ENV, "LMAX", "36"))           # even l up to LMAX
const NL = LMAX ÷ 2 + 1
const NW = parse(Int, get(ENV, "NW", "64"))               # nodes for the Watson's coefficients
const NK = parse(Int, get(ENV, "NK", "40"))               # ... for the stick's
# the Watson's integrand peaks at u = 1 (width about 1/(2κ)): u = 1 − s², s on [0, 1], puts the
# nodes there (du = 2s ds)
const ZW, WW = _gausslegendre01(NW)
const ZK, WK = _gausslegendre01(NK)
# P₀, P₂, ..., P_LMAX at x, by the three-term recurrence (every degree, the even ones kept)
@inline function _legendre_even(x)
    p0, p1 = one(x), x
    v = MVector{NL,typeof(x)}(undef)
    v[1] = p0
    @inbounds for l in 2:LMAX
        p0, p1 = p1, ((2l - 1) * x * p1 - (l - 1) * p0) / l
        iseven(l) && (v[l ÷ 2 + 1] = p1)
    end
    return SVector(v)
end
# w_l/w_0 for the even l
@inline function _watsoncoeffs(κ)
    w = zero(SVector{NL,typeof(κ)})
    @inbounds for k in eachindex(ZW)
        s = ZW[k]; u = 1 - s^2
        w += (WW[k] * 2s * exp(κ * (u^2 - 1))) * _legendre_even(u)
    end
    return w / w[1]
end
# (2l+1)/2 · k_l(β) for the even l
@inline function _stickcoeffs(β)
    c = zero(SVector{NL,typeof(β)})
    @inbounds for k in eachindex(ZK)
        v = ZK[k]
        c += (2WK[k] * exp(-β * v^2)) * _legendre_even(v)
    end
    return SVector(ntuple(i -> (4(i - 1) + 1) / 2 * c[i], Val(NL)))
end
function signal!(out, m::NODDI_SH, p)
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    κ = _kappa(ODI); μ = fiberdirection(theta, phi); τ = _watsoncos2(κ); Dperp = (1 - vic) * Dpar
    w = _watsoncoeffs(κ)
    blast = -one(eltype(b)); ck = _stickcoeffs(zero(Dpar))
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        t = (gx[j], gy[j], gz[j]) ⋅ μ
        if iszero(b[j])
            Aic = one(t)
        else
            b[j] == blast || (ck = _stickcoeffs(b[j] * Dpar); blast = b[j])
            Aic = sum(ck .* w .* _legendre_even(t))
        end
        Aec = exp(-b[j] * (Dperp + (Dpar - Dperp) * (τ * t^2 + (1 - τ) * (1 - t^2) / 2)))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

# 3b. the same series, written for speed: the Legendre values at the fixed nodes tabulated once
# (with the quadrature weights), so per voxel only the exponentials at the nodes are new; the
# coefficients summed into stack arrays (MVector); per image one recurrence, summing as it goes
for M in (:NODDI_SHFAST,)
    @eval begin
        @kwdef struct $M{A,X,Y,Z} <: MRIModel
            b::A; gx::X; gy::Y; gz::Z
        end
        parameters(::Type{<:$M}) = parameters(NODDI)
        datastart(m::$M, y, p) = _tensordirection(m, y)
        relabel(::$M, θ) = _upperdirection(θ)
    end
end
function _ptable(Z, W, map_u)
    tab = zeros(NL, length(Z))
    for k in eachindex(Z)
        u, wt = map_u(Z[k], W[k])
        p0, p1 = 1.0, u
        tab[1, k] = wt
        for l in 2:LMAX
            p0, p1 = p1, ((2l - 1) * u * p1 - (l - 1) * p0) / l
            iseven(l) && (tab[l ÷ 2 + 1, k] = wt * p1)
        end
    end
    return tab
end
const PW = _ptable(ZW, WW, (s, w) -> (1 - s^2, 2s * w))       # u = 1 − s², du = 2s ds
const UW = [1 - s^2 for s in ZW]
const PK = _ptable(ZK, WK, (v, w) -> (v, 2w))
const VK = collect(ZK)
@inline function _coeffs(tab, x, f)
    c = zero(MVector{NL,typeof(f(x[1]))})
    @inbounds for k in eachindex(x)
        e = f(x[k])
        for l in 1:NL; c[l] += tab[l, k] * e; end
    end
    return SVector(c)
end
function signal!(out, m::NODDI_SHFAST, p)
    (; b, gx, gy, gz) = m
    (; S0, vic, viso, ODI, theta, phi, Dpar, Diso) = p
    κ = _kappa(ODI); μ = fiberdirection(theta, phi); τ = _watsoncos2(κ); Dperp = (1 - vic) * Dpar
    w0 = _coeffs(PW, UW, u -> exp(κ * (u^2 - 1)))
    w = w0 / w0[1]                                              # (a new name: a reassigned variable
                                                                #  captured by a closure is boxed)
    blast = -one(eltype(b))
    cw = w                                                      # (2l+1)/2·kₗ·wₗ/w₀, per shell
    @inbounds for j in eachindex(out, b, gx, gy, gz)
        t = (gx[j], gy[j], gz[j]) ⋅ μ
        if iszero(b[j])
            Aic = one(t)
        else
            if b[j] != blast
                β = b[j] * Dpar
                k = _coeffs(PK, VK, v -> exp(-β * v^2))
                cw = SVector(ntuple(i -> (4(i - 1) + 1) / 2 * k[i] * w[i], Val(NL)))
                blast = b[j]
            end
            p0, p1 = one(t), t
            Aic = cw[1] * p0
            for l in 2:LMAX
                p0, p1 = p1, ((2l - 1) * t * p1 - (l - 1) * p0) / l
                iseven(l) && (Aic += cw[l ÷ 2 + 1] * p1)
            end
        end
        Aec = exp(-b[j] * (Dperp + (Dpar - Dperp) * (τ * t^2 + (1 - τ) * (1 - t^2) / 2)))
        out[j] = S0 * ((1 - viso) * (vic * Aic + (1 - vic) * Aec) + viso * iso(b[j], Diso))
    end
    return out
end

# --- measure ------------------------------------------------------------------------------------
function timeit(f, n)
    f(); t = Inf
    for _ in 1:5; t0 = time_ns(); for _ in 1:n; f(); end; t = min(t, (time_ns() - t0) / n); end
    return t
end
g = directionset(30)
acq = diffusionencoding([fill(0.0, 3); fill(1000.0, 30); fill(2500.0, 30)], reduce(hcat, [fill([0.0, 0.0, 0.0], 3); collect.(g); collect.(g)]))
acq3k = diffusionencoding([fill(0.0, 3); fill(3000.0, 30); fill(5000.0, 30)], reduce(hcat, [fill([0.0, 0.0, 0.0], 3); collect.(g); collect.(g)]))
base = (S0 = 1.0, vic = 0.6, viso = 0.1, theta = 0.7, phi = 0.3, Dpar = 1.7e-3, Diso = 3e-3)
models = [("built-in (16 nodes, I0 and I1)", NODDI), ("1. I0 alone", NODDI_I0), ("2. 12 nodes", NODDI_12), ("3. series, l ≤ $LMAX", NODDI_SH), ("3b. series, l ≤ $LMAX, tabulated", NODDI_SHFAST)]
println("| option | largest relative error (b ≤ 2500; ≤ 5000) | one signal, plain | one voxel's fit (1 thread) | dataset fit, 1000 voxels, SNR 30, $(Threads.nthreads()) threads | largest |ΔODI|, |Δvic| against the built-in |")
println("|---|---|---:|---:|---:|---|")
rng = MersenneTwister(1)
nv = 1000
dirs = [normalize(SVector{3}(randn(rng, 3))) for _ in 1:nv]
truth = (S0 = fill(1000.0, nv), vic = fill(0.6, nv), viso = fill(0.05, nv), ODI = 0.05 .+ 0.6 .* rand(rng, nv),
         theta = acos.(getindex.(dirs, 3)), phi = atan.(getindex.(dirs, 2), getindex.(dirs, 1)), Dpar = fill(1.7e-3, nv), Diso = fill(3e-3, nv))
Y = simulate(NODDI(; acq...), truth; snr = 30, noise = :rician, rng)
ref = fit(NODDI(; acq...), Y)
for (label, M) in models
    errs = map((acq, acq3k)) do a
        e = 0.0
        for odi in (EQUIP._ODI_MIN, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0), th in (0.0, 0.5, 1.2), ph in (0.0, 1.0)
            p = merge(base, (ODI = odi, theta = th, phi = ph))
            e = max(e, maximum(abs.(signal(M(; a...), p) ./ signal(NODDI(; a...), p) .- 1)))
        end
        e
    end
    m = M(; acq...); p = merge(base, (ODI = 0.2,))
    out = zeros(63)
    ts = timeit(() -> signal!(out, m, p), 2000)
    c = init(m); y = signal(m, merge(p, (S0 = 1000.0,))) .* 1.01; fit!(c, y)
    tf = timeit(() -> fit!(c, y), 200)
    r = fit(m, Y); td = minimum(@elapsed(fit(m, Y)) for _ in 1:2)
    @printf("| %s | %.1e; %.1e | %.1f µs | %.0f µs | %.0f µs/voxel | %.1e, %.1e |\n", label, errs[1], errs[2], 1e-3ts, 1e-3tf, 1e6td / nv,
            maximum(abs.(r.params.ODI .- ref.params.ODI)), maximum(abs.(r.params.vic .- ref.params.vic)))
    flush(stdout)
end
