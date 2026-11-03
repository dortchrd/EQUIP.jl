# =================================================================================================
# Orientational sampling: the error of a spherical mean taken over a finite set of directions
#
#   directionset(N)                        N directions spread evenly (electrostatic repulsion)
#   orientationcovariance(m, θ, sets)      the covariance, across shells, of the error of the
#                                          shells' means over their direction sets, for one fibre
#                                          population in a random orientation
#
# The spherical mean is an integral over all directions; the data give its average over N. For
# one population of fibres along n, the signal of observation j at gradient direction g depends
# only on t = n·g, so it expands in Legendre polynomials (even orders: g and −g are the same):
#
#   f_j(t) = Σ_l a_l(j)·P_l(t),   a_0(j) the spherical mean (what SMT fits)
#
# and the average over a set G_j of N_j directions is a_0(j) plus an error
#
#   e_j(n) = Σ_{l≥2} a_l(j) · (1/N_j) Σ_{g∈G_j} P_l(n·g).
#
# Over orientations n spread uniformly, e_j has mean 0 and, by the addition theorem
# (the mean over n of P_l(n·g)·P_l(n·h) is P_l(g·h)/(2l+1)), the covariance
#
#   C_jk = Σ_{l≥2} a_l(j)·a_l(k)/(2l+1) · (1/(N_j N_k)) Σ_{g∈G_j} Σ_{h∈G_k} P_l(g·h)
#
# exact, from the tissue (the a_l) and the direction sets alone (the double sums, which a set
# that averages order l exactly makes 0). Shells see the same fibre, so they are correlated.
# One aligned population is the worst case (a peripheral nerve); dispersion or crossing
# populations shrink the a_l. Added to the noise's covariance, it makes crlb's SD the error of
# the fitted parameters over voxels with fibres in any orientation (docs/design.md, "Peripheral
# nerve SMT").
# =================================================================================================

"""
    directionset(N; iterations = 2000) -> Vector{SVector{3,Float64}}

`N` gradient directions spread evenly over the sphere, as unit vectors with z ≥ 0 (a direction
and its opposite are the same for diffusion): electrostatic repulsion between the directions and
their opposites (Jones et al., Magn Reson Med 1999;42:515), as MRtrix's `dirgen` makes them.
From a Fibonacci spiral, so the same `N` always gives the same set.
"""
function directionset(N::Integer; iterations::Integer = 2000)
    N >= 1 || throw(ArgumentError("directionset: N ≥ 1"))
    g = [_fibonacci(k, N) for k in 0:N-1]
    N == 1 && return g
    spacing = sqrt(2π / N)                       # about the distance between neighbours
    F = similar(g)
    for it in 1:iterations
        maxf = 0.0
        for i in 1:N
            f = zero(SVector{3,Float64})
            for j in 1:N
                j == i && continue
                for h in (g[j], -g[j])          # each direction and its opposite repel
                    d = g[i] - h
                    f += d / norm(d)^3
                end
            end
            f -= dot(f, g[i]) * g[i]             # along the sphere
            F[i] = f
            maxf = max(maxf, norm(f))
        end
        step = 0.05 * spacing * (1 - (it - 1) / iterations) / maxf   # shrinking steps
        for i in 1:N
            g[i] = normalize(g[i] + step * F[i])
        end
    end
    return [x[3] < 0 ? -x : x for x in g]
end
function _fibonacci(k, N)                        # the k-th of N points over the upper half
    z = 1 - (k + 0.5) / N
    r = sqrt(max(0.0, 1 - z^2))
    φ = π * (3 - sqrt(5)) * k
    return SVector(r * cos(φ), r * sin(φ), z)
end

# The signal of observation j for one fibre population at cos t to the gradient: what the
# spherical mean averages. Its average over t ∈ [0, 1] is the model's own signal. The same tissue
# equation as `signal!` (smt.jl), with the compartments in their directional view: one function
# for every SMT model and every part switched on (free water, T2s, the extra-axonal diffusivities).
_profile(m::AbstractSMT, p, j, t) =
    _smt(p, 1 / p.T2ax, 1 / p.T2ex, 1 / p.T2iso, m.b[j], _optobs(m.TE, j), t)

# Gauss–Legendre nodes and weights on [0, 1] (n of them: exact for polynomials of degree 2n − 1),
# by Newton's method on P_n from the Chebyshev guesses.
function _gausslegendre01(n)
    x, w = zeros(n), zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        for _ in 1:100
            p0, p1 = 1.0, z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = n * (z * p1 - p0) / (z^2 - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        p0, p1 = 1.0, z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = n * (z * p1 - p0) / (z^2 - 1)
        x[i] = (1 + z) / 2                       # [−1, 1] → [0, 1]
        w[i] = 1 / ((1 - z^2) * dp^2)            # 2/((1 − z²)P'²), halved for [0, 1]
    end
    return x, w
end
const _GL01 = _gausslegendre01(120)

# P_0(x), ..., P_L(x), by the three-term recurrence.
function _legendre!(P, x)
    P[1] = 1.0
    length(P) > 1 && (P[2] = x)
    for l in 2:length(P)-1
        P[l+1] = ((2l - 1) * x * P[l] - (l - 1) * P[l-1]) / l
    end
    return P
end

# The even Legendre coefficients a_0, a_2, ..., a_L of observation j's profile:
# a_l = (2l + 1)·∫₀¹ f(t)·P_l(t) dt (f even in t). With σ, of the magnitude images' mean in each
# direction, ricianmean(f(t), σ): the Rician floor acts direction by direction, before the
# spherical mean, so on the profile, not on its average.
function _legendrecoefs(m, p, j, L, σ = nothing)
    x, w = _GL01
    a = zeros(L ÷ 2 + 1)
    P = zeros(L + 1)
    for (t, wt) in zip(x, w)
        f = σ === nothing ? _profile(m, p, j, t) : ricianmean(abs(_profile(m, p, j, t)), σ)
        _legendre!(P, t)
        for (k, l) in enumerate(0:2:L)
            a[k] += (2l + 1) * wt * f * P[l+1]
        end
    end
    return a
end

"""
    orientationcovariance(m, θ, sets; L = 60) -> Matrix{Float64}

The covariance, across observations, of the error a finite direction set makes in a spherical
mean: one fibre population in an orientation spread uniformly over the sphere, `sets[j]` the
gradient directions averaged for observation j (unit 3-vectors; `directionset(N)` for evenly
spread ones; a single direction or an empty set for a b = 0 shell, which has no error). For the
SMT models (`SMT`, `SMTFreeWater`, `SMTT2`), at truth `θ` (every parameter's value), in signal
units squared:

    C_jk = Σ_{l=2,4,…,L} a_l(j)·a_l(k)/(2l+1) · (1/(N_j N_k)) Σ_{g∈G_j} Σ_{h∈G_k} P_l(g·h)

with a_l(j) the Legendre coefficients of observation j's signal as a function of the cosine
between fibre and gradient (exact: the addition theorem; header of src/models/diffusion/
orientation.jl). Its mean over orientations is 0: the error shows as spread, not bias, over
voxels with fibres in different orientations. One aligned population (a nerve) is the worst
case.

Added to the noise's covariance, crlb gives the fitted parameters' SD over orientations and
noise together:

```julia
dirs = [10, 30, 60]
sets = directionset.(dirs)
kw = riciannoise(m, θ; snr = 20, averages = dirs, averaging = :magnitude)
crlb(m, θ; simulate_with = kw.simulate_with, Σ = kw.Σ + orientationcovariance(m, θ, sets),
     weights = dirs)
```

`sqrt.(diag(C)) ./ signal(m, θ)` is each shell's relative SD over orientations, the relative SD
of Schilling et al. (Magn Reson Imaging 2022) without noise. With `σ` (one image's noise SD, a
number or one per observation), the covariance of the magnitude images' means instead (the
Rician mean of each direction's signal, the profile the data actually average):
[`sphericalmeannoise`](@ref) uses it.
"""
function orientationcovariance(m::AbstractSMT, θ::NamedTuple, sets;
                               σ = nothing, L::Integer = 60)
    n = nobs(m)
    length(sets) == n || throw(DimensionMismatch("orientationcovariance: $(length(sets)) direction " *
                                                 "sets for $n observations"))
    p = resolve(m, θ)
    Le = 2 * (L ÷ 2)
    σv = σ === nothing ? fill(nothing, n) : σ isa Real ? fill(Float64(σ), n) : Float64.(collect(σ))
    a = [_legendrecoefs(m, p, j, Le, σv[j]) for j in 1:n]
    G = [[normalize(SVector{3,Float64}(g...)) for g in s] for s in sets]
    C = zeros(n, n)
    P = zeros(Le + 1)
    for j in 1:n, k in j:n
        (isempty(G[j]) || isempty(G[k])) && continue
        # Σ over the two sets of P_l(g·h), every even l ≥ 2 at once
        S = zeros(Le ÷ 2 + 1)
        for g in G[j], h in G[k]
            _legendre!(P, clamp(dot(g, h), -1.0, 1.0))
            for (q, l) in enumerate(0:2:Le)
                S[q] += P[l+1]
            end
        end
        c = 0.0
        for (q, l) in enumerate(0:2:Le)
            l >= 2 || continue
            c += a[j][q] * a[k][q] / (2l + 1) * S[q] / (length(G[j]) * length(G[k]))
        end
        C[j, k] = C[k, j] = c
    end
    return C
end

# -------------------------------------------------------------------------------------------------
"""
    RicianSphericalMean(m; σ)

A model whose signal is the expected spherical mean of magnitude images: for each observation
(shell), the mean over all fibre–gradient angles of [`ricianmean`](@ref) of the signal in that
direction, ∫₀¹ E|y|(f_j(t)) dt, for one fibre population (one of the SMT models). Unlike
[`RicianMean`](@ref) of the spherical mean, the floor acts direction by direction: directions
near the fibres, where the signal is small at high b, sit on the floor, and since E|y| is convex
in the signal the spherical mean of magnitudes lies above the floor of the mean (2026-11-03:
v_ax's bias 0.156 measured in a simulated pipeline, against 0.033 from the floor of the mean).
`σ` is one image's noise SD after the scanner's (complex) averages, a number or one per
observation. The integral by Gauss–Legendre quadrature, 120 nodes.
"""
struct RicianSphericalMean{M<:AbstractSMT,S<:AbstractVector} <: MRIModel
    model::M
    σ::S
end
function RicianSphericalMean(m::AbstractSMT; σ)
    n = nobs(m)
    σv = σ isa Real ? fill(Float64(σ), n) : Float64.(collect(σ))
    length(σv) == n || throw(DimensionMismatch("RicianSphericalMean: $(length(σv)) σ for $n observations"))
    return RicianSphericalMean(m, σv)
end
parameters(w::RicianSphericalMean) = parameters(w.model)
nobs(w::RicianSphericalMean) = nobs(w.model)
_with_precision(w::RicianSphericalMean, ::Type{T}) where {T} =
    RicianSphericalMean(_with_precision(w.model, T), Vector{T}(w.σ))
function signal!(out, w::RicianSphericalMean, p)
    x, wt = _GL01
    for j in eachindex(out, w.σ)
        s = zero(eltype(out))
        for (t, ω) in zip(x, wt)
            s += ω * ricianmean(abs(_profile(w.model, p, j, t)), w.σ[j])
        end
        out[j] = s
    end
    return out
end

"""
    sphericalmeannoise(m, θ; snr or σ, directions, averages = 1, sets = directionset.(directions))
        -> (; simulate_with, Σ)

crlb's keywords for the whole spherical-mean pipeline of an SMT model `m` at truth `θ`: magnitude
images, one per gradient direction (`directions[j]` of them for shell j, the directions `sets`),
each the complex mean of `averages[j]` acquisitions (the scanner's NEX: σ/√NEX), averaged into
the shell's spherical mean, for one fibre population in a random orientation. Every error the
pipeline has, carried to the fit by crlb:

- the Rician floor, direction by direction (`simulate_with`: [`RicianSphericalMean`](@ref), the
  data's mean over orientations): the bias;
- the noise, each direction's Rician variance averaged over the angles, over the number of
  directions (`Σ`'s diagonal);
- the orientational sampling of a finite direction set, on the magnitude images' means
  ([`orientationcovariance`](@ref) with σ; `Σ`, across shells): spread over orientations.

`snr` is one acquisition's (one direction, one average), relative to S0, as for `simulate`.

```julia
m = SMT(b = [0, 1000, 2500, 4000])
dirs, nex = [4, 30, 30, 30], [1, 2, 2, 2]
crlb(m, θ; sphericalmeannoise(m, θ; snr = 20, directions = dirs, averages = nex)...,
     weights = dirs .* nex)
```
"""
function sphericalmeannoise(m::AbstractSMT, θ::NamedTuple; snr = nothing,
                            σ = nothing, directions, averages = 1,
                            sets = directionset.(directions isa Integer ? fill(directions, nobs(m)) : directions))
    n = nobs(m)
    dirs = directions isa Integer ? fill(directions, n) : collect(directions)
    nex = averages isa Integer ? fill(averages, n) : collect(averages)
    (length(dirs) == n && length(nex) == n && length(sets) == n) || throw(DimensionMismatch(
        "sphericalmeannoise: give `directions` and `averages` one per observation ($n), or one for all"))
    σ1 = sqrt.(diag(noisecovariance(m, θ; snr, σ)))             # one acquisition's σ
    σc = σ1 ./ sqrt.(nex)                                        # after the scanner's averages
    p = resolve(m, θ)
    x, wt = _GL01
    v = [sum(ω * ricianvar(abs(_profile(m, p, j, t)), σc[j]) for (t, ω) in zip(x, wt)) / dirs[j]
         for j in 1:n]
    C = orientationcovariance(m, θ, sets; σ = σc)
    return (simulate_with = RicianSphericalMean(m; σ = σc) => θ, Σ = Diagonal(v) + C)
end
