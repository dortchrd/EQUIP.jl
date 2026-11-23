# =================================================================================================
# IVIM: intravoxel incoherent motion (Le Bihan et al., Radiology 1988;168:497)
#
#   S = S0·[f·exp(−b·D*) + (1 − f)·exp(−b·D)]
#
# Two compartments of one voxel: blood moving in randomly oriented capillaries, a pseudo-
# diffusion with D* about ten times D or more, and tissue water, diffusing with D. At b above a
# few hundred s/mm² the perfusion term has decayed, so the signal is (1 − f)·S0·exp(−b·D) there;
# f is the perfusion fraction. No direction: the b-values only, as ADCDecay.
#
# Poorly conditioned at usual SNR (D* very uncertain and correlated with f), which crlb and
# montecarlo show for a protocol before it is scanned; b-values for IVIM are designed in Lemke
# et al., Magn Reson Imaging 2011;29:766.
#
# Each voxel starts from the segmented fit, the published method (e.g. Le Bihan 1988; Pekar et
# al., Magn Reson Med 1992), as `datastart`:
#   1. D from the images at b ≥ bsplit (200 s/mm² by default): the log-linear fit of ln y on b,
#      weighted by y², where the perfusion term has gone;
#   2. f from that fit's intercept A against the signal at the lowest b, S(b₀): f = 1 − A/S(b₀);
#   3. D* from the images below bsplit: what is left after the tissue term, y − A·exp(−b·D), is
#      the perfusion term, (S(b₀) − A)·exp(−b·D*); its log-linear fit through b₀.
# A step the data do not support (fewer than two b-values above the split, nothing left below
# it) falls back to the parameters' typical values (D 1e-3, f 0.1, D* 10·D).
#
# The components can swap (f, D*, D) ↔ (1 − f, D, D*): `relabel` reports the faster one as D*.
# =================================================================================================

"""
    IVIM(; b, bsplit = 200)

Intravoxel incoherent motion (Le Bihan et al., Radiology 1988;168:497): perfusion as a fast
pseudo-diffusion and tissue water's diffusion,

    S = S0·[f·exp(−b·D*) + (1 − f)·exp(−b·D)]

- acquisition: `b` (s/mm²), the b-values, several below 200 s/mm² (where the perfusion term
  decays) and several above. `bsplit` (s/mm², 200): where the start's segmented fit (below)
  divides them
- free by default: `S0`; `f`, the perfusion fraction (0 to 1); `Dstar`, the pseudo-diffusion
  coefficient D* (mm²/s, 0 to 0.2); `D`, the tissue's diffusivity (mm²/s, 0 to 4e-3). The
  faster component is reported as D* (`relabel`)

Each voxel starts from the segmented fit (`f`, `Dstar` and `D` start at `:data`): D from the
images at b ≥ `bsplit`, f from that fit's intercept, D* from what is left at the lower b-values.
The simplified IVIM, D* fixed: `fit(m, data; Dstar = 0.02)`; D* poorly determined at usual SNR:
see its crlb.

```julia
res = fit(IVIM(b = [0, 10, 20, 50, 100, 200, 400, 600, 800]), data; mask)
```
"""
@kwdef struct IVIM{A<:AbstractVector,B<:Real} <: MRIModel
    b::A                                # b-values (s/mm²)
    bsplit::B = 200.0                   # the start's split: D from b ≥ bsplit (s/mm²)
end

parameters(::Type{<:IVIM}) = (
    S0 = Free(start = :scale, lower = 0),
    f = Free(start = :data, lower = 0, upper = 1),
    Dstar = Free(start = :data, lower = 0, upper = 0.2),    # pseudo-diffusion, mm²/s
    D = Free(start = :data, lower = 0, upper = _DMAX),
)

function signal!(out, m::IVIM, p)
    (; b) = m
    (; S0, f, Dstar, D) = p
    @inbounds for j in eachindex(out, b)
        out[j] = S0 * (f * iso(b[j], Dstar) + (1 - f) * iso(b[j], D))
    end
    return out
end

# The faster component is D*. Only when all three are free: a fixed one pins the labels.
function relabel(::IVIM, θ)
    haskey(θ, :f) && haskey(θ, :Dstar) && haskey(θ, :D) || return θ
    θ.Dstar >= θ.D && return θ
    return merge(θ, (f = 1 - θ.f, Dstar = θ.D, D = θ.Dstar))
end

# The segmented fit (the header): sums over the images, no allocation.
function datastart(m::IVIM, y, p)
    (; b, bsplit) = m
    T = eltype(y)
    D0, f0 = T(1e-3), T(0.1)                       # the fallbacks: typical tissue
    # the signal at the lowest b-value, S(b₀) (the mean of its images)
    b0 = T(minimum(b))
    s0, n0 = zero(T), 0
    @inbounds for j in eachindex(y, b)
        b[j] == b0 && (s0 += y[j]; n0 += 1)
    end
    s0 /= n0
    # 1. D: ln y = ln A − b·D at b ≥ bsplit, weighted by y²
    sw = sb = sz = sbb = sbz = zero(T)
    @inbounds for j in eachindex(y, b)
        (b[j] >= bsplit && y[j] > 0) || continue
        w, bj, z = y[j]^2, T(b[j]), log(y[j])
        sw += w; sb += w * bj; sz += w * z; sbb += w * bj^2; sbz += w * bj * z
    end
    det = sw * sbb - sb^2
    det > eps(T) * sw * sbb || return (f = f0, D = D0, Dstar = 10D0)   # under two b-values
    D = -(sw * sbz - sb * sz) / det
    A = exp((sz + D * sb) / sw)
    (isfinite(D) && D > 0) || (D = D0)
    # 2. f from the intercept
    f = s0 > 0 ? 1 - A / s0 : f0
    (isfinite(f) && 0 < f < 1) || return (f = f0, D = D, Dstar = 10D)
    # 3. D*: ln[(y − A·e^(−b·D))/(S(b₀) − A)] = −(b − b₀)·D* below bsplit, through b₀
    s, ss = zero(T), zero(T)
    @inbounds for j in eachindex(y, b)
        (b0 < b[j] < bsplit) || continue
        r = (y[j] - A * exp(-T(b[j]) * D)) / (s0 - A)
        r > 0 || continue
        Δ = T(b[j]) - b0
        s += Δ * log(r); ss += Δ^2
    end
    Dstar = ss > 0 ? -s / ss : 10D
    (isfinite(Dstar) && Dstar > D) || (Dstar = 10D)
    return (f = f, D = D, Dstar = Dstar)
end
