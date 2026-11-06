# =================================================================================================
# SMT: the spherical mean technique (Kaden et al., NeuroImage 2016;139:346), in the form used for
# multiple sclerosis imaging by Devan et al., Magn Reson Imaging 2020;74:56 (its protocols and
# tissue values are below), with free water and compartment T2s as parts to switch on
#
# Averaging the diffusion-weighted signal over the gradient directions of a shell (its spherical
# mean) removes the fiber orientation distribution: crossing, dispersion and undulation drop out,
# and what is left depends on the diffusion within one fiber and around it. Per shell b and echo
# time TE, from the building blocks (src/models/blocks.jl):
#
#   S̄(b, TE) = S0·[(1 − fiso)·(v_ax·E_ax·S̄_stick(b, D_ax) + (1 − v_ax)·E_ex·S̄_zeppelin(b, D∥ex, D⊥ex))
#                  + fiso·E_iso·exp(−b·Diso)],       E_c = exp(−TE/T2_c)
#   S̄_stick(b, D)            = √π·erf(√(b·D)) / (2√(b·D))       intra-axonal: sticks (D⊥,in = 0)
#   S̄_zeppelin(b, D∥, D⊥)    = exp(−b·D⊥)·S̄_stick(b, D∥ − D⊥)  extra-axonal
#
# Kaden's two-compartment model, Devan's SMT, is this with the parts off: no free water (fiso = 0),
# no T2 weighting (T2ax = T2ex = T2iso = Inf), and the extra-axonal diffusivities tied to D_ax:
# D∥ex = D_ax (the same parallel diffusivity) and D⊥ex = (1 − v_ax)·D_ax (the tortuosity limit).
# So the parameters free by default are S0, v_ax (the apparent axonal volume fraction) and D_ax.
# Each part switches on in a call: `fiso = Free()` (free water, as SMTFreeWater), `T2ax = Free(),
# T2ex = Free()` with spherical means at several TEs (as SMTT2), `Dexperp = Free()` (no
# tortuosity limit), `Dexpar = Free()` (another parallel diffusivity outside the axons).
#
# The spherical means come from the data first: `sphericalmean` averages the volumes of each
# shell (b-values within 100 s/mm² of each other are one shell, as in Devan 2020), and the model
# is fitted to one value per shell. A shell's mean over N directions has noise σ/√N: give
# `averages = directions` to crlb and montecarlo. (A mean of magnitude images keeps the Rician
# floor: in montecarlo, `averaging = :magnitude`; per direction, `sphericalmeannoise`.)
#
# The same tissue equation gives the signal along one direction (`_profile`, in orientation.jl):
# the compartments in their directional view, at cosine ξ between fiber and gradient.
# =================================================================================================

"""
    AbstractSMT

The SMT models: [`SMT`](@ref), [`SMTFreeWater`](@ref), [`SMTT2`](@ref). One signal (free water,
compartment T2s and the extra-axonal diffusivities as parts, src/models/diffusion/smt.jl) and the
same parameters: `S0`, `vax`, `Dax`, `fiso`, `Diso`, `T2ax`, `T2ex`, `T2iso`, `Dexpar`,
`Dexperp`; the models differ in which are on by default, and any part switches on or off in a
call (`fit(SMT(b = b), y; fiso = Free())`).
"""
abstract type AbstractSMT <: MRIModel end

# Every part, with what is on for Kaden's two-compartment SMT; the named models change some
# roles. The order keeps each model's free parameters in the order they had (S0, vax, Dax; with
# fiso; with T2ax, T2ex).
function _smt_parts()
    return (
        S0 = Free(start = :scale, lower = 0),
        vax = Free(start = 0.5, lower = 0, upper = 1),
        Dax = Free(start = 2.0e-3, lower = 0, upper = 3.0e-3),                  # mm²/s
        # free water, a ball (NODDI's isotropic compartment): off
        fiso = Fixed(0.0; free = Free(start = 0.1, lower = 0, upper = 1)),
        Diso = Fixed(3.0e-3; free = Free(start = 3.0e-3, lower = 0)),           # free water at 37 °C
        # each compartment's T2 (s): Inf, no T2 weighting
        T2ax = Fixed(Inf; free = Free(start = 0.08, lower = _TMIN, upper = 1)),
        T2ex = Fixed(Inf; free = Free(start = 0.05, lower = _TMIN, upper = 1)),
        T2iso = Fixed(Inf; free = Free(start = 1.0, lower = _TMIN)),
        # the extra-axonal diffusivities, tied: D∥ex = D_ax, D⊥ex = (1 − v_ax)·D_ax (tortuosity)
        Dexpar = Fixed(:Dax; free = Free(start = 2.0e-3, lower = 0, upper = 3.0e-3)),
        Dexperp = Fixed(p -> (1 - p.vax) * p.Dax; free = Free(start = 0.5e-3, lower = 0, upper = 3.0e-3)),
    )
end
# The T2 upper bound of 1 s: white matter's compartments are far below; it keeps a poorly
# determined fit from wandering, as it did to hundreds of seconds at SNR 50 without it.

# The tissue at one observation: b, TE (false without echo times: no T2 weighting), the rates
# R2 = 1/T2 (0 for T2 = Inf), and the view, `powder` for the spherical mean or a cosine
# ξ for one direction. With the parts off: S0·(v_ax·S̄_stick + (1 − v_ax)·S̄_zeppelin), Kaden's.
@inline function _smt(p, R2ax, R2ex, R2iso, b, TE, v)
    (; S0, vax, Dax, fiso, Diso, Dexpar, Dexperp) = p
    ax = vax * (decay(TE, R2ax) * stick(b, Dax, v))                           # intra-axonal: sticks
    ex = (1 - vax) * (decay(TE, R2ex) * zeppelin(b, Dexpar, Dexperp, v))      # extra-axonal: a zeppelin
    iso = _skip(fiso) ? zero(b * Diso * R2iso) :                               # (skipped when off)
          decay(TE, R2iso) * ball(b, Diso)                                     # free water: a ball
    return S0 * ((1 - fiso) * (ax + ex) + fiso * iso)
end

function signal!(out, m::AbstractSMT, p)
    (; b, TE) = m
    _checkobs(out, TE, :TE, m)          # with echo times, one per observation (once, before the loop)
    R2ax, R2ex, R2iso = 1 / p.T2ax, 1 / p.T2ex, 1 / p.T2iso
    @inbounds for j in eachindex(out, b)
        out[j] = _smt(p, R2ax, R2ex, R2iso, b[j], _optobs(TE, j), powder)
    end
    return out
end

# -------------------------------------------------------------------------------------------------
"""
    SMT(; b, TE = nothing)

The spherical mean technique, two compartments (Kaden et al., NeuroImage 2016), as used for
multiple sclerosis imaging by Devan et al. (Magn Reson Imaging 2020;74:56): the spherical mean
of the diffusion signal in a shell, from sticks (intra-axonal, no radial diffusion) and a
zeppelin around them (extra-axonal), with the same parallel diffusivity D_ax and the
extra-axonal radial diffusivity from the tortuosity limit, D⊥,ex = (1 − v_ax)·D_ax:

    S̄(b) = S0·[v_ax·f(b·D_ax) + (1 − v_ax)·exp(−b·D⊥,ex)·f(b·v_ax·D_ax)],
    f(x²) = √π·erf(x)/(2x)

The fibers' orientations drop out of the spherical mean (crossing, dispersion, undulation).

- acquisition: `b` (s/mm²), one per shell: the data are the shells' spherical means
  ([`sphericalmean`](@ref)); a shell at b = 0 gives S0 directly. `TE` (s, one per shell), only
  for the compartment T2s
- free by default: `S0`, `vax` (the apparent axonal volume fraction, 0 to 1), `Dax` (mm²/s,
  0 to 3e-3, about free water's diffusivity at body temperature)
- off by default, each switched on in a call ([`AbstractSMT`](@ref)): free water `fiso` (with
  `Diso`, fixed at 3e-3 mm²/s), the compartments' T2s `T2ax`, `T2ex`, `T2iso` (Inf: no T2
  weighting; they need `TE`), and the extra-axonal diffusivities `Dexpar` (tied to `Dax`) and
  `Dexperp` (tied to (1 − vax)·Dax): `fit(m, y; Dexperp = Free())` drops the tortuosity limit

Protocols: [`SMT_PROTOCOLS`](@ref) (Devan 2020, Table 2); tissue values: `SMT_NWM`, `SMT_NAWM`,
`SMT_T2lesion`, `SMT_cBH` in [`TISSUES`](@ref) (Table 1).

```julia
means, b, directions = sphericalmean(dwi, bvals)            # one value per shell
res = fit(SMT(b = b), means; mask)
crlb(SMT(b = b), nominal(SMT(b = b), :SMT_cBH); snr = 20, averages = directions)
```
"""
@kwdef struct SMT{A<:AbstractVector,E<:Union{Nothing,AbstractVector}} <: AbstractSMT
    b::A                                # the shells' b-values (s/mm²)
    TE::E = nothing                     # echo times (s), one per shell, or none
end
parameters(::Type{<:SMT}) = _smt_parts()

"""
    SMT_PROTOCOLS

The five SMT protocols compared by Devan et al., Magn Reson Imaging 2020;74:56, Table 2, each
about 100 acquisitions: `b` (s/mm², one per shell) and `directions` (gradient directions in
each shell). `A` and `B` were optimized there (`A`, 10 shells of 10 directions, shells within
100 s/mm² merged; `B`, 9 with one fixed at b = 0); `C` is NODDI's optimized protocol (Zhang et
al., NeuroImage 2012), `D` the protocol of their earlier MS studies, `E` the original SMT
protocol of Kaden et al. 2016, shortened to the same scan time.

```julia
P = SMT_PROTOCOLS.A
m = SMT(b = P.b)
crlb(m, nominal(m, :SMT_cBH); snr = 20, averages = P.directions)   # SNR of one direction
```
"""
const SMT_PROTOCOLS = (
    A = (b = [430.0, 990.0, 4230.0], directions = [10, 10, 80]),
    B = (b = [0.0, 530.0, 3790.0], directions = [10, 10, 80]),
    C = (b = [0.0, 711.0, 2855.0], directions = [10, 32, 64]),
    D = (b = [0.0, 1000.0, 2500.0], directions = [10, 45, 45]),
    E = (b = [0.0, 1000.0, 2000.0, 3000.0], directions = [4, 32, 32, 32]),
)

"""
    sphericalmean(data, b; TE = nothing, tol = 100) -> (; means, b, directions, TE)

The spherical mean of diffusion-weighted data: the volumes of each shell averaged, the input of
[`SMT`](@ref) and its variants. `data` has the volumes along its last dimension (any number of
dimensions before, as for `fit`), `b` one b-value per volume (s/mm², e.g. from the `.bval` file:
`info.found[:b]` after [`loaddata`](@ref)). Volumes whose b-values are within `tol` of the lowest
b-value of their shell form one shell (Devan 2020: within 100 s/mm²); b = 0 volumes are a shell
too. With `TE` (s, one per volume: data at several echo times, for [`SMTT2`](@ref)), a shell is
one echo time and one b-value: the volumes are grouped by TE first (TEs within 1 ms are one).

Returns a named tuple, in order: the means (the volumes replaced by one value per shell, by
increasing TE, then b), each shell's b (the mean of its volumes' b-values), its number of
volumes (`directions`: the `averages` for crlb and montecarlo, since a shell's mean has noise
σ/√N, and the fit's `weights`), and each shell's TE (`nothing` without `TE`). The first three
can be taken by position, `means, b, directions = sphericalmean(data, b)`.
"""
function sphericalmean(data::AbstractArray{<:Real}, b::AbstractVector{<:Real};
                       TE::Union{Nothing,AbstractVector{<:Real}} = nothing, tol::Real = 100)
    N = size(data, ndims(data))
    length(b) == N || throw(DimensionMismatch("sphericalmean: $(length(b)) b-values for $N " *
                                              "volumes (the last dimension of the data)"))
    TE === nothing || length(TE) == N || throw(DimensionMismatch("sphericalmean: " *
        "$(length(TE)) echo times for $N volumes"))
    # The echo times' groups (one group without TE), then the b-value shells within each.
    tegroups = TE === nothing ? [collect(1:N)] : _groups(TE, 1e-3, 1:N)
    groups = reduce(vcat, [_groups(b, tol, g) for g in tegroups])
    shells = [sum(b[g]) / length(g) for g in groups]
    directions = length.(groups)
    tes = TE === nothing ? nothing : [sum(TE[g]) / length(g) for g in groups]
    T = float(eltype(data))
    flat = reshape(data, :, N)
    means = Matrix{T}(undef, size(flat, 1), length(groups))
    for (s, g) in enumerate(groups)
        means[:, s] .= vec(sum(view(flat, :, g); dims = 2)) ./ length(g)
    end
    return (means = reshape(means, size(data)[1:end-1]..., length(groups)), b = shells,
            directions = directions, TE = tes)
end

# The indices `idx` grouped by their values x[idx], in increasing order: a new group where a
# value is more than `tol` above its group's lowest.
function _groups(x, tol, idx)
    groups = Vector{Int}[]
    for k in idx[sortperm(x[idx])]
        if isempty(groups) || x[k] - x[first(groups[end])] > tol
            push!(groups, [k])
        else
            push!(groups[end], k)
        end
    end
    return groups
end

# -------------------------------------------------------------------------------------------------
"""
    SMTFreeWater(; b, TE = nothing)

[`SMT`](@ref) with its free-water part on: a third compartment, free water (CSF, or oedema from
inflammation), isotropic with diffusivity `Diso`, as NODDI's (Zhang et al., NeuroImage 2012).
Free water was the largest confound of SMT in Devan et al. 2020 (Fig. 4: with the
two-compartment model, v_ax falls and D_ax rises towards free water's as its fraction grows):

    S̄(b) = S0·[(1 − fiso)·(v_ax·S̄_ax + (1 − v_ax)·S̄_ex) + fiso·exp(−b·Diso)]

with S̄_ax and S̄_ex as in `SMT` (sticks, and a zeppelin with the tortuosity limit).

- acquisition: `b` (s/mm²), one per shell, as for `SMT`
- free by default: `S0`, `vax` (the intra-axonal fraction of the tissue, without the free water:
  Devan's "relative intra-axonal volume fraction"), `Dax` (mm²/s), `fiso` (free water's signal
  fraction, 0 to 1)
- fixed: `Diso`, 3.0e-3 mm²/s (free water at body temperature); `Diso = Free()` estimates it.
  The other parts as for `SMT`, off

Free water makes the fit much less precise: for v_ax 0.45, D_ax 1.8e-3 mm²/s, 20% free water,
b-values 0–4500 s/mm² in 6 shells, at SNR 50 per shell mean, crlb predicts an SD of 0.10 in vax
and 0.29 in fiso (SNR 500: 0.01 and 0.03; low b-values, 100–300, help fiso a little). Check a
protocol with `crlb` before relying on fiso.
"""
@kwdef struct SMTFreeWater{A<:AbstractVector,E<:Union{Nothing,AbstractVector}} <: AbstractSMT
    b::A                                # the shells' b-values (s/mm²)
    TE::E = nothing                     # echo times (s), or none
end
parameters(::Type{<:SMTFreeWater}) = merge(_smt_parts(), (fiso = Free(start = 0.1, lower = 0, upper = 1),))

# -------------------------------------------------------------------------------------------------
"""
    SMTT2(; b, TE)

[`SMT`](@ref) with its compartment T2s on, for spherical means at several echo times
(TE-dependent diffusion imaging: Veraart et al., NeuroImage 2018 (TEdDI); McKinnon & Jensen,
Magn Reson Med 2019, who measured a longer T2 inside axons than outside in healthy white
matter):

    S̄(b, TE) = S0·[v_ax·exp(−TE/T2ax)·S̄_ax(b) + (1 − v_ax)·exp(−TE/T2ex)·S̄_ex(b)]

with S̄_ax and S̄_ex as in `SMT`. `vax` is then the volume (water) fraction itself, and the
tortuosity limit uses it. At one TE, SMT's fraction is T2-weighted instead, and not exactly the
T2-weighted signal fraction vax·E_ax / (vax·E_ax + (1 − vax)·E_ex), E = exp(−TE/T2), since its
tortuosity limit then uses that fraction too: for vax = 0.45, T2 90 ms inside and 50 ms outside,
TE 80 ms, SMT gives 0.66, the T2-weighted fraction is 0.62. Veraart et al. showed that including
the compartments' T2s also removes the coupling between the fraction and D_ax.

- acquisition: `b` (s/mm²) and `TE` (s), one pair per observation (one shell at one echo time:
  [`sphericalmean`](@ref)`(data, b; TE)`); at least two echo times, to tell the T2s apart
- free by default: `S0` (the signal at b = 0 and TE = 0), `vax`, `Dax` (mm²/s), `T2ax`, `T2ex`
  (s, up to 1 s: white matter's compartments are far below; the bound keeps a poorly determined
  fit from wandering, as it did to hundreds of seconds at SNR 50 without it)
- off by default, as for `SMT`: free water (`fiso = Free()`, with its own `T2iso`, Inf unless
  freed or given) and the extra-axonal diffusivities' ties

The T2s need a protocol made for them: with 3 echo times from 70 to 130 ms and 4 b-values, at
SNR 50 per shell mean, crlb predicts an SD of 47% in T2ax and 0.48 in vax (SNR 500: 5% and 0.05).
`ProtocolDesign` finds better ones.
"""
@kwdef struct SMTT2{A<:AbstractVector,E<:AbstractVector} <: AbstractSMT
    b::A                                # b-values (s/mm²)
    TE::E                               # echo times (s)
end
parameters(::Type{<:SMTT2}) = merge(_smt_parts(), (
    T2ax = Free(start = 0.08, lower = _TMIN, upper = 1),      # s
    T2ex = Free(start = 0.05, lower = _TMIN, upper = 1),      # s
))
