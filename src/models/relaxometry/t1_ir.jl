# =================================================================================================
# Inversion recovery
#
#   T1IR, R1IR       long repetition time: S = M0·[Sf·exp(-TI/T1) + 1 - exp(-TI/T1)]
#   T1IRTD, R1IRTD   any delay TD between readout and inversion, modelled explicitly
#                    (Li et al., Magn Reson Med 2010;64:491, Eq. 1)
#
# All four share the parameters M0, Sf (inversion efficiency, -1 = perfect) and T1 or R1: T1IR is
# T1IRTD with full recovery before each inversion (Mz(TD) = 1), so its Sf means the same.
#
# Fit the quantity you will report, so its standard errors and CRLB come out directly. The
# equations are written once, in the R1 versions; the T1 versions convert and call them (see
# builtin.jl).
#
# Magnitude data has local minima: a fit can settle with the signal null in the wrong place.
# From a single start (T1 = 1 s, R1 = 1/s), 200 voxels at SNR 100, both forms failed in most or
# all voxels when T1 was very short or very long compared with the TIs (T1IR at 0.05 and 4 s,
# T1IRTD at 0.05-0.1 s and about half at 4 s; the T1 form also at T1IR 0.1 s and T1IRTD
# 0.2-0.3 s). So every voxel first tries 30 values of T1 spanning the TIs and starts from the
# best (`T1 = Free(start = m -> BestOf(_t1grid(m.TI)), ...)` in `parameters`: a function of the
# model, as the grid depends on its TIs, computed once per fit; `_t1grid` is in
# builtin.jl: log-spaced from 1/5 of the shortest TI to 5× the longest). Tested with T1 from 0.05 to 4 s at
# SNR 20 and 100, both forms: at most 2 of 200 voxels failed in any case, for about 1 µs more
# per voxel. The R1 forms try the same values as rates, so both forms start from the same place.
# Another start for T1 (or R1) in the call replaces it: `T1 = Free(start = 0.2)`, a start map, or
# other values, `T1 = Free(start = BestOf(...))`.
#
# Long T1, low SNR (T1IR, T1 = 4 s, TIs up to 3.2 s, SNR 20, 2000 voxels): the two forms agree
# except in 15 voxels whose noisy data are best fitted with T1 running off to infinity (an almost
# straight line); both forms stop there with status MaxIterations. Without the grid the forms
# disagreed in 371 voxels, each stuck in a different local minimum.
# =================================================================================================


"""
    T1IR(; TI, magnitude = true)

Inversion recovery with a long repetition time, so Mz has fully recovered before each
inversion:

    S = M0·[Sf·exp(-TI/T1) + 1 - exp(-TI/T1)] = M0·(1 - (1 - Sf)·exp(-TI/T1))

its absolute value for magnitude data. `Sf` is the inversion efficiency (-1 = perfect, the
familiar `M0·(1 - 2·exp(-TI/T1))`), as in [`T1IRTD`](@ref), which this is for a long TD.

- acquisition: `TI` (s), one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- parameters: `M0`, `Sf` (from -1.2 to 0), `T1` (s)

Rate version: [`R1IR`](@ref).
"""
@kwdef struct T1IR{A<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    magnitude::Bool = true              # a default value makes the keyword optional
end

parameters(::Type{<:T1IR}) = (
    M0 = Free(start = :scale, lower = 0),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),  # as in T1IRTD
    T1 = Free(start = m -> BestOf(_t1grid(m.TI)), lower = _TMIN),  # the best of 30 (see above)
)

# The T1 form is the R1 form with R1 = 1/T1: convert, then R1IR's signal (below; see builtin.jl
# for why the time form calls the rate form).
signal!(out, m::T1IR, p) = signal!(out, R1IR(m.TI, m.magnitude), (; p.M0, p.Sf, R1 = 1 / p.T1))

"""
    R1IR(; TI, magnitude = true)

[`T1IR`](@ref) with the rate R1 = 1/T1: `S = M0·[Sf·exp(-R1·TI) + 1 - exp(-R1·TI)]`.

- acquisition: `TI` (s), one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- parameters: `M0`, `Sf` (from -1.2 to 0), `R1` (1/s)
"""
@kwdef struct R1IR{A<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    magnitude::Bool = true
end

parameters(::Type{<:R1IR}) = (
    M0 = Free(start = :scale, lower = 0),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    R1 = Free(start = m -> BestOf(1 ./ _t1grid(m.TI)), lower = 0),  # T1IR's values, as rates
)

function signal!(out, m::R1IR, p)
    (; TI, magnitude) = m
    (; M0, Sf, R1) = p
    @inbounds for j in eachindex(out, TI)
        E_ti = exp(-R1 * TI[j])
        s = M0 * (Sf * E_ti + 1 - E_ti)     # R1IRTD's equation with Mz(TD) = 1 (full recovery)
        out[j] = magnitude ? abs(s) : s     # |S| for magnitude images
    end
    return out
end

# -------------------------------------------------------------------------------------------------
"""
    T1IRTD(; TI, TD, magnitude = true)

Inversion recovery with the delay `TD` between the readout and the next inversion modelled
explicitly, so TD can be short and can vary (Li et al., Magn Reson Med 2010;64:491, Eq. 1):

    S = M0·[Sf·Mz(TD)·exp(-TI/T1) + 1 - exp(-TI/T1)]
    Mz(TD) = 1 - (1 - Mz_td0)·exp(-TD/T1)

The readout (e.g. a spin echo or fast spin echo) is assumed to leave Mz = 0; Mz recovers for TD,
the inversion pulse multiplies it by Sf (-1 = perfect inversion), and it recovers for TI. With
Sf = -1 this is the familiar `M0·(1 - 2·exp(-TI/T1) + exp(-(TI+TD)/T1))`. For a long TD it
reduces to [`T1IR`](@ref), with the same parameters.

`Mz_td0` (an extension of Li's equation, which is the case `Mz_td0 = 0`) is Mz at the start of
TD as a fraction of M0. Use it to test the assumption that the readout saturates Mz, as
`Mzf_td0` does for [`QMTSIR`](@ref).

- acquisition: `TI` (s), one per observation; `TD` (s), one number for all observations or one
  per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- free by default: `M0`, `Sf` (inversion efficiency, -1 = perfect), `T1` (s)
- fixed by default: `Mz_td0 = 0`

Rate version: [`R1IRTD`](@ref).

```julia
m = T1IRTD(TI = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4], TD = [0.5, 3.0, 1.0, 0.2, 2.0, 0.05])
m = T1IRTD(TI = [0.03, 0.1, 0.3, 0.6, 1.2, 2.4], TD = 1.0)     # one TD for all
```
"""
@kwdef struct T1IRTD{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
end

T1IRTD(TI, TD::Real, magnitude) = T1IRTD(TI, fill(TD, length(TI)), magnitude)   # one TD for all

parameters(::Type{<:T1IRTD}) = (
    M0 = Free(start = :scale, lower = 0),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    T1 = Free(start = m -> BestOf(_t1grid(m.TI)), lower = _TMIN),
    Mz_td0 = Fixed(0.0),                # Mz at the start of TD, as a fraction of M0: 0 = saturated
)

# The T1 form: R1 = 1/T1, then R1IRTD's signal (below).
signal!(out, m::T1IRTD, p) =
    signal!(out, R1IRTD(m.TI, m.TD, m.magnitude), (; p.M0, p.Sf, R1 = 1 / p.T1, p.Mz_td0))

"""
    R1IRTD(; TI, TD, magnitude = true)

[`T1IRTD`](@ref) with the rate R1 = 1/T1:

    S = M0·[Sf·Mz(TD)·exp(-R1·TI) + 1 - exp(-R1·TI)]
    Mz(TD) = 1 - (1 - Mz_td0)·exp(-R1·TD)

- acquisition: `TI` (s), one per observation; `TD` (s), one number or one per observation
- option: `magnitude` (true: fit |S| for magnitude images; false: signed S)
- free by default: `M0`, `Sf`, `R1` (1/s)
- fixed by default: `Mz_td0 = 0`
"""
@kwdef struct R1IRTD{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A                               # inversion times (s)
    TD::B                               # delays between readout and inversion (s)
    magnitude::Bool = true
end

R1IRTD(TI, TD::Real, magnitude) = R1IRTD(TI, fill(TD, length(TI)), magnitude)   # one TD for all

parameters(::Type{<:R1IRTD}) = (
    M0 = Free(start = :scale, lower = 0),
    Sf = Free(start = -0.95, lower = -1.2, upper = 0),
    R1 = Free(start = m -> BestOf(1 ./ _t1grid(m.TI)), lower = 0),
    Mz_td0 = Fixed(0.0),
)

function signal!(out, m::R1IRTD, p)
    (; TI, TD, magnitude) = m
    (; M0, Sf, R1, Mz_td0) = p
    @inbounds for j in eachindex(out, TI, TD)
        Mz_td = 1 - (1 - Mz_td0) * exp(-R1 * TD[j])        # Mz at the end of TD (fraction of M0)
        E_ti = exp(-R1 * TI[j])
        s = M0 * (Sf * Mz_td * E_ti + 1 - E_ti)             # Li 2010, Eq. 1
        out[j] = magnitude ? abs(s) : s
    end
    return out
end

