# =================================================================================================
# Building blocks: the pieces signal equations are made of (module EQUIP.Blocks)
#
#   relax(t, T)              exp(−t/T)                        relaxation with a time constant
#                                                             (T2 over TE, T1 over TI); T = Inf: 1
#   relaxrate(t, R)          exp(−t·R)                        the same with a rate (R2 = 1/T2); R = 0: 1
#   stick(b, D)              √π·erf(√(bD))/(2√(bD))           a stick's spherical mean (b in s/mm²,
#   stick(b, D, t)           exp(−b·D·t²)                     D in mm²/s); along one direction at
#                                                             cosine t between fibre and gradient
#   zeppelin(b, D∥, D⊥)      exp(−b·D⊥)·stick(b, D∥ − D⊥)     axially symmetric: a stick with D∥ − D⊥
#   zeppelin(b, D∥, D⊥, t)   exp(−b·D⊥)·stick(b, D∥ − D⊥, t)  times the radial attenuation
#   ball(b, D)               exp(−b·D)                        isotropic (free water, an ADC)
#
# For writing models: `using EQUIP.Blocks`, then e.g.
#
#   @mrimodel MyT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05)) = S0 * relax(TE, T2)
#
# (src/models/macro.jl; more in README, "Building blocks", and examples/building_blocks.jl). The
# names are plain, without the leading underscore that marks Julia's internal names, because they
# are meant to be used; they live in this small module so that `stick` and `ball` stay out of the
# namespace of anyone who does not ask for them. EQUIP's own models use them through `using .Blocks`.
#
# A family of models (the T2 decays, SMT and its variants) has ONE signal equation, written from
# these pieces with every part the family can have. A part a model does not use by default is
# switched off by its parameters' values, not by another type: an offset C = 0, a second
# component's fraction fshort = 0, a compartment's T2 = Inf (no T2 weighting). In `parameters`
# such a part is `Fixed(0.0; free = Free(...))`: fixed at its "off" value, with the role it takes
# when a call switches it on (`fit(m, y; C = Free())`: start and bounds from there). A complete set
# of values does not need it (`simulate(m, (S0 = ..., T2 = ...))`): it is filled in at its value.
# The named variants (T2DecayOffset, T2BiexpDecay, SMTFreeWater, SMTT2) are small types of the
# family that differ only in which parts are on by default.
#
# A part switched off changes no number: x·1 = x and x + 0·y = x for finite y, so a model with its
# parts off gives the values a model written without them gives (measured: bit for bit, both
# precisions, docs/design.md, "Building blocks"). Nor does it cost its computation: a part whose
# weight is a plain number at exactly zero (fixed at its off value) is skipped (`_skip`), and a
# relaxation with a rate of exactly zero (T = Inf) is 1 without an exp. A dual number is never
# skipped, at zero too: it is being differentiated (a free parameter, or crlb's and leakage's
# seeded fixed values), and its derivative is needed. The blocks themselves cost nothing that can
# be measured against the equation written out (bench/blocks_cost.jl): the compiler inlines them.
#
# Diffusion compartments: Kaden et al., NeuroImage 2016; Panagiotaki et al., NeuroImage 2012 (the
# taxonomy of compartments). The two views of a compartment are two methods of one function: Julia
# picks the one that matches the arguments (dispatch), when compiling, so neither costs a test at
# run time.
# =================================================================================================

module Blocks

using SpecialFunctions: erf

export relax, relaxrate, stick, zeppelin, ball

"""
    relax(t, T) = exp(−t/T)

Relaxation with a time constant, in the papers' notation: a T2 (or T2*) decay over the echo time,
`relax(TE, T2)`, or a T1 term, `relax(TI, T1)`. `T = Inf` gives 1 (no relaxation), so a
compartment's T2 can be switched off at Inf. Computed as [`relaxrate`](@ref)`(t, 1/T)`: a model
written with `relax(TE, T2)` gives the same numbers, bit for bit, as the built-in T2 decays,
which convert T2 to a rate once per voxel.

```julia
using EQUIP.Blocks
relax(0.08, 0.08)            # e⁻¹
@mrimodel MyT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05)) = S0 * relax(TE, T2)
```
"""
@inline relax(t, T) = relaxrate(t, 1 / T)

"""
    relaxrate(t, R) = exp(−t·R)

Relaxation with a rate: `relaxrate(TE, R2)` with R2 = 1/T2 (`R = 0`: 1, no relaxation, without
computing an exp when R is a plain number). The exponential unit of every decay: with `t = b`
and `R = D` it is diffusion's attenuation, [`ball`](@ref).
"""
@inline relaxrate(t, R) = _skip(R) ? one(t * R) : exp(-t * R)

# A weight (a fraction, a rate) that is a plain number at exactly zero: its part can be skipped. A
# dual number (`ForwardDiff.Dual <: Real`, not an AbstractFloat) never is: the method is chosen by
# type, when compiling.
@inline _skip(x::AbstractFloat) = iszero(x)
@inline _skip(x) = false

# The spherical-mean view, as a value: what the models' tissue functions pass through to the
# compartments when they average over directions (smt.jl). Users write `stick(b, D)`.
struct _SphericalMean end

"""
    stick(b, D)          the spherical mean: √π·erf(√(b·D)) / (2√(b·D))
    stick(b, D, t)       along one gradient direction: exp(−b·D·t²)

A stick: a cylinder of zero radius, diffusion along the fibre only (an axon in SMT, a neurite in
NODDI), b in s/mm², D in mm²/s. Without `t`, the signal averaged over all gradient directions
(the spherical mean, what SMT fits; 1 at b = 0); with `t`, the cosine of the angle between the
fibre and the gradient, along that one direction. The spherical mean is the average of the
directional signal over t ∈ [0, 1] (Kaden et al., NeuroImage 2016).

```julia
using EQUIP.Blocks
stick(1000.0, 2e-3)           # 0.606: the spherical mean at b = 1000 s/mm²
stick(1000.0, 2e-3, 1.0)      # exp(−2): along the fibre
stick(1000.0, 2e-3, 0.0)      # 1: across it
```
"""
@inline stick(b, D) = stick(b, D, _SphericalMean())
@inline stick(b, D, t::Real) = exp(-b * D * t^2)
@inline stick(b, D, ::_SphericalMean) = _sphmean(b * D)

"""
    zeppelin(b, D∥, D⊥)        the spherical mean: exp(−b·D⊥)·stick(b, D∥ − D⊥)
    zeppelin(b, D∥, D⊥, t)     along one direction: exp(−b·D⊥)·exp(−b·(D∥ − D⊥)·t²)

An axially symmetric tensor (the extra-axonal space in SMT and NODDI): a stick with D∥ − D⊥,
times the attenuation that every direction has, exp(−b·D⊥). The two views as for
[`stick`](@ref). With D⊥ = (1 − v_ax)·D_ax, the tortuosity limit, it is Kaden's extra-axonal
compartment.
"""
@inline zeppelin(b, Dpar, Dperp) = zeppelin(b, Dpar, Dperp, _SphericalMean())
@inline zeppelin(b, Dpar, Dperp, v) = exp(-b * Dperp) * stick(b, Dpar - Dperp, v)

"""
    ball(b, D) = exp(−b·D)

Isotropic diffusion (free water, CSF, an ADC): the same along every direction and in the
spherical mean. The exponential unit, [`relaxrate`](@ref), with b and D.
"""
@inline ball(b, D) = exp(-b * D)

# √π·erf(x)/(2x), from x² = x2: the mean of exp(−x²·cos²ψ) over all directions (a stick's signal
# exp(−b·D·(g·n)²) averaged over gradient directions g, with x² = b·D); 1 at x = 0. Near 0, its
# Taylor series in x², Σ (−1)ⁿ x²ⁿ / (n!·(2n + 1)), 6 terms (error below 2e-12 for x² < 0.05):
# no 0/0 at b = 0, and no square root, whose derivative is infinite at 0 (the fit's dual numbers
# would carry it). The coefficients are integers, and π enters converted to x2's type, so a
# Float32 fit stays in Float32.
@inline function _sphmean(x2)
    if abs(x2) < 0.05                   # (comparing a dual number compares its value)
        t = x2
        s = 1 - t / 3
        t *= x2; s += t / 10
        t *= x2; s -= t / 42
        t *= x2; s += t / 216
        t *= x2; s -= t / 1320
        return s
    end
    x = sqrt(x2)
    return erf(x) / x * (sqrt(oftype(x, π)) / 2)
end

# An acquisition field that a model may or may not have (SMT's TE): its value at observation j,
# or `false` without it. `false` is Julia's "strong zero": false·x is a zero of x's type (a
# Float32 stays Float32, a dual number a dual), so exp(−false·R) = 1 exactly.
Base.@propagate_inbounds _optobs(x::AbstractVector, j) = x[j]
_optobs(::Nothing, j) = false

end # module Blocks
