# =================================================================================================
# Building blocks: the pieces signal equations are made of (module EQUIP.Blocks)
#
#   decay(x, rate)             exp(−x·rate)                       the exponential unit, in rate form:
#                                                                 decay(TE, 1/T2), decay(TE, R2),
#                                                                 decay(b, ADC), 1 − 2·decay(TI, 1/T1)
#   ball(b, D)                 exp(−b·D)                          isotropic diffusion (free water)
#   stick(b, D, view)          exp(−b·D·ξ²) at view = ξ           a cylinder of zero radius
#   zeppelin(b, D∥, D⊥, view)  exp(−b·D⊥)·stick(b, D∥ − D⊥, view) axially symmetric
#   fiberdirection(θ, φ)       (sinθ·cosφ, sinθ·sinφ, cosθ)       a fiber's direction, from angles
#   g ⋅ n                      g₁·n₁ + g₂·n₂ + g₃·n₃              the view ξ of one image (Julia's dot)
#
# b in s/mm², D in mm²/s, times in s. A compartment with a direction takes its view explicitly:
# a cosine ξ, the signal along one gradient direction at cosine ξ between the fiber and the
# gradient (what a direction set samples), or `powder`, its average over all directions (the
# spherical mean, NMR's powder average, what SMT fits). `stick(b, D)` without a view is an error
# that says which to write. The ball is the same in every view and takes none.
#
# For writing models: `using EQUIP.Blocks`, then e.g.
#
#   @mrimodel MyT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05)) = S0 * decay(TE, 1 / T2)
#
# T2 is what is fitted (the model's parameter); the block takes the rate. More: README, "Building
# blocks", and examples/building_blocks.jl. The names are plain, without the leading underscore
# that marks Julia's internal names, because they are meant to be used; they live in this small
# module so that `stick` and `ball` stay out of the namespace of anyone who does not ask for them.
# EQUIP's own models use them through `using .Blocks`.
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
# decay with a rate of exactly zero (a T2 of Inf) is 1 without an exp. A dual number is never
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
using LinearAlgebra: ⋅

export decay, ball, stick, zeppelin, powder, fiberdirection
# The cosine between a gradient direction and a fiber is Julia's own dot product, ξ = g ⋅ n
# (LinearAlgebra's `⋅`, typed \cdot then Tab), on two triples: g = (gx[j], gy[j], gz[j]) and
# n = fiberdirection(θ, φ). It takes tuples as well as vectors, with nothing allocated, so no
# block of our own is needed (a `fibercosine(g, n)` was one: the same numbers bit for bit, and
# as fast; docs/design.md, "Directions per image"). Passed on here, so `using EQUIP.Blocks`
# brings it; with `using LinearAlgebra` too it is the same function, so the names do not clash.
export ⋅

"""
    decay(x, rate) = exp(−x·rate)

The exponential unit, in rate form: a T2 decay `decay(TE, 1/T2)` (or `decay(TE, R2)`), a T1 term
`1 − 2·decay(TI, 1/T1)`, an ADC `decay(b, ADC)`, IVIM's `decay(b, Dstar)`. The model's parameter
is what is fitted, a T2 or an R2: the block takes the rate. `rate = 0` (a T2 of Inf, 1/Inf = 0)
gives 1, without computing an exp when it is a plain number.

!!! warning "The rate, not the time constant"
    `decay(TE, T2)` is exp(−TE·T2), not exp(−TE/T2): with T2 = 0.08 s it barely decays, and
    nothing reports the mistake (both arguments are just numbers). Write `decay(TE, 1/T2)`.
    A fit of simulated data from a known T2 shows it at once.

Written as `decay(TE, 1/T2)` in a model, it gives the built-in T2 decays' numbers bit for bit
(they convert T2 to a rate and call it), and no slower: 1/T2 does not change from echo to echo,
so the compiler computes it once, outside the loop.

```julia
using EQUIP.Blocks
decay(0.08, 1 / 0.08)            # e⁻¹
@mrimodel MyT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05)) = S0 * decay(TE, 1 / T2)
```
"""
@inline decay(x, rate) = _skip(rate) ? one(x * rate) : exp(-x * rate)

# A weight (a fraction, a rate) that is a plain number at exactly zero: its part can be skipped. A
# dual number (`ForwardDiff.Dual <: Real`, not an AbstractFloat) never is: the method is chosen by
# type, when compiling.
@inline _skip(x::AbstractFloat) = iszero(x)
@inline _skip(x) = false

"""
    ball(b, D) = exp(−b·D)

Isotropic diffusion (free water, CSF, an ADC): the same along every direction and in the powder
average, so it takes no view. The exponential unit, [`decay`](@ref), with b and D, named as a
compartment so that diffusion models read like the papers (Panagiotaki et al., NeuroImage 2012).
"""
@inline ball(b, D) = decay(b, D)

"""
    powder

The view of a diffusion compartment averaged over all gradient directions: the spherical mean
(NMR's powder average), what SMT fits. `stick(b, D, powder)`; `stick(b, D, ξ)` is one direction.
"""
struct Powder end
const powder = Powder()
# `Powder` is a type with one value, `powder`: a stick or a zeppelin given it picks its
# spherical-mean method, given a number its directional one. Julia chooses the method by the
# argument's type, when compiling (dispatch), so the view costs no test at run time.

"""
    stick(b, D, powder)    the spherical mean: √π·erf(√(b·D)) / (2√(b·D))
    stick(b, D, ξ)         along one gradient direction: exp(−b·D·ξ²)

A stick: a cylinder of zero radius, diffusion along the fiber only (an axon in SMT, a neurite in
NODDI), b in s/mm², D in mm²/s. With `powder`, the signal averaged over all gradient directions
(1 at b = 0); with a number `ξ`, the cosine of the angle between the fiber and the gradient, the
signal along that one direction. The powder average is the average of the directional signal over
ξ ∈ [0, 1] (Kaden et al., NeuroImage 2016).

```julia
using EQUIP.Blocks
stick(1000.0, 2e-3, powder)      # 0.606: the spherical mean at b = 1000 s/mm²
stick(1000.0, 2e-3, 1.0)         # exp(−2): along the fiber
stick(1000.0, 2e-3, 0.0)         # 1: across it
```
"""
@inline stick(b, D, ::Powder) = _sphmean(b * D)
@inline stick(b, D, ξ::Real) = exp(-b * D * ξ^2)
stick(b, D) = throw(ArgumentError("stick(b, D) needs its view: stick(b, D, powder) for the " *
    "spherical mean, stick(b, D, ξ) along one direction at cosine ξ to the fiber"))

"""
    zeppelin(b, D∥, D⊥, powder)    the spherical mean: exp(−b·D⊥)·stick(b, D∥ − D⊥, powder)
    zeppelin(b, D∥, D⊥, ξ)         along one direction: exp(−b·D⊥)·exp(−b·(D∥ − D⊥)·ξ²)

An axially symmetric tensor (the extra-axonal space in SMT and NODDI): a stick with D∥ − D⊥,
times the attenuation that every direction has, exp(−b·D⊥). The views as for [`stick`](@ref).
With D⊥ = (1 − v_ax)·D_ax, the tortuosity limit, it is Kaden's extra-axonal compartment.
"""
@inline zeppelin(b, Dpar, Dperp, v) = exp(-b * Dperp) * stick(b, Dpar - Dperp, v)
zeppelin(b, Dpar, Dperp) = throw(ArgumentError("zeppelin(b, D∥, D⊥) needs its view: " *
    "zeppelin(b, D∥, D⊥, powder) for the spherical mean, zeppelin(b, D∥, D⊥, ξ) along one direction"))

"""
    fiberdirection(θ, φ) -> (nx, ny, nz)

The unit vector at polar angle θ (from z) and azimuth φ (from x, in the x–y plane), in
radians, as a tuple of three numbers: a fiber's direction n. With one image's gradient direction
g = (gx[j], gy[j], gz[j]), the cosine between them is ξ = g ⋅ n (Julia's dot product, passed on
by this module): the view `ξ` that [`stick`](@ref) and [`zeppelin`](@ref) take for one gradient
direction, and for unit vectors also the gradient's projection on the fiber. Only ξ² enters
their signals, so n and −n are the same fiber.

In a model, compute n once, before the loop over the images: its sines and cosines are the same
for every image, and the compiler does not move them out of the loop (measured: a
stick–zeppelin's fit took twice as long with them per image, docs/design.md, "Directions per
image").

```julia
n = fiberdirection(θ, φ)                                  # once per voxel
for j in eachindex(out, b, gx, gy, gz)
    g = (gx[j], gy[j], gz[j])                             # image j's gradient direction
    ξ = g ⋅ n                                             # the cosine, gradient to fiber
    out[j] = S0 * (vax * stick(b[j], Dax, ξ) + (1 - vax) * zeppelin(b[j], Dax, Dperp, ξ))
end
```
"""
@inline fiberdirection(θ, φ) = (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))

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
