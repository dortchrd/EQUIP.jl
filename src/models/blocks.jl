# =================================================================================================
# Building blocks: the pieces signal equations are made of
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
# relaxation with a rate of exactly zero is 1 without an exp. A dual number is never skipped, at
# zero too: it is being differentiated (a free parameter, or crlb's and leakage's seeded fixed
# values), and its derivative is needed.
#
# Relaxation and decay
#   _relax(t, R)               exp(−t·R): a T2 decay over TE with R = 1/T2 (R = 0, T2 = Inf: none),
#                              or a T1 term, or a b-value's attenuation with R = D
#
# Diffusion compartments at one observation, b in s/mm², diffusivities in mm²/s (Kaden et al.,
# NeuroImage 2016; Panagiotaki et al., NeuroImage 2012, for the taxonomy). Each comes in two views,
# chosen by the last argument:
#   a cosine t (a number)      the signal along one gradient direction, at cosine t between the
#                              fibre and the gradient (what a direction set samples)
#   _SphericalMean()           its average over all directions, ∫₀¹ ... dt (what SMT fits)
#
#   _stick(b, D, v)            exp(−b·D·t²)                       a cylinder of zero radius
#   _zeppelin(b, D∥, D⊥, v)    exp(−b·D⊥)·exp(−b·(D∥ − D⊥)·t²)    axially symmetric: a stick with
#                              D∥ − D⊥ times the radial attenuation, so both views follow
#   _ball(b, D)                exp(−b·D)                          isotropic: the same in every view
# =================================================================================================

@inline _relax(t, R) = _skip(R) ? one(t * R) : exp(-t * R)

# A weight (a fraction, a rate) that is a plain number at exactly zero: its part can be skipped. A
# dual number (`ForwardDiff.Dual <: Real`, not an AbstractFloat) never is: the method is chosen by
# type, when compiling.
@inline _skip(x::AbstractFloat) = iszero(x)
@inline _skip(x) = false

"""The spherical-mean view of a diffusion compartment (see src/models/blocks.jl)."""
struct _SphericalMean end

@inline _stick(b, D, t::Real) = exp(-b * D * t^2)
@inline _stick(b, D, ::_SphericalMean) = _sphmean(b * D)
@inline _zeppelin(b, Dpar, Dperp, v) = exp(-b * Dperp) * _stick(b, Dpar - Dperp, v)
@inline _ball(b, D) = exp(-b * D)
# `t::Real` and `::_SphericalMean` are two methods of one function: Julia picks the one whose
# argument type matches (dispatch), at compile time, so neither view costs a test at run time.

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
