# =================================================================================================
# Compartments: the diffusion compartments models are written with (exported by EQUIP)
#
#   iso(b, D)                  exp(−b·D)                          isotropic diffusion (free water; D = 0: a dot)
#   stick(b, D, view)          exp(−b·D·ξ²) at view = ξ           a cylinder of zero radius
#   zeppelin(b, D∥, D⊥, view)  exp(−b·D∥·ξ² − b·D⊥·(1 − ξ²)) at ξ  axially symmetric
#   powder                                                        the view averaged over directions
#
# b in s/mm², D in mm²/s. A compartment with a direction takes its view explicitly: a cosine ξ,
# the signal along one gradient direction at cosine ξ between the fiber and the gradient (what a
# direction set samples), or `powder`, its average over all directions (the spherical mean,
# NMR's powder average, what SMT fits). `stick(b, D)` without a view is an error that says which
# to write. The isotropic compartment is the same in every view and takes none. The cosine for one image comes from
# `fiberdirection` (orientation.jl) and Julia's dot product: ξ = (gx[j], gy[j], gz[j]) ⋅ n.
#
# The name is the literature's: the compartments of Panagiotaki et al.'s taxonomy (NeuroImage
# 2012: ball, stick, zeppelin, tensor, cylinder, sphere, dot) and of Microstructure.jl (Gong &
# Yendiki, Imaging Neuroscience 2026: Cylinder, Stick, Zeppelin, Sphere, Iso), whose names EQUIP
# follows: `iso` is Panagiotaki's ball, and with D = 0 the dot (Gong & Yendiki, section 2.2.1.5);
# a model made of them is a multi-compartment model. ("Kernel", in the standard-model and SMI
# papers, is the signal of one fiber bundle, its compartments together: a stick plus a zeppelin
# is one; Gong & Yendiki's Eqs. 6–8 use ξ = g·n, as here.)
# Plain functions, exported by EQUIP (`using EQUIP` brings them), for writing a model as its
# paper writes it:
#
#   @mrimodel IVIM(b; S0 = Free(start = :scale, lower = 0), f = Free(start = 0.1, lower = 0, upper = 0.5),
#                  Dstar = Free(start = 0.02), D = Free(start = 1e-3, lower = 0)) =
#       S0 * (f * iso(b, Dstar) + (1 - f) * iso(b, D))
#
# They are diffusion only. (Microstructure.jl's carry a T2 each, the factor e^(−t/T2) of Gong &
# Yendiki's Eqs. 1–5, included only with multi-TE data.) Here a compartment's T2 weighting is
# written in the model as its equation, `vax * exp(-TE / T2ax) * stick(...)`, in a model whose
# acquisition has the echo times to measure it (SMTT2): no T2 to switch off in the others. Every model, EQUIP's own included, is
# written out in the standard format (a struct, `parameters`, `signal!`), with these as its
# compartments. More: README, "Compartments", and examples/compartments.jl. (Until 2026-11-06
# they were "building blocks", first a module of their own, `EQUIP.Blocks`, with an exponential
# unit `decay(x, rate)`, while the T2 and SMT families were each one equation with parts switched
# on and off: docs/design.md, "Building blocks".) They cost nothing against the equation written
# out: the compiler inlines them.
#
# Diffusion compartments: Kaden et al., NeuroImage 2016; Panagiotaki et al., NeuroImage 2012;
# Gong & Yendiki, Imaging Neuroscience 2026 (doi:10.1162/IMAG.a.1102). The
# two views of a compartment are two methods of one function: Julia picks the one that matches
# the arguments (dispatch), when compiling, so neither costs a test at run time.
# =================================================================================================

"""
    iso(b, D) = exp(−b·D)

An isotropic compartment: Gaussian diffusion, the same along every direction and in the powder
average, so it takes no view. As Gong & Yendiki use it (Imaging Neuroscience 2026, Eq. 5): CSF or
free water (D about 3e-3 mm²/s at body temperature), an extra-cellular space (D from the tissue),
or, with D = 0, immobile water (a signal of 1 at every b: the "dot" of Panagiotaki et al.,
NeuroImage 2012, whose name for this compartment is the "ball").

```julia
iso(1000.0, 1e-3)                # exp(−1)
iso(4000.0, 0.0)                 # 1: a dot
```
"""
@inline iso(b, D) = exp(-b * D)

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
stick(1000.0, 2e-3, powder)      # 0.598: the spherical mean at b = 1000 s/mm²
stick(1000.0, 2e-3, 1.0)         # exp(−2): along the fiber
stick(1000.0, 2e-3, 0.0)         # 1: across it
```
"""
# S̄ = ∫₀¹ exp(−b·D·ξ²) dξ = √π·erf(√(b·D)) / (2√(b·D))                     (Gong & Yendiki, Eq. 3)
@inline stick(b, D, ::Powder) = _meanexp(b * D)
@inline stick(b, D, ξ::Real) = exp(-b * D * ξ^2)
stick(b, D) = throw(ArgumentError("stick(b, D) needs its view: stick(b, D, powder) for the " *
    "spherical mean, stick(b, D, ξ) along one direction at cosine ξ to the fiber"))

"""
    zeppelin(b, D∥, D⊥, powder)    the spherical mean: exp(−b·D⊥)·√π·erf(√(b·(D∥ − D⊥))) / (2√(b·(D∥ − D⊥)))
    zeppelin(b, D∥, D⊥, ξ)         along one direction: exp(−b·D∥·ξ² − b·D⊥·(1 − ξ²))

A zeppelin: an axially symmetric tensor, diffusivity D∥ along the fiber and D⊥ across it (the
extra-axonal space in SMT and NODDI), b in s/mm², D in mm²/s. With a number `ξ`, the cosine
between the fiber and the gradient, the signal along that one direction (Gong & Yendiki, Imaging
Neuroscience 2026, Eq. 7); with `powder`, its average over all directions (their Eq. 1; Kaden et
al., NeuroImage 2016). With D⊥ = (1 − v_ax)·D_ax, the tortuosity limit, it is Kaden's
extra-axonal compartment; with D∥ = D⊥ = D, it is `iso(b, D)`.

D⊥ > D∥ (an oblate tensor) is allowed and correct in both views: the spherical mean then takes
the imaginary error function, exp(−b·D⊥)·√π·erfi(√(b·(D⊥ − D∥))) / (2√(b·(D⊥ − D∥))). Whether
it is physical is the model's to say: to keep D⊥ ≤ D∥, give the model a fraction, D⊥ =
fperp·D∥ with 0 ≤ fperp ≤ 1 (as Microstructure.jl does; SMT's tortuosity limit is the case
fperp = 1 − v_ax).

```julia
zeppelin(1000.0, 2e-3, 0.5e-3, powder)   # the spherical mean
zeppelin(1000.0, 2e-3, 0.5e-3, 1.0)      # exp(−2): along the fiber
zeppelin(1000.0, 2e-3, 0.5e-3, 0.0)      # exp(−0.5): across it
```
"""
# S = exp(−b·D∥·ξ² − b·D⊥·(1 − ξ²))                                          (Gong & Yendiki, Eq. 7)
@inline zeppelin(b, Dpar, Dperp, ξ::Real) = exp(-b * Dpar * ξ^2 - b * Dperp * (1 - ξ^2))
# S̄ = exp(−b·D⊥)·∫₀¹ exp(−b·(D∥ − D⊥)·ξ²) dξ
#   = exp(−b·D⊥)·√π·erf(√(b·(D∥ − D⊥))) / (2√(b·(D∥ − D⊥)))                (Gong & Yendiki, Eq. 1)
@inline zeppelin(b, Dpar, Dperp, ::Powder) = exp(-b * Dperp) * _meanexp(b * (Dpar - Dperp))
zeppelin(b, Dpar, Dperp) = throw(ArgumentError("zeppelin(b, D∥, D⊥) needs its view: " *
    "zeppelin(b, D∥, D⊥, powder) for the spherical mean, zeppelin(b, D∥, D⊥, ξ) along one direction"))

# ∫₀¹ exp(−a·ξ²) dξ: the mean over all directions of exp(−a·ξ²), ξ the cosine between a fiber and
# a gradient direction spread uniformly over the sphere. Every axially symmetric compartment's
# spherical mean is made of it: the stick's is _meanexp(b·D), the zeppelin's exp(−b·D⊥)·
# _meanexp(b·(D∥ − D⊥)) (and a cylinder's would be the zeppelin's with its restricted D⊥). Closed
# forms:
#   a > 0:  √π·erf(√a) / (2√a)          (Gong & Yendiki's √(π/(4a))·erf(√a))
#   a < 0:  √π·erfi(√−a) / (2√−a)       (exp(+|a|·ξ²): a zeppelin with D⊥ > D∥, oblate)
#   a = 0:  1
# Near 0 (|a| < 0.05, either sign) its Taylor series instead, Σ (−1)ⁿ aⁿ / (n!·(2n + 1)), 6 terms
# (error below 2e-12): no 0/0 at b = 0 or at D∥ = D⊥, and no square root, whose derivative is
# infinite at 0 (the fit's dual numbers would carry it). The coefficients are integers, and π
# enters converted to a's type, so a Float32 fit stays in Float32.
@inline function _meanexp(a)
    if abs(a) < 0.05                    # (comparing a dual number compares its value)
        t = a
        s = 1 - t / 3
        t *= a; s += t / 10
        t *= a; s -= t / 42
        t *= a; s += t / 216
        t *= a; s -= t / 1320
        return s
    elseif a > 0
        x = sqrt(a)
        return erf(x) / x * (sqrt(oftype(x, π)) / 2)
    else
        x = sqrt(-a)
        return erfi(x) / x * (sqrt(oftype(x, π)) / 2)
    end
end
