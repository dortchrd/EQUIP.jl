# =================================================================================================
# Building blocks: diffusion compartments for writing models (exported by EQUIP)
#
#   ball(b, D)                 exp(−b·D)                          isotropic diffusion (free water)
#   stick(b, D, view)          exp(−b·D·ξ²) at view = ξ           a cylinder of zero radius
#   zeppelin(b, D∥, D⊥, view)  exp(−b·D⊥)·stick(b, D∥ − D⊥, view) axially symmetric
#   powder                                                        the view averaged over directions
#   fiberdirection(θ, φ)       (sinθ·cosφ, sinθ·sinφ, cosθ)       a fiber's direction, from angles
#   g ⋅ n                      g₁·n₁ + g₂·n₂ + g₃·n₃              the view ξ of one image (Julia's dot)
#
# b in s/mm², D in mm²/s. A compartment with a direction takes its view explicitly: a cosine ξ,
# the signal along one gradient direction at cosine ξ between the fiber and the gradient (what a
# direction set samples), or `powder`, its average over all directions (the spherical mean,
# NMR's powder average, what SMT fits). `stick(b, D)` without a view is an error that says which
# to write. The ball is the same in every view and takes none.
#
# Plain functions, exported by EQUIP (`using EQUIP` brings them), for writing a model as its
# paper writes it:
#
#   @mrimodel IVIM(b; S0 = Free(start = :scale, lower = 0), f = Free(start = 0.1, lower = 0, upper = 0.5),
#                  Dstar = Free(start = 0.02), D = Free(start = 1e-3, lower = 0)) =
#       S0 * (f * ball(b, Dstar) + (1 - f) * ball(b, D))
#
# They are the diffusion compartments only, as Microstructure.jl's compartments are (the owner,
# 2026-11-06): relaxation is written as its equation, `exp(-TE / T2)`, and every model, EQUIP's
# own included, is written out in the standard format (a struct, `parameters`, `signal!`), with
# these as its compartments. More: README, "Building blocks", and examples/building_blocks.jl.
# (Until 2026-11-06 they were a module of their own, `EQUIP.Blocks`, with an exponential unit
# `decay(x, rate)`, and the T2 and SMT families were each one equation with parts switched on
# and off: removed, docs/design.md, "Building blocks".) The blocks cost nothing against the
# equation written out: the compiler inlines them.
#
# Diffusion compartments: Kaden et al., NeuroImage 2016; Panagiotaki et al., NeuroImage 2012 (the
# taxonomy of compartments). The two views of a compartment are two methods of one function: Julia
# picks the one that matches the arguments (dispatch), when compiling, so neither costs a test at
# run time.
# =================================================================================================

"""
    ball(b, D) = exp(−b·D)

Isotropic diffusion (free water, CSF, an ADC): the same along every direction and in the powder
average, so it takes no view. Named as a compartment so that diffusion models read like the
papers (Panagiotaki et al., NeuroImage 2012).

```julia
ball(1000.0, 1e-3)               # exp(−1)
```
"""
@inline ball(b, D) = exp(-b * D)

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
g = (gx[j], gy[j], gz[j]), the cosine between them is ξ = g ⋅ n (Julia's dot product, which
EQUIP exports: it takes tuples, with nothing allocated): the view `ξ` that [`stick`](@ref) and [`zeppelin`](@ref) take for one gradient
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
