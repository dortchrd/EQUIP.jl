# =================================================================================================
# Directions: a fiber's direction, and gradient direction sets
#
#   fiberdirection(θ, φ)    a fiber's unit vector n, from its angles: with Julia's dot, the view
#                           ξ = g ⋅ n that the directional compartments take (stick, zeppelin)
#   directionset(N)         N gradient directions spread evenly over the sphere (electrostatic
#                           repulsion), to simulate a scanner's scheme
#   Directions              the type of a directional model's `g`: a vector of SVector{3}s
#   checkencoding(b, g)     a directional model's acquisition checked: a direction per b-value,
#                           unit vectors where b > 50 s/mm² (every directional model's
#                           constructor, the ones @mrimodel and newmodel write, and
#                           `diffusionencoding`)
#
# (Until 2026-11-06 this was orientation.jl, which also held the spherical mean's pipeline
# errors for SMT: `orientationcovariance`, `RicianSphericalMean`, `sphericalmeannoise`. They are
# on the roadmap, to return with the nerve-SMT protocol work; the code is on branch `nerve-smt`:
# docs/design.md, roadmap item 18.)
# =================================================================================================

"""
    Directions

The type of a directional model's gradient directions `g`: a vector with one `SVector{3}` of
numbers per image (what [`diffusionencoding`](@ref) makes from a .bvec). As a type parameter's
bound in a model's struct it says what `g` holds, and Julia refuses anything else (a vector of
plain vectors, a matrix) when the model is made:

```julia
@kwdef struct MyModel{A<:AbstractVector,G<:Directions} <: MRIModel
    b::A                                # b-values (s/mm²)
    g::G                                # gradient direction of each image (a unit SVector{3})
    MyModel(b::A, g::G) where {A,G} = new{A,G}(b, checkencoding(b, g))
end
```

The line inside is the struct's *inner constructor*: Julia calls it however the model is made
(`MyModel(b = ..., g = ...)`, from files, rebuilt in a fit's precision), and it checks b and g
with [`checkencoding`](@ref). Every built-in directional model is written so, and
[`@mrimodel`](@ref) and [`newmodel`](@ref) write it for a model whose acquisition is `b` and `g`.
"""
const Directions = AbstractVector{<:SVector{3,<:Real}}

"""
    checkencoding(b, g) -> g

A directional model's acquisition checked, the rule in one place: one gradient direction per
b-value, and a unit vector (within 1e-2, as DIPY checks) wherever b > 50 s/mm²; at b ≈ 0 the
direction does not matter, and FSL writes 0 0 0 there. Returns `g`; an error names the image
otherwise. Called by every directional model's constructor (see [`Directions`](@ref)) and by
[`diffusionencoding`](@ref), so a model made by hand, or rebuilt in the fit's precision or by the
protocol designer, is checked as one made from files is. The b-values are not changed. One pass
over the images, nothing allocated.
"""
function checkencoding(b, g)
    length(b) == length(g) || _badencodinglength(b, g)
    @inbounds for j in eachindex(b, g)
        b[j] > _B0THRESHOLD || continue
        abs(norm(g[j]) - 1) <= 1e-2 || _notunit(j, b[j], g[j])
    end
    return g
end
const _B0THRESHOLD = 50
# the errors made in functions of their own, so the check's loop stays small (and allocates nothing)
@noinline _badencodinglength(b, g) = throw(DimensionMismatch("$(length(b)) b-values but " *
    "$(length(g)) gradient directions: there must be one direction per image"))
@noinline _notunit(j, bj, gj) = throw(ArgumentError("image $j (b = $bj): the gradient direction " *
    "($(gj[1]), $(gj[2]), $(gj[3])) has length $(round(norm(gj); sigdigits = 4)), not 1: the " *
    "directions must be unit vectors (where b > $_B0THRESHOLD s/mm²)"))

"""
    fiberdirection(θ, φ) -> (nx, ny, nz)

The unit vector at polar angle θ (from z) and azimuth φ (from x, in the x–y plane), in
radians, as a tuple of three numbers: a fiber's direction n. With one image's gradient direction
g[j] (a model's field `g`, a unit `SVector{3}` per image), the cosine between them is
ξ = g[j] ⋅ n (Julia's dot product, which EQUIP exports: it takes tuples and static vectors, with
nothing allocated): the view `ξ` that the compartments
[`stick`](@ref) and [`zeppelin`](@ref) take for one gradient direction, and for unit vectors
also the gradient's projection on the fiber. Only ξ² enters their signals, so n and −n are the
same fiber. Not a compartment: the geometry that gives a directional compartment its view.

In a model, compute n once, before the loop over the images: its sines and cosines are the same
for every image, and the compiler does not move them out of the loop (measured: a
stick–zeppelin's fit took twice as long with them per image, docs/design.md, "Directions per
image").

```julia
n = fiberdirection(θ, φ)                                  # once per voxel
for j in eachindex(out, b, g)
    ξ = g[j] ⋅ n                                          # the cosine, image j's gradient to fiber
    out[j] = S0 * (vax * stick(b[j], Dax, ξ) + (1 - vax) * zeppelin(b[j], Dax, Dperp, ξ))
end
```
"""
@inline fiberdirection(θ, φ) = (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))

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
            f -= (f ⋅ g[i]) * g[i]               # along the sphere
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
