# =================================================================================================
# Directions: a fiber's direction, and gradient direction sets
#
#   fiberdirection(θ, φ)    a fiber's unit vector n, from its angles: with Julia's dot, the view
#                           ξ = g ⋅ n that the directional compartments take (stick, zeppelin)
#   directionset(N)         N gradient directions spread evenly over the sphere (electrostatic
#                           repulsion), to simulate a scanner's scheme
#
# (Until 2026-11-06 this was orientation.jl, which also held the spherical mean's pipeline
# errors for SMT: `orientationcovariance`, `RicianSphericalMean`, `sphericalmeannoise`. They are
# on the roadmap, to return with the nerve-SMT protocol work; the code is on branch `nerve-smt`:
# docs/design.md, roadmap item 18.)
# =================================================================================================

"""
    fiberdirection(θ, φ) -> (nx, ny, nz)

The unit vector at polar angle θ (from z) and azimuth φ (from x, in the x–y plane), in
radians, as a tuple of three numbers: a fiber's direction n. With one image's gradient direction
g = (gx[j], gy[j], gz[j]), the cosine between them is ξ = g ⋅ n (Julia's dot product, which
EQUIP exports: it takes tuples, with nothing allocated): the view `ξ` that the compartments
[`stick`](@ref) and [`zeppelin`](@ref) take for one gradient direction, and for unit vectors
also the gradient's projection on the fiber. Only ξ² enters their signals, so n and −n are the
same fiber. Not a compartment: the geometry that gives a directional compartment its view.

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
