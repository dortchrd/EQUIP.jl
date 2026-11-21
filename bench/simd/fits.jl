# Batching (docs/design.md, Performance notes): julia -t 1 --project=. bench/simd/fits.jl
# Whole fits: QMTSIREqualR1 against a copy whose exp is branch-free (vectorized across
# observations), 10,000 voxels, one thread, best of 3; the maps compared
using EQUIP, ForwardDiff, Random, Statistics, Printf
import EQUIP: parameters, signal!
import ForwardDiff: Dual, value, partials
include(joinpath(@__DIR__, "bexp.jl"))
@inline bexp(d::Dual{T,V,N}) where {T,V,N} = (e = bexp(value(d)); Dual{T,V,N}(e, e * partials(d)))
@kwdef struct SIRv{A<:AbstractVector,B<:AbstractVector} <: MRIModel
    TI::A
    TD::B
    magnitude::Bool = true
end
parameters(::Type{<:SIRv}) = parameters(QMTSIREqualR1)
function signal!(out, m::SIRv, p)
    (; TI, TD, magnitude) = m
    (; PSR, R1f, Sf, M0f, kmf, Sm) = p
    k = kmf * (1 + PSR); a = (Sf + PSR * Sm) / (1 + PSR); b = PSR * (Sf - Sm) / (1 + PSR)
    @inbounds @simd for j in eachindex(out, TI, TD)
        Ei, Ed, Ek = bexp(-R1f * TI[j]), bexp(-R1f * TD[j]), bexp(-k * TI[j])
        s = M0f * (1 + Ei * ((1 - Ed) * (a + b * Ek) - 1))
        out[j] = magnitude ? abs(s) : s
    end
    out
end
N = 10_000; rng = MersenneTwister(5)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
θ = (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = lu(5, 20), Sm = fill(0.83, N))
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:3))
for (scheme, acq, kw) in (("4-point, kmf fixed", SIR_4POINT, (;)), ("5-point, kmf free", SIR_5POINT, (kmf = Free(),)),
                          ("16-point, kmf free", SIR_16POINT, (kmf = Free(),)))
    a, v = QMTSIREqualR1(; acq...), SIRv(; acq...)
    Y = simulate(a, θ; snr = 200, noise = :gaussian, rng = MersenneTwister(1))
    ta = best(() -> fit(a, Y; kw...)); tv = best(() -> fit(v, Y; kw...))
    ra, rv = fit(a, Y; kw...), fit(v, Y; kw...)
    @printf("%-20s QMTSIREqualR1 %.2f µs, branch-free exp %.2f µs a voxel (%.2f×); PSR largest difference %.0e\n",
            scheme, 1e6ta / N, 1e6tv / N, ta / tv, maximum(abs.(ra.params.PSR .- rv.params.PSR)))
end
