# Whole fits, written-out Jacobian (sir_analytic.jl) or not: julia -t 1 --project=. bench/jacobian/compare.jl dual.jls, then with -e 'include("bench/jacobian/sir_analytic.jl"); include("bench/jacobian/compare.jl")' analytic.jls
using Serialization
using EQUIP, Random, Statistics, Printf
N = 10_000; rng = MersenneTwister(3)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
cases = [("4-point", SIR_4POINT, false), ("5-point, kmf free", SIR_5POINT, true), ("16-point", SIR_16POINT, false), ("16-point, kmf free", SIR_16POINT, true)]
best(f) = (f(); minimum((GC.gc(); @elapsed f()) for _ in 1:5))
runs = Dict()
for (name, acq, kf) in cases
    m = QMTSIREqualR1(; acq...)
    θ = (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = kf ? lu(5, 20) : fill(12.5, N), Sm = fill(0.83, N))
    Y = simulate(m, θ; snr = 100, noise = :rician, rng = MersenneTwister(1))
    kw = kf ? (kmf = Free(),) : (;)
    r = fit(m, Y; kw...); t = best(() -> fit(m, Y; kw...))
    runs[name] = (r.params, r.iterations, t)
end
open(ARGS[1], "w") do io; serialize(io, runs); end
