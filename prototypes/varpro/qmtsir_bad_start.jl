include(joinpath(@__DIR__, "vp.jl"))
using Random, Statistics, StaticArrays, Printf
m = QMTSIR(; SIR_5POINT...)
truth = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)
Y = simulate(m, truth; snr = 50, n = 2000, rng = MersenneTwister(1))
good = fit(m, Y)
bad = fit(m, Y; PSR = Free(start = 0.4), R1f = Free(start = 3.0))
q(x) = round.(quantile(vec(x), [0.1, 0.5, 0.9]); sigdigits = 3)
println("from defaults:       PSR ", q(good.params.PSR), "  R1f ", q(good.params.R1f), "  Sf ", q(good.params.Sf), "  SSE ", q(good.sse))
println("from PSR 0.4, R1f 3: PSR ", q(bad.params.PSR), "  R1f ", q(bad.params.R1f), "  Sf ", q(bad.params.Sf), "  SSE ", q(bad.sse))
println("SSE ratio bad/good: ", q(bad.sse ./ good.sse))
println("at a bound (bad): R1f = 10: ", count(==(10), bad.params.R1f), "  PSR = 1: ", count(==(1), bad.params.PSR),
        "  PSR = 0: ", count(==(0), bad.params.PSR), "  Sf = 0: ", count(==(0), bad.params.Sf), "  Sf = -1.2: ", count(==(-1.2), bad.params.Sf))
