# One call, duals or written out (as compare.jl): julia -t 1 --project=. bench/jacobian/percall.jl duals
using EQUIP, Random, Printf, StaticArrays
import EQUIP: normal_equations, VoxelProblem, _pick
function timeit(f, n)
    f(); t = Inf
    for _ in 1:5
        t0 = time_ns(); for _ in 1:n; f(); end; t = min(t, (time_ns() - t0) / n)
    end
    return t
end
res = Dict()
for (name, acq) in (("4-point", SIR_4POINT), ("16-point", SIR_16POINT))
    m = QMTSIREqualR1(; acq...); truth = (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, Sm = 0.83)
    c = init(m); spec = c.spec; F = EQUIP.freenames(spec); P = length(F)
    copyto!(c.y, signal(m, truth) .+ 1.0)
    prob = VoxelProblem{F}(c.model, c.y, map(x -> _pick(x, 1, Float64), spec.fixed), spec.weights)
    θ = SVector{P,Float64}(Tuple(truth[k] for k in F))
    # the exps alone: 3 per observation
    ex = () -> (s = 0.0; @inbounds for j in eachindex(m.TI); s += exp(-1.0 * m.TI[j]) + exp(-1.0 * m.TD[j]) + exp(-14.4 * m.TI[j]); end; s)
    @printf("%s %s: normal_equations %.0f ns; the exps alone %.0f ns\n", ARGS[1], name, timeit(() -> normal_equations(prob, θ, c.buf), 50_000), timeit(ex, 50_000))
end
