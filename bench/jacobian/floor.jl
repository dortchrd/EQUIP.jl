# One normal_equations call against the sums alone (docs/design.md, "A written-out Jacobian"): julia -t 1 --project=. bench/jacobian/floor.jl
# The most a written-out Jacobian could save per model: one `normal_equations` call (the signal
# on dual numbers, then the sums) against the sums alone from rows already made (the floor any
# Jacobian method pays, plus its own arithmetic: here none). One thread.
using EQUIP, Random, Printf, StaticArrays
import EQUIP: normal_equations, VoxelProblem, _pick
TE = collect(0.01:0.01:0.32)
function dmodel(M)
    g = directionset(30)
    b = [0.0; fill(1000.0, 30); fill(2000.0, 30)]
    M(b = b, g = [[SVector(0.0, 0.0, 0.0)]; g; g])
end
md, mk = dmodel(DTI), dmodel(DKI)
cases = [("QMTSIREqualR1 4-point", QMTSIREqualR1(; SIR_4POINT...), (PSR = 0.15, R1f = 1.0, Sf = -0.95, M0f = 1000.0, kmf = 12.5, Sm = 0.83)),
         ("T2, 32 echoes", T2Decay(TE = TE), (S0 = 1000.0, T2 = 0.08)),
         ("bi-exponential, 32 echoes", T2BiexpDecay(TE = TE), (S0 = 1000.0, fshort = 0.2, T2short = 0.015, T2long = 0.09)),
         ("kurtosis, 13 b", ADCKurtosis(b = collect(0.0:250.0:3000.0)), (S0 = 1000.0, ADC = 1e-3, K = 1.0)),
         ("DTI, 61 images", md, nominal(md; MD = 0.8e-3, FA = 0.5, direction = (1.0, 0.0, 0.0))),
         ("DKI, 61 images", mk, nominal(mk, :DKI_WM; direction = (1.0, 0.0, 0.0)))]
function sums(rows::Vector{SVector{P,T}}, res) where {P,T}
    A = zero(SMatrix{P,P,T}); g = zero(SVector{P,T}); sse = zero(T)
    @inbounds for i in eachindex(rows)
        J = rows[i]; r = res[i]
        A += J * J'; g += r * J; sse += r * r
    end
    return sse, A, g
end
function timeit(f, n)
    f(); t = Inf
    for _ in 1:5
        t0 = time_ns(); for _ in 1:n; f(); end; t = min(t, (time_ns() - t0) / n)
    end
    return t
end
println("| model | free | normal_equations ns | the sums alone ns | at most |"); println("|---|---:|---:|---:|---:|")
for (name, m, truth) in cases
    c = init(m); spec = c.spec
    F = EQUIP.freenames(spec); P = length(F)
    y = signal(m, truth) .+ 1.0
    copyto!(c.y, y)
    fixed = map(x -> _pick(x, 1, Float64), spec.fixed)
    prob = VoxelProblem{F}(c.model, c.y, fixed, spec.weights)
    θ = SVector{P,Float64}(Tuple(truth[k] for k in F))
    tn = timeit(() -> normal_equations(prob, θ, c.buf), 20_000)
    rows = [SVector{P,Float64}(randn(P)) for _ in 1:length(y)]; res = randn(length(y))
    ts = timeit(() -> sums(rows, res), 20_000)
    @printf("| %s | %d | %.0f | %.0f | %.1f× |\n", name, P, tn, ts, tn / ts); flush(stdout)
end
