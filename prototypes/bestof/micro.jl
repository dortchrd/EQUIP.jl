# Where the dictionary's time goes: one voxel's start, T1IR with 30 candidates (9 TIs), and the
# biexponential's 8 × 8 with fshort solved (32 echoes)
using EQUIP, Random, StaticArrays
# The best of 5 timings of n calls, per call
function per(f, n = 200_000)
    f()
    best = Inf
    for _ in 1:5
        t = @elapsed for _ in 1:n
            f()
        end
        best = min(best, t / n)
    end
    return best
end
TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]
m = T1IR(TI = TI)
y = signal(m, (M0 = 1000.0, Sf = -0.95, T1 = 1.0)) .+ randn(Xoshiro(1), 9)
for (label, mm, yy, kw) in (("IR 30", m, y, (;)),
        ("biexp 8x8", T2BiexpDecay(TE = collect(0.01:0.01:0.32)),
         signal(T2BiexpDecay(TE = collect(0.01:0.01:0.32)), (S0 = 1000.0, fshort = 0.3, T2short = 0.02, T2long = 0.1)),
         (T2short = Free(start = BestOf(exp.(range(log(0.005), log(0.05), 8)))), T2long = Free(start = BestOf(exp.(range(log(0.05), log(0.64), 8)))))))
    c = init(mm; kw...)
    spec = c.spec; F = EQUIP.freenames(spec); P = length(F)
    copyto!(c.y, yy)
    prob = EQUIP.VoxelProblem{F}(c.model, c.y, spec.fixed, spec.weights)
    θ0 = SVector{P,Float64}(Tuple(spec.start))
    d = c.dict; b = c.buf
    t1 = per(() -> EQUIP._dictionary_start(d, prob, θ0, spec, b))
    t2 = per(() -> EQUIP._best_start(prob, θ0, spec, b))
    t3 = per(() -> fit!(c, yy), 50_000)
    println(label, ": dictionary start ", round(1e6t1; digits = 3), " µs, per voxel ", round(1e6t2; digits = 3), " µs, whole fit! ", round(1e6t3; digits = 3), " µs")
end
