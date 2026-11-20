# Default fits, main against bestof-proto (the per-voxel BestOf path forced on the branch for
# T1IR, as main has it), one thread, 20,000 voxels, best of 5. Results compared too (saved).
using EQUIP, Random, Serialization
const BRANCH = isdefined(EQUIP, :_PROTO_DICTIONARY)
N = 20_000
lu(rng, a, b, n) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, n))
rng = Xoshiro(11)
TE = collect(0.01:0.01:0.32)
cases = [
 ("T1IR", T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]), (M0 = fill(1000.0, N), Sf = fill(-0.95, N), T1 = lu(rng, 0.3, 4, N))),
 ("T2Decay", T2Decay(TE = TE), (S0 = fill(1000.0, N), T2 = lu(rng, 0.02, 0.3, N))),
 ("T2BiexpDecay", T2BiexpDecay(TE = TE), (S0 = fill(1000.0, N), fshort = fill(0.2, N), T2short = lu(rng, 0.01, 0.03, N), T2long = lu(rng, 0.06, 0.15, N))),
 ("QMTSIR", QMTSIR(TI = [1.007, 0.010, 0.278, 0.010], TD = [0.010, 4.174, 2.730, 0.684]),
  (PSR = fill(0.15, N), R1f = lu(rng, 0.6, 1.8, N), Sf = fill(-0.95, N), M0f = fill(1000.0, N), kmf = fill(12.5, N),
   R1m = fill(1.0, N), Sm = fill(0.83, N), Mzf_td0 = zeros(N), Mzm_td0 = zeros(N))),
 ("T1IR, BestOf on Sf too", T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]), (M0 = fill(1000.0, N), Sf = fill(-0.95, N), T1 = lu(rng, 0.3, 4, N))),
]
BRANCH && (EQUIP._PROTO_DICTIONARY[] = false)
out = Dict{String,Any}()
for (name, m, θ) in cases
    θs = name == "QMTSIR" ? merge(θ, (R1m = θ.R1f,)) : θ
    Y = simulate(m, θs; snr = 100, noise = :gaussian, rng = Xoshiro(hash(name)))
    if name == "T1IR, BestOf on Sf too"
        BRANCH || continue        # several BestOf: the branch only
        kw = (Sf = Free(start = BestOf([-1.0, -0.9])),)
    else
        kw = (;)
    end
    f() = fit(m, Y; kw...)
    f()
    t = minimum((GC.gc(); @elapsed f()) for _ in 1:5)
    r = f()
    out[name] = (r.params, r.iterations)
    println(rpad(name, 26), BRANCH ? "branch " : "main   ", round(1e6t / N; digits = 3), " µs/voxel")
end
serialize(ARGS[1], out)
