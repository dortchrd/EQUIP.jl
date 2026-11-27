# The qMT models' speed: one signal (plain numbers, and a fit's dual numbers, 5 partials), one
# voxel's fit (one thread), a whole brain (294,912 voxels, threads as started); for comparing
# versions of the code (docs/design.md, item 6). Each fits data it made, tissue varied.
#   julia -t 4 --project=test bench/qmt/speed.jl
using EQUIP, ForwardDiff, Random, Printf
const D = ForwardDiff.Dual{Nothing}
off = [1e3, 2e3, 4e3, 8e3, 16e3, 32e3]
spgr = (offset = [off; off; 0.0], B1e = [fill(4.0, 6); fill(8.0, 6); 0.0], tm = 0.012, ts = 0.002, TR = 0.030, FA = 10)
blk = blockpulse(180, 1e-3)
mt = (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1000.0, R1obs = 1.1)
sir = (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0)
cases = [("QMTSIR", QMTSIR(; SIR_5POINT...), sir, 200, (:PSR, :R1f, :Sf, :M0f)),
         ("QMTSIR with its pulse", QMTSIR(; SIR_5POINT..., pulse = blk), sir, 200, (:PSR, :R1f, :Sf, :M0f)),
         ("QMTSIREqualR1", QMTSIREqualR1(; SIR_5POINT...), sir, 200, (:PSR, :R1f, :Sf, :M0f)),
         ("QMTSPGRYarnykh", QMTSPGRYarnykh(; spgr...), mt, 800, (:PSR, :kmf, :T2f, :T2m, :M0f))]
isdefined(EQUIP, :QMTSPGRRamani) && push!(cases, ("QMTSPGRRamani", QMTSPGRRamani(; spgr...), mt, 100, (:PSR, :kmf, :T2f, :T2m, :M0f)))
isdefined(EQUIP, :QMTCWHenkelman) && push!(cases, ("QMTCWHenkelman", QMTCWHenkelman(offset = spgr.offset, B1cw = spgr.B1e ./ 2.5), mt, 100,
                                                  (:PSR, :kmf, :T2f, :T2m, :M0f)))
N = 294912
println("| model | one signal | one signal, dual | one voxel's fit | whole brain ($(Threads.nthreads()) threads) |")
println("|---|---:|---:|---:|---:|")
for (lab, m, θ0, snr, free) in cases
    θ = resolve(m, merge(nominal(m), θ0))
    θd = merge(θ, NamedTuple{free}(ntuple(i -> D(θ[free[i]], ntuple(k -> Float64(k == i), 5)), length(free))))
    t(p) = (out = Vector{typeof(p.M0f * 1.0)}(undef, nobs(m)); signal!(out, m, p);
            minimum(@elapsed(for _ in 1:50000; signal!(out, m, p); end) for _ in 1:7) / 50000)
    ts, td = t(θ), t(θd)
    rng = MersenneTwister(1); u(a, b) = a .+ (b - a) .* rand(rng, N)
    vary = haskey(θ0, :kmf) ? (PSR = u(0.05, 0.25), kmf = u(10, 40), T2f = u(0.02, 0.08), T2m = 1e-6 .* u(8, 16), R1obs = u(0.6, 1.4)) :
                              (PSR = u(0.05, 0.25), R1f = u(0.6, 1.4), Sf = u(-1.0, -0.9))
    tt = merge(map(x -> x isa Real ? fill(x, N) : x, merge(nominal(m), θ0)), vary)
    Y = simulate(m, tt; snr, noise = :rician, rng = MersenneTwister(2))
    kw = haskey(θ0, :kmf) ? (R1obs = tt.R1obs,) : (;)
    c = init(m; (haskey(θ0, :kmf) ? (R1obs = 1.1,) : (;))...); fit!(c, view(Y, 1, :))
    t1 = minimum(@elapsed(for i in 1:5000; fit!(c, view(Y, i, :)); end) for _ in 1:3) / 5000
    fit(m, Y[1:100, :]; map(x -> x[1:100], kw)...)
    tb = minimum(@elapsed(fit(m, Y; kw...)) for _ in 1:2)
    @printf("| %s | %.0f ns | %.0f ns | %.2f µs | %.2f s |\n", lab, 1e9ts, 1e9td, 1e6t1, tb)
    flush(stdout)
end
