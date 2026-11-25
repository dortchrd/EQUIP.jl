# The share of each model's fit spent in normal_equations (docs/design.md, "A written-out Jacobian"): julia -t 1 --project=. bench/jacobian/share.jl
# Where a fit's time goes, per model: the share of profile samples inside `normal_equations`
# (the signal on dual numbers and the sums JᵀJ, Jᵀr: what an analytic Jacobian would replace),
# against the whole fit (start, solver steps, Cholesky, maps). One thread, 10,000 voxels, SNR 100.
using EQUIP, Random, Profile, Printf
N = 10_000; rng = MersenneTwister(3)
uu(a, b) = a .+ (b - a) .* rand(rng, N); lu(a, b) = exp.(log(a) .+ (log(b) - log(a)) .* rand(rng, N))
TE = collect(0.01:0.01:0.32)
sir(acq, kmf) = (QMTSIREqualR1(; acq...), (PSR = uu(0.05, 0.25), R1f = lu(0.6, 1.8), Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N),
                 kmf = kmf ? lu(5, 20) : fill(12.5, N), Sm = fill(0.83, N)), kmf ? (kmf = Free(),) : (;))
cases = [("QMTSIREqualR1 4-point", sir(SIR_4POINT, false)...),
         ("QMTSIREqualR1 16-point, kmf free", sir(SIR_16POINT, true)...),
         ("QMTSIR 4-point", QMTSIR(; SIR_4POINT...), (r = lu(0.6, 1.8); (PSR = uu(0.05, 0.25), R1f = r, Sf = uu(-1.05, -0.9), M0f = fill(1000.0, N), kmf = fill(12.5, N), R1m = r, Sm = fill(0.83, N), Mzf_td0 = zeros(N), Mzm_td0 = zeros(N))), (;)),
         ("T2, 32 echoes", T2Decay(TE = TE), (S0 = fill(1000.0, N), T2 = lu(0.02, 0.3)), (;)),
         ("bi-exponential, 32 echoes", T2BiexpDecay(TE = TE), (S0 = fill(1000.0, N), fshort = uu(0.05, 0.4), T2short = lu(0.008, 0.03), T2long = lu(0.06, 0.15)), (;)),
         ("kurtosis, 13 b", ADCKurtosis(b = collect(0.0:250.0:3000.0)), (S0 = fill(1000.0, N), ADC = lu(0.3e-3, 2.5e-3), K = uu(0.3, 2.0)), (;)),
         ("T1IR, 9 TI", T1IR(TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]), (M0 = fill(1000.0, N), Sf = uu(-1.0, -0.85), T1 = lu(0.3, 4.0)), (;))]
function dmodel(M)
    g = directionset(30)
    b = [0.0; fill(1000.0, 30); fill(2000.0, 30)]
    M(b = b, g = [[SVector(0.0, 0.0, 0.0)]; g; g])
end
md, mk = dmodel(DTI), dmodel(DKI)
push!(cases, ("DTI, 61 images", md, nominal(md; MD = 0.8e-3, FA = 0.5, direction = (1.0, 0.0, 0.0)) |> t -> map(x -> fill(Float64(x), N), t), (;)))
push!(cases, ("DKI, 61 images", mk, map(x -> fill(Float64(x), N), nominal(mk, :DKI_WM; direction = (1.0, 0.0, 0.0))), (;)))
function share(m, Y, kw)
    fit(m, Y[1:10, :]; kw...)
    t = minimum(@elapsed(fit(m, Y; kw...)) for _ in 1:3)
    Profile.clear(); Profile.init(n = 10^7, delay = 0.0002)
    @profile for _ in 1:3; fit(m, Y; kw...); end
    io = IOBuffer(); Profile.print(io; format = :flat, C = false, mincount = 0); s = String(take!(io))
    tot = parse(Int, match(r"Total snapshots: (\d+)", s)[1])
    ne = sum((parse(Int, split(l)[1]) for l in split(s, '\n') if occursin(r"\snormal_equations\(", l) || occursin(r"\snormal_equations$", l)); init = 0)
    return t, ne / tot
end
println("| model | µs/voxel | share in signal + Jacobian + sums |"); println("|---|---:|---:|")
for (name, m, θ, kw) in cases
    Y = simulate(m, θ; snr = 100, noise = :gaussian, rng = MersenneTwister(1))
    t, f = share(m, Y, kw)
    @printf("| %s | %.2f | %.0f%% |\n", name, 1e6t / N, 100f); flush(stdout)
end
