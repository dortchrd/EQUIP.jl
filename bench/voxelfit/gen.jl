# The datasets (EQUIP's simulate, Gaussian noise), and a check that each EQUIP model's signal
# is VoxelFit's equation at the same values.
using EQUIP, Random
include(joinpath(@__DIR__, "cases.jl"))
eqmodel(c) = c.eq == "T1MFA" ? T1MFA(FA = c.x, TR = TR) :
             getfield(EQUIP, Symbol(c.eq))(; (c.eq in ("ADCDecay", "ADCKurtosis") ? :b :
                 c.eq == "T1IR" ? :TI : c.eq == "T1SR" ? :TD : :TE) => c.x)
for c in CASES
    m = eqmodel(c)
    θ = c.eq == "T1MFA" ? merge(c.truth, (B1 = 1.0,)) : c.truth
    d = maximum(abs.(signal(m, θ) ./ vfsignal(c, c.x, collect(tovf(c, c.truth))) .- 1))
    println(rpad(c.name, 10), " EQUIP signal vs VoxelFit's equation: max rel diff ", d)
    for snr in SNRS
        Y = simulate(m, θ; snr, noise = :gaussian, n = N, rng = MersenneTwister(hash((c.name, snr))))
        writebin(datafile(c, snr), Y)
    end
end
