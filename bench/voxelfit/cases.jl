# =================================================================================================
# bench/voxelfit: EQUIP against the one-shot version, VoxelFit 0.1 (branch `claude-oneshot`), on
# the eight models both have. Run everything with bench/voxelfit/run.sh; README.md says what each
# script does; the results of 2026-11-20 are in docs/design.md ("Compared with the one-shot
# version").
# =================================================================================================

# The comparison's cases, shared by gen.jl (EQUIP), vf.jl (VoxelFit) and eq.jl (EQUIP): eight
# models both packages have, each with one acquisition (SI units for both: VoxelFit's models
# take any units) and a typical truth. Parameters are stored in VoxelFit's order and form.
const CASES = [
    (name = "T2",       vf = "MonoExp",            eq = "T2Decay",       x = collect(0.01:0.01:0.32),
     truth = (S0 = 1000.0, T2 = 0.08)),
    (name = "T2offset", vf = "MonoExpOffset",      eq = "T2DecayOffset", x = collect(0.01:0.01:0.32),
     truth = (S0 = 1000.0, T2 = 0.08, C = 50.0)),
    (name = "biexp",    vf = "BiExp",              eq = "T2BiexpDecay",  x = collect(0.01:0.01:0.32),
     truth = (S0 = 1000.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)),
    (name = "ADC",      vf = "ADC",                eq = "ADCDecay",      x = collect(0.0:200.0:1000.0),
     truth = (S0 = 1000.0, ADC = 0.8e-3)),
    (name = "kurtosis", vf = "Kurtosis",           eq = "ADCKurtosis",   x = collect(0.0:250.0:3000.0),
     truth = (S0 = 1000.0, ADC = 0.8e-3, K = 1.0)),
    (name = "IR",       vf = "InversionRecovery",  eq = "T1IR",          x = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0],
     truth = (M0 = 1000.0, Sf = -1.0, T1 = 1.0)),
    (name = "SR",       vf = "SaturationRecovery", eq = "T1SR",          x = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
     truth = (M0 = 1000.0, T1 = 1.0)),
    (name = "SPGR",     vf = "SPGR",               eq = "T1MFA",         x = [2.0, 5, 10, 15, 20, 30],
     truth = (M0 = 1000.0, T1 = 1.0)),
]
const TR = 0.015                                   # SPGR / T1MFA
const N = 20_000                                   # voxels per dataset
const SNRS = (50, 200)

# VoxelFit's parameter names, and an EQUIP named tuple (numbers or vectors) in VoxelFit's form
vfnames(c) = c.name == "T2" ? (:S0, :T2) : c.name == "T2offset" ? (:S0, :T2, :C) :
             c.name == "biexp" ? (:A1, :T2a, :A2, :T2b) : c.name == "ADC" ? (:S0, :D) :
             c.name == "kurtosis" ? (:S0, :D, :K) : c.name == "IR" ? (:S0, :B, :T1) :
             c.name == "SR" ? (:S0, :T1) : (:M0, :T1)
function tovf(c, p)
    c.name == "biexp" && return (p.S0 .* p.fshort, p.T2short, p.S0 .* (1 .- p.fshort), p.T2long)
    c.name == "IR" && return (p.M0, 1 .- p.Sf, p.T1)
    c.name in ("ADC", "kurtosis") && return c.name == "ADC" ? (p.S0, p.ADC) : (p.S0, p.ADC, p.K)
    return values(p)[1:length(vfnames(c))]             # same order (B1, fixed, left out for SPGR)
end
# ... and back, for starts given to EQUIP (columns of a VoxelFit-ordered matrix)
function fromvf(c, θ)
    c.name == "biexp" && return (S0 = θ[:, 1] .+ θ[:, 3], fshort = θ[:, 1] ./ (θ[:, 1] .+ θ[:, 3]),
                                 T2short = θ[:, 2], T2long = θ[:, 4])
    c.name == "IR" && return (M0 = θ[:, 1], Sf = 1 .- θ[:, 2], T1 = θ[:, 3])
    names = c.name in ("ADC", "kurtosis") ? (c.name == "ADC" ? (:S0, :ADC) : (:S0, :ADC, :K)) :
            c.name in ("SR", "SPGR") ? (:M0, :T1) : vfnames(c)
    return NamedTuple{names}(Tuple(θ[:, j] for j in 1:length(names)))
end
# VoxelFit's signal equations, written here from its source, to check the EQUIP model against
function vfsignal(c, x, θ)
    n = c.name
    n == "T2" && return θ[1] .* exp.(-x ./ θ[2])
    n == "T2offset" && return θ[1] .* exp.(-x ./ θ[2]) .+ θ[3]
    n == "biexp" && return θ[1] .* exp.(-x ./ θ[2]) .+ θ[3] .* exp.(-x ./ θ[4])
    n == "ADC" && return θ[1] .* exp.(-x .* θ[2])
    n == "kurtosis" && return θ[1] .* exp.(-x .* θ[2] .+ x .^ 2 .* θ[2]^2 .* θ[3] ./ 6)
    n == "IR" && return abs.(θ[1] .* (1 .- θ[2] .* exp.(-x ./ θ[3])))
    n == "SR" && return θ[1] .* (1 .- exp.(-x ./ θ[2]))
    E1 = exp(-TR / θ[2])
    return θ[1] .* sind.(x) .* (1 - E1) ./ (1 .- cosd.(x) .* E1)
end
datafile(c, snr) = joinpath(@__DIR__, "data", "$(c.name)_snr$(snr).bin")
outfile(pkg, c, snr, what) = joinpath(@__DIR__, "out", "$(pkg)_$(c.name)_snr$(snr)_$(what).bin")
writebin(f, A) = open(io -> write(io, A), f, "w")
readbin(f, T, dims...) = (A = Array{T}(undef, dims...); open(io -> read!(io, A), f); A)
