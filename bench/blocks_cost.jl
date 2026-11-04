# =================================================================================================
# What building a signal from blocks costs, separated from what carrying switched-off parts costs
#
# Each equation three ways, in one process, interleaved, best of 7 (docs/design.md, "Building
# blocks"):
#   written   the equation typed out in the loop (as before the refactor)
#   blocks    the same parameters, the signal from the blocks (EQUIP.Blocks: relaxrate, stick, zeppelin)
#   family    the family model (R2Decay, SMT), its switched-off parts carried as fixed parameters
# for a T2 decay (32 echoes), an ADC (7 b-values: the same exponential unit with b and D) and
# Kaden's SMT (4 shells). ns per signal! call (one thread), µs per voxel for a 20,000-voxel fit
# (all threads), and the first fit's time (its compilation, shared code compiled before).
#   julia -t 4 --project=. bench/blocks_cost.jl          (about 1 minute)
# =================================================================================================

using EQUIP, EQUIP.Blocks, Random, Printf, Statistics
import EQUIP: parameters, signal!
const E = EQUIP

# --- T2 decay (rate form): S0·exp(−R2·TE) --------------------------------------------------------
@kwdef struct WrittenR2{A} <: MRIModel; TE::A; end
@kwdef struct BlockR2{A} <: MRIModel; TE::A; end
parameters(::Type{<:Union{WrittenR2,BlockR2}}) =
    (S0 = Free(start = :scale, lower = 0), R2 = Free(start = m -> 1 / mean(m.TE), lower = 0))
function signal!(out, m::WrittenR2, p)
    @inbounds for j in eachindex(out, m.TE); out[j] = p.S0 * exp(-p.R2 * m.TE[j]); end
    return out
end
function signal!(out, m::BlockR2, p)
    @inbounds for j in eachindex(out, m.TE); out[j] = p.S0 * relaxrate(m.TE[j], p.R2); end
    return out
end
# --- ADC: S0·exp(−b·ADC), the same exponential unit with b and D ----------------------------------
@kwdef struct BlockADC{A} <: MRIModel; b::A; end
parameters(::Type{<:BlockADC}) = parameters(ADCDecay)
function signal!(out, m::BlockADC, p)
    @inbounds for j in eachindex(out, m.b); out[j] = p.S0 * relaxrate(m.b[j], p.ADC); end
    return out
end
# --- SMT, Kaden's two compartments ----------------------------------------------------------------
@kwdef struct WrittenSMT{A} <: MRIModel; b::A; end
@kwdef struct BlockSMT{A} <: MRIModel; b::A; end
parameters(::Type{<:Union{WrittenSMT,BlockSMT}}) = (S0 = Free(start = :scale, lower = 0),
    vax = Free(start = 0.5, lower = 0, upper = 1), Dax = Free(start = 2.0e-3, lower = 0, upper = 3.0e-3))
function signal!(out, m::WrittenSMT, p)                  # the code before the refactor
    (; S0, vax, Dax) = p
    Dperp = (1 - vax) * Dax; Dpar_perp = vax * Dax
    @inbounds for j in eachindex(out, m.b)
        b = m.b[j]
        out[j] = S0 * (vax * E._sphmean(b * Dax) + (1 - vax) * exp(-b * Dperp) * E._sphmean(b * Dpar_perp))
    end
    return out
end
function signal!(out, m::BlockSMT, p)                    # blocks, the extra-axonal D's computed here
    (; S0, vax, Dax) = p
    Dperp = (1 - vax) * Dax
    @inbounds for j in eachindex(out, m.b)
        b = m.b[j]
        out[j] = S0 * (vax * stick(b, Dax) + (1 - vax) * zeppelin(b, Dax, Dperp))
    end
    return out
end

te = collect(0.01:0.01:0.32); bd = [0.0, 250, 500, 750, 1000, 1500, 2000]; bs = [0.0, 1000, 2500, 4000]
CASES = [
    ("T2 decay, 32 TEs", "written", WrittenR2(TE = te), (S0 = 1000.0, R2 = 12.5)),
    ("T2 decay, 32 TEs", "blocks",  BlockR2(TE = te),   (S0 = 1000.0, R2 = 12.5)),
    ("T2 decay, 32 TEs", "family",  R2Decay(TE = te),   (S0 = 1000.0, R2 = 12.5)),
    ("ADC, 7 b",         "written", ADCDecay(b = bd),   (S0 = 1000.0, ADC = 0.8e-3)),
    ("ADC, 7 b",         "blocks",  BlockADC(b = bd),   (S0 = 1000.0, ADC = 0.8e-3)),
    ("SMT, 4 shells",    "written", WrittenSMT(b = bs), (S0 = 1000.0, vax = 0.6, Dax = 2.0e-3)),
    ("SMT, 4 shells",    "blocks",  BlockSMT(b = bs),   (S0 = 1000.0, vax = 0.6, Dax = 2.0e-3)),
    ("SMT, 4 shells",    "family",  SMT(b = bs),        (S0 = 1000.0, vax = 0.6, Dax = 2.0e-3)),
]
function sigloop(out, m, p, k)
    for _ in 1:k; signal!(out, m, p); end
    return out
end

# compile time of each model's first fit (shared code compiled first, by another model)
fit(T1SR(TD = [0.1, 0.5, 1.0, 2.0]), simulate(T1SR(TD = [0.1, 0.5, 1.0, 2.0]), (M0 = 1.0, T1 = 1.0); σ = 0.01, noise = :gaussian, n = 10))
first = Dict{String,Float64}()
data = Dict{String,Any}()
for (eq, how, m, θ) in CASES
    data[eq * how] = simulate(m, θ; snr = 100, n = 20_000, noise = :gaussian, rng = MersenneTwister(1))
    first[eq * how] = @elapsed fit(m, data[eq * how][1:10, :])
end
tfit = Dict(k => Inf for k in keys(first)); tsig = Dict(k => Inf for k in keys(first))
for rep in 1:7, (eq, how, m, θ) in CASES
    k = eq * how
    tfit[k] = min(tfit[k], @elapsed(fit(m, data[k])) / 20_000 * 1e6)
    p = resolve(m, θ); out = zeros(nobs(m)); sigloop(out, m, p, 10)
    tsig[k] = min(tsig[k], @elapsed(sigloop(out, m, p, 200_000)) / 200_000 * 1e9)
end
println("threads: ", Threads.nthreads(), "; best of 7, interleaved; fits of 20,000 voxels, SNR 100")
@printf("%-18s %-8s %10s %10s %12s\n", "equation", "how", "signal ns", "fit µs/vox", "first fit s")
for (eq, how, m, θ) in CASES
    k = eq * how
    @printf("%-18s %-8s %10.1f %10.3f %12.2f\n", eq, how, tsig[k], tfit[k], first[k])
end
