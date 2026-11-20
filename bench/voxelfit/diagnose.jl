# Where VoxelFit's and EQUIP's estimates differ: bounds reached (EQUIP's caps), and which of
# the two reached the lower sum of squares (design.md, "Compared with the one-shot version")
using Statistics
include(joinpath(@__DIR__, "cases.jl"))
c = CASES[8]; vf = readbin(outfile("vf", c, 50, "default"), Float64, N, 2); eq = readbin(outfile("eq", c, 50, "default"), Float64, N, 2)
d = abs.(vf[:, 2] .- eq[:, 2]) .> 1e-4
println("SPGR SNR 50: voxels differing ", count(d), "; of them EQUIP's T1 at its cap 5 s: ", count(d .& (eq[:, 2] .>= 5 - 1e-9)),
        ", VoxelFit's T1 beyond 5 s: ", count(d .& (vf[:, 2] .> 5)), " (largest ", round(maximum(vf[:, 2]); sigdigits = 3), " s)")
c = CASES[3]
for snr in (50, 200)
    vf = readbin(outfile("vf", c, snr, "default"), Float64, N, 4); eq = readbin(outfile("eq", c, snr, "default"), Float64, N, 4)
    f(θ) = θ[:, 1] ./ (θ[:, 1] .+ θ[:, 3])
    d = (abs.(vf[:, 4] .- eq[:, 4]) .> 1e-4 * 0.08) .| (abs.(vf[:, 2] .- eq[:, 2]) .> 1e-4 * 0.02)
    println("biexp SNR $snr: voxels differing ", count(d), "; VoxelFit T2long > 2.5 s: ", count(d .& (vf[:, 4] .> 2.5)),
            ", EQUIP at a bound (T2long = 2.5, or fshort = 0 or 1): ",
            count(d .& ((eq[:, 4] .>= 2.5 - 1e-9) .| (f(eq) .<= 1e-9) .| (f(eq) .>= 1 - 1e-9))),
            ", VoxelFit with a zero amplitude: ", count(d .& ((vf[:, 1] .<= 1e-9) .| (vf[:, 3] .<= 1e-9))),
            ", VoxelFit with the components swapped (T2a > T2b): ", count(d .& (vf[:, 2] .> vf[:, 4])))
end

for (ci, snr, j, tol) in ((8, 50, 2, 1e-4), (3, 200, 4, 1e-4 * 0.08), (3, 50, 4, 1e-4 * 0.08))
    c = CASES[ci]; P = length(vfnames(c))
    Y = readbin(datafile(c, snr), Float64, N, length(c.x))
    vf = readbin(outfile("vf", c, snr, "default"), Float64, N, P); eq = readbin(outfile("eq", c, snr, "default"), Float64, N, P)
    d = findall(abs.(vf[:, j] .- eq[:, j]) .> tol)
    sse(θ, v) = sum(abs2, Y[v, :] .- vfsignal(c, c.x, θ[v, :]))
    rel = [(sse(eq, v) - sse(vf, v)) / sse(vf, v) for v in d]
    println(rpad("$(c.name) SNR $snr", 16), length(d), " voxels differ: EQUIP's SSE lower in ", count(<(-1e-9), rel),
            ", VoxelFit's lower in ", count(>(1e-9), rel), ", equal (to 1e-9) in ", count(r -> abs(r) <= 1e-9, rel),
            "; median relative SSE difference ", round(median(rel); sigdigits = 2))
end
