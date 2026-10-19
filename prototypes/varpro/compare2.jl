include(joinpath(@__DIR__, "compare_lib.jl"))
# The two-amplitude models with columns the model supplies: each column computes only its own
# exponential (the biexponential's two T2s; the offset's constant column needs none).
te = collect(0.01:0.01:0.32)
Δ = Free
# Named functions with fixed field access (p.T2short), so the compiler resolves the field.
function col_short(c, m, p)
    @inbounds for j in eachindex(c, m.TE); c[j] = exp(-m.TE[j] / p.T2short); end
    return c
end
function col_long(c, m, p)
    @inbounds for j in eachindex(c, m.TE); c[j] = exp(-m.TE[j] / p.T2long); end
    return c
end
function col_t2(c, m, p)
    @inbounds for j in eachindex(c, m.TE); c[j] = exp(-m.TE[j] / p.T2); end
    return c
end
col_one(c, m, p) = fill!(c, one(eltype(c)))

pairs = vec([SVector(a, b) for a in exp.(range(log(0.005), log(0.04); length = 8)),
                               b in exp.(range(log(0.04), log(0.4); length = 8))])
run_case("T2BiexpDecay, own columns, 15% at 20 ms + 85% at 80 ms, SNR 100", T2BiexpDecay(TE = te),
         (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08); snr = 100,
         linear = (:S0, :fshort), units = (col_short, col_long), nonneg = (true, true),
         starts = [("EQUIP's (defaults)", (;)), ("T2s = 5 ms, T2l = 300 ms", (T2short = Δ(start = 0.005), T2long = Δ(start = 0.3))),
                   ("grid", (;))], grid = pairs)
run_case("T2BiexpDecay, own columns, 40% at 15 ms + 60% at 100 ms, SNR 100", T2BiexpDecay(TE = te),
         (S0 = 1000.0, fshort = 0.4, T2short = 0.015, T2long = 0.1); snr = 100,
         linear = (:S0, :fshort), units = (col_short, col_long), nonneg = (true, true),
         starts = [("EQUIP's (defaults)", (;)), ("grid", (;))], grid = pairs, seed = 2)
run_case("T2DecayOffset, own columns, T2 = 80 ms, C = 30, SNR 50", T2DecayOffset(TE = te), (S0 = 1000.0, T2 = 0.08, C = 30.0); snr = 50,
         linear = (:S0, :C), units = (col_t2, col_one), nonneg = (true, false),
         starts = [("EQUIP's (defaults)", (;)), ("T2 = 1 s", (T2 = Δ(start = 1.0),))])
