include(joinpath(@__DIR__, "compare_lib.jl"))
te = collect(0.01:0.01:0.32); ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
Δ = Free

run_case("T2Decay, T2 = 80 ms, 32 echoes, SNR 50", T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); snr = 50,
         linear = (:S0,), units = ((S0 = 1.0,),), nonneg = (true,),
         starts = [("EQUIP's (log-linear)", (;)), ("T2 = 2 s", (T2 = Δ(start = 2.0),)), ("T2 = 2 ms", (T2 = Δ(start = 0.002),))])

for T1 in (0.1, 0.9)
    run_case("T1IR magnitude, T1 = $T1 s, SNR 50", T1IR(TI = ti), (M0 = 1000.0, Sf = -0.95, T1 = T1); snr = 50,
             linear = (:M0,), units = ((M0 = 1.0,),), nonneg = (true,),
             starts = [("EQUIP's (BestOf)", (;)), ("T1 = 1 s (one start)", (T1 = Δ(start = 1.0),)),
                       ("T1 = 4 s (one start)", (T1 = Δ(start = 4.0),))])
end

run_case("QMTSIR, 5-point SIR, SNR 50", QMTSIR(; SIR_5POINT...), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0); snr = 50,
         linear = (:M0f,), units = ((M0f = 1.0,),), nonneg = (true,),
         starts = [("EQUIP's (defaults)", (;)), ("PSR = 0.4, R1f = 3", (PSR = Δ(start = 0.4), R1f = Δ(start = 3.0)))])

pairs = vec([SVector(a, b) for a in exp.(range(log(0.005), log(0.04); length = 8)),
                               b in exp.(range(log(0.04), log(0.4); length = 8))])
run_case("T2BiexpDecay, 15% at 20 ms + 85% at 80 ms, SNR 100", T2BiexpDecay(TE = te),
         (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08); snr = 100,
         linear = (:S0, :fshort), units = ((S0 = 1.0, fshort = 1.0), (S0 = 1.0, fshort = 0.0)), nonneg = (true, true),
         starts = [("EQUIP's (defaults)", (;)), ("T2s = 5 ms, T2l = 300 ms", (T2short = Δ(start = 0.005), T2long = Δ(start = 0.3))),
                   ("grid", (;))], grid = pairs)

run_case("T2DecayOffset, T2 = 80 ms, C = 30, SNR 50", T2DecayOffset(TE = te), (S0 = 1000.0, T2 = 0.08, C = 30.0); snr = 50,
         linear = (:S0, :C), units = ((S0 = 1.0, C = 0.0), (S0 = 0.0, C = 1.0)), nonneg = (true, false),
         starts = [("EQUIP's (defaults)", (;)), ("T2 = 1 s", (T2 = Δ(start = 1.0),))])
