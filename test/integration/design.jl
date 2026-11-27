# =================================================================================================
# test/integration/design.jl: Protocol design: ProtocolDesign's objective (Eq. 5) at Dortch et al., MRM 2018, Table 2,
# and design's search (named optimize_protocol until 2026-11-09).
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl design
# =================================================================================================

@testset "ProtocolDesign and design: Dortch 2018 Table 2 by search" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    tissues = (:SIR_WM_I, :SIR_WM_II, :SIR_GM, :SIR_lesion)
    # Eq. 5 with the MATLAB code's settings, as in the crlb tests: signed signal, t_FSE 0.3 s,
    # SNR 200, the bounds of its biased fits; TR ≥ 0.8 s (TI + TD ≥ 0.5 s), times 10 ms–10 s
    bounds4 = (PSR = Free(lower = 0, upper = 1), R1f = Free(lower = 0, upper = 3),
               Sf = Free(lower = -1.05, upper = 0), M0f = Free(lower = 0, upper = 10))
    st(m) = sum(m.TI .+ m.TD .+ 0.3)
    g(m) = m.TI .+ m.TD .- 0.5
    space = (TI = (0.01, 10.0), TD = (0.01, 10.0))
    m4 = QMTSIR(; SIR_4POINT..., magnitude = false)
    m5 = QMTSIR(; SIR_5POINT..., magnitude = false)
    s4 = [Scenario(QMTSIR, nominal(QMTSIR, t); kmf = 12.5, bounds4...) for t in tissues]
    s5 = [Scenario(QMTSIR, nominal(QMTSIR, t); kmf = Free()) for t in tissues]
    d4 = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, constraints = g)
    d5 = ProtocolDesign(m5, s5; snr = 200, acquisition = space, scantime = st, constraints = g)
    # the objective: Eq. 5 at the Table 2 schemes, as crlb gives it, both covariances
    # (over the parameters: QMTSIR's derived f, after them in crlb's tables, left out)
    eq5(m, ss; cov = :sandwich) = maximum(sum(abs2, values(params_only(c, c.relrmse))) for c in (crlb(m, s; snr = 200, cov) for s in ss)) * st(m)
    @test d4(m4) ≈ 0.243395 rtol = 2e-6
    @test d4(m4) ≈ eq5(m4, s4) rtol = 1e-12
    @test d5(m5) ≈ 1.602062 rtol = 2e-6
    @test d5(m5) ≈ eq5(m5, s5) rtol = 1e-12
    d4b = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, cov = :biased)
    @test d4b(m4) ≈ eq5(m4, s4; cov = :biased) rtol = 1e-12
    # the same as x (the TIs, then the TDs), without allocation; one cache for the 4 tissues
    x4 = [m4.TI; m4.TD]
    @test d4(x4) == d4(m4) && length(d4.groups) == 1
    allocs(d, x) = @allocated d(x)
    allocs(d4, x4)
    @test_noalloc allocs(d4, x4) == 0
    # a broken constraint: Inf; the options: weights, a budget, another aggregate, a shared TD
    @test d4([0.01, 0.01, 0.278, 0.1, 4.171, 0.684, 2.73, 0.01]) == Inf     # TI + TD < 0.5
    cs = [crlb(m4, s; snr = 200) for s in s4]
    dw = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, weights = (R1f = 0, Sf = 0, M0f = 0))
    @test dw(m4) ≈ maximum(c.relrmse.PSR^2 for c in cs) * st(m4) rtol = 1e-12
    db = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, budget = st(m4))
    @test db(m4) ≈ d4(m4) / st(m4) rtol = 1e-12             # B = T: Eq. 5 without × T
    dm = ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, aggregate = v -> sum(v) / length(v))
    @test dm(m4) ≈ sum(sum(abs2, values(params_only(c, c.relrmse))) for c in cs) / 4 * st(m4) rtol = 1e-12
    m4s = QMTSIR(TI = m4.TI, TD = 2.5, magnitude = false)
    dsh = ProtocolDesign(m4s, s4; snr = 200, acquisition = space, shared = (:TD,), scantime = st)
    @test length(dsh.lower) == 5 && dsh([m4.TI; 2.5]) == dsh(m4s)
    @test dsh(m4s) ≈ eq5(m4s, s4) rtol = 1e-12
    # the search reaches the paper's optima (Fopt4 0.2433, Fopt5 1.6020, the GA's before
    # rounding), and the rounded protocols are no worse than Table 2's own
    r4 = design(d4; starts = 2, generations = 400, rng = MersenneTwister(1))
    @test r4.value <= 0.2433 && r4.rounded_value <= d4(m4)
    @test all(g(r4.rounded) .>= 0) && issorted(r4.protocol.TI)
    @test r4.rounded_value ≈ d4(r4.rounded) rtol = 1e-12   # the order of the points aside
    @test all(x -> isinteger(round(1000x; digits = 9)), [r4.rounded.TI; r4.rounded.TD])  # the ms
    r5 = design(d5; starts = 4, generations = 400, rng = MersenneTwister(2))
    @test r5.value <= 1.6021 && r5.rounded_value <= 1.6021
    @test design(d4; starts = 2, generations = 50, rng = MersenneTwister(3)).x ==
          design(d4; starts = 2, generations = 50, rng = MersenneTwister(3)).x   # repeatable
    # in one call, ProtocolDesign's keywords and the search's together: the same search
    x3 = design(d4; starts = 2, generations = 50, rng = MersenneTwister(3)).x
    @test design(m4, s4; snr = 200, acquisition = space, scantime = st, constraints = g,
                 starts = 2, generations = 50, rng = MersenneTwister(3)).x == x3
    s = sprint(show, MIME"text/plain"(), r4)
    @test startswith(s, "DesignResult for QMTSIR, 4 points: value 0.2432") && occursin("\n  rounded:  TI = [0.01, 0.01,", s)
    @test sprint(show, d4) == "ProtocolDesign(QMTSIR, 4 points: TI ∈ [0.01, 10.0], TD ∈ [0.01, 10.0]; 4 scenarios)"
    # mistakes
    @test occursin("give `scantime", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = space)))
    @test occursin("the acquisition to optimize is `acquisition", msg(() -> ProtocolDesign(m4, s4; snr = 200, vary = space, scantime = st)))
    @test occursin("ProtocolDesign takes no noise", msg(() -> ProtocolDesign(m4, s4; snr = 200, noise = :rician, acquisition = space, scantime = st)))
    @test occursin("give `acquisition`", msg(() -> ProtocolDesign(m4, s4; snr = 200, scantime = st)))
    @test occursin("has no field `TR`", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = (TR = (0.1, 1.0),), scantime = st)))
    @test occursin("not a per-observation field", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = (magnitude = (0, 1),), scantime = st)))
    @test occursin("(lower, upper)", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = (TI = (1.0, 0.1),), scantime = st)))
    @test occursin("`snr` or `σ`", msg(() -> ProtocolDesign(m4, s4; acquisition = space, scantime = st)))
    @test occursin("not a free parameter", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, weights = (kmf = 1,))))
    # values by name are named tuples (`_byname`): a single entry without its comma is the value
    @test occursin("`weights = (PSR = 1, R1f = 0)`", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = space, scantime = st, weights = 1)))
    @test occursin("`acquisition` takes values by name", msg(() -> ProtocolDesign(m4, s4; snr = 200, acquisition = (0.01, 10.0), scantime = st)))
    @test occursin("the scenario is for QMTSIR", msg(() -> ProtocolDesign(T2Decay(TE = TD5), s4; snr = 200, acquisition = (TE = (0.01, 1.0),), scantime = sum)))
    @test occursin("is 0 in a scenario's truth", msg(() -> ProtocolDesign(m4, [Scenario(QMTSIR, nominal(QMTSIR, :SIR_WM_I; PSR = 0.0))];
                                                                          snr = 200, acquisition = space, scantime = st)))
end
