include(joinpath(@__DIR__, "vp.jl"))
using StaticArrays
function check(m, truth, linear, units, nonneg)
    y = collect(Float64, signal(m, truth))
    cache = init(m); spec = cache.spec; F = E.freenames(spec)
    vs = vp_setup(cache, linear, units, nonneg)
    prob = E.VoxelProblem{vs.Fn}(vs.vp, y, vs.fixed)
    θt = SVector(map(n -> Float64(truth[n]), vs.Fn))
    sse_t, = E.normal_equations(prob, θt, nothing)
    s0 = fit(m, y; solver = LM(maxiter = 0)).params
    θ0 = SVector(map(n -> Float64(s0[n]), vs.Fn))
    r = E.levmar(prob, θ0, vs.lo, vs.hi, cache.opts, vs.vp.cols[1])
    rf = fit(m, y)
    println(rpad(string(nameof(typeof(m))), 14), "SSE at truth ", round(sse_t; sigdigits = 2),
            " | VP from EQUIP's start: ", r.status, " iters ", r.iterations, " θ ", round.(r.θ; sigdigits = 5),
            " c ", round.(linear_coeffs(prob, r.θ); sigdigits = 5), " | full: iters ", rf.iterations)
end
te = collect(0.01:0.01:0.32); ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
check(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08), (:S0,), ((S0 = 1.0,),), (true,))
check(T1IR(TI = ti), (M0 = 1000.0, Sf = -0.95, T1 = 0.9), (:M0,), ((M0 = 1.0,),), (true,))
check(QMTSIR(; SIR_5POINT...), (PSR = 0.12, R1f = 1.0, Sf = -0.95, M0f = 1000.0), (:M0f,), ((M0f = 1.0,),), (true,))
check(T2BiexpDecay(TE = te), (S0 = 1000.0, fshort = 0.15, T2short = 0.02, T2long = 0.08), (:S0, :fshort),
      ((S0 = 1.0, fshort = 1.0), (S0 = 1.0, fshort = 0.0)), (true, true))
check(T2DecayOffset(TE = te), (S0 = 1000.0, T2 = 0.08, C = 30.0), (:S0, :C),
      ((S0 = 1.0, C = 0.0), (S0 = 0.0, C = 1.0)), (true, false))
