# EQUIP: load time, the first fit (compiling) and the second, in a fresh process (ARGS[1]: a case's index)
t0 = time(); using EQUIP; t1 = time()
include(joinpath(@__DIR__, "cases.jl"))
c = CASES[parse(Int, ARGS[1])]
m = c.eq == "T1MFA" ? T1MFA(FA = c.x, TR = TR) :
    getfield(EQUIP, Symbol(c.eq))(; (c.eq in ("ADCDecay", "ADCKurtosis") ? :b : c.eq == "T1IR" ? :TI : c.eq == "T1SR" ? :TD : :TE) => c.x)
Y = readbin(datafile(c, 200), Float64, N, length(c.x))[1:2000, :]
t2 = time(); fit(m, Y); t3 = time(); fit(m, Y); t4 = time()
println(join(("EQUIP", c.name, round(t1 - t0; digits = 2), round(t3 - t2; digits = 2), round(1e3 * (t4 - t3); digits = 1)), '\t'))
