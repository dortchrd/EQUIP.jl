# =================================================================================================
# test/integration/macro.jl: Writing a model: @mrimodel (a model from one observation's signal) and newmodel (a model
# file to edit).
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl macro
# =================================================================================================

@testset "@mrimodel: a model from a function of one observation" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    mmerr(ex) = try macroexpand(@__MODULE__, ex); "" catch e; sprint(showerror, e isa LoadError ? e.error : e) end
    rng = MersenneTwister(11)
    # The pieces: the type, its acquisition, the roles (a start from the acquisition, a value
    # fixed, a tie by name, a tie by formula), the docstring
    te = collect(0.01:0.01:0.32)
    m = MacroT2Offset(TE = te)
    @test m isa MRIModel && m.TE === te && nobs(m) == 32
    @test keys(parameters(MacroT2Offset)) == (:S0, :T2, :C)
    @test parameters(m).T2.start == mean(te)                  # Free(start = mean(TE))
    @test parameters(MacroSIR) == parameters(QMTSIR)          # kmf = 12.5, R1m = R1f, ...
    @test parameters(TieFormula).k2 isa Fixed{<:Function}
    @test occursin("made with @mrimodel", string(@doc MacroT2Offset))
    @test MacroSIR(TI = TI5, TD = TD5).magnitude && !MacroSIR(TI = TI5, TD = TD5, magnitude = false).magnitude
    @test resolve(TieFormula(t = [1.0]), (A = 1.0, k1 = 2.0, k2 = 0.5 * 2.0)) == (A = 1.0, k1 = 2.0, k2 = 1.0)

    # The same fits as the built-in models, bit for bit (the same equation, per observation),
    # in both precisions
    mb = T2DecayOffset(TE = te)
    θ = (S0 = fill(1000.0, 4, 3), T2 = collect(reshape(range(0.04, 0.12; length = 12), 4, 3)), C = fill(20.0, 4, 3))
    Y = simulate(mb, θ; snr = 100, noise = :rician, rng)
    @test signal(m, (S0 = 1000.0, T2 = 0.08, C = 20.0)) == signal(mb, (S0 = 1000.0, T2 = 0.08, C = 20.0))
    for T in (Float64, Float32)
        a, b = fit(m, T.(Y); precision = T), fit(mb, T.(Y); precision = T)
        @test isequal(a.params, b.params) && a.iterations == b.iterations && eltype(a.params.T2) == T
    end
    fa = [2.0, 5, 10, 15, 20, 30]
    mm, mt = MacroMFA(FA = fa, TR = 0.015), T1MFA(FA = fa, TR = 0.015)    # one TR for all
    Ym = simulate(mt, (M0 = fill(1000.0, 5), T1 = collect(range(0.6, 1.8; length = 5)), B1 = 1.0); snr = 200,
                  noise = :gaussian, rng)
    for T in (Float64, Float32)
        @test isequal(fit(mm, T.(Ym); precision = T).params, fit(mt, T.(Ym); precision = T).params)
    end
    @test isequal(fit(mm, Ym; B1 = 0.9).params, fit(mt, Ym; B1 = 0.9).params)
    ms, mq = MacroSIR(TI = TI5, TD = TD5), QMTSIR(TI = TI5, TD = TD5)
    θs = map(x -> x isa Real ? fill(Float64(x), 3, 2) : x, nominal(mq, :SIR_lesion))
    Ys = simulate(mq, θs; snr = 200, noise = :rician, rng)
    for T in (Float64, Float32)
        a, b = fit(ms, T.(Ys); precision = T), fit(mq, T.(Ys); precision = T)
        @test isequal(a.params, b.params) && a.iterations == b.iterations
    end
    @test isequal(fit(ms, Ys; kmf = Free(start = 10.0)).params, fit(mq, Ys; kmf = Free(start = 10.0)).params)
    # crlb and montecarlo take it as any model
    c1, c2 = crlb(ms, nominal(mq, :SIR_WM_I); snr = 200), crlb(mq, nominal(mq, :SIR_WM_I); snr = 200)
    @test all(k -> isapprox(c1.sd[k], c2.sd[k]; rtol = 1e-10), keys(c2.sd))

    # No allocation per fit
    c = init(ms)
    y = Ys[1, 1, :]
    fit!(c, y)
    @test_noalloc (@allocated fit!(c, y)) == 0

    # A tie by formula; a free parameter with no start: given in the call
    mt2 = TieFormula(t = collect(0.1:0.1:2.0))
    yt = signal(mt2, (A = 2.0, k1 = 1.5, k2 = 0.75))
    @test fit(mt2, yt).params.k1 ≈ 1.5
    mn = NoStart(t = collect(0.1:0.1:2.0))
    yn = signal(mn, (A = 2.0, k = 1.5))
    @test occursin("`k` has no starting value: give one in the call, `k = Free(start = ...)`",
                   msg(() -> fit(mn, yn)))
    @test fit(mn, yn; k = Free(start = 1.0)).params.k ≈ 1.5

    # A directional model (acquisition b and g), as the built-in ones: g an SVector{3} per image
    # in the body; the same signal as StickZeppelin (whose own is checked written out in
    # diffusion.jl); fitted in Float32 too; g typed and checked when the model is made
    msz = MacroStickZeppelin(; DWI_ACQ...)
    tsz = (S0 = 1000.0, vax = 0.6, Dax = 1.9e-3, theta = 0.7, phi = 0.3)
    @test signal(msz, tsz) ≈ signal(StickZeppelin(; DWI_ACQ...), tsz) rtol = 1e-12
    rz = fit(msz, simulate(msz, map(x -> fill(x, 20), tsz); snr = 100, noise = :gaussian, rng = MersenneTwister(3)); precision = Float32)
    @test eltype(rz.params.vax) == Float32 && all(isconverged, rz.status)
    @test occursin("SVector{3", msg(() -> MacroStickZeppelin(b = DWI_ACQ.b, g = [Vector(x) for x in DWI_ACQ.g])))
    @test occursin("image 2", msg(() -> MacroStickZeppelin(b = DWI_ACQ.b, g = 2 .* DWI_ACQ.g)))

    # Lengths checked once against the output
    @test occursin("MacroSIR: `TD` has 4 values for 5 observations",
                   msg(() -> signal!(zeros(5), MacroSIR(TI = TI5, TD = TD5[1:4]), resolve(mq, nominal(mq)))))

    # Mistakes, said while the macro expands
    @test occursin("no acquisition", mmerr(:(@mrimodel A1(; S0) = S0)))
    @test occursin("no parameters", mmerr(:(@mrimodel A2(t) = t)))
    @test occursin("`t` is used twice", mmerr(:(@mrimodel A3(t; t) = t)))
    @test occursin("no types", mmerr(:(@mrimodel A4(t::Vector; S0) = S0 * t)))
    @test occursin("no types", mmerr(:(@mrimodel A5(t; S0::Float64) = S0 * t)))
    @test occursin("acquisition comes before the options", mmerr(:(@mrimodel A6(t, mag = true, u; S0) = S0)))
    @test occursin("cannot depend on another parameter", mmerr(:(@mrimodel A7(t; a = 1.0, b = Free(start = 2a)) = a * b)))
    @test occursin("only the start may use the acquisition", mmerr(:(@mrimodel A8(t; a = Free(lower = minimum(t))) = a)))
    @test occursin("cannot depend on the acquisition", mmerr(:(@mrimodel A9(t; a = maximum(t)) = a)))
    @test occursin("tied to itself", mmerr(:(@mrimodel A10(t; a = a) = a)))
    @test occursin("write a function definition", mmerr(:(@mrimodel A11)))
    @test occursin("where", mmerr(:(@mrimodel A12(t; a) where {T} = a)))
end

@testset "newmodel: a model file, blank or from an @mrimodel definition" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    dir = mktempdir()
    rng = MersenneTwister(12)
    # A file loaded into a module of its own (as a user's `include`), its model found by name
    # (through `invokelatest`: the names are newer than this code).
    function load(file)
        mod = Module()
        Core.eval(mod, :(using EQUIP, Statistics))
        Base.include(mod, file)
        return mod
    end
    # From @mrimodel: the code written out, loaded, gives the same fits as the macro's model
    # and the built-in one
    f = newmodel(MacroSIR; file = joinpath(dir, "sir.jl"))
    @test f == joinpath(dir, "sir.jl") && occursin("@inline function _macrosir(TI, TD, magnitude, PSR", read(f, String))
    @test occursin("R1m     = Fixed(:R1f)", read(f, String)) && occursin("return magnitude ? abs(s) : s", read(f, String))
    mod = load(f)
    mw = Base.invokelatest(Base.invokelatest(getproperty, mod, :MacroSIR); TI = TI5, TD = TD5)
    mq = QMTSIR(TI = TI5, TD = TD5)
    Ys = simulate(mq, map(x -> x isa Real ? fill(Float64(x), 3, 2) : x, nominal(mq, :SIR_lesion));
                  snr = 200, noise = :rician, rng)
    @test isequal(Base.invokelatest(fit, mw, Ys).params, fit(mq, Ys).params)
    @test Base.invokelatest(parameters, mw) == parameters(mq)
    f = newmodel(MacroT2Offset; file = joinpath(dir, "t2.jl"))
    txt = read(f, String)
    @test occursin("T2 = Free(start = m -> mean(m.TE), lower = 1.0e-6)", txt)
    @test occursin("made with @mrimodel", txt)                  # its docstring kept
    mw = Base.invokelatest(Base.invokelatest(getproperty, load(f), :MacroT2Offset); TE = collect(0.01:0.01:0.32))
    Yt = simulate(T2DecayOffset(TE = mw.TE), (S0 = fill(1000.0, 4), T2 = [0.05, 0.07, 0.09, 0.11],
                  C = fill(10.0, 4)); snr = 100, noise = :gaussian, rng)
    @test isequal(Base.invokelatest(fit, mw, Yt).params, fit(T2DecayOffset(TE = mw.TE), Yt).params)
    @test occursin("MacroMFA(FA, TR::Real)", read(newmodel(MacroMFA; file = joinpath(dir, "mfa.jl")), String))
    # a directional model written out with the built-in models' struct: g of type Directions, the
    # constructor that checks b and g; loaded, the same signal, and the check kept
    f = newmodel(MacroStickZeppelin; file = joinpath(dir, "sz.jl"))
    @test occursin("A2<:Directions", read(f, String)) && occursin("checkencoding(b, g)", read(f, String))
    Msz = Base.invokelatest(getproperty, load(f), :MacroStickZeppelin)
    tsz = (S0 = 1000.0, vax = 0.6, Dax = 1.9e-3, theta = 0.7, phi = 0.3)
    @test Base.invokelatest(signal, Base.invokelatest(Msz; DWI_ACQ...), tsz) == signal(MacroStickZeppelin(; DWI_ACQ...), tsz)
    @test occursin("image 2", msg(() -> Base.invokelatest(Msz; b = DWI_ACQ.b, g = 2 .* DWI_ACQ.g)))

    # A blank model: it loads, its roles are there, its signal is a TODO (zeros)
    f = newmodel(:BlankIR; acquisition = [:TI, :TD], parameters = [:M0, :T1],
                 options = (magnitude = true,), file = joinpath(dir, "sub", "blank.jl"))
    mod = load(f)
    mb = Base.invokelatest(Base.invokelatest(getproperty, mod, :BlankIR); TI = [0.1, 1.0], TD = [2.0, 2.0])
    @test mb.magnitude === true && Base.invokelatest(nobs, mb) == 2
    @test keys(Base.invokelatest(parameters, mb)) == (:M0, :T1)
    @test Base.invokelatest(signal, mb, (M0 = 1.0, T1 = 1.0)) == [0.0, 0.0]
    f = newmodel(:Roles; acquisition = [:TE], file = joinpath(dir, "roles.jl"),
                 parameters = (S0 = Free(start = :scale, lower = 0), T2 = Free(start = m -> mean(m.TE)), C = Fixed(0.0)))
    txt = read(f, String)
    @test occursin("S0 = Free(start = :scale, lower = 0.0),", txt) && occursin("T2 = Free(start = m -> TODO)", txt)
    @test keys(Base.invokelatest(parameters, Base.invokelatest(getproperty, load(f), :Roles))) == (:S0, :T2, :C)   # (the type: the TODO start is not called)

    # Mistakes
    @test occursin("exists", msg(() -> newmodel(:Roles; acquisition = [:TE], parameters = [:S0], file = f)))
    @test newmodel(:Roles; acquisition = [:TE], parameters = [:S0], file = f, force = true) == f
    @test occursin("was not made with @mrimodel", msg(() -> newmodel(T2Decay)))
    @test occursin("`t` is used twice", msg(() -> newmodel(:X; acquisition = [:t], parameters = [:t], file = joinpath(dir, "x.jl"))))
    @test occursin("give the acquisition", msg(() -> newmodel(:X; acquisition = Symbol[], parameters = [:a], file = joinpath(dir, "x.jl"))))
    @test occursin("not a Julia name", msg(() -> newmodel("my model"; acquisition = [:t], parameters = [:a])))
end
