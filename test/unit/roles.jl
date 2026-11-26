# =================================================================================================
# test/unit/roles.jl: Parameter roles and the model interface: nominal and resolve, ties, setup's keywords,
# parameters(m; changes...), and what models and roles print; StatsAPI's names; a docstring for
# every exported name.
#
# Part of EQUIP's tests (test/runtests.jl runs it, after test/setup.jl). Alone, from the repository:
#     julia -t auto --project=test test/runtests.jl roles
# =================================================================================================

@testset "nominal(m) and resolve" begin
    m = QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    # every value from parameters(m): starts, 1 for the scale, fixed values; ties kept
    @test nominal(m) == (PSR = 0.1, R1f = 1.0, Sf = -0.95, M0f = 1.0, kmf = 12.5, R1m = :R1f, Sm = 0.83,
                       Mzf_td0 = 0.0, Mzm_td0 = 0.0)
    p = nominal(m; PSR = 0.15, kmf = 10.0)
    @test p.PSR == 0.15 && p.kmf == 10.0 && p.R1m === :R1f && keys(p) == keys(parameters(m))
    @test nominal(T1MFA(FA = [2, 10], TR = 0.015)) == (M0 = 1.0, T1 = 1.0, B1 = 1.0)
    # a kept tie follows later changes; resolve gives numbers, in model order
    @test resolve(m, (; nominal(m)..., R1f = 0.7)).R1m == 0.7
    @test resolve(m, nominal(m)) == (PSR = 0.1, R1f = 1.0, Sf = -0.95, M0f = 1.0, kmf = 12.5, R1m = 1.0,
                                   Sm = 0.83, Mzf_td0 = 0.0, Mzm_td0 = 0.0)
    n = nominal(m)
    @test resolve(m, NamedTuple{reverse(keys(n))}(reverse(values(n)))) == resolve(m, n)
    @test resolve(m, nominal(m; R1m = q -> 0.8 * q.R1f)).R1m ≈ 0.8
    # signal = signal! on resolved values; simulate takes nominal(m)
    @test signal(m, nominal(m)) == signal!(zeros(3), m, resolve(m, nominal(m)))
    @test size(simulate(m, nominal(m); noise = :rician, snr = 100, n = 2)) == (2, 3)
    # no number to take: must be given, and the error says why
    te = collect(0.01:0.01:0.32)
    @test occursin("`T2` (it starts from the log-linear fit)", msg(() -> nominal(T2Decay(TE = te))))
    @test nominal(T2Decay(TE = te); T2 = 0.08) == (S0 = 1.0, T2 = 0.08)
    @test occursin("`T1` (it starts from BestOf values)", msg(() -> nominal(T1IR(TI = [0.1, 1.0]))))
    # ... and lists what truth takes for every parameter, the keywords' values included
    e = msg(() -> nominal(T2Decay(TE = te)))
    @test occursin("\n  S0    free  :scale       [0.0, Inf]     1.0\n", e)
    @test occursin(r"\n  T2 +free +:loglinear +\[1.0e-6, 2.5\] +required$", e)
    e = msg(() -> nominal(T1IR(TI = [0.1, 1.0]); M0 = 5.0))
    @test occursin(r"\n  M0 +free +:scale +\[0.0, Inf\] +5.0\n", e)
    # mistakes
    # (a model without derived quantities: QMTSIR's names that are not parameters go to its
    # fromderived, MPF and kfm, test/integration/derived.jl)
    @test occursin("unknown parameter `kfm`", msg(() -> nominal(T1IR(TI = [0.1, 1.0]); kfm = 3.0)))
    @test occursin("is a role", msg(() -> nominal(m; kmf = Fixed(10.0))))
    @test occursin("itself a tie", msg(() -> nominal(m; Sm = :R1m)))
    @test occursin("missing: Sf", msg(() -> resolve(m, Base.structdiff(nominal(m), NamedTuple{(:Sf,)}))))
    # with a tissue, a free parameter never takes its start: WM (T1, T2) does not fit QMTSIR
    e = msg(() -> nominal(m, :WM))
    @test occursin("no value for `PSR`, `R1f`, `Sf`, free parameters the tissue does not give", e)
    @test occursin("`nominal(m, :WM; PSR = value, R1f = value, Sf = value)`", e)
    @test occursin(r"\n  name +role +start/value +bounds +nominal\(m, :WM\)\n", e)
    @test occursin(r"\n  PSR +free +0.1 +\[0.0, 1.0\] +required\n", e)       # the start is not used
    @test occursin(r"\n  M0f +free +:scale +\[0.0, Inf\] +1.0\n", e)          # the scale is 1
    @test occursin("`nominal(m, tissue; Sf = value)`", msg(() -> nominal(T1IR(TI = [0.1]), (T1 = 1.0,))))
    @test occursin("is a role", msg(() -> nominal(m, :SIR_WM_II; kmf = Fixed(10.0))))
    @test nominal(m, :SIR_WM_II; PSR = 0.15).PSR == 0.15                     # keywords win
end

@testset "roles print readably" begin
    @test repr(Free()) == "Free()"
    @test repr(Free(start = 0.1, lower = 0, upper = 1)) == "Free(start = 0.1, lower = 0.0, upper = 1.0)"
    @test repr(Free(start = :scale, lower = 0)) == "Free(start = :scale, lower = 0.0)"
    @test repr(Free(start = [0.1, 0.2])) == "Free(start = 2-element Vector{Float64})"
    @test repr(Fixed(12.5)) == "Fixed(12.5)"
    @test repr(Fixed(:R1f)) == "Fixed(:R1f)"
    @test repr(Fixed(p -> p.R1f)) == "Fixed(<function>)"
    @test repr(Fixed(12.5; lower = 0)) == "Fixed(12.5, lower = 0.0)"
    @test_throws "lower (2.0) exceeds upper (1.0)" Fixed(1.5; lower = 2, upper = 1)
    @test repr(Free(start = BestOf([0.3, 0.8]))) == "Free(start = BestOf([0.3, 0.8]))"
end

@testset "parameters on the model's type: no acquisition needed" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    m = QMTSIR(; SIR_4POINT...)
    @test parameters(QMTSIR) == parameters(m)          # no QMTSIR start depends on TI or TD
    @test nominal(QMTSIR, :SIR_WM_I) == nominal(m, :SIR_WM_I)
    @test nominal(QMTSIR; PSR = 0.15) == nominal(m; PSR = 0.15) && defaults(QMTSIR) == defaults(m)
    @test occursin("nominal(QMTSIR, :WM; PSR = value", msg(() -> nominal(QMTSIR, :WM)))
    # a start computed from the acquisition: a function for the type, its value for a model
    b = [0.0, 500, 1000]
    @test parameters(ADCDecay).ADC.start isa Function
    @test parameters(ADCDecay(b = b)).ADC.start == 1 / mean(b) == nominal(ADCDecay(b = b)).ADC
    @test occursin("`ADC` (its start is computed from the acquisition)", msg(() -> nominal(ADCDecay)))
    @test occursin("`nominal(ADCDecay; ADC = value)`", msg(() -> nominal(ADCDecay)))
    @test nominal(ADCDecay; ADC = 1e-3) == (S0 = 1.0, ADC = 1e-3) == nominal(ADCDecay, (ADC = 1e-3,))
    ti = [0.1, 0.5, 1.0]
    @test parameters(T1IR(TI = ti)).T1.start.values == EQUIP._t1grid(ti)   # BestOf from the TIs
    # a function start given in a fit is computed for that model too
    te = collect(0.01:0.01:0.32)
    mt = T2Decay(TE = te)
    y = signal(mt, (S0 = 1000.0, T2 = 0.08))
    @test fit(mt, y; T2 = Free(start = q -> mean(q.TE)), solver = LM(maxiter = 0)).params.T2 == mean(te)
    @test fit(mt, y; T2 = Free(start = q -> mean(q.TE))).params.T2 ≈ 0.08 rtol = 1e-6
    # a model whose roles are defined on the instance (the other way) still works, but its
    # type cannot be asked
    @test parameters(DataStartTest([0.01, 0.02])).c == Fixed(1.0)
    @test occursin("give a model", msg(() -> nominal(DataStartTest)))
    @test occursin("give a model", msg(() -> defaults(DataStartTest)))
    @test occursin("a function of the model", msg(() -> Free(start = "x")))
end

@testset "models print at the REPL" begin
    txt(x) = sprint(show, MIME"text/plain"(), x)     # what typing `m` at the REPL shows
    s = txt(QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5))
    @test startswith(s, "QMTSIR with 3 observations\n  TI = [0.01, 0.1, 1.0]\n" *
                        "  TD = 2.5, every observation\n  magnitude = true\nParameters:\n" *
                        "  name     role   start/value  bounds             nominal(m)\n")
    @test occursin("\n  PSR      free   0.1          [0.0, 1.0]         0.1\n", s)
    @test occursin("\n  M0f      free   :scale       [0.0, Inf]         1.0\n", s)
    @test occursin("\n  kmf      fixed  12.5         freed: [0.0, Inf]  12.5\n", s)   # its bounds if freed
    @test occursin("\n  R1m      fixed  :R1f (tie)                      :R1f\n", s)
    @test endswith(s, "\n  Mzm_td0  fixed  0.0                             0.0")   # nothing to give
    s = txt(T2Decay(TE = 0.01:0.01:0.32))
    @test occursin("\n  TE = 0.01:0.01:0.32\n", s) && occursin(r"\n  T2 +free +:loglinear .+ required\n", s)
    @test endswith(s, "\n`nominal(m)` needs a value for T2: `nominal(m; T2 = value)`")
    @test occursin("TE = 32 values from 0.01 to 0.32", txt(T2Decay(TE = collect(0.01:0.01:0.32))))
    @test occursin(r"T1 +free +BestOf\(30 values from 0.02 to 5.0\)", txt(T1IR(TI = [0.1, 1.0])))
    # a function tie, and a free parameter without a start (no built-in model has either)
    t = sprint(EQUIP._print_parameters, (a = Free(), b = Fixed(p -> 2 * p.a)), (a = nothing, b = :x), "v")
    @test occursin(r"\n  a +free +\[-Inf, Inf\] +required\n  b +fixed +<function> \(tie\) +:x$", t)
    # a custom model prints the same way; also before it has `parameters`, or with
    # acquisition vectors of different lengths (no count then)
    @test startswith(txt(DataStartTest([0.01, 0.02])),
                     "DataStartTest with 2 observations\n  t = [0.01, 0.02]\nParameters:")
    @test txt(NoParametersYet([0.01, 0.02], 3)) ==
          "NoParametersYet with 2 observations\n  TE = [0.01, 0.02]\n  n = 3"
    m = QMTSIR(TI = [0.01, 0.1], TD = [1.0, 2.0, 3.0], magnitude = true)
    @test startswith(txt(m), "QMTSIR\n  TI =")
    # one line (in a vector, interpolated): Julia's default, the constructor call
    @test !occursin("\n", repr(QMTSIR(TI = [0.01, 0.1, 1.0], TD = 2.5)))
end

@testset "ties: by name (Fixed(:R1f)) and by function" begin
    m = QMTSIR(TI = TI5, TD = TD5)
    @test parameters(m).R1m == Fixed(:R1f)                       # the default for QMTSIR
    @test setup(m).fixed.R1m isa EQUIP._Tie{:R1f}       # name known to the compiler
    θfree = merge(defaults(m), (PSR = 0.12, R1f = 1.1, Sf = -0.95, M0f = 1000.0))
    # simulation: tie by name == tie by function == the value written out
    @test signal(m, θfree) == signal(m, merge(θfree, (R1m = p -> p.R1f,))) ==
          signal(m, merge(θfree, (R1m = 1.1,)))
    # fitting: the override forms give the same fit as the default
    y = signal(m, θfree)
    r = fit(m, y)
    @test r.params == fit(m, y; R1m = :R1f).params
    @test r.params.PSR ≈ fit(m, y; R1m = p -> p.R1f).params.PSR rtol = 1e-10
    # a tie can follow a per-voxel map: an R1 map fixes R1f, and R1m follows it
    r1 = [0.7, 1.0, 1.4]
    Y = reduce(vcat, [permutedims(signal(m, merge(θfree, (R1f = v,)))) for v in r1])
    res = fit(m, Y; R1f = r1)
    @test keys(res.params) == (:PSR, :Sf, :M0f)
    @test res.params.PSR ≈ fill(0.12, 3) rtol = 1e-6
    # mistakes are caught before fitting
    @test_throws ArgumentError setup(m; R1m = :R1F)   # not a parameter
    @test_throws ArgumentError setup(m; R1m = :R1m)   # tied to itself
    @test_throws ArgumentError setup(m; Sm = :R1m)    # tied to a tie
    @test_throws ArgumentError signal(m, merge(θfree, (R1m = :nope,)))
end

@testset "parameter roles (setup): keywords named after the parameters" begin
    m = QMTSIR(TI = TI5, TD = TD5)
    s = setup(m)
    @test freenames(s) == (:PSR, :R1f, :Sf, :M0f)
    @test keys(s.fixed) == (:kmf, :R1m, :Sm, :Mzf_td0, :Mzm_td0) && s.fixed.kmf == 12.5
    @test s.lower == [0, 0.05, -1.2, 0] && s.start.PSR == 0.1
    # a value fixes a parameter that is free by default, or changes a fixed value
    @test freenames(setup(m; PSR = 0.1)) == (:R1f, :Sf, :M0f)
    @test setup(m; kmf = 15.0).fixed.kmf == 15.0
    @test setup(m; kmf = Fixed(15.0)).fixed.kmf == 15.0
    # Free() estimates a fixed parameter: its fixed value becomes the start, and its bounds are
    # the ones its Fixed declares for when it is freed (QMTSIR's kmf ≥ 0), or none
    s = setup(m; kmf = Free())
    @test freenames(s) == (:PSR, :R1f, :Sf, :M0f, :kmf) && s.start.kmf == 12.5
    @test s.lower[end] == 0 && s.upper[end] == Inf
    s = setup(m; kmf = Free(start = 20.0, lower = 5, upper = 50))   # with start and bounds
    @test s.lower[end] == 5 && s.upper[end] == 50 && s.start.kmf == 20
    s = setup(m; kmf = Free(start = 20.0))                         # a start alone keeps the bound
    @test s.lower[end] == 0 && s.start.kmf == 20
    @test setup(m; kmf = Free(lower = -Inf)).lower[end] == -Inf     # removed
    @test setup(m; Mzf_td0 = Free()).lower[end] == -Inf            # declares none: no bound
    @test setup(m; kmf = 15.0).fixed.kmf == 15.0                   # the bounds only apply when freed
    # Free(...) over a free parameter keeps whatever it leaves out
    s = setup(m; Sf = Free(lower = -1.5))
    @test s.lower[3] == -1.5 && s.upper[3] == 0 && s.start.Sf == -0.95
    s = setup(m; R1f = Free(start = 0.8))
    @test s.start.R1f == 0.8 && s.lower[2] == 0.05 && s.fromdata == (false, false, false, false)
    @test Free(start = 0.8).start === 0.8 && Free(start = 1).start === 1.0   # numbers: Float64
    @test Free(start = :data).start === :data && Free(start = BestOf([0.3, 0.8])).start isa BestOf
    @test_throws ArgumentError Free(start = :auto)                  # not a start option
    # the start is a keyword, like the bounds: a positional one says how to write it
    @test_throws "write Free(start = 0.8)" Free(0.8)
    @test_throws "write Free(start = :scale, lower = 0)" Free(:scale; lower = 0)
    @test_throws "write Free(start = 0.1, lower = 0, upper = 1)" Free(0.1, 0, 1)
    # starts and precision; an explicit start replaces :scale
    mfa = T1MFA(FA = [2, 10], TR = 0.015)
    s = setup(mfa, Float32; T1 = Free(start = 2))
    @test s.start == (M0 = 1.0f0, T1 = 2.0f0)          # M0 = 1: placeholder for :scale
    @test s.scale == 1                                  # M0 is the :scale parameter
    @test setup(mfa; M0 = Free(start = 500.0)).scale == 0
    @test setup(m).scale == 4                           # M0f
    # mistakes are caught, with the parameters and options listed
    @test_throws ArgumentError setup(m; kmff = 1.0)                  # misspelled
    @test_throws ArgumentError setup(m; R1m = Free())                # a tie freed needs a start
    @test_throws ArgumentError setup(m; mask = trues(3))             # mask is for datasets
    @test_throws "solver = LM(maxiter" setup(m; maxiter = 50)        # solver settings go in LM
    ir = T1IR(TI = [0.1, 0.5, 1.0])
    @test_throws "M0 in T1IR" setup(ir; T1 = Free(start = :scale))                # only the model's scale
    @test_throws "M0 in T1IR" setup(ir; T1 = Free(start = :scale), M0 = Free(start = 1000.0))
    @test setup(ir; M0 = Free(start = 1000.0)).scale == 0                        # turned off: allowed
    @test setup(ir; M0 = Free(start = :scale)).scale == 1                        # the model's own: allowed
    @test_throws ArgumentError setup(T2Decay(TE = [0.01]); S0 = 1.0, T2 = 0.1)   # nothing free
    e = try setup(m; kmff = 1.0) catch err err end
    @test occursin("PSR, R1f", e.msg) && occursin("mask, precision, solver", e.msg)
end

@testset "parameters(m; changes...): roles made by a call, kept and passed" begin
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    m = QMTSIR(; SIR_4POINT...)
    θ = nominal(m, :SIR_WM_I)
    roles = parameters(m; kmf = 10.0, Sm = 0.9)
    # every parameter's role, the two changed by the same rules as fit's keywords
    @test keys(roles) == keys(parameters(m))
    @test roles.kmf isa Fixed && roles.kmf.value == 10.0 && roles.Sm.value == 0.9
    @test roles.PSR == parameters(m).PSR && roles.R1m == parameters(m).R1m
    @test parameters(m; kmf = Free()).kmf == Free(start = 12.5, lower = 0)   # freed: its value, its bound
    @test parameters(m; R1f = Free(lower = 0.5)).R1f == Free(start = 1.0, lower = 0.5, upper = 10.0)
    @test parameters(T1IR(TI = [0.1, 0.5, 1.0, 2.0]); T1 = Free(start = 1.0)).T1.start == 1.0
    # a model with its own `parameters(m)` method: its roles, changed
    dm = DataStartTest([0.01, 0.02])
    @test parameters(dm; c = 2.0).c.value == 2.0 && parameters(dm; c = 2.0).T2 == parameters(dm).T2
    # passed whole, after the data or the truth: the same as the keywords
    y = simulate(m, θ; noise = :rician, snr = 200, rng = MersenneTwister(1))
    Y = simulate(m, θ; noise = :rician, snr = 200, n = 20, rng = MersenneTwister(2))
    kw = (kmf = 10.0, Sm = 0.9)
    @test fit(m, y, roles).params == fit(m, y; kw...).params
    @test isequal(fit(m, Y, roles).params, fit(m, Y; kw...).params)
    @test fit!(init(m, roles), y).params == fit(m, y; kw...).params
    # a scenario's fit, fitted to data
    s = Scenario(m, θ; kw...)
    @test fit(m, y, s.roles).params == fit(m, y; kw..., NamedTuple{(:PSR, :R1f, :Sf, :M0f)}(
        map(n -> Free(start = θ[n]), (:PSR, :R1f, :Sf, :M0f)))...).params
    # mistakes
    @test occursin("`kfm` is not a parameter of QMTSIR", msg(() -> parameters(m; kfm = 1.0)))
    @test occursin("role", msg(() -> crlb(m, roles; snr = 200)))   # roles where the truth goes
end

@testset "fit, fit! and nobs are StatsAPI's functions (no clash with StatsBase etc.)" begin
    @test nameof(parentmodule(fit)) == :StatsAPI
    @test nameof(parentmodule(fit!)) == :StatsAPI
    @test nameof(parentmodule(nobs)) == :StatsAPI
    @test nameof(parentmodule(init)) == :CommonSolve
end

@testset "every exported name has a docstring (`?stick` at the REPL)" begin
    # A docstring documents only the expression right after it: with a comment or a blank line
    # between, Julia discards it (VS Code: "detached_docstring"). `stick`, `zeppelin` and
    # `nominal` lost theirs that way until 2026-11-12, and `SIR_4POINT`, `SIR_16POINT` and
    # `powder` had none of their own. Docs.undocumented_names: Julia 1.11 on, as EQUIP.
    @test isempty(Docs.undocumented_names(EQUIP))
end
