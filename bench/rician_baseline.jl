# =================================================================================================
# The Rician baseline: what the noise floor does to least-squares fits, and what the fits with the
# Rician noise model (`likelihood = :rician`) do instead
#
#   julia -t auto --project=. bench/rician_baseline.jl            (about 5 minutes on 4 threads)
#   N=200 SNRS=10,50 julia -t auto --project=. bench/rician_baseline.jl     (a quick look)
#   ONLY="T1IR,DTI CSF" julia -t auto --project=. bench/rician_baseline.jl   (cases by label)
#
# Step 0 of docs/design.md's "Tier 2, bias from noise at low SNR: the plan": every later step
# (the `likelihood` switch, maximum likelihood) is judged against this table. Magnitude data with
# Rician noise, N voxels a case, SNR = S0/σ (or M0/σ), the unweighted signal over the noise SD, as
# `simulate`'s `snr`; per model, a tissue and CSF. Each image's own SNR, A/σ, is lower where the
# signal has decayed, and it is what the floor's bias depends on (one image's mean +1% at A/σ = 7,
# +6% at 3, +55% at 1): each case's header prints its median and the share of images below 3.
#
#   T2Decay   32 echoes, TE 10–320 ms; WM (T2 69 ms) and CSF (T2 1.75 s; Stanisz et al. 2005)
#   ADCDecay  b 0–3000 s/mm² in steps of 500; tissue (ADC 0.8e-3 mm²/s) and CSF (3.0e-3)
#   T1IR      magnitude, 9 TIs from 50 ms to 5 s; WM (T1 1.084 s) and CSF (4.124 s), so some TIs
#             sit near the inversion's null
#   DTI       4 b = 0, 64 directions at b = 1000 and at 2500; WM (DKI_WM's MD and FA, Maiter et
#             al. 2021, a random fiber direction per voxel) and CSF (3.0e-3, isotropic)
#   DKI       6 b = 0, 30 directions at 1000 and at 2000; DKI_WM (random directions) and DKI_CSF
#
# The fits, each from the same data, with the models' default starts and bounds (the diffusivity
# at most 4e-3 mm²/s and T1 at most 5 s since 2026-11-09: the first run, recorded in design.md,
# also had them without):
#
#   gauss       least squares (the fit without `likelihood`)
#   mean        likelihood = :rician, method = :mean: least squares on the Rician mean, with the
#               true σ, from the least-squares fit's values (step 0 did this by hand, RicianMean)
#   wmean       method = :wmean: the same, weighted by 1/Var
#   ml          method = :ml: the full Rician model, maximum likelihood
#
# For each, the quantity users read (T2, ADC, T1; MD and FA; MD and MKT): bias, SD and RMSE in %
# of the truth (for a truth of 0, FA and MKT in CSF: their mean and SD), the % of voxels
# converged, and µs per voxel on this machine's threads. Below them, crlb's predictions for one
# voxel: least squares on the magnitudes (`noise = :rician`: the floor's bias and the sandwich), least
# squares on the Rician mean unweighted and weighted by 1/Var, and the Rician bound
# (`ricianbound`): the plan's question of whether the weighted fit reaches the bound.
#
# Not here: σ estimated (the true σ is given).
# =================================================================================================

using EQUIP, LinearAlgebra, StaticArrays, Statistics, Random, Printf

const N = parse(Int, get(ENV, "N", "2000"))
const SNRS = parse.(Int, split(get(ENV, "SNRS", "5,10,20,50"), ","))
const ONLY = split(get(ENV, "ONLY", ""), ","; keepempty = false)   # label prefixes; none: all

# RicianMean passes the `derived` hook through (since tier 2's step 1; the first run, on the
# commit before, defined it here), so the Rician fits' MD, FA and MKT are read like the Gaussian
# fits'.

# ------------------------------------------------------------------------------------------------
# The protocols and truths

const TE = collect(0.01:0.01:0.32)
const BADC = collect(0.0:500.0:3000.0)
const TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]

function encoding(nb0, ndirs, shells)
    g = directionset(ndirs)
    bval = [fill(0.0, nb0); reduce(vcat, [fill(b, ndirs) for b in shells])]
    bvec = reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), nb0); reduce(vcat, [g for _ in shells])])
    return diffusionencoding(bval, bvec)
end

# A random fiber direction (uniform on the sphere).
function randomdirection(rng)
    u = 2rand(rng) - 1
    return fiberdirection(acos(u), 2π * rand(rng))
end

# Truth maps from one NamedTuple per voxel: a NamedTuple of vectors, what `simulate` takes.
asmaps(ts) = NamedTuple{keys(first(ts))}(Tuple([t[k] for t in ts] for k in keys(first(ts))))

# Each case: label, model, truth (one NamedTuple, or one per voxel), the quantities reported.
function cases(rng)
    dti = DTI(; encoding(4, 64, (1000.0, 2500.0))...)
    dki = DKI(; encoding(6, 30, (1000.0, 2000.0))...)
    wm = TISSUES.DKI_WM
    return [
        ("T2Decay WM", T2Decay(TE = TE), (S0 = 1000.0, T2 = TISSUES.WM.T2), (:T2,)),
        ("T2Decay CSF", T2Decay(TE = TE), (S0 = 1000.0, T2 = TISSUES.CSF.T2), (:T2,)),
        ("ADCDecay tissue", ADCDecay(b = BADC), (S0 = 1000.0, ADC = 0.8e-3), (:ADC,)),
        ("ADCDecay CSF", ADCDecay(b = BADC), (S0 = 1000.0, ADC = 3.0e-3), (:ADC,)),
        ("T1IR WM", T1IR(TI = TI), nominal(T1IR(TI = TI); M0 = 1000.0, T1 = TISSUES.WM.T1, Sf = -1.0),
         (:T1,)),
        ("T1IR CSF", T1IR(TI = TI), nominal(T1IR(TI = TI); M0 = 1000.0, T1 = TISSUES.CSF.T1, Sf = -1.0),
         (:T1,)),
        ("DTI WM", dti, [nominal(DTI; S0 = 1000.0, MD = wm.MD, FA = wm.FA,
                                 direction = randomdirection(rng)) for _ in 1:N], (:MD, :FA)),
        ("DTI CSF", dti, nominal(DTI; S0 = 1000.0, MD = 3.0e-3, FA = 0.0,
                                 direction = fiberdirection(0, 0)), (:MD, :FA)),
        ("DKI WM", dki, [nominal(DKI, :DKI_WM; S0 = 1000.0, direction = randomdirection(rng))
                         for _ in 1:N], (:MD, :MKT)),
        ("DKI CSF", dki, nominal(DKI, :DKI_CSF; S0 = 1000.0, direction = fiberdirection(0, 0)),
         (:MD, :MKT)),
    ]
end

# ------------------------------------------------------------------------------------------------
# Fitting and measuring

# The values of quantity q in a fit (a parameter or a derived quantity), one per voxel.
function quantity(res, q)
    q in keys(res.params) && return vec(res.params[q])
    return vec(derived(res)[q])
end

struct Row
    fit::String
    q::Symbol
    stats::NTuple{3,Float64}      # bias, SD, RMSE (in % of the truth; or mean, SD, RMS for 0)
    relative::Bool
    converged::Float64
    us::Float64
end

function measure(name, res, truth, qs, t)
    rows = Row[]
    ok = isconverged.(vec(res.status))
    for q in qs
        v = quantity(res, q)
        v = v[isfinite.(v)]
        tq = truth[q]
        if iszerotruth(tq)
            push!(rows, Row(name, q, (mean(v), std(v), sqrt(mean(abs2, v))), false, mean(ok), t))
        else
            e = (v .- tq) ./ tq
            push!(rows, Row(name, q, (100mean(e), 100std(e), 100sqrt(mean(abs2, e))), true,
                            mean(ok), t))
        end
    end
    return rows
end

# A truth of 0 (FA and MKT in CSF; FA computed from an isotropic tensor is 0 to rounding, 1e-16):
# reported as a mean and SD, not in % of it.
iszerotruth(x) = abs(x) < 1e-9

# The truth's quantities (parameters and derived), for one voxel's truth.
function truthof(m, θ, qs)
    d = derived(m, θ)
    return NamedTuple(q => (q in keys(θ) ? θ[q] : d[q]) for q in qs)
end

function timed(f)
    f(1:min(N, 20))                              # compile on a few voxels first
    t = @elapsed r = f(1:N)
    return r, 1e6 * t / N
end

function runcase(io, label, m, truth, qs, snr)
    # Each case's noise from its own seed, so a case run alone (ONLY) gets the data it gets in the
    # full run. (Since 2026-11-09, after the first run recorded in design.md, whose noise came from
    # one stream: a rerun differs from that table within Monte Carlo error.)
    rng = MersenneTwister(1000snr + sum(codeunits(label)))
    θ1 = truth isa AbstractVector ? first(truth) : truth
    tq = truthof(m, θ1, qs)
    S0 = haskey(θ1, :S0) ? θ1.S0 : θ1.M0
    σ = S0 / snr
    Y = truth isa AbstractVector ?
        simulate(m, asmaps(truth); snr, noise = :rician, rng) :
        simulate(m, θ1; snr, noise = :rician, n = N, rng)
    rm = RicianMean(m; σ)
    rows = Row[]

    g, tg = timed(v -> fit(m, Y[v, :]))
    append!(rows, measure("gauss", g, tq, qs, tg))
    for method in (:mean, :wmean, :ml)
        r, tr = timed(v -> fit(m, Y[v, :]; likelihood = :rician, σ, method))
        append!(rows, measure(String(method), r, tq, qs, tr))
    end

    a = abs.(signal(m, θ1)) ./ σ                        # each image's own SNR, A/σ
    @printf(io, "\n%s, SNR %d = S0/σ; per image A/σ median %.2f (%.2f–%.2f), %.0f%% below 3   (truth: %s)\n",
            label, snr, median(a), minimum(a), maximum(a), 100mean(a .< 3),
            join(("$q = $(round(tq[q], sigdigits = 4))" for q in qs), ", "))
    println(io, "  fit            quantity     bias %     SD %   RMSE %   converged   µs/voxel")
    for row in rows
        b, s, e = row.stats
        row.relative ?
            @printf(io, "  %-13s  %-8s  %8.2f %8.2f %8.2f   %7.1f%%   %8.1f\n", row.fit, row.q,
                    b, s, e, 100row.converged, row.us) :
            @printf(io, "  %-13s  %-8s  (truth 0) mean %.3g  SD %.3g  RMS %.3g   %5.1f%%   %8.1f\n",
                    row.fit, row.q, b, s, e, 100row.converged, row.us)
    end

    # crlb's predictions for this voxel (θ1): bias and SD in % (or absolute for a truth of 0).
    A = abs.(signal(m, θ1))
    v = ricianvar.(A, σ)
    preds = [
        "crlb gauss" => () -> crlb(m, θ1; snr, noise = :rician),
        "crlb rician" => () -> crlb(rm, θ1; Σ = Diagonal(v)),
        "crlb rician, 1/Var" => () -> crlb(rm, θ1; Σ = Diagonal(v), weights = 1 ./ v),
        "ricianbound" => () -> ricianbound(m, θ1; snr),
    ]
    for (name, f) in preds
        c = try
            f()
        catch err
            @printf(io, "  %-19s  failed: %s\n", name, sprint(showerror, err)[1:min(end, 70)])
            continue
        end
        for q in qs
            iszerotruth(tq[q]) ?
                @printf(io, "  %-19s  %-8s  bias %.3g  SD %.3g\n", name, q, c.bias[q], c.sd[q]) :
                @printf(io, "  %-19s  %-8s  bias %7.2f%%  SD %7.2f%%\n", name, q,
                        100c.bias[q] / tq[q], 100c.sd[q] / tq[q])
        end
    end
    flush(io)
end


function main(io = stdout)
    rng = MersenneTwister(2026_11_09)
    @printf(io, "Rician baseline: N = %d voxels a case, %d threads, Julia %s, %s\n", N,
            Threads.nthreads(), VERSION, Sys.cpu_info()[1].model)
    t = @elapsed for (label, m, truth, qs) in cases(rng), snr in SNRS
        isempty(ONLY) || any(o -> startswith(label, o), ONLY) || continue
        runcase(io, label, m, truth, qs, snr)
    end
    @printf(io, "\n%.0f s in all\n", t)
end

main()
