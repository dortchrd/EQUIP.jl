# =================================================================================================
# test/validation/rician_fits.jl: the Rician fits (`likelihood = :rician`, maximum likelihood, the
# default method) per model, at 2000–5000 copies: the floor's bias removed against least squares
# on the same data, the SD at the Rician bound (`ricianbound`, crlb's prediction for the fit), the
# cases with no information ending on their caps rather than running away, and a σ map. Step 0's
# protocols and truths (bench/rician_baseline.jl; docs/design.md, "Step 0, measured").
#
# What is asserted is what holds (measured 2026-11-13, docs/design.md "Step 6, built and
# measured"). The fit's own second-order bias remains (about 1% at SNR 10, falling as σ²), so the
# bias is not asserted to be 0; and where it is as large as the floor's (T1 inversion recovery in
# WM, DKI's MKT) maximum likelihood is not better than least squares: recorded, not asserted.
# DKI in CSF is left out: its kurtosis has no information there, and the fit stops at its
# iteration limit (roadmap: design.md item 20).
#
# Part of EQUIP's validation group, run apart (about two minutes):
#     julia -t auto --project=test test/runtests.jl rician_fits
# =================================================================================================

const _TE = collect(0.01:0.01:0.32)
const _BADC = collect(0.0:500.0:3000.0)
const _TI = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0]

# b = 0 images and directions on shells, as a directional model's b, gx, gy, gz
function _encoding(nb0, ndirs, shells)
    g = directionset(ndirs)
    bval = [fill(0.0, nb0); reduce(vcat, [fill(b, ndirs) for b in shells])]
    bvec = reduce(hcat, [fill(SVector(0.0, 0.0, 0.0), nb0); reduce(vcat, [g for _ in shells])])
    return diffusionencoding(bval, bvec)
end

# Least squares and the Rician fit on the same data (the same seed)
_pair(m, θ, snr, n) = (montecarlo(m, θ; snr, noise = :rician, n, rng = MersenneTwister(1)),
                       montecarlo(m, θ; snr, noise = :rician, likelihood = :rician, n,
                                  rng = MersenneTwister(1)))

@testset "Rician fits in tissue: the floor's bias removed, the SD at the Rician bound" begin
    dti = DTI(; _encoding(4, 64, (1000.0, 2500.0))...)
    dki = DKI(; _encoding(6, 30, (1000.0, 2000.0))...)
    wm, dir = TISSUES.DKI_WM, fiberdirection(0.7, 0.3)
    # Each case's quantities, and whether the floor dominates the least-squares bias there (then
    # the Rician fit's is under a third of it, and halves or better from SNR 10 to 20). Measured,
    # least squares → Rician at SNR 10 and 20 (%): T2 +41.2 → +0.76, +10.0 → +0.19; ADC −7.7 →
    # +1.84, −1.9 → +0.34; MD −8.7 → +0.48, −2.1 → +0.11; FA −4.7 → +0.86, −1.2 → +0.19. SD ÷
    # bound 0.96–1.04 everywhere asserted (DKI's MKT 1.38 at SNR 10 and T1's bias +2.9% against
    # −3.4%: recorded, not asserted).
    cases = [("T2Decay WM", T2Decay(TE = _TE), (S0 = 1000.0, T2 = TISSUES.WM.T2), 5000,
              [(:T2, true)]),
             ("ADCDecay tissue", ADCDecay(b = _BADC), (S0 = 1000.0, ADC = 0.8e-3), 5000,
              [(:ADC, true)]),
             ("T1IR WM", T1IR(TI = _TI), nominal(T1IR(TI = _TI); M0 = 1000.0, T1 = TISSUES.WM.T1,
                                                  Sf = -1.0), 5000, [(:T1, false)]),
             ("DTI WM", dti, nominal(DTI; S0 = 1000.0, MD = wm.MD, FA = wm.FA, direction = dir),
              2000, [(:MD, true), (:FA, true)]),
             ("DKI WM", dki, nominal(DKI, :DKI_WM; S0 = 1000.0, direction = dir), 2000,
              [(:MD, false)])]
    for (name, m, θ, n, qs) in cases
        ls10, ml10 = _pair(m, θ, 10, n)
        ls20, ml20 = _pair(m, θ, 20, n)
        @testset "$name" begin
            @test ml10.converged >= 0.995 && ml20.converged >= 0.995
            for (q, floor) in qs, (ls, ml) in ((ls10, ml10), (ls20, ml20))
                @test abs(ml.sd[q] / ml.crlb.sd[q] - 1) < 0.06            # at the Rician bound
                floor && @test abs(ml.bias[q]) < abs(ls.bias[q]) / 3      # the floor's bias gone
            end
            for (q, floor) in qs                                          # what remains falls
                floor && @test abs(ml20.bias[q]) < abs(ml10.bias[q]) / 2  # with the SNR (as σ²)
            end
        end
    end
end

@testset "Rician fits without information (CSF): converged, on their caps, not running away" begin
    # The signal at the highest b or TI is at the noise floor: the fits converge and many voxels
    # stop at a cap (D ≤ 4e-3 mm²/s, T1 ≤ 5 s) rather than running away, as they did before the
    # caps (MD +20 000% in DTI's CSF, step 0). Measured at SNR 10: converged 100%, 100%, 99.1%;
    # on a bound 1225 of 5000, 3668 of 5000, 1887 of 2000; bias +2.1% against least squares'
    # −20.2% (ADC), +13.8% against −37.2% (DTI's MD), +3.2% against +2.1% (T1: not better).
    dti = DTI(; _encoding(4, 64, (1000.0, 2500.0))...)
    for (name, m, θ, q, n, better) in
            (("ADCDecay CSF", ADCDecay(b = _BADC), (S0 = 1000.0, ADC = 3.0e-3), :ADC, 5000, true),
             ("T1IR CSF", T1IR(TI = _TI), nominal(T1IR(TI = _TI); M0 = 1000.0, T1 = TISSUES.CSF.T1,
                                                   Sf = -1.0), :T1, 5000, false),
             ("DTI CSF", dti, nominal(DTI; S0 = 1000.0, MD = 3.0e-3, FA = 0.0,
                                      direction = fiberdirection(0, 0)), :MD, 2000, true))
        ls, ml = _pair(m, θ, 10, n)
        ob = onbound(ml.result)                                           # one map per parameter
        ncap = count(i -> any(v[i] for v in values(ob)), eachindex(first(values(ob))))
        @testset "$name" begin
            @test ml.converged >= 0.98
            @test ncap >= n / 10                                          # stopped at the caps
            better && @test abs(ml.bias[q]) < abs(ls.bias[q])
        end
    end
end

@testset "A σ map: each region's floor removed with its own σ" begin
    # Two regions of 3000 voxels, σ = 50 and 100 (SNR 20 and 10): with the map, both near 0
    # (measured +0.05% and +1.28%, the second the fit's own second-order bias at SNR 10); with one
    # σ = 75 for both, −6.5% and +13.7%; least squares +9.8% and +41.7%.
    m, θ = T2Decay(TE = _TE), (S0 = 1000.0, T2 = TISSUES.WM.T2)
    Y = vcat(simulate(m, θ; σ = 50.0, noise = :rician, n = 3000, rng = MersenneTwister(3)),
             simulate(m, θ; σ = 100.0, noise = :rician, n = 3000, rng = MersenneTwister(4)))
    halves(r) = (mean(r.params.T2[1:3000]) / θ.T2 - 1, mean(r.params.T2[3001:end]) / θ.T2 - 1)
    withmap = halves(fit(m, Y; likelihood = :rician, σ = [fill(50.0, 3000); fill(100.0, 3000)]))
    withone = halves(fit(m, Y; likelihood = :rician, σ = 75.0))
    @test all(b -> abs(b) < 0.02, withmap)
    @test maximum(abs, withone) > 0.05
end
