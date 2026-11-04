# =================================================================================================
# Models written from the building blocks (EQUIP.Blocks), each fitted to simulated data
#
#   decay(x, rate)  ball(b, D)  stick(b, D, view)  zeppelin(b, D∥, D⊥, view)   view: powder or a cosine
#
# `@mrimodel` turns one observation's signal into a model (src/models/macro.jl): the positional
# arguments are the acquisition, the keywords the parameters (Free(...) is fitted, a number is
# fixed). The blocks make the body read like the paper's equation.
#
#   1. T2 decay, decay(TE, 1/T2): T2 is fitted; the same numbers as the built-in T2Decay, bit for bit
#   2. inversion recovery, magnitude
#   3. three-pool T2 (myelin water, intra/extracellular water, CSF), CSF's T2 known
#   4. IVIM: perfusion and diffusion, two balls
#   5. SMT with a T2 per compartment, in one line
#   6. a nerve along a known direction (z): sticks and a zeppelin along each gradient direction,
#      no spherical mean (the directional view of the blocks)
#
# Run from the repository folder (about 30 s):
#   julia --project=. examples/building_blocks.jl
# =================================================================================================

using EQUIP, EQUIP.Blocks, Random, Printf
rng = MersenneTwister(7)
show_fit(label, r, truth) = println(rpad(label, 34), join([@sprintf("%s %.4g (truth %.4g)", k, v, truth[k])
                                                             for (k, v) in pairs(r.params)], ", "))

# --- 1. T2 decay in the T2 form ---------------------------------------------------------------------
@mrimodel MyT2(TE; S0 = Free(start = :scale, lower = 0), T2 = Free(start = 0.05, lower = 1e-6)) =
    S0 * decay(TE, 1 / T2)                                   # the rate: 1/T2

te = collect(0.01:0.01:0.32)
y = simulate(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); σ = 5.0, noise = :gaussian, rng)
mine = fit(MyT2(TE = te), y)
builtin = fit(T2Decay(TE = te), y; T2 = Free(start = 0.05))     # the same start
show_fit("1. T2 decay, decay(TE, 1/T2):", mine, (S0 = 1000.0, T2 = 0.08))
println("   the same as T2Decay, bit for bit: ", mine.params == builtin.params)

# --- 2. Inversion recovery, magnitude ---------------------------------------------------------------
@mrimodel MyIR(TI; M0 = Free(start = :scale, lower = 0), Sf = Free(start = -0.9, lower = -1.2, upper = 0),
               T1 = Free(start = 1.0, lower = 1e-6)) =
    abs(M0 * (1 + (Sf - 1) * decay(TI, 1 / T1)))              # Sf = −1: a perfect inversion

ti = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
θ = (M0 = 1000.0, Sf = -0.95, T1 = 0.9)
show_fit("2. inversion recovery:", fit(MyIR(TI = ti), simulate(MyIR(TI = ti), θ; σ = 2.0, noise = :gaussian, rng)), θ)

# --- 3. Three T2 pools, CSF's T2 known --------------------------------------------------------------
@mrimodel ThreePool(TE; S0 = Free(start = :scale, lower = 0),
                    fmw = Free(start = 0.1, lower = 0, upper = 0.5),      # myelin water
                    fcsf = Free(start = 0.05, lower = 0, upper = 0.5),    # CSF
                    T2mw = Free(start = 0.015, lower = 0.005, upper = 0.04),
                    T2ie = Free(start = 0.08, lower = 0.04, upper = 0.2),
                    T2csf = 2.0) =                                        # a number: fixed
    S0 * (fmw * decay(TE, 1 / T2mw) + (1 - fmw - fcsf) * decay(TE, 1 / T2ie) + fcsf * decay(TE, 1 / T2csf))

te48 = collect(0.008:0.008:0.384)                 # 48 echoes, as a multi-echo CPMG
θ = (S0 = 1000.0, fmw = 0.15, fcsf = 0.1, T2mw = 0.018, T2ie = 0.075, T2csf = 2.0)
show_fit("3. three pools (SNR 1000):", fit(ThreePool(TE = te48), simulate(ThreePool(TE = te48), θ; σ = 1.0, noise = :gaussian, rng)), θ)

# --- 4. IVIM: perfusion and diffusion ---------------------------------------------------------------
@mrimodel IVIM(b; S0 = Free(start = :scale, lower = 0), f = Free(start = 0.1, lower = 0, upper = 0.5),
               Dstar = Free(start = 0.02, lower = 0.003, upper = 0.1), D = Free(start = 1e-3, lower = 0, upper = 3e-3)) =
    S0 * (f * ball(b, Dstar) + (1 - f) * ball(b, D))

bv = [0.0, 10, 20, 40, 60, 100, 200, 400, 600, 800, 1000]
θ = (S0 = 1000.0, f = 0.12, Dstar = 0.025, D = 0.9e-3)
show_fit("4. IVIM:", fit(IVIM(b = bv), simulate(IVIM(b = bv), θ; σ = 1.0, noise = :gaussian, rng)), θ)

# --- 5. SMT with a T2 per compartment, in one line --------------------------------------------------
@mrimodel MySMTT2(b, TE; S0 = Free(start = :scale, lower = 0), vax = Free(start = 0.5, lower = 0, upper = 1),
                  Dax = Free(start = 2e-3, lower = 0, upper = 3e-3),
                  T2ax = Free(start = 0.08, lower = 1e-3, upper = 1), T2ex = Free(start = 0.05, lower = 1e-3, upper = 1)) =
    S0 * (vax * decay(TE, 1 / T2ax) * stick(b, Dax, powder) +
          (1 - vax) * decay(TE, 1 / T2ex) * zeppelin(b, Dax, (1 - vax) * Dax, powder))

bs = repeat([0.0, 1000, 2500, 4000], 2); tes = [fill(0.07, 4); fill(0.12, 4)]
θ = (S0 = 1.0, vax = 0.6, Dax = 2e-3, T2ax = 0.09, T2ex = 0.05)
ys = simulate(MySMTT2(b = bs, TE = tes), θ; σ = 1e-4, noise = :gaussian, rng)
show_fit("5. SMT with T2s (SNR 10000):", fit(MySMTT2(b = bs, TE = tes), ys), θ)
println("   against the built-in SMTT2: ", round(maximum(abs, signal(MySMTT2(b = bs, TE = tes), θ) .-
                                                  signal(SMTT2(b = bs, TE = tes), θ)); sigdigits = 2), " largest difference in signal")

# --- 6. A nerve along z: one direction at a time, no spherical mean ---------------------------------
# Peripheral nerve runs roughly along the bore: with the fibres' direction known, each gradient
# direction g sees the cosine c = g·z, and the directional view of the blocks models every image.
@mrimodel NerveZ(b, c; S0 = Free(start = :scale, lower = 0), vax = Free(start = 0.5, lower = 0, upper = 1),
                 Dax = Free(start = 2e-3, lower = 0, upper = 3e-3)) =
    S0 * (vax * stick(b, Dax, c) + (1 - vax) * zeppelin(b, Dax, (1 - vax) * Dax, c))

g = directionset(30)                               # 30 directions, repeated on each shell
bn = [fill(0.0, 4); fill(1000.0, 30); fill(2500.0, 30)]
cn = [fill(1.0, 4); [d[3] for d in g]; [d[3] for d in g]]     # cosine with z (any value at b = 0)
θ = (S0 = 1000.0, vax = 0.6, Dax = 1.9e-3)
show_fit("6. nerve along z, 64 images:", fit(NerveZ(b = bn, c = cn), simulate(NerveZ(b = bn, c = cn), θ; σ = 10.0, noise = :gaussian, rng)), θ)

# --- What the T2 form costs ------------------------------------------------------------------------
# decay(TE, 1/T2): 1/T2 is the same for every echo, so the compiler computes it once per voxel, as
# the built-in T2Decay does by hand.
Y = simulate(T2Decay(TE = te), (S0 = 1000.0, T2 = 0.08); σ = 5.0, noise = :gaussian, n = 20_000, rng)
fit(MyT2(TE = te), Y); fit(T2Decay(TE = te), Y; T2 = Free(start = 0.05))
t_mine = minimum(@elapsed(fit(MyT2(TE = te), Y)) for _ in 1:5) / 20_000 * 1e6
t_builtin = minimum(@elapsed(fit(T2Decay(TE = te), Y; T2 = Free(start = 0.05))) for _ in 1:5) / 20_000 * 1e6
@printf("\nfit µs per voxel, 32 echoes, %d threads: decay(TE, 1/T2) %.2f, T2Decay %.2f\n", Threads.nthreads(), t_mine, t_builtin)
