# =================================================================================================
# Two-pool (qMT) relaxation and exchange on static 2×2 matrices: the equations the models write
# in their own signals (qmt_spgr.jl, qmt_cw.jl), and the one piece they share, the 2×2 matrix
# exponential in closed form (a mathematical tool, as `inner`; the physics stays in each model).
#
# The longitudinal magnetizations of the free (f) and macromolecular (m) pools, M = [Mzf, Mzm], in
# units of the free pool's equilibrium M0f (so M0 = [1, PSR]), follow the Bloch–McConnell equations
# without transverse terms (Henkelman et al., Magn Reson Med 1993):
#
#   dM/dt = A·M + R1·M0 − W·M
#
#   A  = [ −(R1f + kfm)       kmf      ]     relaxation and exchange; kfm = kmf·PSR (mass balance)
#        [      kfm      −(R1m + kmf)  ]
#   R1·M0 = [R1f, R1m·PSR]                   recovery toward equilibrium (A·M0 = −R1·M0)
#   W  = diag(Wf, Wm)                         saturation by RF, π·ω1²·g(Δ) per pool (lineshapes.jl)
#
# Over a time t with W constant, M relaxes toward its steady state Mss = (W − A)⁻¹·R1·M0:
#
#   M(t) = E·M(0) + (I − E)·Mss,   E = exp((A − W)·t)
#
# (W = 0: Mss = M0, free recovery). The 2×2 exponential in closed form: with τ = tr(X)/2 and
# δ² = ((X₁₁ − X₂₂)/2)² + X₁₂·X₂₁, the eigenvalues are τ ± δ, and (Cayley–Hamilton)
#
#   exp(X) = e^τ·[cosh δ·I + (sinh δ/δ)·(X − τI)]
#
# For A − W the off-diagonals are both ≥ 0 (rates), so δ² ≥ 0 and the eigenvalues are real;
# δ² < 0 (complex eigenvalues) is taken too, by cos and sin, and δ² near 0 by the series of cosh
# and sinh(δ)/δ in δ², so a dual number passes through (no √ of 0). Equal to LinearAlgebra's
# `exp` to 2e-16 (checked in Octave against `expm` on the lab's matrices, 2026-11-26; in the tests
# against `exp` on random matrices). The lab's MATLAB (yarnykh_pulseMT.m) took exponentials by
# `eig`, which has no method for dual numbers.
# =================================================================================================

# cosh δ and sinh(δ)/δ as functions of q = δ²: by cos and sin for q < 0 (δ = i·√−q), by their
# series near 0 (to q³; the next term, q⁴/8! or q⁴/9!, below 3e-29 for |q| < 1e-6)
@inline function _coshsinhc(q)
    pq = _plain(q)
    if abs(pq) < 1e-6
        # integer divisors, so a Float32 (or its dual numbers) stays one: the branches agree in type
        return 1 + q / 2 + q^2 / 24 + q^3 / 720, 1 + q / 6 + q^2 / 120 + q^3 / 5040
    elseif pq > 0
        δ = sqrt(q)
        return cosh(δ), sinh(δ) / δ
    else
        δ = sqrt(-q)
        return cos(δ), sin(δ) / δ
    end
end

# exp(X) for a static 2×2 matrix X (above)
@inline function _expm2(X::SMatrix{2,2})
    a, c, b, d = X                                   # column-major: X₁₁, X₂₁, X₁₂, X₂₂
    τ = (a + d) / 2
    ch, sh = _coshsinhc(((a - d) / 2)^2 + b * c)
    e = exp(τ)
    return SMatrix{2,2}(e * (ch + sh * (a - τ)), e * (sh * c), e * (sh * b), e * (ch + sh * (d - τ)))
end
