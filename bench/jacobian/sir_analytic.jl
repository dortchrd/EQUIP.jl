# QMTSIREqualR1 with its Jacobian written out (prototype, a scratch override of the solver's
# `normal_equations` for this model): the same sums, SSE, JᵀWJ and JᵀWr, from closed-form
# derivatives instead of dual numbers. Results against the dual-number fit, and speed;
# 10,000 voxels, SNR 100, one thread, 4 and 16 points, kmf fixed and free.
#   s = M0f·(1 + Ei·(D·B − 1)),  Ei = e^(−R1f·TI), D = 1 − e^(−R1f·TD), B = a + b·Ek, Ek = e^(−k·TI),
#   k = kmf·q, q = 1 + PSR, a = (Sf + PSR·Sm)/q, b = PSR·(Sf − Sm)/q
using EQUIP, Random, Statistics, Printf, StaticArrays
import EQUIP: normal_equations, VoxelProblem, _params, _wt
const FREE4 = (:PSR, :R1f, :Sf, :M0f); const FREE5 = (:PSR, :R1f, :Sf, :M0f, :kmf)
@inline function _sirrow(m, p, j, ::Val{P}) where {P}
    (; PSR, R1f, Sf, M0f, kmf, Sm) = p
    TI, TD = m.TI[j], m.TD[j]
    q = 1 + PSR; k = kmf * q; a = (Sf + PSR * Sm) / q; b = PSR * (Sf - Sm) / q
    Ei, Ed, Ek = exp(-R1f * TI), exp(-R1f * TD), exp(-k * TI)
    D = 1 - Ed; B = a + b * Ek; inner = D * B - 1
    s = M0f * (1 + Ei * inner)
    c = M0f * Ei * D                                   # ∂s/∂B
    dPSR = c * ((Sm - Sf) / q^2 * (1 - Ek) - b * TI * kmf * Ek)
    dR1f = M0f * Ei * (TD * Ed * B - TI * inner)
    dSf = c * (1 + PSR * Ek) / q
    dM0f = 1 + Ei * inner
    J = P == 4 ? SVector(dPSR, dR1f, dSf, dM0f) : SVector(dPSR, dR1f, dSf, dM0f, -c * b * TI * q * Ek)
    return m.magnitude && s < 0 ? (-s, -J) : (s, J)
end
for F in (FREE4, FREE5)
    @eval @noinline function normal_equations(prob::VoxelProblem{$F,<:QMTSIREqualR1}, θ::SVector{P,T}, buf) where {P,T}
        p = _params(Val($F), θ, prob.fixed)
        A = zero(SMatrix{P,P,T}); g = zero(SVector{P,T}); sse = zero(T); y = prob.y
        @inbounds for i in eachindex(y)
            s, J = _sirrow(prob.model, p, i, Val(P))
            r = s - T(y[i]); w = _wt(prob.w, i, T); wJ = w * J
            A += wJ * J'; g += r * wJ; sse += w * r * r
        end
        return sse, A, g
    end
end
