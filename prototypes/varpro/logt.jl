# =================================================================================================
# Log-transform prototype, NOT part of EQUIP: the solver works in φ = log θ for chosen
# parameters; the model sees θ = exp(φ). The dual numbers apply the chain rule, so no derivative
# code changes. Works on a plain model (LogT{<:MRIModel}) or on a VarPro wrapper (LogT{<:VP}).
# =================================================================================================
include(joinpath(@__DIR__, "vp.jl"))
using LinearAlgebra: diag, dot, Diagonal

struct LogT{M,K}
    inner::M        # a model, or a VP wrapper
    mask::K         # NTuple{P,Bool}: free parameter j is fitted as log θ_j
end

@inline _unlog(θd::SVector{P}, mask) where {P} = SVector{P}(ntuple(j -> mask[j] ? exp(θd[j]) : θd[j], Val(P)))

@inline function _accumulate(buf, y, ::Val{P}, ::Type{T}) where {P,T}
    A = zero(SMatrix{P,P,T}); g = zero(SVector{P,T}); sse = zero(T)
    @inbounds for i in eachindex(buf, y)
        d = buf[i]
        J = SVector{P,T}(partials(d).values)
        r = T(value(d)) - T(y[i])
        A += J * J'; g += r * J; sse += r * r
    end
    return sse, A, g
end

# Full model, some parameters in log coordinates.
@inline function E.normal_equations(prob::E.VoxelProblem{F,<:LogT{<:E.MRIModel}}, φ::SVector{P,T}, buf) where {F,P,T}
    lt = prob.model
    θd = _unlog(E._seed(φ), lt.mask)
    E.signal!(buf, lt.inner, E._params(Val(F), θd, prob.fixed))
    return _accumulate(buf, prob.y, Val(P), T)
end

# VarPro, its nonlinear parameters in log coordinates.
@inline function E.normal_equations(prob::E.VoxelProblem{F,<:LogT{<:VP}}, φ::SVector{P,T}, buf) where {F,P,T}
    lt = prob.model
    θd = _unlog(E._seed(φ), lt.mask)
    c = _vp_core(E.VoxelProblem{F}(lt.inner, prob.y, prob.fixed), θd)
    cols = lt.inner.cols
    A = zero(SMatrix{P,P,T}); g = zero(SVector{P,T}); sse = zero(T)
    y = prob.y
    @inbounds for i in eachindex(y)
        d = _combine(c, cols, i)
        J = SVector{P,T}(partials(d).values)
        r = T(value(d)) - T(y[i])
        A += J * J'; g += r * J; sse += r * r
    end
    return sse, A, g
end

# EQUIP's levmar with one change: the step test is per parameter, |Δφ_j| ≤ rtol for every j.
# With every parameter in log coordinates, that is "every parameter changes by less than a
# relative rtol". (Copied from src/fit/levmar.jl; everything else identical.)
function levmar_pp(prob, θ0::SVector{P,T}, lo::SVector{P,T}, hi::SVector{P,T}, opts, buf, rtol) where {P,T}
    θ = clamp.(θ0, lo, hi)
    sse, A, g = E.normal_equations(prob, θ, buf)
    if !(isfinite(sse) && E._allfinite(g) && E._allfinite(diag(A)))
        return E.SolverResult(θ, sse, Int32(0), EQUIP.Failed)
    end
    d = max.(diag(A), floatmin(T) / eps(T))
    μ = opts.damping; ν = T(2); iter = Int32(0); status = EQUIP.MaxIterations
    E._gradient_small(A, g, E._movable(θ, g, lo, hi), sse, opts.gtol) && return E.SolverResult(θ, sse, iter, EQUIP.ConvergedG)
    while iter < opts.maxiter
        iter += Int32(1)
        fm = map(b -> ifelse(b, one(T), zero(T)), E._movable(θ, g, lo, hi))
        M = (fm * fm') .* (A + Diagonal(μ * d)) + Diagonal(one(T) .- fm)
        L, ok = E._cholesky(M)
        if !ok
            μ *= ν; ν *= 2; continue
        end
        h = E._chol_solve(L, -(fm .* g))
        θn = clamp.(θ + h, lo, hi)
        s = θn - θ
        if maximum(abs, s) <= rtol                       # <- the per-parameter step test
            status = EQUIP.ConvergedX
            break
        end
        ssen, An, gn = E.normal_equations(prob, θn, buf)
        pred = -(2 * dot(g, s) + dot(s, A * s))
        actual = sse - ssen
        if isfinite(ssen) && E._allfinite(gn) && E._allfinite(diag(An)) && pred > 0 && actual > 0 &&
           !E._lost_influence(A, An, fm)
            ρ = actual / pred
            fsmall = actual <= opts.ftol * sse && pred <= opts.ftol * sse
            θ, sse, A, g = θn, ssen, An, gn
            d = max.(d, diag(A))
            μ = max(μ * max(T(1 / 3), 1 - (2ρ - 1)^3), eps(T))
            ν = T(2)
            if fsmall
                status = EQUIP.ConvergedF; break
            elseif E._gradient_small(A, g, E._movable(θ, g, lo, hi), sse, opts.gtol)
                status = EQUIP.ConvergedG; break
            end
        else
            if pred > 0 && pred <= opts.ftol * sse
                status = EQUIP.ConvergedF; break
            end
            μ *= ν; ν *= 2
        end
    end
    return E.SolverResult(θ, sse, iter, status)
end
