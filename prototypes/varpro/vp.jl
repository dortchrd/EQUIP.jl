# =================================================================================================
# Variable projection (VarPro) prototype, NOT part of EQUIP: the parameters the signal is linear
# in are solved exactly inside every LM evaluation, so LM only sees the nonlinear ones.
#
# How it plugs in: `levmar` gets everything from `normal_equations(prob, θ, buf)`. A VoxelProblem
# whose model is wrapped in `VP` dispatches to the method below instead of EQUIP's.
#
# The linear part as columns: with the linear parameters set to "unit" values (e.g. S0 = 1, or
# S0 = 1, fshort = 1 for the biexponential's short component), the model's own signal! gives one
# column s_k of the linear system  S = Σ c_k s_k(θ). The amplitudes c = (SᵀS)⁻¹Sᵀy are computed in
# dual arithmetic, so the residual Σ c_k s_k − y carries the exact derivative of the reduced
# problem (no Kaufman approximation needed).
# =================================================================================================
using EQUIP, StaticArrays
using ForwardDiff: Dual, value, partials
const E = EQUIP

struct VP{M,U,B,N}
    model::M
    units::U        # tuple of NamedTuples: the linear parameters' values giving column k
    cols::B         # tuple of dual buffers, one per column
    nonneg::N       # tuple of Bools: is amplitude k constrained to ≥ 0?
end

# Fill the columns and solve for the amplitudes (duals). One or two columns.
@inline function _vp_core(prob::E.VoxelProblem{F,<:VP}, θd) where {F}
    vp = prob.model
    map((u, c) -> _column!(c, vp.model, Val(F), θd, prob.fixed, u), vp.units, vp.cols)
    return _amplitudes(vp.cols, prob.y, vp.nonneg)
end
# A column either from the model's own signal! with the linear parameters at unit values (generic,
# but evaluates the whole model per column), or from a function the model supplies (only the
# column's own terms).
@inline _column!(c, model, ::Val{F}, θd, fixed, u::NamedTuple) where {F} =
    E.signal!(c, model, E._params(Val(F), θd, merge(fixed, u)))
@inline _column!(c, model, ::Val{F}, θd, fixed, u::Function) where {F} =
    u(c, model, NamedTuple{F}(Tuple(θd)))

function _amplitudes(cols::Tuple{Any}, y, nonneg)
    s = cols[1]
    sy = zero(eltype(s)); ss = zero(eltype(s))
    @inbounds for i in eachindex(s, y)
        sy += s[i] * y[i]; ss += s[i] * s[i]
    end
    a = sy / ss
    nonneg[1] && value(a) < 0 && (a = zero(a))
    return (a,)
end

function _amplitudes(cols::Tuple{Any,Any}, y, nonneg)
    s1, s2 = cols
    D = eltype(s1)
    g11 = g12 = g22 = b1 = b2 = zero(D)
    @inbounds for i in eachindex(s1, s2, y)
        g11 += s1[i] * s1[i]; g12 += s1[i] * s2[i]; g22 += s2[i] * s2[i]
        b1 += s1[i] * y[i]; b2 += s2[i] * y[i]
    end
    det = g11 * g22 - g12 * g12
    c1 = (g22 * b1 - g12 * b2) / det
    c2 = (g11 * b2 - g12 * b1) / det
    # Active set for amplitudes that must be ≥ 0: drop the negative one, solve for the other.
    if nonneg[1] && value(c1) < 0
        c1 = zero(D); c2 = b2 / g22
        nonneg[2] && value(c2) < 0 && (c2 = zero(D))
    elseif nonneg[2] && value(c2) < 0
        c2 = zero(D); c1 = b1 / g11
        nonneg[1] && value(c1) < 0 && (c1 = zero(D))
    end
    return (c1, c2)
end

@inline function E.normal_equations(prob::E.VoxelProblem{F,<:VP}, θ::SVector{P,T}, buf) where {F,P,T}
    c = _vp_core(prob, E._seed(θ))
    cols = prob.model.cols
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
@inline _combine(c::Tuple{Any}, cols, i) = @inbounds c[1] * cols[1][i]
@inline _combine(c::Tuple{Any,Any}, cols, i) = @inbounds c[1] * cols[1][i] + c[2] * cols[2][i]

# The amplitudes at θ, as plain numbers.
linear_coeffs(prob, θ) = map(value, _vp_core(prob, E._seed(θ)))

# Build the reduced problem's pieces from a full EQUIP cache: the nonlinear names, the columns'
# unit values, which amplitudes are ≥ 0, and the bounds of the nonlinear parameters.
function vp_setup(cache, linear::Tuple, units::Tuple, nonneg::Tuple)
    spec = cache.spec
    F = E.freenames(spec)
    Fn = Tuple(n for n in F if !(n in linear))
    Pn = length(Fn); T = eltype(spec.lower); M = length(cache.y)
    keep = SVector{Pn,Int}(ntuple(k -> findfirst(==(Fn[k]), F), Pn))   # static: indexing allocates nothing
    cols = ntuple(_ -> Vector{Dual{E.FitTag,T,Pn}}(undef, M), length(units))
    vp = VP(cache.model, units, cols, nonneg)
    lo = spec.lower[keep]; hi = spec.upper[keep]
    return (; Fn, keep, vp, lo, hi, fixed = spec.fixed)
end
