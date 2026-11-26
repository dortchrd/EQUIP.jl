# =================================================================================================
# Tables of a function of one variable, read by quintic Hermite interpolation
#
#   f(x) ≈ H0·fᵢ + h·H1·f′ᵢ + h²·H2·f″ᵢ + H3·fᵢ₊₁ + h·H4·f′ᵢ₊₁ + h²·H5·f″ᵢ₊₁
#
# on the cell [xᵢ, xᵢ₊₁] of a grid xᵢ = x₀ + i·h, t = (x − xᵢ)/h: the basis matches the value, the
# slope and the curvature at both ends, so the interpolant is twice continuously differentiable
# across the points (a fit's Jacobian and crlb's Hessians are smooth there), with an error of
# order h⁶·f⁽⁶⁾. For functions EQUIP needs often and cannot write in closed form, made once (when
# EQUIP is precompiled) from their exact values and two derivatives, and only read after: the
# Watson stick's kₗ(β) (models/diffusion/dispersion.jl) and the super-Lorentzian lineshape's
# G(x) (models/qmt/lineshapes.jl). Each user checks its argument is inside the table and computes
# the function exactly outside it.
#
# Dual numbers (a fit's derivatives; crlb's nested ones) take the table's own chain rule: the
# value and the derivative at the plain value, then the partials, so the interpolation's
# arithmetic stays in plain numbers (the dual arithmetic through the basis had cost a
# NODDIConvolved fit 25%). The tables are read-only: held in a struct with indexing and no
# assignment, so nothing in EQUIP can write them, and every thread reads them at once.
# =================================================================================================

# a vector that can be read and not written
struct _ReadOnly{T}
    v::Vector{T}
end
Base.@propagate_inbounds Base.getindex(t::_ReadOnly, i::Int) = t.v[i]
Base.length(t::_ReadOnly) = length(t.v)

# The quintic Hermite basis on [0, 1]: each Hᵣ(t) = Σₖ cᵣₖ·tᵏ, k = 0 … 5
const _QUINTIC = ((1.0, 0.0, 0.0, -10.0, 15.0, -6.0),        # H0
                  (0.0, 1.0, 0.0, -6.0, 8.0, -3.0),          # H1
                  (0.0, 0.0, 0.5, -1.5, 1.5, -0.5),          # H2
                  (0.0, 0.0, 0.0, 10.0, -15.0, 6.0),         # H3
                  (0.0, 0.0, 0.0, -4.0, 7.0, -3.0),          # H4
                  (0.0, 0.0, 0.0, 0.5, -1.0, 0.5))           # H5
# dᵐHᵣ/dtᵐ at t: Σₖ cᵣₖ·k!/(k − m)!·t^(k−m), m known when compiling (Val), 0 past the fifth
@inline function _basisderiv(c, t, ::Val{m}) where {m}
    s = zero(t)
    @inbounds for k in 5:-1:m                                 # Horner in t
        s = s * t + c[k+1] * prod(ntuple(q -> k - q + 1, Val(m)); init = 1.0)
    end
    return s
end

# A table: f, f′ and f″ at xᵢ = x₀ + i·h, i = 0 … n − 1. Its values can be numbers or static
# vectors (the stick's 21 kₗ at once).
struct _QuinticTable{T}
    x0::Float64                 # the first point
    h::Float64                  # the spacing
    v::_ReadOnly{T}             # f at the points
    d1::_ReadOnly{T}            # f′
    d2::_ReadOnly{T}            # f″
end
# made from a function returning (f, f′, f″) at x, at n points
function _QuinticTable(f, x0, h, n)
    tab = [f(x0 + i * h) for i in 0:(n - 1)]
    return _QuinticTable(Float64(x0), Float64(h), _ReadOnly(getindex.(tab, 1)), _ReadOnly(getindex.(tab, 2)),
                         _ReadOnly(getindex.(tab, 3)))
end
_tabletop(tab::_QuinticTable) = tab.x0 + (length(tab.v) - 1) * tab.h      # the last point

# The m-th derivative in x of the interpolant (m = 0: f itself) at a plain x in [x₀, the last
# point): the basis' m-th derivatives in t, over hᵐ. The caller checks the range.
@inline function _interp(tab::_QuinticTable, x::Real, ::Val{m}) where {m}
    y = (x - tab.x0) / tab.h
    i = unsafe_trunc(Int, y)
    t = y - i
    H = map(c -> _basisderiv(c, t, Val(m)), _QUINTIC)
    h, h2, s = tab.h, tab.h^2, inv(tab.h)^m
    v, d1, d2 = tab.v, tab.d1, tab.d2
    @inbounds return (s * H[1]) * v[i+1] + (s * h * H[2]) * d1[i+1] + (s * h2 * H[3]) * d2[i+1] +
                     (s * H[4]) * v[i+2] + (s * h * H[5]) * d1[i+2] + (s * h2 * H[6]) * d2[i+2]
end
# ... at a dual number: its value at the value, and its derivative times the partials, the
# chain rule written once. Recursive: the value of a nested dual is itself a dual. `map` takes
# the values one by one, whether a number or a static vector.
@inline function _interp(tab::_QuinticTable, x::Dual{T}, ::Val{m}) where {T,m}
    v, p = value(x), partials(x)
    a, d = _interp(tab, v, Val(m)), _interp(tab, v, Val(m + 1))
    return map((ai, di) -> Dual{T}(ai, di * p), a, d)
end
@inline _interp(tab::_QuinticTable, x) = _interp(tab, x, Val(0))
