# =================================================================================================
# Exponential decay (T2, T2*)
# =================================================================================================

"""
    ExpDecay(; TE)

Mono-exponential decay `S = S0·exp(-TE/T2)` (T2, T2* or any decay time constant).

- acquisition: `TE` (s), one per observation
- parameters: `S0`, `T2` (s)

```julia
m = ExpDecay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct ExpDecay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::ExpDecay) = (
    S0 = Free(:scale; lower = 0),
    T2 = Free(0.05; lower = _TMIN),
)

function signal!(out, m::ExpDecay, p)
    (; TE) = m                          # unpack by name: TE = m.TE
    (; S0, T2) = p                      # S0 = p.S0, T2 = p.T2
    @inbounds for j in eachindex(out)
        out[j] = S0 * exp(-TE[j] / T2)
    end
    return out
end
# `@inbounds` skips index bounds checks inside the loop (safe: `init` checked the lengths).
