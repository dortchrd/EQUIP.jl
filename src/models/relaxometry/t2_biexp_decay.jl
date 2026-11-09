# =================================================================================================
# Bi-exponential T2 decay: two components with different T2, e.g. a short-T2 pool (myelin water)
# and a long-T2 pool (intra- and extracellular water)
#
#   T2BiexpDecay   S = S0·[fshort·exp(-TE/T2short) + (1 - fshort)·exp(-TE/T2long)]
#   R2BiexpDecay   S = S0·[fshort·exp(-R2short·TE) + (1 - fshort)·exp(-R2long·TE)]
#
# The equation is written once, in R2BiexpDecay; T2BiexpDecay converts and calls it (see
# builtin.jl). No offset version: the fit is hard enough already.
#
# A hard fit: four strongly correlated parameters, and a mono-exponential curve has many
# descriptions (with fshort = 0, T2short can be anything; with T2short = T2long, fshort can).
# The two components can also swap: (fshort, T2short, T2long) and (1 - fshort, T2long, T2short)
# give the same signal. The fit itself does not keep T2short < T2long (from the default starts
# the T2 form swapped them in 196 of 200 voxels at SNR 100, the R2 form in none), so after the
# fit the labels are put in order: the shorter T2 is reported as T2short (`relabel`, below).
#
# Starts from the acquisition: T2short = mean(TE)/4, T2long = 2·mean(TE), fshort = 0.2 (the R2
# version starts from the same values as rates). S0 starts at the least-squares scale.
# =================================================================================================

"""
    T2BiexpDecay(; TE)

Bi-exponential decay with time constants, e.g. myelin water (short T2) and intra- and
extracellular water (long T2):

    S = S0·[fshort·exp(-TE/T2short) + (1 - fshort)·exp(-TE/T2long)]

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `fshort` (fraction of S0 in the short-T2 component, 0 to 1),
  `T2short`, `T2long` (s, each at most 2.5 s as [`T2Decay`](@ref)'s)

Starts at `T2short = mean(TE)/4`, `T2long = 2·mean(TE)`, `fshort = 0.2`. The two components
can swap (the same signal with `1 - fshort`), so after the fit the shorter T2 is reported as
`T2short`, when `fshort`, `T2short` and `T2long` are all free. Rate version:
[`R2BiexpDecay`](@ref).

```julia
m = T2BiexpDecay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct T2BiexpDecay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:T2BiexpDecay}) = (
    S0 = Free(start = :scale, lower = 0),
    fshort = Free(start = 0.2, lower = 0, upper = 1),  # fraction of S0 in the short-T2 component
    T2short = Free(start = m -> mean(m.TE) / 4, lower = _TMIN, upper = _T2MAX),   # from the echo times;
    T2long = Free(start = m -> 2 * mean(m.TE), lower = _TMIN, upper = _T2MAX),    # at most 2.5 s (builtin.jl)
)

# The T2 form: R2 = 1/T2 for each component, then R2BiexpDecay's signal (below).
signal!(out, m::T2BiexpDecay, p) =
    signal!(out, R2BiexpDecay(m.TE), (; p.S0, p.fshort, R2short = 1 / p.T2short, R2long = 1 / p.T2long))

# The components can swap; report the shorter T2 as T2short (see `relabel`). Only when all three
# are free: a fixed one pins the labels, and swapping would change the signal.
function relabel(::T2BiexpDecay, θ)
    haskey(θ, :fshort) && haskey(θ, :T2short) && haskey(θ, :T2long) || return θ
    θ.T2short <= θ.T2long && return θ
    return merge(θ, (fshort = 1 - θ.fshort, T2short = θ.T2long, T2long = θ.T2short))
end

"""
    R2BiexpDecay(; TE)

[`T2BiexpDecay`](@ref) with rates, R2short = 1/T2short and R2long = 1/T2long:

    S = S0·[fshort·exp(-R2short·TE) + (1 - fshort)·exp(-R2long·TE)]

`R2short` is the rate of the short-T2 component (the larger rate); after the fit the larger
rate is reported as `R2short`, as for the T2 version.

- acquisition: `TE` (s), one per observation
- free by default: `S0`, `fshort` (0 to 1), `R2short`, `R2long` (1/s, each at least 0.4 s⁻¹ as
  [`R2Decay`](@ref)'s)

Starts at `R2short = 4/mean(TE)`, `R2long = 1/(2·mean(TE))`, `fshort = 0.2`.

```julia
m = R2BiexpDecay(TE = 0.01:0.01:0.32)
```
"""
@kwdef struct R2BiexpDecay{A<:AbstractVector} <: MRIModel
    TE::A                               # echo times (s)
end

parameters(::Type{<:R2BiexpDecay}) = (
    S0 = Free(start = :scale, lower = 0),
    fshort = Free(start = 0.2, lower = 0, upper = 1),
    R2short = Free(start = m -> 4 / mean(m.TE), lower = _R2MIN),  # rate of the short-T2 component;
    R2long = Free(start = m -> 1 / (2 * mean(m.TE)), lower = _R2MIN),   # at least 0.4/s (builtin.jl)
)

function signal!(out, m::R2BiexpDecay, p)
    (; TE) = m
    (; S0, fshort, R2short, R2long) = p
    @inbounds for j in eachindex(out, TE)
        out[j] = S0 * (fshort * exp(-R2short * TE[j]) + (1 - fshort) * exp(-R2long * TE[j]))
    end
    return out
end

# The larger rate (the shorter T2) is reported as R2short.
function relabel(::R2BiexpDecay, θ)
    haskey(θ, :fshort) && haskey(θ, :R2short) && haskey(θ, :R2long) || return θ
    θ.R2short >= θ.R2long && return θ
    return merge(θ, (fshort = 1 - θ.fshort, R2short = θ.R2long, R2long = θ.R2short))
end
