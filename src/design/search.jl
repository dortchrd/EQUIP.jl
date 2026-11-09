# =================================================================================================
# The protocol search: differential evolution, then Nelder–Mead, from several starts on threads
#
#   r = design(d; starts, rng)          (named `optimize_protocol` until 2026-11-09)
#       for each start (a task, with its own copy of the design and its own random stream):
#         1. differential evolution over the bounds (global: finds the right basin)
#         2. Nelder–Mead from its best point (local: the last digits)
#       the best start's protocol, sorted by TI; the same rounded to the ms, evaluated again
#
# Why these two (docs/design.md, "The plan: ProtocolDesign"): the variables are a handful of
# continuous acquisition values within bounds, the case differential evolution was made for
# (Storn & Price, J Glob Optim 1997;11:341): a population, each member challenged by a trial
# built from the scaled difference of two others, so the steps shrink as the population
# converges. The objective has corners (the maximum over scenarios: at a minimax optimum the
# worst ones tie) and no exact gradient with respect to the acquisition yet, so the polish
# compares values only (Nelder–Mead). Both compare points by `_rank`: feasibility first, then
# the objective, so constraints need no penalty weight.
# =================================================================================================

"""
    DesignResult

What [`design`](@ref) returns:

- `protocol`: the best protocol found, a model (points sorted by the first varied field), and
  `value`, the objective there
- `rounded` and `rounded_value`: the same protocol rounded (to the ms by default) and its
  objective: of the ways to round each value down or up, the best, since rounding each to the
  nearest step can raise one of the scenarios that tie at the optimum
- `x`: the best values as the search saw them (the varied fields one after the other)
- `starts`: every start's best value, in order, so their spread shows how reliable the search
  is; `evaluations` (all starts together) and `time` (s)
"""
struct DesignResult{M,D}
    protocol::M
    value::Float64
    rounded::M
    rounded_value::Float64
    x::Vector{Float64}
    starts::Vector{Float64}
    evaluations::Int
    time::Float64
    design::D
end

"""
    design(d::ProtocolDesign; starts = 2 * Threads.nthreads(), rng, popsize,
           generations = 1000, polish = true, round = 0.001) -> DesignResult
    design(m, scenarios; snr, acquisition, scantime, ..., starts, rng, ...) -> DesignResult

Search the protocol that minimizes the objective `d` ([`ProtocolDesign`](@ref)) within its
bounds and constraints: differential evolution, then Nelder–Mead from its best point, from
`starts` independent starts on Julia's threads (start Julia with `-t auto`). The protocol given
to `d` is one member of the first start's population. In one call, `design(m, scenarios;
...)` makes the objective first: the keywords of [`ProtocolDesign`](@ref) and the search's
together, as `fit(m, data)` is `init` then `fit!`. (Named `optimize_protocol` until
2026-11-09: the main functions are short, `fit`, `crlb`, `montecarlo`, `design`, as the command
line's `equip design`.)

- `rng`: the random number generator; one number is drawn from it per start (that start's
  seed), so `MersenneTwister(1)` (from Random) repeats a search exactly, threads or not.
- `popsize`: the population of each start (default 10 per variable); `generations`: at most
  this many, fewer when the population has converged (its values within 1e-6 of each other).
- `polish`: Nelder–Mead after differential evolution.
- `round`: the step the result is rounded to (s), or `nothing`.

```julia
r = design(d; starts = 24, rng = MersenneTwister(1))
r.protocol, r.value            # the best protocol and its Eq. 5
r.rounded, r.rounded_value     # rounded to the ms
```
"""
function design(d::ProtocolDesign; starts::Integer = 2 * Threads.nthreads(),
                rng::AbstractRNG = Random.default_rng(),
                popsize::Integer = 10 * length(d.lower), generations::Integer = 1000,
                polish::Bool = true, round = 0.001)
    starts >= 1 || throw(ArgumentError("`starts` must be at least 1"))
    popsize >= 4 || throw(ArgumentError("`popsize` must be at least 4 (a trial uses three others)"))
    t0 = time()
    seeds = [rand(rng, UInt64) for _ in 1:starts]
    x0 = _x(d, d.model)
    # One task per start, each with its own copy of the design (buffers) and random stream.
    tasks = [Threads.@spawn _onestart(copy(d), Xoshiro(seeds[i]), i == 1 ? x0 : nothing,
                                      popsize, generations, polish) for i in 1:starts]
    results = fetch.(tasks)
    best = argmin(r -> r.score, results)
    x = best.x
    xr, sr, nr = round === nothing ? (copy(x), best.score, 0) : _rounded(d, x, round)
    p, pr = _sortedprotocol(d, x), _sortedprotocol(d, xr)
    return DesignResult(p, _objective(best.score), pr, _objective(sr), x,
                        [_objective(r.score) for r in results], sum(r.evaluations for r in results) + nr,
                        time() - t0, d)
end

# In one call: the objective (ProtocolDesign's keywords), then the search (its own).
function design(m::MRIModel, scenarios; starts::Integer = 2 * Threads.nthreads(),
                rng::AbstractRNG = Random.default_rng(), popsize = nothing,
                generations::Integer = 1000, polish::Bool = true, round = 0.001, kwargs...)
    d = ProtocolDesign(m, scenarios; kwargs...)
    return design(d; starts, rng, popsize = something(popsize, 10 * length(d.lower)),
                  generations, polish, round)
end

# One start: differential evolution, then (optionally) Nelder–Mead from its best point.
function _onestart(d, rng, x0, popsize, generations, polish)
    x, s, n = _differential_evolution(d, rng, x0, popsize, generations)
    if polish
        x, s, m = _nelder_mead(d, x, s)
        n += m
    end
    return (x = x, score = s, evaluations = n)
end

# -------------------------------------------------------------------------------------------------
# Differential evolution, DE/rand/1/bin with self-adapting F and CR (jDE: Brest et al., IEEE
# Trans Evol Comput 2006;10:646)
# -------------------------------------------------------------------------------------------------

# For each member xᵢ of the population, a trial: xₐ + F(x_b − x_c) from three others (a, b, c),
# taken for each variable with probability CR (and for one at random, so the trial differs),
# the rest from xᵢ. The trial replaces xᵢ when it scores no worse. Each member has its own F and
# CR, renewed at random with probability 0.1 and kept when they gave a better trial: the search
# tunes itself. A trial outside the bounds is put back on the bound it crossed (the optima of
# SIR schemes sit on bounds: TI = 10 ms, TD = 10 ms).
function _differential_evolution(d, rng, x0, NP, generations)
    lo, hi = d.lower, d.upper
    D = length(lo)
    pop = [lo .+ rand(rng, D) .* (hi .- lo) for _ in 1:NP]
    x0 === nothing || (pop[1] = copy(x0))
    score = [_rank(d, x) for x in pop]
    F, CR = fill(0.5, NP), fill(0.9, NP)
    trial = zeros(D)
    n = NP
    for _ in 1:generations
        for i in 1:NP
            Fi = rand(rng) < 0.1 ? 0.1 + 0.9 * rand(rng) : F[i]
            CRi = rand(rng) < 0.1 ? rand(rng) : CR[i]
            a, b, c = _three_others(rng, NP, i)
            jr = rand(rng, 1:D)
            @inbounds for j in 1:D
                t = rand(rng) < CRi || j == jr ? pop[a][j] + Fi * (pop[b][j] - pop[c][j]) : pop[i][j]
                trial[j] = clamp(t, lo[j], hi[j])
            end
            st = _rank(d, trial)
            n += 1
            if !isless(score[i], st)             # no worse: feasibility first, then the value
                pop[i] .= trial
                score[i], F[i], CR[i] = st, Fi, CRi
            end
        end
        _converged(score) && break
    end
    i = argmin(score)
    return pop[i], score[i], n
end

# Three distinct members other than i.
function _three_others(rng, NP, i)
    a = rand(rng, 1:NP); while a == i; a = rand(rng, 1:NP); end
    b = rand(rng, 1:NP); while b == i || b == a; b = rand(rng, 1:NP); end
    c = rand(rng, 1:NP); while c == i || c == a || c == b; c = rand(rng, 1:NP); end
    return a, b, c
end

# Converged: every member feasible and their values within 1e-6 of the best (relative).
function _converged(score)
    all(s -> s[1] == 0 && isfinite(s[2]), score) || return false
    lo, hi = extrema(s -> s[2], score)
    return hi - lo <= 1e-6 * abs(lo)
end

# -------------------------------------------------------------------------------------------------
# Nelder–Mead (Lagarias et al., SIAM J Optim 1998;9:112): the polish
# -------------------------------------------------------------------------------------------------

# A simplex of D + 1 points around x (steps of 1% of each variable's range), moved by
# reflecting, expanding, contracting or shrinking toward its best point; every point clamped
# to the bounds. Stops when the simplex is within 1e-10 of the variables' ranges, or after
# 200·D iterations.
function _nelder_mead(d, x, sx)
    lo, hi = d.lower, d.upper
    D = length(x)
    pts = [copy(x) for _ in 1:D+1]
    for j in 1:D
        step = 0.01 * (hi[j] - lo[j])
        pts[j+1][j] = x[j] + step <= hi[j] ? x[j] + step : x[j] - step
    end
    sc = [i == 1 ? sx : _rank(d, pts[i]) for i in 1:D+1]
    n = D
    at(c, y, t) = clamp.(c .+ t .* (y .- c), lo, hi)     # the point c + t(y − c), in the bounds
    for _ in 1:200D
        o = sortperm(sc; lt = isless)
        pts, sc = pts[o], sc[o]
        maximum(j -> maximum(abs(p[j] - pts[1][j]) for p in pts) / (hi[j] - lo[j]), 1:D) <= 1e-10 && break
        c = sum(pts[1:D]) ./ D                          # the centroid of all but the worst
        xr = at(c, pts[end], -1.0); sr = _rank(d, xr); n += 1         # reflection
        if isless(sr, sc[1])
            xe = at(c, pts[end], -2.0); se = _rank(d, xe); n += 1     # expansion
            if isless(se, sr)
                pts[end] = xe; sc[end] = se
            else
                pts[end] = xr; sc[end] = sr
            end
        elseif isless(sr, sc[D])
            pts[end] = xr; sc[end] = sr
        else
            outside = isless(sr, sc[end])
            xc = outside ? at(c, xr, 0.5) : at(c, pts[end], 0.5)       # contraction
            scc = _rank(d, xc); n += 1
            if isless(scc, outside ? sr : sc[end])
                pts[end] = xc; sc[end] = scc
            else                                         # shrink toward the best point
                for i in 2:D+1
                    pts[i] = at(pts[1], pts[i], 0.5); sc[i] = _rank(d, pts[i]); n += 1
                end
            end
        end
    end
    i = argmin(sc)
    return pts[i], sc[i], n
end

# The values rounded to `step` (the ms): of the ways to round each one down or up, the one that
# scores best (all 2^D of them up to D = 12 variables), then single steps up or down while one
# improves. Why not the nearest step: at a minimax optimum the worst scenarios tie, and rounding
# each value to the nearest ms can raise one of them (4-point SIR: WM I and the lesion tie, and
# the nearest ms raised the maximum by 0.3%, past Table 2's own rounded scheme).
function _rounded(d, x, step)
    lo, hi = d.lower, d.upper
    k = inv(step)                                   # values as k-ths: 1000 for the ms
    grid(v, j) = clamp(v / k, lo[j], hi[j])
    down = [grid(floor(x[j] * k), j) for j in eachindex(x)]
    up = [grid(ceil(x[j] * k), j) for j in eachindex(x)]
    best = [grid(Base.round(x[j] * k), j) for j in eachindex(x)]
    sbest = _rank(d, best)
    n, D = 1, length(x)
    if D <= 12
        y = similar(x)
        for bits in 0:(2^D - 1)
            for j in 1:D
                y[j] = isodd(bits >> (j - 1)) ? up[j] : down[j]
            end
            s = _rank(d, y); n += 1
            if isless(s, sbest)
                best .= y; sbest = s
            end
        end
    end
    improved = true
    while improved
        improved = false
        for j in 1:D, δ in (-1, 1)
            y = copy(best)
            y[j] = grid(Base.round(y[j] * k) + δ, j)
            s = _rank(d, y); n += 1
            if isless(s, sbest)
                best = y; sbest = s; improved = true
            end
        end
    end
    return best, sbest, n
end

# -------------------------------------------------------------------------------------------------
# The result: the protocol as a plain model, points sorted
# -------------------------------------------------------------------------------------------------

# The protocol at x, with plain vectors and its points in order of the first varied field (the
# CRLB does not depend on their order).
function _sortedprotocol(d::ProtocolDesign{N,V,W}, x) where {N,V,W}
    p = _protocol(d, x)
    o = sortperm(collect(getfield(p, V[1])))
    sorted(v) = v isa AbstractVector{<:Real} && length(v) == N && !(v isa AbstractVector{Bool}) ?
                Vector{Float64}(v[o]) : v
    return W(map(sorted, _fields(p))...)
end

function Base.show(io::IO, ::MIME"text/plain", r::DesignResult)
    d = r.design
    f(x) = string(round(x; sigdigits = 6))
    good = count(v -> v <= r.value * 1.01, r.starts)
    print(io, "DesignResult for ", nameof(typeof(r.protocol)), ", ", nobs(r.protocol), " points: ",
          "value ", f(r.value), " (rounded: ", f(r.rounded_value), "); ", length(r.starts),
          " starts, ", good, " within 1% of the best; ", r.evaluations, " evaluations, ",
          string(round(r.time; sigdigits = 3)), " s")
    names = [n for n in fieldnames(typeof(r.rounded)) if getfield(r.rounded, n) isa AbstractVector{<:Real}]
    print(io, "\n  rounded:  ", join(("$n = " * _acqvalue(getfield(r.rounded, n)) for n in names), ", "))
    print(io, "\n  found:    ", join(("$n = " * _acqvalue(getfield(r.protocol, n)) for n in names), ", "))
end
Base.show(io::IO, r::DesignResult) =
    print(io, "DesignResult(", nameof(typeof(r.protocol)), ", value ", round(r.value; sigdigits = 6), ")")
