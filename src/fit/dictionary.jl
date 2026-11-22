# =================================================================================================
# BestOf from a dictionary: the candidates' signals made once per fit
#
# PROTOTYPE (branch bestof-proto, 2026-11-20): measured in docs/design.md, item 4, "smart BestOf".
#
# `BestOf` scores every candidate starting point in every voxel (`_best_start`, starts.jl): the
# signal at the candidate, the scale solved, the SSE there. When nothing about the candidates
# depends on the voxel (no map among the fixed values or the starts, no `:data` or `:loglinear`
# start), their signals are the same in every voxel and can be made once per fit, as MR
# fingerprinting matches each voxel to a precomputed dictionary. A voxel then costs one dot
# product per candidate (and per linear parameter, below), not a signal: T1IR's 30 candidates at
# 9 TIs are 270 multiply-adds, not 270 exponentials.
#
# Linear parameters solved in the score. The scale a (M0, S0, ...) multiplies the signal. Some
# other parameters enter it linearly too, in one of two ways, found numerically once per fit
# (`_linearparameters`):
#   - added to it, an offset:            S = a·s₀ + C·g      (T2DecayOffset's C, with g = 1)
#   - inside the scale's shape:          S = a·(s₀ + f·g)    (T2BiexpDecay's fshort; Sf of a
#                                                             signed inversion recovery)
# with s₀ and g depending only on the other parameters (the candidate's). Either way the signal
# is linear in the coefficients c = (a, a·f, ..., C, ...): S = B·c, B = [s₀ g ...], N × M. The best
# c for the data y is the linear least squares c = G⁻¹z, with G = BᵀWB (once per candidate) and
# z = BᵀWy (per voxel), and the SSE there is
#     Σ wᵢ(yᵢ − Sᵢ)² = y·Wy − 2c·z + cᵀGc,
# so each candidate is ranked with its linear parameters at their best values, not at their
# starts: a candidate T2 is not judged with the offset at 0, nor a pair (T2short, T2long) with the
# fraction at its start, 0.2. With the scale alone (M = 1) this is `_score`'s a = (s·y)/(s·s).
#
# Bounds: c is solved without them; then each linear parameter is clamped to its bounds, the
# scale solved again given those values (its best value for them) and clamped to its own, and
# the SSE is the one at those values. A score, not a constrained fit: LM then starts from there.
# =================================================================================================

# Prototype switches, to measure the parts apart: the dictionary at all, and the linear
# parameters solved in its score. Read when a cache is made (`init`), not in the voxel loop.
const _PROTO_DICTIONARY = Ref(true)
const _PROTO_LINEAR = Ref(true)

# The dictionary of one fit, K candidates (the columns of `spec.candidates`), N observations, M
# coefficients (the scale and the linear parameters). `basis[i, k]` holds observation i of
# candidate k's M columns, times the weight wᵢ (so z = Σᵢ basis[i, k]·yᵢ, no weights per voxel);
# stored N × K so that a candidate's observations are contiguous in memory.
struct CandidateDictionary{P,T,M,L}
    points::Vector{SVector{P,T}}        # candidate k's starting point: its BestOf values, the other starts
    basis::Matrix{SVector{M,T}}         # N × K: wᵢ·(s₀, g, ...)ᵢ of candidate k
    gram::Vector{SMatrix{M,M,T,L}}      # G = BᵀWB of each candidate
    graminv::Vector{SMatrix{M,M,T,L}}   # G⁻¹ (NaN where G is singular: the candidate never wins)
    linear::SVector{M,Int}              # which free parameter each coefficient gives: the scale first
    inshape::SVector{M,Bool}            # true: inside the scale's shape (p = c/a); false: an offset (p = c)
end

# The dictionary of a fit, or `nothing` where the candidates depend on the voxel (then
# `_best_start` scores them per voxel, as before) or there is no BestOf or no scale.
function _dictionary(model, spec::FitSpec{F,P,T}, n) where {F,P,T}
    _PROTO_DICTIONARY[] || return nothing
    sc = spec.scale
    (isempty(spec.candindex) || sc == 0) && return nothing
    (any(spec.fromdata) || any(spec.fromlog)) && return nothing         # starts from each voxel's data
    (any(x -> x isa AbstractArray, values(spec.fixed)) ||
     any(x -> x isa AbstractArray, values(spec.start))) && return nothing   # a map: per voxel
    θ0 = Base.setindex(SVector{P,T}(Tuple(spec.start)), one(T), sc)     # the scale at 1
    C = spec.candidates
    points = [_withcandidate(θ0, spec.candindex, C, j) for j in axes(C, 2)]
    # The signal at θ, in plain numbers (setup only: allocating is fine here)
    signal(θ) = (s = Vector{T}(undef, n); signal!(s, model, _params(Val(F), θ, spec.fixed)); s)
    lin, inshape = _PROTO_LINEAR[] ? _linearparameters(signal, points, spec) : (Int[], Bool[])
    return _builddictionary(signal, points, spec, sc, Val(1 + length(lin)), lin, inshape, n)
end

# θ with combination j of the BestOf values
function _withcandidate(θ, ks, C, j)
    for q in eachindex(ks)
        θ = Base.setindex(θ, C[q, j], ks[q])
    end
    return θ
end

# The columns of every candidate: s₀ with the linear parameters at 0, and g = the signal with one
# of them at 1, minus s₀. M is a type parameter (`Val(M)`), so the columns are fixed-size vectors.
function _builddictionary(signal, points::Vector{SVector{P,T}}, spec, sc, ::Val{M}, lin, inshape,
                          n) where {P,T,M}
    K = length(points)
    basis = Matrix{SVector{M,T}}(undef, n, K)
    gram = Vector{SMatrix{M,M,T,M * M}}(undef, K)
    graminv = similar(gram)
    for k in 1:K
        θ = points[k]
        for j in lin
            θ = Base.setindex(θ, zero(T), j)
        end
        s0 = signal(θ)
        cols = vcat([s0], [signal(Base.setindex(θ, one(T), j)) .- s0 for j in lin])
        G = zero(SMatrix{M,M,T})
        for i in 1:n
            b = SVector{M,T}(ntuple(j -> cols[j][i], Val(M)))
            w = _wt(spec.weights, i, T)
            basis[i, k] = w * b
            G += w * b * b'
        end
        gram[k] = G
        graminv[k] = _safeinv(G)
    end
    return CandidateDictionary{P,T,M,M * M}(points, basis, gram, graminv,
                                            SVector{M,Int}(sc, lin...), SVector{M,Bool}(true, inshape...))
end

# G⁻¹, or NaN if G is singular (a column of zeros, two equal columns) or not finite
function _safeinv(G::SMatrix{M,M,T}) where {M,T}
    nan = SMatrix{M,M,T}(ntuple(_ -> T(NaN), Val(M * M)))
    all(isfinite, G) || return nan
    Gi = try
        inv(G)
    catch
        return nan
    end
    return all(isfinite, Gi) ? Gi : nan
end

# Which free parameters (other than the scale and the BestOf ones) the signal is linear in, and
# how (:shape or :offset, `_linearkind`), the same at the first, middle and last candidate; then
# all of them together checked (`_checklinear`), else none (the scale alone).
function _linearparameters(signal, points, spec::FitSpec{F,P,T}) where {F,P,T}
    sc = spec.scale
    tests = unique([points[1], points[cld(length(points), 2)], points[end]])
    lin, inshape = Int[], Bool[]
    for j in 1:P
        (j == sc || j in spec.candindex) && continue
        kinds = [_linearkind(signal, θ, sc, j, T) for θ in tests]
        kind = first(kinds)
        (kind !== nothing && all(==(kind), kinds)) || continue
        push!(lin, j)
        push!(inshape, kind === :shape)
    end
    isempty(lin) || _checklinear(signal, tests, sc, lin, inshape, T) || return Int[], Bool[]
    return lin, inshape
end

# How the signal depends on free parameter j at θ, from the signal at scale a and value p:
# - :shape   S(a, p) = a·(s₀ + p·g):  S(2, 1) = 2·S(1, 1)
# - :offset  S(a, p) = a·s₀ + p·g:    S(2, 1) − S(1, 1) = S(1, 0)
# - nothing: not affine in p (S(1, 2) − S(1, 1) ≠ S(1, 1) − S(1, 0)), no effect, or neither
# Compared to √eps relative to the signal's size (rounding in the model's arithmetic).
function _linearkind(signal, θ, sc, j, ::Type{T}) where {T}
    at(a, p) = signal(Base.setindex(Base.setindex(θ, T(a), sc), T(p), j))
    s = try
        (at(1, 0), at(1, 1), at(1, 2), at(2, 1))
    catch                                   # e.g. a DomainError at one of these values
        return nothing
    end
    all(x -> all(isfinite, x), s) || return nothing
    s10, s11, s12, s21 = s
    tol = sqrt(eps(T)) * max(maximum(abs, s10), maximum(abs, s11), maximum(abs, s21), floatmin(T))
    near(x, y) = maximum(abs, x .- y) <= tol
    near(s12 .- s11, s11 .- s10) || return nothing
    near(s11, s10) && return nothing
    near(s21, 2 .* s11) && return :shape
    near(s21 .- s11, s10) && return :offset
    return nothing
end

# The signal equals what the columns predict, a·s₀ + Σ (a·p or p)·g, at a few other values of
# the scale and the linear parameters together (no product of two of them, as a·f·C would be)
function _checklinear(signal, tests, sc, lin, inshape, ::Type{T}) where {T}
    for θ in tests, (a, p0) in ((1.37, 0.29), (0.61, 1.41), (1.83, -0.53))
        base = θ
        for j in lin
            base = Base.setindex(base, zero(T), j)
        end
        s0 = Base.setindex(base, one(T), sc)
        cols = [signal(Base.setindex(s0, one(T), j)) .- signal(s0) for j in lin]
        p = [T(p0 + 0.17 * q) for q in eachindex(lin)]
        predicted = T(a) .* signal(s0) .+ sum(((inshape[q] ? T(a) : one(T)) * p[q]) .* cols[q]
                                             for q in eachindex(lin))
        θr = Base.setindex(base, T(a), sc)
        for q in eachindex(lin)
            θr = Base.setindex(θr, p[q], lin[q])
        end
        actual = signal(θr)
        tol = sqrt(eps(T)) * max(maximum(abs, actual), floatmin(T))
        (all(isfinite, actual) && maximum(abs, actual .- predicted) <= tol) || return false
    end
    return true
end

# The starting point from the dictionary: each candidate scored with its scale and linear
# parameters solved (`_solvelinear`), the lowest SSE kept. If none is usable (NaN signals), the
# scale start from θ0, which holds the BestOf middle values (as `_best_start`).
function _dictionary_start(d::CandidateDictionary{P,T,M}, prob::VoxelProblem{F}, θ0::SVector{P,T},
                           spec, buf) where {F,P,T,M}
    y = prob.y
    n = size(d.basis, 1)
    yy = zero(T)
    @inbounds for i in 1:n
        yy += _wt(prob.w, i, T) * T(y[i])^2
    end
    lo, hi = spec.lower, spec.upper
    bestk, bestsse, bestq = 0, T(Inf), zero(SVector{M,T})
    @inbounds for k in axes(d.basis, 2)
        z = zero(SVector{M,T})
        for i in 1:n
            z += d.basis[i, k] * T(y[i])                # z = BᵀWy
        end
        q, sse = _solvelinear(d, k, z, yy, lo, hi)
        if sse < bestsse                                # NaN never wins
            bestk, bestsse, bestq = k, sse, q
        end
    end
    bestk == 0 && return _scale_start(prob, θ0, spec.scale, lo, hi, buf)
    θ = @inbounds d.points[bestk]
    for j in 1:M
        θ = Base.setindex(θ, bestq[j], d.linear[j])
    end
    return θ
end

# Candidate k's scale and linear parameters for data with z = BᵀWy and y·Wy = yy (the comment at
# the top of the file), within their bounds; returns them (the scale first, as `d.linear`) and
# the SSE there.
@inline function _solvelinear(d::CandidateDictionary{P,T,M}, k, z, yy, lo, hi) where {P,T,M}
    G = @inbounds d.gram[k]
    c0 = @inbounds d.graminv[k] * z                 # unconstrained least squares
    a0 = c0[1]
    ok = isfinite(a0) && a0 > 0                     # a usable scale, to divide by
    # Each linear parameter from c0 (c0/a0 inside the shape), else its start; within its bounds.
    # (The do-block is a closure: the variables it reads, c0 and a0, are never reassigned, or
    # Julia would box them and allocate.)
    p = SVector{M,T}(ntuple(Val(M)) do j
        j == 1 && return one(T)
        l = d.linear[j]
        x = d.inshape[j] ? c0[j] / a0 : c0[j]
        x = isfinite(x) && (ok || !d.inshape[j]) ? x : @inbounds(d.points[k][l])
        clamp(x, lo[l], hi[l])
    end)
    # The scale's best value given them: S = a·v + e, with v the shape's coefficients (1, p, ...)
    # and e the offsets'; a = (v·z − v·Ge)/(v·Gv)
    v = SVector{M,T}(ntuple(j -> j == 1 ? one(T) : d.inshape[j] ? p[j] : zero(T), Val(M)))
    e = SVector{M,T}(ntuple(j -> j == 1 || d.inshape[j] ? zero(T) : p[j], Val(M)))
    Gv = G * v
    a = (dot(v, z) - dot(Gv, e)) / dot(v, Gv)
    sc = d.linear[1]
    a = clamp(isfinite(a) && a > 0 ? a : one(T), lo[sc], hi[sc])
    c = a * v + e
    sse = yy - 2 * dot(c, z) + dot(c, G * c)
    return Base.setindex(p, a, 1), sse
end
