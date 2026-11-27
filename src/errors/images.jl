# =================================================================================================
# A quantity calculated from the images (MTR, ...): its errors from the images' own noise
#
#   propagate(m, θ, g; snr)            the delta method: g at the noise-free images, its SD and
#                                      covariance √(∇g Σ ∇gᵀ), the noise's own bias to second
#                                      order (`secondorder = true`)
#   montecarlo(m, θ, g; snr, noise)    the same measured: simulate, apply g to each copy
#
# g takes one voxel's images (a vector, in the acquisition's order) and returns a number, a
# vector or a named tuple of numbers: `y -> mtr(m, y; reference = :maxoffset).mtr`. Nothing is
# fitted: the images' mean is the model's noise-free signal μ at θ, their covariance Σ the noise
# model's (`noisecovariance`: `snr` relative to the scale parameter, `σ`, `Σ`, `averages`), and
# with Gaussian noise
#
#   E[g(y)] ≈ g(μ) + ½ tr(∇²g Σ),   Cov g(y) ≈ ∇g Σ ∇gᵀ
#
# (the second-order mean: Taylor's expansion to second order, E[ε] = 0, E[εεᵀ] = Σ). Not
# propagate(c, g), crlb's: that propagates a fit's parameter covariance, and an MTR protocol (a
# reference and a few MT images) has no qMT fit to take one from (docs/design.md, item 5).
#
# The truth's own parameters, through the model: `sensitivity`, ∂g/∂p for every parameter θ gives
# as a number (how MTR moves with PSR, with B1), and `uncertain = (B1 = 0.05,)`: a parameter of
# the truth known only to within an SD (B1 across a region, uncorrected), whose spread adds
# Sᵤ·V·Sᵤᵀ to the covariance (Sᵤ those columns of the sensitivity; V = L·diag(sd²)·Lᵀ, inputs that
# move together given as crlb's are, `R1obs = (sd = 0.02, B1 = 2.0)`). Unlike crlb's uncertain
# inputs (a value the fit assumes, measured with an error), g assumes nothing: the SD is how much
# what the scan measures moves with the true value.
# =================================================================================================

"""
    ImagePropagation

What [`propagate`](@ref)`(m, θ, g)` returns, for the quantity `g` computes from the images:
`truth` (g at the noise-free images), `bias` (the noise's own, to second order with
`secondorder = true`; else 0), `sd`, `rmse`, `relbias`, `relsd`, `relrmse` (divided by
|truth|), each shaped as `g`'s value (a number, a vector or a named tuple); `cov`, the
covariance between g's values (a matrix); `noisebias` (`nothing` without `secondorder`);
`sensitivity`, ∂g/∂p by parameter (each shaped as `g`'s value), for every parameter θ gives as
a number; `model`, `noise` (as text), `uncertain` (as given).
"""
struct ImagePropagation{V,S<:NamedTuple,M<:MRIModel}
    truth::V
    bias::V
    sd::V
    rmse::V
    relbias::V
    relsd::V
    relrmse::V
    cov::Matrix{Float64}
    noisebias::Union{Nothing,V}
    sensitivity::S
    model::M
    noise::String
    uncertain::NamedTuple
end

"""
    ImageMonteCarlo

What [`montecarlo`](@ref)`(m, θ, g)` returns: `truth` (g at the noise-free images), and over
the copies where g is finite, `mean`, `bias` (mean − truth), `sd`, `rmse`, `relbias`, `relsd`,
`relrmse`, `se` (the mean's Monte Carlo standard error, sd/√n), each shaped as `g`'s value;
`cov`; `values` (one column per copy, g's values in order); `n`, `finite` (how many copies, and
how many had every value finite); `predicted`, [`propagate`](@ref)'s prediction alongside (to
second order); `model`, `noise`.
"""
struct ImageMonteCarlo{V,P<:ImagePropagation,M<:MRIModel}
    truth::V
    mean::V
    bias::V
    sd::V
    rmse::V
    relbias::V
    relsd::V
    relrmse::V
    se::V
    cov::Matrix{Float64}
    values::Matrix{Float64}
    n::Int
    finite::Int
    predicted::P
    model::M
    noise::String
end

# g's value as a vector (for the derivatives), and a vector back in the shape g gave
function _gvector(x)
    x isa Real && return [x]
    x isa AbstractVector{<:Real} && return x
    x isa NamedTuple && !isempty(x) && all(v -> v isa Real, values(x)) && return collect(promote(values(x)...))
    throw(ArgumentError("`g` returns a number, a vector or a named tuple of numbers from the images " *
                        "(`y -> mtr(m, y).mtr`); got $(typeof(x))"))
end
_gshape(x, v) = x isa Real ? v[1] : x isa NamedTuple ? NamedTuple{keys(x)}(Tuple(v)) : collect(v)

# `uncertain` for the images' quantity: which of the parameters given as numbers (`K`), and their
# covariance V = L·diag(sd²)·Lᵀ (L: the identity plus each input's slopes on others; as crlb's)
function _imageuncertain(u, K)
    u = _byname(u, "uncertain", "(B1 = 0.05,)")
    isempty(u) && return nothing
    for (n, x) in pairs(u)
        n in K || throw(ArgumentError("`uncertain`: `$n` is not a parameter given as a number " *
                                      "($(join(K, ", ")))"))
        sd = _inputsd(x)
        (sd isa Real && isfinite(sd) && sd >= 0) ||
            throw(ArgumentError("`uncertain`: the SD of `$n` must be a number ≥ 0, got $(repr(sd))"))
        for (j, sl) in pairs(_inputslopes(x))
            haskey(u, j) && u[j] isa Real && sl isa Real && isfinite(sl) || throw(ArgumentError(
                "`uncertain`: `$n` moves with `$j`: give `$j` by its SD, and the slope as a finite number"))
        end
    end
    U = keys(u)
    L = Matrix{Float64}(I, length(U), length(U))
    for (i, n) in enumerate(U), (j, c) in enumerate(U)
        haskey(_inputslopes(u[n]), c) && (L[i, j] = Float64(_inputslopes(u[n])[c]))
    end
    V = L * Diagonal([Float64(_inputsd(u[n]))^2 for n in U]) * L'
    return (index = [findfirst(==(n), K) for n in U], V = V)
end

"""
    propagate(m, θ, g; snr or σ or Σ, averages, secondorder = false, uncertain = (;)) -> ImagePropagation

The errors of a quantity calculated from the images, not fitted (MTR): `g` takes one voxel's
images (a vector, in `m`'s acquisition order) and returns a number, a vector or a named tuple of
numbers. By the delta method from the images' own noise (Gaussian, `noisecovariance`'s: `snr`
relative to the model's scale parameter, `σ`, `Σ`, `averages`): g at the noise-free images
`signal(m, θ)` (`truth`), its SD and the covariance between its values, ∇g·Σ·∇gᵀ (MTRs that share
a reference are correlated), and with `secondorder = true` the noise's own bias, ½·tr(∇²g·Σ)
(a ratio's, at low SNR). `sensitivity` gives ∂g/∂p for every parameter θ gives as a number;
`uncertain = (B1 = 0.05,)` adds the spread of a parameter of the truth known only to within an
SD (B1 across a region, uncorrected): (∂g/∂B1)²·0.05², with correlated inputs as crlb takes them.
[`montecarlo`](@ref)`(m, θ, g; ...)` measures the same.

```julia
m = QMTSPGRYarnykh(...)
θ = (PSR = 0.15, kmf = 20.0, T2f = 0.04, T2m = 12e-6, M0f = 1.0, R1f = 1.2, R1m = 1.0, R1obs = 1.1, B1 = 1.0, B0 = 0.0)
g(y) = mtr(m, y; reference = :maxoffset).mtr
e = propagate(m, θ, g; snr = 800)          # e.truth, e.sd, e.cov; e.sensitivity.PSR: ∂MTR/∂PSR
propagate(m, θ, g; snr = 100, secondorder = true, uncertain = (B1 = 0.05,))
```
"""
function propagate(m::MRIModel, θ::NamedTuple, g; snr = nothing, σ = nothing, Σ = nothing,
                   averages = nothing, secondorder::Bool = false, uncertain = (;))
    t = resolve(m, θ)
    μ = signal(m, t)
    Σy = Matrix{Float64}(noisecovariance(m, t; snr, σ, Σ, averages))
    x0 = g(μ)
    gv(y) = _gvector(g(y))
    v = Float64.(gv(μ))
    J = ForwardDiff.jacobian(gv, μ)                          # ∂g/∂y: one row per value of g
    C = J * Σy * J'
    # the parameters θ gives as numbers, through the model: ∂g/∂p (ties follow, as in signal)
    K = Tuple(k for k in keys(θ) if θ[k] isa Real)
    gp(p) = gv(signal(m, merge(θ, NamedTuple{K}(Tuple(p)))))
    Sp = ForwardDiff.jacobian(gp, Float64[θ[k] for k in K])
    u = _imageuncertain(uncertain, K)
    if u !== nothing
        Su = Sp[:, u.index]
        C = C + Su * u.V * Su'
    end
    nb = secondorder ? [0.5 * sum(ForwardDiff.hessian(y -> gv(y)[i], μ) .* Σy) for i in eachindex(v)] : nothing
    bias = nb === nothing ? zero(v) : nb
    sd = sqrt.(max.(diag(C), 0))
    rmse = sqrt.(bias .^ 2 .+ sd .^ 2)
    sh(a) = _gshape(x0, a)
    rel(a) = sh(a ./ abs.(v))
    noise = _noisetext(sqrt.(diag(Σy)), Σ, averages)
    return ImagePropagation(sh(v), sh(bias), sh(sd), sh(rmse), rel(bias), rel(sd), rel(rmse), C,
                            nb === nothing ? nothing : sh(nb),
                            NamedTuple{K}(Tuple(sh(Sp[:, j]) for j in eachindex(K))), m, noise,
                            u === nothing ? (;) : _byname(uncertain, "uncertain", ""))
end

"""
    montecarlo(m, θ, g; snr or σ or Σ, noise, averages, averaging, n = 10_000, rng, uncertain = (;)) -> ImageMonteCarlo

What [`propagate`](@ref)`(m, θ, g)` predicts, measured: `n` noisy copies of the images
(`simulate`, with `noise = :gaussian` or `:rician`, always given), `g` applied to each; the
mean, bias, SD, RMSE and covariance over the copies where g is finite (an MTR whose reference
came out ≤ 0 is NaN: `finite` counts the rest), with the prediction alongside (`predicted`, to
second order: Gaussian noise; with Rician noise the difference is the magnitude images' own
bias). With `uncertain`, each copy draws its own value of each input (as a region's voxels have
their own B1).

```julia
mc = montecarlo(m, θ, g; snr = 100, noise = :rician, n = 20_000)
mc.bias, mc.se, mc.predicted.bias      # measured, its standard error, predicted
```
"""
function montecarlo(m::MRIModel, θ::NamedTuple, g; snr = nothing, σ = nothing, Σ = nothing,
                    noise = nothing, averages = nothing, averaging = nothing, n::Integer = 10_000,
                    rng::AbstractRNG = Random.default_rng(), uncertain = (;))
    pred = propagate(m, θ, g; snr, σ, Σ, averages, secondorder = true, uncertain)
    K = keys(pred.sensitivity)
    u = _byname(uncertain, "uncertain", "(B1 = 0.05,)")
    truth = if isempty(u)
        θ
    else                                       # each copy its own value of each input
        own = NamedTuple{keys(u)}(Tuple(θ[k] .+ Float64(_inputsd(u[k])) .* randn(rng, n) for k in keys(u)))
        moved(k) = foldl((d, (j, sl)) -> d .+ Float64(sl) .* (own[j] .- θ[j]), pairs(_inputslopes(u[k])); init = own[k])
        merge(θ, NamedTuple{keys(u)}(Tuple(moved(k) for k in keys(u))))
    end
    Y = isempty(u) ? simulate(m, truth; snr, σ, Σ, noise, averages, averaging, n, rng) :
                     simulate(m, truth; snr, σ, Σ, noise, averages, averaging, rng)
    x0 = g(signal(m, resolve(m, θ)))
    G = length(_gvector(x0))
    vals = Matrix{Float64}(undef, G, n)
    for c in 1:n
        vals[:, c] .= _gvector(g(Y[c, :]))
    end
    keep = [all(isfinite, view(vals, :, c)) for c in 1:n]
    X = vals[:, keep]
    nf = size(X, 2)
    t = Float64.(_gvector(x0))
    mean_ = vec(sum(X; dims = 2)) ./ nf
    sd = vec(sqrt.(sum((X .- mean_) .^ 2; dims = 2) ./ (nf - 1)))
    rmse = vec(sqrt.(sum((X .- t) .^ 2; dims = 2) ./ nf))
    bias = mean_ .- t
    cv = (X .- mean_) * (X .- mean_)' ./ (nf - 1)
    sh(a) = _gshape(x0, a)
    rel(a) = sh(a ./ abs.(t))
    return ImageMonteCarlo(sh(t), sh(mean_), sh(bias), sh(sd), sh(rmse), rel(bias), rel(sd), rel(rmse),
                           sh(sd ./ sqrt(nf)), cv, vals, Int(n), nf, pred, m,
                           pred.noise * ", $(noise) noise")
end

# At the REPL: one row per value of g (by name, or by position)
function _imagetable(io, names, cols)
    header = first.(cols)
    rows = [Tuple(string(n) for n in (name, (c[2][i] for c in cols)...)) for (i, name) in enumerate(names)]
    header = ("quantity", header...)
    w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for r in rows
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end
_imagenames(x) = x isa NamedTuple ? collect(keys(x)) : x isa Real ? [:value] : collect(1:length(x))
_imagevec(x) = x isa Real ? [x] : collect(values(x))
_num(x) = isfinite(x) ? round(x; sigdigits = 4) : x
_pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)

function Base.show(io::IO, ::MIME"text/plain", r::ImagePropagation)
    print(io, "propagate for ", nameof(typeof(r.model)), "'s images: ", r.noise, ", by the delta method",
          r.noisebias === nothing ? "" : "; bias the noise's own, to second order",
          isempty(r.uncertain) ? "" : "; uncertain: " * join(("$k" for k in keys(r.uncertain)), ", "))
    v(x) = _imagevec(x)
    _imagetable(io, _imagenames(r.truth), [("true", _num.(v(r.truth))), ("bias", _num.(v(r.bias))),
        ("SD", _num.(v(r.sd))), ("rel SD", _pct.(v(r.relsd))), ("RMSE", _num.(v(r.rmse)))])
end
function Base.show(io::IO, ::MIME"text/plain", r::ImageMonteCarlo)
    print(io, "montecarlo for ", nameof(typeof(r.model)), "'s images: ", r.noise, ", ", r.n, " copies",
          r.finite == r.n ? "" : " ($(r.finite) finite)")
    v(x) = _imagevec(x)
    p = r.predicted
    _imagetable(io, _imagenames(r.truth), [("true", _num.(v(r.truth))), ("bias", _num.(v(r.bias))),
        ("± SE", _num.(v(r.se))), ("predicted", _num.(v(p.bias))), ("SD", _num.(v(r.sd))),
        ("predicted SD", _num.(v(p.sd)))])
end
