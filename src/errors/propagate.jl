# =================================================================================================
# propagate: crlb's bias, SD and RMSE for any quantity computed from the fitted parameters
#
#   propagate(c, p -> 1 / p.T1)                          R1 from a T1 fit
#   propagate(c, p -> (kfm = p.PSR * p.kmf, f = p.PSR / (1 + p.PSR)))   several, by name
#
# The delta method, as crlb computes a model's own derived quantities (`derived`, crlb.jl's
# `_derivedcrlb`), with any function in their place: the noise-free fit's value g(θ̂), its bias
# against g at the truth, SD √(∇g C ∇gᵀ) and the covariance ∇g C ∇gᵀ between several (C crlb's
# covariance, ∇g taken with dual numbers), and with `secondorder = true` the noise's own,
# ∇g·b + ½ tr(∇²g C). After the fact, from a result: one crlb, then as many functions as wanted.
# (The tier 2 plan, step 7; roadmap item 16, "Outputs".)
# =================================================================================================

"""
    Propagation

What [`propagate`](@ref) returns: for the quantity `g` computes, `fit` (g at the noise-free
fit, where the noisy fits center), `truth` (g at the true values), `bias` (with crlb's
`secondorder = true`, the noise's own too, kept apart in `noisebias`), `sd`, `rmse`,
`relbias`, `relsd`, `relrmse` (divided by |truth|), and `cov`, the covariance between the
quantities. Each a number when `g` returns a number, a named tuple by name when it returns a
named tuple (`r.sd.R1`); `crlb` is the result it was propagated from.
"""
struct Propagation{N,B,C<:CRLBResult}
    fit::N
    truth::N
    bias::N
    sd::N
    rmse::N
    relbias::N
    relsd::N
    relrmse::N
    cov::Matrix{Float64}
    noisebias::B               # nothing, or the noise's own bias to second order (in `bias`)
    crlb::C
end

"""
    propagate(c::CRLBResult, g) -> Propagation

The bias, SD and RMSE that crlb predicts for a quantity computed from the fitted parameters:
`g` takes every parameter by name (a named tuple: the free ones as fitted, the fixed ones at the
fit's values) and returns a number, or a named tuple of numbers for several quantities at once.
By the delta method, from `c`'s covariance, as for a model's own [`derived`](@ref) quantities
(the same numbers for those): SD √(∇g C ∇gᵀ), the bias g(θ̂) − g(truth) (what a fixed value
assumed wrong does to g), and with `crlb(...; secondorder = true)` the noise's own,
∇g·b + ½ tr(∇²g C); `cov` holds the covariance between several. First order otherwise, as crlb:
for a strongly curved g at low SNR, [`montecarlo`](@ref) is the reference (compute g from
`mc.result`'s estimates). For a sweep, `propagate.(s.results, Ref(g))`.

```julia
m = T1MFA(FA = [2, 5, 10, 15, 20, 30], TR = 0.015)
c = crlb(m, (M0 = 1000.0, T1 = 1.0, B1 = 1.0); snr = 100, secondorder = true)
propagate(c, p -> 1 / p.T1)                    # R1: its noise bias, SD and RMSE (1/s)
m4 = QMTSIR(; SIR_5POINT...)
c4 = crlb(m4, nominal(m4, :SIR_WM_I); snr = 200, kmf = Free())
propagate(c4, p -> (kfm = p.PSR * p.kmf, f = p.PSR / (1 + p.PSR)))   # the forward rate, the
                                               # macromolecular fraction; r.sd.f, r.cov
```
"""
function propagate(c::CRLBResult, g)
    F = _freenames(c)
    P = length(F)
    fixed = map(_tie, c.fixed)                      # the fit's fixed values, ties as ties
    # g of the free values θ (a static vector, of dual numbers for the derivatives), as a
    # named tuple of numbers: a number becomes `(value = x,)` and is given back as a number
    function gθ(θ)
        x = g(_params(Val(F), θ, fixed))
        x isa Real && return (value = x,)
        x isa NamedTuple && all(v -> v isa Real, values(x)) && return x
        throw(ArgumentError("propagate: `g` returns a number or a named tuple of numbers " *
            "(`p -> 1 / p.T1`, `p -> (R1 = 1 / p.T1, R2 = 1 / p.T2)`); got $(typeof(x))"))
    end
    θ̂ = SVector{P,Float64}(Tuple(c.fit[F]))
    ĝ = map(Float64, gθ(θ̂))
    G = keys(ĝ)
    xt = g(resolve(c.model, c.truth))                # g at the true values
    scalar = xt isa Real                             # a number in, numbers out
    gt = map(Float64, scalar ? (value = xt,) : xt)
    ∇g = ForwardDiff.jacobian(θ -> SVector(values(gθ(θ))), θ̂)        # length(G) × P
    C = SMatrix{P,P,Float64}(c.cov)
    finite = all(isfinite, C)
    Cg = finite ? Matrix(∇g * C * ∇g') : fill(Inf, length(G), length(G))
    sd = NamedTuple{G}(Tuple(finite ? sqrt.(max.(diag(Cg), 0)) : fill(Inf, length(G))))
    # The noise's own bias, when crlb computed the parameters' (`secondorder = true`), from the
    # image noise's covariance; with uncertain inputs too, c.cov holds theirs as well
    nb = if c.noisebias === nothing
        nothing
    else
        c.uncertain === nothing || throw(ArgumentError("propagate of a crlb with both " *
            "`secondorder` and `uncertain`: not yet (its covariance holds the inputs' too)"))
        b2 = SVector{P,Float64}(Tuple(c.noisebias[F]))
        NamedTuple{G}(Tuple(_derivednoisebias(gθ, θ̂, ∇g, b2, C)))
    end
    bias = nb === nothing ? map(-, ĝ, gt) : map((x, t, n) -> x - t + n, ĝ, gt, nb)
    rmse = map((b, s) -> sqrt(b^2 + s^2), bias, sd)
    rel(x) = map((x, t) -> x / abs(t), x, gt)
    out(x) = x === nothing ? nothing : scalar ? x.value : x
    return Propagation(out(ĝ), out(gt), out(bias), out(sd), out(rmse), out(rel(bias)),
                       out(rel(sd)), out(rel(rmse)), Cg, out(nb), c)
end

function Base.show(io::IO, ::MIME"text/plain", r::Propagation)
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    c = r.crlb
    print(io, "propagate for ", nameof(typeof(c.model)), ": crlb's ",
          _noisetext(c.σ, c.Σ, c.averages), ", by the delta method",
          r.noisebias === nothing ? "" : "; bias with the noise's own, to second order")
    as(x) = x isa NamedTuple ? x : (value = x,)
    nb = r.noisebias === nothing ? nothing : as(r.noisebias)
    header = ("quantity", "true", "noise-free fit", "bias", (nb === nothing ? () : ("noise bias",))...,
              "rel bias", "SD", "rel SD", "RMSE", "rel RMSE")
    rows = [(string(n), f(as(r.truth)[n]), f(as(r.fit)[n]), f(as(r.bias)[n]),
             (nb === nothing ? () : (f(nb[n]),))..., pct(as(r.relbias)[n]), f(as(r.sd)[n]),
             pct(as(r.relsd)[n]), f(as(r.rmse)[n]), pct(as(r.relrmse)[n])) for n in keys(as(r.fit))]
    w = [maximum(length, (header[j], (x[j] for x in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for x in rows
        print(io, "\n  ", rstrip(join((rpad(x[j], w[j]) for j in eachindex(x)), "  ")))
    end
end

Base.show(io::IO, r::Propagation) =
    print(io, "Propagation(", nameof(typeof(r.crlb.model)), ": ",
          r.fit isa NamedTuple ? join(keys(r.fit), ", ") : "one quantity", ")")
