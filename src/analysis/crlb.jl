# =================================================================================================
# crlb: the bias, SD and RMSE of a fit, predicted from derivatives (no noisy fits)
#
#   truth θ ──────▶ g_true = signal(m, θ)                 the noise-free data
#   fit settings ─▶ free and fixed parameters, bounds     as `fit` uses them
#   ① noise-free fit  θ̂ = fit of g_true, started at the truth;  b = θ̂ − θ;  r₀ = g(θ̂) − g_true
#   ② derivatives     J₀ = ∂g/∂θ at θ̂ (ForwardDiff, ties followed);  H = J₀ᵀJ₀ + Σᵢ r₀ᵢ∇²gᵢ
#   ③ covariance      :sandwich  H⁻¹ J₀ᵀΣJ₀ H⁻¹                       (default)
#                     :biased    (I+∇b) F⁻¹ (I+∇b)ᵀ,  I+∇b = H⁻¹J₀ᵀJ_t,  F = J_tᵀΣ⁻¹J_t
#   ④ report          SD = √diag, RMSE = √(b² + SD²), absolute and relative to the truth
#
# Why a bias at all: a fit that fixes a parameter at a wrong value (kmf = 12.5 for a tissue with
# kmf = 5) cannot reproduce the data, even without noise. Its noise-free answer θ̂ (the
# "pseudo-true" value) differs from the truth by the bias b; noise then spreads the fit around
# θ̂, not around the truth. When nothing fixed is wrong, θ̂ = θ, b = 0, r₀ = 0, H = JᵀJ, and both
# covariances are the classical CRLB σ²(JᵀJ)⁻¹.
#
# The two covariances (both first order in σ; see docs/design.md, "crlb"):
# - the sandwich, H⁻¹(J₀ᵀΣJ₀)H⁻¹: what the least-squares fit itself achieves. Noise ε moves the
#   fit by δ = H⁻¹J₀ᵀε (expand the fit's zero-gradient condition around θ̂), so Cov(δ) is the
#   "meat" J₀ᵀΣJ₀ (how hard the noise pushes) between two slices of "bread" H⁻¹ (how stiff the
#   cost function is). Huber–White (White, Econometrica 1982;50:1), the misspecified CRB
#   (Fortunati et al., IEEE Signal Process Mag 2017;34(6):142).
# - Eq. 3 of Dortch et al., MRM 2018;80:1824, (I + ∇b)F⁻¹(I + ∇b)ᵀ: a bound for any estimator
#   with this bias gradient. Written out, σ²H⁻¹J₀ᵀP_tJ₀H⁻¹ with P_t = J_t(J_tᵀJ_t)⁻¹J_tᵀ, so it
#   is never larger than the sandwich. Equal when the fit is exact (as many observations as
#   free parameters, e.g. the 4-point scheme).
#
# ∇b without extra fits (the slow part of the MATLAB code, which refits perturbed data): the
# fit's gradient G(θ̂, θ) = J₀ᵀ[g(θ̂) − g(θ; true)] is 0 for every truth θ, so differentiating it
# (implicit function theorem) gives ∂θ̂/∂θ = −H⁻¹·∂G/∂θ = H⁻¹J₀ᵀJ_t, one linear solve.
# =================================================================================================

"""
    CRLBResult

What [`crlb`](@ref) returns. Fields, each a named tuple over the fit's free parameters (`c.sd.PSR`,
like `res.params.PSR` of a [`FitResult`](@ref)):

- `fit`: the noise-free fit's values θ̂ (the "pseudo-true" values, where the noisy fits center)
- `bias` (θ̂ − truth), `sd`, `rmse` (√(bias² + SD²)), in the parameters' units
- `relbias`, `relsd`, `relrmse`: the same divided by |truth|

and `cov` (the covariance matrix, free parameters in model order), `cond` (condition number of
the Jacobian with each column scaled by its parameter's true value: large means the parameters
are hard to tell apart), `onbound` (free parameters on a bound in the noise-free fit: held there,
SD 0), `status` (the noise-free fit's [`FitStatus`](@ref)), `method` (`:sandwich` or `:biased`),
`σ` (the noise SD per observation), `truth`, `fixed` (the fit's fixed values) and `model`.
"""
struct CRLBResult{M<:MRIModel,N<:NamedTuple,Tr<:NamedTuple,X<:NamedTuple}
    fit::N
    bias::N
    sd::N
    rmse::N
    relbias::N
    relsd::N
    relrmse::N
    cov::Matrix{Float64}
    cond::Float64
    onbound::Vector{Symbol}
    status::FitStatus
    method::Symbol
    σ::Vector{Float64}
    truth::Tr
    fixed::X
    model::M
end

"""
    crlb(m, truth; snr or σ, fit = (...), cov = :sandwich) -> CRLBResult

The bias, SD and RMSE of each fitted parameter predicted for one voxel, from derivatives: the
Cramér–Rao lower bound (CRLB, also CRB), biased when the fit fixes a parameter at a value other
than the truth's. No noisy fits: the bias comes from one fit of the noise-free data, and the
SD from the derivatives at that fit's answer, first order in the noise.

- `truth`: every parameter of `m` (numbers, or ties), as for [`simulate`](@ref): e.g.
  `nominal(m, :SIR_WM_I)` or `nominal(m; PSR = 0.15)`.
- `snr` (relative to the model's scale parameter, as for `simulate`: σ = M0/snr) or `σ`, a
  number or one value per observation (Gaussian noise).
- `fit`: the fit's settings, the keywords of [`fit`](@ref) in a named tuple; the model's own
  roles when left out. The prediction is for a fit with these settings: `fit = (kmf = Free(),)`
  fits kmf, `fit = (kmf = 15.0,)` fixes it at 15. The bias is zero unless a fixed value differs
  from the truth; the classical CRLB with kmf fixed at its true value is `fit = (kmf = 5.0,)`.
  Bounds are the fit's.
- `cov`: the covariance, `:sandwich` (default) or `:biased`. The sandwich is the covariance of
  the least-squares fit itself (what EQUIP's fit achieves; the misspecified CRB); `:biased` is
  Eq. 3 of Dortch et al., MRM 2018, a bound for any estimator with the same bias gradient. Both
  are the classical CRLB σ²(JᵀJ)⁻¹ when nothing fixed is wrong, and equal each other when the
  fit is exact (as many observations as free parameters); otherwise `:biased` ≤ `:sandwich`.

A free parameter that ends on a bound in the noise-free fit is held there (SD 0; `c.onbound`).
Parameters the data cannot determine (a singular Jacobian) get SD `Inf`. EQUIP's fits are
unweighted: with σ per observation the sandwich is the SD of the unweighted fit, and `:biased`
the bound a weighted fit would reach. Rician noise, bounds hit by noisy fits and nonlinearity
at low SNR are not modelled: simulate and fit for those.

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)                          # kmf = 5 in this tissue
c = crlb(m, θ; snr = 200)                          # the fit fixes kmf at 12.5: biased
c.bias.PSR, c.sd.PSR, c.relrmse.PSR
crlb(m, θ; snr = 200, fit = (kmf = 5.0,))          # kmf fixed at the truth: unbiased
crlb.(Ref(m), [nominal(m, t) for t in (:SIR_WM_I, :SIR_GM)]; snr = 200)   # several tissues
```
"""
function crlb(m::MRIModel, truth::NamedTuple; snr = nothing, σ = nothing, fit::NamedTuple = (;),
              cov::Symbol = :sandwich)
    # `fit` is a keyword here, so inside this function the name means the settings, not the
    # function `fit`; the noise-free fit below calls `fit!(init(...))`, which is what `fit` does.
    cov in (:sandwich, :biased) ||
        throw(ArgumentError("`cov` must be :sandwich or :biased, got $(repr(cov))"))
    for (k, x) in pairs(fit)
        k in _OPTIONS && throw(ArgumentError("`fit = (; $k)`: crlb runs its own noise-free fit; " *
                                             "give only parameter settings"))
        _hasmap(x) &&
            throw(ArgumentError("`fit = (; $k)`: crlb is for one voxel; give numbers, not maps"))
    end
    any(_hasmap, values(truth)) &&
        throw(ArgumentError("crlb is for one voxel: give numbers, not maps"))
    θ = resolve(m, truth)                    # every true value a number, in model order
    spec = setup(m, Float64; fit...)         # the fit's free and fixed parameters, and bounds
    F = freenames(spec)
    P = length(F)
    N = nobs(m)
    σv = _crlb_sigma(m, θ, snr, σ, N)
    θt = [Float64(θ[n]) for n in F]          # the free parameters' true values

    # ① The noise-free fit, started at the truth (inside the fit's bounds), with tight
    #    tolerances: the bias is a small difference of two values, so it needs every digit.
    g_true = signal(m, θ)
    starts = (; (n => Free(start = θt[j], lower = spec.lower[j], upper = spec.upper[j])
                 for (j, n) in enumerate(F))...)
    res = fit!(init(m; precision = Float64, solver = _CRLB_SOLVER, merge(fit, starts)...), g_true)
    θ̂ = [Float64(res.params[n]) for n in F]
    b = θ̂ .- θt

    # A parameter on a bound stays there: it is left out of the derivatives (SD 0, ∇b row 0).
    near(x, bound) = isfinite(bound) && abs(x - bound) <= 1e-9 * max(1, abs(bound))
    U = [j for j in 1:P if !(near(θ̂[j], spec.lower[j]) || near(θ̂[j], spec.upper[j]))]
    onbound = [F[j] for j in 1:P if !(j in U)]

    # ② Derivatives at θ̂, over the parameters not on a bound. `gfit` is the fit's signal: its
    #    fixed values (ties as written, followed by `signal`) and the free ones from φ.
    #    `expand` puts φ_U back among the held values; `convert` makes room for dual numbers.
    gfit(φ) = signal(m, merge(res.fixed, NamedTuple{F}(Tuple(φ))))
    expand(φU) = (φ = convert(Vector{eltype(φU)}, θ̂); φ[U] = φU; φ)
    C = zeros(P, P)
    condJ = Inf
    if !isempty(U)
        J0 = ForwardDiff.jacobian(φU -> gfit(expand(φU)), θ̂[U])
        H = ForwardDiff.hessian(φU -> sum(abs2, gfit(expand(φU)) .- g_true) / 2, θ̂[U])
        condJ = _condition(J0, θt[U])
        # ③ The covariance. Parameters the data cannot determine (fewer observations than
        #    free parameters, or columns of J that are nearly dependent) get Inf. The test
        #    cannot be left to the factorization: rounding can make a singular H look positive
        #    definite, and the sandwich then gives finite, meaningless values. A failed
        #    Cholesky factorization (H or F not positive definite) also gives Inf, rather than a
        #    negative variance hidden by abs() as in the MATLAB code.
        Hc = cholesky(Symmetric(H); check = false)
        if !(condJ < _COND_MAX) || !issuccess(Hc)
            C[U, U] .= Inf
        elseif cov === :sandwich
            A = Hc \ J0'                                     # H⁻¹J₀ᵀ: how noise moves the fit
            C[U, U] = A * Diagonal(σv .^ 2) * A'
        else
            # J_t: the true signal's Jacobian, the truth's own ties followed (R1m = R1f moves
            # with R1f, as in the MATLAB's `R1 + eps` on both).
            Jt = ForwardDiff.jacobian(x -> signal(m, merge(truth, NamedTuple{F}(Tuple(x)))), θt)
            Fc = cholesky(Symmetric(Jt' * Diagonal(1 ./ σv .^ 2) * Jt); check = false)
            if !(_condition(Jt, θt) < _COND_MAX) || !issuccess(Fc)
                C[U, U] .= Inf
            else
                D = zeros(P, P)                              # I + ∇b = ∂θ̂/∂θ; rows on a bound 0
                D[U, :] = Hc \ (J0' * Jt)
                X = Fc.L \ D'                                # D F⁻¹ Dᵀ = XᵀX, F = LLᵀ
                C = X' * X
            end
        end
    end

    # ④ Report, by name.
    sd = sqrt.(diag(C))
    rmse = sqrt.(b .^ 2 .+ sd .^ 2)
    rel(x) = x ./ abs.(θt)
    nt(x) = NamedTuple{F}(Tuple(x))
    return CRLBResult(nt(θ̂), nt(b), nt(sd), nt(rmse), nt(rel(b)), nt(rel(sd)), nt(rel(rmse)),
                      Matrix(Symmetric(C)), condJ, onbound, res.status, cov, σv, truth,
                      res.fixed, m)
end

# The noise-free fit's solver: the step tolerance alone, tight, and room to get there.
const _CRLB_SOLVER = LM(maxiter = 1000, xtol = 1e-13, ftol = 0.0, gtol = 0.0)

# The condition number of J with each column scaled by its parameter's true value (relative
# sensitivities, so the units drop out; 1 for a true value of 0). Inf when there are fewer
# observations than parameters: `cond` of a wide matrix would ignore the missing directions.
_condition(J, θ) = size(J, 1) < size(J, 2) ? Inf :
                   cond(J * Diagonal([x == 0 ? 1.0 : abs(x) for x in θ]))
# Beyond this the covariance is below rounding error (H = JᵀJ squares the condition number, and
# 1e16 is 1/eps): the parameters cannot be told apart, so their SD is Inf.
const _COND_MAX = 1e8

_hasmap(x::AbstractArray) = true
_hasmap(x::Free) = x.start isa AbstractArray
_hasmap(x::Fixed) = x.value isa AbstractArray
_hasmap(x) = false

# The noise SD of each observation: from `snr` (relative to the scale parameter, as `simulate`)
# or from `σ` (a number, or one per observation).
function _crlb_sigma(m, θ, snr, σ, N)
    (snr === nothing) == (σ === nothing) &&
        throw(ArgumentError("give the noise level as `snr` or as `σ` (one of them)"))
    if snr !== nothing
        scale = _scalename(m)
        scale === nothing && throw(ArgumentError("$(nameof(typeof(m))) has no " *
                                   "`Free(start = :scale)` parameter to define the SNR; give `σ`"))
        return fill(Float64(θ[scale]) / snr, N)
    end
    σ isa Real && return fill(Float64(σ), N)
    length(σ) == N ||
        throw(DimensionMismatch("`σ` has $(length(σ)) values; the model has $N observations"))
    return Float64.(collect(σ))
end

# -------------------------------------------------------------------------------------------------
# Display: at the REPL a table, one row per free parameter, in the style of `compare`
# -------------------------------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", c::CRLBResult)
    f(x) = isfinite(x) ? string(round(x; sigdigits = 4)) : string(x)
    pct(x) = isfinite(x) ? string(round(100 * x; sigdigits = 3), "%") : string(x)
    σ = c.σ
    noise = all(==(σ[1]), σ) ? "σ = $(f(σ[1]))" : "σ = $(f(minimum(σ))) to $(f(maximum(σ)))"
    how = c.method === :sandwich ? "sandwich (the least-squares fit)" :
                                   "biased CRLB (Dortch 2018 Eq. 3)"
    print(io, "crlb for ", nameof(typeof(c.model)), ", ", length(σ), " observations, ", noise,
          ", covariance: ", how)
    # The fit's fixed values, with the truth where it differs: the source of any bias.
    if !isempty(c.fixed)
        item(n, x) = (t = c.truth[n]; s = "$n = $(_showvalue(x))";
                      x isa Real && t isa Real && x != t ? s * " (true $(_showvalue(t)))" : s)
        print(io, "\nfixed in the fit: ", join((item(n, x) for (n, x) in pairs(c.fixed)), ", "))
    end
    isconverged(c.status) ||
        print(io, "\nthe noise-free fit did not converge ($(c.status)): the bias may be inaccurate")
    isempty(c.onbound) ||
        print(io, "\non a bound in the noise-free fit (held there, SD 0): ", join(c.onbound, ", "))
    header = ("parameter", "true", "noise-free fit", "bias", "rel bias", "SD", "rel SD", "RMSE",
              "rel RMSE")
    rows = [(string(n), f(c.fit[n] - c.bias[n]), f(c.fit[n]), f(c.bias[n]), pct(c.relbias[n]),
             f(c.sd[n]), pct(c.relsd[n]), f(c.rmse[n]), pct(c.relrmse[n])) for n in keys(c.fit)]
    w = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    print(io, "\n  ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
    for r in rows
        print(io, "\n  ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(r)), "  ")))
    end
end

Base.show(io::IO, c::CRLBResult) =
    print(io, "CRLBResult(", nameof(typeof(c.model)), ": ", join(keys(c.fit), ", "), ")")
