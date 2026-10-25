# =================================================================================================
# The pieces of crlb, one function each, written as the formulas read
#
#   noisecovariance(m, θ; snr | σ | Σ)     Σ: the noise covariance between observations (N×N)
#   noisefree(m, θ; fit)                   θ̂: the fit of the noise-free data g(θ), started at θ
#   bias(m, θ; fit)                        b = θ̂ − θ, for the fit's free parameters
#   jacobian(m, θ; fit)                    J = ∂g/∂θ (N×P), P the fit's free parameters
#   hessian(m, θ, y; fit, Σ)               H = ∇²[½ rᵀΣ⁻¹r], r = g(θ) − y; = JᵀΣ⁻¹J + Σᵢ (Σ⁻¹r)ᵢ ∇²gᵢ
#   informationmatrix(m, θ; noise, fit)    F = JᵀΣ⁻¹J (Fisher), or the observed information
#   biasgradient(m, θ; fit, method)        ∇b = ∂b/∂θ (P×P): implicit function theorem, or
#                                          ForwardDiff through Gauss–Newton steps
#   sandwich(H, J, Σ)                      H⁻¹JᵀΣJH⁻¹: the least-squares fit's covariance
#   biasedcrlb(∇b, F)                      (I + ∇b)F⁻¹(I + ∇b)ᵀ: Dortch et al., MRM 2018, Eq. 3
#
# θ is always every parameter's value (numbers, or ties), as for `signal`: e.g.
# `nominal(m, :SIR_WM_I)`. `fit` is a fit's settings, the keywords of `fit` in a named tuple
# (`kmf = Free()` frees kmf, `kmf = 15.0` fixes it). A function that runs a fit (`noisefree`,
# `bias`, `biasgradient`) uses all of it, fixed values included: that is the fit's assumption.
# The derivatives (`jacobian`, `hessian`, `informationmatrix`) are taken AT θ: `fit` says which
# parameters are free (the columns of J), and every value, fixed ones included, is θ's. So with
# a fit that fixes kmf at 12.5 for a tissue with kmf = 5:
#
#   jacobian(m, θ; fit)                          the true signal's J (kmf = 5), J_t of Eq. 3
#   jacobian(noisefree(m, θ; fit))               the fit's J at its answer θ̂ (kmf = 12.5), J₀
#   jacobian(m, merge(θ, (kmf = 12.5,)); fit)    the fit's J at the truth's free values
#
# Readable rather than fast: plain matrices, ForwardDiff on the allocating `signal`, and Julia's
# LinearAlgebra (`\`, `cholesky`) for the formulas: a few hundred µs a call. The protocol
# optimizer (millions of calls) uses the allocation-free kernel in crlb_kernel.jl instead, which
# accumulates JᵀJ and J₀ᵀJ_t from dual numbers row by row (as the solver's `normal_equations`
# does) and never forms J; it is tested against these.
# =================================================================================================

# The noise-free fit's solver: the step tolerance alone, tight, and room to get there (the bias
# is a small difference of two values and needs every digit).
const _CRLB_SOLVER = LM(maxiter = 1000, xtol = 1e-13, ftol = 0.0, gtol = 0.0)

# The free parameters of a fit with these settings, in model order, e.g. (:PSR, :R1f, :Sf, :M0f).
_freenames(m::MRIModel, fit::NamedTuple) = freenames(setup(m, Float64; fit...))

# Every value of a one-voxel fit as the fit used them: the fitted ones and the fixed ones, ties
# as written (`R1m = :R1f`), so derivatives follow the ties as the fit did.
_values(r::FitResult) = merge(r.fixed, r.params)

# Symmetric part: sums like JᵀΣ⁻¹J and H⁻¹JᵀΣJH⁻¹ come out symmetric only to rounding, and
# `isposdef`, `cholesky` want them exactly symmetric.
_sym(A) = (A + A') / 2

# -------------------------------------------------------------------------------------------------
# The noise
# -------------------------------------------------------------------------------------------------

"""
    noisecovariance(m, θ; snr or σ or Σ) -> Diagonal or Matrix

The covariance Σ of Gaussian noise between the observations of `m` (N×N), from one of:

- `snr`: relative to the model's scale parameter at the values θ, as for [`simulate`](@ref):
  σ = M0/snr, the same for every observation, independent: Σ = σ²I (a `Diagonal`)
- `σ`: a number (Σ = σ²I), or one SD per observation (Σ = Diagonal(σ²): independent noise, e.g.
  different numbers of averages)
- `Σ`: the covariance matrix itself (correlated noise, e.g. after denoising), checked
  (symmetric, positive definite) and returned

```julia
noisecovariance(m, nominal(m, :SIR_WM_I); snr = 200)       # 4×4 Diagonal, (M0f/200)²
```
"""
function noisecovariance(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, Σ = nothing)
    N = nobs(m)
    count(!isnothing, (snr, σ, Σ)) == 1 ||
        throw(ArgumentError("give the noise level as `snr`, `σ` or `Σ` (one of them)"))
    if snr !== nothing
        scale = _scalename(m)
        scale === nothing && throw(ArgumentError("$(nameof(typeof(m))) has no " *
                                   "`Free(start = :scale)` parameter to define the SNR; give `σ`"))
        s = Float64(resolve(m, θ)[scale]) / snr       # σ = M0/snr
        return Diagonal(fill(s^2, N))
    elseif σ isa Real
        return Diagonal(fill(Float64(σ)^2, N))
    elseif σ !== nothing
        σ isa AbstractVector || throw(ArgumentError("`σ` is a number or one SD per observation; " *
                                                    "for a covariance matrix, give `Σ`"))
        length(σ) == N ||
            throw(DimensionMismatch("`σ` has $(length(σ)) values; the model has $N observations"))
        return Diagonal(Float64.(collect(σ)) .^ 2)
    end
    _noisefactor(Σ, N)                    # the checks `simulate` makes (sim/simulate.jl)
    return Σ
end

# -------------------------------------------------------------------------------------------------
# The noise-free fit and its bias
# -------------------------------------------------------------------------------------------------

"""
    noisefree(m, θ; fit = (...), simulate_with) -> FitResult

The fit of the noise-free data `signal(m, θ)` with the fit settings `fit` (the keywords of
[`fit`](@ref) in a named tuple; the model's own roles when left out), started at θ's free values,
with the fit's bounds and tight tolerances. Its values θ̂ (`r.params`) are where noisy fits
center (the "pseudo-true" values): θ itself unless a fixed value of the fit differs from θ's
(`fit = (kmf = 12.5,)` for a tissue with kmf = 5), or the data come from another model
(`simulate_with = m2 => θ2`, as for [`crlb`](@ref)). A [`FitResult`](@ref): [`bias`](@ref),
[`jacobian`](@ref) and [`hessian`](@ref) take it.
"""
function noisefree(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;), simulate_with = nothing)
    _check_fit_settings(fit)
    dm, dθ = _datasource(m, θ, simulate_with)
    y = signal(dm, dθ)                                   # the noise-free data
    θt = resolve(m, θ)
    F = _freenames(m, fit)
    # Start every free parameter at θ's value, keeping the bounds the fit gives it.
    starts = NamedTuple{F}(map(n -> _start_at(get(fit, n, nothing), θt[n]), F))
    return fit!(init(m; precision = Float64, solver = _CRLB_SOLVER, fit..., starts...), y)
end
_start_at(x::Free, v) = Free(start = v, lower = x.lower, upper = x.upper)
_start_at(x, v) = Free(start = v)                        # the model's role: its bounds kept

"""
    bias(m, θ; fit = (...), simulate_with) -> named tuple
    bias(r::FitResult, θ) -> named tuple

The bias b = θ̂ − θ of each free parameter of the fit: the noise-free fit's values
([`noisefree`](@ref)) minus the truth θ. Zero unless the fit fixes a parameter at a wrong value
or the data come from another model (`simulate_with`). The second form takes the noise-free fit
itself.
"""
bias(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;), simulate_with = nothing) =
    bias(noisefree(m, θ; fit, simulate_with), θ)
function bias(r::FitResult, θ::NamedTuple)
    θt = resolve(r.model, θ)
    F = keys(r.params)
    return NamedTuple{F}(map(n -> r.params[n] - θt[n], F))
end

# Which free parameters of a one-voxel fit ended on one of its bounds (within 1e-9, relative for
# large bounds): crlb holds them there (SD 0; ∂θ̂/∂θ = 0 for them). The bounds are the fit's own,
# from the settings it keeps (`r.settings`: its parameter keywords and options).
function _onbound(r::FitResult)
    spec = setup(r.model, Float64; Base.structdiff(r.settings, NamedTuple{_OPTIONS})...)
    F = keys(r.params)
    return [F[j] for j in eachindex(F)
            if _near(r.params[j], spec.lower[j]) || _near(r.params[j], spec.upper[j])]
end
@inline _near(x, bound) = isfinite(bound) && abs(x - bound) <= 1e-9 * max(1, abs(bound))

# -------------------------------------------------------------------------------------------------
# Derivatives: ForwardDiff on `signal`, with respect to the free parameters F (in model order)
# -------------------------------------------------------------------------------------------------

"""
    jacobian(m, θ; fit = (...)) -> Matrix (N×P)
    jacobian(r::FitResult) -> Matrix

The Jacobian J = ∂g/∂θ of the signal g = `signal(m, θ)`: row i observation i, column j the
j-th free parameter of a fit with settings `fit` (in model order, as `keys(r.params)`; the
model's free parameters when left out). At the values θ, every one of them θ's: `fit` chooses
the columns, not the point (see the top of src/analysis/derivatives.jl). Ties in θ are followed:
with `R1m = :R1f`, the R1f column includes R1m's change. ForwardDiff, exact to rounding.

The second form: at a one-voxel fit's answer, its free parameters, its fixed values and ties
(e.g. `jacobian(noisefree(m, θ; fit))`, the J₀ of [`crlb`](@ref)).

```julia
m = QMTSIR(; SIR_4POINT...)
θ = nominal(m, :SIR_WM_I)
J = jacobian(m, θ)                           # 4×4: PSR, R1f, Sf, M0f
J5 = jacobian(m, θ; fit = (kmf = Free(),))   # 4×5: kmf too
```

Also exported by Zygote and DifferentiationInterface: with either loaded, write
`EQUIP.jacobian`.
"""
jacobian(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;)) =
    _jacobian(m, θ, _freenames(m, fit))
function jacobian(r::FitResult)
    _onevoxel(r) || throw(ArgumentError("jacobian of a fit: give a one-voxel FitResult"))
    return _jacobian(r.model, _values(r), keys(r.params))
end
function _jacobian(m::MRIModel, θ::NamedTuple, F::Tuple)
    θt = resolve(m, θ)
    x = [Float64(θt[n]) for n in F]                       # the free values
    g(x) = signal(m, merge(θ, NamedTuple{F}(Tuple(x))))   # the signal as a function of them
    return ForwardDiff.jacobian(g, x)
end

"""
    hessian(m, θ, y; fit = (...), Σ = I) -> Matrix (P×P)
    hessian(r::FitResult, y; Σ = I) -> Matrix

The Hessian of the cost ½ rᵀΣ⁻¹r, r = g(θ) − y the residual of the data `y`, with respect to the
free parameters (as for [`jacobian`](@ref)), at θ:

    H = JᵀΣ⁻¹J + Σᵢ (Σ⁻¹r)ᵢ ∇²gᵢ

With `Σ = I` (the default), the cost the fit minimizes, ½‖g − y‖², and H = JᵀJ + Σᵢ rᵢ∇²gᵢ:
JᵀJ is the Gauss–Newton part the solver uses; the second, the residual's curvature, is zero
when the model fits `y` exactly and not when an assumption is wrong. With Σ the noise
covariance, the negative log-likelihood's Hessian: the observed information. ForwardDiff
(nested duals), exact to rounding.

The second form: at a one-voxel fit's answer, e.g. `hessian(noisefree(m, θ; fit), signal(m, θ))`,
the H of [`crlb`](@ref).
"""
hessian(m::MRIModel, θ::NamedTuple, y::AbstractVector; fit::NamedTuple = (;), Σ = I) =
    _hessian(m, θ, _freenames(m, fit), y, Σ)
function hessian(r::FitResult, y::AbstractVector; Σ = I)
    _onevoxel(r) || throw(ArgumentError("hessian of a fit: give a one-voxel FitResult"))
    return _hessian(r.model, _values(r), keys(r.params), y, Σ)
end
function _hessian(m::MRIModel, θ::NamedTuple, F::Tuple, y, Σ)
    length(y) == nobs(m) ||
        throw(DimensionMismatch("`y` has $(length(y)) values; the model has $(nobs(m)) observations"))
    θt = resolve(m, θ)
    x = [Float64(θt[n]) for n in F]
    W = inv(Σ)                                            # the weights Σ⁻¹ (I: none)
    function cost(x)
        r = signal(m, merge(θ, NamedTuple{F}(Tuple(x)))) .- y
        return dot(r, W * r) / 2
    end
    return _sym(ForwardDiff.hessian(cost, x))
end

"""
    informationmatrix(m, θ; snr or σ or Σ, fit = (...), expected = true, y) -> Matrix (P×P)

The information matrix of the free parameters (as for [`jacobian`](@ref)) at the values θ, for
Gaussian noise (the noise as for [`noisecovariance`](@ref)). `expected = true`: the Fisher
information F = JᵀΣ⁻¹J (Dortch et al., MRM 2018, Eq. 4: JᵀJ/σ² for one σ); its inverse is the
CRLB of an unbiased estimator. `expected = false`: the observed information, the Hessian of the
negative log-likelihood of the data `y` ([`hessian`](@ref) with Σ), which adds the residual's
curvature; `y` is the noise-free signal at θ when left out, and then the two are equal. At a
noise-free fit's answer with a wrong assumption they differ; [`crlb`](@ref) reports both there
(`informationmatrix(c)`). StatsAPI's function, with EQUIP's methods.

```julia
F = informationmatrix(m, θ; snr = 200)       # at the truth
sqrt.(diag(inv(F)))                          # the classical CRLB of each SD
```
"""
function informationmatrix(m::MRIModel, θ::NamedTuple; snr = nothing, σ = nothing, Σ = nothing,
                           fit::NamedTuple = (;), expected::Bool = true, y = nothing)
    Σn = noisecovariance(m, θ; snr, σ, Σ)
    if expected
        J = jacobian(m, θ; fit)
        return _sym(J' * (Σn \ J))                        # F = JᵀΣ⁻¹J
    end
    return hessian(m, θ, something(y, signal(m, θ)); fit, Σ = Σn)
end

"""
    biasgradient(m, θ; fit = (...), method = :implicit, steps = 3) -> Matrix (P×P)
    biasgradient(r::FitResult, θ) -> Matrix

The bias gradient ∇b = ∂b/∂θ: how the bias of each free parameter of the fit (row) changes with
the true value of each (column), in Eq. 3 of Dortch et al., MRM 2018 (see [`biasedcrlb`](@ref)).
Zero when nothing fixed is wrong. Two ways, which agree (tested):

- `:implicit` (default): the noise-free fit θ̂ ([`noisefree`](@ref)) makes the fit's gradient
  G(θ̂, θ) = J₀ᵀ[g(θ̂) − g(θ)] zero for every truth θ; differentiating that identity (the
  implicit function theorem) gives I + ∇b = ∂θ̂/∂θ = H⁻¹J₀ᵀJ_t, one linear solve: J₀ and H at
  θ̂ ([`jacobian`](@ref), [`hessian`](@ref)), J_t at θ. No refits of perturbed data (the
  MATLAB code's finite differences). A parameter that ends on a bound stays there: its row of
  I + ∇b is 0.
- `:unrolled`: ForwardDiff carried through `steps` Gauss–Newton steps of the noise-free fit
  from the truth, θ̂ ← θ̂ − (JᵀJ)⁻¹Jᵀ(g(θ̂) − g(θ)): the derivative of the iterations
  themselves. No bounds; accurate when a few steps reach the fit (3 for the 4-point scheme,
  kmf assumed 12.5: to 1e-10). The form to use where derivatives with respect to the protocol
  are needed too.

`NaN` when the data cannot determine the parameters (`:implicit`). The second form is the
implicit method at a noise-free fit already made, `r = noisefree(m, θ; fit)` (as `crlb` does).
"""
function biasgradient(m::MRIModel, θ::NamedTuple; fit::NamedTuple = (;), method::Symbol = :implicit,
                      steps::Integer = 3)
    method in (:implicit, :unrolled) ||
        throw(ArgumentError("`method` must be :implicit or :unrolled, got $(repr(method))"))
    method === :unrolled && return _biasgradient_unrolled(m, θ, fit, steps)
    return biasgradient(noisefree(m, θ; fit), θ)
end
function biasgradient(r::FitResult, θ::NamedTuple)
    m = r.model
    J₀ = jacobian(r)                             # N×P at θ̂, the fit's fixed values
    H = hessian(r, signal(m, θ))                 # P×P, of ½‖g(θ̂) − g(θ)‖²
    Jt = _jacobian(m, θ, keys(r.params))         # N×P: the true signal's, at θ
    P = size(Jt, 2)
    onbound = _onbound(r)
    U = findall(n -> !(n in onbound), collect(keys(r.params)))       # not on a bound
    _identifiable(J₀[:, U]' * J₀[:, U], size(J₀, 1)) && isposdef(H[U, U]) || return fill(NaN, P, P)
    D = zeros(P, P)                              # I + ∇b = ∂θ̂/∂θ; 0 for a parameter on a bound
    D[U, :] = H[U, U] \ (J₀[:, U]' * Jt)         # the implicit function theorem
    return D - I
end

# ∂θ̂/∂θ by differentiating `steps` Gauss–Newton iterations from the truth: the noise-free data
# g(θ) from the truth's free values x (the truth's own fixed values and ties), fitted with the
# fit's fixed values. Nested ForwardDiff: the outer duals carry ∂/∂x through every step.
function _biasgradient_unrolled(m::MRIModel, θ::NamedTuple, fit::NamedTuple, steps::Integer)
    spec = setup(m, Float64; fit...)
    F = freenames(spec)
    θfit = merge(θ, map(_untie, spec.fixed))     # the values the fit assumes
    g(vals, x) = signal(m, merge(vals, NamedTuple{F}(Tuple(x))))
    function θ̂(x)
        y = g(θ, x)                              # the noise-free data, true values x
        φ = x                                    # the fit, started at the truth
        for _ in 1:steps
            J = ForwardDiff.jacobian(p -> g(θfit, p), φ)
            φ = φ - (J' * J) \ (J' * (g(θfit, φ) - y))
        end
        return φ
    end
    θt = resolve(m, θ)
    return ForwardDiff.jacobian(θ̂, [Float64(θt[n]) for n in F]) - I
end

# -------------------------------------------------------------------------------------------------
# The two covariances
# -------------------------------------------------------------------------------------------------

"""
    sandwich(H, J, Σ) -> Matrix

The covariance of a least-squares fit, first order in the noise: H⁻¹(JᵀΣJ)H⁻¹, with J the
Jacobian at the fit's answer, H the Hessian of its cost ½‖g − y‖² there ([`hessian`](@ref)) and
Σ the noise covariance ([`noisecovariance`](@ref)). Noise ε moves the fit by δ = −H⁻¹Jᵀε
(expand the fit's zero-gradient condition), so Cov(δ) is the "meat" JᵀΣJ (how hard the noise
pushes) between two slices of "bread" H⁻¹ (how stiff the cost is). Huber–White (White,
Econometrica 1982;50:1); the misspecified CRB (Fortunati et al., IEEE Signal Process Mag
2017;34(6):142). With H = JᵀJ (the fit exact) and Σ = σ²I: the classical σ²(JᵀJ)⁻¹.
"""
function sandwich(H::AbstractMatrix, J::AbstractMatrix, Σ)
    A = H \ J'                          # P×N: column i, how noise in observation i moves the fit
    return _sym(A * Σ * A')
end

"""
    biasedcrlb(∇b, F) -> Matrix

The covariance bound of Dortch et al., MRM 2018, Eq. 3, for any estimator with the bias
gradient ∇b ([`biasgradient`](@ref)): (I + ∇b)F⁻¹(I + ∇b)ᵀ, with F the Fisher information at
the truth ([`informationmatrix`](@ref)). With ∇b = 0, the classical F⁻¹. Computed as XᵀX,
X = L⁻¹(I + ∇b)ᵀ with F = LLᵀ: a sum of squares, so never a negative variance.
"""
function biasedcrlb(∇b::AbstractMatrix, F::AbstractMatrix)
    X = cholesky(Symmetric(F)).L \ (I + ∇b)'
    return X' * X
end

# Can the data tell the parameters apart? G = JᵀJ (or JᵀΣ⁻¹J) over the parameters, from N
# observations. The Cholesky pivots of G with unit diagonal (each column of J normalized) are
# the squared sines of the angles between each column and the span of the ones before it. A
# pivot below `_PIVOT_MIN` (an angle below 1e-6 rad: the condition number of J is then at least
# ~1e6, a relative SD of ~1e6 × noise/signal) means a parameter is a combination of the others,
# to within what rounding in JᵀJ can resolve. A parameter with no influence at all (a zero
# column) is undetermined too, and so are more parameters than observations. Why: a singular H
# can look positive definite after rounding, and the covariance then has finite, meaningless
# values.
function _identifiable(G::AbstractMatrix, N::Integer)
    isempty(G) && return true                # nothing to determine
    N >= size(G, 1) || return false
    g = diag(G)
    all(>(0), g) || return false
    s = 1 ./ sqrt.(g)
    C = cholesky(Symmetric(s .* G .* s'); check = false)
    return issuccess(C) && all(>=(_PIVOT_MIN), diag(C.U) .^ 2)
end
const _PIVOT_MIN = 1e-12

# The condition number of J with each column scaled by its parameter's true value (relative
# sensitivities, so the units drop out; 1 for a true value of 0), reported as `c.cond`. Inf
# when there are fewer observations than parameters: `cond` of a wide matrix would ignore the
# missing directions.
_condition(J, θ) = size(J, 1) < size(J, 2) ? Inf :
                   cond(J * Diagonal([x == 0 ? 1.0 : abs(x) for x in θ]))

# -------------------------------------------------------------------------------------------------
# Checks shared by crlb, montecarlo and the functions above
# -------------------------------------------------------------------------------------------------

_hasmap(x::AbstractArray) = true
_hasmap(x::Free) = x.start isa AbstractArray
_hasmap(x::Fixed) = x.value isa AbstractArray
_hasmap(x) = false

function _check_fit_settings(fit::NamedTuple)
    for (k, x) in pairs(fit)
        k in _OPTIONS && throw(ArgumentError("`fit = (; $k)`: the noise-free fit sets its own " *
                                             "options; give only parameter settings"))
        _hasmap(x) &&
            throw(ArgumentError("`fit = (; $k)`: crlb is for one voxel; give numbers, not maps"))
    end
end

# Where the data come from: the fitted model at θ, or another model (`simulate_with = m2 =>
# θ2`), with as many observations; every value given, no maps.
function _datasource(m::MRIModel, θ::NamedTuple, simulate_with)
    _check_simulate_with(simulate_with)
    dm, dθ = simulate_with === nothing ? (m, θ) : (simulate_with.first, simulate_with.second)
    (any(_hasmap, values(θ)) || any(_hasmap, values(dθ))) &&
        throw(ArgumentError("crlb is for one voxel: give numbers, not maps"))
    resolve(m, θ)                            # checks the values: every parameter, no roles
    resolve(dm, dθ)                          # and the data's
    nobs(dm) == nobs(m) || throw(DimensionMismatch("the data's model has $(nobs(dm)) " *
        "observations, the fitted model $(nobs(m))"))
    return dm, dθ
end
_check_simulate_with(::Nothing) = nothing
_check_simulate_with(s) = s isa Pair{<:MRIModel,<:NamedTuple} || throw(ArgumentError(
    "`simulate_with = model => truth`: a model and every value of its parameters, e.g. " *
    "`T2BiexpDecay(TE = te) => (S0 = 1.0, fshort = 0.2, T2short = 0.02, T2long = 0.08)`"))
