# =================================================================================================
# The reusable cache: everything a voxel fit needs, prepared once
#
#   FitSpec (setup.jl)
#        │
#        ▼  init         FitSpec + model (in the working precision) + solver settings + work
#   FitCache             buffers, prepared ONCE per fit call (`init` is SciML's name: prepare
#                        once, then solve many times).
#
# Next: voxel.jl fits one voxel with it (`fit!`); dataset.jl fits every voxel of a dataset,
# giving each chunk of voxels (one task on a thread) its own copy of the buffers (`_copycache`).
# =================================================================================================

"""
    FitCache

Reusable state for fitting voxels with one model and one set of roles: the model (its
acquisition converted to the working precision), the [`FitSpec`](@ref), solver options and work
buffers. Make it with [`init`](@ref), use it with [`fit!`](@ref). Not thread-safe: every fit
overwrites the buffers, so use one cache per task or thread (`fit` on a dataset does this for
you).
"""
struct FitCache{M,S<:FitSpec,T,B,K,L}
    model::M            # the model, its acquisition vectors converted to precision T
    spec::S
    opts::SolverOptions{T}
    buf::B              # Vector{Dual{FitTag,T,P}}: the model output, one per observation
    y::Vector{T}        # the current voxel's data
    settings::K         # the keywords of `init`, passed on to every FitResult
    logpass::L          # nothing, or the `:loglinear` start's first pass, the same in every voxel
end

function Base.show(io::IO, c::FitCache{M,S,T}) where {M,S,T}
    print(io, "FitCache(", nameof(M), "; free ", freenames(c.spec), ", fixed ", keys(c.spec.fixed),
          ", ", length(c.y), " observations, ", T, ")")
end
# Adding a method to `Base.show` controls how the object prints in the REPL. `nameof(M)` is the
# model's bare type name, e.g. `QMTSIR`.

"""
    init(model; precision = Float64, solver = LM(), weights, params...) -> FitCache

Set up a reusable cache for fitting single voxels with [`fit!`](@ref).

- `params`: keywords named after the model's parameters, to fix one (`kmf = 15.0`, a map, or a
  tie `R1m = :R1f`) or estimate one with its start and bounds (`kmf = Free()`,
  `R1f = Free(start = 0.8)`, `Sf = Free(lower = -1.5)`); see [`setup`](@ref). Maps (arrays) are
  indexed by the `voxel` keyword of `fit!`.
- `precision`: `Float64` or `Float32`.
- `solver`: solver settings, [`LM`](@ref)`(maxiter = 50, ...)`.
- `weights`: one weight per observation, the same in every voxel (`weights = nex`, the numbers
  of averages; or 1/σᵢ²): the fit minimizes ½Σwᵢrᵢ². Left out, unweighted.
- `likelihood`: the noise model the fit assumes, `:gaussian` (least squares, the default) or
  `:rician` (magnitude data), with `σ` (the noise SD of each channel: a number or a map) and
  `method`: `:ml` (the full Rician model, maximum likelihood; the default), `:mean` (least
  squares on the Rician mean) or `:wmean` (the same, weighted by 1/Var). Each voxel is then fitted
  as without it first, then with the Rician model from that result (src/fit/rician.jl); `init`
  returns a [`RicianFitCache`](@ref), used as this cache is.

`init` is `CommonSolve.init`, the SciML convention, so it does not clash with other packages.
"""
function CommonSolve.init(model::MRIModel; precision::Type{<:AbstractFloat} = Float64,
                          solver::LM = LM(), weights = nothing, likelihood = :gaussian,
                          method = nothing, params...)
    likelihood in _LIKELIHOODS || throw(ArgumentError("`likelihood` is the noise model the fit " *
        "assumes: :gaussian (least squares, the default) or :rician (magnitude data, with `σ`), " *
        "got $(repr(likelihood))"))
    likelihood === :rician && return _rician_init(model, method, precision, solver, weights, params)
    method === nothing || throw(ArgumentError("`method` is for likelihood = :rician (:ml, " *
        ":mean or :wmean); least squares, the default, has none"))
    haskey(params, :σ) && !(:σ in keys(parameters(model))) && throw(ArgumentError("`σ`, the " *
        "noise SD, is for likelihood = :rician; to give it to the standard errors of a least-" *
        "squares fit, `stderror(res; σ)`"))
    T = precision
    n = nobs(model)                                # also checks the acquisition's lengths
    spec = setup(model, T; weights, params...)     # the parameter keywords, the weights
    opts = SolverOptions{T}(solver)
    # Kept for the results (`(; solver)` = `(solver = solver,)`); the weights only when given.
    settings = weights === nothing ? (; precision, solver, params...) :
                                     (; precision, solver, weights, params...)
    return _cache(_with_precision(model, T), spec, opts, n, settings)
end
# The roles as one named tuple, made by `parameters(m; changes...)`: `init(m, roles)`.
CommonSolve.init(model::MRIModel, roles::NamedTuple; kwargs...) = init(model; roles..., kwargs...)

# A separate function so that P (the number of free parameters) is available as a type
# parameter, to allocate the dual-number buffer with the right type.
_cache(model, spec::FitSpec{F,P,T}, opts, n, settings) where {F,P,T} =
    FitCache(model, spec, opts, Vector{Dual{FitTag,T,P}}(undef, n), Vector{T}(undef, n), settings,
             _logpass(model, spec, n))

# Same setup, fresh buffers: one copy per task (chunk of voxels) in a dataset fit. The log
# start's first pass is read only, so the copies share it.
_copycache(c::FitCache) = FitCache(c.model, c.spec, c.opts, similar(c.buf), similar(c.y), c.settings,
                                   c.logpass)
