# =================================================================================================
# SMT: the spherical mean technique, two compartments (Kaden et al., NeuroImage 2016;139:346), in
# the form used for multiple sclerosis imaging by Devan et al., Magn Reson Imaging 2020;74:56
# (its protocols and tissue values are below)
#
# Averaging the diffusion-weighted signal over the gradient directions of a shell (its spherical
# mean) removes the fibre orientation distribution: crossing, dispersion and undulation drop out,
# and what is left depends on the diffusion within one fibre and around it. Per shell b
# (Devan 2020, section 2.1):
#
#   S̄(b)  = v_ax·S̄_ax + (1 − v_ax)·S̄_ex
#   S̄_ax  = S0·√π·erf(√(b·D_ax)) / (2√(b·D_ax))                     intra-axonal: sticks (D⊥,in = 0)
#   S̄_ex  = S0·exp(−b·D⊥,ex)·√π·erf(√(b·(D_ax − D⊥,ex))) / (2√(b·(D_ax − D⊥,ex)))
#                                                                    extra-axonal: a zeppelin
#   D⊥,ex = (1 − v_ax)·D_ax                                          the tortuosity limit
#
# The intra- and extra-axonal parallel diffusivities are the same, D_ax. So the parameters are
# S0, v_ax (the apparent axonal volume fraction) and D_ax. D_ax − D⊥,ex = v_ax·D_ax, which is how
# the code computes it (never negative, for v_ax and D_ax in their bounds).
#
# The spherical means come from the data first: `sphericalmean` averages the volumes of each
# shell (b-values within 100 s/mm² of each other are one shell, as in Devan 2020), and the model
# is fitted to one value per shell. A shell's mean over N directions has noise σ/√N: give
# `averages = directions` to crlb and montecarlo. (A mean of magnitude images keeps the Rician
# floor: in montecarlo, `averaging = :magnitude`.)
# =================================================================================================

"""
    SMT(; b)

The spherical mean technique, two compartments (Kaden et al., NeuroImage 2016), as used for
multiple sclerosis imaging by Devan et al. (Magn Reson Imaging 2020;74:56): the spherical mean
of the diffusion signal in a shell, from sticks (intra-axonal, no radial diffusion) and a
zeppelin around them (extra-axonal), with the same parallel diffusivity D_ax and the
extra-axonal radial diffusivity from the tortuosity limit, D⊥,ex = (1 − v_ax)·D_ax:

    S̄(b) = S0·[v_ax·f(b·D_ax) + (1 − v_ax)·exp(−b·D⊥,ex)·f(b·v_ax·D_ax)],
    f(x²) = √π·erf(x)/(2x)

The fibres' orientations drop out of the spherical mean (crossing, dispersion, undulation).

- acquisition: `b` (s/mm²), one per shell: the data are the shells' spherical means
  ([`sphericalmean`](@ref)); a shell at b = 0 gives S0 directly
- free by default: `S0`, `vax` (the apparent axonal volume fraction, 0 to 1), `Dax` (mm²/s,
  0 to 3e-3, about free water's diffusivity at body temperature)

Protocols: [`SMT_PROTOCOLS`](@ref) (Devan 2020, Table 2); tissue values: `SMT_NWM`, `SMT_NAWM`,
`SMT_T2lesion`, `SMT_cBH` in [`TISSUES`](@ref) (Table 1).

```julia
means, b, directions = sphericalmean(dwi, bvals)            # one value per shell
res = fit(SMT(b = b), means; mask)
crlb(SMT(b = b), nominal(SMT(b = b), :SMT_cBH); snr = 20, averages = directions)
```
"""
@kwdef struct SMT{A<:AbstractVector} <: MRIModel
    b::A                                # the shells' b-values (s/mm²)
end

parameters(::Type{<:SMT}) = (
    S0 = Free(start = :scale, lower = 0),
    vax = Free(start = 0.5, lower = 0, upper = 1),
    Dax = Free(start = 2.0e-3, lower = 0, upper = 3.0e-3),   # mm²/s
)

function signal!(out, m::SMT, p)
    (; b) = m
    (; S0, vax, Dax) = p
    Dperp = (1 - vax) * Dax                       # extra-axonal radial diffusivity (tortuosity)
    Dpar_perp = vax * Dax                         # D_ax − D⊥,ex
    @inbounds for j in eachindex(out, b)
        Sax = _sphmean(b[j] * Dax)                                    # intra-axonal (sticks)
        Sex = exp(-b[j] * Dperp) * _sphmean(b[j] * Dpar_perp)         # extra-axonal (zeppelin)
        out[j] = S0 * (vax * Sax + (1 - vax) * Sex)
    end
    return out
end

# √π·erf(x)/(2x), from x² = x2: the mean of exp(−x²·cos²ψ) over all directions (a stick's signal
# exp(−b·D·(g·n)²) averaged over gradient directions g, with x² = b·D); 1 at x = 0. Near 0, its
# Taylor series in x², Σ (−1)ⁿ x²ⁿ / (n!·(2n + 1)), 6 terms (error below 2e-12 for x² < 0.05):
# no 0/0 at b = 0, and no square root, whose derivative is infinite at 0 (the fit's dual numbers
# would carry it). The coefficients are integers, and π enters converted to x2's type, so a
# Float32 fit stays in Float32.
@inline function _sphmean(x2)
    if abs(x2) < 0.05                   # (comparing a dual number compares its value)
        t = x2
        s = 1 - t / 3
        t *= x2; s += t / 10
        t *= x2; s -= t / 42
        t *= x2; s += t / 216
        t *= x2; s -= t / 1320
        return s
    end
    x = sqrt(x2)
    return erf(x) / x * (sqrt(oftype(x, π)) / 2)
end

"""
    SMT_PROTOCOLS

The five SMT protocols compared by Devan et al., Magn Reson Imaging 2020;74:56, Table 2, each
about 100 acquisitions: `b` (s/mm², one per shell) and `directions` (gradient directions in
each shell). `A` and `B` were optimized there (`A`, 10 shells of 10 directions, shells within
100 s/mm² merged; `B`, 9 with one fixed at b = 0); `C` is NODDI's optimized protocol (Zhang et
al., NeuroImage 2012), `D` the protocol of their earlier MS studies, `E` the original SMT
protocol of Kaden et al. 2016, shortened to the same scan time.

```julia
P = SMT_PROTOCOLS.A
m = SMT(b = P.b)
crlb(m, nominal(m, :SMT_cBH); snr = 20, averages = P.directions)   # SNR of one direction
```
"""
const SMT_PROTOCOLS = (
    A = (b = [430.0, 990.0, 4230.0], directions = [10, 10, 80]),
    B = (b = [0.0, 530.0, 3790.0], directions = [10, 10, 80]),
    C = (b = [0.0, 711.0, 2855.0], directions = [10, 32, 64]),
    D = (b = [0.0, 1000.0, 2500.0], directions = [10, 45, 45]),
    E = (b = [0.0, 1000.0, 2000.0, 3000.0], directions = [4, 32, 32, 32]),
)

"""
    sphericalmean(data, b; tol = 100) -> (means, shells, directions)

The spherical mean of diffusion-weighted data: the volumes of each shell averaged, the input of
[`SMT`](@ref). `data` has the volumes along its last dimension (any number of dimensions before,
as for `fit`), `b` one b-value per volume (s/mm², e.g. from the `.bval` file: `info.found[:b]`
after [`loaddata`](@ref)). Volumes whose b-values are within `tol` of the lowest b-value of their
shell form one shell (Devan 2020: within 100 s/mm²); b = 0 volumes are a shell too.

Returns the means (the volumes replaced by one value per shell, in increasing b), each shell's b
(the mean of its volumes' b-values) and its number of volumes (`directions`: the `averages` for
crlb and montecarlo, since a shell's mean has noise σ/√N).
"""
function sphericalmean(data::AbstractArray{<:Real}, b::AbstractVector{<:Real}; tol::Real = 100)
    N = size(data, ndims(data))
    length(b) == N || throw(DimensionMismatch("sphericalmean: $(length(b)) b-values for $N " *
                                              "volumes (the last dimension of the data)"))
    order = sortperm(b)
    groups = Vector{Int}[]
    for k in order                                   # in increasing b: a new shell beyond tol
        if isempty(groups) || b[k] - b[first(groups[end])] > tol
            push!(groups, [k])
        else
            push!(groups[end], k)
        end
    end
    shells = [sum(b[g]) / length(g) for g in groups]
    directions = length.(groups)
    T = float(eltype(data))
    flat = reshape(data, :, N)
    means = Matrix{T}(undef, size(flat, 1), length(groups))
    for (s, g) in enumerate(groups)
        means[:, s] .= vec(sum(view(flat, :, g); dims = 2)) ./ length(g)
    end
    return reshape(means, size(data)[1:end-1]..., length(groups)), shells, directions
end
