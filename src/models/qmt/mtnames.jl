# =================================================================================================
# The qMT models' names, and the other conventions' as derived quantities and hints
#
# The field writes the same few quantities under many names (the owner, 2026-11-26: "different
# terminology ... henkelman R - directional exchange. And lots of names for PSR and MPF").
# EQUIP's MT models all use one set, QMTSIR's: the free pool f and the macromolecular pool m,
#
#   PSR = M0m/M0f          the pool size ratio
#   kmf                    the exchange rate from m to f (1/s); k_xy is always from x to y
#   kfm = kmf·PSR          from f to m (mass balance: kfm·M0f = kmf·M0m)
#   R1f, R1m, T2f, T2m     each pool's relaxation; M0f the free pool's equilibrium magnetization
#
# and give the others as derived quantities, with their errors from crlb and montecarlo by the
# delta method, and as `nominal`'s keywords (`fromderived`):
#
#   MPF = M0m/(M0f + M0m) = PSR/(1 + PSR)     the fraction (also called BPF, or f)
#   kfm = kmf·PSR                             the exchange rate f → m (often kf)
#
# Henkelman's R (Henkelman et al., Magn Reson Med 1993) is not directional: kfm = R·M0m and
# kmf = R·M0f, so with M0f = 1 (as there), R is kmf. A fit or `nominal` given another
# convention's name says which of EQUIP's it is (`_namehint`).
# =================================================================================================

const _MTMODELS = Union{QMTSIR,QMTSIREqualR1,QMTSPGRYarnykh}

# MPF and kfm, from the parameters (`derived`'s rules: no allocation, any number type)
derived(::_MTMODELS, p) = (MPF = p.PSR / (1 + p.PSR), kfm = p.kmf * p.PSR)

# ... and back, for `nominal(m; MPF = 0.13)`: MPF gives PSR = MPF/(1 − MPF); kfm, with MPF, gives
# kmf = kfm/PSR (kfm alone fixes no parameter: it needs the pool size)
fromderived(::Type{<:_MTMODELS}) = (:MPF, :kfm)
function fromderived(::Type{M}, q::NamedTuple) where {M<:_MTMODELS}
    for n in keys(q)                                 # the call's names that are not parameters
        n in (:MPF, :kfm) || throw(ArgumentError("nominal($(nameof(M)); ...): `$n` is neither a parameter " *
            "($(join(keys(parameters(M)), ", "))) nor a derived quantity it takes (MPF, kfm)" * _namehint(M, n)))
    end
    haskey(q, :MPF) || throw(ArgumentError("nominal($(nameof(M)); ...): kfm needs the pool size too: give MPF " *
                                           "with it (or PSR and kmf, the parameters)"))
    0 <= q.MPF < 1 || throw(ArgumentError("nominal($(nameof(M)); ...): MPF = $(q.MPF) is a fraction, in [0, 1)"))
    PSR = q.MPF / (1 - q.MPF)
    return haskey(q, :kfm) ? (PSR = PSR, kmf = q.kfm / PSR) : (PSR = PSR,)
end

# Other conventions' names, and what they are in EQUIP's: added to the message when a fit or
# `nominal` is given a name that is not a parameter
const _MT_NAMEHINTS = let
    size = "the pool size is `PSR` = M0m/M0f (with M0f = 1, the macromolecular pool's size)"
    frac = "the fraction M0m/(M0f + M0m) (MPF, BPF, f) is derived, MPF = PSR/(1 + PSR): fit `PSR` and read " *
           "`derived(r).MPF`; `nominal(m; MPF = ...)` takes it"
    fwd = "the exchange parameter is `kmf`, from m to f; kfm = kmf·PSR, from f to m (often kf), is derived"
    bwd = "the exchange parameter is `kmf`, the rate from the macromolecular pool to the free one"
    hen = "Henkelman's R is not directional: kmf = R·M0f and kfm = R·M0m, so with M0f = 1 the parameter `kmf` is R"
    Dict{Symbol,String}(:F => size, :M0B => size, :M0b => size, :M0r => size, :M0m => size,
                        :f => frac, :BPF => frac, :MPF => frac, :bpf => frac, :mpf => frac,
                        :kf => fwd, :kfm => fwd, :kab => fwd, :k => fwd,
                        :kb => bwd, :kr => bwd, :kba => bwd,
                        :R => hen, :RM0B => hen,
                        :T2r => "the macromolecular pool's T2 is `T2m`", :T2B => "the macromolecular pool's T2 is `T2m`",
                        :T2b => "the macromolecular pool's T2 is `T2m`",
                        :T2A => "the free pool's T2 is `T2f`", :T2a => "the free pool's T2 is `T2f`", :T2F => "the free pool's T2 is `T2f`",
                        :R1r => "the macromolecular pool's R1 is `R1m`", :R1B => "the macromolecular pool's R1 is `R1m`",
                        :R1b => "the macromolecular pool's R1 is `R1m`",
                        :R1A => "the free pool's R1 is `R1f`", :R1a => "the free pool's R1 is `R1f`", :R1F => "the free pool's R1 is `R1f`",
                        :M0A => "the free pool's equilibrium magnetization is `M0f`", :M0a => "the free pool's equilibrium magnetization is `M0f`")
end
_namehint(::Union{_MTMODELS,Type{<:_MTMODELS}}, n::Symbol) =
    haskey(_MT_NAMEHINTS, n) ? " (EQUIP's names for qMT: $(_MT_NAMEHINTS[n]))" : ""
