# =================================================================================================
# The qMT models' names: one set for every model, f derived, and other conventions' names as hints
#
# The field writes the same few quantities under many names (the owner, 2026-11-26: "different
# terminology ... henkelman R - directional exchange. And lots of names for PSR and MPF").
# EQUIP's MT models all use one set, QMTSIR's: the free pool f and the macromolecular pool m,
#
#   PSR = M0m/M0f          the pool size ratio (per free-water proton: as a molality)
#   kmf                    the exchange rate from m to f (1/s); k_xy is always from x to y
#   R1f, R1m, T2f, T2m     each pool's relaxation; M0f the free pool's equilibrium magnetization
#
# and one derived quantity, with its errors from crlb and montecarlo by the delta method, and as
# `nominal`'s keyword (`fromderived`):
#
#   f = M0m/(M0f + M0m) = PSR/(1 + PSR)    the macromolecular fraction of all protons (as a mole
#                                          fraction): linear in the macromolecular concentration
#                                          with no offset, so what a gel series is plotted against
#                                          (the owner, 2026-11-27: "I call it f and it helps to look
#                                          at things like variation with gel concentration where
#                                          we assume linear with f with zero offset"); Yarnykh's
#                                          MPF, others' BPF
#
# Not derived (the owner: "we really dont ever care [about] kfm or R"): kfm = kmf·PSR, the rate
# from f to m, and Henkelman's R (Magn Reson Med 1993; not directional: kfm = R·M0m, kmf = R·M0f,
# so with M0f = 1, R is kmf). A fit or `nominal` given another convention's name for one of these
# says which of EQUIP's it is (`_namehint`).
# =================================================================================================

const _MTMODELS = Union{QMTSIR,QMTSIREqualR1,QMTSPGRYarnykh}

# f, from the parameters (`derived`'s rules: no allocation, any number type)
derived(::_MTMODELS, p) = (f = p.PSR / (1 + p.PSR),)

# ... and back, for `nominal(m; f = 0.12)`: PSR = f/(1 − f)
fromderived(::Type{<:_MTMODELS}) = (:f,)
function fromderived(::Type{M}, q::NamedTuple) where {M<:_MTMODELS}
    for n in keys(q)                                 # the call's names that are not parameters
        n === :f || throw(ArgumentError("nominal($(nameof(M)); ...): `$n` is neither a parameter " *
            "($(join(keys(parameters(M)), ", "))) nor the derived quantity it takes (f)" * _namehint(M, n)))
    end
    0 <= q.f < 1 || throw(ArgumentError("nominal($(nameof(M)); ...): f = $(q.f) is a fraction, in [0, 1)"))
    return (PSR = q.f / (1 - q.f),)
end

# Other conventions' names, and what they are in EQUIP's: added to the message when a fit or
# `nominal` is given a name that is not a parameter
const _MT_NAMEHINTS = let
    size = "the pool size is `PSR` = M0m/M0f (with M0f = 1, the macromolecular pool's size)"
    frac = "the fraction M0m/(M0f + M0m) (MPF, BPF) is `f`, derived, f = PSR/(1 + PSR): fit `PSR` and read " *
           "`derived(r).f`; `nominal(m; f = ...)` takes it"
    fwd = "the exchange parameter is `kmf`, from m to f; kfm = kmf·PSR is the rate from f to m (often kf)"
    bwd = "the exchange parameter is `kmf`, the rate from the macromolecular pool to the free one"
    hen = "Henkelman's R is not directional: kmf = R·M0f and kfm = R·M0m, so with M0f = 1 the parameter `kmf` is R"
    Dict{Symbol,String}(:F => size, :M0B => size, :M0b => size, :M0r => size, :M0m => size,
                        :BPF => frac, :MPF => frac, :bpf => frac, :mpf => frac,
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
