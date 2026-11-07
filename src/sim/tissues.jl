# =================================================================================================
# Tissue presets (published values), and `nominal`: every parameter of a model with a value, from
# a tissue or from the model itself
# =================================================================================================

"""
    TISSUES

Published tissue values, by name (times in s, rates in 1/s). [`tissue`](@ref)`(name)` gives one;
[`nominal`](@ref)`(m, name)` turns it into a model's parameters.

T1 and T2 at 3 T, 37 °C, measured in vitro (Stanisz et al., Magn Reson Med 2005;54:507, Table 1;
mouse liver, muscle and heart, rat kidney and spinal cord, bovine cartilage, optic nerve, white
matter (corpus callosum) and gray matter (cortex), human blood): `liver`, `muscle`, `heart`,
`kidney`, `cartilage_0`, `cartilage_55` (0° and 55° to the field), `WM`, `GM`, `optic_nerve`,
`spinal_cord`, `blood`.

`CSF` (in vivo, 3 T): T1 = 4.124 s, the mean of the three inversion recovery studies in the review
of Bojorquez et al., Magn Reson Imaging 2017;35:69, Table 1c (4391, 3817 and 4163 ms); T2 =
1.75 s, the middle of the 1.5–2 s usually reported at 3 T, a placeholder until a measured
reference is chosen.

Use your own values with a named tuple, `nominal(m, (T1 = 4.3, T2 = 2.0))`, or change one value
of a preset: `nominal(m, merge(tissue(:CSF), (T2 = 2.0,)))`.

SIR tissue classes at 3 T (Dortch et al., Magn Reson Med 2018;80:1824, Table 1), with the values
shared by all classes there (Sm = 0.83, Sf = −1.0; R1m = R1f, QMTSIR's default; M0f = 1 and
SNR = 200 at thermal equilibrium): `SIR_WM_I`, `SIR_WM_II` (maximum and mean white matter),
`SIR_GM`, `SIR_lesion` (RRMS lesion).

SMT values in multiple sclerosis (Bagnato et al., Ann Clin Transl Neurol 2019, as summarized by
Devan et al., Magn Reson Imaging 2020;74:56, Table 1; `vax` as a fraction, `Dax` in mm²/s):
`SMT_NWM` (normal white matter, internal capsule), `SMT_NAWM` (normal-appearing white matter in
MS), `SMT_T2lesion` (T2-hyperintense lesion), `SMT_cBH` (chronic black hole, T1-hypointense).

Diffusion and kurtosis values in healthy adults at 3 T (Lätt et al., J Magn Reson Imaging
2013;37:610, Table 1: 36 volunteers, 19–64 years; 15 directions, b = 0, 500, 1000, 2500 and
2750 s/mm², the kurtosis tensor fitted by nonlinear least squares; the regional means): MD
(mm²/s), FA, MK and RK, by region: `DKI_CC_splenium`, `DKI_CC_genu`, `DKI_CC_body` (corpus
callosum), `DKI_PLIC` (internal capsule, posterior limb), `DKI_centrum_semiovale`,
`DKI_corona_radiata`, `DKI_frontal_WM` (frontal subcortical white matter), `DKI_thalamus`,
`DKI_putamen`, `DKI_caudate` (its head). Derived quantities, not parameters: a model takes them
through its [`fromderived`](@ref), with the direction of the fibers given in the call, DTI an
axially symmetric tensor from MD and FA (its RD within 4% of the table's), DKI also W from MK and
RK, and AK, which the paper did not report (biased by the noise floor, its authors said):
`nominal(DKI, :DKI_CC_splenium; S0 = 1000.0, direction = (1, 0, 0), AK = 0.8)`.
"""
const TISSUES = (
    # Stanisz et al. 2005, Table 1, 3 T: T1 and T2 (s)
    liver = (T1 = 0.812, T2 = 0.042),
    muscle = (T1 = 1.412, T2 = 0.050),          # skeletal muscle
    heart = (T1 = 1.471, T2 = 0.047),
    kidney = (T1 = 1.194, T2 = 0.056),
    cartilage_0 = (T1 = 1.168, T2 = 0.027),     # collagen fibers at 0° to the field
    cartilage_55 = (T1 = 1.156, T2 = 0.043),    # at 55° (the magic angle)
    WM = (T1 = 1.084, T2 = 0.069),
    GM = (T1 = 1.820, T2 = 0.099),
    optic_nerve = (T1 = 1.083, T2 = 0.078),
    spinal_cord = (T1 = 0.993, T2 = 0.078),
    blood = (T1 = 1.932, T2 = 0.275),
    # Bojorquez et al. 2017, Table 1c: T1, the mean of the three inversion recovery studies (in
    # vivo, 3 T). T2: the middle of the usual 1.5-2 s, a placeholder (no reference chosen yet).
    CSF = (T1 = 4.124, T2 = 1.75),
    # Dortch et al. 2018, Table 1: PSR (fraction), kmf and R1f (1/s), and the shared Sf, Sm
    SIR_WM_I = (PSR = 0.20, kmf = 5.0, R1f = 1.5, Sf = -1.0, Sm = 0.83),
    SIR_WM_II = (PSR = 0.12, kmf = 10.0, R1f = 1.0, Sf = -1.0, Sm = 0.83),
    SIR_GM = (PSR = 0.08, kmf = 15.0, R1f = 0.7, Sf = -1.0, Sm = 0.83),
    SIR_lesion = (PSR = 0.05, kmf = 20.0, R1f = 0.5, Sf = -1.0, Sm = 0.83),
    # Devan et al. 2020, Table 1 (from Bagnato et al. 2019): v_ax (fraction), D_ax (μm²/ms → mm²/s)
    SMT_NWM = (vax = 0.607, Dax = 1.9e-3),
    SMT_NAWM = (vax = 0.450, Dax = 1.8e-3),
    SMT_T2lesion = (vax = 0.365, Dax = 1.7e-3),
    SMT_cBH = (vax = 0.241, Dax = 1.7e-3),
    # Lätt et al. 2013, Table 1 (36 healthy adults, 3 T): MD (μm²/ms → mm²/s) and FA, MK and RK.
    # Derived quantities, for models that take them (`fromderived`: DTI's MD and FA; DKI's MK and
    # RK too, with AK given in the call, since the paper did not report it); a direction too.
    DKI_CC_splenium = (MD = 0.89e-3, FA = 0.83, MK = 1.32, RK = 2.72),
    DKI_CC_genu = (MD = 0.93e-3, FA = 0.80, MK = 1.06, RK = 2.07),
    DKI_CC_body = (MD = 0.92e-3, FA = 0.78, MK = 1.17, RK = 2.54),
    DKI_PLIC = (MD = 0.89e-3, FA = 0.71, MK = 1.23, RK = 2.04),   # internal capsule, posterior limb
    DKI_centrum_semiovale = (MD = 0.80e-3, FA = 0.63, MK = 1.09, RK = 1.72),
    DKI_corona_radiata = (MD = 0.84e-3, FA = 0.53, MK = 1.09, RK = 1.49),
    DKI_frontal_WM = (MD = 0.91e-3, FA = 0.48, MK = 0.94, RK = 1.23),   # frontal subcortical WM
    DKI_thalamus = (MD = 0.87e-3, FA = 0.32, MK = 0.86, RK = 0.92),
    DKI_putamen = (MD = 0.79e-3, FA = 0.15, MK = 0.67, RK = 0.61),
    DKI_caudate = (MD = 0.87e-3, FA = 0.14, MK = 0.61, RK = 0.59),    # caudate head
)

"""
    tissue(name) -> named tuple

One tissue's published values from [`TISSUES`](@ref), e.g. `tissue(:WM)` gives
`(T1 = 1.084, T2 = 0.069)`.
"""
function tissue(name::Symbol)
    haskey(TISSUES, name) ||
        throw(ArgumentError("unknown tissue `$name`; available: $(join(keys(TISSUES), ", "))"))
    return TISSUES[name]
end

"""
    nominal(m; values...) -> named tuple
    nominal(m, name::Symbol; values...) -> named tuple
    nominal(m, tissue::NamedTuple; values...) -> named tuple

`m` is a model, or a model's type when no acquisition is at hand: `nominal(QMTSIR, :SIR_WM_I)`.

Every parameter of model `m` with a value: the ground truth for [`signal`](@ref),
[`simulate`](@ref), from the model, the literature or your own values. For
each parameter, the value comes from, in order:

1. the keywords: `nominal(m; PSR = 0.15)`, `nominal(m, :WM; M0 = 1000.0, Sf = -0.95)`;
2. the tissue, if given: a name from [`TISSUES`](@ref) (`:WM`) or your own named tuple
   (`(T1 = 4.3, T2 = 2.0)`), by name or converted between time and rate (R1 = 1/T1,
   R2 = 1/T2);
3. the model, [`parameters`](@ref)`(m)`: a fixed parameter's value ([`defaults`](@ref)`(m)`,
   e.g. QMTSIR's `kmf = 12.5`), 1 for the scale parameter (`M0`, `S0`, `M0f`), and, without a
   tissue, a free parameter's start.

A model with derived quantities can take them in place of its parameters (its
[`fromderived`](@ref)): `nominal(DTI; S0 = 1000, MD = 0.8e-3, FA = 0.7, direction = (1, 0, 0))`
makes the six elements of an axially symmetric tensor (DTI's and DKI's docstrings say what each
takes). A tissue can give them too, the names the model takes (`fromderived(M)`), a keyword
winning: `nominal(DTI, :DKI_CC_splenium; direction = (1, 0, 0))` (the preset's MD and FA).

Every value is in the result, the model's included, so a script shows everything its data were
made with. Ties are kept as written (`R1m = :R1f`), so they follow later changes;
[`resolve`](@ref) turns them into numbers.

Without a tissue, the free parameters take the model's starts: guesses for fitting rather than
tissue values, but a complete set for any model, custom ones included. With a tissue, a free
parameter must come from the tissue or a keyword, never from a start: that would hide a tissue
that does not fit the model (`:WM` has T1 and T2, no PSR for QMTSIR), and values that are not
properties of the tissue (an inversion efficiency `Sf`) are your choice. A free parameter with
no value (without a tissue: one that starts from the data, `:data`, or from [`BestOf`](@ref)
values) is an error that names it and lists what `nominal` takes for every parameter. Typing the
model at the REPL shows the same table. With a type, a start computed from the acquisition
(e.g. ADCDecay's `1/mean(b)`) is not known, so without a tissue that parameter must be given.

```julia
m = QMTSIR(; SIR_4POINT...)
nominal(m)                          # (PSR = 0.1, R1f = 1.0, Sf = -0.95, M0f = 1.0, kmf = 12.5, ...)
nominal(m; PSR = 0.15, kmf = 10.0)  # a couple changed
nominal(m, :SIR_lesion)             # PSR, R1f, Sf, kmf = 20, Sm = 0.83 (the tissue), M0f = 1,
                                  # R1m = :R1f, Mzf_td0 = 0, Mzm_td0 = 0 (the model)
nominal(R1IR(TI = ti), :GM; Sf = -0.95)            # (M0 = 1.0, Sf = -0.95, R1 = 0.549...)
nominal(T2Decay(TE = 0.01:0.01:0.32); T2 = 0.08)   # T2 starts from the data, so give it
```
"""
# Three methods, one per kind of second argument; Julia picks by its type ("dispatch"): none
# (the model's own values, starts included), a preset's name, or your own values. Each passes
# the shared body what it knows: the tissue's values, the call as the user wrote it (for error
# messages), and whether the free parameters may take their starts (only without a tissue).
# Another source of values is another method, e.g. one taking a fitted voxel of a FitResult.
nominal(m::_ModelOrType; changes...) = _nominal(m, (;), _call(m), true; changes...)
nominal(m::_ModelOrType, name::Symbol; changes...) =
    _nominal(m, tissue(name), _call(m) * ", $(repr(name))", false; changes...)
nominal(m::_ModelOrType, values::NamedTuple; changes...) =
    _nominal(m, values, _call(m) * ", tissue", false; changes...)
# The start of the call as the user wrote it, for error messages: `nominal(m` or `nominal(QMTSIR`.
_call(m::MRIModel) = "nominal(m"
_call(M::Type{<:MRIModel}) = "nominal($(nameof(M))"

function _nominal(m::_ModelOrType, props::NamedTuple, call, starts::Bool; changes...)
    roles = _roles(m)                            # a model's starts computed for its acquisition
    ch = _fromderived(m, values(changes), keys(roles), props)   # derived quantities → parameters
    _check_names("nominal(m; ...)", keys(ch), keys(roles))
    _reject_roles(ch)
    function pick(n)                             # the keyword's value, the tissue's, the model's
        haskey(ch, n) && return ch[n]
        haskey(props, n) && return props[n]
        x = _converted(props, n)
        return x === nothing ? _model_value(roles[n], starts) : x
    end
    q = NamedTuple{keys(roles)}(map(pick, keys(roles)))
    missing = Tuple(n for n in keys(q) if q[n] === nothing)
    isempty(missing) || _nominal_error(m, roles, q, call, starts, missing)
    _check_ties(q)                       # a tie given as a keyword must name a real parameter
    return q
end

# The keywords that are not parameters go to the model's `fromderived` (derived quantities, e.g.
# DTI's MD and FA, and what else it takes, such as a direction), with the tissue's values of the
# names it takes (`fromderived(M)`; a keyword wins over the tissue's value); the parameters it
# makes join the keywords that are parameters. A parameter both given and made is refused. A
# model without the hook (`nothing`): the keywords as given, so that `_check_names` names the
# unknown ones. Also how a scenario's truth, `simulate`'s and crlb's data truth may be given by
# derived quantities (no tissue there).
function _fromderived(m::_ModelOrType, ch::NamedTuple{K}, names, props::NamedTuple = (;)) where {K}
    # The usual case, parameters only and no tissue: K and the names are known when compiling,
    # so this is one check, folded (crlb! takes this path for every truth).
    isempty(props) && all(n -> n in names, K) && return ch
    M = m isa Type ? m : typeof(m)
    extra = Tuple(n for n in keys(ch) if !(n in names))
    fromtissue = Tuple(n for n in keys(props) if n in fromderived(M) && !(n in names) && !haskey(ch, n))
    isempty(extra) && isempty(fromtissue) && return ch       # the usual case: parameters only
    q = merge(NamedTuple{fromtissue}(map(n -> props[n], fromtissue)),
              NamedTuple{extra}(map(n -> ch[n], extra)))
    made = fromderived(M, q)
    made === nothing && return ch
    both = Tuple(n for n in keys(made) if haskey(ch, n))
    isempty(both) || throw(ArgumentError("$(_modelname(m)): $(join(("`$n`" for n in both), ", ")) " *
        "given, and also made from $(join(("`$n`" for n in keys(q)), ", ")): give one or the other"))
    given = Tuple(n for n in keys(ch) if n in names)
    return merge(NamedTuple{given}(map(n -> ch[n], given)), made)
end

# A free parameter without a value: say why, how to give it, and what `nominal` takes for every
# parameter (the table the model's REPL display shows).
function _nominal_error(m, roles, q, call, starts, missing)
    start(n) = roles[n] isa Free ? roles[n].start : nothing
    why(n) = start(n) === :data ? "it starts from the data" :
             start(n) === :loglinear ? "it starts from the log-linear fit" :
             start(n) isa BestOf ? "it starts from BestOf values" :
             start(n) isa Function ? "its start is computed from the acquisition" :
             "it has no start"
    one = length(missing) == 1
    list = starts ? join(("`$n` ($(why(n)))" for n in missing), ", ") :
        join(("`$n`" for n in missing), ", ") *
        (one ? ", a free parameter" : ", free parameters") * " the tissue does not give (a " *
        "start is a guess for fitting, not a tissue value)"
    table = sprint(_print_parameters, roles, q, call * ")")
    give = join(("$n = value" for n in missing), ", ")
    throw(ArgumentError("$(_modelname(m)): no value for $list; give $(one ? "it" : "them"): " *
        "`$call; $give)`. What `$call)` takes for each parameter:\n" * table))
end

# Time ↔ rate conversions between a tissue's values and a model's parameter names.
function _converted(props, n)
    n === :R1 && haskey(props, :T1) && return 1 / props.T1
    n === :T1 && haskey(props, :R1) && return 1 / props.R1
    n === :R2 && haskey(props, :T2) && return 1 / props.T2
    n === :T2 && haskey(props, :R2) && return 1 / props.R2
    return nothing
end
