# =================================================================================================
# Tissue presets: published values, and `truth` to turn them into one model's parameters
# =================================================================================================

"""
    TISSUES

Published tissue values, by name (times in s, rates in 1/s). [`tissue`](@ref)`(name)` gives one;
[`truth`](@ref)`(m, name)` turns it into a model's parameters.

T1 and T2 at 3 T, 37 °C, measured in vitro (Stanisz et al., Magn Reson Med 2005;54:507, Table 1;
mouse liver, muscle and heart, rat kidney and spinal cord, bovine cartilage, optic nerve, white
matter (corpus callosum) and gray matter (cortex), human blood): `liver`, `muscle`, `heart`,
`kidney`, `cartilage_0`, `cartilage_55` (0° and 55° to the field), `WM`, `GM`, `optic_nerve`,
`spinal_cord`, `blood`.

`CSF` (in vivo, 3 T): T1 = 4.124 s, the mean of the three inversion recovery studies in the review
of Bojorquez et al., Magn Reson Imaging 2017;35:69, Table 1c (4391, 3817 and 4163 ms); T2 =
1.75 s, the middle of the 1.5–2 s usually reported at 3 T, a placeholder until a measured
reference is chosen.

Use your own values with a named tuple, `truth(m, (T1 = 4.3, T2 = 2.0))`, or change one value
of a preset: `truth(m, merge(tissue(:CSF), (T2 = 2.0,)))`.

SIR tissue classes at 3 T (Dortch et al., Magn Reson Med 2018;80:1824, Table 1), with the values
shared by all classes there (Sm = 0.83, Sf = −1.0; R1m = R1f, QMTSIR's default; M0f = 1 and
SNR = 200 at thermal equilibrium): `SIR_WM_I`, `SIR_WM_II` (maximum and mean white matter),
`SIR_GM`, `SIR_lesion` (RRMS lesion).
"""
const TISSUES = (
    # Stanisz et al. 2005, Table 1, 3 T: T1 and T2 (s)
    liver = (T1 = 0.812, T2 = 0.042),
    muscle = (T1 = 1.412, T2 = 0.050),          # skeletal muscle
    heart = (T1 = 1.471, T2 = 0.047),
    kidney = (T1 = 1.194, T2 = 0.056),
    cartilage_0 = (T1 = 1.168, T2 = 0.027),     # collagen fibres at 0° to the field
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
    truth(m, tissue; values...) -> named tuple

The parameters of model `m` for a tissue: a name from [`TISSUES`](@ref), or your own named
tuple of values. Ready for [`signal`](@ref), [`simulate`](@ref) and [`phantom`](@ref).

For each parameter of `m`, the value comes from, in order:
1. the keywords, e.g. `truth(m, :WM; M0 = 1000.0, Sf = -0.95)`;
2. the tissue, by name, or converted between time and rate (R1 = 1/T1, R2 = 1/T2);
3. for the scale parameter (`M0`, `S0`, `M0f`): 1;
4. for a parameter the model fixes by default: that default ([`defaults`](@ref)`(m)`, e.g.
   QMTSIR's `kmf = 12.5`, `R1m = :R1f`), included in the result, so every value is visible.

A free parameter with no value is an error that names it: values that are not properties of the
tissue (an inversion efficiency `Sf`) are your choice, so give them. For a model no tissue
preset covers, [`nominal`](@ref)`(m; changes...)` gives every value from the model itself.

```julia
truth(R1IR(TI = ti), :GM; Sf = -0.95)      # (M0 = 1.0, Sf = -0.95, R1 = 0.549...)
truth(QMTSIR(; SIR_4POINT...), :SIR_lesion)  # PSR, R1f, Sf, M0f, kmf = 20, Sm = 0.83 (the tissue),
                                             # R1m = :R1f, Mzf_td0 = 0, Mzm_td0 = 0 (the defaults)
```
"""
function truth(m::MRIModel, t; kwargs...)
    props = t isa Symbol ? tissue(t) : t
    props isa NamedTuple || throw(ArgumentError("give a tissue name or a named tuple of values"))
    roles = parameters(m)
    for k in keys(kwargs)
        haskey(roles, k) || throw(ArgumentError("`$k` is not a parameter of $(nameof(typeof(m)))"))
    end
    scale = _scalename(m)
    vals = Pair{Symbol,Any}[]
    for (n, r) in pairs(roles)
        x = haskey(kwargs, n) ? kwargs[n] :
            haskey(props, n) ? props[n] :
            _converted(props, n)
        if x === nothing && n === scale
            x = 1.0
        end
        if x === nothing && r isa Fixed
            x = r.value                                # the model's default, shown in the result
        end
        if x === nothing
            throw(ArgumentError("no value for `$n` in this tissue; give it, e.g. truth(m, ...; $n = value)"))
        end
        push!(vals, n => x)
    end
    return (; vals...)
end

# Time ↔ rate conversions between a tissue's values and a model's parameter names.
function _converted(props, n)
    n === :R1 && haskey(props, :T1) && return 1 / props.T1
    n === :T1 && haskey(props, :R1) && return 1 / props.R1
    n === :R2 && haskey(props, :T2) && return 1 / props.T2
    n === :T2 && haskey(props, :R2) && return 1 / props.R2
    return nothing
end
