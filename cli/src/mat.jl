# =================================================================================================
# MATLAB files: MAT.jl loaded only when a command names a .mat file (`data.mat:S`,
# `--TE @protocol.mat:TE`, `--mask mask.mat:M`), so other commands do not pay its 0.4 s of
# loading. Loading MAT loads MRImages' extension for it (MRImagesMATExt), and MRImages calls the
# extension through `invokelatest`, so it works although it is loaded while the command runs.
# =================================================================================================

const _MAT = Base.PkgId(Base.UUID("23992714-dd62-5051-b70f-ba57cb901cac"), "MAT")

# A word that names a MATLAB file: a path ending in .mat, or .mat:NAME, perhaps after `@`.
_namesmat(w::AbstractString) = occursin(r"\.mat(:[A-Za-z][A-Za-z0-9_]*)?$"i, w)

function _loadmat(words)
    any(_namesmat, words) || return nothing
    MRImages._matloaded() && return nothing
    try
        Base.require(_MAT)
    catch e
        throw(ArgumentError("MATLAB files need MAT.jl, which could not be loaded " *
                            "($(sprint(showerror, e)))"))
    end
    return nothing
end
