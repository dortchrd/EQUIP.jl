# =================================================================================================
# MRImagesMATExt: MATLAB files for MRImages, through MAT.jl. Loaded by Julia when both MRImages
# and MAT are loaded (`using MRImages, MAT`), a package extension; it adds the methods that
# src/mat.jl's functions call.
# =================================================================================================

module MRImagesMATExt

using MRImages: MRImages
using MAT: matopen, matwrite

MRImages._matnames(file::AbstractString) = matopen(f -> String.(collect(keys(f))), file)
MRImages._matread(file::AbstractString, name::String) = matopen(f -> read(f, name), file)
MRImages._matwrite(file::String, vars::Dict{String,Any}) = (matwrite(file, vars); file)

end
