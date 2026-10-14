# Minimal NumPy .npy reader/writer with memory mapping.
#
# .npy is the interchange format for the CLI: NumPy reads and writes it natively, MATLAB needs
# ~40 lines (see matlab/voxelfit.m), and it records shape, dtype and memory order. Inputs are
# memory-mapped so a large dataset is paged in on demand, and outputs can be created as
# memory-mapped files and filled in place.
#
# Order convention: a C-ordered (NumPy default) array of shape (a, b, c) is returned as a Julia
# array of size (c, b, a) without copying; `reversed = true` records that the axes are reversed.

const _NPY_MAGIC = UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')]

const _NPY_DTYPES = Dict{String,DataType}(
    "f4" => Float32, "f8" => Float64, "f2" => Float16,
    "i1" => Int8, "i2" => Int16, "i4" => Int32, "i8" => Int64,
    "u1" => UInt8, "u2" => UInt16, "u4" => UInt32, "u8" => UInt64, "b1" => Bool)

_npy_descr(::Type{T}) where {T} =
    (sizeof(T) == 1 ? "|" : "<") * only(k for (k, v) in _NPY_DTYPES if v === T)

struct NpyHeader
    eltype::DataType
    shape::Vector{Int}     # as NumPy reports it
    fortran_order::Bool
    offset::Int            # byte offset of the data
end

function read_npy_header(io::IO)
    read(io, 6) == _NPY_MAGIC || error("not a .npy file")
    major = read(io, UInt8)
    read(io, UInt8)
    hlen = major == 1 ? Int(ltoh(read(io, UInt16))) : Int(ltoh(read(io, UInt32)))
    hdr = String(read(io, hlen))
    m = match(r"'descr'\s*:\s*'([<>|=])(\w\d+)'", hdr)
    m === nothing && error("unsupported .npy dtype in header: $hdr")
    endian, code = m.captures
    haskey(_NPY_DTYPES, code) || error("unsupported .npy dtype '$code'")
    T = _NPY_DTYPES[code]
    (endian == ">" && sizeof(T) > 1) && error("big-endian .npy files are not supported")
    fo = match(r"'fortran_order'\s*:\s*(True|False)", hdr)
    fo === nothing && error("malformed .npy header: $hdr")
    sh = match(r"'shape'\s*:\s*\(([^)]*)\)", hdr)
    sh === nothing && error("malformed .npy header: $hdr")
    shape = [parse(Int, strip(s)) for s in split(sh.captures[1], ',') if !isempty(strip(s))]
    return NpyHeader(T, shape, fo.captures[1] == "True", position(io))
end

"""
    read_npy(path; mmap=true) -> (A, reversed::Bool)

Read a .npy file. C-ordered files come back with reversed axes (`reversed = true`) so that no
copy is needed; Fortran-ordered files (e.g. written from MATLAB) keep NumPy's axis order.
"""
function read_npy(path::AbstractString; mmap::Bool=true)
    open(path, "r") do io
        h = read_npy_header(io)
        dims = Tuple(h.fortran_order ? h.shape : reverse(h.shape))
        A = if mmap && prod(dims; init=1) > 0
            Mmap.mmap(io, Array{h.eltype,length(dims)}, dims, h.offset)
        else
            seek(io, h.offset)
            read!(io, Array{h.eltype}(undef, dims))
        end
        return A, !h.fortran_order && length(dims) > 1
    end
end

function _npy_header_bytes(::Type{T}, shape, fortran_order::Bool) where {T}
    shp = length(shape) == 1 ? "($(shape[1]),)" : "(" * join(shape, ", ") * ")"
    d = "{'descr': '$(_npy_descr(T))', 'fortran_order': $(fortran_order ? "True" : "False"), 'shape': $shp, }"
    total = 10 + length(d) + 1
    pad = (64 - total % 64) % 64
    hdr = d * " "^pad * "\n"
    io = IOBuffer()
    write(io, _NPY_MAGIC, UInt8(1), UInt8(0), htol(UInt16(length(hdr))), hdr)
    return take!(io)
end

"""
    create_npy(path, T, dims; reversed=false) -> Array{T} (memory-mapped)

Create a .npy file for a Julia array of size `dims` and return it memory-mapped for writing.
With `reversed = true` the file is C-ordered with NumPy shape `reverse(dims)`, matching arrays
read by [`read_npy`](@ref) from C-ordered files.
"""
function create_npy(path::AbstractString, ::Type{T}, dims::Dims; reversed::Bool=false) where {T}
    shape = reversed ? reverse(dims) : dims
    hdr = _npy_header_bytes(T, shape, !reversed)
    io = open(path, "w+")
    write(io, hdr)
    A = Mmap.mmap(io, Array{T,length(dims)}, dims, length(hdr))
    close(io)
    return A
end

"""
    write_npy(path, A; reversed=false)

Write `A` to a .npy file (Fortran order, or C order with reversed shape when `reversed`).
"""
function write_npy(path::AbstractString, A::AbstractArray{T}; reversed::Bool=false) where {T}
    open(path, "w") do io
        write(io, _npy_header_bytes(T, reversed ? reverse(size(A)) : size(A), !reversed))
        write(io, A isa Array ? A : collect(A))
    end
    return path
end
