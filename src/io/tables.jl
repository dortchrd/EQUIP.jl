# =================================================================================================
# Tables out: rows (crlb's, montecarlo's, a sweep's: named tuples with the same names) as a TSV
# file, as BIDS keeps tables; it opens in a spreadsheet, MATLAB (`readtable`), Python (pandas)
# or R.
# =================================================================================================

"""
    writetable(path, rows) -> String

Rows (named tuples with the same names: a [`Sweep`](@ref)'s `rows`, ...) as a TSV file: a
header, then one line per row; numbers in full precision, names as text. The directory is made
if needed. Returns the file written.
"""
function writetable(path::AbstractString, rows)
    isempty(rows) && throw(ArgumentError("no rows to write"))
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    cell(x) = x isa AbstractFloat ? repr(Float64(x)) : string(x)
    open(path, "w") do io
        println(io, join(string.(keys(rows[1])), '\t'))
        for r in rows
            println(io, join((cell(x) for x in values(r)), '\t'))
        end
    end
    return String(path)
end
