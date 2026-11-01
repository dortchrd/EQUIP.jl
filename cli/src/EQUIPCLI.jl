# =================================================================================================
# EQUIPCLI: EQUIP from the command line, and its data files
#
# A companion package, so that EQUIP itself depends on no file format. The `equip` command, a
# subcommand per part of EQUIP (docs/design.md, "The command line and the data reader: the
# plan"):
#   io.jl       data in (NIfTI with BIDS sidecars; text files of values), maps out
#   args.jl     the command line's words: flags, config files of flags, values, roles
#   models.jl   the models offered, built from the flags and the files; --help
#   fit.jl      equip fit
# Run as `julia -t auto --project=cli -m EQUIPCLI fit ...` (cli/bin/equip does that), or from
# Julia: `EQUIPCLI.main(["fit", "T2Decay", "t2.nii.gz"])`.
# =================================================================================================

module EQUIPCLI

using EQUIP
import JSON
import Dates
using NIfTI: niread, niwrite, NIVolume
using PrecompileTools: @setup_workload, @compile_workload

export loaddata, loadmap, readvalues, savemaps

include("io.jl")
include("args.jl")
include("models.jl")
include("fit.jl")

const USAGE = """
equip: EQUIP from the command line

  equip fit MODEL DATA... [flags]       fit maps: NIfTI data in, NIfTI maps out
  equip fit MODEL --help                 the model's acquisition fields and parameters
  equip models                           the models

  crlb, montecarlo and design: next.

Flags are named after the model's fields (--TI, --TE, --b) and parameters (--kmf 12.5,
--R1f free); --config FILE reads flags from a file (one per line, # comments), --save FILE
writes the command's flags to one. Times in seconds.
"""

"""
    main(args; io = stdout, err = stderr) -> exit code

The `equip` command: `main(["fit", "T2Decay", "t2.nii.gz"])`. Mistakes in the command (an
unknown flag, a missing value, data that do not match the model) print a message to `err` and
return 1; anything else is an error in EQUIP, shown in full.
"""
function main(args::AbstractVector{<:AbstractString}; io::IO = stdout, err::IO = stderr)
    # The commands write into buffers, copied to `io` and `err` at the end: then they are
    # compiled once, for IOBuffer, in the precompile workload, whatever `stdout` is (a terminal,
    # a file, a pipe: each a type of its own, and each would compile the command again).
    out, msg = IOBuffer(), IOBuffer()
    code = try
        _main(String.(args), out, msg)
    catch e
        e isa ArgumentError || e isa DimensionMismatch || (_flush(out, io, msg, err); rethrow())
        println(msg, "equip: ", e.msg)
        1
    end
    _flush(out, io, msg, err)
    return code
end
_flush(out, io, msg, err) = (write(io, take!(out)); write(err, take!(msg)); flush(io); flush(err))

function _main(args::Vector{String}, io::IOBuffer, err::IOBuffer)
    isempty(args) && (print(io, USAGE); return 0)
    cmd, rest = args[1], args[2:end]
    cmd in ("-h", "--help", "help") && (print(io, USAGE); return 0)
    cmd == "models" && (println(io, join(keys(MODELS), "\n")); return 0)
    cmd == "fit" && return cmd_fit(rest; io, err)
    cmd in ("crlb", "montecarlo", "design") &&
        throw(ArgumentError("`equip $cmd` is not built yet; `equip fit` is"))
    throw(ArgumentError("unknown command `$cmd`; `equip --help` lists them"))
end

# The entry point for `julia -m EQUIPCLI ...`: its return value is the process's exit code.
(@main)(args) = main(args)

include("precompile.jl")            # last: it runs `main`

end
