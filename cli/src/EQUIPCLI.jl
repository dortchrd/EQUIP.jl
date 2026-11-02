# =================================================================================================
# EQUIPCLI: EQUIP from the command line, and its data files
#
# The `equip` command, a subcommand per part of EQUIP (docs/design.md, "The command line and the
# data reader: the plan"). The data are read and the maps written by EQUIP (`loaddata`,
# `savemaps`, on MRImages); here, the command line:
#   args.jl     the command line's words: flags, config files of flags, values, roles
#   mat.jl      MATLAB files: MAT.jl loaded when a command names one
#   models.jl   the models offered, built from the flags and the files; --help
#   fit.jl      equip fit
#   errors.jl   equip crlb, equip montecarlo
#   design.jl   equip design
# Run as `julia -t auto --project=cli -m EQUIPCLI fit ...` (cli/bin/equip does that), or from
# Julia: `EQUIPCLI.equip(["fit", "T2Decay", "t2.nii.gz"])`.
# =================================================================================================

module EQUIPCLI

using EQUIP
import MRImages
using MRImages: filestem, writesidecar
import Dates
using PrecompileTools: @setup_workload, @compile_workload
using Random: Xoshiro

include("args.jl")
include("mat.jl")
include("models.jl")
include("fit.jl")
include("errors.jl")
include("design.jl")

const USAGE = """
equip: EQUIP from the command line

  equip fit MODEL DATA... [flags]       fit maps: NIfTI (or MATLAB) data in, maps out
  equip crlb MODEL [flags]               a fit's bias, SD and RMSE, predicted (Cramér–Rao)
  equip montecarlo MODEL [flags]         the same, measured by simulating and fitting
  equip design MODEL [flags]             a protocol optimized for a set of scenarios
  equip COMMAND MODEL --help             the flags: the model's fields and parameters
  equip models                           the models

Flags are named after the model's fields (--TI, --TE, --b) and parameters (--kmf 12.5,
--R1f free); --config FILE reads flags from a file (one per line, # comments), --save FILE
writes the command's flags to one. Times in seconds. MATLAB files as file.mat:NAME (the
variable NAME), for data, maps and values (--TE @protocol.mat:TE).
"""

"""
    equip(args; io = stdout, err = stderr) -> exit code

The `equip` command, from Julia: `equip(["fit", "T2Decay", "t2.nii.gz"])`. Mistakes in the
command (an unknown flag, a missing value, data that do not match the model) print a message to
`err` and return 1; anything else is an error in EQUIP, shown in full.
"""
equip(args::AbstractVector{<:AbstractString}; io::IO = stdout, err::IO = stderr) =
    _run(String.(args), io, err)
# The commands write into buffers, copied to `io` and `err` at the end: then they are compiled
# once, for IOBuffer, in the precompile workload, whatever `stdout` is (a terminal, a file, a
# pipe: each a type of its own, and each would compile the command again); `@nospecialize` keeps
# this function itself compiled once for any of them.
function _run(args::Vector{String}, @nospecialize(io::IO), @nospecialize(err::IO))
    out, msg = IOBuffer(), IOBuffer()
    code = try
        _main(args, out, msg)
    catch e
        e isa ArgumentError || e isa DimensionMismatch || (_flush(out, io, msg, err); rethrow())
        println(msg, "equip: ", e.msg)
        1
    end
    _flush(out, io, msg, err)
    return code
end
function _flush(out::IOBuffer, @nospecialize(io::IO), msg::IOBuffer, @nospecialize(err::IO))
    write(io, take!(out)); write(err, take!(msg)); flush(io); flush(err)
    return nothing
end

function _main(args::Vector{String}, io::IOBuffer, err::IOBuffer)
    isempty(args) && (print(io, USAGE); return 0)
    cmd, rest = args[1], args[2:end]
    cmd in ("-h", "--help", "help") && (print(io, USAGE); return 0)
    cmd == "models" && (println(io, join(keys(MODELS), "\n")); return 0)
    cmd == "fit" && return cmd_fit(rest; io, err)
    cmd == "crlb" && return cmd_crlb(rest; io, err)
    cmd == "montecarlo" && return cmd_montecarlo(rest; io, err)
    cmd == "design" && return cmd_design(rest; io, err)
    throw(ArgumentError("unknown command `$cmd`; `equip --help` lists them"))
end

# The entry point for `julia -m EQUIPCLI ...` (Julia calls `main(ARGS)`): its return value is
# the process's exit code. Straight to `_run`, compiled once for any stream, so nothing is
# compiled again for the types the real stdout and stderr have (a terminal, a file, a pipe).
(@main)(args) = _run(String.(args), stdout, stderr)

include("precompile.jl")            # last: it runs `main`

end
