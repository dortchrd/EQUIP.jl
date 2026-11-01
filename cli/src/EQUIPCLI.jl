# =================================================================================================
# EQUIPCLI: EQUIP from the command line, and its data files
#
# A companion package, so that EQUIP itself depends on no file format. Here:
#   io.jl     data in (NIfTI with BIDS sidecars; text files of values), maps out
# Next (docs/design.md, "The command line and the data reader: the plan"): the `equip` command,
# a subcommand per part (fit, crlb, montecarlo, design), flags first, FSL-style config files.
# =================================================================================================

module EQUIPCLI

using EQUIP
import JSON
using NIfTI: niread, niwrite, NIVolume

export loaddata, loadmap, readvalues, savemaps

include("io.jl")

end
