# =================================================================================================
# EQUIP's tests, in groups (docs/design.md, "Tests and benchmarks"):
#
#   unit/          fast: the models' signals against their equations, roles and setup, starts and
#                  the solver, errors and printing
#   integration/   the parts together: datasets, statistics, crlb, montecarlo, sweeps, leakage,
#                  design, files, the diffusion models, derived quantities, @mrimodel
#   validation/    expensive statistical and numerical validation, run apart (none yet)
#
# Which ones run:
#   Pkg.test()                                   unit and integration (CI on every push)
#   Pkg.test(test_args = ["unit"])               a group; "all" for every group
#   Pkg.test(test_args = ["crlb", "io"])         files by name, from any group
#   julia -t auto --project=test test/runtests.jl unit       the same, without Pkg.test's setup
#
# Run with threads so the parallel path is exercised. Each file needs only test/setup.jl before
# it. Split from one file on 2026-11-08, every test set moved as it was (docs/design.md).
# =================================================================================================

include("setup.jl")

# The files of each group, in the order they run (names without .jl).
const GROUPS = (
    unit = ["models", "roles", "starts_solver"],
    integration = ["fitting", "simulation", "crlb", "montecarlo_sweeps", "noise_weights",
                   "scenarios_leakage", "design", "io", "diffusion", "derived", "macro"],
    validation = String[],
)

# The files to run, from the arguments: group names, "all", or file names; none: unit and integration.
function testfiles(args)
    isempty(args) && (args = ["unit", "integration"])
    files = String[]
    for a in args
        if a == "all"
            for g in keys(GROUPS)
                append!(files, joinpath.(String(g), GROUPS[g] .* ".jl"))
            end
        elseif Symbol(a) in keys(GROUPS)
            append!(files, joinpath.(a, GROUPS[Symbol(a)] .* ".jl"))
        else
            g = findfirst(fs -> a in fs, GROUPS)
            g === nothing && error("unknown test group or file `$a`. Groups: " *
                                   "$(join(keys(GROUPS), ", ")), all. Files: " *
                                   "$(join(reduce(vcat, collect(GROUPS)), ", ")).")
            push!(files, joinpath(String(g), a * ".jl"))
        end
    end
    return unique(files)
end

@testset "EQUIP" begin
    for f in testfiles(ARGS)
        @testset "$f" begin
            include(f)
        end
    end
end
