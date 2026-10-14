# Build a system image with VoxelFit and its compiled code baked in, so that the command-line
# tool (and MATLAB wrapper) start almost instantly:
#
#   julia --project=build -e 'using Pkg; Pkg.add("PackageCompiler")'   # once
#   julia --project=build build/build_sysimage.jl
#   export VOXELFIT_SYSIMAGE=$PWD/build/voxelfit.so                   # .dylib / .dll elsewhere
#   bin/voxelfit fit ...
#
# By default the image targets this machine's CPU ("native"), which builds fastest. For an image
# shared across a cluster with mixed CPUs set VOXELFIT_CPU_TARGET=generic (PackageCompiler's
# portable multi-target default; slower to build and needs ~8 GB RAM).
# Rebuild after updating VoxelFit or Julia. Needs a C compiler (gcc/clang).
using PackageCompiler, Libdl

root = dirname(@__DIR__)
target = get(ENV, "VOXELFIT_CPU_TARGET", "native")
create_sysimage(["VoxelFit"];
                sysimage_path=joinpath(@__DIR__, "voxelfit." * Libdl.dlext),
                project=root,
                cpu_target=target == "generic" ? PackageCompiler.default_app_cpu_target() : target,
                precompile_execution_file=joinpath(@__DIR__, "precompile_cli.jl"))
