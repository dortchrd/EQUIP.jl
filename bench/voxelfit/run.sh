#!/usr/bin/env bash
# EQUIP against the one-shot version (VoxelFit 0.1, branch claude-oneshot), every step:
#   bench/voxelfit/run.sh                 (ONESHOT=/path/to/a/checkout to reuse one; THREADS=4)
# About 15 minutes the first time (installing VoxelFit's packages), then about 10.
set -e
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
oneshot="${ONESHOT:-$here/oneshot}"
threads="${THREADS:-4}"
if [ ! -d "$oneshot" ]; then                      # the one-shot version, beside this repository's
    git -C "$repo" fetch -q origin claude-oneshot
    git -C "$repo" worktree add -q --detach "$oneshot" origin/claude-oneshot
fi
julia --project="$oneshot" -e 'using Pkg; Pkg.instantiate()'
julia --project="$repo/bench" -e 'using Pkg; Pkg.instantiate()'
mkdir -p "$here/data" "$here/out"
rm -f "$here/out/"*
echo "== the data, and EQUIP's models against VoxelFit's equations"
julia --project="$repo/bench" "$here/gen.jl"
echo "== the fits, and their accuracy and agreement"
(cd "$oneshot" && julia -t "$threads" --project=. "$here/vf.jl" acc)
julia -t "$threads" --project="$repo/bench" "$here/eq.jl" acc
julia --project="$repo/bench" "$here/compare.jl"
julia --project="$repo/bench" "$here/diagnose.jl"
julia --project="$repo/bench" "$here/iters.jl"
echo "== speed, one thread then $threads (each package alone on the machine)"
for t in 1 "$threads"; do (cd "$oneshot" && julia -t "$t" --project=. "$here/vf.jl" time > /dev/null); done
for t in 1 "$threads"; do julia -t "$t" --project="$repo/bench" "$here/eq.jl" time > /dev/null; done
julia "$here/timetable.jl"
echo "== start-up: load, first fit, second fit (ms, 2000 voxels)"
for i in 1 6; do
    (cd "$oneshot" && julia --project=. "$here/first_vf.jl" "$i")
    julia --project="$repo/bench" "$here/first_eq.jl" "$i"
done
