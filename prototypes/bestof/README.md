# Smart BestOf: the prototype's measurements

Branch `bestof-proto` (2026-11-20). The code is in src/fit/dictionary.jl (the dictionary of
candidate signals, linear parameters solved in the score) and src/fit/setup.jl and starts.jl
(BestOf on several parameters); the results and the decisions left are in docs/design.md, item
4, "Smart BestOf, prototyped and measured".

| script | what it measures | run |
|---|---|---|
| `study.jl` | each model's default start against BestOf per voxel, from the dictionary, with and without linear parameters solved; 10,000 voxels, SNR 50 and 200 (a few minutes); `study_results.md` is its output of 2026-11-20 | `julia -t 1 --project=. prototypes/bestof/study.jl` (model names as arguments to run some; `NVOX=200` for a quick check) |
| `micro.jl` | one voxel's start: the dictionary against scoring per voxel | `julia -t 1 --project=. prototypes/bestof/micro.jl` |
| `vsmain.jl` | default fits, results and speed, on `main` and on the branch (a worktree of `main` beside) | `julia -t 1 --project=PATH prototypes/bestof/vsmain.jl out.jls`, once per checkout |
| `../../bench/voxelfit/smart.jl` | the smart starts on the VoxelFit comparison's data | see bench/voxelfit/README.md |

The switches `EQUIP._PROTO_DICTIONARY[]` and `EQUIP._PROTO_LINEAR[]` (true by default) turn the
two parts off, read when a cache is made.
