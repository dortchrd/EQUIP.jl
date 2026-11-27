# EQUIP experiments, kept but not adopted

This branch holds prototypes that were tried and not merged into `main`. Their outcomes are
recorded in `main`'s docs/design.md. They are kept so the code can be looked at or picked up
again.

Each archiving is one commit whose parents are the archive before it and the prototypes' last commits, so every commit of
every prototype is still here, exactly as it was. A commit's own content is only this file: the
prototypes were not merged into each other (they change the solver at different points in its
history, so merging them would mean resolving conflicts in code that nobody wants).

## Archived 2026-11-27 (this branch's last commit; its parents)

| parent | was branch | last commit | made from `main` at | what it tried, and what came of it |
|---|---|---|---|---|
| 1 | (the archive of 2026-11-23, below) | 4b47226 | | the seven earlier prototypes |
| 2 | `lineshape-cache` | d822550 (2026-11-27), 2 commits | a7bb930 | QMTSPGRYarnykh's lineshapes made once per distinct MT offset: a cache on the stack in `signal!` (up to 16 offsets), the same offset at several MT powers taking the values made there. Bit for bit; a signal 15–20% faster and a whole brain about 10% where offsets repeat, 2–4% slower where they do not (bench/qmt/lineshape_cache.jl). Not adopted (the owner, 2026-11-27: "worry about very specialized code for marginal benefit"): about 25 lines of cache in place of one line of physics. design.md item 6: "The lineshapes once per distinct offset". |

## Archived 2026-11-23 (the commit 4b47226, this branch's first parent; its parents)

| parent | was branch | last commit | made from `main` at | what it tried, and what came of it |
|---|---|---|---|---|
| 1 | (the archive of 2026-11-08, below) | bc996fe | | the four earlier prototypes |
| 2 | `stopping-proto` | 95f6c04 (2026-11-22), 9 commits | 1d5857e | Stopping rules for the solver: the statistical stop (`LM(ctol)`), per-parameter tolerances calibrated from a sample of voxels (`stoptolerances`, `LM(atol)`), tolerances scaled per voxel (`rtol`, `setol`), the stall rule (`LM(stall)`, status `StalledOnBound`). 6–37% faster; not adopted (the owner, 2026-11-22: too complicated for the benefit). Its `writemaps(compress)` went into `main` differently (maps gzipped as the data are). design.md: "Stopping tolerances against the noise" and "Voxels a poor mask lets in", Performance notes. |
| 3 | `geodesic-proto` | 1fd6d99 (2026-11-21), 1 commit | d44a173 | Geodesic acceleration for Levenberg–Marquardt (`LM(geodesic = true)`). Slower in every model measured, not more robust; not adopted. design.md: "Geodesic acceleration, prototyped", Performance notes. |
| 4 | `combined-proto` | 94eeaa2 (2026-11-22), 14 commits | 1d5857e | `stopping-proto` and `bestof-proto` merged, only to time them together on the owner's scan at full size (the stopping rules, QMTSIREqualR1, uncompressed maps and the dictionary starts: about 8 s against 20); and the 4-point SIR's two exact fits found there. design.md: "Voxels a poor mask lets in". `bestof-proto` itself is kept as a branch (on hold, with roadmap item 6). |

## Archived 2026-11-08 (the commit bc996fe, this branch's first parent; its parents)

| parent | was branch | last commit | made from `lean` at | what it tried, and what came of it |
|---|---|---|---|---|
| 1 | `varpro-proto` | 4b4f410 (2026-10-23), 6 commits | 866ea79 | Variable projection on the benchmark's biexponential cases, and log coordinates. Not faster as prototyped; kept as a design for later. design.md: "Variable projection (VarPro): proposed design". |
| 2 | `step-proto` | 6c3ad9a (2026-10-23), 1 commit | ab853a3 | Step-control variants for the Levenberg–Marquardt solver (`scaling`, `update`, `gnfirst`, `guard`, `trust`; bench/steps.jl). Today's step control kept. design.md: "Step-control prototype", under "Performance notes". |
| 3 | `sens-proto` | e8411b5 (2026-10-29), 3 commits | 7506c8b | `sensitivity` and then `sensitivityefficiency` maps (Dortch 2018, Eq. 6 and Fig. 2A). Built, then removed: Eq. 6 is `jacobian ./ √t`, shown in `jacobian`'s docstring; its plot is on the roadmap. design.md: "Diagnostics: why a protocol works". |
| 4 | `prototype/solver-deadzone` | 8ca4a95 (2026-10-17), 1 commit | 3063d6d | Solver options for steps toward bounds. Superseded by the solver's dead-zone rule (`_lost_influence` in src/fit/levmar.jl). |

## Getting one back

```
git fetch origin experiments
git log --oneline experiments^2                # lineshape-cache's commits
git switch -c lineshape-cache experiments^2     # a local branch, exactly as it was
git log --oneline experiments^1^2              # the 2026-11-23 ones: stopping-proto (^1^3 geodesic, ^1^4 combined)
git switch -c stopping-proto experiments^1^2
git log --oneline experiments^1^1^2            # the 2026-11-08 ones: step-proto (^1^1^1 varpro, ^1^1^3 sens, ^1^1^4 deadzone)
git push -u origin stopping-proto               # and on GitHub again, if wanted
```

The last commits' hashes (column 3) work in place of `experiments^N` too.
