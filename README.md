# EQUIP experiments, kept but not adopted

This branch holds four prototypes that were tried and not merged into `main`. Their outcomes are
recorded in `main`'s docs/design.md. They are kept so the code can be looked at or picked up
again.

The branch has one commit, whose four parents are the four prototypes' last commits. Every
commit of every prototype is still here, exactly as it was. This commit's own content is only
this file: the prototypes were not merged into each other (they change the solver at different
points in its history, so merging them would mean resolving conflicts in code that nobody
wants).

| parent | was branch | last commit | made from `lean` at | what it tried, and what came of it |
|---|---|---|---|---|
| 1 | `varpro-proto` | 4b4f410 (2026-10-23), 6 commits | 866ea79 | Variable projection on the benchmark's biexponential cases, and log coordinates. Not faster as prototyped; kept as a design for later. design.md: "Variable projection (VarPro): proposed design". |
| 2 | `step-proto` | 6c3ad9a (2026-10-23), 1 commit | ab853a3 | Step-control variants for the Levenberg–Marquardt solver (`scaling`, `update`, `gnfirst`, `guard`, `trust`; bench/steps.jl). Today's step control kept. design.md: "Step-control prototype", under "Performance notes". |
| 3 | `sens-proto` | e8411b5 (2026-10-29), 3 commits | 7506c8b | `sensitivity` and then `sensitivityefficiency` maps (Dortch 2018, Eq. 6 and Fig. 2A). Built, then removed: Eq. 6 is `jacobian ./ √t`, shown in `jacobian`'s docstring; its plot is on the roadmap. design.md: "Diagnostics: why a protocol works". |
| 4 | `prototype/solver-deadzone` | 8ca4a95 (2026-10-17), 1 commit | 3063d6d | Solver options for steps toward bounds. Superseded by the solver's dead-zone rule (`_lost_influence` in src/fit/levmar.jl). |

## Getting one back

```
git fetch origin experiments
git log --oneline experiments^2              # step-proto's commits (^1 varpro, ^3 sens, ^4 deadzone)
git switch -c step-proto experiments^2        # a local branch, exactly as it was
git push -u origin step-proto                 # and on GitHub again, if wanted
```

The last commits' hashes (column 3) work in place of `experiments^N` too.
