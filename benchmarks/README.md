# benchmarks/

Evaluation harness for the [TabPFN-3.5 transfer backlog](../TODO.md).

**Status: not yet implemented.** This directory currently holds only the results ledger.
Building the harness is item **BM-01** and it blocks Phases 2–5 — see
[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) §"Phase 0".

## Intended layout

```
benchmarks/
  README.md            # this file
  RESULTS.md           # the ledger — append-only, single source of truth
  _core/
    datasets.py        # fetch + on-disk cache, checksum-pinned
    runner.py          # per-(dataset, fold, model-config) execution, resumable
    schema.py          # one result row: run id, commit sha, ckpt id, dataset, fold,
                       #   metric, value, wall-clock, peak VRAM, seed
    aggregate.py       # mean rank, Elo, bootstrap CIs, win rates
  suites/
    speed.py           # BM-02  inference speed, memory, KV-cache bytes
    real_small.py      # BM-03  fixed real-data suite, frozen day one
    synthetic.py       # BM-04  frozen held-out prior stream
    scoringbench.py    # TP-14  101 OpenML regression datasets, CRPS
    fev.py             # TP-15  time-series forecasting
```

## Non-negotiables

Cheap to build in from the start, expensive to retrofit:

- **Resumable** — keyed by `(run_id, dataset, fold)`. Long sweeps get interrupted.
- **Commit-pinned** — every row records commit SHA and checkpoint identity.
- **Seed-explicit** — ≥3 seeds for anything gating a phase decision.
- **Cost-aware** — wall-clock and peak VRAM per row, always.
- **Frozen evaluation sets** — BM-03's dataset list and folds never change. A moving evaluation
  set makes the ledger worthless.

## Validate the harness before trusting it

Reproduce a *published* baseline number before reporting any of our own. This is the method the
TabPFN-3.5 report used on ScoringBench: they re-ran the released TabPFN-3 under the benchmark's
own harness and confirmed it reproduced the published CRPS to within 1e-3 relative and landed on
the same mean rank, *then* reported their model. A harness that cannot reproduce a known number
cannot be used to claim a new one.
