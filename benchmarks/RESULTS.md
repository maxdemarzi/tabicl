# Results ledger

Single source of truth for every measurement taken against the
[TabPFN-3.5 transfer backlog](../TODO.md). See
[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for the protocol.

## Rules

1. **Append only.** Rows are never edited or deleted. A superseded result is superseded by a new
   row that references the old one, not by rewriting it.
2. **Provenance is mandatory.** Commit SHA, checkpoint identity, seed, hardware. A number without
   provenance is not a result.
3. **Cost is recorded alongside accuracy.** Wall-clock and peak VRAM. The target is a Pareto
   frontier, not a single number.
4. **Negative results are recorded with the same care as positive ones**, with a one-line reading
   of why. Knowing which of TabPFN's changes do *not* transfer is half the value of this work.
5. **≥3 seeds** for anything that gates a phase decision.

---

## Reference points (external, not ours)

From the TabPFN-3.5 technical report, 14 Sep 2026. Recorded so our numbers have something to be
compared against. **These were not measured by us** and the BeyondArena values are read off
Figures 3–4 (labels rounded to the nearest 10, wide CIs on small slices).

| Benchmark | Metric | TabICLv2 | TabPFN-3 | TabPFN-3.5 | Source |
|---|---|---|---|---|---|
| BeyondArena (all) | Elo | ~1200 | ~1280 | ~1420 | Fig. 3 |
| BeyondArena (high-card) | Elo | ~1180 | ~1290 | ~1340 | Fig. 3 |
| BeyondArena (text) | Elo | ~1230 | ~1180 | ~1580 | Fig. 3 |
| STRABLE (TF-IDF) | Elo | 1383 | — | 1714 | Fig. 5a |
| MulTaBench | Elo | 1637 | — | 1856 | Fig. 5b |
| ScoringBench | CRPS mean rank | 12.05 *(finetuned)* | 7.64 | 2.85 | Fig. 22 |
| fev-bench | SQL skill | not entered | 43.1 | 44.2 | Fig. 21 |

---

## BM-05 — Baseline (current released checkpoints)

**Status: not yet run.** Blocks all of Phases 2–5.

### Speed and memory (BM-02)

| Run | Commit | Ckpt | Train rows | Cols | Fwd time (s) | Cached 1-row (s) | Cached 100-row (s) | Cache (MB) | Peak VRAM (GB) | HW |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

### Real-data accuracy (BM-03)

| Run | Commit | Ckpt | Suite | Seeds | Mean rank | Elo | Median time/1k (s) | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

### Synthetic held-out (BM-04)

| Run | Commit | Ckpt | Eval set | Seeds | Metric | Value | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## BM-06 — Proxy-scale validation

Sign agreement between proxy-scale ablations and the published TabICLv2 full-scale results.
The proxy is not trusted until the signs agree.

| Knob | Published full-scale effect | Proxy steps | Proxy effect | Sign agrees? | Notes |
|---|---|---|---|---|---|
| `--col_ssmax` | | | | | |
| `--col_target_aware` | | | | | |
| `--row_rope_interleaved` | | | | | |

---

## Phase 1 — No retrain

### TP-14 ScoringBench

Harness must first reproduce a published baseline before our own numbers are trusted.

| Run | Commit | Config | Harness reproduces baseline? | CRPS mean rank | vs published 12.05 | Reading |
|---|---|---|---|---|---|---|
| | | | | | | |

### TP-15 fev-bench

| Run | Commit | Ckpt | Harness | SQL skill | MASE skill | Rank /29 | Time vs TabPFN-TS-3.5 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### TP-12 Native date/text preprocessing

| Run | Commit | Encoding | STRABLE Elo | vs 1383 baseline | vs XGBoost-tuned 1336 | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

---

## Phase 2 — Cell encoding and prior

Each item ablated **separately** at proxy scale before any combination is trained.
Prior changes are evaluated on **both** frozen synthetic sets (old prior and new) — a change that
only helps on its own distribution has proven nothing.

| Run | Commit | Item | Scale | Seeds | BM-04 old | BM-04 new | BM-03 | High-card slice | Text slice | Cost delta | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | TP-01 | proxy | | | | | | | | |
| | | TP-02 | proxy | | | | | | | | |
| | | TP-08 | proxy | | | | | | | | |
| | | combined | proxy | | | | | | | | |
| | | combined | full | | | | | | | | |

### TP-11 preprocessing simplification (gated on TP-02)

| Run | Commit | `norm_methods` | BM-03 | Inference time | Reading |
|---|---|---|---|---|---|
| | | `["none","power"]` (baseline) | | | |
| | | `["none"]` | | | |

---

## Phase 3 — Capacity

**TP-05 gate:** cache bytes and single-test-row latency must be flat versus baseline at unchanged
width before TP-06 changes the width at all.

| Run | Commit | Item | Params | Fwd time (s) | Cached 1-row (s) | Cache (MB) | BM-03 | Gate met? | Reading |
|---|---|---|---|---|---|---|---|---|---|
| | | TP-04 | | | | | | n/a | |
| | | TP-05 | | | | | | | |
| | | TP-06 | | | | | | n/a | |

---

## Phase 4 — Multitask checkpoint

Joint training is compared against a **single-task control run at the same proxy scale**, not
against the published claim.

| Run | Commit | Config | Clf metric | Reg metric | Train cost | vs single-task control | Reading |
|---|---|---|---|---|---|---|---|
| | | single-task control | | | | n/a | |
| | | joint (TP-07) | | | | | |

---

## Phase 5 — Research

### TP-09 grouped / non-IID prior

Report both with and without the group identifier exposed as a feature — the deviation from
BeyondArena's standard procedure is TabPFN's to defend and ours to disclose.

| Run | Commit | Prior config | Group id exposed | BeyondArena grouped | BeyondArena temporal | vs tuned+ens MLP | Reading |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## Dropped / did not transfer

Items investigated and abandoned. Kept permanently — this is the record of what does *not* carry
over from TabPFN's model shape to ours.

| Item | Date | Investigated to | Why dropped |
|---|---|---|---|
| | | | |
