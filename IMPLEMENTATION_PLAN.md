# Implementation plan — TabPFN-3.5 transfer

Execution plan for the backlog in [TODO.md](TODO.md). Item IDs (`TP-nn`, `BM-nn`) are shared
between the two documents. All measurements land in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

---

## Guiding principles

1. **Measure before changing anything.** There is currently no evaluation harness in this
   repository — four unit tests and nothing else. Every claim in the TabPFN report about what
   helped is a claim about *their* model on *their* prior. We verify on ours or we do not ship it.
2. **One change, one ablation, one row in the ledger.** Bundled changes that move a number
   together teach us nothing about which half to keep.
3. **Cheap signal first, expensive signal last.** Held-out synthetic (minutes) → small real-data
   suite (hours) → full benchmark (days). A change must clear each tier before it earns the next.
4. **Never regress cached inference silently.** KV caching is a headline feature of this project.
   Cache size in bytes and single-test-row latency are tracked as first-class metrics on every
   architecture change, not checked at the end.
5. **Retrains are the scarce resource.** Group changes that require a retrain into as few
   full-scale runs as possible, and use proxy scale to decide what goes into each one.

---

## Phase 0 — Measurement infrastructure

**Nothing else begins until BM-05 lands.** This phase requires no model changes and no GPU
training, only inference.

### BM-01 — Harness skeleton

Create `benchmarks/` with:

```
benchmarks/
  README.md            # how to run, how to add a benchmark
  RESULTS.md           # the ledger (see §Tracking)
  _core/
    datasets.py        # fetch + on-disk cache, checksum-pinned
    runner.py          # per-(dataset, fold, model-config) execution, resumable
    schema.py          # one result row: run id, commit sha, ckpt id, dataset, fold,
                       #   metric name, value, wall-clock, peak VRAM, seed
    aggregate.py       # mean rank, Elo, bootstrap CIs, win rates
  suites/
    speed.py           # BM-02
    real_small.py      # BM-03
    synthetic.py       # BM-04
    scoringbench.py    # TP-14
    fev.py             # TP-15
```

Requirements that are easy to get wrong and expensive to retrofit:

- **Resumable.** Long sweeps will be interrupted. Keyed by `(run_id, dataset, fold)`.
- **Commit-pinned.** Every result row records the commit SHA and checkpoint identity. A number
  without provenance is not a result.
- **Seed-explicit.** Record the seed; run ≥3 seeds for anything that gates a decision.
- **Cost-aware.** Record wall-clock and peak VRAM per row. Accuracy gains that cost 3x inference
  need to be visible as such — the report's own framing is a Pareto frontier, not a single number.

### BM-02 — Speed and memory microbenchmark

Mirror report Figure 7 so our numbers are directly comparable to theirs:

- Forward time vs training rows, 1k → 1M, at 100 columns and 1,024 test rows, default estimators.
- Cached-predict time for **1** and **100** test rows against a cache built over the training rows.
- **KV-cache size in bytes** as a function of training rows. Their figure omits this; it is
  precisely the quantity TP-05 exists to hold flat while TP-06 quadruples parameters.

The single-test-row cached number is the one that matters for online serving — it is bound by
memory bandwidth moving the cache and weights, not by compute.

### BM-03 — Real-data accuracy suite

Pick a fixed subset — TabArena-lite, or a TALENT subset — sized so a full sweep runs in hours,
not days. Freeze the dataset list and fold definitions on day one and never change them; a moving
evaluation set makes the ledger worthless. Larger boards (BeyondArena, STRABLE, MulTaBench) are
run only at phase gates, not per-ablation.

### BM-04 — Frozen synthetic eval set

Pre-generate a held-out stream with `python -m tabicl.prior` (`SavePriorDataset`, see
[`prior/_genload.py`](src/tabicl/prior/_genload.py)) and freeze it. Two properties make this the
inner loop:

- Every ablation sees a byte-identical data stream, so runs differ *only* in the model change.
- It is cheap enough to evaluate every few thousand training steps, giving a learning curve rather
  than a single end-point.

Generate **two** sets: one from the current prior, and — once TP-08/TP-09 land — one from the
revised prior. Report both, always. A prior change that improves scores only on its own
distribution has proven nothing.

### BM-05 — Baseline

Run the current released v2 classifier and regressor checkpoints through BM-02/03/04. Write the
numbers to the ledger. This is the line every subsequent row is compared against.

### BM-06 — Proxy-scale training recipe

The blocking constraint on this entire project: **Stage 1 is 500K steps at batch 64 on 4 GPUs**
(see [`scripts/train_v2_clf_stage1.sh`](scripts/train_v2_clf_stage1.sh)), and the full curriculum
is three stages × two checkpoints = six runs. Per-ablation full-scale training is not viable.

Define a proxy: **Stage-1 architecture and hyperparameters, reduced step count**, on the frozen
BM-04 prior stream. Start at ~5% of Stage 1 (25K steps) and adjust once wall-clock is known.

Then **validate the proxy before trusting it**, which is the step most likely to be skipped and
most expensive to skip. Take two or three architecture choices whose full-scale effect is already
known from the TabICLv2 paper's own ablations — SSMax (`--col_ssmax` / `--icl_ssmax`),
target-aware embeddings (`--col_target_aware`), the RoPE variant (`--row_rope_interleaved`) — run
them at proxy scale, and check the **sign** of each effect agrees with the published result. If
signs disagree, the proxy is too short; increase and repeat. Record the validation in the ledger.

We are buying sign agreement and rough ordering, not effect-size fidelity. Decisions that hinge on
a small effect size need full-scale confirmation regardless.

---

## Phase 1 — No retrain required

Runs concurrently with Phase 0 once BM-01 exists. These items touch only inference and
preprocessing, so they carry no training cost and no risk to the checkpoints.

### TP-14 — ScoringBench investigation *(do this first)*

`TabICL v2 (finetuned)` at mean rank 12.05 against TabPFN-3 at 7.64 is an outlier relative to
every other board, and we have native quantile regression. Sequence:

1. Reproduce their harness: 101 OpenML regression datasets, 3,000 rows each, 5-fold, CRPS.
2. Confirm the harness is sound by reproducing a *published* baseline's number, exactly as the
   report did when they re-ran TabPFN-3 to within 1e-3 relative before trusting their own runs.
3. Only then evaluate our checkpoints — default config *and* the finetuned path they used.

Three outcomes, all useful: a harness/config bug (cheapest win in the backlog), a genuine
calibration weakness (which becomes a new backlog item, likely touching
[`quantile_dist.py`](src/tabicl/_model/quantile_dist.py)), or a fair result we simply lose
(closes the question).

### TP-15 — fev-bench entry

`TabICLForecaster` exists and is not on the board. TabPFN ran their general tabular checkpoint
through the TabPFN-TS harness — casting each series as regression over calendar and seasonal
features, context 32768, 12 AutoSeasonal periods — with no time-series finetuning, and beat their
own TS-specific checkpoint at 2.4x the speed. Test whether that holds for us.

Set expectations honestly: TabPFN-TS-3.5 places **6th of 29**, behind TimesFM-3, both Chronos-2
variants, TiRex-2 and Toto-2.0. This is a visibility and diagnostic item, not a win condition.

### TP-12 — Native date and text preprocessing

Today [`preprocessing.py:141`](src/tabicl/_sklearn/preprocessing.py#L141) raises and redirects to
skrub. Add a built-in path:

- **Dates** → calendar decomposition (year, month, day-of-week, day-of-year, hour) plus cyclical
  sin/cos encodings for the periodic components.
- **Strings** → TF-IDF by default. The report notes TF-IDF was the *strongest* string encoding on
  STRABLE, which is convenient: it needs no extra model dependency. Keep skrub as the documented
  advanced path rather than replacing it.

Gate on STRABLE. Our 1383 against tuned XGBoost's 1336 is the number to move.

### TP-16 — README correction

Depends on BM-02. Replace the stale "10x faster than TabPFN-2.5" claim
([`README.md:25`](README.md#L25)) with measured numbers against the current generation. Being
accurate here costs nothing and being wrong is noticed.

---

## Phase 2 — Cell encoding and prior *(first retrain)*

The highest value-per-unit-risk group, and the one the report attributes most of the
high-cardinality and text gains to. All three are ablated separately at proxy scale, then the
survivors are combined into one full-scale run.

### TP-01 — Fourier value encoding

Replace the bare linear cell projection at
[`embedding.py:153`](src/tabicl/_model/embedding.py#L153):

```
current:  SkippableLinear(feature_group_size if feature_group else 1, embed_dim)
target:   x → [sin(x·f_i), cos(x·f_i)] for F=32 learned frequencies f
          → summed over the G group positions
          → Linear(2F → embed_dim)
```

Add behind a config flag (`--col_fourier_value`, `--col_fourier_freqs`) defaulting off, so the
existing checkpoints keep loading. Initialization of the frequency bank is the detail most likely
to decide whether this works — log-spaced is the conventional starting point; treat it as a tuned
hyperparameter, not a given.

### TP-02 — In-context ECDF features

Midrank each cell against its column's training rows → `u ∈ [0,1]` → `[sin(kπu), cos(kπu)]` for
`k = 1..K`, `K=4` → concatenate into the metadata encoder alongside the scaled values and the
missing/infinite indicators.

Design notes:

- Ranks come from **training rows only**. Ranking against train+test leaks, and the whole point is
  that the statistic is computable at cache-build time.
- The scaled value is still passed through, so exact magnitudes survive alongside the rank. This
  is explicit in the report and easy to get wrong by replacing rather than adding.
- **Ablate honestly against our column embedder.** Our `ColEmbedding` set transformer is already
  distribution-aware and may already encode this. The novel content here is cheapness and
  cacheability, not necessarily representational power. If the accuracy delta is nil but TP-11
  becomes possible, that is still a win — but record it as *that* win, not an accuracy one.

### TP-08 — High-cardinality prior

At [`_reg2cls.py:333-341`](src/tabicl/prior/_reg2cls.py#L333-L341), widen the cardinality
distribution so hundreds-of-levels categoricals actually occur, and raise `cat_prob` from 0.2.
Note that `max_categories` is already `inf` in `DEFAULT_FIXED_HP` — the binding constraint is
`gammavariate(1, 10)`, mean 10. A heavier-tailed distribution over `num_cats` is the change.

Ship with TP-01: ordinal codes are exactly the case where value resolution matters, and the report
is explicit that they tuned the prior toward high cardinality *in conjunction with* the encoding
change. Evaluate on both BM-04 synthetic sets (old prior and new) to separate a real gain from
train/test distribution matching.

### Phase 2 gate

Combined candidate must show, at proxy scale, no regression on BM-04-old and a gain on the
high-cardinality and text slices of BM-03. Then one full three-stage run for the classifier, and
a full BM-02/03 plus BeyondArena and STRABLE evaluation before it is considered.

### TP-11 — Preprocessing simplification *(gated on TP-02 shipping)*

Once an ECDF-equipped checkpoint exists, test dropping the `norm_methods` ensemble
([`classifier.py:560`](src/tabicl/_sklearn/classifier.py#L560), default `["none", "power"]`).
TabPFN-3.5 removed quantile transforms, robust scaling and SVD augmentation outright once their
cell encodings carried the information. If it holds here, we collapse an inference-time ensemble
axis into a model input — cheaper *and* strictly more expressive. If it does not hold, keep the
ensemble and say so; this is a consequence of TP-02, not a goal in itself.

---

## Phase 3 — Capacity *(second retrain; strict ordering)*

**TP-04 → TP-05 → TP-06.** Reversing this order means discovering instability and cache blowup
simultaneously, in a run too expensive to repeat.

### TP-04 — QK-norm and normalization (first)

RMSNorm on queries and keys in the attention blocks, LayerNorm after the input encoding, RMSNorm
before the task head. Their stated purpose is stability *at the larger width* and *across the two
tasks* — so it lands before both TP-06 and TP-07. Verify at proxy scale that it is at worst
neutral at current width before relying on it.

### TP-05 — Grouped-query attention for test rows (second)

Route test-row attention through a single 64-dim KV head so cache size stays flat as width grows.
Touches [`kv_cache.py`](src/tabicl/_model/kv_cache.py) and
[`attention.py`](src/tabicl/_model/attention.py); PyTorch SDPA has native `enable_gqa` support,
which should keep the change modest.

**Gate: BM-02 cache bytes and single-test-row latency must be flat versus baseline at unchanged
width, before TP-06 touches the width at all.** Their Figure 7 shows TabPFN-3.5 and TabPFN-3
aligning exactly on the single-test-row curve despite 4x the parameters; that alignment is the
target, and it is the thing that makes TP-06 affordable.

### TP-06 — Width (third)

`--row_num_cls 4 → 8` and `--icl_nhead 8 → 16`, holding head dim at 64. Since
[`tabicl.py:203`](src/tabicl/_model/tabicl.py#L203) derives `icl_dim = embed_dim * row_num_cls`,
this is a recipe change across all six `scripts/train_v2_*.sh` files rather than a code change —
though [`ColEmbedding`](src/tabicl/_model/embedding.py#L18) `reserve_cls_tokens` and the
downstream shapes need checking for hardcoded assumptions.

Expect ~4x parameters and ~2x inference time on large training sets. Decide explicitly whether we
accept that trade or ship the wider model alongside the current one; the report's answer was to
ship both plus a 4-estimator fast variant, which is a product decision as much as a technical one.

### TP-03 and TP-13 — Cache optimizations

TP-03 (bucketed ECDF cache) is **required** for TP-02 to be cache-compatible rather than silently
reintroducing an O(n_train) term — sequence it here, but do not let TP-02 ship to a release
without it. TP-13 (cache the many-class decoder's input keys rather than the final ICL layer's
train embeddings) is independent and lands whenever convenient.

---

## Phase 4 — Multitask checkpoint

**TP-07.** Train classification and regression jointly, task type as an input; only the label
encoder and output head stay task-specific. Depends on TP-04 for stability.

Halves training cost — six curriculum runs become three — which is why it is worth doing *before*
Phase 5's prior work rather than after. [`prior/_reg2cls.py`](src/tabicl/prior/_reg2cls.py)
already derives both tasks from a shared SCM, so the data pipeline is largely in place; the work
is in the model's task conditioning, the two heads, and a training loop that mixes task types
within a batch.

Risk to watch: task interference. The report claims a joint improvement, but they also added
normalization layers specifically to make it stable, and their prior differs from ours. Hold a
single-task control run at proxy scale to compare against, not just the published claim.

---

## Phase 5 — Research

### TP-09 — Grouped / non-IID prior

Highest ceiling in the backlog, least certain payoff. BeyondArena's Grouped and Temporal slices
are where every TFM still loses to tuned + ensembled MLPs, and TabPFN closed that to parity on
non-large datasets *through the prior alone*.

Two separable pieces:

1. **Generation.** Add a latent group variable to the SCM and draw the test block from a different
   group than train. The "group" concept in [`_dataset.py`](src/tabicl/prior/_dataset.py) today is
   dataset *batching* and is unrelated — do not overload the term; it will cause confusion.
2. **Exposure.** TabPFN passes the group identifier as a feature at inference (their §C.2.3, which
   they flag as differing from BeyondArena's standard procedure). Worth replicating, and worth
   reporting both with and without, since the deviation is theirs to defend and ours to disclose.

### TP-10 — Wide tables

Our recipe trains at `--max_features 100`; TabPFN now claims 1M rows × 6k features. Likely gated
on Phase 3 memory work. Scope after Phase 3 lands, when the memory picture is known.

---

## Ordering summary

| Phase | Items | Retrain | Gate to proceed |
|---|---|---|---|
| 0 | BM-01…06 | none | BM-05 baseline + BM-06 proxy validated |
| 1 | TP-14, 15, 12, 16 | none | — (runs concurrently) |
| 2 | TP-01, 02, 08 → TP-11 | 1 full run | no BM-04-old regression; gain on high-card/text |
| 3 | TP-04 → 05 → 06, TP-03, 13 | 1 full run | cache bytes + 1-row latency flat after TP-05 |
| 4 | TP-07 | 1 full run | joint ≥ single-task control at proxy scale |
| 5 | TP-09, TP-10 | research | scoped after Phase 3 |

Phases 2, 3 and 4 each cost one full curriculum. Phase 4 pays for itself by halving every
subsequent run.

---

## Tracking

**[benchmarks/RESULTS.md](benchmarks/RESULTS.md) is the single source of truth.** Every run
appends a row; no row is edited or deleted after the fact. A superseded result is superseded by a
*new* row, not by rewriting the old one.

Discipline:

- Branch and commit names carry the item ID: `TP-01: add Fourier value encoding`.
- Every ablation records the commit SHA, checkpoint identity, seed, and both accuracy and cost.
- **Negative results are recorded with the same care as positive ones**, with a one-line reading of
  why. Half the value of this backlog is knowing which of TabPFN's changes do *not* transfer to a
  differently-shaped model — and that knowledge only exists if we write it down when it is
  inconvenient.
- When an item is dropped, mark it `[-]` in TODO.md with the reason inline. Do not delete it.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Proxy scale does not predict full scale | BM-06 sign-agreement validation against known TabICLv2 ablations; small effects always confirmed full-scale |
| TP-02 is redundant with our column embedder | Ablate explicitly against unmodified `ColEmbedding`; if accuracy-neutral, justify on TP-11 and cacheability or drop |
| Prior changes only help on the new prior | Always evaluate on both frozen synthetic sets, old and new |
| TP-06 regresses cached inference | TP-05 gated on flat cache bytes *before* any width change |
| TP-07 task interference | Single-task control run at proxy scale, not the published claim |
| Benchmark harness itself is wrong | Reproduce a published baseline number before trusting our own — the method the report used for TabPFN-3 on ScoringBench |
| Evaluation set drifts and invalidates the ledger | BM-03 dataset list and folds frozen on day one |
