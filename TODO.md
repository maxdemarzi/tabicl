# TODO — TabPFN-3.5 transfer backlog

Findings from a full read of the **TabPFN-3.5 Technical Report** (Prior Labs, 14 Sep 2026,
<https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf>),
cross-referenced against this codebase.

Companion documents:
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — phased execution order, ablation protocol, gates.
- [benchmarks/RESULTS.md](benchmarks/RESULTS.md) — the results ledger. Every measurement lands there.

Item IDs are stable. Reference them in branches, commits and PRs (`TP-01: add Fourier value encoding`).

---

## Status as of 2026-09-17

Branch `tabpfn-3.5-transfer`, 7 commits. **236 tests pass, zero failures** — `BUG-01` is fixed, so the suite is
green for the first time on this branch.

| | Item | State |
|---|---|---|
| ✅ | BM-01 | Harness complete: ledger, provenance, aggregation, resumable runner, datasets |
| ✅ | TP-12 | Datetime + text preprocessing, shipped |
| ✅ | BM-02, BM-03, BM-05 | **Baseline measured** on RTX PRO 6000 (the report's own card), $1.32 |
| ✅ | BM-06 | **PASS** — proxy reproduces the published sign at all 8 checkpoints, $117. Phase 2 screening re-priced $520 → ~$115 |
| ✅ | BM-08 | Eval suite 10 → 62 datasets (OpenML-CC18, mechanical rule) |
| 🔧 | TP-01, TP-04 | Code + tests done, **off by default, untrained** |
| 🔧 | TP-14 | Scoring rules done; needs the benchmark's dataset list |
| ⛔ | TP-02, 03, 05, 06, 07, 08, 09, 10, 11 | Not started |
| ⛔ | TP-15, TP-16 | Blocked on measurement |

**Everything that can be decided on a laptop has been.** Every remaining question is a
training run or a benchmark sweep. The GPU lane is ready — see [docs/RUNPOD.md](docs/RUNPOD.md)
and the queue in [scripts/ablations/README.md](scripts/ablations/README.md).

**The baseline is measured; no *ablation* has been.** `TP-01` and `TP-04` are implemented because they are cheap
to implement and their *structure* is testable without training; neither has been shown to
help this model. The first thing to run is not an ablation, it is `BM-06`'s sign-agreement
check — a proxy that cannot reproduce the sign of a known full-scale effect would make every
ablation after it meaningless.

---

## 0. Why this matters: where TabICLv2 currently stands

TabPFN-3.5 benchmarks explicitly against TabICLv2 on seven benchmarks and wins all seven.
Their report is unusually candid about *how*, and the mapping onto this repo is close to
one-to-one — partly because TabPFN-3.5 has converged on a TabICL-shaped design. They state
outright that their synthetic prior takes inspiration from the TabICLv2 prior (report §3.3).

| Benchmark | TabICLv2 | TabPFN-3.5 | Notes |
|---|---|---|---|
| TabArena | — (not in their fig.) | 1 of 89 | 57% win rate vs best other TFM |
| BeyondArena (all) | ~1200 Elo | ~1420 Elo | TabICLv2 behind in **every** slice; 83% win rate against us |
| STRABLE (strings, TF-IDF) | 1383 | 1714 | we are barely above tuned XGBoost (1336); TabPFN-2.5 is 1499 |
| MulTaBench | 1637 | 1856 | we are the best non-TabPFN entry |
| ScoringBench (CRPS mean rank) | 12.05 *(finetuned)* | 2.85 | TabPFN-3 7.64, EXAONE-Tabular 7.59 — **see TP-14** |
| TALENT | — | 1 of 37 | 80% win rate vs us |
| fev-bench | not entered | 6 of 29 | TabPFN-TS-3.5 at 44.2 SQL skill |

BeyondArena Elo values are read off Figures 3–4 and are approximate (their labels are rounded
to the nearest 10 and confidence intervals on the smaller slices are wide).

Two things worth keeping in perspective:

- **TabPFN-3.5-Plus and -Thinking are proprietary and deliberately undocumented** (report §3.4).
  There is nothing to transfer there. Everything in this backlog comes from the open parts.
- **fev-bench is the one board they do not dominate** — 6th, behind TimesFM-3, both Chronos-2
  variants, TiRex-2 and Toto-2.0. Time-series specialists still lead. Calibrate ambitions on
  TP-15 accordingly.

---

## 1. Backlog

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` dropped (record why)

### Phase 0 — Measurement (no model changes)

- [~] **BM-01** Build `benchmarks/` harness skeleton: dataset fetch + cache, per-dataset runner,
      fold handling, result serialization to a stable schema, Elo/mean-rank aggregation.
      *Done:* `_core/schema.py` (provenance + append-only ledger), `_core/aggregate.py`
      (mean rank, win rate, Bradley-Terry Elo, bootstrap CIs), `_core/runner.py` (resumable,
      failures recorded not dropped), 39 passing tests pinning the arithmetic to closed-form cases.
      `_core/datasets.py` (OpenML fetch via scikit-learn, on-disk cache, content digests,
      deterministic folds) with the 10-dataset `REAL_SMALL` suite validated and frozen;
      `suites/real_small.py` (BM-03) and `report.py` (leaderboard CLI). **BM-01 complete.**
- [~] **BM-02** Inference speed + memory microbenchmark. *Implemented* as
      `benchmarks/suites/speed.py`; smoke-run on CPU, full sweep still needs a CUDA box. Mirror report Figure 7: forward time vs
      training rows (1k → 1M) at 100 columns / 1,024 test rows; cached-predict time for 1 and
      100 test rows; **and KV-cache bytes**, which their figure omits but which is the quantity
      TP-05 exists to hold flat.
- [~] **BM-03** Real-data accuracy suite. *Implemented* as `benchmarks/suites/real_small.py`
      over the frozen 10-dataset `REAL_SMALL` list; scores accuracy, balanced accuracy,
      ROC-AUC and log-loss. Smoke-run on CPU; baseline sweep needs a GPU. Original note: TabArena-lite or a TALENT subset — enough datasets to
      rank changes, small enough to run per-ablation.
- [ ] **BM-04** Held-out synthetic eval set, frozen. Pre-generate with `python -m tabicl.prior`
      (`SavePriorDataset`) so every ablation sees an identical stream and differs only in the
      model change. This is the fast inner-loop signal.
- [ ] **BM-05** Record baseline numbers for the current released v2 clf + reg checkpoints across
      BM-02/03/04 into `benchmarks/RESULTS.md`. **Nothing else starts before this lands.**
- [~] **BM-06** Establish and validate the proxy-scale training recipe.
      *Done:* [`scripts/train_proxy.sh`](scripts/train_proxy.sh) (Stage-1 architecture at
      ~5% of the steps, every non-ablated knob pinned) and the sign-agreement queue in
      [`scripts/ablations/README.md`](scripts/ablations/README.md).
      *Remaining:* running it — needs a GPU. Original note: (see
      [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) §"Ablation protocol"). Full Stage 1 is
      500K steps on 4 GPUs — not viable per-ablation.

### Phase 1 — No retrain required

- [~] **TP-14** Investigate the ScoringBench result. **Premise withdrawn — see
      [RESULTS.md](benchmarks/RESULTS.md).** I described 12.05 as anomalous and "the cheapest
      potential win"; three candidate mechanisms were tested against their actual code and all
      three came back neutral (calibrated-vs-raw quantiles, finetuned-vs-base, and
      mse-vs-crps selection, each within 0.25%). 12.05 of **53** methods on a board of
      purpose-built probabilistic regressors is a plausible honest result, not an artifact.
      What remains is *verification*: run their 101-dataset harness and check we reproduce it.
      One genuine gap was found and closed on the way — `FinetunedTabICLRegressor` trained on
      pinball loss but could only early-stop on point metrics; `eval_metric="crps"` now exists
      (no measured benefit; kept on principle). *Scoring machinery done:*
      [`benchmarks/_core/scoring.py`](benchmarks/_core/scoring.py) — CRPS from predictive
      quantiles via the pinball identity (with the factor of 2 and trapezoidal integration
      over levels, both of which are commonly dropped and both of which make the number
      non-comparable to a published one), plus Winkler interval score and coverage. 13 tests
      pinning it to closed forms, including CRPS of a standard normal and the exact size of
      the tail-truncation bias — a scoring bug here would be indistinguishable from the model
      weakness we are trying to rule out.
      *Remaining:* the benchmark's own 101-dataset list, which is not published in the report
      and must come from the ScoringBench repo; and `TabICLRegressor` returns quantiles via
      `predict(output_type="quantiles", alphas=...)`, which still needs wiring into a suite.
      Then the sequence below. Original note: `TabICL v2 (finetuned)` ranks 12.05 vs
      TabPFN-3 at 7.64 — behind a model we beat elsewhere. We have native quantile regression
      ([`_model/quantile_dist.py`](src/tabicl/_model/quantile_dist.py)), so this looks anomalous.
      Reproduce under their harness (101 OpenML regression datasets, subsampled to 3,000 rows,
      5-fold). Plausibly a harness/config issue rather than a model deficiency. Cheapest
      potential win in the whole backlog.
- [ ] **TP-15** Enter fev-bench. `TabICLForecaster` already exists
      ([`forecast/_forecaster.py`](src/tabicl/forecast/_forecaster.py)) and is not on the board.
      TabPFN ran their *general* checkpoint through the TabPFN-TS harness with no time-series
      finetuning and beat their own TS-specific checkpoint at 2.4x the speed — worth testing
      whether the same holds for us.
- [x] **TP-12** Native date + text preprocessing. *Done:* `DatetimeEncoder` (epoch trend,
      calendar parts, cyclical sin/cos pairs, constant-feature dropping), `TextEncoder`
      (char n-gram TF-IDF + SVD), and `classify_string_columns` in
      [`_sklearn/preprocessing.py`](src/tabicl/_sklearn/preprocessing.py); 20 new tests,
      109 sklearn-compliance tests still green. **Datetime is on by default** because those
      columns were previously dropped outright, so there is no prior behaviour to preserve.
      **Text stays ordinal by default** (`text_encoding="tfidf"` opts in) — flipping it is a
      behaviour change and is gated on the STRABLE measurement, per the plan. Previously:
      [`preprocessing.py:141`](src/tabicl/_sklearn/preprocessing.py#L141) raises and points users
      at skrub's `TableVectorizer`. TabPFN-3.5's *open-source* release now handles both natively.
      Given STRABLE has us at 1383, a built-in TF-IDF path for string columns is the cheapest
      move on that board. Inference-side only.
- [ ] **TP-16** Add a current-generation speed comparison to the README.
      **Rescoped after looking closely.** [`README.md:25`](README.md#L25) and
      [`README.md:350`](README.md#L350) say "10x faster than TabPFN-2.5" — which is not
      *wrong*, since it names the model it compares against. It is merely two generations
      old. So this is not a text fix to be applied now; it needs a measurement against
      TabPFN-3.5 and TabPFN-3.5-Fast, and editing the claim before having one would just
      swap a dated number for an invented one. Blocked on BM-02 on a GPU, plus installing
      `tabpfn` to measure against. For reference: TabPFN-3.5 is ~2x slower than TabPFN-3 on
      large training sets, while TabPFN-3.5-Fast is 3x *faster* and still ~150 Elo ahead of
      TabPFN-3 on TabArena.

### Phase 2 — Cell encoding + prior (one retrain)

- [~] **TP-01** Fourier value encoding. **Screened 2026-09-20: null at 20K steps, but the
      test could not see what the change is for.** Identical to control at step 20,000
      (log-loss 0.2946 vs 0.2947 over 62 datasets); starts behind, catches up by 5K. On the
      max-cardinality>=10 slice it is better on 66.7% of datasets — the predicted direction —
      but n=9, p=0.18. CC18's most extreme categorical column has 71 levels where the claim
      concerns hundreds; and the prior it trained against generates ~10 levels on average, so
      the encoder never saw the distribution it exists to resolve. **Re-screen only after
      TP-08 lands and a high-cardinality slice exists, and then screen the two together** —
      as this file already said to do. See [RESULTS.md](benchmarks/RESULTS.md). *Code:*
      `FourierValueEncoder` in [`_model/layers.py`](src/tabicl/_model/layers.py), wired through
      `ColEmbedding` -> `TabICL` -> the training CLI as `--col_fourier_value` /
      `--col_fourier_freqs`, **off by default** so existing checkpoints load unchanged.
      14 tests; the load-bearing ones assert the structural property rather than a benchmark
      number — under a linear projection the embeddings of a value sequence are provably
      collinear (rank 1 centred), so adjacent ordinal codes *must* be near-identical whatever
      the weights; Fourier features break that. Costs ~8K parameters against 27.5M.
      *Remaining:* the ablation itself — queued in
      [`scripts/ablations/README.md`](scripts/ablations/README.md), needs a GPU.
      Previously, [`embedding.py:153`](src/tabicl/_model/embedding.py#L153) was
      `SkippableLinear(feature_group_size if feature_group else 1, embed_dim)` — a bare linear
      projection of the scaled scalar. TabPFN replaced exactly this with a bank of `F=32` learned
      frequencies → `[sin, cos]` → summed over the `G` group positions → `Linear(2F → E)`.
      Their stated motivation: *"captures small differences better than a linear projection of the
      raw number… particularly advantageous for modeling ordinal-encoded categorical variables
      with high cardinality."* Credited to TabFM. Self-contained and ablatable in isolation.
- [ ] **TP-02** In-context ECDF features. Midrank each cell against its column's **training** rows
      → `u ∈ [0,1]` → expand into `K=4` low-frequency sin/cos harmonics. Monotone-transform
      invariant: a skewed feature and its log produce identical terms, and heavy tails that
      standard scaling crushes against the clip spread evenly over `[0,1]`.
      That last clause lands directly on
      [`CustomStandardScaler`](src/tabicl/_sklearn/preprocessing.py#L388) (`clip_min=-100,
      clip_max=100`).
      **Caveat to test, not assume:** our `ColEmbedding` set transformer is *already*
      distribution-aware over each column and can in principle learn ECDF-like statistics. The
      case for making it explicit is that it is O(n log n), free, and — critically — cacheable.
      TP-02 must be ablated against the unmodified column embedder before we commit to it.
- [~] **TP-08** High-cardinality categoricals in the prior. *Code done, untrained.*
      `high_card_prob` in [`_reg2cls.py`](src/tabicl/prior/_reg2cls.py) mixes a log-uniform
      heavy tail into the cardinality draw, capped at `n_rows // 4` because
      `MulticlassAssigner` takes its boundaries from the data and more levels than rows just
      yields empty ones. Exposed as `--prior_high_card_prob` / `--prior_min_high_categories`
      / `--prior_cat_prob`. **Defaults to 0.0 so the prior stays bit-identical** — a prior
      ablation's control has to be. Measured: columns with >=100 levels go from ~1% to ~20%
      at 0.25 and ~30% at 0.4; median levels 7 -> 19. 9 tests. Original note: At
      [`_reg2cls.py:333-341`](src/tabicl/prior/_reg2cls.py#L333-L341):
      `cat_prob=0.2`, and `num_cats = min(max(round(random.gammavariate(1, 10)), 2), max_categories)`
      has mean 10 — so hundreds-of-levels categoricals are essentially never generated. Note
      `DEFAULT_FIXED_HP["max_categories"] = float("inf")`, so the cap is *not* what binds; the
      gamma is. TabPFN retuned toward high cardinality explicitly *in conjunction with* the
      encoding changes. Ship with TP-01 — ordinal codes need value resolution to be useful.
- [ ] **TP-11** Once TP-02 lands, delete preprocessing machinery. TabPFN-3.5 dropped quantile
      transforms, robust scaling and SVD augmentation outright (§3.2). Here the analogue is the
      `norm_methods` inference ensemble, defaulting to `["none", "power"]` at
      [`classifier.py:560`](src/tabicl/_sklearn/classifier.py#L560). Collapsing an ensemble axis
      into a model input is cheaper *and* strictly more expressive. Gated on TP-02 proving out.

### Phase 3 — Capacity (order matters: TP-04 → TP-05 → TP-06)

- [~] **TP-04** QK-norm and extra normalization. *Partially done, untrained.*
      `RMSNorm` in [`_model/attention.py`](src/tabicl/_model/attention.py) (written out rather
      than `nn.RMSNorm`, which needs torch>=2.4 while `pyproject.toml` allows >=2.2), applied
      to per-head queries and keys and threaded through every block up to `--qk_norm`.
      Off by default. 21 norm modules when on (6 column + 3 row + 12 ICL), ~900 parameters.
      **The care went into the cache boundary:** `k` is normed *before* RoPE and before being
      returned for caching, so a stored key and a fresh one are normalized identically and the
      cached path normalizes `q` only. Tests assert cached/uncached equivalence with a
      non-unit learned weight, which is what would expose a double application — that failure
      would stay finite and plausible while being wrong, on a feature the README advertises.
      *Remaining:* the report also adds a LayerNorm after the input encoding and an RMSNorm
      before the task head; neither is implemented yet. And the ablation, which needs a GPU.
      Original note: `grep` found no QK-norm anywhere in
      [`_model/`](src/tabicl/_model/). TabPFN added RMSNorm on queries and keys in attention
      blocks, LayerNorm after the input encoding, and RMSNorm before the task head — specifically
      to stabilize the wider model *and* joint multitask training. Prerequisite for TP-06 and TP-07.
- [ ] **TP-05** Grouped-query attention for test rows. TabPFN keeps test rows attending through a
      *single* 64-dim KV head, so cache size and cached-predict latency stay flat as `d_model`
      doubles — their Figure 7 shows TabPFN-3.5 and TabPFN-3 single-test-row times aligning
      exactly. [`kv_cache.py`](src/tabicl/_model/kv_cache.py) caches full MHA K/V today, so
      widening without this scales the cache linearly. The README markets KV caching as a headline
      feature, so this is the gate on TP-06, not an optional extra.
      **Measured (BM-05-smoke):** our cache is 48 KiB/row/estimator =
      `12 blocks x 2 x 512 dim x 4 B`, independent of column count — ~37 GiB at 100K rows and
      8 estimators in fp32, doubling to ~73 GiB after TP-06. A single 64-dim KV head cuts it ~8x
      at current width and ~16x after TP-06. See [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
- [ ] **TP-06** Width scaling — **this is literally the same knob we already have**.
      [`tabicl.py:203`](src/tabicl/_model/tabicl.py#L203) computes
      `icl_dim = embed_dim * row_num_cls` = 128 × 4 = 512 with `icl_nhead 8` (head dim 64) —
      the identical starting point to TabPFN-3. Their change is 4 → 8 CLS tokens, giving
      1024 dim and 16 heads at a fixed head dim of 64. In our recipe
      ([`train_v2_clf_stage3.sh:81-85`](scripts/train_v2_clf_stage3.sh#L81-L85)) that is
      `--row_num_cls 8 --icl_nhead 16`. Params go 4x; inference time only ~2x, since it is
      linear in width rather than parameter count.
- [ ] **TP-03** Bucketed ECDF cache. TabPFN caches a bucketed ECDF so cache size decouples from
      training-row count. Required for TP-02 to be compatible with the KV-cache path rather than
      silently reintroducing an O(n_train) term.
- [ ] **TP-13** Cache the many-class decoder's *input keys* rather than the final ICL layer's
      train embeddings. Smaller cache and less compute. TabPFN calls this out as a specific
      TabPFN-3 → 3.5 optimization.

### Phase 4 — Training economics

- [ ] **TP-07** Single multitask checkpoint. TabPFN trains classification and regression jointly
      with task type as an input; only the label encoder and output head are task-specific — the
      cell encoder, column distribution embedder, feature aggregator and in-context transformer
      are all shared. We ship two checkpoints and two full 3-stage curricula
      (`scripts/train_v2_{clf,reg}_stage{1,2,3}.sh`) — six training runs.
      [`prior/_reg2cls.py`](src/tabicl/prior/_reg2cls.py) already derives both tasks from the same
      SCM, so the data side is largely in place. Halves training cost, and they report it improves
      both tasks. Depends on TP-04 for stability.

### Phase 5 — Research-grade, highest ceiling

- [ ] **TP-09** Grouped / non-IID splits in the prior. No such notion exists today — the "group"
      concept in [`_dataset.py`](src/tabicl/prior/_dataset.py) is dataset *batching*, not
      distribution shift. TabPFN now generates datasets whose test block comes from a different
      latent group than train, and exposes the group identifier as a feature at inference
      (their §C.2.3). BeyondArena's Grouped and Temporal slices are where **every** TFM still
      loses to tuned + ensembled MLPs — yet TabPFN-3.5 closed that gap to parity on non-large
      datasets purely through the prior. Highest ceiling, least certain payoff.
- [ ] **TP-10** Wide tables. TabPFN now claims 1M rows × 6k features simultaneously, and up to
      20k features with more estimators. Our recipe trains at `--max_features 100`.

---

## 1b. Found along the way (not from the report)

- [x] **BUG-01** *Fixed.* [`forecast/_preprocessing.py`](src/tabicl/forecast/_preprocessing.py)
      allocated the forecast timestamp buffer as a hardcoded `datetime64[ns]`. pandas 3
      preserves the input's resolution (commonly `datetime64[us]`), so the forecast index came
      back at a different resolution than the context it was produced from — which breaks any
      caller that joins or compares the two, not just the test. Now follows the input dtype.
      **The suite is green for the first time: 236 passed, 0 failed.** Previously:
      **pandas 3.0.3**: `pd.testing.assert_index_equal` reports a dtype mismatch on the
      predicted timestamp index. Confirmed pre-existing — it fails identically with the whole
      TabPFN-3.5 branch stashed, so it is not a regression from this work. `pyproject.toml`
      pins only `pandas>=2.1.2` for the forecast extra, so a fresh install on pandas 3.x hits
      this. Either fix the dtype handling in
      [`forecast/_ts_dataframe.py`](src/tabicl/forecast/_ts_dataframe.py) or cap the pin.
      Relevant to **TP-15**, which runs the forecaster.

- [x] **BM-07** *Done.* `drop_uninformative()` in
      [`_core/aggregate.py`](benchmarks/_core/aggregate.py) removes, **at read time**, any
      dataset on which every method scores identically; `report.py` does this by default and
      says which it dropped, with `--keep-uninformative` to opt out. Read-time rather than
      suite-time on purpose: the suite stays frozen, because dropping datasets after seeing
      results is how a benchmark becomes flattering. Original note:
      `banknote-authentication` and `steel-plates-fault` both score 1.0000 accuracy with
      log-loss ~0. A dataset every method solves perfectly cannot separate methods and only
      dilutes a mean rank. Keep them — the suite is frozen and dropping datasets after seeing
      results is how benchmarks become flattering — but exclude or separately report them in
      ablation readouts. Add the split to `benchmarks/report.py`.

- [~] **PERF-01** **Single-row cached prediction is ~1.5x slower than 100-row.**
      *Investigated; the actionable part is done and documented, the anomaly itself is deferred.*
      The 1-row-slower-than-100-row **inversion does not reproduce on CPU** (there 1 row is
      8.6x *faster*), so it is GPU-specific — most likely kernel-launch latency dominating when
      every tensor is tiny — and diagnosing it needs a GPU with a profiler. What is established
      and now in the README: cached latency is **flat in training rows** (48→79 ms for a 16x
      increase) and **scales with `n_estimators`** (12.8→60.9 ms from 1 to 8), so the ensemble
      is the lever for online serving, not the training set size. Trimming it is a trade, not a
      free win: on 6 datasets x 2 folds, accuracy was within noise but 8 estimators ranked best
      on both ROC-AUC and log-loss — ensembling buys ranking and calibration more than accuracy.
      All CIs overlap; directional only. Original note:
      Measured across every row count from 1k to 256k (0.0419 s vs 0.0278 s at 1k). Per-call
      overhead dominates the online path, so the marginal cost of 99 extra rows is negative.
      This is the exact case the report highlights for cached inference and the shape of real
      online serving. Profile it: being flat in row count, it is likely Python-side setup,
      preprocessing, or the ensembling loop rather than attention.
      **Not from the report — found by running BM-02.**

- [x] **DOC-01** *Done* — documented on `--batch_size_per_gp` in the training CLI.
      `micro_batch_size` cannot exceed `batch_size_per_gp` when
      `seq_len_per_gp=True`; every dataset in a micro-batch must share a training size. Raising
      one alone fails at step 0 with `ValueError: All datasets in the micro batch must have the
      same training size`. Add it to the training CLI help.

- [x] **BM-08** *Done.* Evaluation suite grown from 10 datasets to **62**: OpenML-CC18
      restricted by a rule fixed before any result was seen — features <= 500 — and applied
      mechanically, so the 10 excluded are exactly the image-like and very wide ones. All 62
      validated and cached. This is now the **gate on BM-06**: at 8 informative datasets a
      true ~64% win rate (TabICLv2's published effect size) clears p < 0.05 about 15% of the
      time; at ~60 it is about 70%.
      Two bugs surfaced on the way, both fixed. Rows were labelled with the *runner* name
      rather than the dataset suite, and `steel-plates-fault` is two different datasets in the
      two suites (binary 1504 vs 7-class 40982), so they would have been averaged together
      silently. And the suite runner had no way to load a proxy checkpoint — **BM-06's
      evaluation step could not have measured anything but the released model.** Added
      `--model-path`.

- [x] **BUG-02** *Fixed.* Any dataset with an **entirely-missing column crashed at predict
      time**: `IndexError: boolean index did not match indexed array`. `SimpleImputer` drops
      all-missing columns during fit by default, while the estimators build a `feature_mask`
      of all-NaN columns in the *original* feature space precisely so they can be masked —
      the two disagreed by one column. `keep_empty_features=True` lets the mask do the job it
      was written for. Found because OpenML `sick` (in CC18) has an all-NaN `TBG` column and
      failed every fold of the BM-06 anchor evaluation. **Pre-existing** — verified on the
      code from before any change on this branch. 7 regression tests.
      **Effect on the running BM-06:** none on validity. The pod has the unfixed copy, so
      `sick` fails identically for every arm and is dropped symmetrically; comparisons run on
      61 datasets rather than 62.

- [x] **BM-10** *Done.* `HIGH_CARD` suite in
      [`_core/datasets.py`](benchmarks/_core/datasets.py): 14 OpenML datasets chosen by a rule
      fixed before any result — a categorical column with >=100 levels, <=500 features,
      >=2000 rows, <=10 classes. Cardinalities reach **15,415** levels (`KDDCup09_*`), 7,518
      (`Amazon_employee_access`), 7,019 (`okcupid-stem`), against CC18's maximum of 71. All
      14 validated and cached. **Not independent** — the three `KDDCup09_*` tasks share a
      feature matrix and differ only in target, as do the `ipums_la_9x` variants, so effective
      n is nearer 8 than 14. Report it alongside `cc18_narrow`, never instead of it.
      Original note: across all 62
      CC18 datasets the largest categorical column is 71 levels (`cylinder-bands`) and only 9
      reach 10 levels. TP-01, TP-02 and TP-08 all target hundreds-to-thousands of levels, so
      none of them can currently be measured. Needs a dedicated slice — OpenML has candidates
      (`KDDCup09`, `porto-seguro`, `amazon_employee_access`, road-safety), and BeyondArena's
      high-cardinality slice is the reference.

- [ ] **BM-09** Pull ablation checkpoints, or `stop` rather than `terminate`, when the trained
      weights have onward value. BM-06's control checkpoints were destroyed with the pod, so no
      future ablation can reuse that control; each must re-run its own (~$29 at 10K steps).

- [ ] **CAVEAT-01** A proxy is **biased against capacity increases**, not just noisy. TabICLv2's
      own deeper-model ablation showed no clear gain at 280K steps "likely due to insufficient
      pretraining for the larger model to fully converge". A 5K-step proxy is far more
      starved than that. A proxy "no" on **TP-06** must not be read as a real "no".

## 2. Deliberately not pursued

- **TabPFN-3.5-Plus / -Thinking internals** — proprietary, intentionally undescribed (§3.4).
- **FP8 attention** — mentioned as the source of Plus's latency win, but the implementation is
  part of the proprietary stack. Revisit only if it becomes independently available.
- **A "Fast" variant checkpoint** — TabPFN-3.5-Fast is a 4-estimator alpha checkpoint on their
  Pareto front. We already default to `n_estimators=8`, so the same lever exists as a config,
  not a separate artifact. Not worth a distinct checkpoint until TP-06 makes the base model
  slow enough to warrant one.
