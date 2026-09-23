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

**Status: measured 2026-09-17.** RunPod pod `2ne93b4ii3qfkq`, **NVIDIA RTX PRO 6000 Blackwell
Server Edition, 95 GB, sm_120, 128 vCPU**, torch 2.8.0+cu128, checkpoint
`tabicl-classifier-v2-20260212`, commit `23e8e88`, `n_estimators=8`.

The card is deliberately the one the TabPFN-3.5 report used for its own Figure 7, so these
numbers sit on the same hardware as the ones we are trying to beat. FlashAttention-3 was NOT
installed (hardware-capable, sm_120); irrelevant for inference, relevant for any stage-2/3
training run on this pod.

Pod cost for everything in this section: **$1.32**.

### Speed and memory (BM-02)

| Rows | fit+predict (s) | cached 1-row (s) | cached 100-row (s) | cache (GiB) | KiB/row/est |
|---|---|---|---|---|---|
| 1,000 | 0.70 | 0.0419 | 0.0278 | 0.34 | 44.0 |
| 2,000 | 0.81 | 0.0412 | 0.0265 | 0.52 | 34.0 |
| 4,000 | 1.09 | 0.0413 | 0.0263 | 0.88 | 29.0 |
| 8,000 | 1.63 | 0.0419 | 0.0267 | 1.62 | 26.5 |
| 16,000 | 2.60 | 0.0412 | 0.0266 | 3.08 | 25.2 |
| 32,000 | 4.82 | 0.0413 | 0.0264 | 6.01 | 24.6 |
| 64,000 | 9.92 | 0.0420 | 0.0298 | 11.87 | 24.3 |
| 128,000 | 23.68 | 0.0479 | 0.0378 | 23.59 | 24.2 |
| 256,000 | 98.47 | 0.0627 | 0.0538 | 47.03 | 24.1 |

Zero errors. 100 columns, 1,024 test rows for the uncached path, median of 3 repeats
(1 repeat at 256k).

#### The cache law, confirmed at scale

Cache per row per estimator converges to **24.1 KiB**, and the earlier CPU smoke test measured
**48 KiB in fp32**. That is exactly 2x: the GPU path runs under AMP in fp16. The prediction
made from two points on a laptop holds across three orders of magnitude on real hardware, and
the residual at small row counts is the fixed term amortizing, not a different law.

`12 ICL blocks x 2 (K+V) x 512 dim x 2 bytes = 24 KiB`.

**Ceiling on this card:** 47 GiB at 256k rows. 512k would need ~94 GiB against 95 GB of VRAM
and did not complete within the session. So `kv_cache="kv"` tops out somewhere around
**400-500k training rows on a 96 GB card**, at 8 estimators, today. TP-05 (grouped-query
attention) would cut that by ~8x; TP-06 (doubled width) would halve it again without TP-05.

#### Finding: single-row cached prediction is *slower* than 100-row

| Rows | 1 test row | 100 test rows | ratio |
|---|---|---|---|
| 1,000 | 0.0419 | 0.0278 | **1.51x slower** |
| 32,000 | 0.0413 | 0.0264 | **1.56x slower** |
| 256,000 | 0.0627 | 0.0538 | 1.17x slower |

Predicting **one** row costs ~1.5x predicting a hundred. Per-call overhead dominates the
online path, so the marginal cost of the first 99 rows is negative. This is not a rounding
artefact — it holds at every row count, across 3 repeats.

It matters because the single-test-row case is exactly what the report highlights for cached
inference ("a single test example is scored against the training context"), and it is the
shape of real online serving. Worth a profile: it is a fixed cost, so it is likely Python-side
setup, preprocessing, or an ensembling loop rather than attention.

**New backlog item.** Not from the report — found here.

#### Cached prediction does not scale with training rows

0.041 s at 1k rows, 0.042 s at 64k, 0.048 s at 128k, 0.063 s at 256k. Essentially flat over a
256x range. The KV cache delivers what it promises on latency; its cost is memory, and that
is where TP-05 applies.

#### Quadratic onset above ~64k rows

Uncached fit+predict: 9.92 s at 64k, 23.68 s at 128k (2.4x for 2x rows), 98.47 s at 256k
(**4.2x for 2x rows**). Consistent with the report's note that these scale quadratically past
128k, and it lands slightly earlier here.

### Real-data accuracy (BM-03)

10 frozen OpenML datasets, 3 folds, capped at 3,000 rows. 120 rows, **zero errors**, 48 s
total.

| Dataset | accuracy | balanced acc | ROC-AUC | log-loss |
|---|---|---|---|---|
| banknote-authentication | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| blood-transfusion | 0.7954 | 0.6298 | 0.7580 | 0.4684 |
| credit-g | 0.7590 | 0.6726 | 0.8054 | 0.4783 |
| diabetes | 0.7669 | 0.7248 | 0.8437 | 0.4654 |
| kc1 | 0.8672 | 0.6307 | 0.8537 | 0.3156 |
| phoneme | 0.9000 | 0.8713 | 0.9586 | 0.2354 |
| qsar-biodeg | 0.8711 | 0.8483 | 0.9376 | 0.3014 |
| steel-plates-fault | 1.0000 | 1.0000 | 1.0000 | 0.0001 |
| vehicle | 0.8853 | 0.8868 | 0.9791 | 0.2337 |
| wdbc | 0.9719 | 0.9700 | 0.9962 | 0.0693 |

**Two datasets are saturated and should be treated as dead weight:**
`banknote-authentication` (1.0000/1.0000/1.0000) and `steel-plates-fault`
(1.0000, log-loss 0.0001). A dataset every method solves perfectly contributes nothing to a
rank or an Elo and only dilutes the mean. They are kept — the suite is frozen, and silently
dropping datasets after seeing results is how benchmarks become flattering — but they must be
excluded from, or reported separately in, any ablation readout. Recorded as **BM-07** in
[TODO.md](../TODO.md).

---

## BM-06 — Proxy-scale training: throughput and cost

**Status: throughput measured, calibration NOT run.** This is the section that changes the plan.

Clean runs, nothing else on the GPU, 40 steps each, `NJOBS=16`:

| micro_batch_size | batch_size_per_gp | s/it |
|---|---|---|
| 4 (recipe default) | 4 | 3.11 |
| 8 | 8 | 3.04 |
| 16 | 16 | 3.00 |

**Micro-batching is not the bottleneck.** Tripling it buys 3.5%. ~3.0 s/it is the floor for
this configuration on a 96 GB Blackwell card, and `prior_time` was 0 after the first step, so
the CPU prior is keeping up — the step is GPU-bound at `batch_size 64 x 1024 samples`.

### The cost problem

| | s/it | hours | $ at $2.09/hr |
|---|---|---|---|
| One 25,000-step proxy run | 3.0 | **20.8** | **$43.50** |
| BM-06 calibration (control + 3 knobs) | | 83 | **$174** |
| Phase 2 at 3 seeds (3 ablations + control, x3) | | ~250 | **~$520** |

Against an account balance of **$98.33**, the ablation programme as specified in
[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) is **not affordable** — Phase 2 alone is
roughly five times the balance, and that is before Phases 3 and 4.

This is precisely what BM-06 existed to discover, and it is better to have discovered it for
$1.32 than 80 hours in. It does not invalidate the plan; it re-prices it. Options, in the order
worth trying:

1. **Find the cheapest step count that still passes sign agreement.** The plan fixes 25,000
   steps as "~5% of Stage 1", which was a guess, not a measurement. Run the calibration at
   5,000 steps first ($8.70/run, 4 runs = $35). If the signs agree there, the whole programme
   drops by 5x. Only raise steps if they do not.
2. **Move training to a cheaper card.** This lane does not need 96 GB — the proxy model is
   `embed_dim 128`, `d_model 512`. An RTX A6000 is $0.33/hr against $2.09. Even at 3x the
   wall-clock that is a 2x saving, and it costs FlashAttention-3 (sm_86), which only matters
   for stage 2/3 recipes. **Never mix cards within one comparison** — re-run the control on
   whichever card the treatment used.
3. **Cut seeds from 3 to 2 for screening**, keeping 3 only for whatever survives to a
   full-scale run.

### TP-07 — RESULT: no task interference, but no free halving either (2026-09-23)

**$22.22.** Pod `v9uxq8avcu4ch4`, one RTX PRO 6000 Blackwell (95 GB, sm_120), torch
2.8.0+cu128, container capped at 13.6 vCPU, commit `61c9d39` (plus the `NJOBS_PER_ARM` knob
added to launch it), seed 42, 5,000 steps, 10.4 h wall clock. Three arms — `clf_control`,
`reg_control`, `joint` — trained **concurrently on the one GPU** because no host had three
free: 7.35 s/step each, ≈2.45 s of GPU time per arm-step, against 2.65 s/step measured for a
single arm alone. Per-step times from this run are therefore not comparable with single-arm
measurements; divide by three. Prior generation kept up (`prior_time` 0) on 4 workers per arm.
5,820 ledger rows, **zero errors**.

#### At equal steps, the joint arm loses on both tasks

| task | metric | control | joint | joint worse on | p (Bonferroni) |
|---|---|---|---|---|---|
| classification | log-loss, `cc18_narrow` (62) | **0.3108** | 0.3156 | 71.5% | 4.1e-05 |
| regression | CRPS, `ctr23` (35) | **0.2039** | 0.2141 | 89.5% | 1.9e-07 |

Both directions hold at step 2,500 as well. **The gate as written fails**: joint is not ≥ each
single-task control at the same step count.

#### At equal data per task — which is equal total compute — it does not

A step costs the same in every arm, and the joint arm splits its 64 datasets 32/32. So a joint
run at 2N steps costs exactly what the *pair* of single-task runs at N steps costs, and gives
each task the same number of datasets. This run contains that comparison: joint@5000 against
each control@2500.

| task | metric | control @2500 | joint @5000 | joint worse on | p |
|---|---|---|---|---|---|
| classification | log-loss | 0.3213 | **0.3156** | 35.5% | **better, p = 0.0016** |
| regression | CRPS | 0.2149 | 0.2141 | 51.4% | no difference, 0.41 / 0.59 |

Median per-dataset change at equal data: **−1.2%** log-loss, **+0.1%** CRPS.

**Reading: sharing the trunk costs nothing; the deficit at equal steps is a data-rate effect,
not task interference.** That is the risk the plan flagged, and it did not materialise — at
proxy scale, for one seed, without TP-04's stability norms, which were not needed to make
joint training train stably here. What is *not* supported is the report's headline framing as
it applies to us: the saving is **one checkpoint instead of two, not half the compute**. A
joint run matched to the pair on data costs what the pair costs. It still halves the number of
curriculum runs (six → three), removes a whole pipeline, and ships 29.10M parameters instead
of 56.1M.

Anchor, for scale: released v2 scores 0.2759 log-loss and 0.1718 CRPS against the proxy
control's 0.3108 / 0.2039. These arms ran 5,000 steps — 1% of stage 1.

**Provisional.** One seed, 5,000 steps (a quarter of the 20,000-step screen, which BM-06
validated only for *sign* agreement), proxy scale, and CAVEAT-01 applies. Three seeds at
20,000 steps are what the gate asks for; the equal-data comparison is the one to repeat.

### TP-07 — one checkpoint for both tasks: built, verified for $0, ready to launch (2026-09-21)

**Everything short of the training run.** Model (`580f263`), trainer (`bb607a5`), and — in the
commit that adds this entry — the regression suite, the three-arm launcher and the joint
recipe. Nothing trained; whether joint training *helps* is the GPU question below.

What is established without a GPU:

| Claim | How checked |
|---|---|
| A joint model can be *exactly* either released-architecture single-task model | load its weights, compare every forward path (train, inference, KV cache, repr cache), both tasks |
| The joint step's gradient is exactly CE-grad + w·pinball-grad | one step each way, compared per parameter; mutation-checked |
| Single-task training is unchanged | new vs pre-TP-07 `_run.py` from git: **bit-identical** weights and metrics after 3 steps, both tasks, AdamW and Muon |
| The joint recipe builds | `train_v2_joint_stage1.sh` parsed and built on CPU: 29.10M params, 32/32 split, two Muon groups |

Params: **29.10M joint vs 27.55M classifier + 28.56M regressor** — +5.6% over one checkpoint,
in place of two. The cost stake, from the per-stage measurement below: one full curriculum
is ~$1,470–2,740, so the regressor checkpoint TP-07 removes is worth that much again.

#### CTR23: a regression suite, because there was none

Every suite until now was classification, so a regression checkpoint could not be screened.
`ctr23` is OpenML-CTR23 (study 353), the regression counterpart of CC18. The `cc18_narrow`
rule (≤ 500 features), fixed before any result, removes none of the 35 (widest: 116).
Validated with `python -m benchmarks._core.datasets --suite ctr23 --validate`:

| dataset | id | rows | cols | cat | digest |
|---|---|---|---|---|---|
| Moneyball | 41021 | 1,232 | 14 | 6 | f1ee9cbf08a6280c |
| abalone | 44956 | 4,177 | 8 | 1 | 81dd4cc70daca341 |
| airfoil_self_noise | 44957 | 1,503 | 5 | 0 | 817e15162dd5da33 |
| auction_verification | 44958 | 2,043 | 7 | 2 | a7dd7929698dbb9b |
| concrete_compressive_strength | 44959 | 1,030 | 8 | 0 | f114808cb45a4723 |
| energy_efficiency | 44960 | 768 | 8 | 0 | 6223c6b6405c3139 |
| forest_fires | 44962 | 517 | 12 | 0 | d6b434cb15d95225 |
| physiochemical_protein | 44963 | 45,730 | 9 | 0 | ff518e68d936f275 |
| superconductivity | 44964 | 21,263 | 81 | 0 | ca4c66d4a080f5d7 |
| geographical_origin_of_music | 44965 | 1,059 | 116 | 0 | bb8e55b57ee78160 |
| solar_flare | 44966 | 1,066 | 10 | 8 | 5721aadf9c2032b1 |
| student_performance_por | 44967 | 649 | 30 | 17 | 985530c7bbe6df09 |
| naval_propulsion_plant | 44969 | 11,934 | 14 | 0 | dd8edc80728474dc |
| QSAR_fish_toxicity | 44970 | 908 | 6 | 0 | ef261ece782acfca |
| white_wine | 44971 | 4,898 | 11 | 0 | 829013f733afb21a |
| red_wine | 44972 | 1,599 | 11 | 0 | e78b9233c00d8515 |
| grid_stability | 44973 | 10,000 | 12 | 0 | 3f2c9fba11d0520d |
| video_transcoding | 44974 | 68,784 | 18 | 2 | 2debc92e4971ece1 |
| wave_energy | 44975 | 72,000 | 48 | 0 | 88f125172d9088a2 |
| sarcos | 44976 | 48,933 | 21 | 0 | 6fdc585acd43ae37 |
| california_housing | 44977 | 20,640 | 8 | 0 | e3362fdc49f86b6a |
| cpu_activity | 44978 | 8,192 | 21 | 0 | 4d47407cf353bf76 |
| diamonds | 44979 | 53,940 | 9 | 3 | b3192abac78b00d5 |
| kin8nm | 44980 | 8,192 | 8 | 0 | 3822d8bed53b34a0 |
| pumadyn32nh | 44981 | 8,192 | 32 | 0 | efd5b03ac25b0e33 |
| miami_housing | 44983 | 13,932 | 15 | 0 | 95573d2ac6597352 |
| cps88wages | 44984 | 28,155 | 6 | 4 | d3d5776e48b2baf3 |
| socmob | 44987 | 1,156 | 5 | 4 | ddc242724be0af95 |
| kings_county | 44989 | 21,613 | 21 | 4 | 71c80ddcbaddd12d |
| brazilian_houses | 44990 | 10,692 | 9 | 4 | da6006ca0b44eedd |
| fps_benchmark | 44992 | 24,624 | 43 | 13 | cfa687a08b13bea7 |
| health_insurance | 44993 | 22,272 | 11 | 7 | 72e15471cc8885fa |
| cars | 44994 | 804 | 17 | 0 | ebe63c1cf6d2a779 |
| fifa | 45012 | 19,178 | 28 | 1 | 79ac7ce7cd32947c |
| space_ga | 45402 | 3,107 | 6 | 0 | e18a83b1e5546b60 |

Scored on **CRPS** (from 99 predicted quantiles, on a fixed grid, since a coarser grid
underestimates it), RMSE, MAE and R², all on targets standardized by the training fold — so a
paired test across datasets is not dominated by whichever has the largest units.

**The metrics measure what they should** (CPU, 2 folds, 2 datasets):

| model | energy_efficiency CRPS / R² | QSAR_fish_toxicity CRPS / R² |
|---|---|---|
| untrained tiny joint model | 0.761 / −0.004 | 0.653 / −0.010 |
| released v2 regressor | **0.021 / 0.998** | **0.303 / 0.648** |

An untrained model sits at R² ≈ 0 and RMSE ≈ 1 in standardized units, as it must; a good one
is far below on CRPS. Not a benchmark — a check that the scale is right before paying for one.

#### The experiment: `scripts/tp07_run.sh`

Three proxy arms, one GPU each, same host: **clf_control**, **reg_control**, **joint**. The
evaluator scores each checkpoint on the suites its own config can do — the joint one on both
`cc18_narrow` and `ctr23` under one config_id — and `bm06_analyze` reads each task against
that task's own control (`--metric crps` for regression).

- **Equal steps; the joint arm splits each 64-dataset step 32/32.** It sees half as many
  datasets of each task as that task's control, and the pair of controls costs twice the
  joint arm. That is exactly the trade TP-07 claims is free, so it is the one tested.
- **All three arms use the classifier's LayerNorm with biases,** the regression control
  included, although the released regressor was trained bias-free. The question is
  joint-vs-single-task; a control with a different norm would differ in two ways at once.
- **Pass:** joint not worse than either control (no OPPOSITE SIGN at the final checkpoints).
  Parity at half the compute is the claim, so "not separated" passes.
- **Cost:** stage 1 runs 2.65 s/step on one RTX PRO 6000, so 20,000 steps ≈ 15 h per arm,
  **≈ 44 GPU-hours per seed** plus evaluation — ≈ $90 at the measured $2.09/GPU-hour, ≈ $280
  for the three seeds the gate asks for.
- **If it fails:** re-run with `SHARED_EXTRA="--qk_norm True --input_norm True"` in all three
  arms — TP-04, the report's stated fix for joint-training stability — before tuning the loss
  weight. `--input_norm` (the report's LayerNorm after the input encoding) is new in this
  commit; the other norm the report adds, before the task head, already exists here as the
  LayerNorm that `norm_first` puts before the decoder.

### TP-05 — grouped-query attention: ~7.6x smaller cache, measured, no GPU needed (2026-09-21)

**Implemented and measured for $0.** TP-05's headline benefit is memory, not accuracy, and
memory is an architectural fact — it can be measured on an untrained model. Only the
accuracy-neutrality needs a training run, which has not been done.

Keys and values in the in-context transformer are projected to `icl_num_kv_heads` heads
instead of `icl_nhead`. That smaller tensor is what gets **cached**; it is expanded back to
full width immediately before the attention call, so the arithmetic is unchanged and only the
stored tensors shrink. Off by default — separate q/kv projections replace the packed one, so
checkpoints are not interchangeable.

#### The cache has two parts, and only one of them scales with rows

This corrects a sloppy reading of the earlier BM-02 result. Measured at 100 features:

| Component | Size | Scales with |
|---|---|---|
| column cache | 40.9 MB | **features only** — it holds inducing points, so it is fixed in rows |
| ICL cache | 49 KiB/row (fp32) | **rows** — 12 blocks x 2 x 512 dim, = 24 KiB/row/estimator in fp16 |

That per-row figure is exactly the 24 KiB/row/estimator measured on the released model on a
Blackwell, so the two measurements agree.

GQA shrinks the ICL part by `nhead / num_kv_heads` and leaves the column part alone, so the
**total** reduction depends on how the two compare — which depends on training-set size:

| n_train (100 features) | cache | with `icl_num_kv_heads=1` | reduction |
|---|---|---|---|
| 512 | 66 MB | 44 MB | 1.50x |
| 2,048 | 142 MB | 53 MB | 2.65x |
| 8,192 | 444 MB | 91 MB | 4.86x |
| 100,000 *(extrapolated)* | 4.6 GiB | 0.6 GiB | **7.56x** |
| 500,000 *(extrapolated)* | 22.9 GiB | 2.9 GiB | **7.91x** |

At small training sets the fixed column cache dominates and the reduction looks poor; at the
sizes that actually strain memory it approaches the full 8x. Anyone benchmarking this on a
toy dataset would conclude it barely helps.

#### It also removes 20% of the parameters

27,552,258 -> 22,036,482 at `icl_num_kv_heads=1`. Separate q and kv projections cost less than
the packed qkv they replace. This only holds because the now-unused packed `in_proj_weight`
that `nn.MultiheadAttention` allocates is explicitly deleted; left in place it was never read
and still added 4M parameters, turning a 5.5M saving into a 4M cost.

#### What remains

Accuracy is untested — that needs a training run. But the practical consequence is already
concrete: it lifts the `kv_cache="kv"` ceiling from ~400-500K training rows on a 96 GB card to
roughly 3-4M, and it is the prerequisite for TP-06, which would otherwise double the cache.

### Full-curriculum cost, measured per stage (2026-09-21)

**$1.77.** One RTX PRO 6000 Blackwell, each stage's exact config from
`scripts/train_v2_clf_stage{1,2,3}.sh`, run on the same pod so the ratios compare like with
like. Steady state excludes the first step, which carries prior warm-up.

| Stage | Steps | Rows | s/step | vs stage 1 | my estimate | 
|---|---|---|---|---|---|
| 1 | 500,000 | 1,024 | **2.65** | 1x | — |
| 2 | 40,000 | 400–10,240 | **11.8** | **4.5x** | 2.5x |
| 3 | 10,000 | 400–60,000 | **74** | **28x** | 12x |

**Both estimates were low**, and stage 3's was outside the ±2x band I put on it. No OOM at
60,000 rows on 96 GB, so `--recompute` is not needed on this card.

#### What a full classifier checkpoint costs

| | Stage 1 | Stage 2 | Stage 3 | **Total** | **Cost** |
|---|---|---|---|---|---|
| **1 GPU** | 368 h | 131 h | 206 h | **705 h, 29 days** | **$1,473** |
| **4 GPUs** | 171 h | 61 h* | 96 h* | **328 h, 14 days** | **$2,742** |

\* Stages 2–3 on 4 GPUs assume the same 2.15x DDP speedup measured for stage 1. That is
probably conservative — longer sequences do more work per synchronisation — but it is not
measured.

The pair (classifier + regressor) is double — **and TP-07 does not change that.** This line
previously read "unless TP-07's multitask checkpoint lands", which implied the joint model
halves the bill. It does not. TP-07's own result (d62bc16) finds no task interference, but a
joint run matched to the pair *on data per task* is also matched on total compute, so it costs
what the pair costs. What TP-07 saves is **one checkpoint instead of two** — 29.10M parameters
rather than 56.1M, and one run to babysit rather than two — not half the money.

So a full classifier+regressor capability is **~$2,950 on one GPU or ~$5,480 on four**,
whichever way it is trained.

#### Three corrections to what I said before measuring

1. **Total is ~$1,470–2,740, not ~$1,250–2,100.**
2. **Stages 2 and 3 are half the run, not a small tail.** At one GPU they are 337 of 705
   hours. Stage 3's 10,000 steps alone take 56% as long as stage 1's 500,000.
3. **Shortening stage 1 to 280K saves ~23%, not ~45%.** The paper's ablation length gives
   543 h / $1,135 at one GPU — the saving was overstated because the tail was underestimated.

#### FA3 measurement attempt — FAILED, $9.10, no result (2026-09-21)

Goal: stage 3 on one H100 SXM (sm_90, where FA3 is fully supported), FA3 off vs on, same card.
**The FA3 source build never finished.** 224 cores, 1.9 TB RAM, `MAX_JOBS=112`, and it was
still compiling when the pod's 2.5 h hard deadline terminated it. No timing was taken.

Cause: FA3 instantiates a kernel for every combination of head dim, dtype and feature, and
the build compiled out only FP8 and the SM80 path. `scripts/cloud/build_fa3.sh` now compiles
**only what tabicl calls** — backward, varlen, fp16, head dim 64 — switches off split, paged-KV,
append-KV, local, softcap, pack-GQA, cluster and head dims 96–256, reports any flag the
checkout does not recognise instead of ignoring it, and builds a **wheel** that can be pulled
back and reinstalled via `FA3_WHEEL`, so the compile is paid for once.

Two bugs were fixed on the way, both of which would have made a successful measurement lie:
`INSTALL_FA3` installed FA2 (`flash-attn`) when the code imports FA3 (`flash_attn_interface`);
and `flash-attn-3` on PyPI is a 0.0.0 pure-Python wheel, not FA3. The runpod image also keeps
`nvcc` at `/usr/local/cuda/bin` off the PATH.

What I got wrong: I called this "another few dollars" and sized the deadline for the
measurement, not for an uncharted multi-hour compile. The deadline did its job — it bounded
the loss — but the estimate was wrong.

#### The one thing that could move it most: FlashAttention-3

These numbers are **without FA3**. The reference recipe enables it for stages 2 and 3 — exactly
where 10K–60K-row sequences make attention dominate — and it is not installed here, while
Blackwell's (sm_120) support for it is partial. On an H100 (sm_90), where FA3 is fully
supported, stages 2–3 could be substantially cheaper even at a similar hourly rate. **Before
paying for a full run, measure stage 3 on an H100 with FA3** — another few dollars, and it
bears on half the bill.

### TP-06 — width scaling: the report's central trade, reproduced for $0 (2026-09-21)

TabPFN-3.5's headline architectural claim is that they **doubled the model width while the
cache size stayed largely unchanged**. That is an architectural fact, so it can be checked
without training anything. It reproduces here — and slightly better than their framing,
because with a single KV head the cache does not merely stay flat, it falls.

Measured at 2,048 training rows and 100 features:

| Config | params | ICL cache | total cache |
|---|---|---|---|
| baseline (512 dim, 8 heads) | 27.6M | 48 KiB/row | 142 MB |
| **TP-06 alone** (1024 dim, 16 heads) | **105.1M** (3.8x) | **96 KiB/row** (2x) | 244 MB |
| **TP-06 + TP-05** (1 KV head) | **81.5M** (3.0x) | **6 KiB/row** (0.125x) | 55 MB |
| baseline + TP-05 | 22.0M | 6 KiB/row | 54 MB |

The 3.8x parameter growth from widening alone matches the report's "roughly quadrupled".

**Why widening becomes free.** With one KV head the cached tensor is
`blocks x 2 x head_dim`, and head_dim is held at 64 as width grows — so the cache stops
depending on width at all. That is the whole trick, and it is why TP-05 gates TP-06 rather
than being an optional extra. Two tests pin it: without GQA, doubling width doubles the cache;
with GQA, doubling width leaves it unchanged.

**Cost of the width.** Forward pass 1.53x baseline on CPU for width alone, 1.37x with GQA —
consistent with the report's point that inference time is linear in width rather than in
parameter count, so 3x the parameters does not cost 3x the time. All three configs were
confirmed to train: real forward, backward and optimizer step, finite gradients.

**Net:** 3x the parameters, 1.37x the forward time, and **8x less cache per row** than today.

#### What is still unknown

Whether the wider model is actually *better*. That needs a full training run, and
**CAVEAT-01 applies with full force** — the TabICLv2 authors' own depth ablation showed no
gain at 280K steps *because the larger model had not converged*. A proxy-scale screen of
TP-06 would be biased against it and should not be run as a go/no-go.

### TP-01 + TP-08 joint screen — RESULT: still no benefit, now on the right data (2026-09-21)

**Second null, and this time the objections to the first one were removed.** Control vs
`--col_fourier_value True`, **both arms on the high-cardinality prior**
(`--prior_cat_prob 0.5 --prior_high_card_prob 0.4`) so only the encoder differs, 10,000
steps, one GPU per arm, seed 42. Scored on **both** suites. **$30.93**, 7 h.

**High-cardinality suite (14 datasets) — the primary readout:**

| Step | control ll | fourier ll | fourier worse on | p (adj) |
|---|---|---|---|---|
| 2,500 | 0.3838 | 0.3833 | 47.6% | 1.00 |
| 5,000 | 0.3785 | 0.3813 | 64.3% | 1.00 |
| 7,500 | 0.3763 | 0.3793 | 66.7% | 1.00 |
| **10,000** | **0.3748** | **0.3786** | **71.4%** | **1.00** |

**CC18 (62 datasets):** 0.3003 vs 0.2991 at step 10,000, worse on 54.8%, not separated.

On the suite where the mechanism is supposed to pay off, the Fourier arm is **slightly worse**,
and the trend grows with training rather than shrinking (47.6% -> 71.4%). Uncorrected, the
negative direction at step 10,000 is p ~ 0.02; after correction across four checkpoints it is
not significant, and the suite's effective n is nearer 8 than 14. So: not evidence of harm,
but certainly no evidence of benefit.

#### Why this null is worth more than the first one

The first screen had two defensible objections — the prior never generated high-cardinality
columns, and the evaluation set contained none. Both were removed here: the prior was verified
on-pod to produce columns with >=50 levels in 43 of 86 discrete columns, and the evaluation
includes datasets with up to 15,415 levels. The result did not change.

#### What could still explain it

1. **Length.** 10K steps against TabPFN's 500K-plus. CAVEAT-01 applies most strongly to a
   representation change, and this one starts *behind* in both screens before catching up —
   the signature of something that needs time. This remains the leading benign explanation.
2. **A regime gap that survives.** Training uses `max_seq_len 1024`, so the prior's
   high-cardinality columns are capped at ~256 levels, while the evaluation data reaches
   15,415. The encoder is still not trained on the regime it is tested on.
3. **It may simply matter less here than in TabPFN.** TabICL's `ColEmbedding` is a set
   transformer over each column, so it already sees the column's distribution; TabPFN-3's
   cell encoder is a plain per-cell projection. The marginal value of Fourier features should
   be *lower* against a distribution-aware embedder. This was flagged as a risk for TP-02 at
   the start of the backlog and applies just as well to TP-01.

#### Reading

Two independent screens, one with every stated objection removed, show no benefit. **Stop
screening TP-01 at proxy scale.** It is not worth more money at this length; the open question
is whether it pays off at full scale, and that is a several-hundred-dollar question, not a
$30 one. The code stays, off by default, with this result recorded next to it.

### TP-01 — RESULT: no benefit at proxy scale, but the suite cannot test the claim (2026-09-20)

**Null, and partly my own fault for screening it alone.** Control vs `--col_fourier_value True`,
20,000 steps, 4 GPUs per arm, seed 42, same host/card. **$113.90**, 6.5 h. Checkpoints pulled
this time (BM-09), so this control is reusable.

`sick` now evaluates, so this run has **62 datasets** against BM-06's 61 — BUG-02's fix
shipped in this payload.

| Step | control ll | TP-01 ll | control acc | TP-01 acc | separated? |
|---|---|---|---|---|---|
| 2,500 | 0.3247 | 0.3385 | 0.8578 | 0.8529 | **TP-01 worse** (p=0.001) |
| 5,000 | 0.3140 | 0.3133 | 0.8615 | 0.8628 | no |
| 10,000 | 0.3061 | 0.3015 | 0.8645 | 0.8669 | no |
| 15,000 | 0.2968 | 0.2968 | 0.8689 | 0.8687 | no |
| **20,000** | **0.2946** | **0.2947** | **0.8699** | **0.8695** | **no** |

TP-01 starts **behind** — it has to learn its frequency bank — draws level by 5,000, runs
marginally ahead between 7,500 and 12,500, and finishes indistinguishable.

#### The suite cannot test what this change is for

The report credits Fourier encoding specifically for *"ordinal-encoded categorical variables
with high cardinality"*. Sliced at step 20,000 by maximum categorical cardinality:

| Slice | n | control ll | TP-01 ll | TP-01 better on | p |
|---|---|---|---|---|---|
| all | 62 | 0.2946 | 0.2947 | 40.3% | 0.81 |
| has categoricals | 22 | 0.3713 | 0.3688 | 45.5% | 0.38 |
| **max cardinality >= 10** | **9** | **0.3715** | **0.3672** | **66.7%** | **0.18** |
| max cardinality >= 20 | 4 | 0.5235 | 0.5170 | 50.0% | — |
| purely numeric | 40 | 0.2524 | 0.2539 | 37.5% | 0.92 |

The direction is right where the mechanism predicts (66.7% on cardinality >= 10, and nothing
or slightly negative on purely numeric data) — but **n = 9 and p = 0.18**. That is a hint, not
evidence.

The deeper problem is that **CC18 has no high-cardinality categorical data**. The most
extreme column in all 62 datasets is `cylinder-bands` at 71 levels; only 9 datasets reach 10
levels. TabPFN's claim is about hundreds to thousands of levels. **The evaluation set does not
contain the regime the change targets**, so this run could not have detected the effect even
if it is real.

#### The screen was also mis-scoped, by my own earlier note

TODO.md already said of TP-08: *"Ship with TP-01 — ordinal codes are exactly the case where
value resolution matters, and the report is explicit that they tuned the prior toward high
cardinality **in conjunction with** the encoding change."* I then screened TP-01 on its own.
The prior currently generates categoricals with a mean of ~10 levels
(`gammavariate(1, 10)`), so during training the encoder almost never saw the input
distribution it exists to resolve. Testing the pair jointly is what the report's own account
implies.

#### Reading

**Not "Fourier encoding does not work". "Not demonstrated, on a suite and a prior that both
lack the data regime it targets."** Three things must change before a rerun is worth paying
for: a high-cardinality evaluation slice, TP-08 in the prior, and the two screened together.

Cost of learning this: $113.90 — and it is worth noting the null was cheap precisely because
BM-06 had already established that the proxy can detect a real effect, so "no signal" here
means something rather than nothing.

### BM-06 — RESULT: the proxy reproduces the published sign (2026-09-20)

**PASS.** Control (Muon) against AdamW, 20,000 steps, 4 GPUs per arm on one 8-GPU
RTX PRO 6000 Blackwell pod, evaluated on `cc18_narrow` every 2,500 steps.
**$116.94**, 6.9 h wall clock, pod terminated by the watcher after results were verified.

| Step | Muon acc | AdamW acc | Muon log-loss | AdamW log-loss | AdamW worse on | p (Bonferroni) |
|---|---|---|---|---|---|---|
| 2,500 | 0.8558 | 0.8330 | 0.3292 | 0.3895 | 92.9% | ~0 |
| 5,000 | 0.8577 | 0.8422 | 0.3208 | 0.3633 | 86.3% | ~0 |
| 7,500 | 0.8619 | 0.8455 | 0.3143 | 0.3535 | 90.2% | ~0 |
| 10,000 | 0.8644 | 0.8477 | 0.3079 | 0.3466 | 89.1% | ~0 |
| 12,500 | 0.8656 | 0.8508 | 0.3052 | 0.3393 | 84.7% | ~0 |
| 15,000 | 0.8660 | 0.8528 | 0.3025 | 0.3334 | 85.8% | ~0 |
| 17,500 | 0.8667 | 0.8538 | 0.3000 | 0.3326 | 85.8% | ~0 |
| **20,000** | **0.8673** | **0.8539** | **0.2995** | **0.3320** | **86.3%** | **~0** |

The sign agrees at every checkpoint, on 61 datasets (`sick` excluded by BUG-02, identically
for both arms). A short proxy run **can** detect an effect the paper measured at 280K steps,
in the right direction.

#### Three things the curve says that the plan did not anticipate

**1. The gap does not converge toward the published 64%.** It falls once, 92.9% -> 86.3%, then
plateaus between 84.7% and 90.2% for the remaining six checkpoints. The prediction recorded
earlier — that this was purely Muon converging faster, and would decay toward 64% — is only
right about the first step. Either the decay is far slower than implied (the log-loss gap is
still drifting down, 0.060 -> 0.033) or the proxy genuinely magnifies this effect. **Treat
proxy effect *sizes* as inflated; trust only the sign and the ordering.**

**2. The proxy is a closer stand-in than expected.** Control reaches log-loss 0.2995 at 20K
steps against the released checkpoint's 0.2798 on the same 61 datasets — within ~7%, from
20K steps versus the full 500K-plus three-stage curriculum.

**3. Screening can start early.** The sign is significant from the very first checkpoint at
2,500 steps and never lapses. Combined with (1), read ablations at the **latest** checkpoint
available, where the exaggeration is smallest.

#### What this re-prices

Measured: **$0.0029 per arm-step** (two 20,000-step arms, 8 GPUs, 6.9 h, $116.94).

| Screen length | Cost per arm |
|---|---|
| 5,000 steps | ~$15 |
| 10,000 steps | ~$29 |
| 20,000 steps | ~$58 |

Phase 2 was priced at ~$520 when a 25,000-step run on one GPU was the unit. Screening TP-01,
TP-02 and TP-08 against a shared control at 10,000 steps is **~$115**, inside the current
balance.

#### A mistake worth recording: the checkpoints were destroyed

`pod_finish.py` pulled the ledger, logs and `pod_doctor.json`, verified them, then terminated
— which destroyed the volume holding every checkpoint, including the control arm's. The
measurement survived, and that was the stated deliverable, but a *reusable control* did not.
Future comparisons must now either re-run their own control (~$29 at 10K steps) or
regenerate this one once.

Ablation checkpoints should be pulled, or the pod stopped rather than terminated, whenever
the trained weights themselves have onward value. Recorded as **BM-09**.

#### What this does NOT establish

Muon's advantage is substantially about convergence *speed*, which a short run flatters — it
was chosen as the most favourable available test, so passing is necessary, not sufficient.
Nothing here shows the proxy correctly ranks a change that learns slowly and finishes better,
which is exactly what **TP-06** (doubled width) and possibly **TP-01** may be. **CAVEAT-01
stands**, and point (1) above strengthens it. One seed, one of three published effects,
classification only, stage 1 only.

### The calibration was not launched, and why (2026-09-19)

Before spending on BM-06 I read TabICLv2's own ablation section (arXiv 2602.11139, §7 and
Appendix C) to get the reference signs the calibration is supposed to agree with. It changed
the plan in four ways, one of which is a correction of my own work.

**1. One of my four calibration knobs had no reference at all.** The plan listed
`--row_rope_interleaved` as "a knob whose full-scale effect is already known from the
TabICLv2 paper". It is not. RoPE interleaving is never ablated in the paper — I asserted a
reference that does not exist. Removed. The published effects that *do* exist:

| Component | Published effect | Reference model |
|---|---|---|
| Prior (v2 vs v1) | **Largest effect.** v2 architecture on the v1 prior *fails* — below TabICL, validation loss degrades in the second half | — |
| Early target inclusion (`--col_target_aware`) | ~100 Elo, ~64% win rate | added to reference |
| Muon instead of AdamW (`--muon`) | ~100 Elo, ~64% win rate | AdamW at lr 1e-4, regular weight decay |
| QASSMax (`--col_ssmax`/`--icl_ssmax`) | ~100 Elo, ~64% win rate | reference has none |
| Repeated feature grouping, prior filtering | smaller gains | |
| Gaussian noise on prior edges | negligible | |
| Deeper model (4/4/18 layers) | no clear gain — "likely due to insufficient pretraining for the larger model to fully converge" | |

**2. Those effects were measured at 280,000 steps, not 5,000.** The authors' ablations are
56% of a full Stage 1 — 56x longer than the proxy the plan proposed — and they note that
per-step validation "noise decreases as the learning rate decays", i.e. early training is
the noisiest part to read.

**3. Our evaluation suite cannot see effects of this size.** The paper evaluates on 60
datasets x 2 splits. Ours has 10, of which 2 are saturated (BM-07), leaving 8. For a true
64% win rate:

| Datasets | Wins needed for p < 0.05 | True 64% effect clears it |
|---|---|---|
| **8 (ours)** | **7** | **15% of the time** |
| 20 | 15 | 22% |
| 40 | 26 | 52% |
| 60 (paper) | 37 | 70% |

A sign test is the least powerful choice — paired tests on continuous log-loss differences
do better — but the order of magnitude is not in doubt. **With 8 datasets, 85% of real
~100-Elo effects would read as nothing.** Running the calibration on this suite could not
distinguish "the proxy works" from "the proxy is noise", which is the only question it exists
to answer.

**4. A proxy will systematically *under*-estimate capacity changes.** The depth ablation found
no clear gain *because the larger model had not converged* at 280K steps. A 5K-step proxy is
far more convergence-starved than that. So a proxy result for **TP-06** (doubled width) is
not merely noisy — it is **biased against** the change. A proxy "no" on TP-06 must not be
read as a real "no".

#### What changes

- **BM-03 must grow to ~40–60 datasets before BM-06 runs.** That is the new gate, and it is
  CPU work. Recorded as **BM-08**.
- The calibration knobs become **target-aware, Muon, and QASSMax** — the three with a
  published ~100-Elo reference — plus the control. RoPE is dropped.
- The AdamW arm must use **lr 1e-4 with regular weight decay**, per the paper, not the Muon
  learning rate of 8e-4.
- Read results with a **paired test on log-loss**, not a sign test on accuracy.
- Phase 3 capacity items get a **standing caveat**: a proxy can reject them wrongly.

Cost of discovering all of this: **$0.03** (one broken community pod) and reading a paper.
The calibration as specified would have cost $35–80 and returned an uninterpretable answer.

### Two bugs found by running it

- **`NJOBS = cpu_count - 2` is catastrophic on a large pod.** 128 vCPUs gave 126 prior-worker
  processes, each opening a 64-thread OpenBLAS pool; the run died with
  `pthread_create failed ... Resource temporarily unavailable` before step 1. The visible
  error is a DataLoader worker "killed by signal: Interrupt", which points nowhere near the
  cause. Fixed in `scripts/train_proxy.sh`: cap at 32 workers and pin
  `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
- **`micro_batch_size` cannot exceed `batch_size_per_gp` when `seq_len_per_gp=True`**, because
  every dataset in a micro-batch must share a training size
  (`ValueError: All datasets in the micro batch must have the same training size`). Raising one
  without the other fails at step 0. Undocumented; worth a line in the training CLI help.

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

### TP-14 ScoringBench — three hypotheses tested, three negatives

**The premise was probably wrong, and that is the finding.**

I called the published `TabICL v2 (finetuned)` mean rank of 12.05 "an outlier relative to
every other board", "plausibly a harness/config issue rather than a model deficiency", and
"the cheapest potential win in the whole backlog". Having read their code and tested the
three mechanisms that could produce it, **none of them do**. That characterization was too
strong and is withdrawn.

Source: <https://github.com/jonaslandsgesell/ScoringBench>, wrapper at
`scoringbench/univariate/wrappers/tabicl.py`, config at `univariate/config.py`
(seed 42, 5 folds, 3,000-row cap — matching the report). All experiments below use their CV
config and their 200-level alpha grid.

| # | Hypothesis | Mechanism | Result |
|---|---|---|---|
| 1 | Wrapper uses the post-hoc calibrated path | It calls `output_type="quantiles"`; our changelog says `raw_quantiles` are "direct outputs … without post-hoc calibration" | **Identical.** 3 datasets x 2 folds: `wine_quality` 0.2913 vs 0.2913, `boston` 1.1108 vs 1.1118, `kin8nm` 0.0407 vs 0.0407. Under 0.1% apart. |
| 2 | Finetuning degrades calibration | The published entry is the *finetuned* variant (`epochs=80`), not base TabICL | **Neutral.** `boston`, 2 folds: base 1.1108, finetuned 1.1087 (0.2% *better*). |
| 3 | Selection metric mismatched to scoring | `models.py` sets `eval_metric="mse"` — a point metric — while the benchmark scores CRPS | **Neutral.** `boston`, 3 folds: select-on-mse 1.2327, select-on-crps 1.2358 (0.25% *worse*). |

**Caveat, stated plainly:** hypotheses 2 and 3 were tested on `boston` (506 rows) across 2-3
folds. That is underpowered and cannot exclude a small effect. It can exclude the *large*
effect that would be needed to move a mean rank from ~7.6 to 12.05, which is what was being
claimed.

#### What the number probably means

12.05 is a mean rank among **53 methods** on a board densely populated with purpose-built
probabilistic regressors — NGBoost, XGBLSS, BART, normalizing flows, conditional density
estimators, conformal wrappers. TabPFN-3 (7.64) and EXAONE-Tabular (7.59) beat us;
Nori-30M (13.09) does not. Mid-upper-pack on a benchmark that scores *only* the predictive
distribution is a plausible honest result for a model whose headline strength is point
accuracy. It is not obviously an artifact, and I no longer think it is one.

**What would settle it:** running their harness over the real 101 datasets and checking we
reproduce ~12.05. If we do, the item closes as "accurate, and a genuine weakness to work on".
Their dataset list is built lazily from OpenML suites and needs the `openml` package. That is
the remaining work on TP-14, and it is now a verification task, not a bug hunt.

#### One real gap, found and closed

`FinetunedTabICLRegressor` **trains** against pinball loss over the raw quantile head
(`_compute_batch_loss`) but only offered `{"mse", "mae", "r2"}` for early stopping — all point
metrics. Selection and objective disagreed by construction. Added `eval_metric="crps"` so they
can agree, with the CRPS helper pinned against the closed form for a standard normal.

It showed **no measurable benefit** in the one A/B above. It is kept because selecting on the
metric you trained against is principled and the option costs nothing — not because it was
shown to win. Do not cite it as a fix.

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
| TP-07 seed42, 5K steps, equal steps | `61c9d39` | clf_control / reg_control | log-loss 0.3108 | CRPS 0.2039 | $22.22, 10.4 h, 3 arms on one RTX PRO 6000 | n/a | control |
| TP-07 seed42, 5K steps, equal steps | `61c9d39` | joint (TP-07) | log-loss 0.3156 | CRPS 0.2141 | shared with the row above | **worse on both** (p 4.1e-05 / 1.9e-07) | at equal steps the joint arm sees half the data per task |
| TP-07 seed42, equal data per task = equal total compute | `61c9d39` | joint @5000 vs control @2500 | 0.3156 vs 0.3213 | 0.2141 vs 0.2149 | same total compute as the pair | **clf better** (p 0.0016), reg a wash (p 0.41) | no task interference; the saving is one checkpoint, not half the compute |

Not yet run. Launch with `scripts/tp07_run.sh` (three arms, one GPU each); design, pass rule
and cost are in the TP-07 entry above. Each task has its own control — `clf_control` on
`cc18_narrow`, `reg_control` on `ctr23` — so fill one row per task, per seed.

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
