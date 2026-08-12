# Why this package exists and not a dependency

The question this answers: **`scaling/` is ~15k lines. Which of it could have been a library
call, and how do we know?**

The short answer is that one part has been measured against the standard library for the job
and earned its place, one part is ruled out by a signature rather than a benchmark, and one
part — most of the line count — rests on an argument nobody has tested. Those three are
different kinds of claim and are labelled as such throughout. **A rejection nobody measured is
a preference, and it is written here as a preference.**

| evidence | what it means | which alternatives |
|---|---|---|
| **measured** | run head-to-head, same model, same seeds, only the component swapped | Featuretools / DFS |
| **read** | rejected from its API, because the required operation is absent | skrub |
| **not evaluated** | never tried; listed so the gap is visible | getml, RDBLearn as a dependency |

---

## 1. Featuretools (DFS) — MEASURED, and it earned the build

Deep Feature Synthesis (Kanter & Veeramachaneni, IEEE DSAA 2015) is the standard automated
approach to exactly our problem: apply aggregation primitives across foreign-key paths,
recursively, to some depth. `flatten_relational` does the same thing. The overlap is not
incidental and pretending otherwise would be dishonest — see `STATUS.md`, *Relation to Deep
Feature Synthesis*, for the row-by-row comparison.

**The comparison was built to be able to lose.** From `eval_dfs_baseline.py`: *"Ours wins and
the engineering is justified. Or DFS wins, and the honest framing becomes 'Featuretools plus
label features plus a foundation model' — still a result, and a far cheaper thing to
maintain."* Same TabICL, same context, same seeds, per-row `cutoff_time` given to DFS so it
respects the same temporal boundary, and the tuned primitive set rather than the default.

| task | DFS | ours | ours − DFS | feature build |
|---|---:|---:|---:|---|
| rel-f1 / driver-top3 | 76.81 | 81.84 | **+5.03** (sd 1.49, 5/5) | 28 s → **1 s** |
| rel-event / user-ignore | 77.95 | 80.34 | **+2.39** (sd 3.16, 4/5) | 119 s → **25 s** |
| rel-trial / study-outcome | 69.12 | 69.56 | +0.45 (sd 0.50, 5/5) | 20 s → **2 s** |
| rel-avito / user-visits | 65.81 | 65.51 | −0.30 (sd 0.52, 2/5) | 414 s → **18 s** |

Two wins, two ties inside the ±0.6 floor, features built 5–23× faster.

**Which of our differences actually paid.** The correctness differences are real but small:
factorized sufficient statistics make a depth-2 mean the *true* mean rather than a mean of
means (DFS's stacked `MEAN(sessions.SUM(txns.amount))` is a different quantity, and the error
is unbounded as group sizes vary), and per-row cutoffs are an `O(n log n)` prefix scan rather
than a per-row re-scan, which is where the 5–23× lives. **But the thing that moved a headline
number is the one DFS deliberately will not do: use the target.** `key_target_history` was
worth **+5.45** on rel-trial, more than every structural lever in this package combined. DFS
never touches the target by design — that is a defensible design choice and it is also the
ceiling.

**Three caveats that keep this from being a bigger claim than it is.** Both sides used three
child tables, so it is DFS-on-3 against ours-on-3, not against the full layer. It ran on the
original four tasks and the benchmark is twelve now. And **rel-avito is the one task where DFS
is ahead**, which is also the task where we sit nearest a published win — those two facts
probably belong together, and neither has been chased.

---

## 2. skrub — READ, and ruled out by a missing parameter

Checked against the stable API on 2026-08-11. The relevant class is `AggJoiner`, and its full
signature is:

```python
skrub.AggJoiner(aux_table, operations, *, key=None, main_key=None,
                aux_key=None, cols=None, suffix='')
```

**There is no `cutoff_time`, no as-of semantics, no time parameter of any kind.** Nor is there
one on `MultiAggJoiner`, `Joiner`, `InterpolationJoiner`, `AggTarget` or `fuzzy_join` — no
join or aggregation primitive in skrub carries temporal semantics.

That single absence is disqualifying here, and not as a matter of taste. Every RelBench task
predicts an entity's future from its past at a per-row cutoff. An aggregation that cannot be
bounded by a timestamp aggregates rows recorded after the prediction time, which is leakage —
so the failure mode is not "skrub would be slower" but "skrub would score higher and be
wrong." Nothing downstream would flag it.

**The near miss is worth recording**, because it is the closest any library gets to the
feature that matters most here. `AggTarget` — *"aggregate a target `y` before joining its
aggregation on a base dataframe"* — is structurally the same idea as `key_target_history`, the
+5.45 on rel-trial. Same idea, and without a cutoff it is the leaky version of it. The thing
that makes target aggregation usable on this benchmark is precisely the parameter skrub does
not have.

**None of this is a knock on skrub**, which is not aimed at temporal relational learning, and
the traffic runs the other way too: upstream PR #139 (merged into this branch) documents
`skrub.tabular_pipeline` calling TabICL as an estimator. Skrub calling our model works; our
pipeline calling skrub's joins does not.

---

## 3. getml, and RDBLearn as a dependency — NOT EVALUATED

Listed so the gap is visible rather than implied away.

**getml** has a relational feature-learning engine (FastProp, RelBoost, and relatives) aimed
at exactly this problem shape. It has never been run here, and nothing in these documents
justifies not running it. If it turns out to handle per-row cutoffs — which is the bar skrub
failed — it is a genuine candidate for the flattening layer and the DFS comparison is the
template for testing it.

**RDBLearn** appears about twenty-five times across `PERFORMANCE.md`, and every one of them is
a benchmark column to be ranked against. Whether its implementation is something we could have
depended on rather than competed with has never been asked. Note the standing comparison is
confounded in a way that makes this worth asking: the whole flatten-then-tabular-foundation-
model peer group — TabPFN-REL, RDBLearn+v3, KumoRFMv2 — **runs on TabPFN-3 while we run on
TabICL**, so our remaining gap to them has never been separated into featurization versus
backbone. Running their featurizer under our model is the experiment that would separate it,
and it is the same design as the DFS comparison.

---

## 4. The harness — ARGUMENT, NOT EVIDENCE

This is the honest weak point, and it is most of the folder: `eval_track_record.py`, the
calibrated selection protocol, the leak controls, the noise floor, the salvage and paired-
comparison tooling, `cycle.py` and the pod runner. **None of it has been compared against an
alternative, because no comparison was ever attempted.** The argument for it is:

* **The leak controls are specific to the claims made here.** The permutation null, the
  temporal shift controls, the arm-eligibility veto, the resolution horizon on target
  features — each exists because a specific feature block could otherwise read the future in a
  specific way. A general-purpose experiment tracker has no opinion about whether
  `n_linked` reaches past a cutoff.
* **The failures encoded are ours.** `cycle.py`'s comment block is a list of things that each
  cost a round — CRLF line endings, a missing `LICENSE`, a stale payload, fetching logs on the
  happy path, `pip | tail -3` eating a build error. That is scar tissue, not architecture, and
  no library would have carried it.
* **The protocol is the object of study.** Several rounds tested the *selection procedure*
  itself (random validation splits, lexicographic selection, gap-matched validation,
  temporal-distance extrapolation). A harness you cannot open is not usable for that.

The first and third are strong. The second is an argument for keeping what exists, not for
having written it. And the counter-argument has never been given a hearing: MLflow or Weights
& Biases plus a thin task layer would have covered the tracking, the paired comparisons and
the artefact plumbing, and roughly none of the domain logic. Nobody costed that.

**What would change this section:** running one round's bookkeeping through an off-the-shelf
tracker and reporting what it could not express. Until someone does, "we built our own harness"
is a preference with reasons, not a measured decision, and it should be read that way.

---

## Summary

| component | alternative | verdict | basis |
|---|---|---|---|
| relational flattening | Featuretools / DFS | **keep ours** — +5.03/+2.39, two ties, 5–23× faster | measured, 4 of 12 tasks |
| relational joins | skrub | **cannot be used** — no cutoff parameter anywhere | read the API |
| target features | skrub `AggTarget` | **cannot be used** — the leaky version of `key_target_history` | read the API |
| flattening | getml | **unknown** | never tried |
| flattening | RDBLearn | **unknown**, and would separate featurization from backbone | never tried |
| benchmark harness | MLflow / W&B + task layer | **unmeasured preference** | argument only |

The strongest honest statement: **the core feature builder has been tested against the
standard tool for the job and won on half the tasks it was tried on, while the largest part of
the package has never been tested against anything.**
