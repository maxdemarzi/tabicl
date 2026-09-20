"""Read BM-06: does the proxy reproduce the published sign of Muon vs AdamW?

TabICLv2 (section 7) reports Muon beating AdamW by ~100 Elo / ~64% win rate at 280K steps.
The proxy is trusted only if it reproduces that SIGN -- AdamW worse -- and this script says
at which checkpoint, if any, it first does so and then keeps doing so.

The test is a paired Wilcoxon signed-rank test on per-dataset log-loss (folds averaged
first), one-sided in the published direction. Log-loss, not accuracy, because accuracy
alone hides calibration and the paper states its ablation ordering holds on log-loss.
Datasets on which every arm ties are dropped at read time (BM-07), and a dataset counts
only if every arm scored it.

**p-values are Bonferroni-corrected across the checkpoints tested.** The question "from
which step does the sign appear?" is asked once per checkpoint, so with eight checkpoints one
will likely cross p < 0.05 by chance -- and because every later checkpoint genuinely
separates, the uncorrected rule reports that chance hit as the answer and overstates how
early the proxy becomes usable. That exact failure was caught on a synthetic ledger with no
effect at the first step. The published effect is large enough (~100 Elo, 60+ datasets)
that correction costs little power where it matters.

    python -m benchmarks.bm06_analyze benchmarks/_results/bm06/bm06.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from ._core.aggregate import Record, drop_uninformative, records_from_ledger, win_rate
from ._core.schema import Ledger

STEP_RE = re.compile(r"^(?P<arm>.+)-step(?P<step>\d+)$")


def per_dataset(records, method):
    acc = defaultdict(list)
    for r in records:
        if r.method == method:
            acc[r.dataset].append(r.value)
    return {d: float(np.mean(v)) for d, v in acc.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--control", default="control")
    ap.add_argument("--treatment", default="adamw")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--expect", choices=("worse", "better"), default="worse",
                    help="Direction the published result predicts for the TREATMENT. BM-06 asked "
                         "whether AdamW is worse; an improvement like TP-01 is expected better, and "
                         "reporting that as an 'opposite sign' would invert the conclusion.")
    ap.add_argument("--slice", default=None,
                    help="Restrict to datasets whose cc18_narrow slices include this tag")
    args = ap.parse_args(argv)

    rows = list(Ledger(Path(args.ledger)))
    ll = records_from_ledger(rows, metric="log_loss")
    acc = records_from_ledger(rows, metric="accuracy")
    if not ll:
        print("no log_loss rows yet")
        return 1

    steps = sorted({int(m["step"]) for r in ll if (m := STEP_RE.match(r.method))})
    anchor = "released-v2" if any(r.method == "released-v2" for r in ll) else None

    print(f"ledger: {args.ledger}")
    if args.slice:
        from ._core.datasets import SUITES
        keep = {d.name for d in SUITES["cc18_narrow"] if args.slice in d.slices}
        ll = [r for r in ll if r.dataset in keep]
        acc = [r for r in acc if r.dataset in keep]
        print(f"slice: {args.slice} ({len(keep)} datasets in suite)")
    want_worse = args.expect == "worse"
    print(f"question: is {args.treatment} {'WORSE' if want_worse else 'BETTER'} than "
          f"{args.control}?  (expected: yes)\n")

    hdr = (f"{'step':>6} {'n':>3} {'ctrl acc':>9} {'trt acc':>8} {'ctrl ll':>8} {'trt ll':>8} "
           f"{'trt worse':>10} {'p raw':>8} {'p adj':>8}  verdict")  # 'trt worse' = share of datasets the control wins
    print(hdr)
    print("-" * len(hdr))

    first_agree = None
    verdicts = []
    n_tests = max(len(steps), 1)
    for step in steps:
        c, t = f"{args.control}-step{step}", f"{args.treatment}-step{step}"
        sub = [r for r in ll if r.method in (c, t)]
        sub, _dropped = drop_uninformative(sub)
        dc, dt = per_dataset(sub, c), per_dataset(sub, t)
        common = sorted(set(dc) & set(dt))
        if len(common) < 5:
            print(f"{step:>6} {len(common):>3}  (too few datasets scored by both arms yet)")
            continue

        diff = np.array([dt[d] - dc[d] for d in common])  # >0 means treatment worse
        hyp = "greater" if want_worse else "less"   # diff = treatment - control, on log-loss
        opp = "less" if want_worse else "greater"
        try:
            p_raw = stats.wilcoxon(diff, alternative=hyp).pvalue
            p_opp = min(1.0, stats.wilcoxon(diff, alternative=opp).pvalue * n_tests)
        except ValueError:
            p_raw = p_opp = float("nan")
        p = min(1.0, p_raw * n_tests)  # Bonferroni across checkpoints
        worse = win_rate(sub, c, t, higher_is_better=False)  # share of datasets control wins

        ac = per_dataset([r for r in acc if r.method == c], c)
        at = per_dataset([r for r in acc if r.method == t], t)
        agree = p < args.alpha
        opposite = p_opp < args.alpha
        verdict = ("SIGN AGREES" if agree else "OPPOSITE SIGN" if opposite else "not separated")
        verdicts.append((step, agree, opposite))
        if agree and first_agree is None:
            first_agree = step
        print(f"{step:>6} {len(common):>3} {np.mean([ac[d] for d in common if d in ac]):>9.4f} "
              f"{np.mean([at[d] for d in common if d in at]):>8.4f} "
              f"{np.mean([dc[d] for d in common]):>8.4f} {np.mean([dt[d] for d in common]):>8.4f} "
              f"{worse:>9.1%} {p_raw:>8.4f} {p:>8.4f}  {verdict}")

    if anchor:
        da = per_dataset([r for r in ll if r.method == anchor], anchor)
        last = steps[-1] if steps else None
        if last is not None:
            dc = per_dataset([r for r in ll if r.method == f"{args.control}-step{last}"],
                             f"{args.control}-step{last}")
            common = sorted(set(da) & set(dc))
            if common:
                print(f"\nreference: released v2 log-loss {np.mean([da[d] for d in common]):.4f} vs "
                      f"proxy control at step {last} {np.mean([dc[d] for d in common]):.4f} "
                      f"on {len(common)} datasets -- how far the proxy is from a real model")

    print()
    if not verdicts:
        print("VERDICT: nothing to read yet.")
        return 0
    held = first_agree is not None and all(a for s, a, _ in verdicts if s >= first_agree)
    last_step, _, last_opposite = verdicts[-1]
    if held and len(verdicts) < 3:
        # "Holds at every later checkpoint" is vacuous with nothing later to hold at.
        print(f"VERDICT (PROVISIONAL, {len(verdicts)} checkpoint(s)): the published sign appears at "
              f"step {first_agree}. Persistence cannot be judged until more checkpoints are scored.")
    elif held:
        print(f"VERDICT: the proxy reproduces the published sign from step {first_agree} onward. "
              f"Ablations can be screened at >= {first_agree} steps.")
    elif first_agree is not None:
        print(f"VERDICT: the sign appears at step {first_agree} but does not hold at every later "
              f"checkpoint -- not yet a trustworthy proxy.")
    elif last_opposite:
        # A distinct and more serious outcome than "cannot see it": the proxy separates the
        # arms, confidently, in the direction the paper says is wrong.
        print(f"VERDICT: the proxy separates the arms in the OPPOSITE direction to the published "
              f"result at step {last_step} ({args.treatment} "
              f"{'better' if want_worse else 'worse'}). It contradicts the expectation -- "
              f"do not use it to screen, and check the arms were configured as intended.")
    else:
        print(f"VERDICT: no separation at corrected p < {args.alpha} by step {last_step}. "
              f"The proxy cannot see a published ~100-Elo effect at this length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
