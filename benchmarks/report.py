"""Turn ledger rows into a leaderboard.

    PYTHONPATH=src python -m benchmarks.report --ledger benchmarks/_results/real_small.jsonl \\
        --metric accuracy

    # compare two configurations head to head
    PYTHONPATH=src python -m benchmarks.report --ledger ... --metric log_loss \\
        --lower-is-better --against v2-default
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ._core.aggregate import bootstrap_ci, elo, mean_rank, records_from_ledger, win_rate
from ._core.schema import Ledger


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--run-id", default=None, help="Restrict to one sweep")
    parser.add_argument("--lower-is-better", action="store_true", help="For error metrics: log_loss, CRPS, time")
    parser.add_argument("--against", default=None, help="Report win rates against this config_id")
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args(argv)

    rows = list(Ledger(Path(args.ledger)))
    if args.run_id:
        rows = [r for r in rows if r.get("run_id") == args.run_id]

    records = records_from_ledger(rows, metric=args.metric, suite=args.suite)
    if not records:
        print(f"No usable rows for metric '{args.metric}'"
              + (f" in suite '{args.suite}'" if args.suite else ""))
        errored = sum(1 for r in rows if r.get("error"))
        if errored:
            print(f"({errored} row(s) in this ledger carry an error)")
        return 1

    higher = not args.lower_is_better
    methods = sorted({r.method for r in records})
    datasets = sorted({r.dataset for r in records})

    print(f"metric: {args.metric}   ({'higher' if higher else 'lower'} is better)")
    print(f"methods: {len(methods)}   datasets: {len(datasets)}   records: {len(records)}\n")

    ranks = mean_rank(records, higher_is_better=higher)
    ratings = elo(records, higher_is_better=higher)
    cis = {} if args.no_ci or len(methods) < 2 else bootstrap_ci(
        records, lambda r: mean_rank(r, higher_is_better=higher), n_boot=args.n_boot
    )

    width = max(len(m) for m in methods)
    header = f"{'method':{width}s}  {'mean rank':>10s}  {'elo':>7s}"
    if cis:
        header += f"  {'rank 95% CI':>18s}"
    if args.against:
        header += f"  {'win vs ' + args.against:>22s}"
    print(header)
    print("-" * len(header))

    for method in sorted(ranks, key=lambda m: ranks[m]):
        line = f"{method:{width}s}  {ranks[method]:>10.3f}  {ratings.get(method, float('nan')):>7.0f}"
        if cis and method in cis:
            lo, hi = cis[method]
            line += f"  {f'[{lo:.2f}, {hi:.2f}]':>18s}"
        if args.against:
            if method == args.against:
                line += f"  {'—':>22s}"
            else:
                wr = win_rate(records, method, args.against, higher_is_better=higher)
                line += f"  {wr:>21.1%}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
