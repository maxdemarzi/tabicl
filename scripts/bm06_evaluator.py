#!/usr/bin/env python3
"""BM-06 evaluator: score every permanent proxy checkpoint as it appears.

Runs alongside training and evaluates each `step-N.ckpt` whose step is a multiple of
--every on the frozen `cc18_narrow` suite, appending to the ledger. Evaluating as
checkpoints land, rather than at the end, means a run that dies at hour 16 still leaves
a learning curve behind, and the curve can be read mid-run.

Two details that each would have silently invalidated the experiment:

* **Slim export.** Training checkpoints carry optimizer and scheduler state; the classifier
  loads with ``weights_only=True``. Each checkpoint is re-saved as just ``{config,
  state_dict}`` -- the only two keys the classifier reads -- before evaluation.
* **--model-path.** Without it the suite loads the *released* checkpoint and every row
  measures the wrong model while looking perfectly normal.

Each checkpoint is scored only on the suites of the tasks it can do, read from its own
config (TP-07): a classifier on the classification suites, a regressor on the regression
ones, and a multitask checkpoint on both, under one config_id. So one evaluator can follow
a classification control, a regression control and a joint arm at once, and a regressor is
never handed to TabICLClassifier.

Idempotent: the ledger's resume keys skip anything already scored, so re-running after a
crash picks up where it stopped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Run as a script from anywhere; the suite registry lives in the repo's benchmarks package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def slim(src: Path, dst: Path) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)  # our own file
    torch.save({"config": ckpt["config"], "state_dict": ckpt["state_dict"]}, dst)


def suite_task(suite: str) -> str:
    from benchmarks._core.datasets import SUITES  # noqa: PLC0415

    tasks = {spec.task for spec in SUITES[suite]}
    if len(tasks) != 1:
        raise ValueError(f"suite {suite!r} mixes tasks {sorted(tasks)}")
    return tasks.pop()


def checkpoint_tasks(config: dict) -> set[str]:
    """Which tasks a checkpoint's config says it can do."""

    if config.get("multitask"):
        return {"classification", "regression"}
    return {"regression"} if config.get("max_classes", 10) == 0 else {"classification"}


def permanent_steps(ckpt_dir: Path, every: int) -> list[int]:
    steps = []
    for f in ckpt_dir.glob("step-*.ckpt"):
        try:
            n = int(f.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if n > 0 and n % every == 0:
            steps.append(n)
    return sorted(steps)


def evaluate(arm: str, step: int, model_path: Path | None, args, gpu: str) -> int:
    config_id = f"{arm}-step{step}" if model_path else arm
    if model_path:
        config = torch.load(model_path, map_location="cpu", weights_only=True)["config"]
        tasks = checkpoint_tasks(config)
    else:
        tasks = {"classification", "regression"}  # the released anchor exists for both
    suites = [suite for suite in args.suite if suite_task(suite) in tasks]
    if not suites:
        print(f"[eval] WARNING: {config_id} does {sorted(tasks)} but no --suite given is for that; "
              f"nothing scored. Pass e.g. --suite ctr23 for regression.", flush=True)
    rc = 0
    for suite in suites:
        rc |= _evaluate_one(suite, config_id, model_path, args, gpu)
    return rc


def _evaluate_one(suite: str, config_id: str, model_path: Path | None, args, gpu: str) -> int:
    cmd = [sys.executable, "-m", "benchmarks.suites.real_small",
           "--suite", suite, "--run-id", args.run_id, "--config-id", config_id,
           "--n-folds", str(args.n_folds), "--max-rows", str(args.max_rows),
           "--device", "cuda", "--ledger", args.ledger,
           "--checkpoint", str(model_path) if model_path else "released-v2"]
    if model_path:
        cmd += ["--model-path", str(model_path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    print(f"[eval] {config_id} / {suite} on gpu {gpu}", flush=True)
    return subprocess.call(cmd, env=env, cwd=args.repo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True,
                    help="name=ckpt_dir=gpu, e.g. control=/workspace/ckpt/c=0 (repeatable)")
    ap.add_argument("--every", type=int, default=2500)
    ap.add_argument("--final-step", type=int, required=True)
    ap.add_argument("--suite", action="append", default=None,
                    help="Dataset suite to score on; repeat to score several. The ledger records "
                         "the suite per row, so one config_id can span suites safely.")
    ap.add_argument("--run-id", default="BM-06-C")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=3000)
    ap.add_argument("--ledger", default="benchmarks/_results/bm06.jsonl")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--poll", type=float, default=120.0)
    ap.add_argument("--anchor", action="store_true", help="Also score the released checkpoint once")
    args = ap.parse_args()
    args.suite = args.suite or ["cc18_narrow"]

    arms = []
    for spec in args.arm:
        name, d, gpu = spec.split("=")
        arms.append((name, Path(d), gpu))

    if args.anchor:
        evaluate("released-v2", 0, None, args, arms[0][2])

    done: set[tuple[str, int]] = set()
    while True:
        for name, d, gpu in arms:
            for step in permanent_steps(d, args.every):
                if (name, step) in done:
                    continue
                src = d / f"step-{step}.ckpt"
                dst = d / f"eval-step-{step}.pt"
                # Give the trainer a moment to finish writing a checkpoint that just appeared.
                if time.time() - src.stat().st_mtime < 30:
                    continue
                if not dst.exists():
                    slim(src, dst)
                rc = evaluate(name, step, dst, args, gpu)
                if rc == 0:
                    done.add((name, step))
        finished = all((name, args.final_step) in done for name, _, _ in arms)
        if finished:
            Path(args.repo, "BM06_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
            print("[eval] all arms reached final step and were scored -- done", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
