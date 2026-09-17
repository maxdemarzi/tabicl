"""Tests for the resumable runner.

Focused on the behaviours that only show up mid-sweep: interruption, retry of
failures, and not silently dropping work.
"""

from __future__ import annotations

from benchmarks._core.runner import Runner, Task
from benchmarks._core.schema import Ledger


def _task(dataset="ds0", config_id="cfg", metrics=("acc",), **kw):
    return Task(suite="s", dataset=dataset, fold=0, config_id=config_id, metrics=metrics, **kw)


def test_runner_records_one_row_per_metric(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    runner = Runner(ledger, run_id="R", verbose=False)

    runner.run([_task(metrics=("acc", "auc"))], lambda t: {"acc": 0.9, "auc": 0.8})

    rows = list(ledger)
    assert {r["metric"]: r["value"] for r in rows} == {"acc": 0.9, "auc": 0.8}


def test_runner_skips_completed_tasks_on_resume(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    tasks = [_task(dataset=f"ds{i}") for i in range(3)]

    calls = []

    def execute(task):
        calls.append(task.dataset)
        return {"acc": 1.0}

    executed = Runner(ledger, run_id="R", verbose=False).run(tasks, execute)
    assert executed == 3 and calls == ["ds0", "ds1", "ds2"]

    calls.clear()
    executed = Runner(ledger, run_id="R", verbose=False).run(tasks, execute)
    assert executed == 0 and calls == []


def test_runner_resumes_only_the_unfinished_tail(tmp_path):
    """The interruption case: half a sweep done, re-invoke, finish the rest."""

    ledger = Ledger(tmp_path / "l.jsonl")
    tasks = [_task(dataset=f"ds{i}") for i in range(4)]

    def execute_partial(task):
        if task.dataset == "ds2":
            raise KeyboardInterrupt("cluster time limit")
        return {"acc": 1.0}

    try:
        Runner(ledger, run_id="R", verbose=False).run(tasks, execute_partial)
    except KeyboardInterrupt:
        pass

    done = []
    Runner(ledger, run_id="R", verbose=False).run(
        tasks, lambda t: (done.append(t.dataset), {"acc": 1.0})[1]
    )
    assert done == ["ds2", "ds3"]


def test_runner_records_failures_as_rows_and_retries_them(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")

    def boom(task):
        raise RuntimeError("CUDA out of memory")

    Runner(ledger, run_id="R", verbose=False).run([_task()], boom)

    rows = list(ledger)
    assert len(rows) == 1
    assert rows[0]["value"] is None
    assert "CUDA out of memory" in rows[0]["error"]

    # The failure must not count as done.
    calls = []
    Runner(ledger, run_id="R", verbose=False).run(
        [_task()], lambda t: (calls.append(1), {"acc": 1.0})[1]
    )
    assert calls == [1]


def test_runner_failure_does_not_abort_the_sweep(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    tasks = [_task(dataset=f"ds{i}") for i in range(3)]

    def flaky(task):
        if task.dataset == "ds1":
            raise ValueError("bad dataset")
        return {"acc": 1.0}

    executed = Runner(ledger, run_id="R", verbose=False).run(tasks, flaky)
    assert executed == 3

    rows = list(ledger)
    assert sum(1 for r in rows if r["error"]) == 1
    assert sum(1 for r in rows if r["value"] == 1.0) == 2


def test_runner_multi_metric_partial_completion_is_rerun(tmp_path):
    """A task that produced only some of its metrics is not done."""

    ledger = Ledger(tmp_path / "l.jsonl")
    task = _task(metrics=("acc", "auc"))

    Runner(ledger, run_id="R", verbose=False).run([task], lambda t: {"acc": 0.9})

    calls = []
    Runner(ledger, run_id="R", verbose=False).run(
        [task], lambda t: (calls.append(1), {"acc": 0.9, "auc": 0.8})[1]
    )
    assert calls == [1]


def test_runner_different_run_ids_do_not_share_resume_state(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    Runner(ledger, run_id="R1", verbose=False).run([_task()], lambda t: {"acc": 1.0})

    calls = []
    Runner(ledger, run_id="R2", verbose=False).run(
        [_task()], lambda t: (calls.append(1), {"acc": 1.0})[1]
    )
    assert calls == [1]


def test_runner_resume_false_reruns_everything(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    Runner(ledger, run_id="R", verbose=False).run([_task()], lambda t: {"acc": 1.0})

    calls = []
    Runner(ledger, run_id="R", resume=False, verbose=False).run(
        [_task()], lambda t: (calls.append(1), {"acc": 1.0})[1]
    )
    assert calls == [1]
    assert len(list(ledger)) == 2  # both rows kept; append-only


def test_runner_stamps_provenance_and_cost(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    Runner(ledger, run_id="R", device="cpu", verbose=False).run(
        [_task(seed=3, checkpoint="v2-clf")], lambda t: {"acc": 1.0}
    )

    row = list(ledger)[0]
    assert row["seed"] == 3
    assert row["checkpoint"] == "v2-clf"
    assert row["device"] == "cpu"
    assert row["wall_clock_s"] is not None and row["wall_clock_s"] >= 0
    assert row["provenance"]["python_version"]


def test_runner_config_is_recorded_verbatim(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    cfg = {"n_estimators": 8, "norm_methods": ["none", "power"]}
    Runner(ledger, run_id="R", verbose=False).run([_task(config=cfg)], lambda t: {"acc": 1.0})
    assert list(ledger)[0]["config"] == cfg


def test_runner_warns_when_tabicl_is_not_local(tmp_path, capsys):
    ledger = Ledger(tmp_path / "l.jsonl")
    runner = Runner(ledger, run_id="R", verbose=True)
    runner._provenance["tabicl_is_local"] = False
    runner._provenance["tabicl_path"] = "/site-packages/tabicl/__init__.py"

    assert runner.warn_if_not_local_tabicl() is False
    assert "NOT this working tree" in capsys.readouterr().out
