"""Task selection for a single checkpoint that serves both classification and regression.

TP-07 in ``TODO.md``. TabPFN-3.5 trains one model on both tasks with the task type as an
input; only the label encoder and the output head are task-specific, and everything between
them is shared. :class:`TaskConditioned` holds the bookkeeping that the two modules owning
task-specific parts (``ColEmbedding`` and ``ICLearning``) have in common.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

TASKS = ("classification", "regression")


def check_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return task


class TaskConditioned:
    """Mixin for an ``nn.Module`` whose behaviour depends on the task.

    A single-task module has its task fixed by ``max_classes`` at construction, exactly as
    before. A multitask module starts with no task and refuses to run until one is set, so a
    forgotten ``set_task`` fails loudly instead of silently using the wrong head.

    Multitask modules also own a learned per-task embedding, added to every row (train and
    test) at the module's input. It is zero-initialized: at initialization the task reaches
    the model only through its label encoder, and a joint model loaded with single-task
    weights reproduces that model exactly. Zero still gets a gradient, so it learns.
    """

    def _init_task(self, max_classes: int, multitask: bool, dim: int) -> None:
        self.multitask = multitask
        self.task: Optional[str] = None
        if multitask:
            if max_classes <= 0:
                raise ValueError("A multitask model needs max_classes > 0 for its classification head.")
            self.task_embed = nn.Parameter(torch.zeros(len(TASKS), dim))
        else:
            self.task = "classification" if max_classes > 0 else "regression"

    def set_task(self, task: str) -> None:
        check_task(task)
        if not self.multitask and task != self.task:
            raise ValueError(f"This is a single-task {self.task} model; it cannot run {task}.")
        self.task = task

    def _require_task(self) -> str:
        if self.task is None:
            raise RuntimeError("Multitask model has no task selected; call set_task() first.")
        return self.task

    @property
    def is_classification(self) -> bool:
        return self._require_task() == "classification"

    def add_task_embedding(self, src: Tensor) -> Tensor:
        if not self.multitask:
            return src
        return src + self.task_embed[TASKS.index(self._require_task())].to(src.dtype)
