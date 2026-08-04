"""Shared model configuration for the RelBench evaluation scripts.

Every script here used to hardcode ``device="cpu", n_estimators=4``. Both are
handicaps -- the default estimator count is 8, and CPU inference is slow enough
that it silently caps how much of a sweep is worth running -- and neither made it
into any recorded result. That matters because three of the five tasks in the
window sweep landed at 0.50-0.63 AUC, and "the model was weakened" is a live
explanation for that which no measurement on this branch has ruled out.

Centralised so the settings are chosen once and, more importantly, *printed*: a
number recorded without the configuration that produced it cannot be compared
against anything later.

Moving to GPU then turned up the thing that actually mattered. TabICL runs mixed
precision on CUDA and fp32 on CPU, so a device swap is silently a precision swap.
On rel-event / user-ignore, ``+ motif features``:

    device  n_est  precision   AUC
    cpu     4      fp32        0.7332
    cuda    4      fp32        0.7332     <- identical; the device is neutral
    cuda    8      fp32        0.7364
    cuda    4      amp         0.6938     <- -0.039
    cuda    8      amp         0.7167     <- -0.020

Two conclusions, and the second is the important one.

The estimator count was never the problem. In fp32, 4 -> 8 moves this task by
+0.003, so the weak 0.50-0.63 scores in the five-task window sweep are not an
artifact of a handicapped model. That hypothesis is now closed.

AMP is louder than the signal. -0.039 exceeds every effect this branch has
attributed -- typing +0.021, triangles +0.016, causality -0.011 -- so a sweep run
under AMP cannot resolve its own conclusions. That is why these scripts default to
fp32 on GPU and make ``--amp`` opt in, which also keeps new numbers comparable
against the CPU results already recorded in the branch history.
"""

from __future__ import annotations

import argparse

import torch

from .._sklearn.classifier import TabICLClassifier

__all__ = ["add_model_args", "resolve_device", "make_classifier", "describe_model"]

# Matches TabICLClassifier's own default. Stated here so a drift in either is visible.
DEFAULT_N_ESTIMATORS = 8


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--device`` / ``--n-estimators`` pair to an eval script."""
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' (CUDA when present), 'cpu', 'cuda', or an explicit device string",
    )
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="run the GPU path in mixed precision. Off by default here -- see the "
        "module docstring; AMP moves rel-event/user-ignore by up to 0.039 AUC, "
        "which is larger than any effect these scripts are built to measure.",
    )
    return parser


def resolve_device(spec: str = "auto") -> str:
    """Turn ``"auto"`` into a concrete device, leaving anything explicit alone."""
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_classifier(args, **kwargs) -> TabICLClassifier:
    """Build the classifier every eval script should be using.

    Row chunking is set to ``"auto"`` on GPU. It is exactly equivalent to the
    unchunked path, so it cannot move an AUC; it only stops a wide feature block
    -- windows triple the column count, and the widest arm here reaches 570
    columns -- from turning the switch to GPU into an OOM on a small card. On CPU
    it is left off, where it would only add overhead.
    """
    device = resolve_device(args.device)
    config = None
    if not device.startswith("cpu"):
        config = {
            "COL_CONFIG": {"row_chunk": "auto"},
            "ICL_CONFIG": {"row_chunk": "auto"},
        }
        if not getattr(args, "amp", False):
            for section in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"):
                config.setdefault(section, {})["use_amp"] = False

    return TabICLClassifier(
        n_estimators=args.n_estimators,
        device=device,
        random_state=0,
        inference_config=config,
        **kwargs,
    )


def describe_model(args) -> str:
    """One line naming what produced the numbers below it."""
    device = resolve_device(args.device)
    name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"
    if device.startswith("cpu"):
        precision = "fp32"
    else:
        precision = "amp" if getattr(args, "amp", False) else "fp32"
    return (
        f"model: TabICL n_estimators={args.n_estimators} "
        f"device={device} ({name}) precision={precision}"
    )
