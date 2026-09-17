#!/usr/bin/env python3
"""Record what this machine actually is, before anything is measured on it.

Writes ``pod_doctor.json`` and prints a summary. Run it first on every pod and keep the
output next to the results, because a benchmark number is only interpretable alongside the
hardware that produced it.

Four things here are not cosmetic for this project:

* **compute capability** decides whether FlashAttention-3 exists. FA3 is Hopper or newer
  (sm_90+); ``scripts/train_v2_clf_stage{2,3}.sh`` pass ``--use_flash_attn3 True``, and
  ``tabicl._model.attention`` silently falls back when the import fails. A stage-2 run on an
  sm_86 card is therefore a *different run* from the same script on an H100, with no error
  to tell you so.
* **VRAM** decides how far up the BM-02 curve you can go before the KV cache stops fitting.
* **CPU count** decides ``--n_jobs`` for prior generation, which is what actually feeds the
  GPU during proxy training.
* **which tabicl is importable** -- a PyPI wheel shadowing the checkout means the pod
  benchmarks released code while you think it is measuring your branch.

Stdlib plus torch only; it has to run before the project is installed.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "pod_doctor.json"


def _torch_info() -> dict:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}

    info = {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        info.update(
            {
                "gpu": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 1),
                "compute_capability": f"{major}.{minor}",
                "sm": major * 10 + minor,
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return info


def _flash_attn_info(sm: int | None) -> dict:
    """FA3 presence, and whether the hardware could support it at all.

    The distinction matters: 'not installed' is fixable with pip, 'sm_86' is not.
    """

    out: dict = {"fa3_importable": False, "fa3_error": None}
    try:
        import flash_attn_interface  # noqa: F401

        out["fa3_importable"] = True
    except Exception as exc:  # noqa: BLE001
        out["fa3_error"] = repr(exc)[:200]

    if sm is not None:
        out["hardware_supports_fa3"] = sm >= 90
        if sm < 90:
            out["note"] = (
                f"sm_{sm} is pre-Hopper: FlashAttention-3 cannot run here regardless of "
                "install. --use_flash_attn3 True will silently fall back, so stage 2/3 "
                "runs on this card are NOT comparable to the reference recipe."
            )
    return out


def _tabicl_info() -> dict:
    try:
        import tabicl
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": repr(exc)[:200]}

    path = os.path.realpath(tabicl.__file__)
    try:
        from importlib.metadata import version

        ver = version("tabicl")
    except Exception:
        ver = getattr(tabicl, "__version__", None)
    return {
        "importable": True,
        "path": path,
        "version": ver,
        "is_checkout": "/site-packages/" not in path,
    }


def _glibc() -> str | None:
    try:
        out = subprocess.run(["ldd", "--version"], capture_output=True, text=True, timeout=10)
        return out.stdout.splitlines()[0] if out.stdout else None
    except Exception:
        return None


def main() -> int:
    torch_info = _torch_info()
    sm = torch_info.get("sm")

    report = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "glibc": _glibc(),
        "disk_free_gb": round(shutil.disk_usage("/").free / 1024**3, 1),
        "torch": torch_info,
        "flash_attn": _flash_attn_info(sm),
        "tabicl": _tabicl_info(),
    }

    OUT.write_text(json.dumps(report, indent=2))

    print(f"host      {report['hostname']}  ({report['platform']})")
    print(f"python    {report['python']}   cpus {report['cpu_count']}   "
          f"disk free {report['disk_free_gb']} GB")
    if torch_info.get("available"):
        if torch_info.get("cuda_available"):
            print(f"gpu       {torch_info['gpu']}  {torch_info['vram_gb']} GB  "
                  f"sm_{torch_info['sm']}  x{torch_info['device_count']}")
        else:
            print("gpu       NONE VISIBLE -- torch.cuda.is_available() is False")
        print(f"torch     {torch_info['version']}  cuda {torch_info['cuda_version']}")

    fa = report["flash_attn"]
    if sm is None:
        print("flash-3   unknown -- no CUDA device to judge against")
    elif fa.get("hardware_supports_fa3") is False:
        print(f"flash-3   UNAVAILABLE ON THIS HARDWARE -- {fa.get('note')}")
    elif fa["fa3_importable"]:
        print("flash-3   available")
    else:
        print("flash-3   hardware capable, not installed "
              "(stage 2/3 will fall back; install flash-attn to match the recipe)")

    tab = report["tabicl"]
    if not tab.get("importable"):
        print(f"tabicl    NOT IMPORTABLE -- {tab.get('error')}")
    elif not tab.get("is_checkout"):
        print(f"tabicl    {tab['version']} from site-packages -- NOT the checkout. "
              "Benchmarks would measure released code. Use PYTHONPATH=src or pip install -e .")
    else:
        print(f"tabicl    {tab.get('version')} from checkout  ({tab['path']})")

    print(f"\nwrote {OUT}")

    # Non-zero when the pod cannot do the job it was rented for.
    if not torch_info.get("cuda_available"):
        print("\nFAIL: no CUDA device. This pod cannot run training or the BM-02 sweep.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
