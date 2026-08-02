"""Build the optional compiled hash-join backend, in place.

    python -m tabicl.scaling.build_native

The extension is optional: ``wcoj_join`` falls back to the pure-Python leapfrog
triejoin when it is absent, so nothing here is required to use ``tabicl``. It exists
because a Python join loses to compiled binary joins by 10-60x on wall clock -- the
algorithm is not the problem, the interpreter is -- and comparing an interpreted
implementation against ``pandas`` or ``scipy`` measures the language, not the join.

Requires a C++17 compiler and ``pybind11``:

* Linux/macOS: gcc or clang, already present on most systems.
* Windows: MSVC Build Tools with the C++ workload. Clang alone is not sufficient --
  it does not ship the Windows SDK headers and import libraries a CPython extension
  must link against.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


def main() -> int:
    here = pathlib.Path(__file__).parent
    source = here / "_wcoj_native.cpp"
    if not source.exists():
        print(f"missing source: {source}", file=sys.stderr)
        return 1

    try:
        import pybind11
    except ImportError:
        print("pybind11 is required: pip install pybind11", file=sys.stderr)
        return 1

    # setuptools drives the platform's compiler so this stays one command on
    # Windows and POSIX alike, rather than two hand-written command lines.
    setup_py = f"""
from setuptools import setup, Extension
import pybind11

setup(
    name="_wcoj_native",
    ext_modules=[
        Extension(
            "_wcoj_native",
            [r"{source}"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=(
                ["/std:c++17", "/O2", "/EHsc"] if __import__("sys").platform == "win32"
                else ["-std=c++17", "-O3", "-fvisibility=hidden"]
            ),
        )
    ],
    script_args=["build_ext", "--inplace", "--build-temp", r"{here / '_build_tmp'}"],
)
"""
    driver = here / "_setup_native.py"
    driver.write_text(setup_py, encoding="utf-8")
    try:
        result = subprocess.run([sys.executable, str(driver)], cwd=str(here))
    finally:
        driver.unlink(missing_ok=True)

    if result.returncode != 0:
        return result.returncode

    built = sorted(here.glob("_wcoj_native*.pyd")) + sorted(here.glob("_wcoj_native*.so"))
    print(f"built: {[b.name for b in built]}" if built else "no artifact produced")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
