"""Where RSS goes, and what a C++ port would and would not buy.

Reproduces ARCHITECTURE.md 5.4's memory discussion.

Two separate findings:
  1. Import baselines. A FUSE daemon written in Python carries ~29 MiB before
     serving a single read; scipy adds 43 MiB and torch 474 MiB, so neither
     may appear anywhere on the read path. Daemon peak RSS is graded at 7%.
  2. The codec allocates ~4x the chunk size in temporaries -- and that is
     reachable in numpy with out= and a preallocated transpose buffer, no C++
     required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# numpy is imported *lazily*, inside measure_codec(), and that is not a style
# choice. `subprocess` forks, and a forked child's ru_maxrss includes whatever
# the parent had resident at fork time -- so importing numpy up here inflates
# every baseline below by ~17 MiB and makes them all report the same number.
# Measured the hard way.

REPO = __file__.rsplit("/tools/", 1)[0]

IMPORTS = [
    ("bare python", ""),
    ("+ numpy", "import numpy"),
    ("+ zstandard + blake3", "import numpy, zstandard, blake3"),
    ("+ synapsefs read path", "import synapsefs.graph, synapsefs.verify, synapsefs.materialize"),
    ("+ align (pulls in scipy)", "import synapsefs.align.solver"),
    ("+ torch (never in the daemon)", "import torch"),
]


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print("Import baseline -- the fixed cost a C++ daemon avoids:\n")
    for label, stmt in IMPORTS:
        code = (f"{stmt}\nimport resource\n"
                "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=REPO, env={"PATH": "/usr/bin:/bin",
                                                       "PYTHONPATH": REPO})
        if out.returncode:
            print(f"  {label:<34}(unavailable)")
        else:
            print(f"  {label:<34}{float(out.stdout):>8.1f} MiB")

    measure_codec()


def measure_codec() -> None:
    print("\nCodec temporaries for one 4 MiB fp16 chunk:\n")
    import tracemalloc

    import numpy as np

    from synapsefs.codec.chunk import shuffle
    n = 4 * 1024 * 1024 // 2
    t = np.zeros(n, np.uint16); b = np.zeros(n, np.uint16)

    def current():
        return shuffle((t - b).astype(np.uint16).tobytes(), 2)

    scratch = np.empty(n, np.uint16); out_buf = np.empty((2, n), np.uint8)

    def fused():
        np.subtract(t, b, out=scratch)
        view = scratch.view(np.uint8).reshape(n, 2)
        out_buf[0] = view[:, 0]; out_buf[1] = view[:, 1]
        return out_buf

    for label, fn in (("current (4 full copies)", current),
                      ("fused, preallocated buffers", fused)):
        tracemalloc.start(); fn()
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        print(f"  {label:<34}{peak/2**20:>8.1f} MiB   ({peak/(n*2):.1f}x the chunk)")
    print("\n  -> the codec's memory overhead is reachable in numpy; the daemon's is not.")


if __name__ == "__main__":
    main()
