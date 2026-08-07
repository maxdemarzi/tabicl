"""Re-pair every plain-arm A/B in the retrieved logs, using the corrected method.

The per-seed tables were always printed. What was missing was using them: every A/B this
session was computed from the CALIBRATED block, where arms correlate at r = -0.03 to +0.34
because selection varies per seed. On fixed configurations they correlate at 0.88-0.94 and
pairing is worth 2.4-3.0x on the standard error.

The first version of this script flushed a pending seed table only on the next ARM header,
never on a BLOCK header, so the last arm of one task was attributed to the first arm of the
next -- and it duly reported +17.66 for a task whose whole range is two points. Any delta
larger than a couple of points on this benchmark is a parsing bug, so that is asserted.
"""
import re, sys
import numpy as np

BLOCK  = re.compile(r"^#{5,}\s*(.+?)\s*$")
ARMHDR = re.compile(r"^----\s*(.+?)\s*$")
SEEDHDR= re.compile(r"^\s*seed\s+((?:\S+\s*)+)$")
SEEDROW= re.compile(r"^\s*(\d+)\s+((?:\s*\d+\.\d+)+)")

def parse(path):
    out, block, arm, cols, cur = [], "", "", None, None
    def flush():
        nonlocal cur, cols
        if cur and any(len(v) >= 3 for v in cur.values()):
            out.append((block, arm, cur))
        cur, cols = None, None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip()
        m = BLOCK.match(line)
        if m and ("::" in m.group(1) or "TASK" in m.group(1)):
            flush(); block = m.group(1); arm = ""; continue
        m = ARMHDR.match(line)
        if m:
            flush(); arm = m.group(1); continue
        m = SEEDHDR.match(line)
        if m and "base" in m.group(1):
            cols = m.group(1).split(); cur = {c: [] for c in cols}; continue
        if cols:
            m2 = SEEDROW.match(line)
            if m2:
                vals = [float(v) for v in m2.group(2).split()]
                for c, v in zip(cols, vals): cur[c].append(v)
            elif line.strip() and not line.startswith(" "):
                flush()
    flush()
    return out

def report(path):
    by = {}
    for blk, arm, arms in parse(path):
        by.setdefault(blk, []).append((arm, arms))
    for blk, entries in by.items():
        if len(entries) != 2:
            continue                      # not a clean two-arm comparison; skip rather than guess
        (la, A), (lb, B) = entries
        common = [c for c in A if c in B and len(A[c]) >= 3 and len(B[c]) >= 3]
        if not common: continue
        print(f"\n{blk}\n  A = {la}\n  B = {lb}")
        for c in common:
            n = min(len(A[c]), len(B[c]))
            x, y = np.array(A[c][:n]), np.array(B[c][:n]); d = y - x
            assert abs(d.mean()) < 8, f"delta {d.mean():.2f} on {blk}/{c} -- parsing bug"
            se = d.std(ddof=1)/np.sqrt(n); r = np.corrcoef(x, y)[0, 1]
            flag = "  <-- outside floor" if abs(d.mean()) >= 0.6 else ""
            print(f"    {c:<10} {x.mean():6.2f} -> {y.mean():6.2f}  delta {d.mean():+6.2f}  "
                  f"SE {se:.2f}  t {d.mean()/se:+5.2f}  r {r:+.2f}  {int((d>0).sum())}/{n}{flag}")

for p in sys.argv[1:]:
    report(p)
