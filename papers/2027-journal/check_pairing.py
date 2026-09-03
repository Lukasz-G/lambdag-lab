# Pre-analysis validity gate for the genre-absent grid: assert that all four
# arms of every cell scored the IDENTICAL case list, that no arm contains a
# degenerate (all-zero) donor column, and that no case was scored against a
# donor fitted on its own questioned author. An arm comparison is meaningless
# unless these hold, so they are checked before any AUC is computed.
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
XS = Path(__file__).resolve().parent.parent / "scores" / "xabsent2"
ARMS = ["oracle", "borrowed", "native-wrong", "pooled"]

def namekey(stem):
    import re
    return tuple(sorted(p for p in re.sub(r"^\d+_", "", stem).split("_") if p))

cells = sorted({f.name.split("__")[0] for f in XS.glob("*.jsonl")})
ok = True
for c in cells:
    keys, present = {}, []
    for a in ARMS:
        f = XS / f"{c}__{a}__K10000w100.jsonl"
        if not f.exists():
            continue
        present.append(a)
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        keys[a] = {(r["cand"], r["quest"], r["qw"]) for r in rows}
        M = np.array([r["lam_j"] for r in rows], dtype=object)
        zero = any(np.allclose([float(x) for x in row], 0) for row in M)
        self_ref = any(namekey(d.split(":", 1)[1]) == namekey(r["quest"])
                       for r in rows for d in r["donors"])
        if zero or self_ref:
            ok = False
            print(f"  FAIL {c}/{a}: degenerate={zero} self-reference={self_ref}")
    if len(present) < 2:
        continue
    base = keys[present[0]]
    same = all(keys[a] == base for a in present)
    print(f"{c:28s} arms={len(present)} cases={len(base):4d} "
          f"identical case list: {'YES' if same else 'NO'}")
    if not same:
        ok = False
        for a in present:
            print(f"    {a:13s} shares {len(keys[a] & base)}/{len(base)}")
print("\nVALIDITY GATE:", "PASS" if ok else "FAIL")
