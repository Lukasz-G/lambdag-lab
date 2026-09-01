# Label-free calibration test on Lithuanian, L=5000 symmetric.
#
# Question: can the +7 location offset (sqrt-Cllr 0.951 despite AUC 1.0) be fixed
# WITHOUT labelled cases? The reference bank's author identities are metadata, so
# same-author pseudo-cases (one author's two disjoint windows) and different-author
# pseudo-cases (two authors' windows) are free. Score the full 12x12 pair matrix,
# then for each evaluation case fit a logistic calibrator on all pseudo-cases that
# involve NEITHER case author (leave-both-out), and report Cllr of the calibrated
# LLRs next to the uncalibrated sqrt correction.

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, read_tsv  # noqa: E402
from run_longtexts import window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG, cllr, cllr_min  # noqa: E402

DS, L = "lithuanian_novels", 5000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    bank = {}
    for f in sorted((MASKED / DS / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    names = sorted(n for n in bank if bank[n]["ntok"] >= 2 * L)
    known = {n: window(bank[n]["sents"], 0, L) for n in names}
    quest = {n: window(bank[n]["sents"], L, L) for n in names}

    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    t0 = time.time()
    rows = []                       # (known author, questioned author, sqrt score)
    for a in names:                 # one reference set per known author: exclude a
        import random
        pool = [s for n in names if n != a for s in bank[n]["sents"]]
        rng = random.Random(f"{DS}|{L}|{a}")
        rng.shuffle(pool)
        tot, kept = 0, []
        for s in pool:
            kept.append(s); tot += len(s)
            if tot >= 60 * L:
                break
        lg.set_reference(kept)
        for b in names:
            r = lg.score(quest[b], known[a], with_details=False)
            rows.append((a, b, r.lambda_sqrt))
    print(f"scored {len(rows)} pairs in {time.time()-t0:.0f}s", flush=True)

    s = np.array([r[2] for r in rows])
    y = np.array([int(a == b) for a, b, _ in rows])
    ka = [r[0] for r in rows]; qa = [r[1] for r in rows]

    from sklearn.linear_model import LogisticRegression
    llr = np.empty(len(rows))
    for i in range(len(rows)):
        tr = [j for j in range(len(rows))
              if {ka[j], qa[j]}.isdisjoint({ka[i], qa[i]})]
        X, yy = s[tr].reshape(-1, 1), y[tr]
        m = LogisticRegression(C=1e6).fit(X, yy)
        prior = np.log(yy.mean() / (1 - yy.mean()))   # remove training prior odds
        llr[i] = (m.coef_[0, 0] * s[i] + m.intercept_[0] - prior) / np.log(10)

    print(f"\n{'':24s}{'Cllr':>8}{'Cllr_min':>10}")
    print(f"{'sqrt, uncalibrated':24s}{cllr(s[y==1], s[y==0]):8.3f}"
          f"{cllr_min(s[y==1], s[y==0]):10.3f}")
    print(f"{'bank-calibrated (LBO)':24s}{cllr(llr[y==1], llr[y==0]):8.3f}"
          f"{cllr_min(llr[y==1], llr[y==0]):10.3f}")
    print(f"\nn = {len(rows)} ({int(y.sum())} same / {int((1-y).sum())} diff); "
          f"calibrator saw no case author (leave-both-out)")


if __name__ == "__main__":
    main()
