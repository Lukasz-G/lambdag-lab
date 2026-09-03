# Cross-genre verification by impostor standardisation (journal paper,
# cross-genre section): make the genre gap common-mode instead of removing it.
#
# The questioned window (genre G2) is scored under the candidate's grammar
# fitted on his GENRE-G1 text and under a cohort of impostor grammars fitted
# on OTHER authors' G1 text of the same size -- every model in the comparison
# crosses the same genre gap, so the gap cancels in the standardised statistic
#   t(Q) = (lam_cand(Q) - mean_j lam_imp_j(Q)) / sd_j(lam_imp_j(Q)).
# A single fixed G2 reference donor supplies the (common, cancelling)
# denominator of each lambda; r=1 throughout.
#
#   python experiments/run_ximpostor.py --direction novels2poetree
#
# Output: scores/xgenre/{dir}__impostor__K{KT}.jsonl, one row per case:
#   {"cand","quest","qw","within","lam_cand","lam_imp":[...]}

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import rechunk  # noqa: E402
from run_xgenre import GERMAN, load_bank, namekey, window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

SCORES = HERE.parent / "scores" / "xgenre"
KT = 10000        # known-document tokens (candidate and impostors alike)
N_IMP = 20        # impostor grammars per direction
SAME_W = 3
NCROSS = 4
L = 1000


def run(g1, g2, seg=0):
    tag = f"{g1}2{g2}__impostor__K{KT}"
    wtag = f"w{seg}" if seg else ""
    fn = SCORES / f"{tag}{wtag}.jsonl"
    if fn.exists():
        print(f"{tag}: exists, skipped", flush=True)
        return
    b1, b2 = load_bank(GERMAN[g1]), load_bank(GERMAN[g2])
    k1 = {namekey(a): a for a in b1}
    k2 = {namekey(a): a for a in b2}
    shared = sorted(k for k in k1 if k in k2 if b2[k2[k]]["ntok"] >= L)
    # impostors: G1 authors who are not cross-genre case authors, >= KT tokens
    imp_pool = sorted(a for a in b1
                      if namekey(a) not in shared and b1[a]["ntok"] >= KT)
    rng = random.Random(f"{tag}|impostors")
    imps = sorted(rng.sample(imp_pool, min(N_IMP, len(imp_pool))))
    # fixed common reference donor: largest non-case G2 author
    refa = max((a for a in b2 if namekey(a) not in shared),
               key=lambda a: b2[a]["ntok"])
    refdoc = window(b2[refa]["sents"], 0, KT)
    print(f"{tag}: {len(shared)} candidates, {len(imps)} impostors, "
          f"ref={refa}", flush=True)

    kdoc = {a: window(b1[a]["sents"], 0, min(b1[a]["ntok"], KT))
            for a in imps}
    for k in shared:
        kdoc[k1[k]] = window(b1[k1[k]]["sents"], 0,
                             min(b1[k1[k]]["ntok"], KT))
    qwins = {a: [window(b2[a]["sents"], i * L, L)
                 for i in range(min(b2[a]["ntok"] // L, SAME_W))]
             for a in b2 if b2[a]["ntok"] >= L}

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    if seg:   # the segmentation protocol: every stream in w-token units
        kdoc = {a: rechunk(s, seg) for a, s in kdoc.items()}
        refdoc = rechunk(refdoc, seg)

    def lam(q, ka):
        return lg.score(rechunk(q, seg), kdoc[ka], ref_sentences=refdoc,
                        r=1, with_details=False).lambda_G

    t0, rows = time.time(), []
    for k in shared:
        a1, a2 = k1[k], k2[k]
        cases = [(a2, j, 1) for j in range(len(qwins.get(a2, [])))]
        prng = random.Random(f"{tag}|{a1}|cross")
        cand = [b for b in qwins if b != a2]
        for b in prng.sample(cand, min(NCROSS, len(cand))):
            cases.append((b, prng.randrange(len(qwins[b])), 0))
        for (qa, qw, within) in cases:
            q = qwins[qa][qw]
            rows.append({"cand": a1, "quest": qa, "qw": qw, "within": within,
                         "lam_cand": round(lam(q, a1), 3),
                         "lam_imp": [round(lam(q, j), 3) for j in imps]})
        print(f"  {tag} {a1}: {len(rows)} cases, {time.time()-t0:.0f}s",
              flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{tag}: {len(rows)} cases in {time.time()-t0:.0f}s", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, help="e.g. novels2poetree")
    ap.add_argument("--seg", type=int, default=0)
    args = ap.parse_args()
    g1, g2 = args.direction.split("2")
    run(g1, g2, args.seg)


if __name__ == "__main__":
    main()
