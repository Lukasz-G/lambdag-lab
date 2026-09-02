# KN-LambdaG driver for the multilingual grid.
#
# Staged design (agreed with the user, deliberately NOT a full cross-product -- 3 engines
# x 4 segmentations x 5 lengths x 44 datasets would be ~2700 runs and a multiple-comparisons
# trap):
#   PRIMARY     engine=kn, segmentation=sentences, all datasets, all lengths
#   ABLATION 1  segmentation in {sent, w10, w20, w30}  -- run per GENRE, since the mechanism
#               ("sentence" is a dubious unit in verse and stichomythia) predicts windows
#               should win outright on poetry/drama, not only at short lengths as measured
#               for German novels
#   ABLATION 2  engine in {kn, hpy, ppmd*}  -- on a language subset chosen for the
#               morphology contrast (rich: hu, lv, lt, cs, ru, pl; analytic: en, fr)
#
# Lengths are SYMMETRIC: both the questioned and the known text are truncated to L tokens
# (L=0 means "full"), because that is the honest "how little text do we need" question.
#
# Every score file carries lambda_G AND the two model-free corrections of Barlow, Nini &
# Manino (2026), so calibration-free LLRs are available for languages that have no
# calibration corpus.
#
#   python experiments/run_kn_grid.py --stage primary
#   python experiments/run_kn_grid.py --stage segmentation --genres poetree,dracor
#   python experiments/run_kn_grid.py --stage engines --engines kn,hpy,ppmd_static
#
# Output: experiments/scores/{dataset}__{engine}__{seg}__L{len}.jsonl  + results.csv

import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, cllr, cllr_min, sqrt_correction, hapax_correction

MASKED = ROOT / "masked"
SCORES = HERE / "scores"; SCORES.mkdir(parents=True, exist_ok=True)

LENGTHS = [0, 1200, 600, 300, 150]          # 0 = full text, otherwise symmetric truncation
SEGMENTS = {"sent": 0, "w10": 10, "w20": 20, "w30": 30}
# engine label -> kwargs for LambdaG(engine=..., engine_kwargs=...)
ENGINES = {
    "kn":          ("kn", {}),
    "hpy":         ("hpy", {}),
    "ppmd_static": ("ppmd", {"adaptive": False}),
    "ppmd_sent":   ("ppmd", {"adaptive": True, "reset": "sentence"}),
    "ppmd_doc":    ("ppmd", {"adaptive": True, "reset": "document"}),
}
# HPY parameter sweep (journal engines chapter): theta = concentration,
# min/exp = table estimator, dXX = discount. theta=0 + minimal + d=0.75 == KN
# exactly (the oracle), so every variant below probes one departure from KN.
for _th in (0, 1, 5, 10):
    for _est in ("minimal", "expected"):
        ENGINES[f"hpy_t{_th}_{_est[:3]}"] = (
            "hpy", {"concentration": float(_th), "table_estimator": _est,
                    "discount": 0.75})
for _d in (0.5, 0.9):
    ENGINES[f"hpy_t0_min_d{int(_d*100)}"] = (
        "hpy", {"concentration": 0.0, "table_estimator": "minimal",
                "discount": _d})
    ENGINES[f"hpy_t1_exp_d{int(_d*100)}"] = (
        "hpy", {"concentration": 1.0, "table_estimator": "expected",
                "discount": _d})
# morphology contrast for the engine ablation
ENGINE_LANGS = ["hungarian", "latvian", "lithuanian", "czech", "russian", "polish",
                "english", "french"]
MIN_REF_AUTHORS = 10                        # datasets below this are reported, not analysed


def read_tsv(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            out.append(line.split("\t"))
    return out


def rechunk(sents, w):
    """Re-tile a masked token stream into fixed w-token pseudo-sentences (no re-masking)."""
    if not w:
        return sents
    toks = [t for s in sents for t in s]
    return [toks[i:i + w] for i in range(0, len(toks), w) if toks[i:i + w]]


def trunc(sents, L):
    if L <= 0:
        return sents
    out, n = [], 0
    for s in sents:
        if n + len(s) <= L:
            out.append(s); n += len(s)
        else:
            take = L - n
            if take > 0:
                out.append(s[:take])
            break
    return out


def datasets():
    for d in sorted(MASKED.iterdir()):
        if d.is_dir() and (d / "DONE").exists():
            info = json.loads((d / "DONE").read_text(encoding="utf-8"))
            yield d, info


def score_dataset(d, engine_label, seg, lengths, limit=None):
    engine, ekw = ENGINES[engine_label]
    w = SEGMENTS[seg]
    man = [l.rstrip("\n").split("\t") for l in open(d / "pairs.tsv", encoding="utf-8")][1:]
    if limit:
        man = man[:limit]

    ref = []
    for f in sorted((d / "bank").glob("*.tsv")):
        ref += rechunk(read_tsv(f), w)
    if not ref:
        return []

    lg = LambdaG(N=10, r=30, engine=engine, random_state=0,
                 **({"engine_params": ekw} if ekw else {}))
    lg.set_reference(ref)

    rows = []
    for L in lengths:
        fn = SCORES / f"{d.name}__{engine_label}__{seg}__L{L}.jsonl"
        if fn.exists():
            continue
        t0 = time.time(); recs = []
        for pid, lab, ka, qa in man:
            k = rechunk(read_tsv(d / "pairs" / f"{pid}_known.tsv"), w)
            q = rechunk(read_tsv(d / "pairs" / f"{pid}_q.tsv"), w)
            kt, qt = trunc(k, L), trunc(q, L)          # SYMMETRIC truncation
            if not kt or not qt:
                continue
            r = lg.score(qt, kt, with_details=False)
            recs.append({"id": int(pid), "label": int(lab), "lambda_G": r.lambda_G,
                         "n_q": r.n_query_tokens, "v1_q": r.n_query_hapax,
                         "sqrt": r.lambda_sqrt, "hapax": r.lambda_hapax})
        with open(fn, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        rows.append(summarise(d.name, engine_label, seg, L, recs, time.time() - t0))
    return rows


def summarise(name, engine, seg, L, recs, secs=0.0):
    from sklearn.metrics import roc_auc_score
    y = np.array([r["label"] for r in recs])
    out = {"dataset": name, "engine": engine, "seg": seg, "L": L, "n": len(recs),
           "secs": round(secs)}
    if len(set(y.tolist())) < 2:
        return out
    for key in ("lambda_G", "sqrt", "hapax"):
        s = np.array([r[key] for r in recs], dtype=float)
        out[f"auc_{key}"] = round(float(roc_auc_score(y, s)), 4)
        out[f"cllr_{key}"] = round(float(cllr(s[y == 1], s[y == 0])), 4)
        out[f"cllrmin_{key}"] = round(float(cllr_min(s[y == 1], s[y == 0])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="primary", choices=["primary", "segmentation", "engines"])
    ap.add_argument("--genres", default="", help="restrict to these genres")
    ap.add_argument("--langs", default="")
    ap.add_argument("--engines", default="")
    ap.add_argument("--segs", default="", help="restrict segmentation stage to these units")
    ap.add_argument("--lengths", default="")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else LENGTHS
    genres = {g.strip() for g in args.genres.split(",") if g.strip()}
    langs = {l.strip() for l in args.langs.split(",") if l.strip()}

    if args.stage == "primary":
        engines, segs = ["kn"], ["sent"]
    elif args.stage == "segmentation":
        engines = ["kn"]
        segs = ([s.strip() for s in args.segs.split(",") if s.strip()]
                or list(SEGMENTS))
    else:
        engines = [e.strip() for e in args.engines.split(",")] or ["kn", "hpy", "ppmd_static"]
        segs = ["sent"]
        if not langs:
            langs = set(ENGINE_LANGS)

    rows, skipped = [], []
    for d, info in datasets():
        lang, _, corpus = d.name.rpartition("_")
        if genres and corpus not in genres:
            continue
        if langs and lang not in langs:
            continue
        if info.get("ref", 0) < MIN_REF_AUTHORS:
            skipped.append((d.name, info.get("ref", 0))); continue
        for engine_label in engines:
            for seg in segs:
                try:
                    rows += score_dataset(d, engine_label, seg, lengths, args.limit_pairs or None)
                except Exception as e:
                    print(f"  !! {d.name} {engine_label}/{seg}: {type(e).__name__}: {str(e)[:90]}",
                          flush=True)
                for r in rows[-len(lengths):]:
                    if r.get("auc_lambda_G") is not None:
                        print(f"  {r['dataset']:24s} {r['engine']:11s} {r['seg']:4s} "
                              f"L={str(r['L'] or 'full'):>4s}  AUC {r['auc_lambda_G']:.3f}  "
                              f"Cllr {r['cllr_lambda_G']:8.3f}  sqrt {r['cllr_sqrt']:.3f}  "
                              f"hapax {r['cllr_hapax']:.3f}  ({r['secs']}s)", flush=True)

    if skipped:
        print(f"\nexcluded ({MIN_REF_AUTHORS}+ reference authors required): "
              + ", ".join(f"{n} ({r})" for n, r in skipped))
    if rows:
        import csv
        out = HERE / f"results_{args.stage}.csv"
        keys = sorted({k for r in rows for k in r})
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"\n{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
