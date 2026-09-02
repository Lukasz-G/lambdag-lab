# Per-token KN surprisal-PROFILE features (the DeBERTa-gap idea): LambdaG computes logP for
# every token and then sums it away; the profile SHAPE is dense, graded, order-sensitive
# evidence against the best prior we own -- exactly what tiny fragments need.
#
# Prior = ONE pooled KN model (N=10, D=0.75) fit on the phase1/bank reference corpus (authors
# disjoint from both the 28 TM-training authors and the 500 test pairs). Each fragment/document
# gets a 23-scalar profile of its per-token logP vector: 11 distribution stats + 12 bin rates.
#
# Outputs (TSV, parsed by tm_scale.jl when P4_PROF=1):
#   phase4/prof_authors.tsv : side(k|q)  author_idx0  frag_idx0  s1..s23   (slicing order MUST
#       mirror tm_scale.jl: authors sorted by filename; frags = concat over KLENS/QLENS scales)
#   phase4/prof_test.tsv    : pair_id  L  side(k|q)  s1..s23   (L=0 full; L>0 both sides
#       truncated to L tokens, the symmetric protocol)
#
#   python phase4/prof_feats.py

import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import Vocabulary, KNGrammarModel

KLENS = [150, 300, 600, 1200, 3000]; QLENS = [150, 300, 600, 1200]
KSTRIDE = 2.0; QSTRIDE = 2.0
EVALLENS = [0, 1200, 600, 300, 150]
BINS = [-14, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1]   # 12 rates: (-inf,-14],(-14,-12],..,(-1,inf)


def read_tsv(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        if line: out.append(line.split("\t"))
    return out


def fragments(sents, tok_target, stride_frac):          # exact port of tm_scale.jl
    frags = []
    buf = [(s, len(s)) for s in sents]
    i = 0
    while i < len(buf):
        cur = []; n = 0; j = i
        while j < len(buf) and n < tok_target:
            cur.append(buf[j][0]); n += buf[j][1]; j += 1
        if n >= tok_target // 2: frags.append(cur)
        if j >= len(buf): break
        i += max(1, round((j - i) * stride_frac))
    return frags


def trunc(sents, L):
    if L <= 0: return sents
    out, n = [], 0
    for s in sents:
        if n + len(s) <= L: out.append(s); n += len(s)
        else:
            t = L - n
            if t > 0: out.append(s[:t])
            break
    return out


def profile(lp):
    if lp.size == 0: return np.zeros(23)
    q = np.quantile(lp, [0.10, 0.25, 0.50, 0.75, 0.90])
    stats = [lp.mean(), lp.std(), lp.min(), *q,
             float((lp < -12).mean()), float((lp < -8).mean()), float((lp > -2).mean())]
    hist = np.histogram(lp, bins=[-np.inf, *BINS, np.inf])[0] / lp.size
    return np.array([*stats, *hist])


def main():
    ref = []
    for f in sorted((ROOT / "phase1" / "bank").glob("*.tsv")): ref += read_tsv(f)
    vocab = Vocabulary()
    store = vocab.encode(ref, grow=True)
    t0 = time.time()
    G = KNGrammarModel.fit(store, N=10, D=0.75)
    print(f"prior: KN N=10 on phase1/bank ({store.n_tokens} tokens, {time.time()-t0:.0f}s)", flush=True)

    def prof(sents):
        return profile(G.token_logprobs(vocab.encode(sents, grow=False)))

    afiles = sorted(p for p in (HERE / "authors").glob("*.tsv"))
    t0 = time.time(); nk = nq = 0
    with open(HERE / "prof_authors.tsv", "w", encoding="utf-8") as out:
        for a, p in enumerate(afiles):
            sents = read_tsv(p)
            tot = sum(len(s) for s in sents); cut = 0; n = 0
            for i, s in enumerate(sents):
                n += len(s)
                if n >= 0.6 * tot: cut = i + 1; break     # Julia's 1:cut inclusive
            for side, pool, lens, stride in (("k", sents[:cut], KLENS, KSTRIDE),
                                             ("q", sents[cut:], QLENS, QSTRIDE)):
                idx = 0
                for L in lens:
                    for f in fragments(pool, L, stride):
                        v = prof(f)
                        out.write(side + "\t" + str(a) + "\t" + str(idx) + "\t"
                                  + "\t".join(f"{x:.6g}" for x in v) + "\n")
                        idx += 1
                if side == "k": nk += idx
                else: nq += idx
    print(f"authors: {len(afiles)}, known frags {nk} (~{nk/len(afiles):.1f}/author), "
          f"q frags {nq} (~{nq/len(afiles):.1f}/author), {time.time()-t0:.0f}s", flush=True)

    man = []
    with open(ROOT / "phase3" / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t"); man.append(pid)
    pdir = ROOT / "phase3" / "pairs500"
    t0 = time.time()
    with open(HERE / "prof_test.tsv", "w", encoding="utf-8") as out:
        for pid in man:
            k = read_tsv(pdir / f"{pid}_known.tsv"); q = read_tsv(pdir / f"{pid}_q.tsv")
            if not k or not q: continue
            for L in EVALLENS:
                kk = prof(trunc(k, L) if L > 0 else k)
                qq = prof(trunc(q, L))
                out.write(f"{pid}\t{L}\tk\t" + "\t".join(f"{x:.6g}" for x in kk) + "\n")
                out.write(f"{pid}\t{L}\tq\t" + "\t".join(f"{x:.6g}" for x in qq) + "\n")
    print(f"test profiles written ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
