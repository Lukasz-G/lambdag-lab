# Parse the German harvest once, and keep only what an encoding needs.
#
# WHY THIS EXISTS. POSNoise runs with the parser switched off -- lambdag.py loads
# spaCy with disable=("parser", "ner") -- so nothing downstream knows how the
# words relate. A dependency alphabet does, and it cannot be derived from the
# masked banks: masking has already replaced content words with class sigils and
# dropped the alignment a parser needs. So the parse has to start from the raw
# harvest, and it is worth doing exactly once.
#
# WHAT IS PERSISTED, and why not spaCy docs. Three arrays per author and genre:
# the POSNoise symbol, the dependency relation, and the head as a RELATIVE offset.
# Relative offsets are what make the long records tractable -- a 2.4M-character
# prose record has to be parsed in chunks, and a relative head stays correct when
# the chunks are concatenated where an absolute index would not. The whole
# harvest is about 40M tokens, so this is roughly 400MB of int arrays against
# several GB of serialised docs, and every encoding variant and every corruption
# level is then a cheap re-derivation rather than a re-parse.
#
# THE GATE. The parser is itself genre-sensitive: verse parses worse than prose,
# and historical orthography worse than modern. If parse error is correlated with
# genre it manufactures a cross-genre penalty out of nothing, which is precisely
# the effect under test. So the same pass emits parse-quality proxies per author
# per genre -- the parser's own fallback label, sentence fragmentation, arc
# length, crossing arcs -- and those are read BEFORE any scoring is commissioned.
# Measured on ordinary hardware the parse costs about 6,000 tokens/s/core against
# 10,000 with the parser off, so the whole harvest is near 110 core-minutes.
#
#   python data_prep/parse_dep_banks.py                     # all three genres
#   python data_prep/parse_dep_banks.py --genres verse --procs 4
#   python data_prep/parse_dep_banks.py --proxies-only      # re-read the gate
#
# Output: parsed_dep/{genre}/{NNN}_{author}.npz   sym / dep / head / sent_len
#                                                 + the record's own vocab/deps
#         parsed_dep/parse_quality.tsv            the gate

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lambdag import POSNoiseMasker  # noqa: E402

RAW = HERE / "raw" / "textgrid"
OUT = HERE.parent / "parsed_dep"
GENRES = {"prose": "german_tgproseall", "verse": "german_tgverseall",
          "drama": "german_tgdramaall"}
MODEL = "de_core_news_lg"
# spaCy holds the whole Doc in memory and the parser's cost is superlinear in
# practice on very long inputs; 200k characters keeps peak memory flat without
# splitting often enough for the boundary to matter.
CHUNK = 200_000
NONPROJ_SENTS = 1500        # crossing arcs are O(n^2) per sentence, so sample


def chunks(text, cap=CHUNK):
    """Split on a paragraph break, else a sentence end, else a space.

    Never mid-token: a chunk boundary inside a word would produce two spurious
    tokens and one spurious arc, and those errors would land preferentially in
    the longest records, which are the prose ones.
    """
    out = []
    while len(text) > cap:
        cut = text.rfind("\n\n", 0, cap)
        if cut <= 0:
            cut = max(text.rfind(". ", 0, cap), text.rfind("! ", 0, cap),
                      text.rfind("? ", 0, cap))
            cut = cut + 1 if cut > 0 else text.rfind(" ", 0, cap)
        if cut <= 0:
            cut = cap
        out.append(text[:cut])
        text = text[cut:].lstrip()
    if text.strip():
        out.append(text)
    return out


def quality(doc):
    """Parse-quality counters for one chunk. Counts, not rates -- summed later."""
    q = Counter()
    n = len(doc)
    q["tok"] = n
    for t in doc:
        if t.is_space:
            continue
        q["tok_real"] += 1
        # spaCy's generic fallback label: the parser declining to commit.
        if t.dep_ in ("dep", ""):
            q["dep_fallback"] += 1
        q["arclen"] += abs(t.head.i - t.i)
    # NOT roots-per-sentence: spaCy derives doc.sents FROM the parse, so every
    # sentence has exactly one root by construction and the measure is vacuous.
    # What does vary is how far the parser's segmentation drifts from the
    # punctuation, and whether it produces runaway sentences -- a parser that
    # has lost the structure swallows whole passages into one tree.
    sents = 0
    for s in doc.sents:
        sents += 1
        n_s = len(s)
        q["sent_tok"] += n_s
        if n_s > 80:
            q["long_sent_tok"] += n_s
    q["sents"] = sents
    # Crossing arcs, sampled. German is genuinely non-projective at a low rate,
    # so this is read as a DIFFERENCE between genres, never as an absolute.
    seen = 0
    for s in doc.sents:
        if seen >= NONPROJ_SENTS:
            break
        arcs = [(min(t.i, t.head.i), max(t.i, t.head.i)) for t in s
                if t.head.i != t.i]
        if len(arcs) > 60:
            continue
        seen += 1
        q["nonproj_sents_seen"] += 1
        for a in range(len(arcs)):
            i, j = arcs[a]
            for b in range(a + 1, len(arcs)):
                k, l = arcs[b]
                if i < k < j < l or k < i < l < j:
                    q["crossing"] += 1
        q["arcs_seen"] += len(arcs)
    return q


def parse_record(nlp, masker, text):
    """One author-genre record -> (sym, dep, head_rel, sent_len, quality).

    The masker's own segmentation is authoritative: mask_doc() is called for the
    symbols and the sentence boundaries, and the relation arrays are built by a
    parallel walk over the SAME tokens with the SAME space filter. The two are
    then asserted equal in length, so a divergence is caught here rather than
    surfacing as a silent misalignment several experiments downstream.
    """
    # Vocabularies are PER RECORD. A single shared table would couple every
    # record to every other and force the parse to be one sequential job;
    # keeping the ids local costs a few kilobytes per file and lets the
    # harvest be sharded across processes with no coordination at all. The
    # encoder takes the union, where one process sees everything anyway.
    sym, dep, head, slen = [], [], [], []
    symtab, deptab = {}, {}
    q = Counter()
    for doc in nlp.pipe(chunks(text), batch_size=1):
        q += quality(doc)
        sents = masker.mask_doc(doc)
        # the punctuation-defined unit, for the segmentation-drift proxy
        q["mask_sents"] += len(sents)
        flat = [s for sent in sents for s in sent]

        pos = [masker._pos_of(t) for t in doc]
        keys = [masker._keys_of(t) for t in doc]
        raw = [t.text for t in doc]
        keep = masker._safe_mask(keys, pos, raw)
        kept, at = [], np.full(len(doc), -1, dtype=np.int64)
        for i, t in enumerate(doc):
            if (not t.text.strip()) or pos[i] == "SPACE" or t.pos_ == "SPACE":
                continue
            at[i] = len(kept)
            kept.append(i)
        if len(kept) != len(flat):
            raise RuntimeError(f"mask/parse misalignment: {len(kept)} vs {len(flat)}")

        base = len(sym)
        for j, i in enumerate(kept):
            s = flat[j]
            sid = symtab.get(s)
            if sid is None:
                sid = symtab[s] = len(symtab)
            d = doc[i].dep_ or "dep"
            did = deptab.get(d)
            if did is None:
                did = deptab[d] = len(deptab)
            # Heads that land on a dropped whitespace token are lifted to the
            # nearest kept ancestor; a token whose whole ancestry is dropped
            # becomes its own root rather than being silently reattached.
            h, hops = doc[i].head, 0
            while at[h.i] == -1 and h.head.i != h.i and hops < 16:
                h = h.head
                hops += 1
            hj = at[h.i]
            sym.append(sid)
            dep.append(did)
            head.append(0 if hj < 0 else int(hj - j))
        slen.extend(len(s) for s in sents)
        assert len(sym) - base == len(kept)
    voc = np.array([k for k, _ in sorted(symtab.items(), key=lambda kv: kv[1])],
                   dtype=object)
    dps = np.array([k for k, _ in sorted(deptab.items(), key=lambda kv: kv[1])],
                   dtype=object)
    return (np.array(sym, dtype=np.int32), np.array(dep, dtype=np.int16),
            np.array(head, dtype=np.int32), np.array(slen, dtype=np.int32),
            voc, dps, q)


def matched_quality(bins=((5, 9), (10, 19), (20, 39), (40, 79))):
    """The gate proper: parse quality compared at MATCHED sentence length.

    Read off the persisted arrays, so it costs nothing and needs no re-parse.
    Three questions, in the order they have to be asked:

    1. Does the parser decline more often in one genre? (the fallback rate, above)
    2. At the same sentence length, are its trees more tangled in one genre?
       A confused parser produces MORE crossing arcs, not fewer.
    3. Is a genre's tidiness instead a COLLAPSE -- a parser that has given up and
       returned a flat chain? That shows as adjacent attachment: a right-
       branching chain is nearly all |offset| = 1. This question exists because
       (2) answering "verse is tidier" is otherwise indistinguishable from
       "verse is not really being parsed", and the two have opposite meanings.
    """
    acc = {g: {b: np.zeros(5) for b in bins} for g in GENRES}
    for g in GENRES:
        d0 = OUT / g
        for f in sorted(d0.glob("*.npz")) if d0.exists() else []:
            z = np.load(f, allow_pickle=True)
            head, sl = z["head"], z["sent_len"]
            cuts = np.concatenate([[0], np.cumsum(sl)])
            for a, b in zip(cuts[:-1], cuts[1:]):
                n = int(b - a)
                for bb in bins:
                    if not (bb[0] <= n <= bb[1]):
                        continue
                    h = head[a:b]
                    arcs = [(min(j, j + int(h[j])), max(j, j + int(h[j])))
                            for j in range(n)
                            if h[j] != 0 and 0 <= j + int(h[j]) < n]
                    cr = 0
                    for x in range(len(arcs)):
                        i1, j1 = arcs[x]
                        for y in range(x + 1, len(arcs)):
                            k1, l1 = arcs[y]
                            if i1 < k1 < j1 < l1 or k1 < i1 < l1 < j1:
                                cr += 1
                    r = acc[g][bb]
                    r[0] += len(arcs)
                    r[1] += cr
                    r[2] += sum(j1 - i1 for i1, j1 in arcs)
                    r[3] += sum(1 for i1, j1 in arcs if j1 - i1 == 1)
                    r[4] += 1
                    break
    print("\nPARSE QUALITY AT MATCHED SENTENCE LENGTH (the gate proper)")
    print(f"  {'bin':>9s} {'genre':7s} {'cross/arc':>10s} {'arc len':>8s} "
          f"{'adjacent':>9s} {'sents':>9s}")
    for bb in bins:
        for g in GENRES:
            arcs, cr, al, adj, ns = acc[g][bb]
            if arcs < 1:
                continue
            print(f"  {str(bb):>9s} {g:7s} {cr / arcs:10.4f} {al / arcs:8.3f} "
                  f"{100 * adj / arcs:8.1f}% {int(ns):9d}")
    print("\n  A genre parses WORSE if its crossing rate is higher at matched "
          "length.\n  It is not being parsed AT ALL if adjacent attachment "
          "approaches 100%.\n  Only if one genre is worse on (2) without (3) is "
          "the corruption control\n  needed -- run encode_dep.py "
          "--corrupt-attach at the observed excess and\n  check that the "
          "within-genre trend survives.")


def collect_proxies():
    """Read the gate off every persisted record, not just this run's."""
    rows = []
    for g in GENRES:
        for f in sorted((OUT / g).glob("*.npz")) if (OUT / g).exists() else []:
            d = np.load(f, allow_pickle=True)
            if "qk" not in d:
                continue
            q = {str(k): int(v) for k, v in zip(d["qk"], d["qv"])}
            tr = max(1, q.get("tok_real", 0))
            rows.append({
                "genre": g, "author": f.stem.split("_", 1)[1],
                "tokens": int(len(d["sym"])),
                "dep_fallback_pct": 100.0 * q.get("dep_fallback", 0) / tr,
                "long_sent_pct": 100.0 * q.get("long_sent_tok", 0)
                                 / max(1, q.get("sent_tok", 0)),
                "mean_arclen": q.get("arclen", 0) / tr,
                "crossing_per_arc": q.get("crossing", 0) / max(1, q.get("arcs_seen", 0)),
                "sent_drift": (q.get("sent_tok", 0) / max(1, q.get("sents", 0)))
                              / max(1e-9, tr / max(1, q.get("mask_sents", 0)))})
    return rows


def write_proxies(rows, path):
    cols = ["genre", "author", "tokens", "dep_fallback_pct", "long_sent_pct",
            "mean_arclen", "crossing_per_arc", "sent_drift"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float)
                               else str(r[c]) for c in cols) + "\n")


def summarise(rows):
    """The gate, read out. A large genre gap here invalidates what follows."""
    print(f"\n{'PARSE QUALITY BY GENRE (the gate)':50s}")
    print(f"  {'genre':8s} {'n':>3s} {'dep-fallback':>13s} {'long sents':>11s} "
          f"{'arc len':>8s} {'crossing/arc':>13s} {'sent drift':>11s}")
    per = {}
    for g in GENRES:
        sub = [r for r in rows if r["genre"] == g]
        if not sub:
            continue
        m = {k: float(np.median([r[k] for r in sub])) for k in
             ("dep_fallback_pct", "long_sent_pct", "mean_arclen",
              "crossing_per_arc", "sent_drift")}
        per[g] = m
        print(f"  {g:8s} {len(sub):3d} {m['dep_fallback_pct']:12.2f}% "
              f"{m['long_sent_pct']:10.2f}% {m['mean_arclen']:8.2f} "
              f"{m['crossing_per_arc']:13.4f} {m['sent_drift']:11.2f}")
    # DO NOT read a dose off the table above. Every proxy in it except the
    # fallback rate scales with SENTENCE LENGTH, and the genres differ in
    # sentence length for reasons of language: prose sentences are longer than
    # dramatic ones, so prose shows longer arcs and more crossings whether or not
    # it is parsed any worse. Taken at face value the aggregate spread reaches
    # several hundred per cent and would prescribe a corruption dose that is
    # measuring German, not the parser.
    matched_quality()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genres", default="", help="comma-separated subset")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0, help="first N authors, for a smoke test")
    ap.add_argument("--shard", default="", help="k/N -- take every Nth record. "
                    "Records are independent and vocabularies are local, so N "
                    "processes with --shard 0/N ... (N-1)/N parse the harvest "
                    "in parallel with no locking and no shared state.")
    ap.add_argument("--proxies-only", action="store_true",
                    help="re-read parse_quality.tsv without parsing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.proxies_only:
        import csv
        rows = [{k: (float(v) if k not in ("genre", "author", "tokens") else v)
                 for k, v in r.items()}
                for r in csv.DictReader(open(OUT / "parse_quality.tsv",
                                             encoding="utf-8"), delimiter="\t")]
        summarise(rows)
        return

    import spacy
    # The parser is the point, so it is explicitly NOT disabled here; the
    # lemmatizer stays because the masker's key lookup can use it.
    nlp = spacy.load(MODEL, disable=["ner"])
    nlp.max_length = CHUNK * 3
    print(f"model {MODEL}  pipes {nlp.pipe_names}", flush=True)
    masker = POSNoiseMasker(language="de", nlp=nlp, lowercase=True,
                            disable=("ner",))

    rows = []
    todo = [g for g in (args.genres.split(",") if args.genres else GENRES) if g in GENRES]
    for g in todo:
        recs = [json.loads(l) for l in
                open(RAW / f"german_tg{g}_preprocessed.jsonl", encoding="utf-8")]
        recs.sort(key=lambda r: r["author_id"])
        if args.limit:
            recs = recs[:args.limit]
        idx = list(range(len(recs)))
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            idx = [i for i in idx if i % n == k]
        d = OUT / g
        d.mkdir(parents=True, exist_ok=True)
        for i in idx:
            r = recs[i]
            fn = d / f"{i:03d}_{r['author_id']}.npz"
            t0 = time.time()
            if fn.exists():
                print(f"  [{g}] {r['author_id'][:34]:34s} exists", flush=True)
                continue
            sym, dep, head, slen, voc, dps, q = parse_record(nlp, masker,
                                                             r["text"])
            np.savez_compressed(fn, sym=sym, dep=dep, head=head, sent_len=slen,
                                vocab=voc, deps=dps,
                                qk=np.array(sorted(q), dtype=object),
                                qv=np.array([q[k] for k in sorted(q)],
                                            dtype=np.int64))
            tr = max(1, q["tok_real"])
            rows.append({"genre": g, "author": r["author_id"], "tokens": len(sym),
                         "dep_fallback_pct": 100.0 * q["dep_fallback"] / tr,
                         "long_sent_pct": 100.0 * q["long_sent_tok"] / max(1, q["sent_tok"]),
                         "mean_arclen": q["arclen"] / tr,
                         "crossing_per_arc": q["crossing"] / max(1, q["arcs_seen"]),
                         "sent_drift": q["sent_tok"] / max(1, q["sents"])
                                       / max(1e-9, q["tok_real"] / max(1, q["mask_sents"]))})
            print(f"  [{g}] {r['author_id'][:34]:34s} {len(sym):8d} tok  "
                  f"{len(sym) / max(1e-9, time.time() - t0):7.0f} tok/s  "
                  f"fallback {rows[-1]['dep_fallback_pct']:5.2f}%  "
                  f"long {rows[-1]['long_sent_pct']:5.2f}%", flush=True)

    rows = collect_proxies()
    if rows:
        qf = OUT / "parse_quality.tsv"
        write_proxies(rows, qf)
        print(f"  {len(rows)} rows -> {qf.name}")
        summarise(rows)


if __name__ == "__main__":
    main()
