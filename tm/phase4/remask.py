# ENRICHED re-masking pass: one spaCy run (parser ON) over the three corpora of the best
# configuration, emitting per token FOUR fields so every downstream consumer picks its column:
#
#   cur    current POSNoise logic, byte-identical to the existing TSVs (verified below)
#   morph  cur + inflectional subtype on masked classes, from spaCy morph features ONLY
#          (project rule: machine-derived, nothing hand-written):
#            VERB/AUX Ø -> Ø_prs Ø_pst Ø_fin Ø_inf Ø_prt   (VerbForm/Tense)
#            ADJ @      -> @c @s                            (Degree=Cmp/Sup)
#            ADV ©      -> ©c ©s
#          kept (function) tokens are unchanged -- their surface IS the feature.
#   dep    UD dependency relation of the token
#   head   1-based position of the head token within the SAME emitted sentence, 0 if the
#          head lies outside it (or was dropped as space) -- arcs never cross sentences.
#
# Line format: sentences as TSV rows; token fields joined by the unit separator U+241F-free
# plain "|" (verified absent from all masked vocabularies).
#
# Corpora (selection logic replicated line-for-line from the original prep scripts):
#   phase4/authors_enr/   28 TM-training authors, sorted by name, tx[:P4_CHARS] (prep_authors.py)
#   phase4/bank_enr/      12 KN reference authors: by-length rank 1..12, tx[:200k] (phase1/prep.py)
#   phase4/pairs500_enr/  the 500 test pairs, known = larger text (phase3/prep_test500.py)
#
#   python phase4/remask.py          # ~30-60 min, prints per-corpus verification

import json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import POSNoiseMasker, _EOS_PUNCT_RE

REF = ROOT / "german" / "av_reference_novels_de.jsonl"
TEST = ROOT / "german" / "av_test_novels_de.jsonl"
CHARS = int(os.environ.get("P4_CHARS", "900000"))
BANK_CH = 200_000

def slug(a): return re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_")


def morph_token(cur, pos, t):
    """Inflectional subtype for a MASKED token, from spaCy morph features only."""
    if pos in ("VERB", "AUX"):                      # glyph Ø
        vf = t.morph.get("VerbForm"); tn = t.morph.get("Tense")
        if "Part" in vf: return cur + "_prt"
        if "Inf" in vf: return cur + "_inf"
        if "Fin" in vf:
            if "Past" in tn: return cur + "_pst"
            if "Pres" in tn: return cur + "_prs"
            return cur + "_fin"
        return cur
    if pos in ("ADJ", "ADV"):                       # glyphs @ / ©
        dg = t.morph.get("Degree")
        if "Cmp" in dg: return cur + "c"
        if "Sup" in dg: return cur + "s"
        return cur
    return cur


def mask_doc_enriched(m, doc):
    """Mirror of POSNoiseMasker.mask_doc (sentence segmenter), emitting 4-field records."""
    toks = [t for t in doc]
    raw = [t.text for t in toks]
    pos = [m._pos_of(t) for t in toks]
    keys = [m._keys_of(t) for t in toks]
    keep = m._safe_mask(keys, pos, raw)

    def is_space(i, t):
        return (not t.text.strip()) or pos[i] == "SPACE" or t.pos_ == "SPACE"

    sentences = []                                  # list of list of (cur,morph,dep,doc_idx)
    where = {}                                      # doc token idx -> (sent#, 1-based pos)
    cur_sent = []

    def close():
        nonlocal cur_sent
        if cur_sent:
            sentences.append(cur_sent); cur_sent = []

    for i, t in enumerate(toks):
        newline = "\n" in t.text or "\n" in t.whitespace_
        if not is_space(i, t):
            cur = m._emit(t, i, pos, keep)
            masked = (not keep[i]) and pos[i] in m.abbrev_pos_tags
            mo = morph_token(cur, pos[i], t) if masked else cur
            cur_sent.append((cur, mo, t.dep_, i, t.head.i))
            where[i] = (len(sentences), len(cur_sent))
            if _EOS_PUNCT_RE.match(t.text) and cur_sent:
                close()
        if newline and cur_sent:
            close()
    close()

    out = []
    for si, sent in enumerate(sentences):
        row = []
        for cur, mo, dep, i, hi in sent:
            h = where.get(hi, (None, 0))
            hpos = h[1] if (h[0] == si and hi != i) else 0     # 0 = root / outside sentence
            row.append(f"{cur}|{mo}|{dep}|{hpos}")
        out.append(row)
    return out


def write_enr(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if r: f.write("\t".join(r) + "\n")


def cur_column(rows):
    return [[f.split("|", 1)[0] for f in r] for r in rows if r]


def verify(enr_rows, tsv_path, name):
    want = [line.rstrip("\n").split("\t") for line in open(tsv_path, encoding="utf-8") if line.rstrip("\n")]
    got = cur_column(enr_rows)
    ok = got == want
    if not ok:
        nd = sum(1 for a, b in zip(got, want) if a != b) + abs(len(got) - len(want))
        print(f"  VERIFY FAIL {name}: {len(got)} vs {len(want)} sents, {nd} differing", flush=True)
    return ok


def main():
    recs = []
    with open(REF, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); recs.append((r["author"], r["text"]))

    print("building enriched masker (parser ON) ...", flush=True)
    m = POSNoiseMasker(language="de", spacy_model="de_core_news_lg", lowercase=True, disable=("ner",))

    t0 = time.time(); nver = nok = 0

    # ---- 28 TM-training authors (sorted by name, as prep_authors.py) ----
    outa = HERE / "authors_enr"; outa.mkdir(exist_ok=True)
    for i, (au, tx) in enumerate(sorted(recs, key=lambda r: r[0])):
        rows = mask_doc_enriched(m, m.nlp(tx[:CHARS]))
        assert not any("|" in f.split("|", 1)[0] for r in rows for f in r), f"'|' in token ({au})"
        fn = f"{i:02d}_{slug(au)}.tsv"
        write_enr(outa / fn, rows)
        old = HERE / "authors" / fn
        if old.exists():
            nver += 1; nok += verify(rows, old, f"authors/{fn}")
        if (i + 1) % 7 == 0:
            print(f"  authors {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"authors_enr done, verification {nok}/{nver} files identical ({time.time()-t0:.0f}s)", flush=True)

    # ---- 12 KN bank authors (by-length rank 1..12, as phase1/prep.py) ----
    outb = HERE / "bank_enr"; outb.mkdir(exist_ok=True)
    bylen = sorted(recs, key=lambda t: len(t[1]), reverse=True)
    nver = nok = 0
    for i, (au, tx) in enumerate(bylen[1:13]):
        rows = mask_doc_enriched(m, m.nlp(tx[:BANK_CH]))
        fn = f"{i:02d}_{slug(au)}.tsv"
        write_enr(outb / fn, rows)
        old = ROOT / "phase1" / "bank" / fn
        if old.exists():
            nver += 1; nok += verify(rows, old, f"bank/{fn}")
    print(f"bank_enr done, verification {nok}/{nver} files identical ({time.time()-t0:.0f}s)", flush=True)

    # ---- 500 test pairs (known = larger text, as prep_test500.py) ----
    ids = set()
    with open(ROOT / "phase3" / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            ids.add(line.split("\t", 1)[0])
    outp = HERE / "pairs500_enr"; outp.mkdir(exist_ok=True)
    nver = nok = ndone = 0
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if str(r["id"]) not in ids: continue
            t0_, t1_ = r["pair"]
            known, q = (t1_, t0_) if len(t1_) >= len(t0_) else (t0_, t1_)
            for side, txt in (("known", known), ("q", q)):
                rows = mask_doc_enriched(m, m.nlp(txt))
                write_enr(outp / f"{r['id']}_{side}.tsv", rows)
                old = ROOT / "phase3" / "pairs500" / f"{r['id']}_{side}.tsv"
                if old.exists():
                    nver += 1; nok += verify(rows, old, f"pairs500/{r['id']}_{side}")
            ndone += 1
            if ndone % 100 == 0:
                print(f"  pairs {ndone}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"pairs500_enr done ({ndone} pairs), verification {nok}/{nver} files identical", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
