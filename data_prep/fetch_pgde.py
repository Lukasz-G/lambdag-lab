# Projekt Gutenberg-DE -> per-work JSONL, in the schema the other fetchers emit.
#
# WHY THIS SOURCE, AND WHY ONLY NAMED WORKS. The TextGrid harvest cannot feed
# the self-writing domain (its purity audit found ONE letters author above
# 20k), and gutenberg.org is near-barren for German memoirs -- its sole find
# for our authors being letters written TO Ludwig Tieck by other people, which
# is an identity trap, not a source. Projekt Gutenberg-DE holds the shelf, but
# its title keywords are treacherous: half of the keyword matches for our
# authors are FICTION wearing memoir clothes (Heine's Schnabelewopski, Keller's
# Liebesbriefe novella, Tieck's epistolary novel, Wieland's verse epistles).
# So this fetcher takes an explicit list of works whose kind has been decided
# by a person, and refuses to crawl by keyword.
#
# ACCESS PATH. The current site front end is a WordPress application whose
# catalogue sits behind nonce-gated AJAX, but the LEGACY per-work pages
# ({authordir}/{workdir}/*.html) are still served statically. The work list
# used to choose paths came from the Wayback copy of the legacy index; the
# texts themselves are fetched LIVE.
#
#   python data_prep/fetch_pgde.py
#
# Output: data_prep/raw/pgde/german_pgde_records.jsonl
#         {author_id, author_name, id_source, domain, text} -- one per work,
#         with per-work audit counters printed (verse-line share, orthography
#         markers, size), because a second edition tradition is entering banks
#         built from a first one.

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get  # noqa: E402
from fetch_textgrid import normalise_historical  # noqa: E402

RAW = HERE / "raw" / "pgde"
CACHE = HERE / "cache" / "pgde"
BASE = "https://www.projekt-gutenberg.org"

# Hand-audited work list (2026-09-10): title-level kind decisions made by a
# person against the Wayback legacy index; the per-text audit below still runs.
WORKS = [
    ("thoma_ludwig", "Thoma, Ludwig", "thoma", "autobio", "memoir"),
    ("keller_gottfried", "Keller, Gottfried", "keller", "tagebuch", "diary"),
    ("tieck_ludwig", "Tieck, Ludwig", "tieck", "tagebuch", "diary"),
]

MARK = re.compile(
    r"^(th(eil|al|ur|un|at|eur)\w*|seyn|sey|seynd|ihro|itzt|jtzt|ietzt"
    r"|ohnerachtet|dieweil|allhier|derjenige\w*|weyl|meyn\w*|deyn\w*"
    r"|freyheit\w*|frey|beyde\w*|bey|zwey\w*|drey\w*|\w+irt|\w+iret|\w+iren)$",
    re.I)


WB = "https://web.archive.org/web/2020id_/"


def is_chrome(t):
    """The WordPress front end answers dead legacy paths with HTTP 200 and its
    own chrome; treat any page carrying the app skeleton as not-the-text."""
    return t is None or "wp-json" in t or "gutenberg-library" in t


def page(url):
    cf = CACHE / (re.sub(r"[^a-z0-9]+", "_", url.lower())[-80:] + ".html")
    if cf.exists():
        t = cf.read_text(encoding="utf-8", errors="replace")
        if not is_chrome(t):
            return t
    t = get(url, as_json=False, timeout=45)
    if is_chrome(t):
        # the 2020 Wayback crawl holds the full legacy tree; id_ returns the
        # original bytes without the archive's own banner markup
        t = get(WB + url, as_json=False, timeout=90)
        time.sleep(0.6)
        if is_chrome(t):
            return None
    cf.write_text(t, encoding="utf-8")
    time.sleep(0.3)
    return t


def chapters(ad, wd):
    """Every page of the work, in first-link order from the entry page.

    Legacy works split chapters into sibling files in the same directory; the
    entry page links them (and a 'weiter' chain repeats them). Order follows
    first appearance, which follows the table of contents.
    """
    entry = f"{BASE}/{ad}/{wd}/{wd}.html"
    t = page(entry)
    if t is None:
        return []
    seen, order = {f"{wd}.html"}, [f"{wd}.html"]
    # wayback-served pages carry archive-prefixed absolute links; reduce every
    # href to its final same-directory filename before deciding membership
    for m in re.finditer(r'href="([^"]*?([a-z0-9_-]+\.html?))"', t):
        full, h = m.group(1), m.group(2)
        if ("/" + ad + "/" + wd + "/") not in full and not re.fullmatch(
                r"[a-z0-9_-]+\.html?", full):
            continue
        if h not in seen:
            seen.add(h)
            order.append(h)
    return [f"{BASE}/{ad}/{wd}/{h}" for h in order]


def extract(html):
    """Paragraphs and headings from a legacy page, nav and boilerplate dropped."""
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html,
                  flags=re.S | re.I)
    body = re.search(r"<body.*?>(.*)</body>", html, re.S | re.I)
    if body:
        html = body.group(1)
    out = []
    for m in re.finditer(r"<(p|h[1-4])\b[^>]*>(.*?)</\1>", html, re.S | re.I):
        txt = re.sub(r"<br\s*/?>", "\n", m.group(2), flags=re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"&nbsp;?", " ", txt)
        txt = re.sub(r"&amp;?", "&", txt)
        txt = re.sub(r"[ \t]+", " ", txt).strip()
        if not txt or len(txt) < 3:
            continue
        # navigation and licence boilerplate on every legacy page
        if re.match(r"(<<?\s*zurück|weiter\s*>?>|inhalt|impressum|projekt "
                    r"gutenberg|autoren|information zum projekt)", txt, re.I):
            continue
        out.append(txt)
    return out


def audit(paras):
    """Counters that decide whether the text may enter a bank."""
    toks = [t for p in paras for t in p.split()]
    n = len(toks)
    # verse embedded in prose shows as runs of short <br>-separated lines
    lines = [l for p in paras for l in p.split("\n") if l.strip()]
    short = sum(1 for l in lines if 2 <= len(l.split()) <= 8)
    mark = sum(1 for t in toks if t[:1].isalpha() and MARK.match(t))
    return {"tokens": n, "short_line_share": short / max(1, len(lines)),
            "markers_per_1k": 1000 * mark / max(1, n)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = []
    for slug, name, ad, wd, dom in WORKS:
        urls = chapters(ad, wd)
        paras = []
        for u in urls:
            t = page(u)
            if t:
                paras.extend(extract(t))
        text = normalise_historical("\n\n".join(paras))
        a = audit(paras)
        print(f"  {slug[:24]:26s} /{ad}/{wd}/ {len(urls):3d} pages "
              f"{a['tokens']:7,d} tokens  short-line {100*a['short_line_share']:4.1f}%  "
              f"markers {a['markers_per_1k']:5.2f}/1k")
        if a["tokens"] < 5000:
            print(f"    -> below floor, NOT emitted")
            continue
        recs.append({"author_id": slug, "author_name": name,
                     "id_source": f"pgde:{ad}/{wd}", "domain": dom,
                     "text": text})
    p = RAW / "german_pgde_records.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(recs)} records -> {p}")


if __name__ == "__main__":
    main()
