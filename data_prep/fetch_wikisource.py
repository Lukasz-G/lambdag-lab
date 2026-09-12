# French Wikisource -> per-letter JSONL for named epistolary works.
#
# WHY. Project Gutenberg's French shelf carries Sand's correspondence but not
# Flaubert's, whose Conard edition (1926, nine tomes) lives on fr.wikisource as
# one subpage PER LETTER -- 1,990 of them -- with the addressee in every page
# header and the tomes in chronological order. Letters are the register that
# joins this corpus to the forensic use case, and per-letter structure is what
# the German source could not supply at all.
#
# WHAT IS TAKEN, AND WHAT IS REFUSED. Only works listed in WORKS, which name
# single-author letter editions: a "Correspondance entre X et Y" is both-sided
# and enters only through a sender split that this fetcher deliberately does
# not attempt. Salutations and closings are LEFT IN PLACE but counted, so the
# banking step downstream can strip them with the dose on record -- formulae
# are function-word-heavy, POSNoise keeps them, and unlogged stripping is how
# an inflation artefact becomes invisible.
#
#   python data_prep/fetch_wikisource.py
#
# Output: data_prep/raw/wikisource/fr_ws_letters.jsonl
#         one record per LETTER: {author_id, author_name, id_source, work,
#         tome, letter_no, addressee, text}

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get  # noqa: E402

RAW = HERE / "raw" / "wikisource"
CACHE = HERE / "cache" / "wikisource"
API = "https://fr.wikisource.org/w/api.php"

WORKS = [
    ("flaubert_gustave", "Flaubert, Gustave",
     "Correspondance de Gustave Flaubert/"),
]

SALUT = re.compile(r"^(mon |ma |cher|chère|bien |vieux|pauvre|carissimo|mon "
                   r"bon|adieu|je t'embrasse|mille (baisers|tendresses))", re.I)


def api(params):
    u = API + "?format=json&formatversion=2&" + params
    return get(u, as_json=True, timeout=60)


def letter_pages(prefix):
    out, cont = [], ""
    while True:
        d = api("action=query&list=allpages&aplimit=500&apprefix="
                + prefix.replace(" ", "%20") + cont)
        ps = (d or {}).get("query", {}).get("allpages", [])
        out += [p["title"] for p in ps if re.search(r"/\d{3,4}$", p["title"])]
        c = (d or {}).get("continue", {}).get("apcontinue")
        if not c:
            return sorted(out)
        cont = "&apcontinue=" + c.replace(" ", "%20").replace("&", "%26")


def parsed(title):
    key = re.sub(r"[^A-Za-z0-9]+", "_", title)[-90:]
    cf = CACHE / f"{key}.html"
    if cf.exists():
        return cf.read_text(encoding="utf-8", errors="replace")
    d = api("action=parse&prop=text&page=" + title.replace(" ", "%20")
            .replace("&", "%26"))
    html = (d or {}).get("parse", {}).get("text", "")
    if html:
        cf.write_text(html, encoding="utf-8")
        time.sleep(0.5)
    return html


def clean(html):
    """Letter text + addressee from one parsed page.

    The page opens with a navigation/metadata block (edition, volume, page
    numbers, prev/next arrows, and the addressee line 'À <person>'); the block
    ends where the letter's own dateline or salutation begins. The addressee is
    captured from the header's own 'À ...' title line rather than guessed.
    """
    html = re.sub(r"<style.*?</style>|<script.*?</script>", " ", html, flags=re.S)
    # the header table/divs carry class attributes; drop whole nav containers
    html = re.sub(r'<div class="ws-noexport.*?</div>', " ", html, flags=re.S)
    html = re.sub(r'<table[^>]*class="[^"]*(headertemplate|ws_summary)[^"]*".*?'
                  r"</table>", " ", html, flags=re.S)
    addressee = ""
    m = re.search(r"<h[23][^>]*>\s*(?:<[^>]+>\s*)*À\s+([^<]{2,60})", html)
    if m:
        addressee = m.group(1).strip()
    txt = re.sub(r"<br\s*/?>", "\n", html)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = (txt.replace("&#160;", " ").replace("&nbsp;", " ")
              .replace("&amp;", "&").replace("&#x25c4;", " ")
              .replace("&#x25ba;", " "))
    txt = re.sub(r"[ \t]+", " ", txt)
    lines = [l.strip() for l in txt.splitlines()]
    # drop the residual header run: everything up to and including the last
    # early line that repeats edition metadata or the addressee title
    body = []
    started = False
    for i, l in enumerate(lines):
        if not started:
            if not l or re.search(r"Conard|Correspondance|Volume \d|Tome |"
                                  r"^À |◄|►|book |Paris", l):
                continue
            started = True
        body.append(l)
    return addressee, "\n".join(body).strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = RAW / "fr_ws_letters.jsonl"
    n_out = 0
    with open(out, "w", encoding="utf-8") as fh:
        for aslug, aname, prefix in WORKS:
            pages = letter_pages(prefix)
            print(f"{aname}: {len(pages)} letter pages", flush=True)
            words = salv = 0
            for i, title in enumerate(pages):
                html = parsed(title)
                if not html:
                    continue
                addressee, text = clean(html)
                w = len(text.split())
                if w < 20:
                    continue
                if SALUT.match(text):
                    salv += 1
                m = re.search(r"/Tome ([^/]+)/(\d{3,4})$", title)
                fh.write(json.dumps({
                    "author_id": aslug, "author_name": aname,
                    "id_source": f"wikisource-fr:{title}",
                    "work": prefix.rstrip("/"),
                    "tome": m.group(1) if m else "",
                    "letter_no": m.group(2) if m else "",
                    "addressee": addressee, "text": text},
                    ensure_ascii=False) + "\n")
                n_out += 1
                words += w
                if (i + 1) % 200 == 0:
                    print(f"  ...{i + 1}/{len(pages)}  {words:,} words",
                          flush=True)
            print(f"  {aname}: {n_out} letters, {words:,} words, "
                  f"{salv} opening with a salutation formula", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
