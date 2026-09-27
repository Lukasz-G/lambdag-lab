# The normalised-edition corpus: many named authors, several works each.
#
# WHY THIS SOURCE. Every witness corpus this repository holds is organised by
# MANUSCRIPT: the Kaiserchronik in fifty copies, Parzival in eighty-nine, Iwein
# in thirty. That is the right shape for asking what a scribe does to a text and
# the wrong shape for asking whether an author can be recognised at all, because
# the same-author class then collapses onto the two or three poets whose works
# happen to survive in several books -- in practice Hartmann alone. The Middle
# High German Conceptual Database publishes one normalised text per WORK with a
# lemma and a part of speech on every word, which is the complementary shape:
# twenty-six named authors with two or more works, and no manuscript dimension at
# all.
#
# WHAT IS AND IS NOT COMPARABLE. These are editors' normalised texts, not
# diplomatic transcriptions, so a profile taken from them may not be set beside a
# profile taken from a witness bank -- the orthography has been regularised by a
# different hand for a different purpose. They are used for the author axis on
# their own terms and kept in their own bank.
#
# WHAT IS FETCHED. `authority-files/works.xml` for the work-author register,
# `authority-files/lexicon.xml` for the lemma forms the texts point at, and the
# TEI of every work by a named author who has at least two. Files already on disk
# are left alone, so the run may be repeated.
#
#   python medieval/fetch_mhdbdb.py --list
#   python medieval/fetch_mhdbdb.py
#
# Data: CC BY-NC-SA 4.0 (analysis, not redistribution), MHDBDB, Salzburg;
# repository DigitalHumanitiesCraft/mhdbdb-tei-only.
#
# Output: medieval/raw/mhdbdb/{works.xml,lexicon.xml,tei/<SIGLE>.tei.xml}

import argparse
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw" / "mhdbdb"
BASE = ("https://raw.githubusercontent.com/DigitalHumanitiesCraft/"
        "mhdbdb-tei-only/main")
NS = {"t": "http://www.tei-c.org/ns/1.0"}
UA = {"User-Agent": "authorship-verification-research/1.0"}


def get(url, dest, tries=4):
    """fetch once, keep what is already there, and say how big it was"""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size, True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as fh:
                data = fh.read()
            dest.write_bytes(data)
            return len(data), False
        except (urllib.error.URLError, TimeoutError) as e:
            if k == tries - 1:
                print(f"  FAILED {url}: {e}")
                return 0, False
            time.sleep(2 * (k + 1))
    return 0, False


def registry():
    """(sigle, author, title) for every work with a named author"""
    p = RAW / "works.xml"
    get(f"{BASE}/authority-files/works.xml", p)
    rows = []
    for b in ET.parse(p).getroot().iter("{%s}bibl" % NS["t"]):
        sig = b.find("t:idno[@type='sigle']", NS)
        au = b.find("t:author", NS)
        ti = b.find("t:title", NS)
        if sig is None or sig.text is None:
            continue
        a = (au.text or "").strip() if au is not None else ""
        if not a or "anonym" in a.lower():
            continue
        rows.append((sig.text.strip(), a,
                     (ti.text or "").strip() if ti is not None else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-works", type=int, default=2,
                    help="an author with fewer works than this cannot supply a "
                         "same-author pair across works, which is the whole "
                         "reason for this corpus")
    ap.add_argument("--only", default="", help="substring of the author's name")
    ap.add_argument("--list", action="store_true",
                    help="print the panel and fetch nothing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    by = defaultdict(list)
    for sig, au, ti in registry():
        by[au].append((sig, ti))
    panel = {a: v for a, v in by.items() if len(v) >= args.min_works
             and (not args.only or args.only.lower() in a.lower())}
    n_works = sum(len(v) for v in panel.values())
    print(f"{len(panel)} named authors with {args.min_works}+ works, "
          f"{n_works} works")
    for a, v in sorted(panel.items(), key=lambda kv: -len(kv[1])):
        print(f"  {a:36} {len(v):3} works")
    if args.list:
        return

    get(f"{BASE}/authority-files/lexicon.xml", RAW / "lexicon.xml")
    print(f"  lexicon.xml {(RAW / 'lexicon.xml').stat().st_size / 1e6:.1f} MB")
    done = new = 0
    total = 0
    for a, v in sorted(panel.items()):
        for sig, ti in sorted(v):
            n, cached = get(f"{BASE}/tei/{sig}.tei.xml",
                            RAW / "tei" / f"{sig}.tei.xml")
            total += n
            done += 1
            new += not cached
            if done % 20 == 0 or done == n_works:
                print(f"  {done:4}/{n_works} works, {total / 1e6:7.1f} MB "
                      f"({new} fetched)", flush=True)
    print(f"\n{done} works in {RAW.relative_to(ROOT)}/tei, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
