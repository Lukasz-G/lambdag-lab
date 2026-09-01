# Install the taggers. Both spaCy's and Stanza's downloaders make their own HTTPS
# calls, which fail on this machine's intercepting proxy -- so inject the OS trust
# store FIRST (see _net.py), then call their download APIs in-process.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _net  # noqa: F401  (imports truststore and injects it)

SPACY = ["en_core_web_lg", "es_core_news_lg", "it_core_news_lg", "pt_core_news_lg",
         "ro_core_news_lg", "ru_core_news_lg", "uk_core_news_lg", "sl_core_news_lg",
         "hr_core_news_lg", "el_core_news_lg", "lt_core_news_lg", "nl_core_news_lg",
         "sv_core_news_lg", "nb_core_news_lg"]
STANZA = ["cs", "hu", "sr", "lv"]

import importlib.util
from spacy.cli.download import download as spacy_download

for m in SPACY:
    if importlib.util.find_spec(m):
        print(f"have {m}", flush=True); continue
    try:
        print(f"=== downloading {m}", flush=True)
        spacy_download(m)
        print(f"ok {m}", flush=True)
    except SystemExit:
        print(f"ok {m} (installer exited)", flush=True)
    except Exception as e:
        print(f"FAILED {m}: {type(e).__name__} {str(e)[:100]}", flush=True)

try:
    import stanza
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "stanza", "--quiet"])
    import stanza
for lg in STANZA:
    try:
        stanza.download(lg, processors="tokenize,pos,lemma", verbose=False)
        print(f"stanza ok: {lg}", flush=True)
    except Exception as e:
        print(f"stanza FAILED {lg}: {type(e).__name__} {str(e)[:100]}", flush=True)
print("ALL TAGGERS DONE", flush=True)
