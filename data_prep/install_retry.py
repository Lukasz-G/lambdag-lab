# Retry loop for the tagger downloads: the intercepting proxy resets large transfers
# (WinError 10054), so a single pass leaves gaps. Each round only attempts what is
# still missing; models already installed are skipped instantly.
import importlib, importlib.util, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _net  # injects the OS trust store

SPACY = ["en_core_web_lg", "es_core_news_lg", "it_core_news_lg", "pt_core_news_lg",
         "ru_core_news_lg", "uk_core_news_lg", "sl_core_news_lg", "hr_core_news_lg",
         "lt_core_news_lg", "ro_core_news_lg", "el_core_news_lg", "nl_core_news_lg",
         "sv_core_news_lg", "nb_core_news_lg"]
STANZA = ["cs", "hu", "sr", "lv"]
ROUNDS = 6

for rnd in range(1, ROUNDS + 1):
    importlib.invalidate_caches()
    missing = [m for m in SPACY if not importlib.util.find_spec(m)]
    if not missing:
        print("all spaCy models present", flush=True); break
    print(f"--- round {rnd}: {len(missing)} spaCy models missing", flush=True)
    for m in missing:
        # pip handles resume/retries better than spacy.cli for these large wheels
        url = (f"https://github.com/explosion/spacy-models/releases/download/"
               f"{m}-3.8.0/{m}-3.8.0-py3-none-any.whl")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "--retries", "5", "--timeout", "120", url],
                           capture_output=True, text=True)
        importlib.invalidate_caches()
        print(("  ok " if importlib.util.find_spec(m) else "  fail ") + m, flush=True)
    time.sleep(3)

try:
    import stanza
    for rnd in range(1, ROUNDS + 1):
        left = []
        for lg in STANZA:
            try:
                stanza.download(lg, processors="tokenize,pos,lemma", verbose=False)
                print(f"  stanza ok: {lg}", flush=True)
            except Exception as e:
                print(f"  stanza retry {lg}: {type(e).__name__}", flush=True); left.append(lg)
        if not left:
            break
        STANZA = left
        time.sleep(5)
except ImportError:
    print("stanza not installed", flush=True)
print("RETRY DONE", flush=True)
