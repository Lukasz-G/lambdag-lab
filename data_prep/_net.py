# Shared HTTP helper for the corpus fetchers.
#
# NB: on this machine an SSL-intercepting proxy signs every certificate with a root
# CA that lives in the *Windows* trust store, which certifi's bundle knows nothing
# about -- so plain `requests` fails with CERTIFICATE_VERIFY_FAILED while curl works.
# `truststore` injects the OS trust store into Python's ssl module, which fixes the
# failure with certificate verification left ON (never use verify=False here).

import time

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:  # pip install truststore
    pass

import requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "LambdaG-CHR2027/1.0 (academic authorship verification)"


def get(url, as_json=True, tries=4, timeout=60, session=None):
    """GET with retries; returns parsed JSON / text, or None on a hard failure."""
    s = session or SESSION
    for k in range(tries):
        try:
            r = s.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json() if as_json else r.text
            if r.status_code in (400, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def download(url, dest, tries=3, timeout=600, session=None):
    """Stream a (large) file to `dest`; skips the download when dest already exists."""
    s = session or SESSION
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            with s.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    time.sleep(2 * (k + 1)); continue
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            return dest
        except requests.RequestException:
            time.sleep(2 * (k + 1))
    return None
