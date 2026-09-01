"""
lambdag.py
==========

A self-contained, Numba-accelerated Python implementation of **LambdaG** (λG),
the Authorship Verification (AV) method described in:

    Nini, A., Halvani, O., Graner, L., Gherardi, V., & Ishihara, S. (2025).
    "Grammar as a Behavioral Biometric: Using Cognitively Motivated Grammar
    Models for Authorship Verification." arXiv:2403.08462v2.

and the topic-masking front end described in:

    Halvani, O., & Graner, L. (2021). "POSNoise: An Effective Countermeasure
    Against Topic Biases in Authorship Analysis." ARES '21.
    https://github.com/Halvani/POSNoise

Pipeline (Algorithm 1 of the paper)
-----------------------------------
    D_U, D_A, D_ref
        -> POSNoise topic masking            (POSNoiseMasker)
        -> sentence segmentation             (POSNoiseMasker, punctuation / newline)
        -> n-gram "Grammar Model"            (N=10; engine: kn | hpy | ppmd)
        -> G_A from S_A;  G_1..G_r from r bootstrap samples of S_ref of size |S_A|
        -> lambda_G(S_U) = sum_i sum_k (1/r) sum_j log10 P(t_k|t_<k; G_A)/P(t_k|t_<k; G_j)
        -> logistic-regression calibration    (LambdaGCalibrator)  -> Lambda_G (log10 LR)
        -> Cllr / Cllr_min                    (cllr, cllr_min)

Probability engines
-------------------
The paper estimates P(t_k | t_<k; G) with interpolated Kneser-Ney. Two alternatives
are provided behind the same interface, selectable via ``LambdaG(engine=...)``:

    "kn"    Interpolated Kneser-Ney, D = 0.75.        Sec. 6.3. The published method.
    "hpy"   Hierarchical Pitman-Yor process.          Teh (2006). KN is the special
                                                      case theta=0 + minimal tables,
                                                      so this is a strict
                                                      generalisation with power-law
                                                      (Zipfian) count behaviour.
    "ppmd"  Prediction by Partial Matching, escape D. The COAV engine (Sec. 1.4.3),
                                                      here over POSNoise function
                                                      tokens instead of characters.

Only the estimator changes: masking, sampling of the r reference models, the
log-ratio and the calibration are shared, so the three are directly comparable.

Design notes
------------
* **Numba** carries the two hot paths: (a) rolling 64-bit hashing of all n-gram
  windows during model construction, and (b) the per-token Kneser-Ney recursion,
  which for a single verification case runs
  ``|D_U| tokens x (r + 1) models x N levels x 2 binary searches``
  (~2M binary searches at r=100, N=10, |D_U|=1000). Pure Python needs minutes for
  that; the JIT kernels need well under a second. Cython would buy nothing extra
  here, so it is deliberately not used -- everything below is Numba or vectorised
  NumPy. The module degrades gracefully (but slowly) if Numba is absent.

* n-grams are keyed by a 64-bit SplitMix-based hash into **sorted** key arrays and
  looked up with binary search (the same trick KenLM uses). Collision probability
  at ~1e6 distinct keys is ~5e-8.

* Following the reference R implementation (`idiolect::lambdaG`, which drives
  `kgrams::language_model(smoother="kn", D=0.75)`), **each grammar model carries
  its own dictionary**, built from its own training sentences; tokens of D_U that
  are unseen by a given model map to <UNK>. Set ``vocab_mode="shared"`` to instead
  share one dictionary across G_A and all G_j.

Requirements
------------
    pip install numpy numba scikit-learn spacy posnoise
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
    # (or: python -m spacy download en_core_web_lg   -- the POSNoise default)

Usage examples are at the bottom of this file, in a commented-out block.
"""

from __future__ import annotations

import html as _html
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "NUMBA_AVAILABLE",
    "Vocabulary",
    "SentenceStore",
    "POSNoiseMasker",
    "windowize",
    "SUPPORTED_LANGUAGES",
    "load_hits_to_ud",
    "KNGrammarModel",
    "HPYGrammarModel",
    "PPMGrammarModel",
    "ENGINES",
    "LambdaG",
    "LambdaGResult",
    "LambdaGCalibrator",
    "cllr",
    "cllr_min",
    "cllr_decomposition",
    "heatmap_html",
    "UNK_ID",
    "EOS_ID",
    "BOS_ID",
]

# --------------------------------------------------------------------------- #
# 0. Optional Numba                                                            #
# --------------------------------------------------------------------------- #

try:  # pragma: no cover
    from numba import njit as _numba_njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def _numba_njit(*args, **kwargs):
        """No-op stand-in so the module still imports (and runs, slowly) w/o Numba."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator


def njit(*args, **kwargs):
    kwargs.setdefault("cache", True)
    kwargs.setdefault("nogil", True)
    return _numba_njit(*args, **kwargs)


# --------------------------------------------------------------------------- #
# 1. Special token ids                                                          #
# --------------------------------------------------------------------------- #
# Paper, Sec. 6.3.1: <UNK> in T, but <BOS>, <EOS> not in T. T* = T u {<EOS>}.

UNK_ID: int = 0
EOS_ID: int = 1
BOS_ID: int = 2
N_SPECIAL: int = 3

SPECIAL_TOKENS: Tuple[str, str, str] = ("<UNK>", "<EOS>", "<BOS>")


# --------------------------------------------------------------------------- #
# 2. 64-bit hashing kernels (SplitMix64)                                       #
# --------------------------------------------------------------------------- #
# NOTE: every constant is a np.uint64 *global* so that Numba treats it as a
# compile-time uint64 literal. Mixing a uint64 with a Python int inside nopython
# mode silently promotes to float64 -- that would be a correctness bug, hence the
# explicit casts everywhere below.

_H_INIT = np.uint64(0xCBF29CE484222325)
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_U1 = np.uint64(1)


@njit(inline="always")
def _mix64(x):
    """SplitMix64 finaliser."""
    z = x + _C1
    z = (z ^ (z >> _S30)) * _C2
    z = (z ^ (z >> _S27)) * _C3
    return z ^ (z >> _S31)


@njit(inline="always")
def _hstep(h, tok_id):
    """Absorb one token id into the running hash (left-to-right, order sensitive)."""
    return _mix64(h ^ (np.uint64(tok_id) + _U1))


@njit
def _bsearch(keys, lo, hi, key):
    """Binary search for `key` in the sorted slice keys[lo:hi]. Returns index or -1."""
    end = hi
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < end and keys[lo] == key:
        return lo
    return -1


@njit
def _level_windows(flat, offsets, n, bos, eos):
    """
    Enumerate every n-gram window of a sentence corpus.

    Padding follows the paper (Sec. 6.3.1): for order n, each sentence is padded
    with *exactly n* <BOS> on the left and a single <EOS> on the right, giving
    z + 2 windows for a sentence of z tokens. Hence c(<BOS>^n) = #sentences and
    c(<BOS>^(n-1) t_1) = #sentences starting with t_1.

    Returns
    -------
    gram_h : uint64[W]   hash of the full n-gram        (t_1..t_n)
    pref_h : uint64[W]   hash of its (n-1)-prefix       (t_1..t_{n-1})  = the context g
    sfx_h  : uint64[W]   hash of its (n-1)-suffix       (t_2..t_n)      -> for N1+(*g)
    last_t : int32[W]    the final token id t_n         -> to exclude t_n == <BOS>
    """
    n_sent = offsets.shape[0] - 1
    total = 0
    for s in range(n_sent):
        total += (offsets[s + 1] - offsets[s]) + 2

    gram_h = np.empty(total, dtype=np.uint64)
    pref_h = np.empty(total, dtype=np.uint64)
    sfx_h = np.empty(total, dtype=np.uint64)
    last_t = np.empty(total, dtype=np.int32)

    w = 0
    for s in range(n_sent):
        a = offsets[s]
        b = offsets[s + 1]
        z = b - a
        pad = np.empty(n + z + 1, dtype=np.int32)
        for i in range(n):
            pad[i] = bos
        for i in range(z):
            pad[n + i] = flat[a + i]
        pad[n + z] = eos

        for st in range(0, z + 2):
            h = _H_INIT
            for i in range(st, st + n - 1):
                h = _hstep(h, pad[i])
            pref_h[w] = h
            gram_h[w] = _hstep(h, pad[st + n - 1])

            h2 = _H_INIT
            for i in range(st + 1, st + n):
                h2 = _hstep(h2, pad[i])
            sfx_h[w] = h2

            last_t[w] = pad[st + n - 1]
            w += 1
    return gram_h, pref_h, sfx_h, last_t


# --------------------------------------------------------------------------- #
# 3. Kneser-Ney query kernel                                                    #
# --------------------------------------------------------------------------- #


@njit
def _sentence_logprobs(
    ids,
    N,
    D,
    inv_TV,
    gram_keys,
    gram_ckn,
    gram_off,
    ctx_keys,
    ctx_S,
    ctx_N1p,
    ctx_off,
    bos,
    eos,
    out,
    out_start,
):
    """
    Natural-log P_KN(t_k | t_<k) for every position of one sentence, plus <EOS>.

    Implements the recursion of Eqs. (15)-(19), bottom-up:

        p_1 = alpha(t|empty) + gamma(empty) * 1/(|T|+1)
        p_m = alpha(t|g_{m-1}) + gamma(g_{m-1}) * p_{m-1}      m = 2..N

    with (constant discount D, i.e. r = 1 in modified-KN terms):

        alpha(t|g) = max(c_KN(gt) - D, 0) / S(g),  gamma(g) = D * N1+(g*) / S(g)
        S(g)       = sum_{t' in T*} c_KN(g t')

    and alpha = 0, gamma = 1 whenever the context g was never observed.
    """
    z = ids.shape[0]
    hist = np.empty(N, dtype=np.int32)  # only [0 : N-1] is used

    for k in range(z + 1):
        t = ids[k] if k < z else eos

        # left-pad the history with <BOS> up to N-1 tokens
        for j in range(N - 1):
            idx = k - (N - 1) + j
            hist[j] = bos if idx < 0 else ids[idx]

        p = inv_TV  # the uniform base of Eq. (16)
        for m in range(1, N + 1):
            cstart = N - 1 - (m - 1)  # context = hist[cstart : N-1]

            hc = _H_INIT
            for i in range(cstart, N - 1):
                hc = _hstep(hc, hist[i])
            hg = _hstep(hc, t)

            ci = _bsearch(ctx_keys, ctx_off[m - 1], ctx_off[m], hc)
            if ci < 0:
                continue  # c(g) == 0 -> alpha = 0, gamma = 1 -> p unchanged

            S = ctx_S[ci]
            gi = _bsearch(gram_keys, gram_off[m - 1], gram_off[m], hg)
            c = gram_ckn[gi] if gi >= 0 else 0.0

            a = c - D
            if a < 0.0:
                a = 0.0
            p = a / S + (D * ctx_N1p[ci] / S) * p

        out[out_start + k] = math.log(p)
    return z + 1


@njit
def _model_logprobs(
    flat,
    offsets,
    in_dict,
    unk,
    N,
    D,
    inv_TV,
    gram_keys,
    gram_ckn,
    gram_off,
    ctx_keys,
    ctx_S,
    ctx_N1p,
    ctx_off,
    bos,
    eos,
):
    """
    Natural-log token probabilities for a whole document under one grammar model.

    Output length = n_tokens + n_sentences (one extra slot per sentence for <EOS>).
    Out-of-dictionary ids are folded onto <UNK> *here*, so the caller can encode a
    document once against the global vocabulary and reuse it for all r+1 models.
    """
    n_sent = offsets.shape[0] - 1
    out = np.empty(flat.shape[0] + n_sent, dtype=np.float64)
    vsz = in_dict.shape[0]
    w = 0
    for s in range(n_sent):
        a = offsets[s]
        b = offsets[s + 1]
        z = b - a
        ids = np.empty(z, dtype=np.int32)
        for i in range(z):
            g = flat[a + i]
            if g < vsz and in_dict[g] == 1:
                ids[i] = g
            else:
                ids[i] = unk
        w += _sentence_logprobs(
            ids, N, D, inv_TV,
            gram_keys, gram_ckn, gram_off,
            ctx_keys, ctx_S, ctx_N1p, ctx_off,
            bos, eos, out, w,
        )
    return out


# --------------------------------------------------------------------------- #
# 4. Vocabulary + sentence storage                                             #
# --------------------------------------------------------------------------- #


class Vocabulary:
    """Global string <-> int32 id map. Ids 0/1/2 are reserved for <UNK>/<EOS>/<BOS>."""

    __slots__ = ("_itos", "_stoi", "frozen")

    def __init__(self) -> None:
        self._itos: List[str] = list(SPECIAL_TOKENS)
        self._stoi: Dict[str, int] = {t: i for i, t in enumerate(self._itos)}
        self.frozen: bool = False

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def id_of(self, token: str) -> int:
        return self._stoi.get(token, UNK_ID)

    def token_of(self, idx: int) -> str:
        return self._itos[idx]

    def add(self, token: str) -> int:
        i = self._stoi.get(token, -1)
        if i >= 0:
            return i
        if self.frozen:
            return UNK_ID
        i = len(self._itos)
        self._itos.append(token)
        self._stoi[token] = i
        return i

    def freeze(self) -> "Vocabulary":
        self.frozen = True
        return self

    def encode(self, sentences: Sequence[Sequence[str]], grow: bool = True) -> "SentenceStore":
        """Encode ``[[tok, tok, ...], ...]`` into a flat int32 buffer + offsets."""
        add = self.add if (grow and not self.frozen) else self.id_of
        offsets = np.zeros(len(sentences) + 1, dtype=np.int64)
        flat: List[int] = []
        for i, sent in enumerate(sentences):
            for tok in sent:
                flat.append(add(tok))
            offsets[i + 1] = len(flat)
        return SentenceStore(np.asarray(flat, dtype=np.int32), offsets)


@dataclass
class SentenceStore:
    """A corpus of tokenised sentences as a flat id buffer + CSR-style offsets."""

    flat: np.ndarray  # int32[n_tokens]
    offsets: np.ndarray  # int64[n_sentences + 1]

    def __post_init__(self) -> None:
        self.flat = np.ascontiguousarray(self.flat, dtype=np.int32)
        self.offsets = np.ascontiguousarray(self.offsets, dtype=np.int64)

    @property
    def n_sentences(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return len(self.flat)

    def __len__(self) -> int:
        return self.n_sentences

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def select(self, idx: np.ndarray) -> "SentenceStore":
        """Materialise the sub-corpus made of sentences `idx` (order preserved)."""
        idx = np.asarray(idx, dtype=np.int64)
        lens = self.offsets[idx + 1] - self.offsets[idx]
        new_off = np.zeros(len(idx) + 1, dtype=np.int64)
        np.cumsum(lens, out=new_off[1:])
        flat = np.empty(int(new_off[-1]), dtype=np.int32)
        for j, s in enumerate(idx):
            flat[new_off[j] : new_off[j + 1]] = self.flat[self.offsets[s] : self.offsets[s + 1]]
        return SentenceStore(flat, new_off)

    def reverse(self) -> "SentenceStore":
        """
        Reverse the tokens *within* each sentence (sentence order is unchanged).

        Running the ordinary forward machinery on this is exactly a backward
        language model: <BOS> then pads the sentence's end and <EOS> marks its
        start. P_bwd(S) = prod_k P(t_k | t_{k+1} ... t_{k+N-1}) is a proper
        distribution over sentences -- reversal is a bijection -- so lambda_G
        computed this way is a genuine likelihood ratio, merely under a different
        conditional-independence assumption than the forward model.
        """
        flat = self.flat.copy()
        for i in range(self.n_sentences):
            a, b = self.offsets[i], self.offsets[i + 1]
            flat[a:b] = self.flat[a:b][::-1]
        return SentenceStore(flat, self.offsets.copy())

    @staticmethod
    def concat(stores: Sequence["SentenceStore"]) -> "SentenceStore":
        stores = [s for s in stores if s.n_sentences > 0]
        if not stores:
            return SentenceStore(np.zeros(0, np.int32), np.zeros(1, np.int64))
        flat = np.concatenate([s.flat for s in stores])
        offs = [np.zeros(1, dtype=np.int64)]
        base = 0
        for s in stores:
            offs.append(s.offsets[1:] + base)
            base += s.n_tokens
        return SentenceStore(flat, np.concatenate(offs))


# --------------------------------------------------------------------------- #
# 5. POSNoise front end                                                        #
# --------------------------------------------------------------------------- #


_EOS_PUNCT_RE = re.compile(r"^[.!?\u2026]+$")

#: POS -> placeholder, identical to `posnoise.core.POSNoise.abbrev_pos_tags`.
#: Any POS *not* listed here (DET, ADP, PRON, CCONJ, SCONJ, PART, PUNCT, INTJ)
#: is a function category and survives masking untouched.
DEFAULT_ABBREV_POS_TAGS: Dict[str, str] = {
    "NOUN": "#",
    "PROPN": "\u00a7",
    "VERB": "\u00d8",
    "AUX": "\u00d8",
    "ADJ": "@",
    "ADV": "\u00a9",
    "NUM": "\u00b5",
    "SYM": "$",
    "X": "\u00a5",
}

ENGLISH_CONTRACTIONS = {"'m", "'d", "'s", "'t", "'ve", "'ll", "'re", "'ts", "'em", "'Tis"}
GERMAN_STYLE_TOKENS = {"'s", "\u2019s"}


#: Languages with a safe-pattern list. All nine lists live in ./posnoise_lists/
#: next to this module; en/de are the upstream Halvani & Graner (2021) lists, the
#: rest are this repo's UD/ReM/ReN builds -- see `docs/posnoise_lists.md`.
SUPPORTED_LANGUAGES = (
    "en", "de", "fr", "es", "it", "pl", "ru", "gmh", "gml",
    # UD-mined lists added for the multilingual study (see data_prep/build_posnoise.py):
    "cs", "el", "hr", "hu", "lt", "lv", "nl", "no", "pt", "ro", "sl", "sr", "sv", "uk",
)

_PATTERN_STEMS = {
    "en": "POSNoise_PatternList_En_*.txt", "de": "POSNoise_PatternList_De_*.txt",
    "fr": "POSNoise_PatternList_Fr_*.txt", "es": "POSNoise_PatternList_Es_*.txt",
    "it": "POSNoise_PatternList_It_*.txt", "pl": "POSNoise_PatternList_Pl_*.txt",
    "ru": "POSNoise_PatternList_Ru_*.txt",
    # historical: gmh = Middle High German, gml = Middle Low German (ISO 639-3)
    "gmh": "POSNoise_PatternList_Gmh_*.txt", "gml": "POSNoise_PatternList_Gml_*.txt",
    # UD-mined, lemma-based (build_posnoise.py); all take match_on="both"
    "cs": "POSNoise_PatternList_Cs_*.txt", "el": "POSNoise_PatternList_El_*.txt",
    "hr": "POSNoise_PatternList_Hr_*.txt", "hu": "POSNoise_PatternList_Hu_*.txt",
    "lt": "POSNoise_PatternList_Lt_*.txt", "lv": "POSNoise_PatternList_Lv_*.txt",
    "nl": "POSNoise_PatternList_Nl_*.txt", "no": "POSNoise_PatternList_No_*.txt",
    "pt": "POSNoise_PatternList_Pt_*.txt", "ro": "POSNoise_PatternList_Ro_*.txt",
    "sl": "POSNoise_PatternList_Sl_*.txt", "sr": "POSNoise_PatternList_Sr_*.txt",
    "sv": "POSNoise_PatternList_Sv_*.txt", "uk": "POSNoise_PatternList_Uk_*.txt",
}

SPACY_MODELS = {
    "en": "en_core_web_lg", "de": "de_core_news_lg", "fr": "fr_core_news_lg",
    "es": "es_core_news_lg", "it": "it_core_news_lg", "pl": "pl_core_news_lg",
    "ru": "ru_core_news_lg",
    # added with the UD-mined lists; Norwegian uses the Bokmaal pipeline, and there is
    # no Serbian/Czech/Hungarian/Latvian spaCy model at all -- those are tagged with
    # Stanza and fed in through mask_tagged() (see data_prep/mask_corpora.py).
    "el": "el_core_news_lg", "hr": "hr_core_news_lg", "lt": "lt_core_news_lg",
    "nl": "nl_core_news_lg", "no": "nb_core_news_lg", "pt": "pt_core_news_lg",
    "ro": "ro_core_news_lg", "sl": "sl_core_news_lg", "sv": "sv_core_news_lg",
    "uk": "uk_core_news_lg",
    "cs": None, "hu": None, "lv": None, "sr": None,
    # No model ships for either historical stage. For gmh, point `spacy_model` at
    # the ReM-trained GMH-Tagger (MIT):
    #   github.com/Middle-High-German-Conceptual-Database/Spacy-Model-for-Middle-High-German
    # It emits HiTS in token.tag_ and leaves token.pos_ empty, so pass
    # tag_map=HITS_TO_UD (the default for gmh/gml). For gml there is no spaCy model
    # at all -- see POSNOISE_LISTS.md.
    "gmh": None, "gml": None,
}

#: Languages whose lists are lemma-based and so need match_on="both" (and a
#: lemmatiser in the pipeline). en/de ship as surface-form lists.
_LEMMA_LANGUAGES = ("fr", "es", "it", "pl", "ru",
                    "cs", "el", "hr", "hu", "lt", "lv", "nl", "no", "pt",
                    "ro", "sl", "sr", "sv", "uk")

#: Historical stages: the lists already enumerate every attested spelling variant
#: straight from ReM/ReN, so surface matching is right and no lemmatiser exists.
_HISTORICAL_LANGUAGES = ("gmh", "gml")


def _posnoise_roots():
    """Candidate ``posnoise_lists/`` directories, nearest first.

    Looks next to this module and in the CWD, then walks **up** each parent chain, so a
    copy of ``lambdag.py`` sitting in a sub-folder (e.g. ``medieval/``) still finds a
    single shared ``posnoise_lists/`` higher up the tree (e.g. at the repo root).
    """
    seen = set()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            root = d / "posnoise_lists"
            if str(root) not in seen:
                seen.add(str(root))
                yield root


def load_hits_to_ud() -> Dict[str, str]:
    """
    HiTS (Historisches Tagset, Ruhr-Uni Bochum) -> Universal Dependencies POS.

    ReM and ReN tag with HiTS, not UD, and the ReM-trained spaCy model therefore
    fills ``token.tag_`` and leaves ``token.pos_`` empty -- which POSNoise reads.
    Passing this as `tag_map` bridges the two. ReM and ReN use slightly different
    HiTS variants (DDART vs DDARTA), so both are covered.
    """
    import json
    for root in _posnoise_roots():
        f = root / "hits_to_ud.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "hits_to_ud.json not found in any posnoise_lists/ (searched next to the module "
        "and up from the CWD)."
    )


def _find_pattern_list(language: str) -> Path:
    """
    Locate the safe-pattern list for `language`.

    Searches ``posnoise_lists/`` next to this module and in the CWD, then **up** each
    parent chain (see :func:`_posnoise_roots`). All nine lists (en/de plus fr/es/it/pl/ru
    and gmh/gml) live there, so no external `posnoise` install is needed.
    """
    if language not in _PATTERN_STEMS:
        raise ValueError(f"language must be one of {SUPPORTED_LANGUAGES}")
    stem = _PATTERN_STEMS[language]

    for root in _posnoise_roots():
        if root.exists():
            hits = sorted(root.rglob(stem))
            if hits:
                return hits[-1]  # highest version
    raise FileNotFoundError(
        f"No POSNoise pattern list matching {stem!r}. Expected it in a posnoise_lists/ "
        "next to this module or up from the CWD (all nine lists ship there)."
    )


class POSNoiseMasker:
    """
    POSNoise topic masking + sentence segmentation, producing LambdaG's input.

    Turns raw text into ``List[List[str]]`` -- a list of sentences, each a list of
    *function tokens* (Sec. 3 of the paper: function words, punctuation marks and
    abstract grammatical categories).

    This is a faithful but much faster re-implementation of
    ``posnoise.POSNoise.pos_noise_``. The shipped version rescans the whole token
    list once per safe pattern (1,034 patterns for English) -- O(P x T). Here the
    patterns are bucketed by their first token and the document is scanned once,
    O(T x avg-bucket). The leftmost/non-overlapping per-pattern match semantics of
    the original are reproduced exactly via the ``_next_allowed`` guard.

    Parameters
    ----------
    language : {"en", "de", "fr", "es", "it", "pl", "ru"}
        en/de use the lists shipped with the `posnoise` package. fr/es/it/pl/ru use
        the lists in ``posnoise_lists/`` next to this module -- see
        ``POSNOISE_LISTS.md`` for how they were built, and for the caveats. Those are
        v1.0, not native-speaker reviewed, and unevaluated on any AV corpus; the
        paper's twelve corpora are all English (Sec. 5.1).
    spacy_model : str, optional
        Defaults to the ``_lg`` model for the language (the POSNoise default). ``_sm``
        is a lighter substitute for experimentation, but tagging errors leak straight
        into the mask.
    match_on : {"text", "lemma", "both"}, optional
        What a pattern entry is compared against. Defaults to ``"text"`` for en/de
        (their lists enumerate surface forms) and ``"both"`` for fr/es/it/pl/ru
        (their lists are lemma-based).

        Matching on the lemma while *emitting* the surface form is what makes rich
        morphology tractable: Italian has 193 AUX surface forms but only 10 AUX
        lemmas, Russian 22 vs 2. Inflection survives untouched in the output -- a
        whitelisted token is copied verbatim; only the lookup is lemmatised. Needs a
        lemmatiser in the pipeline, so it is not disabled by default when in use.
    mode : {"posnoise", "star", "none"}
        * ``"posnoise"`` -- full POSNoise (paper's Algorithm 1).
        * ``"star"``     -- ablation of the paper's Table 4: mask with ``*`` instead
                            of POS labels (the POS tags turn out to add very little).
        * ``"none"``     -- ablation of Table 3: tokenise/segment only, no masking.
    segment : {"sentence", "window"}
        Unit each masked document is cut into -- the unit of independence in
        lambda_G (paper Eq. 13).
        * ``"sentence"`` (default) -- split on end-of-sentence punctuation / newlines,
                          the paper's unit.
        * ``"window"``  -- ignore those boundaries and cut the masked token stream
                          into fixed-length windows of ``window`` tokens. Use for
                          verse / unpunctuated corpora where sentence segmentation is
                          unreliable (e.g. gmh/gml), or simply to make the unit a
                          fixed "text snippet" of N words rather than a sentence.
    window : int
        Window length in (masked) tokens when ``segment="window"``; ignored
        otherwise. Typical values: 10, 20, 50.
    nlp : spacy.language.Language, optional
        Pre-loaded pipeline; skips the internal load.
    require_tagger : bool
        If ``False`` no spaCy pipeline is loaded at all -- for input that already
        carries POS tags (and lemmas), such as ReM (gmh) and ReN (gml). Build the
        masker with ``POSNoiseMasker.pretagged(language)`` (or this flag directly)
        and feed it via :meth:`mask_tagged` / :meth:`mask_tagged_batch`. ``mask`` /
        ``mask_batch`` (which need the tagger) then raise.
    emit : {"surface", "lemma"}
        What to emit for the *kept* (unmasked) tokens. ``"surface"`` (default) keeps
        the original word form; ``"lemma"`` emits the lemma instead, normalising away
        spelling/inflectional variation. For historical corpora (gmh/gml) this strips
        scribal-spelling noise (scribe != author) from the function-word scaffold --
        pair it with ``lowercase=True``. Masked tokens are always the POS placeholder;
        falls back to the surface when a lemma is missing.
    """

    def __init__(
        self,
        language: str = "en",
        spacy_model: Optional[str] = None,
        mode: str = "posnoise",
        segment: str = "sentence",
        window: int = 20,
        nlp=None,
        require_tagger: bool = True,
        pattern_list_path: Optional[Union[str, Path]] = None,
        abbrev_pos_tags: Optional[Dict[str, str]] = None,
        lowercase: bool = False,
        emit: str = "surface",
        match_on: Optional[str] = None,
        tag_map: Optional[Dict[str, str]] = None,
        disable: Optional[Sequence[str]] = None,
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {SUPPORTED_LANGUAGES}")
        if mode not in ("posnoise", "star", "none"):
            raise ValueError("mode must be 'posnoise', 'star' or 'none'")
        if segment not in ("sentence", "window"):
            raise ValueError("segment must be 'sentence' or 'window'")
        if emit not in ("surface", "lemma"):
            raise ValueError("emit must be 'surface' or 'lemma'")
        if int(window) < 1:
            raise ValueError("window must be a positive integer")
        self.segment = segment
        self.window = int(window)
        self.emit = emit

        # en/de ship surface-form lists; fr/es/it/pl/ru are lemma-based.
        if match_on is None:
            match_on = "both" if language in _LEMMA_LANGUAGES else "text"
        if tag_map is None and language in _HISTORICAL_LANGUAGES:
            tag_map = load_hits_to_ud()
        self.tag_map = dict(tag_map) if tag_map else None
        if match_on not in ("text", "lemma", "both"):
            raise ValueError("match_on must be 'text', 'lemma' or 'both'")
        self.match_on = match_on

        if disable is None:
            disable = ("parser", "ner") if match_on != "text" else ("parser", "ner", "lemmatizer")

        self.language = language
        self.mode = mode
        self.lowercase = lowercase
        self.abbrev_pos_tags = dict(abbrev_pos_tags or DEFAULT_ABBREV_POS_TAGS)

        if nlp is None and require_tagger:
            import spacy

            model = spacy_model or SPACY_MODELS[language]
            if model is None:
                raise ValueError(
                    f"No spaCy model ships for {language!r}. Pass spacy_model=<path> "
                    "or nlp=<Language>. For gmh use the ReM-trained GMH-Tagger; for "
                    "gml no model exists yet -- see POSNOISE_LISTS.md. For text that "
                    "already carries POS/lemmas (ReM, ReN) use require_tagger=False "
                    "and mask_tagged()."
                )
            nlp = spacy.load(model, disable=[d for d in disable if d != ""])
        self.nlp = nlp  # may be None in pre-tagged mode (require_tagger=False)
        if self.nlp is not None and match_on != "text" and not self.nlp.has_pipe("lemmatizer"):
            warnings.warn(
                f"match_on={match_on!r} needs lemmas, but the pipeline has no "
                "lemmatizer; lemma keys will fall back to the surface form. The "
                f"{language} list is lemma-based, so coverage will suffer.",
                RuntimeWarning,
            )

        path = Path(pattern_list_path) if pattern_list_path else _find_pattern_list(language)
        self.pattern_list_path = path
        self._patterns, self._buckets = self._compile_patterns(path)

    @classmethod
    def pretagged(cls, language: str, **kwargs) -> "POSNoiseMasker":
        """
        Masker for input that already has POS tags / lemmas -- no spaCy loaded.

        Thin wrapper for ``POSNoiseMasker(language, require_tagger=False, **kwargs)``.
        Then call :meth:`mask_tagged` / :meth:`mask_tagged_batch` with (surface, pos,
        lemma) tuples. This is the entry point for ReM (gmh) and ReN (gml), whose
        HiTS POS layer is bridged to UD by the default ``tag_map`` -- and it is the
        only way to use gml, which has no spaCy model at all.

        For the historical stages this defaults ``match_on="both"``: their POSNoise
        lists are in *normalised* orthography while the corpus surface tokens are
        *diplomatic* (long-s, macrons, ...), so matching on the (normalised) lemma
        while still emitting the surface rescues far more function words than the
        diplomatic surface would -- and the scribal spelling survives in the output.
        """
        if language in _HISTORICAL_LANGUAGES:
            kwargs.setdefault("match_on", "both")
        return cls(language, require_tagger=False, **kwargs)

    # -- pattern compilation ------------------------------------------------ #

    def _compile_patterns(self, path: Path):
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if self.nlp is not None:
            # tokenizer only: identical segmentation to the full pipeline, ~100x faster
            patterns = [
                tuple(t.text.lower() for t in doc)
                for doc in self.nlp.tokenizer.pipe(lines)
            ]
        else:
            # spaCy-free: whitespace tokenisation. The gmh/gml surface lists are built
            # straight from ReM/ReN tokens, so their (incl. multiword) entries split on
            # whitespace exactly as the corpus tokens do.
            patterns = [tuple(ln.lower().split()) for ln in lines]
        patterns = [p for p in patterns if p]
        buckets: Dict[str, List[int]] = {}
        for pid, p in enumerate(patterns):
            buckets.setdefault(p[0], []).append(pid)
        return patterns, buckets

    # -- core --------------------------------------------------------------- #

    def _pos_of(self, token) -> str:
        """
        UD POS for a token, via `tag_map` when the pipeline only fills ``tag_``.

        The ReM-trained MHG tagger is a ['tok2vec', 'tagger'] pipeline: it predicts
        HiTS into ``tag_`` and never populates ``pos_``. Without this bridge every
        token would look like POS "" -- absent from abbrev_pos_tags -- and POSNoise
        would silently mask nothing at all.
        """
        if self.tag_map is None:
            return token.pos_
        if token.pos_:
            return token.pos_
        return self.tag_map.get(token.tag_, self.tag_map.get(token.tag_.split(".")[0], ""))

    def _keys_of(self, token) -> Tuple[str, ...]:
        """The string(s) this token may be matched on, per `match_on`."""
        if self.match_on == "text":
            return (token.lower_,)
        lem = (token.lemma_ or token.text).lower()
        if self.match_on == "lemma":
            return (lem,)
        return (token.lower_,) if lem == token.lower_ else (token.lower_, lem)


    def _safe_mask(self, keys: List[Tuple[str, ...]], pos_tags: List[str],
                   raw: List[str]) -> np.ndarray:
        """
        Boolean mask: True = whitelisted, keep the surface form.

        `keys[i]` is the set of strings token i may match on -- its lower-cased
        surface form, its lemma, or both, per `match_on`. Matching on the lemma but
        *emitting* the surface form is what makes rich morphology tractable: Italian
        has 193 AUX surface forms but only 10 AUX lemmas, Russian 22 vs 2. The
        inflection survives in the output, because a whitelisted token is copied
        verbatim -- only the lookup is lemmatised.
        """
        n = len(keys)
        keep = np.zeros(n, dtype=bool)
        patterns, buckets = self._patterns, self._buckets
        next_allowed: Dict[int, int] = {}

        for i in range(n):
            cand = []
            for k in keys[i]:
                cand.extend(buckets.get(k, ()))
            for pid in dict.fromkeys(cand):  # de-dup, preserve order
                if i < next_allowed.get(pid, 0):
                    continue
                p = patterns[pid]
                L = len(p)
                if i + L > n:
                    continue
                ok = True
                for j in range(1, L):
                    if p[j] not in keys[i + j]:
                        ok = False
                        break
                if ok:
                    keep[i : i + L] = True
                    next_allowed[pid] = i + L

        contractions = ENGLISH_CONTRACTIONS if self.language == "en" else GERMAN_STYLE_TOKENS
        for i in range(n):
            if raw[i] in contractions:
                keep[i] = True
            elif pos_tags[i] == "NUM" and not re.fullmatch(r"\d+", raw[i]):
                # written-out / mixed numerals are stylistic; bare digits are topical
                keep[i] = True
        return keep

    def _emit_surface(self, surface: str, pos: str, keep: bool, lemma: str = "") -> str:
        """Masked output for one token, from raw fields (spaCy-free core of _emit)."""
        # what a *kept* token shows: its lemma (emit="lemma") or its surface form.
        # lowercase applies to the WORD only -- placeholders keep their canonical glyph.
        kept = lemma if (self.emit == "lemma" and lemma) else surface
        if self.lowercase:
            kept = kept.lower()
        if keep or self.mode == "none":
            return kept
        if self.mode == "star":
            return "*" if pos in self.abbrev_pos_tags else kept
        return self.abbrev_pos_tags.get(pos, kept)

    def _emit(self, t, i, pos, keep) -> str:
        """The masked surface string for token i (shared by both segmenters)."""
        return self._emit_surface(t.text, pos[i], bool(keep[i]), t.lemma_)

    def _resolve_pos(self, pos: str) -> str:
        """
        UD POS from a pre-supplied tag (the spaCy-free twin of _pos_of).

        With a ``tag_map`` set (gmh/gml) the incoming tag is HiTS and is mapped to
        UD; otherwise it is assumed to already be UD. Unmapped tags become "" -- not
        in ``abbrev_pos_tags``, so the token is kept, matching _pos_of's behaviour.
        """
        if self.tag_map is None:
            return pos
        # ReM/ReN univerbations carry '+'-joined tags for one written token
        # (e.g. "VVFIN+PPER" for "gestu" = gan+du): decide on the HEAD (leftmost)
        # component, which carries the token's primary grammatical role.
        head = pos.split("+", 1)[0]
        return self.tag_map.get(head, self.tag_map.get(head.split(".")[0], ""))

    def _keys_from(self, surface: str, lemma: str) -> Tuple[str, ...]:
        """The string(s) to match on, from raw fields (spaCy-free twin of _keys_of)."""
        low = surface.lower()
        if self.match_on == "text":
            return (low,)
        lem = (lemma or surface).lower()
        if self.match_on == "lemma":
            return (lem,)
        return (low,) if lem == low else (low, lem)

    def mask_doc(self, doc) -> List[List[str]]:
        """
        Mask + segment an already-parsed spaCy ``Doc``.

        Segmentation follows ``self.segment``: ``"sentence"`` cuts on end-of-sentence
        punctuation / newlines (the paper's unit, Eq. 13); ``"window"`` ignores those
        boundaries and tiles the masked token stream into fixed-length windows of
        ``self.window`` tokens.
        """
        toks = [t for t in doc]
        raw = [t.text for t in toks]
        pos = [self._pos_of(t) for t in toks]
        keys = [self._keys_of(t) for t in toks]

        if self.mode == "none":
            keep = np.ones(len(toks), dtype=bool)
        else:
            keep = self._safe_mask(keys, pos, raw)

        def is_space(i, t):
            return (not t.text.strip()) or pos[i] == "SPACE" or t.pos_ == "SPACE"

        # Fixed-length windows: flatten the masked (non-space) tokens and tile them,
        # ignoring sentence boundaries entirely.
        if self.segment == "window":
            flat = [self._emit(t, i, pos, keep)
                    for i, t in enumerate(toks) if not is_space(i, t)]
            w = self.window
            return [flat[j : j + w] for j in range(0, len(flat), w)]

        sentences: List[List[str]] = []
        cur: List[str] = []
        for i, t in enumerate(toks):
            newline = "\n" in t.text or "\n" in t.whitespace_

            if not is_space(i, t):
                cur.append(self._emit(t, i, pos, keep))

                # Sec. 3: sentence boundary at an end-of-sentence punctuation mark ...
                if _EOS_PUNCT_RE.match(t.text) and cur:
                    sentences.append(cur)
                    cur = []

            # ... or at a new line
            if newline and cur:
                sentences.append(cur)
                cur = []

        if cur:
            sentences.append(cur)
        return sentences

    def _require_nlp(self) -> None:
        if self.nlp is None:
            raise RuntimeError(
                "This masker was built without spaCy (require_tagger=False). Feed "
                "already-tagged input via mask_tagged()/mask_tagged_batch() instead "
                "of mask()/mask_batch()."
            )

    def mask(self, text: str) -> List[List[str]]:
        """Mask + segment one raw document."""
        self._require_nlp()
        return self.mask_doc(self.nlp(text))

    def mask_batch(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        n_process: int = 1,
        progress: bool = False,
    ) -> List[List[List[str]]]:
        """Vectorised masking of many documents via ``nlp.pipe`` (use this for corpora)."""
        self._require_nlp()
        texts = list(texts)
        it = self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
        if progress:
            try:
                from tqdm.auto import tqdm

                it = tqdm(it, total=len(texts), desc="POSNoise", unit="doc")
            except ImportError:
                pass
        return [self.mask_doc(d) for d in it]

    def mask_tagged(
        self, sentences: Sequence[Sequence[Sequence[str]]]
    ) -> List[List[str]]:
        """
        Mask an already-tagged, already-segmented document -- no spaCy required.

        Parameters
        ----------
        sentences : list of sentences, each a list of ``(surface, pos, lemma)``.
            ``pos`` is a HiTS tag when the masker has a ``tag_map`` (gmh/gml, mapped
            to UD) or a UD tag otherwise. ``lemma`` may be ``""`` / omitted and is
            used only when ``match_on != "text"``. This is the entry point for
            corpora that ship their own POS/lemma layer (ReM, ReN): no tagger or
            lemmatiser is loaded.

        Returns
        -------
        The masked document as ``List[List[str]]``, honouring ``self.segment``:
        ``"sentence"`` keeps the *provided* boundaries (historical punctuation is
        unreliable, so we trust the corpus's own segmentation); ``"window"`` ignores
        them and tiles the flat token stream into ``self.window``-token windows.
        """
        out: List[List[str]] = []
        flat: List[str] = []
        for sent in sentences:
            if not sent:
                continue
            raw = [tok[0] for tok in sent]
            lem = [tok[2] if len(tok) > 2 else "" for tok in sent]
            pos = [self._resolve_pos(tok[1]) for tok in sent]
            keys = [self._keys_from(raw[i], lem[i]) for i in range(len(sent))]
            keep = (
                np.ones(len(sent), dtype=bool)
                if self.mode == "none"
                else self._safe_mask(keys, pos, raw)
            )
            emitted = [self._emit_surface(raw[i], pos[i], bool(keep[i]), lem[i]) for i in range(len(sent))]
            if self.segment == "window":
                flat.extend(emitted)
            elif emitted:
                out.append(emitted)

        if self.segment == "window":
            w = self.window
            return [flat[i : i + w] for i in range(0, len(flat), w)]
        return out

    def mask_tagged_batch(
        self,
        docs: Iterable[Sequence[Sequence[Sequence[str]]]],
        progress: bool = False,
    ) -> List[List[List[str]]]:
        """:meth:`mask_tagged` over many documents (each a list of tagged sentences)."""
        docs = list(docs)
        it = docs
        if progress:
            try:
                from tqdm.auto import tqdm

                it = tqdm(docs, desc="POSNoise", unit="doc")
            except ImportError:
                pass
        return [self.mask_tagged(d) for d in it]

    def mask_to_string(self, text: str) -> str:
        """Debug helper: the classic flat POSNoise string, sentences joined by ' | '."""
        return " | ".join(" ".join(s) for s in self.mask(text))


def windowize(masked_doc: Sequence[Sequence[str]], window: int) -> List[List[str]]:
    """
    Re-segment an already-masked document into fixed-length windows.

    Takes one masked document (a list of sentences, i.e. the output of
    ``POSNoiseMasker.mask``) and, ignoring the sentence boundaries, tiles its
    flattened token stream into consecutive windows of ``window`` tokens. The last
    window keeps whatever remains (possibly shorter). This is the post-hoc twin of
    ``POSNoiseMasker(segment="window", window=...)``: use it to switch an existing
    sentence-segmented corpus to the fixed-length "text snippet" unit without
    re-running spaCy. Windows never cross document boundaries -- map it per document.

    The window becomes the unit of independence in lambda_G in place of the
    linguistic sentence (paper Eq. 13); handy for verse / unpunctuated text where
    sentence segmentation is unreliable.
    """
    if window < 1:
        raise ValueError("window must be a positive integer")
    flat = [tok for sent in masked_doc for tok in sent]
    return [flat[i : i + window] for i in range(0, len(flat), window)]


# --------------------------------------------------------------------------- #
# 6. Grammar models                                                            #
# --------------------------------------------------------------------------- #
#
# Three interchangeable probability engines, all sharing the same n-gram tables,
# the same <BOS>/<EOS> padding, the same dictionary semantics and the same
# `token_logprobs()` interface, so that LambdaG can swap between them freely:
#
#   "kn"    Interpolated Kneser-Ney, D = 0.75.            Sec. 6.3 of the paper.
#   "hpy"   Hierarchical Pitman-Yor process.              Teh (2006).
#   "ppmd"  Prediction by Partial Matching, escape D.     Cleary & Witten (1984),
#                                                         Howard (1993).
#
# They differ along two axes:
#
#                     lower-order counts        combination rule
#   kn                continuation N1+(*g)      interpolation
#   hpy               table counts t_uw         interpolation
#   ppmd              raw counts c(g)           backoff + exclusion
#
# and the first two are relatives: Teh (2006) showed that interpolated KN *is* the
# HPY predictive distribution under theta = 0 and the "minimal" table assumption
# t_uw = 1[c_uw > 0]. That is not just a remark here -- `HPYGrammarModel` with
# ``concentration=0, table_estimator="minimal", discount=0.75`` reproduces
# `KNGrammarModel` to the last bit (see the test suite). KN's continuation count
# N1+(*g) is literally "sum of the children's table counts" when every child keeps
# exactly one table.


def _build_levels(store: "SentenceStore", N: int) -> List[dict]:
    """
    Raw n-gram inventory for orders 1..N, shared by every engine.

    For each order n returns the distinct n-grams (hashed, sorted), their raw
    counts, and -- for one representative window per distinct n-gram -- the hash of
    the (n-1)-prefix (the context g), the hash of the (n-1)-suffix (used to
    aggregate into the next order down) and the final token (used to drop <BOS>).
    """
    levels = []
    for n in range(1, N + 1):
        gh, ph, sh, lt = _level_windows(store.flat, store.offsets, np.int64(n), BOS_ID, EOS_ID)
        order = np.argsort(gh, kind="stable")
        gh_s = gh[order]
        new = np.empty(len(gh_s), dtype=bool)
        new[0] = True
        np.not_equal(gh_s[1:], gh_s[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        levels.append(
            {
                "keys": gh_s[starts],
                "counts": np.diff(np.append(starts, len(gh_s))).astype(np.float64),
                "rep": order[starts],  # one representative window per distinct gram
                "ph": ph,
                "sh": sh,
                "lt": lt,
            }
        )
    return levels


def _resolve_dict(
    store: "SentenceStore", vocab_size: Optional[int], in_dict: Optional[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """Dictionary membership mask + |T| (= #types + <UNK>). See Sec. 6.3.1."""
    vs = int(vocab_size) if vocab_size is not None else int(store.flat.max()) + 1
    if in_dict is None:
        in_dict = np.zeros(vs, dtype=np.uint8)
        if store.n_tokens:
            in_dict[np.unique(store.flat)] = 1
        in_dict[UNK_ID] = 0
        in_dict[EOS_ID] = 0
        in_dict[BOS_ID] = 0
    else:
        in_dict = np.ascontiguousarray(in_dict, dtype=np.uint8)
        if len(in_dict) < vs:
            in_dict = np.pad(in_dict, (0, vs - len(in_dict)))
    return in_dict, int(in_dict.sum()) + 1


def _per_level(value: Union[float, Sequence[float]], N: int, name: str) -> np.ndarray:
    """Broadcast a scalar hyper-parameter to one value per order 1..N."""
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        arr = np.full(N, float(arr[0]))
    elif arr.size != N:
        raise ValueError(f"`{name}` must be a scalar or a sequence of length N={N}.")
    return np.ascontiguousarray(arr)


class BaseGrammarModel:
    """Common interface of every probability engine: a distribution over sentences."""

    __slots__ = ()

    N: int
    V: int
    in_dict: np.ndarray

    def token_logprobs(self, store: "SentenceStore") -> np.ndarray:  # pragma: no cover
        """Natural-log P(t_k | t_<k; G) per token of `store`, <EOS> included."""
        raise NotImplementedError

    def sentence_logprobs(self, store: "SentenceStore") -> np.ndarray:
        """Natural-log P(S_i; G) per sentence -- Eq. (1) / Eq. (14)."""
        lp = self.token_logprobs(store)
        bounds = np.zeros(store.n_sentences + 1, dtype=np.int64)
        np.cumsum(store.lengths() + 1, out=bounds[1:])
        return np.add.reduceat(lp, bounds[:-1]) if store.n_sentences else np.zeros(0)

    def perplexity(self, store: "SentenceStore") -> float:
        """exp of the cross-entropy rate (per token incl. <EOS>) -- see Sec. 5."""
        return float(np.exp(-self.token_logprobs(store).mean()))

    def code_length_bits(self, store: "SentenceStore") -> float:
        """
        -log2 P(store; G): the length in bits of an optimal encoding of `store`
        under this model. Sec. 5 of the LambdaG paper spells out the equivalence --
        "a language model can be seen as a text-compression scheme".

        With :class:`PPMGrammarModel` fitted on S_A and ``adaptive=True``, this is
        exactly C(D_U | D_A), the *conditional* compression of the questioned
        document given the known one. Halvani et al. (arXiv:1706.00516, Sec. 3) note
        that off-the-shelf compressors cannot take an auxiliary input and therefore
        approximate C(x|y) ~ C(xy) - C(y); here it is computed directly.
        """
        return float(-self.token_logprobs(store).sum() / math.log(2.0))


class KNGrammarModel(BaseGrammarModel):
    """
    An order-N interpolated Kneser-Ney n-gram model = a "Grammar Model" G.

    Implements Sec. 6.3 of the paper with a constant discount ``D`` (default 0.75,
    i.e. the original Kneser & Ney (1995) formulation, r = 1 in the modified-KN
    parameterisation of Chen & Goodman).

    Stored tables, per order m = 1..N (all keys sorted for binary search):
        gram_keys[m] : hash of each distinct m-gram
        gram_ckn[m]  : c_KN, i.e. the raw count c(.) at m == N, and the
                       continuation count N1+(*g) at m < N     (Eq. 19)
        ctx_keys[m]  : hash of each distinct (m-1)-gram context g
        ctx_S[m]     : S(g) = sum_{t' in T*} c_KN(g t')  -- the shared denominator
        ctx_N1p[m]   : N1+(g*) = #distinct continuations of g in T*
    """

    __slots__ = (
        "N", "D", "V", "in_dict",
        "gram_keys", "gram_ckn", "gram_off",
        "ctx_keys", "ctx_S", "ctx_N1p", "ctx_off",
        "n_sentences", "n_tokens",
    )

    def __init__(self, N: int, D: float) -> None:
        self.N = int(N)
        self.D = float(D)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def fit(
        cls,
        store: SentenceStore,
        N: int = 10,
        D: float = 0.75,
        vocab_size: Optional[int] = None,
        in_dict: Optional[np.ndarray] = None,
    ) -> "KNGrammarModel":
        """
        Estimate G from a set of tokenised sentences (line 10/13 of Algorithm 1).

        Parameters
        ----------
        store : SentenceStore
            Training sentences, encoded against the global vocabulary.
        in_dict : uint8[vocab_size], optional
            Explicit dictionary membership mask. If ``None`` (default, and what the
            reference R implementation does), the dictionary is the set of types
            occurring in ``store`` -- so every model gets its own |T|.
        """
        if store.n_sentences == 0:
            raise ValueError("Cannot fit a grammar model on an empty sentence set.")

        self = cls(N, D)
        self.n_sentences = store.n_sentences
        self.n_tokens = store.n_tokens
        # |T| = #dictionary types + <UNK>;  Eq. (16) divides by |T| + 1 (for <EOS>)
        self.in_dict, self.V = _resolve_dict(store, vocab_size, in_dict)
        levels = _build_levels(store, N)

        # ---- c_KN, Eq. (19): raw counts at the top order, continuation counts below
        ckn: List[np.ndarray] = [None] * N  # type: ignore[list-item]
        ckn[N - 1] = levels[N - 1]["counts"]
        for n in range(N - 1, 0, -1):  # n = order (1-indexed) < N; levels[n] is order n+1
            up = levels[n]
            sfx = up["sh"][up["rep"]]  # (n)-suffix of every *distinct* (n+1)-gram
            keys = levels[n - 1]["keys"]
            idx = np.searchsorted(keys, sfx)
            # every such suffix is guaranteed to exist as an n-gram under n-padding
            ckn[n - 1] = np.bincount(idx, minlength=len(keys)).astype(np.float64)

        # ---- context tables: aggregate over continuations t' in T* (i.e. t' != <BOS>)
        ctx_keys_l, ctx_S_l, ctx_N1p_l = [], [], []
        for n in range(1, N + 1):
            lv = levels[n - 1]
            rep = lv["rep"]
            mask = lv["lt"][rep] != BOS_ID  # drop the <BOS>^n gram: <BOS> not in T*
            pk = lv["ph"][rep][mask]
            cv = ckn[n - 1][mask]
            if len(pk) == 0:
                ctx_keys_l.append(np.zeros(0, np.uint64))
                ctx_S_l.append(np.zeros(0, np.float64))
                ctx_N1p_l.append(np.zeros(0, np.float64))
                continue
            o = np.argsort(pk, kind="stable")
            pk_s, cv_s = pk[o], cv[o]
            new = np.empty(len(pk_s), dtype=bool)
            new[0] = True
            np.not_equal(pk_s[1:], pk_s[:-1], out=new[1:])
            starts = np.flatnonzero(new)
            ctx_keys_l.append(pk_s[starts])
            ctx_S_l.append(np.add.reduceat(cv_s, starts))
            ctx_N1p_l.append(np.diff(np.append(starts, len(pk_s))).astype(np.float64))

        # ---- flatten into contiguous arrays + per-order offsets (Numba-friendly)
        self.gram_keys = np.concatenate([lv["keys"] for lv in levels])
        self.gram_ckn = np.concatenate(ckn)
        self.gram_off = np.zeros(N + 1, dtype=np.int64)
        np.cumsum([len(lv["keys"]) for lv in levels], out=self.gram_off[1:])

        self.ctx_keys = np.concatenate(ctx_keys_l)
        self.ctx_S = np.concatenate(ctx_S_l)
        self.ctx_N1p = np.concatenate(ctx_N1p_l)
        self.ctx_off = np.zeros(N + 1, dtype=np.int64)
        np.cumsum([len(k) for k in ctx_keys_l], out=self.ctx_off[1:])
        return self

    # -- inference ---------------------------------------------------------- #

    def token_logprobs(self, store: SentenceStore) -> np.ndarray:
        """
        Natural-log P(t_k | t_<k; G) for every token of `store`, <EOS> included.

        Length = store.n_tokens + store.n_sentences.
        """
        return _model_logprobs(
            store.flat, store.offsets, self.in_dict, UNK_ID,
            np.int64(self.N), np.float64(self.D), np.float64(1.0 / (self.V + 1)),
            self.gram_keys, self.gram_ckn, self.gram_off,
            self.ctx_keys, self.ctx_S, self.ctx_N1p, self.ctx_off,
            BOS_ID, EOS_ID,
        )


# --------------------------------------------------------------------------- #
# 6b. Hierarchical Pitman-Yor process grammar model                            #
# --------------------------------------------------------------------------- #


@njit
def _expected_tables_kernel(c, d, theta, out):
    """
    E[#tables | c customers] in a Pitman-Yor CRP(d, theta), per dish.

        E[T_c] = (theta/d) * [ Gamma(theta+d+c) Gamma(theta)
                             / (Gamma(theta+d) Gamma(theta+c)) - 1 ]

    with the theta -> 0 limit  E[T_c] = Gamma(c+d) / (Gamma(d+1) Gamma(c)).
    Sanity: E[T_1] = 1 and E[T_2] = 1 + d, as the CRP requires. Table counts grow
    like c^d -- the power law that separates PY from both KN (t = 1, flat) and a
    Dirichlet process (t ~ log c).

    This is a per-dish approximation: it ignores the coupling between dishes that
    runs through the restaurant-wide total t_u. The exact posterior needs the
    Gibbs sampler of Teh (2006), which is far too slow for LambdaG's r+1 models
    per case. Counts are clamped to the admissible range max(1, .) <= t <= c.
    """
    for i in range(c.shape[0]):
        ci = c[i]
        if ci <= 0.0:
            out[i] = 0.0
            continue
        if theta == 0.0:
            t = math.exp(math.lgamma(ci + d) - math.lgamma(d + 1.0) - math.lgamma(ci))
        else:
            # expm1 rather than exp(.) - 1: the bracket is close to 0 for small c
            t = (theta / d) * math.expm1(
                math.lgamma(theta + d + ci)
                + math.lgamma(theta)
                - math.lgamma(theta + d)
                - math.lgamma(theta + ci)
            )
        lo = ci if ci < 1.0 else 1.0
        if t < lo:
            t = lo
        if t > ci:
            t = ci
        out[i] = t


def _estimate_tables(c: np.ndarray, d: float, theta: float, estimator: str) -> np.ndarray:
    if estimator == "minimal":
        # t_uw = 1[c_uw > 0]  ->  with theta = 0 this is exactly interpolated KN
        return np.minimum(c, 1.0)
    if estimator == "expected":
        if not (0.0 < d < 1.0):
            raise ValueError("table_estimator='expected' needs a discount d in (0, 1).")
        out = np.empty_like(c)
        _expected_tables_kernel(c, float(d), float(theta), out)
        return out
    raise ValueError("table_estimator must be 'minimal' or 'expected'")


@njit
def _sentence_logprobs_hpy(
    ids, N, dvec, thvec, inv_TV,
    gram_keys, gram_c, gram_t, gram_off,
    ctx_keys, ctx_C, ctx_T, ctx_off,
    bos, eos, out, out_start,
):
    """
    Natural-log of the HPY predictive distribution (Teh 2006, Eq. 8) per token.

        P(w|u) = [c_uw - d_|u| t_uw] / [theta_|u| + c_u.]
               + [theta_|u| + d_|u| t_u.] / [theta_|u| + c_u.] * P(w|pi(u))

    where pi(u) drops the *first* token of u, bottoming out at the uniform base
    measure 1/(|T|+1) over T*. Structurally identical to the KN kernel -- only the
    coefficients change -- which is exactly Teh's point.
    """
    z = ids.shape[0]
    hist = np.empty(N, dtype=np.int32)

    for k in range(z + 1):
        t = ids[k] if k < z else eos
        for j in range(N - 1):
            idx = k - (N - 1) + j
            hist[j] = bos if idx < 0 else ids[idx]

        p = inv_TV
        for m in range(1, N + 1):
            cstart = N - 1 - (m - 1)
            hc = _H_INIT
            for i in range(cstart, N - 1):
                hc = _hstep(hc, hist[i])
            hg = _hstep(hc, t)

            ci = _bsearch(ctx_keys, ctx_off[m - 1], ctx_off[m], hc)
            if ci < 0:
                continue  # unseen context -> the restaurant is empty, defer to parent

            d = dvec[m - 1]
            th = thvec[m - 1]
            denom = th + ctx_C[ci]

            gi = _bsearch(gram_keys, gram_off[m - 1], gram_off[m], hg)
            a = 0.0
            if gi >= 0:
                a = gram_c[gi] - d * gram_t[gi]
                if a < 0.0:
                    a = 0.0
            p = a / denom + ((th + d * ctx_T[ci]) / denom) * p

        out[out_start + k] = math.log(p)
    return z + 1


@njit
def _model_logprobs_hpy(
    flat, offsets, in_dict, unk, N, dvec, thvec, inv_TV,
    gram_keys, gram_c, gram_t, gram_off,
    ctx_keys, ctx_C, ctx_T, ctx_off, bos, eos,
):
    n_sent = offsets.shape[0] - 1
    out = np.empty(flat.shape[0] + n_sent, dtype=np.float64)
    vsz = in_dict.shape[0]
    w = 0
    for s in range(n_sent):
        a = offsets[s]
        b = offsets[s + 1]
        z = b - a
        ids = np.empty(z, dtype=np.int32)
        for i in range(z):
            g = flat[a + i]
            ids[i] = g if (g < vsz and in_dict[g] == 1) else unk
        w += _sentence_logprobs_hpy(
            ids, N, dvec, thvec, inv_TV,
            gram_keys, gram_c, gram_t, gram_off,
            ctx_keys, ctx_C, ctx_T, ctx_off,
            bos, eos, out, w,
        )
    return out


class HPYGrammarModel(BaseGrammarModel):
    """
    An order-N hierarchical Pitman-Yor process language model -- Teh (2006),
    "A Hierarchical Bayesian Language Model based on Pitman-Yor Processes".

    Each context u is a restaurant in a Chinese restaurant franchise:

        G_u ~ PY(d_|u|, theta_|u|, G_pi(u)),      G_empty ~ PY(d_0, theta_0, Uniform)

    with c_uw customers and t_uw tables serving dish w. The customers of a parent
    restaurant *are* the tables of its children, so the counts propagate downwards:

        c_{pi(u),w} = sum_{u' : pi(u')=u} t_{u'w}

    Setting t_uw = 1[c_uw > 0] turns that sum into "count the distinct left
    extensions", i.e. exactly Kneser-Ney's continuation count N1+(*g) -- which is
    why ``concentration=0, table_estimator="minimal"`` reproduces KN exactly.

    Why bother, given LambdaG already works with KN? The PY process generates
    Zipfian/power-law type distributions, which is what natural language actually
    does; ``table_estimator="expected"`` lets a context that has seen a token many
    times open many tables (t ~ c^d), so frequent-in-context tokens push more mass
    down to the parent than rare ones. KN flattens that to t = 1 regardless. In the
    paper's own terms (Sec. 2), this is a graded rather than binary notion of how
    much a chunk's entrenchment propagates to less specific contexts.

    Parameters
    ----------
    N : int, default 10
    discount : float or sequence of N floats, default 0.75
        d_|u| in (0, 1), per context length 0..N-1. Teh fits these by sampling and
        finds them increasing with order (~0.6 -> ~0.9); a scalar 0.75 keeps the
        comparison against the paper's KN (D=0.75) controlled. Pass e.g.
        ``np.linspace(0.5, 0.9, N)`` for a schedule.
    concentration : float or sequence of N floats, default 0.0
        theta_|u| > -d. 0 recovers a pure Pitman-Yor discount rule; larger values
        pull the model toward its parent (more smoothing).
    table_estimator : {"expected", "minimal"}, default "expected"
        How t_uw is approximated -- see :func:`_expected_tables_kernel`.
    """

    __slots__ = (
        "N", "d", "theta", "V", "in_dict", "table_estimator",
        "gram_keys", "gram_c", "gram_t", "gram_off",
        "ctx_keys", "ctx_C", "ctx_T", "ctx_off",
        "n_sentences", "n_tokens",
    )

    def __init__(self, N: int, d: np.ndarray, theta: np.ndarray, table_estimator: str) -> None:
        self.N = int(N)
        self.d = d
        self.theta = theta
        self.table_estimator = table_estimator

    @classmethod
    def fit(
        cls,
        store: SentenceStore,
        N: int = 10,
        discount: Union[float, Sequence[float]] = 0.75,
        concentration: Union[float, Sequence[float]] = 0.0,
        table_estimator: str = "expected",
        vocab_size: Optional[int] = None,
        in_dict: Optional[np.ndarray] = None,
    ) -> "HPYGrammarModel":
        if store.n_sentences == 0:
            raise ValueError("Cannot fit a grammar model on an empty sentence set.")
        d = _per_level(discount, N, "discount")
        th = _per_level(concentration, N, "concentration")
        if np.any(d < 0) or np.any(d >= 1):
            raise ValueError("discount must lie in [0, 1).")
        if np.any(th <= -d):
            raise ValueError("concentration must satisfy theta > -d.")

        self = cls(N, d, th, table_estimator)
        self.n_sentences = store.n_sentences
        self.n_tokens = store.n_tokens
        self.in_dict, self.V = _resolve_dict(store, vocab_size, in_dict)
        levels = _build_levels(store, N)

        # ---- customers and tables, top order downwards
        c: List[np.ndarray] = [None] * N  # type: ignore[list-item]
        t: List[np.ndarray] = [None] * N  # type: ignore[list-item]
        c[N - 1] = levels[N - 1]["counts"]  # top order: real customers = raw counts
        t[N - 1] = _estimate_tables(c[N - 1], d[N - 1], th[N - 1], table_estimator)
        for n in range(N - 1, 0, -1):
            up = levels[n]  # order n+1
            sfx = up["sh"][up["rep"]]  # its n-suffix = the parent's n-gram
            keys = levels[n - 1]["keys"]
            idx = np.searchsorted(keys, sfx)
            # the parent's customers are the children's tables (cf. KN's bincount
            # with no weights, which is this with every t == 1)
            c[n - 1] = np.bincount(idx, weights=t[n], minlength=len(keys))
            t[n - 1] = _estimate_tables(c[n - 1], d[n - 1], th[n - 1], table_estimator)

        # ---- restaurant totals c_u. and t_u. over dishes in T* (i.e. w != <BOS>)
        ck_l, cC_l, cT_l = [], [], []
        for n in range(1, N + 1):
            lv = levels[n - 1]
            rep = lv["rep"]
            mask = lv["lt"][rep] != BOS_ID
            pk = lv["ph"][rep][mask]
            cv, tv = c[n - 1][mask], t[n - 1][mask]
            if len(pk) == 0:
                ck_l.append(np.zeros(0, np.uint64))
                cC_l.append(np.zeros(0, np.float64))
                cT_l.append(np.zeros(0, np.float64))
                continue
            o = np.argsort(pk, kind="stable")
            pk_s, cv_s, tv_s = pk[o], cv[o], tv[o]
            new = np.empty(len(pk_s), dtype=bool)
            new[0] = True
            np.not_equal(pk_s[1:], pk_s[:-1], out=new[1:])
            starts = np.flatnonzero(new)
            ck_l.append(pk_s[starts])
            cC_l.append(np.add.reduceat(cv_s, starts))
            cT_l.append(np.add.reduceat(tv_s, starts))

        self.gram_keys = np.concatenate([lv["keys"] for lv in levels])
        self.gram_c = np.concatenate(c)
        self.gram_t = np.concatenate(t)
        self.gram_off = np.zeros(N + 1, dtype=np.int64)
        np.cumsum([len(lv["keys"]) for lv in levels], out=self.gram_off[1:])

        self.ctx_keys = np.concatenate(ck_l)
        self.ctx_C = np.concatenate(cC_l)
        self.ctx_T = np.concatenate(cT_l)
        self.ctx_off = np.zeros(N + 1, dtype=np.int64)
        np.cumsum([len(k) for k in ck_l], out=self.ctx_off[1:])
        return self

    def token_logprobs(self, store: SentenceStore) -> np.ndarray:
        return _model_logprobs_hpy(
            store.flat, store.offsets, self.in_dict, UNK_ID,
            np.int64(self.N), self.d, self.theta, np.float64(1.0 / (self.V + 1)),
            self.gram_keys, self.gram_c, self.gram_t, self.gram_off,
            self.ctx_keys, self.ctx_C, self.ctx_T, self.ctx_off,
            BOS_ID, EOS_ID,
        )


# --------------------------------------------------------------------------- #
# 6c. PPMd grammar model                                                       #
# --------------------------------------------------------------------------- #

_ESC_A, _ESC_C, _ESC_D = 0, 1, 2
_ESCAPE_CODES = {"a": _ESC_A, "c": _ESC_C, "d": _ESC_D}


@njit
def _ov_probe(ov_key, ov_state, key):
    """Open-addressing probe of the adaptive overlay. Returns (slot, found)."""
    mask = ov_key.shape[0] - 1
    slot = np.int64(key & np.uint64(mask))
    while ov_state[slot] == 1:
        if ov_key[slot] == key:
            return slot, True
        slot = (slot + 1) & mask
    return slot, False


@njit
def _sentence_logprobs_ppm(
    ids, N, esc_code, delta, exclusion, A_size, adaptive,
    ctx_keys, ctx_off, ctx_ptr, cont_tok, cont_c,
    bos, eos, excl, excl_list,
    delta_base, touched, ov_key, ov_state, ov_head, ov_used,
    ov_tok, ov_cnt, ov_next, counters,
    out, out_start,
):
    """
    Natural-log of the PPM predictive distribution per token.

    PPM *backs off* rather than interpolating: start at the longest context, and if
    the symbol has been seen there, stop -- lower orders contribute nothing. If not,
    pay an escape and descend, having *excluded* every symbol that the longer
    context did offer (they are known not to be the answer, so their counts must not
    dilute the shorter context). Contexts absent from the model are skipped for
    free. The chain ends at order -1: uniform over the not-yet-excluded part of T*.

    Escape methods (n = total count in context, q = #distinct symbols, both after
    exclusion):
        A   P = c/(n+1)        escape = 1/(n+1)
        C   P = c/(n+q)        escape = q/(n+q)          (Moffat)
        D   P = (c-delta)/n    escape = delta*q/n        (Howard; delta = 1/2)

    Method D is absolute discounting -- the very same (c - delta)/n and
    delta*q/n shape as KN's alpha and gamma. The differences that matter are that
    PPM uses *raw* counts at every order rather than continuation counts, and backs
    off with exclusion rather than interpolating. The telescoping
    sum_k (prod_{j>k} e_j)(1 - e_k) + prod_j e_j = 1 makes this a proper
    distribution over T*.

    If `adaptive`, counts are incremented after each token is coded, exactly as a
    real compressor does, and the increments live in a small overlay on top of the
    immutable base tables: `delta_base` for (context, token) pairs the base already
    knows, and a chained hash table for genuinely novel ones. Both are cleared by
    the caller at whatever reset boundary was requested. Counters:
        counters[0] = #touched base entries, counters[1] = #used overlay slots,
        counters[2] = #overlay nodes allocated.
    """
    z = ids.shape[0]
    hist = np.empty(N, dtype=np.int32)
    hcs = np.empty(N, dtype=np.uint64)  # context hash per order, computed once

    for k in range(z + 1):
        t = ids[k] if k < z else eos
        for j in range(N - 1):
            idx = k - (N - 1) + j
            hist[j] = bos if idx < 0 else ids[idx]

        for m in range(1, N + 1):
            h = _H_INIT
            for i in range(N - 1 - (m - 1), N - 1):
                h = _hstep(h, hist[i])
            hcs[m - 1] = h

        esc = 1.0
        n_excl = 0
        p = -1.0

        for m in range(N, 0, -1):  # longest context first
            hc = hcs[m - 1]
            ci = _bsearch(ctx_keys, ctx_off[m - 1], ctx_off[m], hc)
            lo = 0
            hi = 0
            if ci >= 0:
                lo = ctx_ptr[ci]
                hi = ctx_ptr[ci + 1]

            node0 = -1
            if adaptive:
                slot, found = _ov_probe(ov_key, ov_state, _hstep(hc, m))
                if found:
                    node0 = ov_head[slot]

            if ci < 0 and node0 < 0:
                continue  # context never seen: descend without charging an escape

            ns = 0.0
            qs = 0.0
            cw = -1.0
            for j in range(lo, hi):  # continuations the base model knows
                w = cont_tok[j]
                if exclusion and excl[w] == 1:
                    continue
                cj = cont_c[j] + delta_base[j]
                ns += cj
                qs += 1.0
                if w == t:
                    cw = cj
            node = node0
            while node >= 0:  # continuations only this document has produced
                w = ov_tok[node]
                if not (exclusion and excl[w] == 1):
                    ns += ov_cnt[node]
                    qs += 1.0
                    if w == t:
                        cw = ov_cnt[node]
                node = ov_next[node]

            if ns <= 0.0:
                continue  # everything here is already excluded: no information

            if cw >= 0.0:  # symbol seen in this context -> emit and stop
                if esc_code == 0:
                    a_num = cw / (ns + 1.0)
                elif esc_code == 1:
                    a_num = cw / (ns + qs)
                else:
                    a_num = (cw - delta) / ns
                if m == 1 and exclusion:
                    # An escape must be zero when there is nothing left to escape
                    # to. Order -1 hands its uniform mass to the symbols of T* that
                    # no context offered; if that set is empty, the order-0 escape
                    # mass has nowhere to go and would simply leak. This cannot
                    # happen to a static model -- <UNK> is by construction never
                    # observed in training, so it always keeps order -1 non-empty --
                    # but an *adaptive* model counts D_U's own OOV tokens as <UNK>
                    # and can exhaust the alphabet. Suppress e_0 and rescale.
                    if A_size - n_excl - qs <= 0.0:
                        if esc_code == 0:
                            e0 = 1.0 / (ns + 1.0)
                        elif esc_code == 1:
                            e0 = qs / (ns + qs)
                        else:
                            e0 = delta * qs / ns
                        a_num = a_num / (1.0 - e0)
                p = esc * a_num
                break

            if esc_code == 0:  # escape to the next shorter context
                esc *= 1.0 / (ns + 1.0)
            elif esc_code == 1:
                esc *= qs / (ns + qs)
            else:
                esc *= delta * qs / ns
            if exclusion:
                for j in range(lo, hi):
                    w = cont_tok[j]
                    if excl[w] == 0:
                        excl[w] = 1
                        excl_list[n_excl] = w
                        n_excl += 1
                node = node0
                while node >= 0:
                    w = ov_tok[node]
                    if excl[w] == 0:
                        excl[w] = 1
                        excl_list[n_excl] = w
                        n_excl += 1
                    node = ov_next[node]

        if p < 0.0:  # order -1
            p = esc / (A_size - n_excl if exclusion else A_size)

        for i in range(n_excl):  # cheap reset: only touch what we set
            excl[excl_list[i]] = 0

        out[out_start + k] = math.log(p)

        # ---- adaptive update: the symbol has now been "coded", so count it
        if adaptive:
            for m in range(1, N + 1):
                hc = hcs[m - 1]
                ci = _bsearch(ctx_keys, ctx_off[m - 1], ctx_off[m], hc)
                in_base = False
                if ci >= 0:
                    lo = ctx_ptr[ci]
                    hi = ctx_ptr[ci + 1]
                    j = _bsearch(cont_tok, lo, hi, t)  # tokens sorted within context
                    if j >= 0:
                        if delta_base[j] == 0.0:
                            touched[counters[0]] = j
                            counters[0] += 1
                        delta_base[j] += 1.0
                        in_base = True
                if not in_base:
                    okey = _hstep(hc, m)
                    slot, found = _ov_probe(ov_key, ov_state, okey)
                    if not found:
                        ov_state[slot] = 1
                        ov_key[slot] = okey
                        ov_head[slot] = -1
                        ov_used[counters[1]] = slot
                        counters[1] += 1
                    node = ov_head[slot]
                    hit = -1
                    while node >= 0:
                        if ov_tok[node] == t:
                            hit = node
                            break
                        node = ov_next[node]
                    if hit >= 0:
                        ov_cnt[hit] += 1.0
                    else:
                        nn = counters[2]
                        counters[2] += 1
                        ov_tok[nn] = t
                        ov_cnt[nn] = 1.0
                        ov_next[nn] = ov_head[slot]
                        ov_head[slot] = nn
    return z + 1


@njit
def _ov_clear(delta_base, touched, ov_state, ov_used, counters):
    """Roll the model back to the reference counts. O(what was touched)."""
    for i in range(counters[0]):
        delta_base[touched[i]] = 0.0
    for i in range(counters[1]):
        ov_state[ov_used[i]] = 0
    counters[0] = 0
    counters[1] = 0
    counters[2] = 0


@njit
def _model_logprobs_ppm(
    flat, offsets, in_dict, unk, N, esc_code, delta, exclusion, A_size,
    adaptive, reset_per_sentence,
    ctx_keys, ctx_off, ctx_ptr, cont_tok, cont_c, bos, eos,
):
    n_sent = offsets.shape[0] - 1
    n_tok = flat.shape[0]
    out = np.empty(n_tok + n_sent, dtype=np.float64)
    vsz = in_dict.shape[0]
    excl = np.zeros(vsz, dtype=np.uint8)
    excl_list = np.empty(vsz, dtype=np.int32)

    # Overlay: at most N new (context, token) increments per coded symbol.
    n_slots = (n_tok + n_sent) * N + 1 if adaptive else 1
    cap = 1
    while cap < 2 * n_slots:
        cap *= 2
    delta_base = np.zeros(cont_c.shape[0], dtype=np.float64)
    touched = np.empty(n_slots, dtype=np.int64)
    ov_key = np.zeros(cap, dtype=np.uint64)
    ov_state = np.zeros(cap, dtype=np.uint8)
    ov_head = np.full(cap, -1, dtype=np.int64)
    ov_used = np.empty(n_slots, dtype=np.int64)
    ov_tok = np.empty(n_slots, dtype=np.int32)
    ov_cnt = np.empty(n_slots, dtype=np.float64)
    ov_next = np.empty(n_slots, dtype=np.int64)
    counters = np.zeros(3, dtype=np.int64)

    w = 0
    for s in range(n_sent):
        a = offsets[s]
        b = offsets[s + 1]
        z = b - a
        ids = np.empty(z, dtype=np.int32)
        for i in range(z):
            g = flat[a + i]
            ids[i] = g if (g < vsz and in_dict[g] == 1) else unk
        w += _sentence_logprobs_ppm(
            ids, N, esc_code, delta, exclusion, A_size, adaptive,
            ctx_keys, ctx_off, ctx_ptr, cont_tok, cont_c,
            bos, eos, excl, excl_list,
            delta_base, touched, ov_key, ov_state, ov_head, ov_used,
            ov_tok, ov_cnt, ov_next, counters, out, w,
        )
        if adaptive and reset_per_sentence:
            _ov_clear(delta_base, touched, ov_state, ov_used, counters)
    return out


class PPMGrammarModel(BaseGrammarModel):
    """
    An order-N PPM model -- Cleary & Witten (1984); escape method D from Howard (1993).

    The LambdaG paper (Sec. 1.4.3) notes that PPM "could also be seen as a
    character-level language model", that COAV uses a PPM variant called PPMd as its
    engine, and (Sec. 5) that language modelling and compression are the same thing
    viewed through cross-entropy. This engine takes that seriously: it drops PPMd in
    as LambdaG's P(t_k|t_<k; G), giving a hybrid of the paper's two strongest
    non-neural baselines -- COAV's compression engine inside LambdaG's likelihood
    ratio, over POSNoise function tokens rather than characters.

    Note what COAV itself (arXiv:1706.00516) actually does: it calls PPMd as an
    off-the-shelf black box and reads three *byte counts* C(x), C(y), C(xy) into a
    dissimilarity measure. It never sees a per-token probability. This class does,
    which is what a likelihood ratio needs.

    Two deliberate deviations from a real compressor:

    * **Not Shkarin's PPMd.** The `PPMd`/PPMII program in 7-Zip and RAR that COAV
      invokes is a heavily engineered heuristic (information inheritance, secondary
      escape estimation, model restarts on memory exhaustion) and is not a clean,
      reproducible probability model. What is implemented here is escape method D,
      the well-defined textbook "PPMd".
    * **Full count updates.** Counts are incremented at every order, i.e. no update
      exclusion.

    Parameters
    ----------
    N : int, default 10
    escape : {"d", "c", "a"}, default "d"
    discount : float, default 0.5
        The delta of method D. 0.5 is the classic value; it is the same knob as KN's D.
    exclusion : bool, default True
        Exclude symbols offered by longer contexts when computing shorter-context
        statistics. This is what makes PPM strong.

        **Leave this on.** With exclusion, the escape chain telescopes and
        P(.|g) sums to exactly 1 over T*. Without it, escape mass is spread over
        symbols the longer contexts had already offered, so the model is
        *sub-normalised*: sum_w P(w|g) < 1, by ~0.2-0.4 in practice. The deficiency
        depends on the count profile, so it does **not** cancel between G_A and
        G_j -- it biases lambda_G rather than shifting it. The flag is kept only for
        ablation and speed; anything reported from it is not a likelihood ratio in
        the sense of Sec. 6.4.
    adaptive : bool, default False
        If True, code the questioned document the way a compressor would: increment
        counts as each token is consumed, so that P(t_k | t_<k) conditions on the
        document's own history as well as on the training data.

        This is the *Bayesian predictive* P(D_U | D_A) rather than the plug-in
        estimate P(D_U | theta-hat(D_A)) -- arguably the more coherent numerator for
        the LR framework of Sec. 6.4 -- and it makes
        :meth:`BaseGrammarModel.code_length_bits` equal to C(D_U | D_A) exactly,
        the conditional compression that COAV could only approximate as
        C(xy) - C(y). It remains a proper distribution: the model state is a
        deterministic function of t_<k, so sum_w P(w | t_<k) = 1 still holds.

        The catch is that adaptation is **asymmetric inside a ratio**: a reference
        model G_j that has never seen a pattern gains proportionally far more from
        each new observation of it than G_A, which has. So a long D_U lets the
        denominator learn from the very document it is being tested against, pulling
        lambda_G toward 0. See `reset`.
    reset : {"sentence", "document"}, default "sentence"
        Where the adaptive counts are rolled back to the training-only state.

        ``"sentence"`` restarts at every sentence boundary, so only the reference
        counts ever persist. This bounds the dilution above to the ~15-25 tokens of
        a single POSNoise sentence, and it restores Eq. (13)'s assumption that
        sentences are independent -- yielding a sentence-level predictive
        P(D_U|D_A) = prod_i P(S_i|D_A). Recommended.

        ``"document"`` never rolls back, so D_U is coded as one adaptive stream,
        exactly as C(D_A D_U) - C(D_A) would. Faithful to compression, but it
        violates sentence independence and maximises the dilution.

        Ignored when ``adaptive=False``.
    """

    __slots__ = (
        "N", "escape", "delta", "exclusion", "adaptive", "reset", "V", "in_dict",
        "ctx_keys", "ctx_off", "ctx_ptr", "cont_tok", "cont_c",
        "n_sentences", "n_tokens",
    )

    def __init__(
        self, N: int, escape: str, delta: float, exclusion: bool,
        adaptive: bool = False, reset: str = "sentence",
    ) -> None:
        self.N = int(N)
        self.escape = escape
        self.delta = float(delta)
        self.exclusion = bool(exclusion)
        self.adaptive = bool(adaptive)
        self.reset = reset

    @classmethod
    def fit(
        cls,
        store: SentenceStore,
        N: int = 10,
        escape: str = "d",
        discount: float = 0.5,
        exclusion: bool = True,
        adaptive: bool = False,
        reset: str = "sentence",
        vocab_size: Optional[int] = None,
        in_dict: Optional[np.ndarray] = None,
    ) -> "PPMGrammarModel":
        if store.n_sentences == 0:
            raise ValueError("Cannot fit a grammar model on an empty sentence set.")
        escape = str(escape).lower()
        if escape not in _ESCAPE_CODES:
            raise ValueError("escape must be one of 'a', 'c', 'd'")
        if escape == "d" and not (0.0 < discount < 1.0):
            raise ValueError("discount (the delta of method D) must lie in (0, 1).")
        if reset not in ("sentence", "document"):
            raise ValueError("reset must be 'sentence' or 'document'")

        self = cls(N, escape, discount, exclusion, adaptive, reset)
        self.n_sentences = store.n_sentences
        self.n_tokens = store.n_tokens
        self.in_dict, self.V = _resolve_dict(store, vocab_size, in_dict)
        levels = _build_levels(store, N)

        # PPM needs to *enumerate* a context's continuations (for exclusion) and to
        # look one up (for the count), so grams are stored grouped by context and
        # sorted by token within it -- a CSR layout -- rather than hashed per gram.
        # Sorting by token also lets the adaptive update binary-search the base.
        ck_l, ptr_l, tok_l, cnt_l = [], [np.zeros(1, dtype=np.int64)], [], []
        ctx_off = np.zeros(N + 1, dtype=np.int64)
        base = 0
        for n in range(1, N + 1):
            lv = levels[n - 1]
            rep = lv["rep"]
            mask = lv["lt"][rep] != BOS_ID  # <BOS> is not a symbol of T*
            pk = lv["ph"][rep][mask]
            tk = lv["lt"][rep][mask]
            cv = lv["counts"][mask]  # PPM uses *raw* counts at every order
            if len(pk) == 0:
                ctx_off[n] = ctx_off[n - 1]
                continue
            o = np.lexsort((tk, pk))  # primary: context hash, secondary: token id
            pk_s, tk_s, cv_s = pk[o], tk[o], cv[o]
            new = np.empty(len(pk_s), dtype=bool)
            new[0] = True
            np.not_equal(pk_s[1:], pk_s[:-1], out=new[1:])
            starts = np.flatnonzero(new)
            ck_l.append(pk_s[starts])
            ptr_l.append(np.append(starts, len(pk_s))[1:] + base)
            tok_l.append(tk_s.astype(np.int32))
            cnt_l.append(cv_s)
            base += len(pk_s)
            ctx_off[n] = ctx_off[n - 1] + len(starts)

        self.ctx_keys = np.concatenate(ck_l) if ck_l else np.zeros(0, np.uint64)
        self.ctx_ptr = np.concatenate(ptr_l)
        self.cont_tok = np.concatenate(tok_l) if tok_l else np.zeros(0, np.int32)
        self.cont_c = np.concatenate(cnt_l) if cnt_l else np.zeros(0, np.float64)
        self.ctx_off = ctx_off
        return self

    def token_logprobs(self, store: SentenceStore) -> np.ndarray:
        return _model_logprobs_ppm(
            store.flat, store.offsets, self.in_dict, UNK_ID,
            np.int64(self.N), np.int64(_ESCAPE_CODES[self.escape]),
            np.float64(self.delta), self.exclusion, np.float64(self.V + 1),
            self.adaptive, self.reset == "sentence",
            self.ctx_keys, self.ctx_off, self.ctx_ptr, self.cont_tok, self.cont_c,
            BOS_ID, EOS_ID,
        )


#: The probability engines LambdaG can be driven by.
ENGINES: Dict[str, type] = {
    "kn": KNGrammarModel,
    "hpy": HPYGrammarModel,
    "ppmd": PPMGrammarModel,
}


# --------------------------------------------------------------------------- #
# 7. LambdaG                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class LambdaGResult:
    """Output of :meth:`LambdaG.score`, decomposed for explainability (Sec. 7)."""

    lambda_G: float
    tokens: List[List[str]] = field(default_factory=list)  # per sentence, <EOS> appended
    token_lambda: List[np.ndarray] = field(default_factory=list)
    sentence_lambda: np.ndarray = field(default_factory=lambda: np.zeros(0))
    log_base: float = 10.0
    n_ref_models: int = 0
    direction: str = "forward"
    lambda_forward: Optional[float] = None
    lambda_backward: Optional[float] = None
    # Statistics of the *masked* questioned document D_U, needed by the model-free
    # normalisation calibrations of Barlow, Nini & Manino (2026, arXiv:2607.09501).
    # n_query_tokens = N(Q) (masked token count); n_query_hapax = V1(Q) (types seen
    # exactly once in the masked text). See :func:`sqrt_correction` / :func:`hapax_correction`.
    n_query_tokens: int = 0
    n_query_hapax: int = 0

    @property
    def lambda_sqrt(self) -> float:
        """Square Root Correction (Eq. 7): a model-free log-LR = lambda_G / sqrt(N(Q))."""
        return sqrt_correction(self.lambda_G, self.n_query_tokens)

    @property
    def lambda_hapax(self) -> float:
        """Hapax Correction (Eq. 8): a model-free log-LR = lambda_G * V1(Q) / N(Q)."""
        return hapax_correction(self.lambda_G, self.n_query_tokens, self.n_query_hapax)

    @property
    def features(self) -> np.ndarray:
        """
        The score(s) to hand to :class:`LambdaGCalibrator`.

        For ``direction="both"`` this is ``[lambda_forward, lambda_backward]``, to be
        *fused* by the calibrator rather than pre-summed. `lambda_G` is their plain
        sum, which is convenient but is a product-of-experts score missing a per-case
        log Z(G_A)/Z(G_j); prefer fusing the two features.
        """
        if self.direction == "both":
            return np.array([self.lambda_forward, self.lambda_backward], dtype=np.float64)
        return np.array([self.lambda_G], dtype=np.float64)

    def top_sentences(self, k: int = 10, most_authorial: bool = True):
        """The k sentences that push hardest toward (or away from) the candidate."""
        order = np.argsort(self.sentence_lambda)
        order = order[::-1] if most_authorial else order
        return [
            (int(i), float(self.sentence_lambda[i]), self.tokens[i], self.token_lambda[i])
            for i in order[:k]
        ]

    def token_table(self):
        """Flat ``(sentence_idx, token, lambda)`` rows -- pipe into pandas if you like."""
        rows = []
        for si, (toks, lams) in enumerate(zip(self.tokens, self.token_lambda)):
            for t, l in zip(toks, lams):
                rows.append((si, t, float(l)))
        return rows


class LambdaG:
    """
    The Likelihood Ratio of Grammar Models (Algorithm 1 of the paper).

    Parameters
    ----------
    N : int, default 10
        n-gram order. Paper default; Fig. 3 shows the method is robust here and that
        nothing is gained beyond N=10 (N=2 suffices for chat/email corpora).
    r : int, default 30
        Number of reference grammar models. The paper's tables use r=100, but Fig. 3
        shows performance plateaus at r=30 -- which is also the R package's default.
    D : float, default 0.75
        Kneser-Ney discount (Sec. 6.3.4). Ignored unless ``engine="kn"``.
    engine : {"kn", "hpy", "ppmd"}, default "kn"
        Which probability engine estimates P(t_k | t_<k; G). Only the estimator
        changes -- POSNoise, the sampling of the r reference models, the log-ratio
        and the calibration are all identical, so the three are directly comparable.
        ``"kn"`` is the published method.
    direction : {"forward", "backward", "both"}, default "forward"
        Which way the grammar models read a sentence.

        ``"backward"`` fits and scores on reversed sentences, i.e. P(t_k | t_>k).
        This is a proper likelihood -- reversal is a bijection on sentences -- so
        lambda_G stays a genuine likelihood ratio, just under a mirrored
        independence assumption.

        ``"both"`` computes both (2x the cost) and exposes them as
        ``LambdaGResult.features = [lambda_forward, lambda_backward]``, to be *fused*
        by :class:`LambdaGCalibrator`, which accepts multi-feature input. This is
        standard forensic score fusion (Brummer & du Preez 2006 -- the same reference
        the paper cites for Cllr). ``result.lambda_G`` is their plain sum, which is
        convenient but is a product-of-experts score missing a per-case
        log Z(G_A)/Z(G_j); the fusion is the principled route.

        Caveat worth weighing before using anything but "forward": the paper's whole
        claim to superiority is scientific plausibility (Sec. 5), and that rests on
        grammar being *procedural memory for sequential prediction* (Sec. 2). Nobody
        produces language right-to-left, so a backward model has no entrenchment
        story behind it -- it detects the same chunks from the other side, but as a
        bare biometric rather than a model of production. In a courtroom (Sec. 6.4)
        that is a harder thing to defend. Treat "both" as an empirical option to test
        with `cllr_min` on your own data, not as a free upgrade.
    engine_params : dict, optional
        Forwarded to the engine's ``fit()``. Defaults per engine:
            kn    : ``D=0.75``
            hpy   : ``discount=0.75, concentration=0.0, table_estimator="expected"``
            ppmd  : ``escape="d", discount=0.5, exclusion=True,
                      adaptive=False, reset="sentence"``
    log_base : float, default 10.0
        Base of the reported log-ratio. 10 matches ``idiolect::lambdaG``; ``math.e``
        gives nats. Sign and ranking are unaffected.
    vocab_mode : {"per_model", "shared"}
        ``"per_model"`` (default) mirrors the reference implementation: G_A and each
        G_j carry their own dictionary. ``"shared"`` gives every model the union
        dictionary of S_A and S_ref, which makes |T| -- and therefore the uniform
        floor 1/(|T|+1) -- identical across numerator and denominator.
    sample_with_replacement : bool, default False
        R's ``sample()`` draws without replacement; this only matters when
        |S_ref| < |S_A|, where we fall back to replacement with a warning.
    random_state : int | np.random.Generator | None
        LambdaG is stochastic (Sec. 5.1); fix this for reproducible runs.
    """

    def __init__(
        self,
        N: int = 10,
        r: int = 30,
        D: float = 0.75,
        engine: str = "kn",
        engine_params: Optional[Dict] = None,
        direction: str = "forward",
        log_base: float = 10.0,
        vocab_mode: str = "per_model",
        sample_with_replacement: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        if vocab_mode not in ("per_model", "shared"):
            raise ValueError("vocab_mode must be 'per_model' or 'shared'")
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {sorted(ENGINES)}")
        if direction not in ("forward", "backward", "both"):
            raise ValueError("direction must be 'forward', 'backward' or 'both'")
        self.direction = direction
        self.N = int(N)
        self.r = int(r)
        self.D = float(D)
        self.engine = engine
        self.engine_params = dict(engine_params or {})
        if engine == "kn":
            self.engine_params.setdefault("D", self.D)
        self.log_base = float(log_base)
        self.vocab_mode = vocab_mode
        self.sample_with_replacement = bool(sample_with_replacement)
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        self.vocab = vocab if vocab is not None else Vocabulary()
        self._ref: Optional[SentenceStore] = None
        self._ref_groups: Optional[np.ndarray] = None

    # -- data ingestion ----------------------------------------------------- #

    def _fit_model(
        self, store: SentenceStore, vocab_size: int, in_dict: Optional[np.ndarray]
    ) -> BaseGrammarModel:
        """Build one grammar model with the configured engine."""
        return ENGINES[self.engine].fit(
            store, N=self.N, vocab_size=vocab_size, in_dict=in_dict, **self.engine_params
        )

    def encode(self, sentences: Sequence[Sequence[str]], grow: bool = True) -> SentenceStore:
        """POSNoise sentences -> SentenceStore against this instance's vocabulary."""
        return self.vocab.encode(sentences, grow=grow)

    def set_reference(
        self,
        ref_sentences: Union[Sequence[Sequence[str]], SentenceStore],
        groups: Optional[Sequence] = None,
    ) -> "LambdaG":
        """
        Register D_ref (line 9 of Algorithm 1).

        Parameters
        ----------
        ref_sentences : list of tokenised sentences, or a SentenceStore
            The *pooled* sentences of the reference population.
        groups : sequence, optional
            One label (e.g. author id) per reference sentence. If given, ``score(...,
            exclude_groups=[...])`` can drop the candidate's / Q author's own sentences
            from the pool, as the R implementation does.
        """
        self._ref = (
            ref_sentences
            if isinstance(ref_sentences, SentenceStore)
            else self.encode(ref_sentences)
        )
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != self._ref.n_sentences:
                raise ValueError("`groups` must have one entry per reference sentence.")
        self._ref_groups = groups
        return self

    # -- the algorithm ------------------------------------------------------ #

    def score(
        self,
        q_sentences: Union[Sequence[Sequence[str]], SentenceStore],
        k_sentences: Union[Sequence[Sequence[str]], SentenceStore],
        ref_sentences: Optional[Union[Sequence[Sequence[str]], SentenceStore]] = None,
        exclude_groups: Optional[Sequence] = None,
        r: Optional[int] = None,
        with_details: bool = True,
    ) -> LambdaGResult:
        """
        Compute lambda_G(S_U) for the verification case c = (D_U, D_A).

        Parameters
        ----------
        q_sentences : D_U, the questioned document (POSNoise sentences).
        k_sentences : D_A, the known document(s) of candidate A.
        ref_sentences : overrides the registered reference pool for this call.
        exclude_groups : labels to drop from the reference pool (needs ``groups``
            to have been passed to :meth:`set_reference`).
        with_details : also return per-token / per-sentence decompositions
            (Eqs. 2-4). Turn off in large batch runs to save a little memory.

        Returns
        -------
        LambdaGResult
        """
        r = self.r if r is None else int(r)

        S_U = q_sentences if isinstance(q_sentences, SentenceStore) else self.encode(q_sentences)
        S_A = k_sentences if isinstance(k_sentences, SentenceStore) else self.encode(k_sentences)

        if ref_sentences is not None:
            S_ref = (
                ref_sentences
                if isinstance(ref_sentences, SentenceStore)
                else self.encode(ref_sentences)
            )
            ref_groups = None
        else:
            if self._ref is None:
                raise RuntimeError("No reference corpus. Call set_reference() first.")
            S_ref, ref_groups = self._ref, self._ref_groups

        if exclude_groups is not None:
            if ref_groups is None:
                raise ValueError("exclude_groups requires groups= in set_reference().")
            keep = ~np.isin(ref_groups, np.asarray(exclude_groups))
            S_ref = S_ref.select(np.flatnonzero(keep))

        if S_A.n_sentences == 0 or S_U.n_sentences == 0:
            raise ValueError("D_U and D_A must each contain at least one sentence.")
        if S_ref.n_sentences == 0:
            raise ValueError("The reference pool is empty (all sentences excluded?).")

        vocab_size = len(self.vocab)
        shared_dict = None
        if self.vocab_mode == "shared":
            shared_dict = np.zeros(vocab_size, dtype=np.uint8)
            seen = np.unique(np.concatenate([S_A.flat, S_ref.flat]))
            shared_dict[seen] = 1
            shared_dict[[UNK_ID, EOS_ID, BOS_ID]] = 0

        direction = self.direction
        need_f = direction in ("forward", "both")
        need_b = direction in ("backward", "both")

        # Reversing a store is a bijection, so the same tokens/dictionary apply.
        S_U_r = S_U.reverse() if need_b else None
        S_A_r = S_A.reverse() if need_b else None
        S_ref_r = S_ref.reverse() if need_b else None

        # --- line 10: G_A, the candidate's grammar model
        logp_A = logp_A_r = None
        if need_f:
            logp_A = self._fit_model(S_A, vocab_size, shared_dict).token_logprobs(S_U)
        if need_b:
            logp_A_r = self._fit_model(S_A_r, vocab_size, shared_dict).token_logprobs(S_U_r)

        # --- lines 11-13: r reference grammar models, each on |S_A| sampled sentences
        n_draw = S_A.n_sentences
        replace = self.sample_with_replacement
        if n_draw > S_ref.n_sentences and not replace:
            warnings.warn(
                f"|S_A| = {n_draw} > |S_ref| = {S_ref.n_sentences}; sampling with "
                "replacement instead. Consider a larger reference corpus.",
                RuntimeWarning,
            )
            replace = True

        acc = np.zeros_like(logp_A) if need_f else None
        acc_r = np.zeros_like(logp_A_r) if need_b else None
        for _ in range(r):
            # one sample, both directions -- keeps the two scores strictly comparable
            idx = self.rng.choice(S_ref.n_sentences, size=n_draw, replace=replace)
            if need_f:
                acc += self._fit_model(S_ref.select(idx), vocab_size, shared_dict).token_logprobs(S_U)
            if need_b:
                acc_r += self._fit_model(
                    S_ref_r.select(idx), vocab_size, shared_dict
                ).token_logprobs(S_U_r)

        # --- lines 14-20: lambda_G(t_k|t_<k) = (1/r) sum_j log P(.;G_A)/P(.;G_j)
        scale = 1.0 / math.log(self.log_base)
        tok_f = (logp_A - acc / r) * scale if need_f else None
        tok_r = (logp_A_r - acc_r / r) * scale if need_b else None

        # Align the backward per-token scores onto the forward token order. The
        # backward pass reads a reversed sentence, so its slot k holds token z-k; the
        # final slot holds its <EOS>, which marks the *start* of the real sentence --
        # the mirror of the forward pass's sentence-final <EOS>. Reversing the first z
        # entries and keeping that boundary marker last puts the two arrays in
        # register, so per-token evidence from both directions can simply be added.
        tok_b = None
        if need_b:
            tok_b = np.empty_like(tok_r)
            for i in range(S_U.n_sentences):
                a = S_U.offsets[i] + i
                z = S_U.offsets[i + 1] - S_U.offsets[i]
                tok_b[a : a + z] = tok_r[a : a + z][::-1]
                tok_b[a + z] = tok_r[a + z]

        if direction == "forward":
            tok_lambda, lam_f, lam_b = tok_f, float(tok_f.sum()), None
        elif direction == "backward":
            tok_lambda, lam_f, lam_b = tok_b, None, float(tok_b.sum())
        else:
            tok_lambda = tok_f + tok_b
            lam_f, lam_b = float(tok_f.sum()), float(tok_b.sum())
        lam = float(tok_lambda.sum())

        # Masked-document statistics for the model-free normalisation calibrations
        # (arXiv:2607.09501). Both counts are taken on the *masked* questioned text --
        # the actual input to the model (paper Sec. 5.4) -- i.e. on S_U.flat, which
        # holds the encoded function tokens without <BOS>/<EOS> padding.
        n_q_tokens = int(S_U.n_tokens)
        if n_q_tokens:
            _, _counts = np.unique(S_U.flat, return_counts=True)
            n_q_hapax = int((_counts == 1).sum())
        else:
            n_q_hapax = 0

        res = LambdaGResult(
            lambda_G=lam, log_base=self.log_base, n_ref_models=r,
            direction=direction, lambda_forward=lam_f, lambda_backward=lam_b,
            n_query_tokens=n_q_tokens, n_query_hapax=n_q_hapax,
        )
        if with_details:
            bounds = np.zeros(S_U.n_sentences + 1, dtype=np.int64)
            np.cumsum(S_U.lengths() + 1, out=bounds[1:])
            res.sentence_lambda = np.add.reduceat(tok_lambda, bounds[:-1])
            res.token_lambda = [tok_lambda[bounds[i] : bounds[i + 1]] for i in range(S_U.n_sentences)]
            itos = self.vocab.token_of
            res.tokens = [
                [itos(int(t)) for t in S_U.flat[S_U.offsets[i] : S_U.offsets[i + 1]]] + ["<EOS>"]
                for i in range(S_U.n_sentences)
            ]
        return res

    # -- batch -------------------------------------------------------------- #

    def score_many(
        self,
        cases: Sequence[Tuple],
        progress: bool = True,
        correction: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Score a list of verification cases.

        Parameters
        ----------
        cases : sequence of ``(q_sentences, k_sentences)`` or
                ``(q_sentences, k_sentences, exclude_groups)``.
        correction : ``None`` (default) returns raw ``lambda_G``; ``"sqrt"`` or
            ``"hapax"`` returns the corresponding model-free normalisation
            calibration of Barlow, Nini & Manino (2026) as a ready-to-use log-LR
            (see :func:`sqrt_correction` / :func:`hapax_correction`). With a
            correction the result is always the 1-D corrected score, taken from the
            summed ``lambda_G`` even under ``direction="both"``.

        Returns
        -------
        float64[n_cases] of lambda_G values (or corrected log-LRs if ``correction``
        is set), or float64[n_cases, 2] of ``[lambda_forward, lambda_backward]``
        when ``direction="both"`` and no correction -- feed either straight into
        :class:`LambdaGCalibrator`, or (for a correction) straight into ``cllr``.
        """
        if correction not in (None, "sqrt", "hapax"):
            raise ValueError("correction must be None, 'sqrt' or 'hapax'.")
        it = range(len(cases))
        if progress:
            try:
                from tqdm.auto import tqdm

                it = tqdm(it, desc="LambdaG", unit="case")
            except ImportError:
                pass
        k = 2 if (self.direction == "both" and correction is None) else 1
        out = np.empty((len(cases), k), dtype=np.float64)
        for i in it:
            c = cases[i]
            excl = c[2] if len(c) > 2 else None
            res = self.score(c[0], c[1], exclude_groups=excl, with_details=False, **kwargs)
            if correction == "sqrt":
                out[i, 0] = res.lambda_sqrt
            elif correction == "hapax":
                out[i, 0] = res.lambda_hapax
            else:
                out[i] = res.features
        return out[:, 0] if k == 1 else out


# --------------------------------------------------------------------------- #
# 8. Calibration + the Likelihood Ratio framework (Sec. 6.4)                   #
# --------------------------------------------------------------------------- #


class LambdaGCalibrator:
    """
    Logistic-regression calibration of lambda_G into Lambda_G, a forensic log-LR.

    lambda_G is an *uncalibrated* score (Table 2: Cllr(lambda_G) up to 18 vs
    Cllr(Lambda_G) < 1). Fitting a 1-D logistic regression on training Y/N cases
    yields a well-calibrated log-likelihood ratio. The empirical training prior is
    divided out so the output is a likelihood ratio and not a posterior odds --
    with balanced training data (as in the paper) that term is 0 anyway.

    Decision rule (Sec. 3): classify Y if Lambda_G > 0, i.e. sigmoid(logit) > 0.5.
    """

    def __init__(
        self, log_base: float = 10.0, remove_prior: bool = True, C: float = 1e12
    ) -> None:
        self.log_base = float(log_base)
        self.remove_prior = bool(remove_prior)
        self.C = float(C)  # lower this if a small/imbalanced training set overfits
        self.coef_: np.ndarray = np.zeros(1)
        self.intercept_: float = 0.0
        self._log_prior_odds: float = 0.0

    @staticmethod
    def _as_2d(scores) -> np.ndarray:
        x = np.asarray(scores, dtype=np.float64)
        return x.reshape(-1, 1) if x.ndim == 1 else x

    def fit(self, lambda_scores: np.ndarray, y: np.ndarray) -> "LambdaGCalibrator":
        """
        ``y``: 1 / True for same-author (Y) cases, 0 / False for different-author (N).

        ``lambda_scores`` may be 1-D (one score per case) or 2-D (n_cases, n_scores),
        in which case this performs **score fusion**: the regression learns how to
        weight several scores into a single calibrated log-LR. That is the standard
        forensic recipe (Brummer & du Preez 2006), and is how to combine
        ``direction="both"``, or several engines, or LambdaG with another method.
        """
        from sklearn.linear_model import LogisticRegression

        x = self._as_2d(lambda_scores)
        y = np.asarray(y).astype(int).ravel()
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration needs both Y- and N-cases.")

        # An (effectively) unpenalised fit is what makes the output a calibrated LR
        # rather than a shrunken one. A huge C is used rather than `penalty=None`
        # (deprecated in sklearn 1.8) or `C=np.inf` (warns in 1.8): it is
        # version-agnostic, warning-free, and numerically identical.
        lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)
        lr.fit(x, y)
        self.coef_ = np.asarray(lr.coef_[0], dtype=np.float64)  # one weight per score
        self.intercept_ = float(lr.intercept_[0])
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        self._log_prior_odds = math.log(n_pos / n_neg) if self.remove_prior else 0.0
        return self

    def transform(self, lambda_scores: np.ndarray) -> np.ndarray:
        """lambda_G (or a fused score vector) -> Lambda_G (log LR in ``log_base``)."""
        x = self._as_2d(lambda_scores)
        llr_nat = x @ self.coef_ + self.intercept_ - self._log_prior_odds
        return llr_nat / math.log(self.log_base)

    def fit_transform(self, lambda_scores, y) -> np.ndarray:
        return self.fit(lambda_scores, y).transform(lambda_scores)

    def predict(self, lambda_scores) -> np.ndarray:
        """Y (True) iff Lambda_G > 0."""
        return self.transform(lambda_scores) > 0.0

    def __repr__(self) -> str:
        c = ", ".join(f"{v:.4g}" for v in np.atleast_1d(self.coef_))
        return (
            f"LambdaGCalibrator(coef=[{c}], intercept={self.intercept_:.4g}, "
            f"log_base={self.log_base})"
        )


# --------------------------------------------------------------------------- #
# Model-free normalisation calibrations (Barlow, Nini & Manino 2026,           #
# arXiv:2607.09501). These turn a raw lambda_G into a log-LR *without* fitting  #
# LambdaGCalibrator -- useful when no labelled calibration set is available.    #
#                                                                              #
# The paper's key empirical observation (Sec. 5.1) is that lambda_G is already  #
# sign-aligned with the ground truth (True > 0, False < 0) but *inflated* in    #
# magnitude, because summing token-level contributions under the naive-Bayes    #
# independence assumption double-counts the redundant evidence of repeated      #
# patterns. Both corrections only *scale* lambda_G by a per-case factor derived #
# from the masked questioned document, so the decision boundary stays at 0 and  #
# the ranking is preserved; only the over-stated strength is damped.            #
#                                                                              #
# Reported result: averaged over 15 corpora the Hapax Correction beats logistic #
# regression calibration (by Cllr) in ~45% of tests, and ties it in most of the #
# rest -- competitive with a trained calibrator while needing no training data. #
# --------------------------------------------------------------------------- #


def sqrt_correction(
    lambda_G: Union[float, np.ndarray], n_query_tokens: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Square Root Correction (arXiv:2607.09501, Eq. 7): ``lambda_G / sqrt(N(Q))``.

    ``N(Q)`` is the number of *masked* tokens in the questioned document D_U (the
    actual model input). The paper's rationale (Sec. 5.2): dispersion of a sum of
    weakly-dependent per-token contributions grows like ``sqrt(N)`` (the same
    scaling as attention's ``1/sqrt(d_k)``), so dividing by ``sqrt(N(Q))`` removes
    the length-driven inflation without over-penalising long texts. It addresses
    text length only, not repetition -- see :func:`hapax_correction` for the latter.

    Accepts scalars or NumPy arrays (broadcast element-wise); ``N(Q) == 0`` -> 0.0.
    Returns a log-LR in the *same base* as ``lambda_G`` (base-10 by default here).
    """
    lam = np.asarray(lambda_G, dtype=np.float64)
    n = np.asarray(n_query_tokens, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n > 0, lam / np.sqrt(n), 0.0)
    return float(out) if out.ndim == 0 else out


def hapax_correction(
    lambda_G: Union[float, np.ndarray],
    n_query_tokens: Union[int, np.ndarray],
    n_query_hapax: Union[int, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Hapax Correction (arXiv:2607.09501, Eq. 8): ``lambda_G * V1(Q) / N(Q)``.

    ``V1(Q)`` is the number of hapax legomena -- types occurring exactly once -- in
    the *masked* questioned document, and ``N(Q)`` its masked token count. The ratio
    ``V1(Q)/N(Q)`` in [0, 1] is a measure of lexical productivity / diversity (Sec.
    5.4): a text that repeats itself has few hapaxes and gets damped hard, whereas a
    stylistically varied text (ratio near 1) is left almost unchanged. Unlike the
    square-root scaling this explicitly discounts *redundant* repeated evidence,
    which also makes it more robust to adversarial feature repetition (Sec. 5.3).

    Both counts are on the masked text so that normalisation operates over exactly
    the tokens that drove the score (Sec. 5.4). Accepts scalars or arrays; ``N(Q)
    == 0`` -> 0.0. Returns a log-LR in the same base as ``lambda_G``.
    """
    lam = np.asarray(lambda_G, dtype=np.float64)
    n = np.asarray(n_query_tokens, dtype=np.float64)
    v1 = np.asarray(n_query_hapax, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n > 0, lam * v1 / n, 0.0)
    return float(out) if out.ndim == 0 else out


def _to_natural_llr(llr: np.ndarray, base: float) -> np.ndarray:
    return np.asarray(llr, dtype=np.float64).ravel() * math.log(base)


def cllr(llr_same: np.ndarray, llr_diff: np.ndarray, base: float = 10.0) -> float:
    """
    Log-likelihood-ratio cost, Eq. (22).

        Cllr = 0.5 * [ mean_Y log2(1 + 1/LR) + mean_N log2(1 + LR) ]

    0 is perfect; ~1 means the system is uninformative; > 1 means it is misleading.
    Computed via ``logaddexp`` so that infinite LRs (PAV blocks) are handled exactly.
    """
    a = _to_natural_llr(llr_same, base)
    b = _to_natural_llr(llr_diff, base)
    ln2 = math.log(2.0)
    t1 = np.logaddexp(0.0, -a).mean() / ln2 if a.size else 0.0
    t2 = np.logaddexp(0.0, b).mean() / ln2 if b.size else 0.0
    return float(0.5 * (t1 + t2))


def cllr_min(scores_same: np.ndarray, scores_diff: np.ndarray) -> float:
    """
    Cllr^min: the discrimination loss, i.e. Cllr after PAV-optimal recalibration.

    Any monotonic score works as input (lambda_G and Lambda_G give the same value,
    as Table 2 of the paper notes). Cllr - Cllr^min = Cllr^cal, the calibration loss.
    """
    from sklearn.isotonic import IsotonicRegression

    a = np.asarray(scores_same, dtype=np.float64).ravel()
    b = np.asarray(scores_diff, dtype=np.float64).ravel()
    x = np.concatenate([a, b])
    y = np.concatenate([np.ones(a.size), np.zeros(b.size)])

    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    p = ir.fit_transform(x, y)

    log_prior_odds = math.log(a.size / b.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        llr_nat = np.log(p) - np.log1p(-p) - log_prior_odds
    llr_nat = np.nan_to_num(llr_nat, nan=0.0, posinf=np.inf, neginf=-np.inf)
    return cllr(llr_nat[: a.size] / math.log(10.0), llr_nat[a.size :] / math.log(10.0), base=10.0)


def cllr_decomposition(llr_same, llr_diff, base: float = 10.0) -> Dict[str, float]:
    """``{"cllr", "cllr_min", "cllr_cal"}`` -- total = discrimination + calibration loss."""
    total = cllr(llr_same, llr_diff, base=base)
    cmin = cllr_min(np.asarray(llr_same).ravel(), np.asarray(llr_diff).ravel())
    return {"cllr": total, "cllr_min": cmin, "cllr_cal": total - cmin}


# --------------------------------------------------------------------------- #
# 9. Explainability (Sec. 7)                                                   #
# --------------------------------------------------------------------------- #


def heatmap_html(
    result: LambdaGResult,
    max_sentences: Optional[int] = None,
    order_by_lambda: bool = True,
    title: str = "LambdaG text heat map",
) -> str:
    """
    Colour-coded rendering of D_U: red = entrenched for A, blue = typical of D_ref.

    Mirrors ``idiolect::lambdaG_visualize``. Drop the string into
    ``IPython.display.HTML(...)`` inside a notebook.
    """
    idx = list(range(len(result.tokens)))
    if order_by_lambda and len(result.sentence_lambda):
        idx = list(np.argsort(result.sentence_lambda)[::-1])
    if max_sentences is not None:
        idx = idx[:max_sentences]

    flat = np.concatenate(result.token_lambda) if result.token_lambda else np.zeros(1)
    scale = float(np.percentile(np.abs(flat), 98)) or 1.0

    def _colour(v: float) -> str:
        x = max(-1.0, min(1.0, v / scale))
        if x >= 0:
            return f"rgba(214,39,40,{0.08 + 0.62 * x:.3f})"
        return f"rgba(31,119,180,{0.08 + 0.62 * -x:.3f})"

    parts = [
        "<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        "font-size:13px;line-height:2.1'>",
        f"<h3 style='font-family:system-ui,sans-serif'>{_html.escape(title)}</h3>",
        f"<p style='font-family:system-ui,sans-serif;color:#555'>"
        f"&lambda;<sub>G</sub> = <b>{result.lambda_G:.3f}</b> "
        f"(log<sub>{result.log_base:g}</sub>, r = {result.n_ref_models}) &nbsp;|&nbsp; "
        "<span style='background:rgba(214,39,40,.55);padding:1px 6px'>entrenched for A</span> "
        "<span style='background:rgba(31,119,180,.55);padding:1px 6px'>typical of the reference"
        "</span></p>",
    ]
    for i in idx:
        parts.append(
            f"<div style='margin:.5em 0'><span style='color:#999;font-size:11px'>"
            f"[S{i}: {result.sentence_lambda[i]:+.2f}]</span> "
        )
        for tok, lam in zip(result.tokens[i], result.token_lambda[i]):
            parts.append(
                f"<span title='{lam:+.3f}' style='background:{_colour(float(lam))};"
                f"padding:1px 3px;border-radius:3px;margin-right:2px'>"
                f"{_html.escape(tok)}</span>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def warmup_jit() -> None:
    """
    Trigger Numba compilation on a toy corpus.

    Worth calling once at the top of a notebook so the first real case is not
    dominated by ~2-4 s of JIT time.
    """
    v = Vocabulary()
    sents = [["the", "#", "\u00d8", "the", "#", "."], ["I", "do", "n't", "\u00d8", "@", "."]]
    store = v.encode(sents)
    for engine, kw in (
        ("kn", {}),
        ("hpy", {"table_estimator": "expected"}),
        ("hpy", {"table_estimator": "minimal"}),
        ("ppmd", {"exclusion": True}),
        ("ppmd", {"exclusion": False}),
        ("ppmd", {"adaptive": True, "reset": "sentence"}),
        ("ppmd", {"adaptive": True, "reset": "document"}),
    ):
        ENGINES[engine].fit(store, N=3, vocab_size=len(v), **kw).token_logprobs(store)


# =========================================================================== #
#                              USAGE EXAMPLES                                 #
# =========================================================================== #
#
# ---------------------------------------------------------------------------
# Example 0 -- setup
# ---------------------------------------------------------------------------
# from lambdag import (LambdaG, POSNoiseMasker, LambdaGCalibrator,
#                      cllr, cllr_min, cllr_decomposition, heatmap_html, warmup_jit)
#
# warmup_jit()                       # ~3 s of Numba compilation, once per session
#
# masker = POSNoiseMasker(language="en", spacy_model="en_core_web_lg")
# # en_core_web_lg is the POSNoise default; en_core_web_sm is ~10x lighter and,
# # per Table 4 of the paper, the POS labels barely matter anyway.
#
#
# ---------------------------------------------------------------------------
# Example 1 -- what POSNoise actually produces
# ---------------------------------------------------------------------------
# masker.mask("I don't understand why they destroy stability and make trouble "
#             "for ordinary people.")
# # -> [['I', 'do', "n't", 'Ø', 'why', 'they', 'Ø', '#', 'and', 'make', '#',
# #      'for', '@', '#', '.']]                       # cf. Fig. 5 of the paper
#
# masker.mask_to_string("Our meal started with a basket of freshly baked bread.")
# # -> 'Our # started with a # of © Ø #.'             # cf. Table 7
#
#
# ---------------------------------------------------------------------------
# Example 2 -- AV Core: one verification case, c = (D_U, D_A)
# ---------------------------------------------------------------------------
# D_U   = "..."                       # the questioned document
# D_A   = "..."                       # the known document(s) of candidate A
# D_ref = ["...", "...", "..."]       # the reference population (genre-matched!)
#
# S_U   = masker.mask(D_U)
# S_A   = masker.mask(D_A)
# S_ref = [s for doc in masker.mask_batch(D_ref, progress=True) for s in doc]
#
# lg  = LambdaG(N=10, r=30, random_state=42)      # paper: N=10; Fig. 3: r=30 is enough
# lg.set_reference(S_ref)
# res = lg.score(S_U, S_A)
# print(res.lambda_G)                 # uncalibrated log10 LR: > 0 favours A
#
#
# ---------------------------------------------------------------------------
# Example 3 -- AV Known / AV Batch: many documents per author, and a reference
#              pool that must exclude the candidate and the Q author
# ---------------------------------------------------------------------------
# corpus = {"alice": ["doc1 ...", "doc2 ..."], "bob": [...], "carol": [...]}
#
# masked, groups, sents = {}, [], []
# for author, docs in corpus.items():
#     per_author = [s for d in masker.mask_batch(docs) for s in d]
#     masked[author] = per_author
#     sents  += per_author
#     groups += [author] * len(per_author)
#
# lg = LambdaG(N=10, r=30, random_state=42)
# lg.set_reference(sents, groups=groups)          # groups -> per-sentence author labels
#
# res = lg.score(
#     q_sentences=masked["alice"][:20],           # S_U
#     k_sentences=masked["bob"],                  # S_A  (candidate = bob)
#     exclude_groups=["bob", "alice"],            # keep the candidate + Q author
# )                                               # out of the reference pool
# print(res.lambda_G)
#
#
# ---------------------------------------------------------------------------
# Example 4 -- calibration into Lambda_G, plus the forensic metrics
# ---------------------------------------------------------------------------
# train_cases = [(S_U1, S_A1), (S_U2, S_A2), ...]        # author-disjoint from test
# y_train     = np.array([1, 0, ...])                    # 1 = Y-case, 0 = N-case
#
# lam_train = lg.score_many(train_cases)
# lam_test  = lg.score_many(test_cases)
#
# cal = LambdaGCalibrator(log_base=10).fit(lam_train, y_train)
# LLR = cal.transform(lam_test)          # Lambda_G: a calibrated log10 LR
# pred = LLR > 0                         # Sec. 3 decision rule
#
# from sklearn.metrics import accuracy_score, roc_auc_score
# print("Acc", accuracy_score(y_test, pred), "AUC", roc_auc_score(y_test, LLR))
# print(cllr_decomposition(LLR[y_test == 1], LLR[y_test == 0]))
# # -> {'cllr': ..., 'cllr_min': ..., 'cllr_cal': ...}   cf. Table 2
# # Cllr ~ 0 is ideal, ~1 is uninformative, > 1 is misleading.
# # Cllr_min is identical for lambda_G and Lambda_G (it is calibration-invariant):
# print(cllr_min(lam_test[y_test == 1], lam_test[y_test == 0]))
#
#
# ---------------------------------------------------------------------------
# Example 5 -- explainability: the text heat map of Sec. 7
# ---------------------------------------------------------------------------
# res = lg.score(S_U, S_A, with_details=True)
#
# for idx, val, toks, lams in res.top_sentences(k=5):
#     print(f"[{val:+.2f}] {' '.join(toks)}")
# # red-hot sequences are the "seemingly unremarkable" but idiosyncratic ones,
# # e.g. 'enough to be © Ø' or 'so PRONOUN cant'
#
# from IPython.display import HTML, display
# display(HTML(heatmap_html(res, max_sentences=15)))
#
# import pandas as pd
# df = pd.DataFrame(res.token_table(), columns=["sent", "token", "lambda"])
# df.groupby("token")["lambda"].agg(["sum", "mean", "count"]).sort_values("sum")
#
#
# ---------------------------------------------------------------------------
# Example 5b -- swapping the probability engine
# ---------------------------------------------------------------------------
# All three take the same POSNoise input and return a comparable lambda_G, so the
# only thing that changes is how P(t_k | t_<k; G) is estimated. Cost is ~the same
# (~120 ms/case at N=10, r=30, |D_A| ~ 800 tokens, one core).
#
# lg = LambdaG(N=10, r=30, engine="kn", random_state=0)             # the paper
#
# lg = LambdaG(N=10, r=30, engine="hpy", random_state=0,            # Teh (2006)
#              engine_params=dict(discount=0.75, concentration=0.0,
#                                 table_estimator="expected"))
#
# lg = LambdaG(N=10, r=30, engine="ppmd", random_state=0,           # the COAV engine
#              engine_params=dict(escape="d", discount=0.5, exclusion=True))
#
# # PPMd as a real (adaptive) compressor rather than a frozen model. This computes
# # the Bayesian predictive P(D_U | D_A) instead of the plug-in P(D_U | theta-hat),
# # and makes G.code_length_bits(S_U) equal C(D_U | D_A) exactly -- the conditional
# # compression that COAV (arXiv:1706.00516) had to approximate as C(xy) - C(y),
# # because off-the-shelf compressors take no auxiliary input.
# lg = LambdaG(N=10, r=30, engine="ppmd", random_state=0,
#              engine_params=dict(adaptive=True, reset="sentence"))
#
# # KEEP reset="sentence". Adaptation is asymmetric inside a ratio: a reference model
# # G_j that has never seen a pattern gains far more from each fresh observation of
# # it than G_A does, so a long D_U lets the denominator learn from the very document
# # under test. Measured on synthetic data (see the repo tests), per-sentence
# # lambda_G across the four quartiles of D_U:
# #     reset="sentence":  +1.31  +1.30  +1.44  +1.44     (flat)
# #     reset="document":  +0.76  +0.47  +0.44  +0.13     (collapses to 17%)
# # and the Y/N gap fell from 143 (static) to 130 (sentence) to 41 (document).
# # reset="document" is faithful to C(D_A D_U) - C(D_A), but it also violates the
# # sentence-independence of Eq. (13). Use it only to reproduce compression numbers.
#
# # A per-level PY schedule, closer to what Teh actually fits (d rises with order):
# lg = LambdaG(N=10, engine="hpy",
#              engine_params=dict(discount=np.linspace(0.5, 0.9, 10),
#                                 concentration=np.linspace(0.0, 2.0, 10)))
#
# # Sanity check you can run yourself -- Teh (2006) proved interpolated KN is the
# # HPY predictive distribution at theta=0 with minimal tables, and these agree to
# # ~1e-12 token-by-token AND end-to-end through lambda_G:
# a = LambdaG(N=10, r=30, engine="kn",  random_state=7).set_reference(S_ref)
# b = LambdaG(N=10, r=30, engine="hpy", random_state=7,
#             engine_params=dict(discount=0.75, concentration=0.0,
#                                table_estimator="minimal")).set_reference(S_ref)
# assert abs(a.score(S_U, S_A).lambda_G - b.score(S_U, S_A).lambda_G) < 1e-9
#
# # Which engine wins is an empirical question the paper does not answer -- decide it
# # on YOUR data, with Cllr_min (calibration-free, so no need to fit the calibrator):
# for eng, kw in [("kn", {}), ("hpy", {}), ("ppmd", {})]:
#     lam = LambdaG(N=10, r=30, engine=eng, engine_params=kw,
#                   random_state=0).set_reference(S_ref).score_many(test_cases)
#     print(eng, cllr_min(lam[y_test == 1], lam[y_test == 0]))
#
# # Engines are usable standalone too (they share BaseGrammarModel):
# from lambdag import ENGINES, HPYGrammarModel, PPMGrammarModel
# G = ENGINES["ppmd"].fit(v.encode(S_A), N=10, escape="d")
# G.perplexity(v.encode(S_U, grow=False))
#
# # The COAV bridge: code lengths in bits, for any engine.
# G_static = PPMGrammarModel.fit(v.encode(S_A), N=10)
# G_adapt  = PPMGrammarModel.fit(v.encode(S_A), N=10, adaptive=True, reset="sentence")
# G_static.code_length_bits(S_U_store)   # -log2 P(D_U ; theta-hat(D_A))
# G_adapt.code_length_bits(S_U_store)    # C(D_U | D_A), exactly
#
#
# ---------------------------------------------------------------------------
# Example 5c -- reading sentences backwards, and score fusion
# ---------------------------------------------------------------------------
# lg = LambdaG(N=10, r=30, direction="backward", random_state=0)   # P(t_k | t_>k)
# lg = LambdaG(N=10, r=30, direction="both",     random_state=0)   # 2x cost
#
# res = lg.set_reference(S_ref).score(S_U, S_A)
# res.lambda_forward, res.lambda_backward
# res.features            # -> [lambda_f, lambda_b], for the calibrator to FUSE
# res.lambda_G            # their plain sum: convenient, but a product-of-experts
#                         # score missing a per-case log Z(G_A)/Z(G_j). Prefer fusion.
#
# X = lg.score_many(train_cases)            # (n, 2) when direction="both"
# cal = LambdaGCalibrator().fit(X, y_train) # multi-feature -> score fusion
# LLR = cal.transform(lg.score_many(test_cases))
#
# BEFORE YOU SPEND 2x THE COMPUTE ON THIS, know what the measurement says:
# forward and backward are not two views of the data, they are the *same n-gram
# counts re-factorised*. For an MLE bigram,
#     P_fwd(S) = prod c(t_{k-1} t_k) / c(t_{k-1} .)
#     P_bwd(S) = prod c(t_k t_{k+1}) / c(. t_{k+1})
# share their numerators, and c(t .) = c(. t) for every t up to boundary terms, so
# the two telescope to the same joint. The same holds at any order: both reduce to
# prod(N-grams) / prod((N-1)-grams). Only the padding and KN's direction-dependent
# continuation counts survive. Measured correlation of lambda_f with lambda_b:
#     N   =  1        2        3        5       10       20
#     corr=  1.00000  1.00000  0.99993  0.99972  0.99929  0.99912
# i.e. essentially nothing to fuse. Test it on your own data with cllr_min before
# believing otherwise -- and note that a backward model has no entrenchment story
# behind it (Sec. 2 is about procedural memory for *production*), which costs you
# the scientific-plausibility argument that is LambdaG's main claim (Sec. 5).
#
# Where "both" may genuinely pay off is EXPLAINABILITY, not the score. The forward
# chain rule already uses the whole sentence, so lambda_G loses nothing; but a
# *single token's* forward lambda is lop-sided (early tokens have almost no
# context). res.token_lambda under direction="both" is the sum of both directions
# per token, i.e. a symmetric attribution for the Sec. 7 heat maps.
#
# Fusion is not limited to directions -- it is the standard forensic recipe for
# combining anything (Brummer & du Preez 2006):
# X = np.column_stack([
#     LambdaG(engine="kn"  ).set_reference(S_ref).score_many(cases),
#     LambdaG(engine="ppmd").set_reference(S_ref).score_many(cases),
# ])
# cal = LambdaGCalibrator().fit(X, y)
#
# NOT provided: conditioning a token on BOTH sides at once, prod_k P(t_k|t_<k,t_>k).
# That is the pseudo-likelihood (Besag 1975) -- it is not a joint over sentences and
# does not normalise, so it forfeits the likelihood-ratio reading that Sec. 6.4 is
# built on, and each token gets explained by neighbours that are themselves being
# predicted. It has precedent (masked-LM scoring, Salazar et al. 2020), but note
# what the naive-Bayes form of it reduces to:
#     P(t|L,R) ~ P_f(t|L) P_b(t|R) / P(t)   =>   lambda_bi ~ lambda_f + lambda_b - lambda_unigram
# which is direction="both" plus a unigram correction. Given corr ~ 0.999 above,
# it is a much more expensive route (normalising over T* at every position) to
# almost exactly the same number.
#
#
# ---------------------------------------------------------------------------
# Example 6 -- the ablations of Tables 3 and 4
# ---------------------------------------------------------------------------
# POSNoiseMasker(mode="posnoise")   # Algorithm 1 as published
# POSNoiseMasker(mode="star")       # Table 4: mask with '*', drop the POS labels
# POSNoiseMasker(mode="none")       # Table 3: no masking at all (always worse)
#
#
# ---------------------------------------------------------------------------
# Example 6b -- fixed-length windows instead of sentences as the unit
# ---------------------------------------------------------------------------
# The unit of independence in lambda_G (Eq. 13) is the sentence by default. For
# verse / unpunctuated corpora (gmh/gml), or just to make the unit a fixed "text
# snippet" of N words, tile the masked stream into windows instead:
#
# mw = POSNoiseMasker(language="de", segment="window", window=20)
# S_U = mw.mask(D_U)                # -> list of <=20-token windows, not sentences
#
# # ... everything downstream is unchanged; a window IS a "sentence" to the store:
# lg = LambdaG(N=10, r=30).set_reference([s for d in mw.mask_batch(D_ref) for s in d])
# lg.score(S_U, mw.mask(D_A)).lambda_G
#
# # Or re-window an already-masked corpus without re-running spaCy (same result):
# from lambdag import windowize
# S_U = windowize(masker.mask(D_U), 20)
#
# # Windows never cross a document boundary; sweep the size like any other knob:
# for w in (10, 20, 50):
#     mw = POSNoiseMasker(language="de", segment="window", window=w)
#     ...                            # pick w by cllr_min on held-out data
#
# # Hyper-parameter sweep of Fig. 3:
# for N in (1, 2, 3, 4, 8, 10, 20):
#     for r in (1, 30, 50, 100):
#         lg = LambdaG(N=N, r=r, random_state=0).set_reference(S_ref)
#         ...
#
#
# ---------------------------------------------------------------------------
# Example 6c -- already-tagged historical corpora (ReM / ReN), no spaCy
# ---------------------------------------------------------------------------
# ReM (Middle High German, gmh) and ReN (Middle Low German, gml) ship their own
# HiTS POS tags and lemmas, and there is no spaCy model for gml at all. Build the
# masker WITHOUT a tagger and feed it (surface, pos, lemma) tuples per sentence:
#
# m = POSNoiseMasker.pretagged("gml")          # no spaCy; HiTS->UD via the tag_map
# doc = [ [("De","DDART","de"), ("man","NA","man"), ("gink","VVFIN","gan"), ...],
#         [("unde","KON","unde"), ("he","PPER","he"), ...] ]   # one doc = list of sents
# S = m.mask_tagged(doc)                        # -> masked List[List[str]]
#
# # historical punctuation is unreliable, so sentence mode trusts the CORPUS's own
# # segmentation; window mode ignores it (often the better unit for verse):
# mw = POSNoiseMasker.pretagged("gmh", segment="window", window=20)
#
# # corpus-scale + straight into LambdaG (same pipeline as every other language):
# ref = [s for d in m.mask_tagged_batch(docs) for s in d]
# lg  = LambdaG(N=10, r=30).set_reference(ref)
# # NB: mask()/mask_batch() raise here -- there is no tagger to run.
#
#
# ---------------------------------------------------------------------------
# Example 7 -- the grammar models on their own
# ---------------------------------------------------------------------------
# from lambdag import KNGrammarModel, Vocabulary
#
# v  = Vocabulary()
# GA = KNGrammarModel.fit(v.encode(S_A), N=10, D=0.75, vocab_size=len(v))
# GA.perplexity(v.encode(S_U, grow=False))     # Sec. 5: lambda_G ~ comparing
#                                              # perplexities of G_A vs G_ref on D_U
# GA.sentence_logprobs(v.encode(S_U, grow=False))   # Eq. (1), natural log
#
#
# ---------------------------------------------------------------------------
# Example 8 -- notes on reproducing the paper
# ---------------------------------------------------------------------------
# * LambdaG is stochastic (Sec. 5.1). The paper averages five repetitions;
#   do the same by re-running with different `random_state` values.
# * `vocab_mode="per_model"` (default) reproduces idiolect::lambdaG, where every
#   model has its own dictionary. `vocab_mode="shared"` makes |T| identical across
#   the numerator and denominator models -- arguably cleaner, slightly different
#   scores.
# * Cost is ~0.14 s per case at N=10, r=30, |D_A| ~ 800 tokens on one core.
#   The r loop is embarrassingly parallel -- wrap `score_many` in joblib if needed.
# * Genre matters most: keep D_ref as close to the case data as you can
#   (Fig. 4), though LambdaG degrades far more gracefully than the Impostors Method.
