# POSNoise safe-pattern lists: fr, es, it, pl, ru

New lists for French, Spanish, Italian, Polish and Russian, following Table 6 of
Nini et al. (2025) / Halvani & Graner (2021). English and German still come from
the `posnoise` package unchanged.

| lang | entries | multiword | match_on | source of the auxiliary system |
|------|--------:|----------:|----------|--------------------------------|
| en   | 1034 | 322 | `text`  | shipped (Halvani & Graner 2021) |
| de   | 2002 | –   | `text`  | shipped |
| fr   |  400 | 122 | `both`  | UD_French-GSD |
| es   |  448 | 166 | `both`  | UD_Spanish-AnCora |
| it   |  383 |  49 | `both`  | UD_Italian-ISDT |
| pl   |  708 | 201 | `both`  | UD_Polish-PDB |
| ru   |  839 | 142 | `both`  | UD_Russian-SynTagRus |

## Two design decisions worth understanding

**1. The list has a much narrower job than it looks.** POSNoise only masks
`{ADJ, ADV, AUX, NOUN, NUM, PROPN, SYM, VERB, X}`. Determiners, prepositions,
pronouns, conjunctions, particles and punctuation are *kept automatically by the
tagger* — roughly half the English list is redundant belt-and-braces. What a list
genuinely has to rescue is: **auxiliaries, modals, light/delexical verbs, function
adverbs, quantifiers, and multiword functional units.**

**2. Entries are lemmas, matched with `match_on="both"`.** Italian has **193 AUX
surface forms but only 10 AUX lemmas**; Russian, 22 vs 2. Enumerating surface forms
is why the German list is 2002 lines. Matching on surface-form-OR-lemma and
*emitting the surface form* collapses that: inflection survives untouched in the
output (a whitelisted token is copied verbatim), only the lookup is lemmatised.
The paper anticipates this (Sec. 5.1: *"The way Grammar Models are trained in
languages with a rich morphology is likely to be different ... different algorithms
such as POSNoise"*).

## Provenance — what is measured vs. what is my judgement

Rebuild everything with `tools/ud_extract.py`, `tools/ud_mine_multiword.py`, then
`tools/build_posnoise_lists.py`.

**Empirical (from UD treebanks, not recalled):**
- the complete attested **AUX inventory** per language;
- frequent **closed-class types** (DET/ADP/PRON/CCONJ/SCONJ/PART);
- **multiword functional units**, mined from the UD `fixed` dependency relation,
  filtered to those whose head carries a functional deprel (`case`/`mark`/`cc`/
  `advmod`/…), frequency ≥ 3. This is the category I trust most and could least
  have written from memory: `sin embargo`, `потому что`, `przede wszystkim`,
  `tem не менее`, `ainsi que`, `anche se`.

**My linguistic judgement (needs review):** the category sets in
`build_posnoise_lists.py` — modals, light/delexical verbs, and the split of
adverbs into function (degree/frequency/place/time/focusing/conjunctive/
pronominal → **keep**) vs manner (`-ment`/`-mente`/`-nie`/`-о` → **mask**).

**Every curated entry is validated against UD**: anything not attested at all, or
dominantly NOUN/PROPN, is dropped and *reported*. That check earns its keep — it
caught FR `point` (NOUN ×174, since *ne…point* is archaic) and PL `czasem`/`czasami`
("sometimes", but also the instrumental of *czas* "time", NOUN ×48).

## Calibration against the published English list

Reverse-engineered from `POSNoise_PatternList_En_v2.1.txt` and mirrored:

- **IN**: auxiliaries/modals/light verbs *in every form*; degree, frequency, place,
  time, focusing, conjunctive and pronominal adverbs; quantifiers; transitions.
- **OUT**: manner adverbs (`slowly`, `carefully`, `badly`); calendar deictics
  (`today`, `yesterday`, `tomorrow` — though `now`/`then`/`already`/`still` are in);
  content words.

Note the English list is itself hand-curated and imperfect — `quickly` and `well`
are in while `slowly` is out; `less` is in but `least` is out; `work` and `think`
are in. These lists do not need to be flawless, but they must be *reviewable*.

## Limits — read before forensic use

1. **These have not been reviewed by native speakers.** I am most confident in
   fr/es/it, least in pl/ru. The multiword and AUX portions are UD-derived and
   solid; the adverb function/manner split is the part most likely to contain
   errors. **Have a native linguist audit each category before any casework.**
2. **They are not Halvani's lists.** en/de were built over two published iterations;
   these are v1.0 and unvalidated on any AV corpus. Expect them to be worse.
3. **Tagger quality now matters more.** With `sm` models we see `capisco`→ADJ,
   `niszczą`→ADJ. Use `lg` (the POSNoise default) or `trf`.
4. **No AV evaluation exists for these languages.** The paper's twelve corpora are
   all English; Sec. 5.1 calls other languages an open question. Nothing here shows
   LambdaG *works* in fr/es/it/pl/ru — only that the preprocessing runs.
5. **Zero-copula languages differ structurally.** Russian has 2 AUX lemmas (быть, бы)
   against Italian's 10, because the present-tense copula is null. The Russian
   grammar model therefore sees a systematically different token stream. Whether
   POSNoise's category inventory even transfers is untested.

## Baseline you should run first

Before trusting a hand-made list, run the language-independent control:
`POSNoiseMasker(mode="star")` masks by POS only. Table 4 of the paper found POS
labels contribute almost nothing over asterisks, and Sec. 5.1 notes this is
*"encouraging for the replicability of this algorithm for languages with less
computational resources."* If `mode="star"` matches your curated list on a
held-out corpus, the curation is not paying for itself.

---

# Historical stages: Middle High German (gmh) and Middle Low German (gml)

| lang | entries | source corpus | spaCy model | status |
|------|--------:|---------------|-------------|--------|
| gmh  | 1809 → **2784** (v0.2) | **ReM** v1.0 (1050–1350), via `cltk/gmh_models_cltk`; v0.2 lemma-augmented over ReM v2.1 (2.15 M tokens) | ReM-trained *GMH-Tagger*, MIT | runs end-to-end |
| gml  | 2981 → **3323** (v0.2) | **ReN** v0.6 (1200–1650), via `cltk/gml_models_cltk`; v0.2 lemma-augmented over ReN v1.1 (1.31 M tokens) | none — but the **pre-tagged path** (`POSNoiseMasker.pretagged("gml")`) needs no spaCy model | runs end-to-end (pre-tagged) |

## Read this first: whose grammar are you measuring?

LambdaG's premise is that grammar is a **behavioural biometric** — entrenchment in
an individual's procedural memory (Sec. 2). Medieval vernacular texts do not reach
us from their authors. They reach us through **scribal copies**, and scribes
routinely re-spelled, and often re-dialected, what they copied. ReM and ReN are
diplomatic transcriptions *of manuscripts*.

So the orthography and much of the morphology in these corpora is the **scribe's**,
not the author's — and POSNoise deliberately keeps function words *with their
surface spelling*, which is precisely the layer the scribe overwrote. Run naively,
λG on MHG/MLG is closer to a **scribal-hand biometric than an authorial one**.

That may be exactly what you want (localising a scriptorium, grouping hands,
detecting copy layers — all real philological problems, and λG's per-token heat maps
suit them). But it is not the same claim as "author A wrote D_U", and the paper's
cognitive-linguistic justification does not straightforwardly transfer. Decide which
question you are asking before trusting a number.

Two consequences:
- **ReM ships normalised forms** alongside diplomatic ones. Normalising removes
  scribal orthography — which either removes the confound or removes the signal,
  depending on your question. The lists here are built on **diplomatic** forms.
- **Reference-corpus matching gets much harder.** Fig. 4 shows λG degrades across
  genre; here you must match genre *and* period *and* dialect *and* scribal
  tradition, from ~2.5M tokens spanning three centuries.

## Why these lists could be built empirically after all

Your three links are unreachable from my sandbox (`host_not_allowed`), and ReM/ReN
require registration, so I could not process them directly. But both corpora reach
CLTK in derived form, which *is* reachable — so neither list is written from memory:

- **gmh**: `cltk/gmh_models_cltk` ships a ReM-derived token→HiTS inventory.
- **gml**: `cltk/gml_models_cltk` ships an NLTK backoff tagger trained on ReN v0.6;
  its `UnigramTagger._context_to_tag` *is* a ReN-derived token→tag inventory.

This matters more here than for any modern language, because **there is no
standard orthography**. ReM attests **19 spellings of the negation particle**
(`ne, niht, nieht, niet, niuht, niwet, niwiht, niuwet, niewet, nieuht, niuweht,
niut, niwit, …`); ReN attests ~15 of the preposition *an* (`an, ane, aen, ahn,
ahne, am, ame, amm, amme, …`) plus enclitic fusions like `hestu` (hest+du) and
`datck` (dat+ick) — Table 6's "Contractions", and stylistically gold. Any list I
wrote from a modern-German intuition would have missed nearly all of them.

Construction (`tools/build_hist_posnoise_lists.py`):
1. **Empirical**: every attested spelling variant of every closed-class HiTS tag
   (articles, pronouns, prepositions, conjunctions, particles) plus the complete
   `VAFIN/VAINF/VAPP` auxiliary and `VMFIN/VMINF` modal inventories — taken
   wholesale from the corpus.
2. **Curated + validated**: function adverbs and light/delexical verbs, each
   checked against the corpus inventory. This caught two modern-German intrusions
   of mine — `immer` (MHG *iemer*) and `vielleicht` (MHG *vil lîhte*).
3. Latin code-switching (`cum`, `aut`, `adhuc` — HiTS `FM`) and editorial brackets
   are filtered out.

**gmh v0.2 — lemma augmentation (`build_gmh_lemma_augment.py`).** v0.1 matched
*surface* forms; when the masker emits lemmas (`emit="lemma"`) or meets a scribal
spelling whose particular surface is unlisted, function words got masked because a
few ReM *lemmas* differ from the normalised list entries (`wërden` vs `werden`).
v0.2 closes this **from the corpus, not from memory**: over the full 2.15 M-token
ReM corpus it adds a lemma L iff some token with lemma L has an *already-whitelisted
surface* and that token's POS is a function class — `ADV`/`AUX`, or `VERB` whose
lemma also occurs as `AUX` (light verbs *wërden, sîn, haben, müezen, soln, wellen*).
Four corpus-derived groups, all verb-free (any lemma embedding a known VERB component
is dropped, so periphrases/particle+verb constructions never leak content):

* **129 single lemmas** — the lemma of a token whose surface is already whitelisted,
  POS `ADV`/`AUX` or a light `VERB` (`wërden`, `sîn`, `müezen`, `soln`, `wellen`);
  content POS excluded, so `man` vs noun `mann` can't sneak in.
* **140 aux + enclitic-pronoun univerbations** — `sîn+dû`=`biſtu`, `haben+dû`=`haſtu`,
  `sol(e)n+dû`=`ſaltu`; head POS `AUX`, head lemma whitelisted, no embedded verb.
* **696 pronominal adverbs** — the closed `PAV*` class (`dâr/+zuo`=dazu, `dâr/+mit(e)`
  =damit, `nâh/dâr+`=danach, `wâr/+umbe`=warumbe, `hièr`, …), minus `++` multi-component
  and verb-embedding constructions.
* **11 curated function adverbs** — `gërne, lange, niène, vërre, schône, balde, vruo,
  lèider, übel(e), wær-lîche, êwig-lîche` (curator-approved, each validated as an ADV
  lemma in the corpus).

**+976 forms; ADV/AUX occurrences rescuable 66.7 % → 92.8 % (AUX 99.4 %, ADV 88.4 %);
v0.2 = 2784 entries.** Rerunning `build_gmh_lemma_augment.py` rebuilds v0.2 idempotently
from v0.1 (which `_find_pattern_list` auto-selects as the highest version). The residual
~12 % of ADV are hapax/manner adverbs left for hand review. To add more curated adverbs,
extend `CURATED_ADV` in the tool and rerun.

**gml v0.2 — lemma augmentation (`build_gml_lemma_augment.py`).** The Middle **Low**
German twin of the above, applying the identical construction over the full **ReN v1.1**
corpus (161 files, 1.31 M tokens) extracted by `rem_extract.ipynb`'s ReN cell into the
`surface⇥lemma⇥POS⇥morph` tab format. v0.1 was surface-only (2981 spellings); v0.2 adds
lemma forms so the masker can emit lemmas and rescue scribal spelling variants whose
particular surface is unlisted. Three corpus-derived groups (**+342 forms → 3323 entries**):

* **95 single lemmas** — the lemma of a token whose surface is already whitelisted, POS
  `ADV`/`AUX` or a light `VERB` whose lemma also occurs as `AUX` (`hebben`, `wēsen`,
  `wērden`, `schȫlen`, `willen`, `mȫgen`, `dôn`, …); content POS excluded.
* **125 aux + enclitic-pronoun univerbations** — `hebben+dû`=`hestu`, `schȫlen+dû`=`scaltu`,
  `künnen+dû`=`kanstu`, `wēsen+dû`=`bistu` (ReN's "Contractions", Table 6, stylistically
  gold).
* **122 pronominal adverbs** — the closed `PAV*` class (`dâr+ümme`=darum, `dâr+nâ`=danach,
  `dâr+in`, `hîr`, `wôr`, …), minus `++` multi-component and verb-embedding constructions
  (73 dropped).

**ADV/AUX occurrences rescuable 81.0 % → 91.8 %.** `CURATED_ADV` is deliberately **empty**
here: the eleven curated adverbs in Gmh_v0.2 are Middle *High* German forms, and inventing
MLG adverbs from memory would break the corpus-derived rule (gml is the list we trust
least). Rerunning the tool rebuilds v0.2 idempotently from v0.1.

> **One deliberate divergence from the gmh build, forced by the data.** ReN double-tags its
> auxiliary/modal lemmas (`hebben`, `wēsen`, `schȫlen`, `künnen`, …) as `VERB` as well —
> their main-verb readings — so *every* one of the 149 aux+clitic fusion candidates has a
> head that is also a `VERB` lemma. The gmh build's whole-lemma "embeds a verb" test would
> therefore drop **all** of them (measured: 149 → 0 contractions). The gml build instead
> tests only the **non-head** components, plus the fusion token's own tail **POS** (a
> `VAFIN+VVPP` perfect periphrasis like `hebben+upspȫren` is caught by its `VVPP` tail even
> when the participle never occurs standalone). This keeps the 125 real contractions and
> still drops the periphrases. *The same latent over-drop exists in `build_gmh_lemma_augment.py`:
> applying the tail-only test there recovers ~23 additional `sîn+…` contractions ReM currently
> masks. That fix is deliberately **not** applied to the shipped Gmh_v0.2 — it would change an
> in-use artifact — but it is a known, defensible improvement if a gmh rebuild is ever wanted.*

   **Caveat on Latin filtering (corrected in v0.1):** several core Germanic
   function words are *homographs* of Latin ones and must not be filtered as
   foreign — `ut` (MLG "out"; cf. Du. *uit*, Ger. *aus*, Eng. *out*), `in`, `an`.
   The ReN tagger itself is inconsistent here: it tags `ut` as `FM` (foreign) but
   the h-spelling `uth` correctly as `APPR` (preposition), and `vt` as both. The
   Latin blocklist in the build tool therefore excludes these cognates by name;
   only unambiguous Latin (`cum`, `vel`, `sed`, `quod`, `ad`, …) is dropped.

## HiTS → UD, and why `tag_map` exists

ReM/ReN tag with **HiTS** (Historisches Tagset, Bochum), not UD. The ReM-trained
spaCy model is a `['tok2vec', 'tagger']` pipeline: it fills `token.tag_` with HiTS
and leaves `token.pos_` **empty**. POSNoise reads `pos_`, so without a bridge every
token would look like POS `""` and *nothing would be masked at all* — silently.
`load_hits_to_ud()` supplies the mapping (102 tags, covering both the ReM and ReN
HiTS variants, e.g. `DDART` vs `DDARTA`); it is the default for gmh/gml.

```python
from lambdag import POSNoiseMasker
m = POSNoiseMasker(language="gmh", spacy_model="path/to/GMH-Tagger/models/model-best")
m.mask_to_string("Swer an rehte güete wendet sîn gemüete dem volget sælde und êre")
# 'Swer an rehte # Ø sîn # dem Ø # und #'
```

## Limits — these are v0.1, weaker than the modern lists

1. **The tagger is the bottleneck, not the list.** The ReM GMH-Tagger mis-tags
   sentence-initial capitalised pronouns as proper nouns (`Uns/NE`, `Ich/NE`,
   `Swer/NE`) and `vil` as a noun. Those tokens survive *only because the whitelist
   rescues them* — which is a good argument for the whitelist and a bad sign for the
   tag stream. Masking rate on Nibelungenlied verse is ~59% vs ~40% for English.
2. **gml needs no spaCy model via the pre-tagged path.** ReN ships its own POS +
   lemma annotation, so `POSNoiseMasker.pretagged("gml")` + `mask_tagged`/
   `mask_tagged_batch` consume the ReN tags directly (HiTS→UD via `hits_to_ud.json`) —
   verified end-to-end on ReN snippets. Training a spaCy tagger from CorA XML (à la the
   GMH-Tagger repo's `cora2spacy.py`) is only needed to mask *un-annotated* MLG text.
   Note: 16 of ReN's finer HiTS tags (`DPDS`, `DPRELS`, `DNEGA`, `PAVKO`, `PTKN`, …) are
   absent from `hits_to_ud.json`; all are closed-class DET/PRON/PART/PUNCT that map to
   `""` and are therefore *kept by default* — the safe outcome — so the gap does not
   mis-mask, but extending the map would make the behaviour explicit.
3. **No sentence boundaries.** These corpora are verse and unpunctuated prose;
   POSNoise's `[.!?]`/newline rule will mostly produce one huge "sentence", which
   breaks Eq. (13) and the N=10 order. **Segment by verse line or by ReM's
   punctuation annotation** (cf. `matthias-stemmler/rem-punctuation-sentences`).
   This is a genuine blocker, not a detail.
4. **Not native-reviewed, no AV evaluation.** I am more confident in gmh than gml.
   Nothing here shows λG works on either.
5. **Licensing.** ReM and ReN require registration and have their own terms; the
   lists here are derived from CLTK's redistributions. Check the terms before
   publishing. Cite: Wich-Reif (ReM v1.0, Bonn/Bochum 2016); ReN-Team (ReN v0.6,
   Hamburg 2018); Schröder (2014); Peters & Nagel (2014).
6. **Köbler's dictionaries were not usable.** koeblergerhard.de is unreachable from
   my sandbox, and the lexica are lemma inventories with glosses — for POSNoise you
   need *attested spelling variants tagged in running text*, which is what ReM/ReN
   give and a dictionary does not. Lübben's *Mittelniederdeutsches Handwörterbuch*
   (Gutenberg #61948) is reachable and public domain, but marks POS only sparsely
   (360 `adv.`, 23 `conj.`, 8 `pron.`) — useful for auditing the gml list by hand,
   not for building it.
