# The anonymous works of the Ambraser Heldenbuch, read against one another.
#
# The codex was written out by one hand between 1504 and 1516, so every text in it
# has passed through the same copyist. That is what makes it the best material in
# this corpus for an authorship question: the scribal toll measured on the witness
# design -- 0.055 in the area under the curve on the lemma stream, 0.151 on the
# scribe's own spelling -- does not enter a comparison made inside the book.
#
# The questions asked here need no candidate author at all, which is why they are
# asked first. Each is relational: is this anonymous poem by the poet of that one?
# The Nibelungenklage against the Nibelungenlied, the Rabenschlacht against
# Dietrichs Flucht, Ortnit against Wolfdietrich, the Nibelungenlied against
# Kudrun. The controls in each case are the other anonymous epics of the codex,
# which share the genre, the metre, the matter and the copyist and differ only in
# the poet -- the same role Lambrecht's Alexanderlied plays for the Kaiserchronik.
#
# THREE ARMS, because the first attempt showed why the scale matters more than
# anything else in this design. Read against a cohort of normalised editions,
# every text of the codex stands so far above the scale that candidate and control
# alike saturate the bound and the odds separate nothing; the codex's own named
# pieces are the same layer as the case and are therefore the scale that can carry
# a verdict.
#
#   codex     the scale from the codex's named pieces, grammars from the tagged
#             editions of the poems' own period, questioned passages of 4,000
#   short     the same with passages of 2,000, which is the granularity the
#             witness design calls the floor for a question about a hand
#   borrowed  the grammars from modern German verse in the shared alphabet of
#             grammatical classes, which the Kaiserchronik chapter finds
#             admissible for an ordering and not for odds -- so this arm is
#             reported on the field test alone
#   edition   the same questions asked of the same poems in their normalised
#             editions: a different hand, a different orthography, and a scale
#             that is native to it. The arms above rest on one copy, so this is
#             the control that separates a property of the poems from a property
#             of Hans Ried.
#
#   python experiments/run_ambraser_pairs.py [--arm codex] [--only dietrich]
#
# Output: scores/konrad_profile/regions__d__r*__q*__amb<arm>_<pair>.jsonl, and per
# pair the calibrated verdict and the field test.

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A = "ambraser__anon__"

# known, its regions, the poem whose authorship is in question, and the poems of
# the same kind that say how high the genre alone reaches
PAIRS = [
    ("nibelungen", f"{A}nibelungenlied__d", 10, f"{A}nibelungenklage__d",
     [f"{A}kudrun__d", f"{A}biterolf_und_dietleib__d", f"{A}ortnit__d",
      f"{A}rabenschlacht__d"],
     "the Klage against the Nibelungenlied"),
    ("dietrich", f"{A}dietrichs_flucht__d", 8, f"{A}rabenschlacht__d",
     [f"{A}nibelungenlied__d", f"{A}kudrun__d", f"{A}biterolf_und_dietleib__d",
      f"{A}ortnit__d"],
     "the Rabenschlacht against Dietrichs Flucht"),
    ("ortnit", f"{A}wolfdietrich_a__d", 4, f"{A}ortnit__d",
     [f"{A}nibelungenlied__d", f"{A}kudrun__d", f"{A}biterolf_und_dietleib__d",
      f"{A}rabenschlacht__d"],
     "Ortnit against Wolfdietrich A"),
    ("kudrun", f"{A}kudrun__d", 9, f"{A}nibelungenlied__d",
     [f"{A}biterolf_und_dietleib__d", f"{A}ortnit__d", f"{A}rabenschlacht__d",
      f"{A}dietrichs_flucht__d"],
     "the Nibelungenlied against Kudrun"),
    # Verse form against hand. The Klage is in couplets and the poems it lost to
    # against the Nibelungenlied are strophic, so its distance there may be a
    # property of the form rather than of the poet. These two put a COUPLET epic
    # on the known side and the Klage in the candidate's place: if form governs,
    # the Klage should rise to the top; if the hand does, the Rabenschlacht --
    # strophic, and the one poem this study finds to share a poet with Dietrichs
    # Flucht -- should stay above it.
    ("formdf", f"{A}dietrichs_flucht__d", 8, f"{A}nibelungenklage__d",
     [f"{A}rabenschlacht__d", f"{A}nibelungenlied__d", f"{A}kudrun__d",
      f"{A}ortnit__d", f"{A}biterolf_und_dietleib__d"],
     "the Klage against Dietrichs Flucht, a poem in couplets"),
    ("formbi", f"{A}biterolf_und_dietleib__d", 10, f"{A}nibelungenklage__d",
     [f"{A}rabenschlacht__d", f"{A}nibelungenlied__d", f"{A}kudrun__d",
      f"{A}ortnit__d", f"{A}dietrichs_flucht__d"],
     "the Klage against Biterolf und Dietleib, the other couplet epic"),
]

# The same poems as their editors print them. Every result above rests on ONE
# copy, Hans Ried's, and a reader may fairly ask whether a verdict is a property
# of the poems or of that copy -- most sharply for the Klage, which comes out as
# the least Nibelungenlied-like of the five poems tested and did not repeat on a
# second witness. These poems are also published in normalised editions, tagged
# here by the same model, and an edition is a different hand with different
# spelling: a relational verdict that survives the move is a verdict about the
# poets. The layer boundary that the Kaiserchronik study had to price as a
# residual disappears, because the case and its scale are then in one layer.
EDITION = {
    f"{A}nibelungenlied__d": "anonym__NBB",
    f"{A}nibelungenklage__d": "anonym__KLA",
    f"{A}kudrun__d": "anonym__KU",
    f"{A}biterolf_und_dietleib__d": "anonym__BRF",
    f"{A}ortnit__d": "anonym__ORT",
    f"{A}wolfdietrich_a__d": "anonym__WDA",
    f"{A}rabenschlacht__d": "anonym__RAB",
    f"{A}dietrichs_flucht__d": "anonym__DFL",
}

# the scale always comes from the codex's named pieces: --cohort-source keeps it
# inside the book and --exclude keeps the anonymous epics, which are the case, out
CODEX = ["--cohort-bank", "witnesses", "--cohort-source", "ambraser"]
# in the editors' layer the scale is native, so the cohort is the edition bank
# itself; the window runs to 1300 because the Dietrich epics are dated there and a
# window that stopped at 1250 would drop half the questions
EDITIONS = ["--case-bank", "editions", "--case-years", "1150-1300",
            "--cohort-bank", "editions", "--cohort-years", "1150-1300",
            "--donor-bank", "editions", "--donor-years", "1150-1300"]
ARMS = {
    "codex": dict(L=4000, calibrate=True,
                  extra=CODEX + ["--donor-bank", "editions",
                                 "--donor-years", "1150-1250"]),
    "short": dict(L=2000, calibrate=True,
                  extra=CODEX + ["--donor-bank", "editions",
                                 "--donor-years", "1150-1250"]),
    "borrowed": dict(L=4000, calibrate=False,
                     extra=CODEX + ["--encoding", "catranks",
                                    "--donor-bank", "german_tgverseall_v12"]),
    "edition": dict(L=4000, calibrate=True, layer="edition", extra=EDITIONS),
}


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True,
                       cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="codex", choices=sorted(ARMS))
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    arm, py = ARMS[args.arm], sys.executable
    L = arm["L"]
    for name, known, nreg, cand, ctrl, label in PAIRS:
        if args.only and args.only != name:
            continue
        if arm.get("layer") == "edition":
            # the same question, asked of the editors' texts. A pair whose poems
            # are not all published in the tagged editions cannot be asked here,
            # and is skipped rather than asked of a mixture of layers.
            want = [known, cand] + ctrl
            if any(w not in EDITION for w in want):
                print(f"  skipped {name}: {[w for w in want if w not in EDITION]}"
                      f" is not in the edition bank", flush=True)
                continue
            known, cand = EDITION[known], EDITION[cand]
            ctrl = [EDITION[w] for w in ctrl]
        tag = f"__amb{args.arm}_{name}"
        log = ROOT / f"logs_amb_{args.arm}_{name}.log"
        log.write_text("", encoding="utf-8")
        run_name = f"regions__{known.split('__')[-1]}__r{nreg}__q{L}{tag}.jsonl"
        cmd = [py, "experiments/run_kc_regions.py", "--witness", known,
               "--regions", str(nreg), "--L", str(L),
               "--candidates", cand, "--control", ",".join(ctrl),
               "--exclude", "anonym__*" if arm.get("layer") == "edition"
               else f"{A}*", "--tag", tag] + arm["extra"]
        if args.dry_run:
            cmd.append("--dry-run")
        t0 = time.time()
        print(f"\n=== {label} [{args.arm}] ===", flush=True)
        if (ROOT / "scores" / "konrad_profile" / run_name).exists() \
                and not args.dry_run:
            print("  the scores are already there; calibrating only", flush=True)
        else:
            run(cmd, log)
        if args.dry_run:
            print(log.read_text(encoding="utf-8").strip()[-420:])
            continue
        if arm["calibrate"]:
            run([py, "experiments/run_calibration_cases.py", "--from-run",
                 run_name, "--case-bank", "mhdbdb", "--years",
                 "1150-1300" if arm.get("layer") == "edition" else "1150-1250",
                 "--tag", f"amb{args.arm}{name}"], log)
            run([py, "experiments/calibrate_crosswork.py", "--calibration",
                 f"calib__amb{args.arm}{name}__q{L}.jsonl", "--regions",
                 run_name, "--tag", f"amb{args.arm}{name}"], log)
        run([py, "experiments/rank_candidate_vs_field.py", "--run", run_name,
             "--tag", f"amb{args.arm}{name}"], log)
        print(f"  done in {time.time() - t0:.0f}s -> {log.name}", flush=True)


if __name__ == "__main__":
    main()
