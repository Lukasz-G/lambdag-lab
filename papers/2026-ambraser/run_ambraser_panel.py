# The short anonymous pieces of the Ambraser Heldenbuch, against a panel.
#
# Four of the codex's anonymous works are too short to be cut into regions and
# stand on the known side: Moriz von Craun at 8,776 masked tokens, Der Mantel at
# 4,893, Von dem ubeln Wibe at 4,391, the Zweites Buchlein at 4,240. The design
# therefore turns round. Each named author's long work becomes the known text in
# turn, cut into regions, and the short pieces are the questioned passages -- the
# direction the chronicle study found to be the right one, and the only one these
# extents allow.
#
# THE CODEX SUPPLIES ITS OWN POSITIVE CONTROLS. Hartmann stands in it three times
# and the Stricker twice, so a run with Erec as the known text scores the Zweites
# Buchlein beside Hartmann's own Klage: if the disputed piece behaves like the
# Klage rather than like the strangers, that is evidence of a kind an anonymous
# case rarely affords. Every questioned text is scored against every known text,
# so the result is a panel rather than a verdict.
#
# AND THE PANEL ANSWERS THE OTHER QUESTION TOO. Two short pieces cannot be scored
# against each other -- both sides would be grammars of four thousand tokens --
# but if one poet wrote two of them, the two should rank the panel's authors
# alike. Their profiles across the panel are therefore correlated, which is the
# same statistic that showed the Rolandslied's two witnesses agreeing at r = 0.94.
#
#   python experiments/run_ambraser_panel.py [--dry-run]
#   python experiments/panel_matrix.py
#
# Output: scores/konrad_profile/regions__d__r*__q2000__panel_<known>.jsonl

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A = "ambraser__anon__"

# the short anonymous pieces, which are the question
QUESTIONED = [f"{A}moriz_von_craun__d", f"{A}der_mantel__d",
              f"{A}von_dem_uebelen_wibe__d", f"{A}zweites_buechlein__d"]
# short pieces under a name, which give the panel its scale of comparison and,
# where the author matches the known text, its positive control
NAMED_SHORT = ["ambraser__hartmann_von_aue__klage__d",
               "ambraser__stricker__frauenehre__d",
               "ambraser__wolfram_von_eschenbach__titurel__d",
               "ambraser__herrand_von_wildonie__der_nackte_kaiser__d"]

# known text, regions, and the author's own other works to keep out of the scale
PANEL = [
    ("erec", "ambraser__hartmann_von_aue__erec__d", 2,
     "ambraser__hartmann_von_aue__*", "Hartmann, Erec"),
    ("iwein", "hvad__hartmann_von_aue__iwein__d_lc", 2,
     "ambraser__hartmann_von_aue__*,hvad__hartmann_von_aue__*",
     "Hartmann, Iwein"),
    ("amis", "ambraser__stricker__pfaffe_amis__d", 1,
     "ambraser__stricker__*", "Der Stricker"),
    ("frauenbuch", "ambraser__ulrich_von_liechtenstein__frauenbuch__d", 1,
     "ambraser__ulrich_von_liechtenstein__*", "Ulrich von Liechtenstein"),
    ("helmbrecht", "ambraser__wernher_der_gartenaere__helmbrecht__d", 1,
     "ambraser__wernher_der_gartenaere__*", "Wernher der Gartenaere"),
    ("nibelungenlied", f"{A}nibelungenlied__d", 3, "", "the Nibelungenlied"),
    ("dietrichs_flucht", f"{A}dietrichs_flucht__d", 2, "", "Dietrichs Flucht"),
]
L = 3000


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True,
                       cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    py = sys.executable
    for name, known, nreg, own, label in PANEL:
        if args.only and args.only != name:
            continue
        log = ROOT / f"logs_panel_{name}.log"
        log.write_text("", encoding="utf-8")
        run_name = f"regions__{known.split('__')[-1]}__r{nreg}__q{L}__panel_{name}.jsonl"
        # the anonymous epics may not set the scale for their own codex-mates, and
        # a known author's other works may not either
        excl = [f"{A}*"] + ([own] if own else [])
        cmd = [py, "experiments/run_kc_regions.py", "--witness", known,
               "--regions", str(nreg), "--L", str(L),
               "--candidates", ",".join(QUESTIONED),
               "--control", ",".join(NAMED_SHORT),
               "--exclude", ",".join(excl),
               "--cohort-bank", "witnesses", "--cohort-source", "ambraser",
               "--donor-bank", "editions", "--donor-years", "1150-1250",
               "--tag", f"__panel_{name}"]
        if args.dry_run:
            cmd.append("--dry-run")
        t0 = time.time()
        print(f"\n=== {label} as the known text ===", flush=True)
        if (ROOT / "scores" / "konrad_profile" / run_name).exists() \
                and not args.dry_run:
            print("  the scores are already there", flush=True)
            continue
        run(cmd, log)
        if args.dry_run:
            print(log.read_text(encoding="utf-8").strip()[-380:])
            continue
        print(f"  done in {time.time() - t0:.0f}s -> {log.name}", flush=True)


if __name__ == "__main__":
    main()
