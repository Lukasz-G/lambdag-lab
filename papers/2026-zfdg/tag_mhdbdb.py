# Tag the normalised editions with the model that annotated the witnesses.
#
# The tagger's own command line would do this in one call, but two things here
# need control it does not offer, so the driver is written out:
#
#   THE BATCH. The inference batch is read from the checkpoint, where it is set
#   for a machine whose card is free. This one shares its card with other work,
#   and at the checkpoint's setting the run dies of memory on the first long
#   text. The batch is therefore lowered in the loaded configuration -- the
#   checkpoint on disk is not touched -- and the size is a flag, so a run that
#   dies can be resumed smaller.
#
#   RESUMING. 222 texts of two and a half million tokens is long enough that a
#   failure in the middle must not cost the whole run, so a text whose output
#   already exists is skipped and the number skipped is reported.
#
# The model is `runs/s_crf`, which is the one `build_masked_banks.py` reads the
# vocabulary of: the point of tagging these texts at all is to put them on the
# SAME annotation basis as the witnesses, and a different checkpoint would defeat
# it.
#
#   python medieval/tag_mhdbdb.py [--batch 3] [--cpu]
#
# Output: medieval/hydra_out_mhdbdb/<author>__<sigle>.txt

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN = HERE / "hydra_in_mhdbdb"
OUT = HERE / "hydra_out_mhdbdb"
TAGGER = Path.home() / "github" / "hydra_august25" / "hydra"
MODEL = TAGGER / "runs" / "s_crf" / "model_only.pt"


def main():
    global IN, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=3,
                    help="inference batch in chunks; the checkpoint asks for 12, "
                         "which needs a card to itself")
    ap.add_argument("--cpu", action="store_true",
                    help="tag on the processor; correct but far slower, and the "
                         "fallback when the card is busy")
    ap.add_argument("--inputs", default=str(IN),
                    help="a directory of tagger inputs. Any corpus written in "
                         "this shape is tagged the same way and, more to the "
                         "point, by the same checkpoint -- which is the only "
                         "reason its texts may afterwards stand beside the "
                         "witnesses in one comparison.")
    ap.add_argument("--outputs", default=str(OUT))
    args = ap.parse_args()
    # resolved, so that a directory given relative to the working directory can
    # still be reported relative to the repository at the end
    IN, OUT = Path(args.inputs).resolve(), Path(args.outputs).resolve()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(TAGGER))
    import torch
    from hydra.tag import (load_model_for_inference, read_input_file,
                           tag_document)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda:0")
    model, vocabs, cfg = load_model_for_inference(MODEL, device)
    # the configuration is frozen, as it should be -- it came out of the
    # checkpoint and nothing may silently differ from what was trained. A new
    # inference block with one field changed says the change out loud.
    import dataclasses
    cfg = dataclasses.replace(
        cfg, infer=dataclasses.replace(cfg.infer, batch_chunks=args.batch))
    snapper = None
    if cfg.infer.snap_lemmas and vocabs.lemma_counts:
        from hydra.snap import LemmaSnapper
        snapper = LemmaSnapper(vocabs.lemma_inventory)

    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(IN.glob("*.txt"))
    todo = [f for f in files if not (OUT / f.name).exists()]
    print(f"{len(files)} texts, {len(files) - len(todo)} already tagged, "
          f"{len(todo)} to do; device {device}, batch {args.batch}")
    t0, done, ntok = time.time(), 0, 0
    for f in todo:
        entries = read_input_file(f, "auto")
        surfaces = [t for kind, t in entries if kind == "token"]
        preds = tag_document(model, vocabs, cfg, surfaces, device)
        # the same three columns the witness files carry, in the same order
        with (OUT / f.name).open("w", encoding="utf-8") as fh:
            it = iter(preds)
            for kind, text in entries:
                if kind == "raw":
                    fh.write(text + "\n")
                    continue
                lemma, pos, morph = next(it)
                if snapper is not None:
                    lemma = snapper.snap(lemma, len(pos.split("+")))
                fh.write(f"{text}\t{lemma}\t{pos}\t{morph}\n")
        done += 1
        ntok += len(surfaces)
        el = time.time() - t0
        print(f"  {done:4}/{len(todo)} {f.stem[:44]:46} {len(surfaces):7,} "
              f"tokens  {ntok / max(el, 1):,.0f} tok/s", flush=True)
    print(f"\n{done} texts, {ntok:,} tokens in {time.time() - t0:.0f}s "
          f"-> {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
