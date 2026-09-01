"""Minimal command-line front end for lambdag-lab.

lambdag.py itself deliberately has no __main__ (single-file library, examples in
its trailing comment block); this thin wrapper exists so that R (reticulate-free),
shell pipelines and remote boxes can drive the pipeline without writing Python.

    lambdag-lab info
    lambdag-lab mask  --lang de --in raw.txt --out masked.jsonl
    lambdag-lab score --known known.jsonl --questioned q.jsonl --reference ref.jsonl

File formats: masked documents are JSONL, one sentence per line, each line a JSON
array of tokens (the pipeline lingua franca; see README).
"""
import argparse
import json
import sys


def _read_masked(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cmd_info(_args):
    import lambdag
    print("lambdag-lab, module:", lambdag.__file__)
    print("numba available:", lambdag.NUMBA_AVAILABLE)
    print("languages:", ", ".join(lambdag.SUPPORTED_LANGUAGES))


def cmd_mask(args):
    from lambdag import POSNoiseMasker
    masker = POSNoiseMasker(args.lang, segment=args.segment, window=args.window)
    text = open(args.infile, encoding="utf-8").read()
    with open(args.out, "w", encoding="utf-8") as fh:
        for sent in masker.mask(text):
            fh.write(json.dumps(sent, ensure_ascii=False) + "\n")
    print(f"masked -> {args.out}")


def cmd_score(args):
    from lambdag import LambdaG, sqrt_correction
    lg = LambdaG(N=args.order, r=args.r, engine=args.engine,
                 random_state=args.seed)
    lg.set_reference(_read_masked(args.reference))
    res = lg.score(_read_masked(args.questioned), _read_masked(args.known),
                   with_details=False)
    out = {"lambda_G": res.lambda_G, "n_q": res.n_query_tokens,
           "v1_q": res.n_query_hapax,
           "sqrt": float(sqrt_correction(res.lambda_G, res.n_query_tokens))}
    print(json.dumps(out))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lambdag-lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="versions, numba, supported languages")
    m = sub.add_parser("mask", help="POSNoise-mask a raw text file (needs [spacy])")
    m.add_argument("--lang", required=True)
    m.add_argument("--in", dest="infile", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--segment", default="sentence", choices=["sentence", "window"])
    m.add_argument("--window", type=int, default=20)
    s = sub.add_parser("score", help="lambda_G for one case from masked JSONL files")
    s.add_argument("--known", required=True)
    s.add_argument("--questioned", required=True)
    s.add_argument("--reference", required=True)
    s.add_argument("--order", type=int, default=10)
    s.add_argument("--r", type=int, default=30)
    s.add_argument("--engine", default="kn", choices=["kn", "hpy", "ppmd"])
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"info": cmd_info, "mask": cmd_mask, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
