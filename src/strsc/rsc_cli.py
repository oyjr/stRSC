"""``strsc rsc``: :func:`strsc.rsc` and :func:`strsc.regions` from the command line.

    strsc rsc sample.h5ad -p my_model=predictions.csv --layout spots --out strsc-output

Each ``-p`` names one model, as ``name=file`` or a file alone (the model is then named after the
file). The output folder receives the files of :meth:`strsc.Result.save`.
"""
from __future__ import annotations

import argparse
import sys

from .api import LAYOUTS, regions, rsc
from .files import InputError, _model_name, read_counts


def _prediction(text: str) -> tuple[str | None, str]:
    name, sep, path = text.partition("=")
    if sep and name and path:
        return name, path
    return None, text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strsc rsc",
        description="Reproducible-signal correlation (RSC) of one or more predictions, at each spatial scale, "
                    "or between and within annotated regions.")
    parser.add_argument("counts", nargs="+",
                        help="raw counts: a CSV or TSV table, an .h5ad file, a 10x .h5 matrix, or a Space Ranger or Xenium folder; "
                             "several give one section each")
    parser.add_argument("-p", "--prediction", action="append", required=True, metavar="NAME=FILE",
                        help="a prediction file, CSV/TSV or .h5ad, optionally named; repeat for each model")
    parser.add_argument("--layout", choices=LAYOUTS, default="spots", help="spots, grid or cells (default: spots)")
    parser.add_argument("--spacing-um", type=float, help="grid spacing in micrometers (layout grid)")
    parser.add_argument("--within", action="append", default=[], metavar="CLASS",
                        help="score the cells of this class on their own graph (layout cells); repeat for several classes")
    parser.add_argument("--only-within", action="store_true", help="with --within, do not score all cells as well")
    parser.add_argument("--cell-class", default="cell_class", help="column that holds the cell classes (default: cell_class)")
    parser.add_argument("--regions", metavar="COLUMN",
                        help="score between and within the regions in this column instead of spatial scales")
    parser.add_argument("--focus", metavar="REGION", help="with --regions, also score within this region and its structures")
    parser.add_argument("--labels", metavar="FILE", help="a CSV or TSV table with unit_id and columns such as region or cell_class")
    parser.add_argument("--um-per-unit", type=float, help="micrometers per coordinate unit, when the files do not say")
    parser.add_argument("--genes", help="comma-separated genes to score (default: every gene in the counts and all predictions)")
    parser.add_argument("--splits", type=int, help="use only the first count splits of the paper")
    parser.add_argument("--out", default="strsc-output", help="output folder (default: strsc-output)")
    parser.add_argument("--title", default="RSC computed with strsc", help="title of the evaluation bundle")
    parser.add_argument("--quiet", action="store_true", help="do not print progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.focus and not args.regions:
        print("strsc rsc: --focus needs --regions.", file=sys.stderr)
        return 2
    genes = [g.strip() for g in args.genes.split(",") if g.strip()] if args.genes else None

    def progress(event: dict) -> None:
        if args.quiet or "done" not in event or not event.get("total"):
            return
        where = " ".join(str(event[k]) for k in ("section", "population", "model") if event.get(k))
        print(f"\r{event['done']}/{event['total']} {where}".ljust(60), end="", file=sys.stderr, flush=True)

    try:
        counts = read_counts(args.counts[0] if len(args.counts) == 1 else args.counts, um_per_unit=args.um_per_unit)
        if args.labels:
            counts = counts.with_labels(args.labels)
        predictions: dict[str, str] = {}
        for text in args.prediction:
            name, path = _prediction(text)
            if name is None:
                name = _model_name(path)
            if name in predictions:
                raise InputError(f"Two predictions are named {name}. Name each model once, as name=file.")
            predictions[name] = path
        if not args.quiet:
            print(counts, file=sys.stderr)
        if args.regions:
            options = {"splits": args.splits} if args.splits else {}
            result = regions(counts, predictions, region=args.regions, focus=args.focus, genes=genes, progress=progress, **options)
        else:
            result = rsc(counts, predictions, args.layout, spacing_um=args.spacing_um, cell_class=args.cell_class,
                         within=args.within or None, all_cells=not args.only_within, genes=genes, splits=args.splits,
                         progress=progress)
    except InputError as error:
        print(f"\nstrsc rsc: {error}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(file=sys.stderr)
    for note in result.notes:
        print(f"Note: {note}", file=sys.stderr)
    print(result)
    written = result.save(args.out, args.title)
    print(f"Wrote {', '.join(path.name for path in written)} to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
