"""Command-line entry point."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="strsc")
    parser.add_argument("command", choices=["hest", "her2", "fourier", "export", "export-results", "example"])
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.argv = [f"strsc {args.command}", *remaining]
    if args.command == "hest":
        from .hest import main as run
    elif args.command == "export":
        from .export import main as run
    elif args.command == "export-results":
        from .evaluation_export import main as run
    elif args.command == "her2":
        from .her2 import main as run
    elif args.command == "fourier":
        from .fourier import main as run
    else:
        from .example import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
