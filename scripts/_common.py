"""Shared helpers for the figure scripts (argument parsing, paths)."""

from __future__ import annotations

import argparse
import os
import sys

# Make the package importable when the script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Step-count presets.  "paper" reproduces the published 3e5 + 3e5 steps and
# is slow; "quick" is for a fast sanity check; "default" is a sensible
# compromise that already reproduces the shape of the diagrams.
PRESETS = {
    "quick":   dict(relax_steps=20_000,  measure_steps=20_000),
    "default": dict(relax_steps=60_000,  measure_steps=60_000),
    "paper":   dict(relax_steps=300_000, measure_steps=300_000),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=list(PRESETS), default="default",
                        help="step-count preset (default: %(default)s)")
    parser.add_argument("--relax-steps", type=int, default=None,
                        help="override number of relaxation steps")
    parser.add_argument("--measure-steps", type=int, default=None,
                        help="override number of measurement steps")
    parser.add_argument("--L", type=float, default=17.3,
                        help="ring length in metres (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="base RNG seed (default: %(default)s)")
    parser.add_argument("--no-show", action="store_true",
                        help="save figure without opening a window")


def resolve_steps(args) -> tuple[int, int]:
    preset = PRESETS[args.preset]
    relax = args.relax_steps if args.relax_steps is not None else preset["relax_steps"]
    measure = args.measure_steps if args.measure_steps is not None else preset["measure_steps"]
    return relax, measure
