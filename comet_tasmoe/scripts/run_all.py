from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from run_single import run_experiment  # type: ignore
else:
    from .run_single import run_experiment

SCENARIOS = ["baseline", "taps", "tacs", "tas"]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run all TAS-MoE scenarios")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--tas", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for scenario in SCENARIOS:
        run_experiment(args.arch, args.workload, args.tas, scenario, args.out)


if __name__ == "__main__":
    main()
