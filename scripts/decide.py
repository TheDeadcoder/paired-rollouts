"""Apply the pre-registered decision rules of one condition to accepted run directories and write the decision
register.

    python scripts/decide.py --condition C2 --runs outputs/runs/gate1-c2-paired-s0 outputs/runs/t1-c2-independent-s0 ... \
        --out registers/decisions/tier1_c2.json [--boot 10000] [--seed 20260911]

Every run must be COMPLETE and complete (verify_run ACCEPT); the rules are applied exactly as registered and the
outcome, pass or miss, is written as is.
"""

import argparse

from pairedrl.analysis.hypotheses import decide, format_decision, write_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["C2", "C4"])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    decision = decide(args.condition, args.runs, n_boot=args.boot, seed=args.seed)
    write_decision(decision, args.out)
    print(format_decision(decision))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
