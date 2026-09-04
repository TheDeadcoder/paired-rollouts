"""Build registers/calibration.json from the downloaded calibration runs and print the verdict."""

import argparse

from pairedrl.analysis.registers import build_calibration_register, format_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run directories under outputs/runs")
    parser.add_argument("--out", default="registers/calibration.json")
    args = parser.parse_args()
    register = build_calibration_register(args.runs, args.out)
    print(format_calibration(register))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
