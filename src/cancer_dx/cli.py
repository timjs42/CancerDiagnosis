"""Command-line interface for the tumor-marker Naive Bayes classifier."""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

from .data import DEFAULT_PATIENTS_PATH, load_patients
from .model import ALL_MARKERS, predict_class


def _print_patient_result(name: str, markers: dict, ranked: List[Tuple[str, float]], top: int) -> None:
    marker_str = ", ".join(f"{m}={markers[m]:.1f}" for m in ALL_MARKERS)
    print(f"\nPatient: {name}")
    print(f"  {marker_str}")
    print(f"  -> Predicted Class: ** {ranked[0][0]} **\n")
    print("  Top Predictions:")
    print("  ┌──────────────────────────────┬───────────┐")
    print("  │ Cancer / Stage               │Probability│")
    print("  ├──────────────────────────────┼───────────┤")
    for cls, prob in ranked[:top]:
        print(f"  │ {cls:28s} │ {prob:9.3f} │")
    print("  └──────────────────────────────┴───────────┘")
    print("-" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cancer-dx",
        description="Rank candidate cancer types/stages from tumor marker levels "
        "(HE4, AFP, CA19-9) using a Gaussian Naive Bayes classifier.",
    )
    parser.add_argument(
        "--patients-file",
        type=str,
        default=str(DEFAULT_PATIENTS_PATH),
        help="Path to a JSON file of patient records. Defaults to the bundled example set.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top-ranked classes to display per patient (default: 5).",
    )

    single = parser.add_argument_group("single patient (overrides --patients-file)")
    single.add_argument("--he4", type=float, help="HE4 marker level for a single patient")
    single.add_argument("--afp", type=float, help="AFP marker level for a single patient")
    single.add_argument("--ca19-9", type=float, help="CA19-9 marker level for a single patient")
    single.add_argument("--name", type=str, default="Patient", help="Name label for the single patient")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    single_values = [args.he4, args.afp, args.ca19_9]
    if any(v is not None for v in single_values):
        if any(v is None for v in single_values):
            parser.error("--he4, --afp, and --ca19-9 must all be provided together")
        patients = [{"name": args.name, "HE4": args.he4, "AFP": args.afp, "CA19-9": args.ca19_9}]
    else:
        patients = load_patients(args.patients_file)

    print("=" * 60)
    print(" Tumor-Marker Naive Bayes Cancer Classification Results ")
    print("=" * 60)

    for record in patients:
        name = record.get("name", "Patient")
        markers = {m: record[m] for m in ALL_MARKERS}
        _, ranked = predict_class(markers)
        _print_patient_result(name, markers, ranked, args.top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())