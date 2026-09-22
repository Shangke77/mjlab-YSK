#!/usr/bin/env python3
"""Compare policy-rate state traces from two sim2sim runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

COMMON_FIELDS = (
  "root_pos",
  "root_vel_b",
  "joint_pos",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--label-reference", default="reference")
  parser.add_argument("--label-candidate", default="candidate")
  parser.add_argument(
    "--first-n",
    type=int,
    default=None,
    help="Compare only the first N policy-rate samples.",
  )
  return parser.parse_args()


def load_field(
  trace: np.lib.npyio.NpzFile,
  trace_name: str,
  field: str,
  count: int,
) -> np.ndarray:
  if field not in trace:
    raise KeyError(f"{trace_name} is missing field {field!r}")
  value = trace[field]
  return value[:count]


def print_error(field: str, reference: np.ndarray, candidate: np.ndarray) -> None:
  error = candidate - reference
  rmse = float(np.sqrt(np.mean(np.square(error))))
  max_abs = float(np.max(np.abs(error)))
  last_abs = float(np.max(np.abs(error[-1])))
  print(
    f"[COMPARE] {field}: rmse={rmse:.6f} "
    f"max_abs={max_abs:.6f} last_max_abs={last_abs:.6f}"
  )


def main() -> None:
  args = parse_args()
  reference = np.load(args.reference)
  candidate = np.load(args.candidate)
  if not np.isclose(reference["dt"], candidate["dt"]):
    raise RuntimeError(f"Trace dt differs: {reference['dt']} vs {candidate['dt']}")

  sample_count = min(len(reference["root_pos"]), len(candidate["root_pos"]))
  if args.first_n is not None:
    if args.first_n <= 0:
      raise ValueError("--first-n must be positive.")
    sample_count = min(sample_count, args.first_n)
  if sample_count == 0:
    raise RuntimeError("No overlapping trace samples to compare.")

  print(
    "[INFO] Comparing "
    f"{sample_count} samples at dt={float(reference['dt']):.4f}s: "
    f"{args.label_reference} -> {args.label_candidate}"
  )
  for field in COMMON_FIELDS:
    print_error(
      field,
      load_field(reference, str(args.reference), field, sample_count),
      load_field(candidate, str(args.candidate), field, sample_count),
    )
  for optional_field in ("joint_target", "torque", "foot_normal_force"):
    if optional_field in reference and optional_field in candidate:
      print_error(
        optional_field,
        load_field(reference, str(args.reference), optional_field, sample_count),
        load_field(candidate, str(args.candidate), optional_field, sample_count),
      )
  print(
    f"[FINAL] {args.label_reference}_root_pos="
    f"{np.array2string(reference['root_pos'][sample_count - 1], precision=4)}"
  )
  print(
    f"[FINAL] {args.label_candidate}_root_pos="
    f"{np.array2string(candidate['root_pos'][sample_count - 1], precision=4)}"
  )


if __name__ == "__main__":
  main()
