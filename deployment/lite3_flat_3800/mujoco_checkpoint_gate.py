#!/usr/bin/env python3
"""Rank Lite3 checkpoints with the native MuJoCo closed-loop gate."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SUMMARY_RE = re.compile(
  r"\[SUMMARY\] "
  r"mean_vx=(?P<mean_vx>[+-]?\d+\.\d+) "
  r"mean_vy=(?P<mean_vy>[+-]?\d+\.\d+) "
  r"mean_wz=(?P<mean_wz>[+-]?\d+\.\d+) "
  r"mean_z=(?P<mean_z>\d+\.\d+) "
  r"mean_tilt=(?P<mean_tilt>\d+\.\d+)"
)
ACTION_RE = re.compile(
  r"\[SUMMARY_ACTION\] "
  r"mean_abs_raw=(?P<mean_abs_raw>\d+\.\d+) "
  r"max_abs_raw=(?P<max_abs_raw>\d+\.\d+) "
  r"frac_abs_gt_1\.4=(?P<frac_abs_gt_1_4>\d+\.\d+)"
)

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoints", type=Path, nargs="+")
  parser.add_argument("--cmd-vx", type=float, default=0.4)
  parser.add_argument("--cmd-vy", type=float, default=0.0)
  parser.add_argument("--cmd-wz", type=float, default=0.0)
  parser.add_argument("--duration-s", type=float, default=10.0)
  parser.add_argument("--warmup-s", type=float, default=1.0)
  parser.add_argument(
    "--clip-actions",
    type=float,
    default=None,
    help="Optional policy-action clamp. Defaults to disabled.",
  )
  return parser.parse_args()


def run_gate(args: argparse.Namespace, checkpoint: Path) -> dict[str, float | str]:
  command = [
    sys.executable,
    str(SCRIPT_DIR / "mujoco_sim2real.py"),
    "--checkpoint",
    str(checkpoint),
    "--duration-s",
    str(args.duration_s),
    "--warmup-s",
    str(args.warmup_s),
    "--cmd-vx",
    str(args.cmd_vx),
    "--cmd-vy",
    str(args.cmd_vy),
    "--cmd-wz",
    str(args.cmd_wz),
    "--all-feet-contact",
  ]
  if args.clip_actions is not None:
    command.extend(["--clip-actions", str(args.clip_actions)])

  result = subprocess.run(
    command,
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
  )
  summary = SUMMARY_RE.search(result.stdout)
  action = ACTION_RE.search(result.stdout)
  if result.returncode != 0 or summary is None or action is None:
    return {
      "checkpoint": str(checkpoint),
      "status": "failed",
      "score": float("-inf"),
    }

  metrics = {
    key: float(value)
    for key, value in (summary.groupdict() | action.groupdict()).items()
  }
  vx_error = abs(metrics["mean_vx"] - args.cmd_vx)
  yaw_error = abs(metrics["mean_wz"] - args.cmd_wz)
  height_deficit = max(0.0, 0.30 - metrics["mean_z"])
  score = (
    -vx_error
    - 0.25 * abs(metrics["mean_vy"] - args.cmd_vy)
    - 0.10 * yaw_error
    - 0.50 * metrics["mean_tilt"]
    - 2.00 * height_deficit
    - 0.02 * metrics["frac_abs_gt_1_4"]
  )
  return {
    "checkpoint": str(checkpoint),
    "status": "ok",
    "score": score,
    **metrics,
  }


def main() -> None:
  args = parse_args()
  rows = [run_gate(args, checkpoint) for checkpoint in args.checkpoints]
  rows.sort(key=lambda row: float(row["score"]), reverse=True)

  print(
    "checkpoint,status,score,mean_vx,mean_vy,mean_wz,mean_z,mean_tilt,"
    "mean_abs_raw,max_abs_raw,frac_abs_gt_1_4"
  )
  for row in rows:
    if row["status"] != "ok":
      print(f"{row['checkpoint']},{row['status']},{row['score']}")
      continue
    print(
      f"{row['checkpoint']},{row['status']},{row['score']:.6f},"
      f"{row['mean_vx']:.6f},{row['mean_vy']:.6f},{row['mean_wz']:.6f},"
      f"{row['mean_z']:.6f},{row['mean_tilt']:.6f},"
      f"{row['mean_abs_raw']:.6f},{row['max_abs_raw']:.6f},"
      f"{row['frac_abs_gt_1_4']:.6f}"
    )


if __name__ == "__main__":
  main()
