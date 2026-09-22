#!/usr/bin/env python3
"""Compare official-runner policy trace scale against native MuJoCo traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


POLICY_DEFAULT = np.array(
  [
    0.0,
    -0.65,
    1.30,
    0.0,
    -0.65,
    1.30,
    0.0,
    -0.65,
    1.30,
    0.0,
    -0.65,
    1.30,
  ],
  dtype=np.float64,
)

ROBOT_DEFAULT = np.array(
  [
    0.0844,
    -0.7224,
    1.4200,
    -0.0789,
    -0.7242,
    1.4206,
    0.0737,
    -0.7104,
    1.4100,
    -0.0684,
    -0.7030,
    1.4120,
  ],
  dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--official-csv", type=Path, required=True)
  parser.add_argument("--native-actions", type=Path)
  parser.add_argument("--native-state", type=Path)
  parser.add_argument("--skip-warmup-rows", type=int, default=0)
  return parser.parse_args()


def summarize(name: str, values: np.ndarray) -> None:
  abs_values = np.abs(values)
  flat = abs_values.reshape(-1)
  print(f"[{name}] shape={values.shape}")
  print(
    "  mean_abs={:.4f} p50_abs={:.4f} p95_abs={:.4f} "
    "p99_abs={:.4f} max_abs={:.4f}".format(
      float(np.mean(flat)),
      float(np.percentile(flat, 50)),
      float(np.percentile(flat, 95)),
      float(np.percentile(flat, 99)),
      float(np.max(flat)),
    )
  )
  print(
    "  per_joint_mean_abs=[{}]".format(
      ", ".join(f"{x:.4f}" for x in np.mean(abs_values, axis=0))
    )
  )
  print(
    "  per_joint_max_abs=[{}]".format(
      ", ".join(f"{x:.4f}" for x in np.max(abs_values, axis=0))
    )
  )


def load_official_csv(path: Path, skip_rows: int) -> dict[str, np.ndarray]:
  data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
  if data.ndim == 0:
    data = data.reshape(1)
  if skip_rows:
    data = data[skip_rows:]
  action_cols = [f"action_{i}" for i in range(12)]
  target_cols = [f"target_{i}" for i in range(12)]
  missing = [name for name in action_cols + target_cols if name not in data.dtype.names]
  if missing:
    raise KeyError(f"{path} is missing columns: {missing}")
  actions = np.column_stack([data[name] for name in action_cols])
  targets = np.column_stack([data[name] for name in target_cols])
  commands = np.column_stack([data[name] for name in ("cmd_vx", "cmd_vy", "cmd_wz")])
  return {
    "actions": actions,
    "targets": targets,
    "target_delta": targets - ROBOT_DEFAULT,
    "commands": commands,
  }


def main() -> None:
  args = parse_args()
  official = load_official_csv(args.official_csv, args.skip_warmup_rows)
  print(f"[INFO] official trace: {args.official_csv}")
  summarize("official_action_raw", official["actions"])
  summarize("official_target_delta_from_robot_default", official["target_delta"])
  summarize("official_cmd", official["commands"])

  if args.native_actions is not None:
    native_actions = np.load(args.native_actions)["actions"]
    print(f"[INFO] native action trace: {args.native_actions}")
    summarize("native_action_raw", native_actions)

  if args.native_state is not None:
    native_state = np.load(args.native_state)
    if "joint_target" not in native_state:
      raise KeyError(f"{args.native_state} has no 'joint_target' array")
    native_target_delta = native_state["joint_target"] - POLICY_DEFAULT
    print(f"[INFO] native state trace: {args.native_state}")
    summarize("native_target_delta_from_policy_default", native_target_delta)


if __name__ == "__main__":
  main()
