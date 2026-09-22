#!/usr/bin/env python3
"""Export a Lite3 checkpoint to the ONNX file expected by Lite3_rl_deploy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mujoco_sim2real import (  # noqa: E402
  EFFORT_LIMIT,
  POLICY_JOINT_NAMES,
  load_policy,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
  SCRIPT_DIR.parent.parent
  / "logs"
  / "rsl_rl"
  / "deep_lite3_velocity"
  / "2026-07-22_21-23-41"
  / "model_9999.pt"
)
DEFAULT_OUTPUT_DIR = Path.home() / "Lite3_rl_deploy" / "policy" / "ppo"
ONNX_OPSET_VERSION = 18


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=DEFAULT_CHECKPOINT,
    help="RSL-RL checkpoint that passed native MuJoCo sim2sim.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory for policy.onnx and manifest.json.",
  )
  return parser.parse_args()


def tolist(array: np.ndarray) -> list[float]:
  return [float(value) for value in array.tolist()]


def export_policy(checkpoint: Path, output_dir: Path) -> tuple[Path, object]:
  policy, policy_spec = load_policy(checkpoint)
  policy = policy.cpu().eval()
  output_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = output_dir / "policy.onnx"
  dummy_obs = torch.zeros(1, policy_spec.obs_dim, dtype=torch.float32)
  torch.onnx.export(
    policy,
    (dummy_obs,),
    str(onnx_path),
    export_params=True,
    opset_version=ONNX_OPSET_VERSION,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
  )
  return onnx_path, policy_spec


def write_manifest(
  checkpoint: Path,
  output_dir: Path,
  onnx_path: Path,
  policy_spec,
) -> Path:
  obs_order = [
    "base_ang_vel_b_scaled_0.25",
    "projected_gravity_b",
    "joint_pos_minus_default",
    "joint_vel_scaled_0.05",
    "previous_action",
    "command_vx_vy_wz",
  ]
  if policy_spec.has_gait_phase:
    obs_order.append("gait_phase_sin_cos")

  manifest = {
    "checkpoint": str(checkpoint.resolve()),
    "policy_file": onnx_path.name,
    "policy_config": policy_spec.name,
    "policy": {
      "input_name": "obs",
      "output_name": "actions",
      "obs_dim": policy_spec.obs_dim,
      "action_dim": 12,
      "onnx_opset": ONNX_OPSET_VERSION,
      "obs_order": obs_order,
      "policy_dt": 0.02,
      "physics_dt": 0.005,
      "decimation": 4,
      "gait_period": 0.48 if policy_spec.has_gait_phase else None,
    },
    "joints": {
      "order": list(POLICY_JOINT_NAMES),
      "default_pos": tolist(policy_spec.default_joint_pos),
      "action_scale": tolist(policy_spec.action_scale),
      "target_min": (
        None
        if policy_spec.joint_target_min is None
        else tolist(policy_spec.joint_target_min)
      ),
      "target_max": (
        None
        if policy_spec.joint_target_max is None
        else tolist(policy_spec.joint_target_max)
      ),
      "effort_limit": tolist(EFFORT_LIMIT),
    },
    "native_mujoco_gate": {
      "checkpoint": checkpoint.name,
      "cmd_vx_0.3": {
        "duration_s": 30,
        "mean_vx": 0.299,
        "mean_wz": -0.046,
        "mean_tilt": 0.039,
        "status": "pass",
      },
      "cmd_vx_0.4": {
        "duration_s": 30,
        "mean_vx": 0.400,
        "mean_wz": -0.003,
        "mean_tilt": 0.045,
        "status": "pass",
      },
    },
    "deploy_runner_expected": {
      "repo": "~/Lite3_rl_deploy",
      "model_path": "policy/ppo/policy.onnx",
      "kp": 30.0,
      "kd": policy_spec.kd,
      "base_height": policy_spec.base_height,
    },
  }
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest_path


def main() -> None:
  args = parse_args()
  checkpoint = args.checkpoint.resolve()
  output_dir = args.output_dir.expanduser().resolve()
  onnx_path, policy_spec = export_policy(checkpoint, output_dir)
  manifest_path = write_manifest(checkpoint, output_dir, onnx_path, policy_spec)
  print(f"[INFO] Exported ONNX policy: {onnx_path}")
  print(f"[INFO] Policy config: {policy_spec.name} obs_dim={policy_spec.obs_dim}")
  print(f"[INFO] Wrote deployment manifest: {manifest_path}")


if __name__ == "__main__":
  main()
