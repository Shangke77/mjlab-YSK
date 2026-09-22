#!/usr/bin/env python3
"""Probe Lite3 policy action statistics in the original MJLab/MuJoCo env."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ID = "Mjlab-Velocity-Flat-DeepRobotics-Lite3"
POLICY_JOINT_NAMES = (
  "FL_HipX_joint",
  "FL_HipY_joint",
  "FL_Knee_joint",
  "FR_HipX_joint",
  "FR_HipY_joint",
  "FR_Knee_joint",
  "HL_HipX_joint",
  "HL_HipY_joint",
  "HL_Knee_joint",
  "HR_HipX_joint",
  "HR_HipY_joint",
  "HR_Knee_joint",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=SCRIPT_DIR / "model_3800.pt",
    help="RSL-RL checkpoint to load.",
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--cmd-vx", type=float, default=0.4)
  parser.add_argument("--cmd-vy", type=float, default=0.0)
  parser.add_argument("--cmd-wz", type=float, default=0.0)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--save-actions",
    type=Path,
    help="Optional .npz path for the first environment's raw policy action trace.",
  )
  parser.add_argument(
    "--save-state-trace",
    type=Path,
    help="Optional .npz path for the first environment's MJLab state trace.",
  )
  return parser.parse_args()


def format_vector(values: torch.Tensor) -> str:
  return "[" + ", ".join(f"{value:+.3f}" for value in values.tolist()) + "]"


def configure_fixed_command(cfg, args: argparse.Namespace) -> None:
  """Configure the Lite3 play env to use one fixed body-frame velocity command."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.resampling_time_range = (1.0e9, 1.0e9)
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_world_envs = 0.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.init_velocity_prob = 0.0
  twist_cmd.ranges.lin_vel_x = (args.cmd_vx, args.cmd_vx)
  twist_cmd.ranges.lin_vel_y = (args.cmd_vy, args.cmd_vy)
  twist_cmd.ranges.ang_vel_z = (args.cmd_wz, args.cmd_wz)


def configure_deterministic_deployment_reset(cfg) -> None:
  """Match the standalone MuJoCo deployment reset and startup friction."""
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }
  reset_base.params["velocity_range"] = {}
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)


def print_action_stats(actions: torch.Tensor) -> None:
  abs_actions = actions.abs()
  flat = abs_actions.flatten()
  print("[ACTION_STATS]")
  print(f"  samples={actions.shape[0]} action_dim={actions.shape[1]}")
  print(f"  mean_abs={flat.mean().item():.4f}")
  print(f"  p95_abs={flat.quantile(0.95).item():.4f}")
  print(f"  p99_abs={flat.quantile(0.99).item():.4f}")
  print(f"  max_abs={flat.max().item():.4f}")
  for threshold in (1.0, 1.2, 1.4, 1.6):
    frac = (flat > threshold).float().mean().item()
    print(f"  frac_abs_gt_{threshold:.1f}={frac:.4f}")
  print(f"  per_joint_max_abs={format_vector(abs_actions.max(dim=0).values.cpu())}")
  print(
    "  per_joint_frac_abs_gt_1="
    f"{format_vector((abs_actions > 1.0).float().mean(dim=0).cpu())}"
  )


def print_motion_stats(lin_vel_b: torch.Tensor, ang_vel_b: torch.Tensor) -> None:
  print("[MOTION_STATS]")
  print(f"  mean_vx={lin_vel_b[:, 0].mean().item():+.4f}")
  print(f"  mean_vy={lin_vel_b[:, 1].mean().item():+.4f}")
  print(f"  mean_wz={ang_vel_b[:, 2].mean().item():+.4f}")
  print(f"  p50_vx={lin_vel_b[:, 0].quantile(0.50).item():+.4f}")
  print(f"  p10_vx={lin_vel_b[:, 0].quantile(0.10).item():+.4f}")
  print(f"  p90_vx={lin_vel_b[:, 0].quantile(0.90).item():+.4f}")


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = args.num_envs
  configure_fixed_command(env_cfg, args)
  configure_deterministic_deployment_reset(env_cfg)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint.resolve()),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  obs, _ = wrapped.reset()
  robot = wrapped.unwrapped.scene["robot"]
  action_term = wrapped.unwrapped.action_manager.get_term("joint_pos")
  if not isinstance(action_term, JointPositionAction):
    raise TypeError(f"Expected joint_pos action term, got {type(action_term).__name__}")
  if tuple(action_term.target_names) != POLICY_JOINT_NAMES:
    raise RuntimeError(
      "MJLab action joint order does not match the policy.\n"
      f"Expected: {POLICY_JOINT_NAMES}\n"
      f"Found:    {tuple(action_term.target_names)}"
    )
  joint_ids = action_term.target_ids
  initial_root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
  initial_root_vel_b = robot.data.root_link_lin_vel_b[0].detach().cpu().numpy()
  initial_joint_pos = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
  initial_joint_vel = robot.data.joint_vel[0, joint_ids].detach().cpu().numpy()
  actions_log: list[torch.Tensor] = []
  lin_vel_log: list[torch.Tensor] = []
  ang_vel_log: list[torch.Tensor] = []
  state_root_pos: list[np.ndarray] = []
  state_root_vel_b: list[np.ndarray] = []
  state_joint_pos: list[np.ndarray] = []
  state_target: list[np.ndarray] = []
  state_torque: list[np.ndarray] = []
  state_foot_force: list[np.ndarray] = []

  feet_contact = wrapped.unwrapped.scene["feet_ground_contact"]
  if not isinstance(feet_contact, ContactSensor):
    raise TypeError(
      f"Expected feet_ground_contact sensor, got {type(feet_contact).__name__}"
    )
  command = wrapped.unwrapped.command_manager.get_term("twist")
  if not isinstance(command, UniformVelocityCommand):
    raise TypeError(f"Expected twist command term, got {type(command).__name__}")
  print(
    "[INFO] MuJoCo probe "
    f"num_envs={args.num_envs} steps={args.steps} "
    f"cmd=({args.cmd_vx:.3f}, {args.cmd_vy:.3f}, {args.cmd_wz:.3f}) "
    f"device={args.device}"
  )

  with torch.inference_mode():
    for _ in range(args.steps):
      # Keep the command fixed even if a reset/resample happens.
      command.vel_command_b[:, 0] = args.cmd_vx
      command.vel_command_b[:, 1] = args.cmd_vy
      command.vel_command_b[:, 2] = args.cmd_wz
      actions = policy(obs)
      actions_log.append(actions.detach().cpu())
      lin_vel_log.append(robot.data.root_link_lin_vel_b.detach().cpu())
      ang_vel_log.append(robot.data.root_link_ang_vel_b.detach().cpu())
      obs, _, _, _ = wrapped.step(actions)
      if args.save_state_trace is not None:
        state_root_pos.append(robot.data.root_link_pos_w[0].detach().cpu().numpy())
        state_root_vel_b.append(
          robot.data.root_link_lin_vel_b[0].detach().cpu().numpy()
        )
        state_joint_pos.append(
          robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        )
        state_target.append(
          robot.data.joint_pos_target[0, joint_ids].detach().cpu().numpy()
        )
        state_torque.append(
          robot.data.qfrc_actuator[0, joint_ids].detach().cpu().numpy()
        )
        assert feet_contact.data.force is not None
        state_foot_force.append(
          feet_contact.data.force[0, :, 2].detach().cpu().numpy()
        )

  print_action_stats(torch.cat(actions_log, dim=0))
  print_motion_stats(torch.cat(lin_vel_log, dim=0), torch.cat(ang_vel_log, dim=0))
  if args.save_actions is not None:
    action_trace = torch.stack(actions_log, dim=0)[:, 0].numpy()
    args.save_actions.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save_actions, actions=action_trace, dt=env.step_dt)
    print(
      f"[INFO] Saved {action_trace.shape[0]} raw policy actions to {args.save_actions}"
    )
  if args.save_state_trace is not None:
    args.save_state_trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
      args.save_state_trace,
      dt=env.step_dt,
      root_pos=np.asarray(state_root_pos),
      root_vel_b=np.asarray(state_root_vel_b),
      joint_pos=np.asarray(state_joint_pos),
      joint_target=np.asarray(state_target),
      torque=np.asarray(state_torque),
      foot_normal_force=np.asarray(state_foot_force),
      initial_root_pos=initial_root_pos,
      initial_root_vel_b=initial_root_vel_b,
      initial_joint_pos=initial_joint_pos,
      initial_joint_vel=initial_joint_vel,
      joint_names=np.asarray(POLICY_JOINT_NAMES),
      foot_geom_names=np.asarray(feet_contact.primary_names),
    )
    print(
      f"[INFO] Saved {len(state_root_pos)} MJLab state samples to "
      f"{args.save_state_trace}"
    )
  wrapped.close()


if __name__ == "__main__":
  main()
