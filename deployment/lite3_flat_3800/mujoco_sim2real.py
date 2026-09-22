#!/usr/bin/env python3
"""Validate the Lite3 checkpoint in native MuJoCo before hardware deployment.

This is deliberately independent from the MJLab environment and MuJoCo Warp.
It runs the frozen actor in the reference C MuJoCo Python bindings while
reproducing the Lite3 training-time observation, position action, collision,
and actuator settings.  It is the first sim2sim gate for sim2real.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
# model_1050 was trained with gait_phase_clock(period=0.48). This must match
# the saved run configuration, not the helper function's default period.
GAIT_PERIOD = 0.48
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
FOOT_GEOM_NAMES = (
  "FL_FOOT_collision",
  "FR_FOOT_collision",
  "HL_FOOT_collision",
  "HR_FOOT_collision",
)
TRAINING_FOOT_CONTACT_GEOM_NAMES = (
  "FL_FOOT_collision",
  "FR_FOOT_collision",
)
OLD_DEFAULT_JOINT_POS = np.array([0.0, -0.65, 1.3] * 4, dtype=np.float64)
OLD_ACTION_SCALE = np.array([0.10, 0.20, 0.20] * 4, dtype=np.float64)
OLD_JOINT_TARGET_MIN = np.array([-0.20, -1.05, 0.85] * 4, dtype=np.float64)
OLD_JOINT_TARGET_MAX = np.array([0.20, -0.25, 1.75] * 4, dtype=np.float64)
DEEP_DEFAULT_JOINT_POS = np.array([0.0, -0.75, 1.3] * 4, dtype=np.float64)
DEEP_ACTION_SCALE = np.array([0.20, 0.20, 0.30] * 4, dtype=np.float64)
EFFORT_LIMIT = np.array([24.0, 24.0, 36.0] * 4, dtype=np.float64)


@dataclass(frozen=True)
class PolicySpec:
  obs_dim: int
  name: str
  default_joint_pos: np.ndarray
  action_scale: np.ndarray
  joint_target_min: np.ndarray | None
  joint_target_max: np.ndarray | None
  base_height: float
  kd: float
  has_gait_phase: bool


def policy_spec_from_obs_dim(obs_dim: int) -> PolicySpec:
  if obs_dim == 45:
    return PolicySpec(
      obs_dim=45,
      name="deep_lite3",
      default_joint_pos=DEEP_DEFAULT_JOINT_POS,
      action_scale=DEEP_ACTION_SCALE,
      joint_target_min=None,
      joint_target_max=None,
      base_height=0.35,
      kd=3.0,
      has_gait_phase=False,
    )
  if obs_dim == 47:
    return PolicySpec(
      obs_dim=47,
      name="legacy_gait_phase",
      default_joint_pos=OLD_DEFAULT_JOINT_POS,
      action_scale=OLD_ACTION_SCALE,
      joint_target_min=OLD_JOINT_TARGET_MIN,
      joint_target_max=OLD_JOINT_TARGET_MAX,
      base_height=0.375,
      kd=1.0,
      has_gait_phase=True,
    )
  raise ValueError(f"Unsupported actor observation dimension: {obs_dim}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--xml",
    type=Path,
    default=SCRIPT_DIR / "robot" / "Lite3.xml",
    help="Full Lite3 deployment MJCF, including the flat terrain plane.",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=SCRIPT_DIR / "model_3800.pt",
    help="RSL-RL checkpoint containing actor_state_dict.",
  )
  parser.add_argument("--cmd-vx", type=float, default=0.4)
  parser.add_argument("--cmd-vy", type=float, default=0.0)
  parser.add_argument("--cmd-wz", type=float, default=0.0)
  parser.add_argument("--duration-s", type=float, default=20.0)
  parser.add_argument("--warmup-s", type=float, default=1.0)
  parser.add_argument(
    "--base-height",
    type=float,
    default=None,
    help="Initial base height. Default is inferred from the checkpoint config.",
  )
  parser.add_argument("--kp", type=float, default=30.0)
  parser.add_argument(
    "--kd",
    type=float,
    default=None,
    help="Walking PD damping. Default is inferred from the checkpoint config.",
  )
  parser.add_argument(
    "--control-latency-steps",
    type=int,
    default=0,
    help=(
      "Delay position targets by this many 5 ms physics steps. Use 1 to emulate "
      "the minimum asynchronous UDP control latency."
    ),
  )
  parser.add_argument(
    "--stand-kp",
    type=float,
    default=80.0,
    help="Position gain used by the default-pose standing PD state.",
  )
  parser.add_argument(
    "--stand-kd",
    type=float,
    default=2.0,
    help="Damping gain used by the default-pose standing PD state.",
  )
  parser.add_argument(
    "--stand-threshold",
    type=float,
    default=0.25,
    help=(
      "Use default-pose standing PD instead of the RL policy when the planar "
      "command magnitude is below this threshold."
    ),
  )
  parser.add_argument(
    "--min-policy-vx",
    type=float,
    default=0.3,
    help=(
      "Minimum absolute vx command sent to the walking policy after the standing "
      "gate opens. Set to 0 to disable command clamping."
    ),
  )
  parser.add_argument(
    "--disable-stand-gate",
    action="store_true",
    help="Always send commands to the RL policy, including low-speed commands.",
  )
  parser.add_argument(
    "--all-feet-contact",
    action="store_true",
    help=(
      "Configure all four feet as high-friction foot contacts. Default reproduces "
      "the current checkpoint's MJLab training regex, which only matched FL/FR."
    ),
  )
  parser.add_argument(
    "--clip-actions",
    type=float,
    default=None,
    help="Optional policy-action clamp. Default matches the checkpoint: disabled.",
  )
  parser.add_argument(
    "--no-joint-target-clip",
    action="store_true",
    help="Disable the Lite3 joint target clamp used by the training action term.",
  )
  parser.add_argument(
    "--action-trace",
    type=Path,
    help="Optional .npz file with an (N, 12) raw policy action trace to replay.",
  )
  parser.add_argument(
    "--save-actions",
    type=Path,
    help="Optional .npz output for raw policy actions, one row per 50 Hz step.",
  )
  parser.add_argument(
    "--save-state-trace",
    type=Path,
    help="Optional .npz output path for sampled MuJoCo root/joint/contact states.",
  )
  parser.add_argument(
    "--viewer",
    action="store_true",
    help="Open MuJoCo's native viewer instead of running headless.",
  )
  return parser.parse_args()


def load_policy(checkpoint_path: Path) -> tuple[nn.Module, PolicySpec]:
  """Reconstruct the deterministic actor and infer its training config."""
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  actor_state = checkpoint.get("actor_state_dict")
  if not isinstance(actor_state, dict):
    raise KeyError(f"No actor_state_dict in checkpoint: {checkpoint_path}")
  first_weight = actor_state.get("mlp.0.weight")
  if first_weight is None or first_weight.ndim != 2:
    raise KeyError(f"No mlp.0.weight in checkpoint: {checkpoint_path}")
  spec = policy_spec_from_obs_dim(int(first_weight.shape[1]))
  policy = nn.Sequential(
    nn.Linear(spec.obs_dim, 512),
    nn.ELU(),
    nn.Linear(512, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 12),
  )
  policy.load_state_dict(
    {
      name.removeprefix("mlp."): value
      for name, value in actor_state.items()
      if name.startswith("mlp.")
    },
    strict=True,
  )
  return policy.eval(), spec


def joint_addresses(
  model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return qpos, dof, and actuator addresses in policy joint order."""
  joint_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in POLICY_JOINT_NAMES
    ]
  )
  if np.any(joint_ids < 0):
    missing = [
      name
      for name, joint_id in zip(POLICY_JOINT_NAMES, joint_ids, strict=True)
      if joint_id < 0
    ]
    raise ValueError(f"MJCF is missing policy joints: {missing}")
  actuator_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      for name in POLICY_JOINT_NAMES
    ]
  )
  if np.any(actuator_ids < 0):
    actuator_matches = [
      np.flatnonzero(model.actuator_trnid[:, 0] == joint_id) for joint_id in joint_ids
    ]
    if any(matches.shape != (1,) for matches in actuator_matches):
      raise ValueError(
        "Expected either policy-named position actuators or exactly one actuator "
        "per policy joint; "
        f"got matches={[matches.tolist() for matches in actuator_matches]}"
      )
    actuator_ids = np.array([matches.item() for matches in actuator_matches])
  return (
    model.jnt_qposadr[joint_ids],
    model.jnt_dofadr[joint_ids],
    actuator_ids,
  )


def geom_ids(model: mujoco.MjModel, geom_names: tuple[str, ...]) -> np.ndarray:
  ids = np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geom_names]
  )
  if np.any(ids < 0):
    missing = [
      name for name, geom_id in zip(geom_names, ids, strict=True) if geom_id < 0
    ]
    raise ValueError(f"MJCF is missing geoms: {missing}")
  return ids


def add_training_position_actuators(spec: mujoco.MjSpec) -> None:
  """Append the same policy position actuators that MJLab adds at build time."""
  for joint_name, effort_limit in zip(POLICY_JOINT_NAMES, EFFORT_LIMIT, strict=True):
    actuator = spec.add_actuator(name=joint_name, target=joint_name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm[0] = 30.0
    actuator.biasprm[1] = -30.0
    actuator.biasprm[2] = -1.0
    actuator.inheritrange = 0.0
    actuator.ctrllimited = False
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit])


def load_training_model(xml_path: Path) -> mujoco.MjModel:
  """Load XML and append MJLab-style training position actuators before compile."""
  spec = mujoco.MjSpec.from_file(str(xml_path))
  add_training_position_actuators(spec)
  return spec.compile()


def configure_training_physics(
  model: mujoco.MjModel,
  actuator_address: np.ndarray,
  args: argparse.Namespace,
  kp: float,
  kd: float,
) -> None:
  """Apply the MJLab training configuration to the standalone MjModel."""
  model.opt.timestep = PHYSICS_DT
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.impratio = 10.0
  model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  model.opt.ccd_iterations = 500

  # The source XML stores portable torque motors. MJLab replaces them with
  # native MuJoCo <position> actuators when building the training entity.
  # Recreate the same implicit affine PD law in this standalone model.
  model.actuator_gaintype[actuator_address] = mujoco.mjtGain.mjGAIN_FIXED
  model.actuator_biastype[actuator_address] = mujoco.mjtBias.mjBIAS_AFFINE
  model.actuator_gainprm[actuator_address] = 0.0
  model.actuator_biasprm[actuator_address] = 0.0
  model.actuator_gainprm[actuator_address, 0] = kp
  model.actuator_biasprm[actuator_address, 1:3] = (-kp, -kd)
  model.actuator_ctrllimited[actuator_address] = 0
  model.actuator_forcelimited[actuator_address] = 1
  model.actuator_forcerange[actuator_address, 0] = -EFFORT_LIMIT
  model.actuator_forcerange[actuator_address, 1] = EFFORT_LIMIT

  foot_contact_names = (
    FOOT_GEOM_NAMES if args.all_feet_contact else TRAINING_FOOT_CONTACT_GEOM_NAMES
  )
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name is None or not name.endswith("_collision"):
      continue
    model.geom_contype[geom_id] = 1
    model.geom_conaffinity[geom_id] = 1
    model.geom_solref[geom_id] = (0.01, 1.0)
    if name in foot_contact_names:
      model.geom_condim[geom_id] = 6
      model.geom_priority[geom_id] = 1
      model.geom_friction[geom_id] = (1.0, 5.0e-3, 5.0e-4)
    else:
      model.geom_condim[geom_id] = 1


def reset(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos_address: np.ndarray,
  args: argparse.Namespace,
  policy_spec: PolicySpec,
) -> None:
  mujoco.mj_resetData(model, data)
  base_height = args.base_height if args.base_height is not None else policy_spec.base_height
  data.qpos[:3] = (0.0, 0.0, base_height)
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  data.qpos[qpos_address] = policy_spec.default_joint_pos
  mujoco.mj_forward(model, data)


def body_frame_values(
  model: mujoco.MjModel, data: mujoco.MjData, torso_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Return body-frame angular/linear velocity and projected gravity."""
  rotation_wb = data.xmat[torso_id].reshape(3, 3)
  rotation_bw = rotation_wb.T
  angular_velocity_b = rotation_bw @ data.cvel[torso_id, :3]
  linear_velocity_b = rotation_bw @ data.cvel[torso_id, 3:]
  gravity_b = rotation_bw @ model.opt.gravity
  gravity_b /= np.linalg.norm(model.opt.gravity)
  return angular_velocity_b, linear_velocity_b, gravity_b, rotation_bw


def build_observation(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  torso_id: int,
  gyro_address: slice,
  qpos_address: np.ndarray,
  dof_address: np.ndarray,
  previous_action: np.ndarray,
  command: np.ndarray,
  policy_step: int,
  policy_spec: PolicySpec,
) -> np.ndarray:
  _, _, gravity_b, _ = body_frame_values(model, data, torso_id)
  angular_velocity_b = data.sensordata[gyro_address]
  terms = [
    angular_velocity_b * 0.25,
    gravity_b,
    data.qpos[qpos_address] - policy_spec.default_joint_pos,
    data.qvel[dof_address] * 0.05,
    previous_action,
    command,
  ]
  if policy_spec.has_gait_phase:
    angle = 2.0 * math.pi * (policy_step * POLICY_DT / GAIT_PERIOD)
    terms.append(np.array((math.sin(angle), math.cos(angle))))
  observation = np.concatenate(tuple(terms)).astype(np.float32)
  if observation.shape != (policy_spec.obs_dim,) or not np.isfinite(observation).all():
    raise RuntimeError(f"Invalid policy observation: shape={observation.shape}")
  return observation


def apply_position_control(
  data: mujoco.MjData,
  actuator_address: np.ndarray,
  target: np.ndarray,
) -> None:
  data.ctrl[actuator_address] = target


def foot_contact_forces(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  foot_geom_ids: np.ndarray,
) -> np.ndarray:
  """Return per-foot MuJoCo contact normal force in FOOT_GEOM_NAMES order."""
  normal = np.zeros(len(foot_geom_ids), dtype=np.float64)
  for contact_index in range(data.ncon):
    contact = data.contact[contact_index]
    geom1_matches = np.flatnonzero(foot_geom_ids == contact.geom1)
    geom2_matches = np.flatnonzero(foot_geom_ids == contact.geom2)
    if geom1_matches.size == 0 and geom2_matches.size == 0:
      continue
    foot_index = int(geom1_matches[0]) if geom1_matches.size else int(geom2_matches[0])
    wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_index, wrench)
    normal[foot_index] += wrench[0]
  return normal


def run(args: argparse.Namespace) -> None:
  if not args.xml.is_file():
    raise FileNotFoundError(f"MJCF not found: {args.xml}")
  if args.clip_actions is not None and args.clip_actions <= 0.0:
    raise ValueError("--clip-actions must be positive")
  if args.stand_threshold < 0.0:
    raise ValueError("--stand-threshold must be non-negative")
  if args.min_policy_vx < 0.0:
    raise ValueError("--min-policy-vx must be non-negative")
  if args.control_latency_steps < 0:
    raise ValueError("--control-latency-steps must be non-negative")
  if args.stand_kp <= 0.0:
    raise ValueError("--stand-kp must be positive")
  if args.stand_kd < 0.0:
    raise ValueError("--stand-kd must be non-negative")

  user_command = np.array((args.cmd_vx, args.cmd_vy, args.cmd_wz), dtype=np.float32)
  command_norm = float(np.linalg.norm(user_command[:2]) + abs(user_command[2]))
  standing_mode = not args.disable_stand_gate and command_norm < args.stand_threshold
  policy_command = user_command.copy()
  if standing_mode:
    policy_command[:] = 0.0
  elif args.min_policy_vx > 0.0:
    vx_abs = abs(float(policy_command[0]))
    if 0.0 < vx_abs < args.min_policy_vx:
      policy_command[0] = math.copysign(args.min_policy_vx, float(policy_command[0]))

  model = load_training_model(args.xml)
  data = mujoco.MjData(model)
  qpos_address, dof_address, actuator_address = joint_addresses(model)
  foot_geom_ids = geom_ids(model, FOOT_GEOM_NAMES)
  torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "TORSO")
  if torso_id < 0:
    raise ValueError("MJCF is missing TORSO body")
  gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
  if gyro_id < 0 or model.sensor_dim[gyro_id] != 3:
    raise ValueError("MJCF must contain a three-axis imu_ang_vel gyro sensor")
  gyro_address = slice(
    model.sensor_adr[gyro_id], model.sensor_adr[gyro_id] + model.sensor_dim[gyro_id]
  )
  action_trace: np.ndarray | None = None
  if args.action_trace is None:
    policy, policy_spec = load_policy(args.checkpoint)
  else:
    policy_spec = policy_spec_from_obs_dim(45)
    trace_data = np.load(args.action_trace)
    action_trace = trace_data["actions"]
    if action_trace.ndim != 2 or action_trace.shape[1] != 12:
      raise ValueError(
        f"Expected action trace with shape (N, 12), got {action_trace.shape}."
      )
    policy = None
  walking_kd = args.kd if args.kd is not None else policy_spec.kd
  drive_kp = args.stand_kp if standing_mode else args.kp
  drive_kd = args.stand_kd if standing_mode else walking_kd
  configure_training_physics(model, actuator_address, args, drive_kp, drive_kd)
  previous_action = np.zeros(12, dtype=np.float32)
  target = policy_spec.default_joint_pos.copy()
  delayed_targets = deque(
    (target.copy() for _ in range(args.control_latency_steps)),
  )
  reset(model, data, qpos_address, args, policy_spec)
  _, initial_root_vel_b, _, _ = body_frame_values(model, data, torso_id)
  initial_root_pos = data.qpos[:3].copy()
  initial_joint_pos = data.qpos[qpos_address].copy()
  initial_joint_vel = data.qvel[dof_address].copy()

  warmup_steps = round(args.warmup_s / PHYSICS_DT)
  duration_steps = round(args.duration_s / PHYSICS_DT)
  policy_step = 0
  samples: list[np.ndarray] = []
  action_samples: list[np.ndarray] = []
  raw_actions: list[np.ndarray] = []
  state_root_pos: list[np.ndarray] = []
  state_root_vel_b: list[np.ndarray] = []
  state_joint_pos: list[np.ndarray] = []
  state_target: list[np.ndarray] = []
  state_torque: list[np.ndarray] = []
  state_foot_force: list[np.ndarray] = []
  print(
    "[INFO] Native MuJoCo sim2real gate: "
    f"xml={args.xml}, dt={PHYSICS_DT:.4f}, policy_dt={POLICY_DT:.4f}, "
    f"cmd=({args.cmd_vx:.3f}, {args.cmd_vy:.3f}, {args.cmd_wz:.3f})"
  )
  print(
    "[INFO] Deployment state gate: "
    f"{'standing PD' if standing_mode else 'RL walking policy'}, "
    f"stand_threshold={args.stand_threshold:.3f}, "
    f"policy_cmd=({policy_command[0]:.3f}, {policy_command[1]:.3f}, "
    f"{policy_command[2]:.3f})"
  )
  print(
    "[INFO] Training-aligned drive: "
    f"policy={policy_spec.name} obs_dim={policy_spec.obs_dim}, "
    f"base_height={args.base_height if args.base_height is not None else policy_spec.base_height:.3f}, "
    f"Kp={drive_kp:.1f}, Kd={drive_kd:.1f}, effort=(24, 24, 36) Nm per leg, "
    f"clip_actions={args.clip_actions}, "
    f"joint_target_clip={not args.no_joint_target_clip}"
  )
  print(
    "[INFO] Contact mode: "
    f"{'all four feet' if args.all_feet_contact else 'training regex match: FL/FR only'}"
  )
  if args.control_latency_steps:
    print(
      "[INFO] Delaying commanded joint targets by "
      f"{args.control_latency_steps} physics step(s)."
    )
  if action_trace is not None:
    print(f"[INFO] Action replay: {args.action_trace} ({len(action_trace)} steps)")

  viewer = None
  if args.viewer:
    viewer = mujoco.viewer.launch_passive(model, data)

  try:
    for physics_step in range(warmup_steps + duration_steps):
      active_step = physics_step - warmup_steps
      if physics_step >= warmup_steps and active_step % POLICY_DECIMATION == 0:
        if standing_mode:
          raw_action = np.zeros(12, dtype=np.float32)
          action = raw_action
          previous_action = action.copy()
          target = policy_spec.default_joint_pos.copy()
          action_samples.append(np.abs(raw_action))
          raw_actions.append(raw_action.copy())
          policy_step += 1
        elif action_trace is None:
          observation = build_observation(
            model,
            data,
            torso_id,
            gyro_address,
            qpos_address,
            dof_address,
            previous_action,
            policy_command,
            policy_step,
            policy_spec,
          )
          assert policy is not None
          with torch.inference_mode():
            raw_action = (
              policy(torch.from_numpy(observation).unsqueeze(0)).squeeze(0).numpy()
            )
          action = (
            raw_action
            if args.clip_actions is None
            else np.clip(raw_action, -args.clip_actions, args.clip_actions)
          )
          previous_action = action.astype(np.float32, copy=True)
          target = policy_spec.default_joint_pos + action * policy_spec.action_scale
          if (
            not args.no_joint_target_clip
            and policy_spec.joint_target_min is not None
            and policy_spec.joint_target_max is not None
          ):
            target = np.clip(
              target, policy_spec.joint_target_min, policy_spec.joint_target_max
            )
          action_samples.append(np.abs(raw_action))
          raw_actions.append(raw_action.astype(np.float32, copy=True))
          policy_step += 1
        else:
          if policy_step >= len(action_trace):
            break
          raw_action = action_trace[policy_step]
          action = (
            raw_action
            if args.clip_actions is None
            else np.clip(raw_action, -args.clip_actions, args.clip_actions)
          )
          previous_action = action.astype(np.float32, copy=True)
          target = policy_spec.default_joint_pos + action * policy_spec.action_scale
          if (
            not args.no_joint_target_clip
            and policy_spec.joint_target_min is not None
            and policy_spec.joint_target_max is not None
          ):
            target = np.clip(
              target, policy_spec.joint_target_min, policy_spec.joint_target_max
            )
          action_samples.append(np.abs(raw_action))
          raw_actions.append(raw_action.astype(np.float32, copy=True))
          policy_step += 1

      if args.control_latency_steps:
        delayed_targets.append(target.copy())
        applied_target = delayed_targets.popleft()
      else:
        applied_target = target
      apply_position_control(data, actuator_address, applied_target)
      mujoco.mj_step(model, data)
      if viewer is not None:
        viewer.sync()

      if physics_step < warmup_steps:
        continue
      angular_velocity_b, linear_velocity_b, gravity_b, _ = body_frame_values(
        model, data, torso_id
      )
      if (
        args.save_state_trace is not None and (active_step + 1) % POLICY_DECIMATION == 0
      ):
        state_root_pos.append(data.qpos[:3].copy())
        state_root_vel_b.append(linear_velocity_b.copy())
        state_joint_pos.append(data.qpos[qpos_address].copy())
        state_target.append(target.copy())
        state_torque.append(data.qfrc_actuator[dof_address].copy())
        state_foot_force.append(foot_contact_forces(model, data, foot_geom_ids))
      samples.append(
        np.array(
          (
            linear_velocity_b[0],
            linear_velocity_b[1],
            angular_velocity_b[2],
            data.qpos[2],
            np.linalg.norm(gravity_b[:2]),
          )
        )
      )
      if active_step % round(1.0 / PHYSICS_DT) == 0:
        print(
          f"[STATE] t={active_step * PHYSICS_DT:6.2f}s z={data.qpos[2]:.3f} "
          f"v_b=({linear_velocity_b[0]:+.3f}, {linear_velocity_b[1]:+.3f}) "
          f"wz={angular_velocity_b[2]:+.3f} tilt={np.linalg.norm(gravity_b[:2]):.3f} "
          f"grav_z={gravity_b[2]:+.3f}"
        )
  finally:
    if viewer is not None:
      viewer.close()

  metric = np.stack(samples)
  action_metric = np.stack(action_samples)
  print(
    "[SUMMARY] "
    f"mean_vx={metric[:, 0].mean():+.3f} mean_vy={metric[:, 1].mean():+.3f} "
    f"mean_wz={metric[:, 2].mean():+.3f} mean_z={metric[:, 3].mean():.3f} "
    f"mean_tilt={metric[:, 4].mean():.3f} samples={len(metric)}"
  )
  print(
    "[SUMMARY_ACTION] "
    f"mean_abs_raw={action_metric.mean():.3f} max_abs_raw={action_metric.max():.3f} "
    f"frac_abs_gt_1.4={(action_metric > 1.4).mean():.3f}"
  )
  if args.save_actions is not None:
    args.save_actions.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save_actions, actions=np.stack(raw_actions), dt=POLICY_DT)
    print(f"[INFO] Saved raw action trace: {args.save_actions}")
  if args.save_state_trace is not None:
    args.save_state_trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
      args.save_state_trace,
      dt=POLICY_DT,
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
      foot_geom_names=np.asarray(FOOT_GEOM_NAMES),
    )
    print(
      f"[INFO] Saved {len(state_root_pos)} MuJoCo state samples to "
      f"{args.save_state_trace}"
    )


if __name__ == "__main__":
  run(parse_args())
