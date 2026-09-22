#!/usr/bin/env python3
"""Record a contact-free Lite3 PD step response in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
JOINT_NAMES = (
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
DEFAULT_Q = np.tile((0.0, -0.65, 1.3), 4)
TARGET_DELTA = np.tile((0.10, -0.15, 0.20), 4)
EFFORT_LIMIT = np.tile((24.0, 24.0, 36.0), 4)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, default=SCRIPT_DIR / "robot/Lite3.xml")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--dt", type=float, default=0.005)
  parser.add_argument("--duration-s", type=float, default=0.5)
  parser.add_argument("--kp", type=float, default=30.0)
  parser.add_argument("--kd", type=float, default=1.0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.dt <= 0.0 or args.duration_s <= 0.0:
    raise ValueError("--dt and --duration-s must be positive.")

  model = mujoco.MjModel.from_xml_path(str(args.xml))
  model.opt.timestep = args.dt
  model.opt.gravity[:] = 0.0
  model.geom_contype[:] = 0
  model.geom_conaffinity[:] = 0
  model.actuator_ctrlrange[:, 0] = -EFFORT_LIMIT
  model.actuator_ctrlrange[:, 1] = EFFORT_LIMIT
  data = mujoco.MjData(model)

  joint_ids = np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
  )
  actuator_ids = np.array(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_ctrl")
      for name in JOINT_NAMES
    ]
  )
  if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
    raise RuntimeError("Could not resolve every Lite3 joint/actuator in the MJCF.")
  qpos_ids = model.jnt_qposadr[joint_ids]
  qvel_ids = model.jnt_dofadr[joint_ids]
  data.qpos[qpos_ids] = DEFAULT_Q
  data.qvel[qvel_ids] = 0.0
  mujoco.mj_forward(model, data)

  steps = round(args.duration_s / args.dt)
  q = np.empty((steps, 12), dtype=np.float64)
  qvel = np.empty_like(q)
  torque = np.empty_like(q)
  target = DEFAULT_Q + TARGET_DELTA
  for step in range(steps):
    position = data.qpos[qpos_ids]
    velocity = data.qvel[qvel_ids]
    command = np.clip(
      args.kp * (target - position) - args.kd * velocity, -EFFORT_LIMIT, EFFORT_LIMIT
    )
    data.ctrl[actuator_ids] = command
    mujoco.mj_step(model, data)
    q[step] = data.qpos[qpos_ids]
    qvel[step] = data.qvel[qvel_ids]
    torque[step] = data.actuator_force[actuator_ids]

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    args.output,
    dt=args.dt,
    kp=args.kp,
    kd=args.kd,
    q=q,
    qvel=qvel,
    torque=torque,
    target=target,
    joint_names=np.asarray(JOINT_NAMES),
  )
  print(f"[RESULT] Wrote {steps} MuJoCo contact-free steps to {args.output}")
  print(f"[RESULT] final_q={np.array2string(q[-1], precision=4)}")


if __name__ == "__main__":
  main()
