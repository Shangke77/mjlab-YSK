#!/usr/bin/env python3
"""Measure Lite3 foot contact forces in the MuJoCo training model."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
JOINT_NAMES = tuple(
  f"{leg}_{joint}_joint"
  for leg in ("FL", "FR", "HL", "HR")
  for joint in ("HipX", "HipY", "Knee")
)
FOOT_GEOMS = tuple(f"{leg}_FOOT_collision" for leg in ("FL", "FR", "HL", "HR"))
DEFAULT_Q = np.tile((0.0, -0.65, 1.3), 4)
EFFORT_LIMIT = np.tile((24.0, 24.0, 36.0), 4)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, default=SCRIPT_DIR / "robot/Lite3.xml")
  parser.add_argument("--duration-s", type=float, default=3.0)
  parser.add_argument("--kp", type=float, default=80.0)
  parser.add_argument("--kd", type=float, default=2.0)
  args = parser.parse_args()
  model = mujoco.MjModel.from_xml_path(str(args.xml))
  model.opt.timestep = 0.005
  data = mujoco.MjData(model)
  joint_ids = np.asarray(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
  )
  actuator_ids = np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_ctrl")
      for name in JOINT_NAMES
    ]
  )
  qpos_ids = model.jnt_qposadr[joint_ids]
  qvel_ids = model.jnt_dofadr[joint_ids]
  foot_geom_ids = np.asarray(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOMS]
  )
  data.qpos[qpos_ids] = DEFAULT_Q
  mujoco.mj_forward(model, data)
  foot_z = data.geom_xpos[foot_geom_ids, 2]
  data.qpos[2] += 0.022 - np.min(foot_z)
  mujoco.mj_forward(model, data)
  sums = np.zeros(4)
  samples = 0
  for _ in range(round(args.duration_s / model.opt.timestep)):
    effort = args.kp * (DEFAULT_Q - data.qpos[qpos_ids]) - args.kd * data.qvel[qvel_ids]
    data.ctrl[actuator_ids] = np.clip(effort, -EFFORT_LIMIT, EFFORT_LIMIT)
    mujoco.mj_step(model, data)
    normal = np.zeros(4)
    for contact_index in range(data.ncon):
      contact = data.contact[contact_index]
      if contact.geom1 not in foot_geom_ids and contact.geom2 not in foot_geom_ids:
        continue
      foot_index = (
        int(np.where(foot_geom_ids == contact.geom1)[0][0])
        if contact.geom1 in foot_geom_ids
        else int(np.where(foot_geom_ids == contact.geom2)[0][0])
      )
      wrench = np.zeros(6)
      mujoco.mj_contactForce(model, data, contact_index, wrench)
      normal[foot_index] += wrench[0]
    if data.time > 1.0:
      sums += normal
      samples += 1
  print(
    f"[INFO] base_z={data.qpos[2]:.4f} base_vz={data.qvel[2]:+.4f} "
    f"contacts={data.ncon} constraint_fz={data.qfrc_constraint[2]:+.3f} "
    f"gravity_fz={data.qfrc_gravcomp[2]:+.3f}"
  )
  for name, force, geom_id in zip(
    FOOT_GEOMS, sums / samples, foot_geom_ids, strict=True
  ):
    pos = data.geom_xpos[geom_id]
    print(f"[FOOT_CONTACT] {name} z={pos[2]:.4f} normal_force={force:.3f}")


if __name__ == "__main__":
  main()
