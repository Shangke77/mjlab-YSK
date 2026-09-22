#!/usr/bin/env python3
"""Compare MJLab's compiled Lite3 MuJoCo model with the standalone model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_action_probe import (  # noqa: E402
  TASK_ID,
  configure_deterministic_deployment_reset,
  configure_fixed_command,
)
from mujoco_sim2real import (  # noqa: E402
  POLICY_JOINT_NAMES,
  configure_training_physics,
  joint_addresses,
  load_training_model,
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, default=SCRIPT_DIR / "robot" / "Lite3.xml")
  parser.add_argument("--cmd-vx", type=float, default=0.4)
  parser.add_argument("--cmd-vy", type=float, default=0.0)
  parser.add_argument("--cmd-wz", type=float, default=0.0)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--kp", type=float, default=30.0)
  parser.add_argument("--kd", type=float, default=1.0)
  parser.add_argument("--all-feet-contact", action="store_true")
  return parser.parse_args()


def actuator_rows(model: mujoco.MjModel) -> list[str]:
  rows = []
  for actuator_id in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    rows.append(
      f"{actuator_id:02d} {name} joint={joint_name} "
      f"gain={model.actuator_gainprm[actuator_id, 0]:+.3f} "
      f"bias=({model.actuator_biasprm[actuator_id, 0]:+.3f},"
      f"{model.actuator_biasprm[actuator_id, 1]:+.3f},"
      f"{model.actuator_biasprm[actuator_id, 2]:+.3f}) "
      f"ctrlimited={int(model.actuator_ctrllimited[actuator_id])} "
      f"forcelimited={int(model.actuator_forcelimited[actuator_id])} "
      f"forcerange={model.actuator_forcerange[actuator_id].tolist()}"
    )
  return rows


def print_section(title: str, rows: list[str]) -> None:
  print(f"[{title}]")
  for row in rows:
    print(f"  {row}")


def policy_joint_dof_values(model: mujoco.MjModel, field: str) -> np.ndarray:
  joint_ids = np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in POLICY_JOINT_NAMES
    ]
  )
  dof_ids = model.jnt_dofadr[joint_ids]
  return getattr(model, field)[dof_ids]


def geom_summary(model: mujoco.MjModel) -> list[str]:
  rows = []
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name is None or not name.endswith("_collision"):
      continue
    rows.append(
      f"{name}: contype={model.geom_contype[geom_id]} "
      f"conaffinity={model.geom_conaffinity[geom_id]} "
      f"condim={model.geom_condim[geom_id]} "
      f"priority={model.geom_priority[geom_id]} "
      f"friction={model.geom_friction[geom_id].tolist()} "
      f"solref={model.geom_solref[geom_id].tolist()}"
    )
  return rows


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  configure_fixed_command(env_cfg, args)
  configure_deterministic_deployment_reset(env_cfg)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  mjlab_model = env.sim.mj_model

  native_model = load_training_model(args.xml)
  qpos_address, dof_address, actuator_address = joint_addresses(native_model)
  del qpos_address, dof_address
  configure_training_physics(native_model, actuator_address, args)

  print(
    "[MODEL_SIZE] "
    f"mjlab: nq={mjlab_model.nq} nv={mjlab_model.nv} nu={mjlab_model.nu} "
    f"ngeom={mjlab_model.ngeom}; "
    f"native: nq={native_model.nq} nv={native_model.nv} nu={native_model.nu} "
    f"ngeom={native_model.ngeom}"
  )
  print(
    "[OPT] "
    f"mjlab dt={mjlab_model.opt.timestep} integrator={int(mjlab_model.opt.integrator)} "
    f"cone={int(mjlab_model.opt.cone)} impratio={mjlab_model.opt.impratio}; "
    f"native dt={native_model.opt.timestep} "
    f"integrator={int(native_model.opt.integrator)} "
    f"cone={int(native_model.opt.cone)} impratio={native_model.opt.impratio}"
  )

  print_section("MJLAB_ACTUATORS", actuator_rows(mjlab_model))
  print_section("NATIVE_ACTUATORS", actuator_rows(native_model))

  for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
    mjlab = policy_joint_dof_values(mjlab_model, field)
    native = policy_joint_dof_values(native_model, field)
    print(
      f"[{field}] max_abs_diff={np.max(np.abs(native - mjlab)):.6g} "
      f"mjlab={mjlab.tolist()} native={native.tolist()}"
    )

  print_section("MJLAB_COLLISIONS", geom_summary(mjlab_model))
  print_section("NATIVE_COLLISIONS", geom_summary(native_model))
  env.close()


if __name__ == "__main__":
  main()
