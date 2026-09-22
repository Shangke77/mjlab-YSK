#!/usr/bin/env python3
"""Bridge the official Lite3 C++ controller to the validated native MuJoCo model.

Run this in one terminal, then run ``build_x86_sim/rl_deploy`` from the
official Lite3_rl_deploy checkout in another. The bridge implements the UDP
packets expected by ``SimulationInterface``:

* receive 12 x (Kp, target position, Kd, target velocity, feed-forward torque)
  on port 20001;
* send a double timestamp followed by 45 float state values on port 30010.

Unlike the upstream simulation scripts, this uses the flat deployment
``robot/Lite3.xml`` and the same native MuJoCo actuator/contact configuration
that model_1050 passed during sim2sim validation.
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from mujoco_sim2real import (
  DEFAULT_JOINT_POS,
  EFFORT_LIMIT,
  JOINT_TARGET_MAX,
  JOINT_TARGET_MIN,
  PHYSICS_DT,
  body_frame_values,
  configure_training_physics,
  joint_addresses,
  load_training_model,
  reset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COMMAND_FLOATS = 12 * 5
COMMAND_PACKET_BYTES = COMMAND_FLOATS * np.dtype("<f4").itemsize
STATE_PACKET_FORMAT = "<d45f"
STATE_PACKET_BYTES = struct.calcsize(STATE_PACKET_FORMAT)


@dataclass
class JointCommand:
  kp: np.ndarray
  target: np.ndarray
  kd: np.ndarray
  target_velocity: np.ndarray
  feedforward_torque: np.ndarray
  received_at: float = float("-inf")

  @classmethod
  def standing(cls) -> JointCommand:
    return cls(
      kp=np.full(12, 80.0, dtype=np.float64),
      target=DEFAULT_JOINT_POS.copy(),
      kd=np.full(12, 2.0, dtype=np.float64),
      target_velocity=np.zeros(12, dtype=np.float64),
      feedforward_torque=np.zeros(12, dtype=np.float64),
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--xml",
    type=Path,
    default=SCRIPT_DIR / "robot" / "Lite3.xml",
    help="Validated flat Lite3 MJCF.",
  )
  parser.add_argument("--base-height", type=float, default=0.375)
  parser.add_argument("--duration-s", type=float, default=60.0)
  parser.add_argument("--command-port", type=int, default=20001)
  parser.add_argument("--state-port", type=int, default=30010)
  parser.add_argument("--state-host", default="127.0.0.1")
  parser.add_argument("--command-timeout-s", type=float, default=0.1)
  parser.add_argument("--all-feet-contact", action="store_true", default=True)
  parser.add_argument("--viewer", action="store_true")
  parser.add_argument(
    "--no-realtime",
    action="store_true",
    help="Run as fast as possible instead of pacing the UDP control loop at 200 Hz.",
  )
  return parser.parse_args()


def rotation_to_rpy(rotation_wb: np.ndarray) -> np.ndarray:
  """Return the ZYX roll/pitch/yaw used by the official C++ RpyToRm helper."""
  pitch = math.asin(float(np.clip(-rotation_wb[2, 0], -1.0, 1.0)))
  roll = math.atan2(float(rotation_wb[2, 1]), float(rotation_wb[2, 2]))
  yaw = math.atan2(float(rotation_wb[1, 0]), float(rotation_wb[0, 0]))
  return np.array((roll, pitch, yaw), dtype=np.float32)


def open_command_socket(port: int) -> socket.socket:
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind(("127.0.0.1", port))
  sock.setblocking(False)
  return sock


def read_latest_command(sock: socket.socket, command: JointCommand) -> JointCommand:
  while True:
    try:
      packet, _ = sock.recvfrom(COMMAND_PACKET_BYTES)
    except BlockingIOError:
      return command
    if len(packet) != COMMAND_PACKET_BYTES:
      print(f"[WARN] Ignoring UDP command with {len(packet)} bytes.")
      continue
    fields = np.frombuffer(packet, dtype="<f4", count=COMMAND_FLOATS)
    command = JointCommand(
      kp=fields[0:12].astype(np.float64),
      target=fields[12:24].astype(np.float64),
      kd=fields[24:36].astype(np.float64),
      target_velocity=fields[36:48].astype(np.float64),
      feedforward_torque=fields[48:60].astype(np.float64),
      received_at=time.monotonic(),
    )


def is_idle_command(command: JointCommand) -> bool:
  return bool(np.max(np.abs(command.kp)) < 1.0e-6)


def sanitized_command(command: JointCommand) -> JointCommand:
  return JointCommand(
    kp=np.clip(command.kp, 0.0, 100.0),
    target=np.clip(command.target, JOINT_TARGET_MIN, JOINT_TARGET_MAX),
    kd=np.clip(command.kd, 0.0, 3.0),
    target_velocity=np.clip(command.target_velocity, -30.0, 30.0),
    feedforward_torque=np.clip(command.feedforward_torque, -EFFORT_LIMIT, EFFORT_LIMIT),
    received_at=command.received_at,
  )


def apply_command(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  actuator_address: np.ndarray,
  dof_address: np.ndarray,
  command: JointCommand,
) -> None:
  model.actuator_gainprm[actuator_address, 0] = command.kp
  model.actuator_biasprm[actuator_address, 1] = -command.kp
  model.actuator_biasprm[actuator_address, 2] = -command.kd
  data.ctrl[actuator_address] = command.target
  data.qfrc_applied[dof_address] = command.feedforward_torque


def state_packet(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  torso_id: int,
  gyro_address: slice,
  accel_address: slice,
  qpos_address: np.ndarray,
  dof_address: np.ndarray,
  actuator_address: np.ndarray,
) -> bytes:
  _, _, _, rotation_bw = body_frame_values(model, data, torso_id)
  rotation_wb = rotation_bw.T
  rpy = rotation_to_rpy(rotation_wb)
  omega = data.sensordata[gyro_address].astype(np.float32)
  acceleration = data.sensordata[accel_address].astype(np.float32)
  joint_pos = data.qpos[qpos_address].astype(np.float32)
  joint_vel = data.qvel[dof_address].astype(np.float32)
  torque = data.actuator_force[actuator_address].astype(np.float32)
  fields = np.concatenate((rpy, acceleration, omega, joint_pos, joint_vel, torque))
  if fields.shape != (45,):
    raise RuntimeError(f"Invalid official state payload shape: {fields.shape}")
  return struct.pack(STATE_PACKET_FORMAT, float(data.time), *fields.tolist())


def sensor_address(model: mujoco.MjModel, name: str) -> slice:
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  if sensor_id < 0:
    raise ValueError(f"MJCF is missing sensor: {name}")
  dimension = int(model.sensor_dim[sensor_id])
  if dimension != 3:
    raise ValueError(
      f"Expected a three-axis sensor for {name}, got dimension={dimension}"
    )
  start = int(model.sensor_adr[sensor_id])
  return slice(start, start + dimension)


def run(args: argparse.Namespace) -> None:
  if not args.xml.is_file():
    raise FileNotFoundError(f"MJCF not found: {args.xml}")
  if args.duration_s <= 0.0:
    raise ValueError("--duration-s must be positive")

  model = load_training_model(args.xml)
  data = mujoco.MjData(model)
  qpos_address, dof_address, actuator_address = joint_addresses(model)
  configure_training_physics(
    model,
    actuator_address,
    argparse.Namespace(all_feet_contact=args.all_feet_contact),
    kp=30.0,
    kd=1.0,
  )
  reset(model, data, qpos_address, argparse.Namespace(base_height=args.base_height))
  torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "TORSO")
  if torso_id < 0:
    raise ValueError("MJCF is missing TORSO body")
  gyro_address = sensor_address(model, "imu_ang_vel")
  accel_address = sensor_address(model, "imu_lin_acc")

  command_socket = open_command_socket(args.command_port)
  state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  state_destination = (args.state_host, args.state_port)
  command = JointCommand.standing()
  standing_command = JointCommand.standing()
  deadline = time.monotonic() + args.duration_s
  next_wall_step = time.monotonic()
  viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None

  print(
    "[INFO] Native MuJoCo UDP bridge "
    f"xml={args.xml} dt={PHYSICS_DT:.4f} command_port={args.command_port} "
    f"state={args.state_host}:{args.state_port} all_feet={args.all_feet_contact}"
  )
  print(
    "[INFO] Start the official x86 rl_deploy in another terminal. "
    "Zero-Kp or stale commands hold the verified standing PD target."
  )

  step = 0
  walking_samples: list[np.ndarray] = []
  try:
    while time.monotonic() < deadline:
      command = read_latest_command(command_socket, command)
      stale = time.monotonic() - command.received_at > args.command_timeout_s
      active_command = (
        standing_command if stale or is_idle_command(command) else command
      )
      apply_command(
        model,
        data,
        actuator_address,
        dof_address,
        sanitized_command(active_command),
      )
      mujoco.mj_step(model, data)
      is_walking_policy = active_command is not standing_command and np.all(
        np.abs(active_command.kp - 30.0) < 0.5
      )
      if is_walking_policy:
        _, linear_velocity_b, gravity_b, _ = body_frame_values(model, data, torso_id)
        walking_samples.append(
          np.array(
            (
              linear_velocity_b[0],
              linear_velocity_b[1],
              data.qpos[2],
              math.acos(float(np.clip(-gravity_b[2], -1.0, 1.0))),
            ),
            dtype=np.float64,
          )
        )
      state_socket.sendto(
        state_packet(
          model,
          data,
          torso_id,
          gyro_address,
          accel_address,
          qpos_address,
          dof_address,
          actuator_address,
        ),
        state_destination,
      )
      if viewer is not None:
        viewer.sync()
      step += 1
      if step % round(1.0 / PHYSICS_DT) == 0:
        _, linear_velocity_b, gravity_b, _ = body_frame_values(model, data, torso_id)
        tilt = math.acos(float(np.clip(-gravity_b[2], -1.0, 1.0)))
        print(
          f"[STATE] t={data.time:5.2f}s z={data.qpos[2]:.3f} "
          f"v_b=({linear_velocity_b[0]:+.3f}, {linear_velocity_b[1]:+.3f}) "
          f"tilt={tilt:.3f} "
          f"active={'stand' if active_command is standing_command else 'official'}"
        )
      if not args.no_realtime:
        next_wall_step += PHYSICS_DT
        sleep_s = next_wall_step - time.monotonic()
        if sleep_s > 0.0:
          time.sleep(sleep_s)
  finally:
    if viewer is not None:
      viewer.close()
    command_socket.close()
    state_socket.close()
  if walking_samples:
    metrics = np.stack(walking_samples)
    print(
      "[SUMMARY_WALK] "
      f"duration_s={len(metrics) * PHYSICS_DT:.2f} "
      f"mean_vx={metrics[:, 0].mean():+.3f} "
      f"mean_vy={metrics[:, 1].mean():+.3f} "
      f"mean_z={metrics[:, 2].mean():.3f} "
      f"mean_tilt={metrics[:, 3].mean():.3f} "
      f"max_tilt={metrics[:, 3].max():.3f}"
    )
  else:
    print("[SUMMARY_WALK] no Kp=30 walking-policy segment observed")


def main() -> None:
  args = parse_args()
  if STATE_PACKET_BYTES != 188:
    raise RuntimeError(f"Unexpected official state packet size: {STATE_PACKET_BYTES}")
  run(args)


if __name__ == "__main__":
  main()
