"""Deep Robotics Lite3 velocity environment configurations."""

import math

from mjlab.asset_zoo.robots import (
  LITE3_ACTION_SCALE,
  get_lite3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def deep_robotics_lite3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Deep Robotics Lite3 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_lite3_robot_cfg()}

  del cfg.observations["actor"].terms["base_lin_vel"]
  del cfg.observations["actor"].terms["height_scan"]
  cfg.observations["actor"].terms["base_ang_vel"].scale = 0.25
  cfg.observations["actor"].terms["joint_vel"].scale = 0.05
  cfg.observations["actor"].terms["gait_phase"] = ObservationTermCfg(
    func=mdp.gait_phase_clock,
    params={"period": 0.48},
  )
  cfg.observations["critic"].terms["gait_phase"] = ObservationTermCfg(
    func=mdp.gait_phase_clock,
    params={"period": 0.48},
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "TORSO"

  leg_names = ("FL", "FR", "HL", "HR")
  foot_body_names = tuple(f"{leg}_FOOT" for leg in leg_names)
  foot_site_names = tuple(f"{name}_site" for name in foot_body_names)
  foot_geom_names = tuple(f"{leg}_FOOT_collision" for leg in leg_names)
  thigh_geom_names = tuple(f"{leg}_THIGH_collision" for leg in leg_names)
  shank_geom_names = tuple(f"{leg}_SHANK_collision" for leg in leg_names)

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in foot_site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="TORSO", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="TORSO", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=shank_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("TORSO_collision",),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LITE3_ACTION_SCALE
  joint_pos_action.clip = {
    ".*_HipX_joint": (-0.20, 0.20),
    ".*_HipY_joint": (-1.05, -0.25),
    ".*_Knee_joint": (0.85, 1.75),
  }

  cfg.viewer.body_name = "TORSO"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.4

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("TORSO",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*_HipX_joint": 0.05,
    r".*_HipY_joint": 0.08,
    r".*_Knee_joint": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_HipX_joint": 0.25,
    r".*_HipY_joint": 0.4,
    r".*_Knee_joint": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_HipX_joint": 0.3,
    r".*_HipY_joint": 0.5,
    r".*_Knee_joint": 0.7,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("TORSO",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("TORSO",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = foot_site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_ground_cfg.name},
  )

  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def deep_robotics_lite3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Deep Robotics Lite3 flat terrain velocity configuration."""
  cfg = deep_robotics_lite3_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  for key in ("self_collisions", "shank_collision", "trunk_collision"):
    cfg.rewards.pop(key, None)

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  cfg.curriculum.pop("terrain_levels", None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 1.0

  if play:
    cfg.events.pop("push_robot", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  else:
    cfg.events["reset_base"].params["pose_range"] = {
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.04),
      "roll": (-0.08, 0.08),
      "pitch": (-0.08, 0.08),
      "yaw": (-0.20, 0.20),
    }
    cfg.events["reset_base"].params["velocity_range"] = {
      "x": (-0.15, 0.15),
      "y": (-0.10, 0.10),
      "z": (-0.05, 0.05),
      "roll": (-0.15, 0.15),
      "pitch": (-0.15, 0.15),
      "yaw": (-0.20, 0.20),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.08, 0.08)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.4, 0.4)

    cfg.events["foot_friction"].params["ranges"] = (0.5, 1.5)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.04, 0.04),
      1: (-0.04, 0.04),
      2: (-0.04, 0.04),
    }
    cfg.events["push_robot"].interval_range_s = (4.0, 8.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.15, 0.15),
      "y": (-0.12, 0.12),
      "z": (-0.05, 0.05),
      "roll": (-0.15, 0.15),
      "pitch": (-0.15, 0.15),
      "yaw": (-0.20, 0.20),
    }
    cfg.events["body_pseudo_inertia"] = EventTermCfg(
      func=envs_mdp.dr.pseudo_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("TORSO",)),
        "alpha_range": (-0.03, 0.03),
      },
    )

  cfg.rewards["track_linear_velocity"].weight = 7.5
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["upright"].weight = 3.5
  cfg.rewards["pose"].weight = 0.65
  cfg.rewards["body_ang_vel"].weight = -0.10
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["action_l2"] = RewardTermCfg(
    func=envs_mdp.action_l2,
    weight=-0.008,
  )
  cfg.rewards["action_magnitude_target"] = RewardTermCfg(
    func=mdp.action_magnitude_target,
    weight=1.2,
    params={
      "command_name": "twist",
      "target": 1.2,
      "std": 0.45,
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["forward_velocity"] = RewardTermCfg(
    func=mdp.forward_velocity,
    weight=1.5,
    params={"command_name": "twist", "command_threshold": 0.05},
  )
  cfg.rewards["forward_overspeed_l2"] = RewardTermCfg(
    func=mdp.forward_overspeed_l2,
    weight=-3.0,
    params={
      "command_name": "twist",
      "margin": 0.1,
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["lateral_velocity_l2"] = RewardTermCfg(
    func=mdp.lateral_velocity_l2,
    weight=-0.5,
  )
  cfg.rewards["yaw_velocity_l2"] = RewardTermCfg(
    func=mdp.yaw_velocity_l2,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["air_time"].params["command_threshold"] = 0.05
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
  cfg.rewards["foot_slip"].weight = -0.2
  cfg.rewards["swing_forward_velocity"] = RewardTermCfg(
    func=mdp.swing_forward_velocity,
    weight=1.1,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": cfg.rewards["foot_slip"].params["asset_cfg"],
    },
  )
  cfg.rewards["trot_phase_gait"] = RewardTermCfg(
    func=mdp.trot_phase_gait,
    weight=1.4,
    params={
      "sensor_name": "feet_ground_contact",
      "period": 0.48,
      "duty_factor": 0.55,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["diagonal_gait"] = RewardTermCfg(
    func=mdp.diagonal_gait_contact,
    weight=2.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["termination"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-25.0,
  )

  if play:
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.rel_forward_envs = 1.0
    twist_cmd.ranges.lin_vel_x = (0.4, 0.8)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  else:
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.rel_forward_envs = 1.0
    twist_cmd.ranges.lin_vel_x = (0.35, 0.65)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.1, 0.1)
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.35, 0.65),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.1, 0.1),
      },
      {
        "step": 6000 * 24,
        "lin_vel_x": (0.2, 0.9),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.2, 0.2),
      },
      {
        "step": 12000 * 24,
        "lin_vel_x": (-0.2, 1.0),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.4, 0.4),
      },
    ]

  return cfg
