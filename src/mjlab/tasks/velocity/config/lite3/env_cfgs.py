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

BASE_BODY_NAME = "TORSO"
FOOT_NAMES = ("FL", "FR", "HL", "HR")
FOOT_GEOM_NAMES = tuple(f"{name}_FOOT_collision" for name in FOOT_NAMES)
FOOT_SITE_NAMES = tuple(f"{name}_FOOT_site" for name in FOOT_NAMES)
THIGH_GEOM_NAMES = tuple(f"{name}_THIGH_collision" for name in FOOT_NAMES)
SHANK_GEOM_NAMES = tuple(f"{name}_SHANK_collision" for name in FOOT_NAMES)
TORSO_GEOM_NAMES = ("TORSO_collision",)


def deep_lite3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Deep Robotics Lite3 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.observations["actor"].terms.pop("base_lin_vel", None)

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_lite3_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = BASE_BODY_NAME
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=BASE_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=BASE_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(mode="geom", pattern=THIGH_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(mode="geom", pattern=SHANK_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  torso_ground_cfg = ContactSensorCfg(
    name="torso_ground_touch",
    primary=ContactMatch(mode="geom", pattern=TORSO_GEOM_NAMES, entity="robot"),
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
    torso_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LITE3_ACTION_SCALE

  cfg.viewer.body_name = BASE_BODY_NAME
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FL|FR|HL|HR)_HipX_joint.*": 0.05,
    r".*(FL|FR|HL|HR)_HipY_joint.*": 0.05,
    r".*(FL|FR|HL|HR)_Knee_joint.*": 0.10,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FL|FR|HL|HR)_HipX_joint.*": 0.30,
    r".*(FL|FR|HL|HR)_HipY_joint.*": 0.30,
    r".*(FL|FR|HL|HR)_Knee_joint.*": 0.60,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FL|FR|HL|HR)_HipX_joint.*": 0.30,
    r".*(FL|FR|HL|HR)_HipY_joint.*": 0.30,
    r".*(FL|FR|HL|HR)_Knee_joint.*": 0.60,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES
  cfg.rewards["foot_clearance"].params["target_height"] = 0.08
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
  cfg.rewards["foot_slip"].weight = -0.25

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
  cfg.rewards["torso_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": torso_ground_cfg.name},
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


def deep_lite3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Deep Robotics Lite3 flat terrain velocity configuration."""
  cfg = deep_lite3_rough_env_cfg(play=play)

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
  }
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in remove_sensors
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  cfg.rewards.pop("self_collisions", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    # Viser's command GUI requires every axis slider max to be at least 0.1.
    # Lite3 is trained with zero lateral command, so keep this play-only range tiny.
    twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
