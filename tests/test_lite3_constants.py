"""Tests for Lite3 robot constants and base velocity task config."""

import re
from pathlib import Path
from unittest.mock import Mock
from xml.etree import ElementTree as ET

import pytest
import torch
from conftest import get_test_device, initialize_entity

from mjlab.asset_zoo.robots.deep_lite3 import lite3_constants
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.lite3.env_cfgs import deep_lite3_flat_env_cfg
from mjlab.tasks.velocity.config.lite3.rl_cfg import deep_lite3_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_lite3_constants_match_mjlab_learn_configuration() -> None:
  assert lite3_constants.INIT_STATE.pos == (0.0, 0.0, 0.35)
  joint_pos = lite3_constants.INIT_STATE.joint_pos
  assert joint_pos is not None
  assert joint_pos[".*HipX_joint"] == 0.0
  assert joint_pos[".*HipY_joint"] == -0.75
  assert joint_pos[".*Knee_joint"] == 1.3
  assert lite3_constants.LITE3_HIP_ACTUATOR_CFG.stiffness == 30.0
  assert lite3_constants.LITE3_HIP_ACTUATOR_CFG.damping == 3.0
  assert lite3_constants.LITE3_HIP_ACTUATOR_CFG.effort_limit == 24.0
  assert lite3_constants.LITE3_KNEE_ACTUATOR_CFG.stiffness == 30.0
  assert lite3_constants.LITE3_KNEE_ACTUATOR_CFG.damping == 3.0
  assert lite3_constants.LITE3_KNEE_ACTUATOR_CFG.effort_limit == 36.0
  assert lite3_constants.LITE3_ACTION_SCALE == pytest.approx(
    {
      ".*_HipX_joint": 0.2,
      ".*_HipY_joint": 0.2,
      ".*_Knee_joint": 0.3,
    }
  )


def test_lite3_foot_collision_regex_matches_all_feet() -> None:
  foot_regex = lite3_constants.FULL_COLLISION.condim
  assert isinstance(foot_regex, dict)
  foot_pattern = next(pattern for pattern, condim in foot_regex.items() if condim == 6)
  matcher = re.compile(foot_pattern)

  assert {
    name
    for name in (
      "FL_FOOT_collision",
      "FR_FOOT_collision",
      "HL_FOOT_collision",
      "HR_FOOT_collision",
    )
    if matcher.fullmatch(name)
  } == {
    "FL_FOOT_collision",
    "FR_FOOT_collision",
    "HL_FOOT_collision",
    "HR_FOOT_collision",
  }


def test_lite3_flat_cfg_matches_mjlab_learn_base_task() -> None:
  cfg = deep_lite3_flat_env_cfg()
  action = cfg.actions["joint_pos"]

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["critic"].terms
  assert "self_collisions" not in cfg.rewards
  assert "out_of_terrain_bounds" not in cfg.terminations
  assert "fell_over" in cfg.terminations
  assert "terrain_levels" not in cfg.curriculum

  assert isinstance(action, JointPositionActionCfg)
  assert action.scale == lite3_constants.LITE3_ACTION_SCALE
  assert cfg.rewards["foot_clearance"].params["target_height"] == 0.08
  assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.08
  assert cfg.rewards["foot_slip"].weight == -0.25

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)


def test_lite3_ppo_runner_matches_mjlab_learn_config() -> None:
  cfg = deep_lite3_ppo_runner_cfg()

  assert cfg.experiment_name == "deep_lite3_velocity"
  assert cfg.save_interval == 200
  assert cfg.num_steps_per_env == 24
  assert cfg.max_iterations == 10_000
  assert cfg.actor.distribution_cfg is not None
  assert cfg.actor.distribution_cfg["init_std"] == 1.0
  assert cfg.algorithm.entropy_coef == 0.01


def test_lite3_zero_action_targets_default_joint_positions() -> None:
  """Zero joint-position action should hold Lite3's default standing pose."""
  device = get_test_device()
  entity = Entity(lite3_constants.get_lite3_robot_cfg())
  entity, _ = initialize_entity(entity, device=device, num_envs=2)

  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = 2
  env.device = device
  env.scene = {"robot": entity}

  cfg = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=lite3_constants.LITE3_ACTION_SCALE,
  )
  action = cfg.build(env)

  zero_action = torch.zeros(env.num_envs, action.action_dim, device=device)
  action.process_actions(zero_action)
  action.apply_actions()

  expected = entity.data.default_joint_pos[:, action.target_ids]
  actual = entity.data.joint_pos_target[:, action.target_ids]

  assert action.action_dim == 12
  assert torch.all(action.raw_action == 0.0)
  torch.testing.assert_close(action._processed_actions, expected)
  torch.testing.assert_close(actual, expected)


def test_lite3_velocity_task_required_sites_and_sensors_exist() -> None:
  """Lite3 MJCF should expose frames and sensors used by velocity tasks."""
  xml_path = (
    Path(__file__).resolve().parents[1]
    / "src/mjlab/asset_zoo/robots/deep_lite3/xmls/Lite3.xml"
  )
  root = ET.parse(xml_path).getroot()
  site_names = {
    site.attrib["name"] for site in root.iter("site") if "name" in site.attrib
  }
  sensor_root = root.find("sensor")
  assert sensor_root is not None
  sensor_names = {
    sensor.attrib["name"] for sensor in sensor_root if "name" in sensor.attrib
  }

  for site_name in (
    "FL_FOOT_site",
    "FR_FOOT_site",
    "HL_FOOT_site",
    "HR_FOOT_site",
    "imu_site",
  ):
    assert site_name in site_names

  for sensor_name in ("imu_lin_vel", "imu_ang_vel"):
    assert sensor_name in sensor_names
