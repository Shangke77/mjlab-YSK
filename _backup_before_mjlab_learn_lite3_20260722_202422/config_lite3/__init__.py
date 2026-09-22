from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  deep_robotics_lite3_flat_env_cfg,
  deep_robotics_lite3_rough_env_cfg,
)
from .rl_cfg import deep_robotics_lite3_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-DeepRobotics-Lite3",
  env_cfg=deep_robotics_lite3_rough_env_cfg(),
  play_env_cfg=deep_robotics_lite3_rough_env_cfg(play=True),
  rl_cfg=deep_robotics_lite3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-DeepRobotics-Lite3",
  env_cfg=deep_robotics_lite3_flat_env_cfg(),
  play_env_cfg=deep_robotics_lite3_flat_env_cfg(play=True),
  rl_cfg=deep_robotics_lite3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
