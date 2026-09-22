"""Deeprobotics lite3 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

LITE3_XML: Path = Path(__file__).resolve().parent / "xmls" / "Lite3.xml"
# Path(__file__).parent 代表当前 lite3_constants.py 所在的目录
assert LITE3_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(LITE3_XML))


##
# Actuator config.
##

STIFFNESS = 30.0
DAMPING = 3.0
HIP_EFFORT_LIMIT = 24.0
KNEE_EFFORT_LIMIT = 36.0

LITE3_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HipX_joint", ".*_HipY_joint"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=HIP_EFFORT_LIMIT,
  armature=0.0,
)
LITE3_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=KNEE_EFFORT_LIMIT,
  armature=0.0,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.375),
  joint_pos={
    ".*_HipY_joint": -0.65,
    ".*_Knee_joint": 1.3,
    ".*[FH]R_HipX_joint": 0.0,
    ".*[FH]L_HipX_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FH][LR]_FOOT_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions.
# Foot collisions are given custom condim, friction.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  # Harden all collision geoms.
  solref=(0.01, 1),
  # Configure feet colliders. Other colliders are frictionless (condim=1).
  condim={_foot_regex: 6, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

LITE3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    LITE3_HIP_ACTUATOR_CFG,
    LITE3_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.99,
)


def get_lite3_robot_cfg() -> EntityCfg:
  """Get a fresh Lite3 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=LITE3_ARTICULATION,
  )


LITE3_ACTION_SCALE = {
  ".*_HipX_joint": 0.10,
  ".*_HipY_joint": 0.20,
  ".*_Knee_joint": 0.20,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_lite3_robot_cfg())

  viewer.launch(robot.spec.compile())
