"""Deep Robotics Lite3 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

LITE3_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "deep_lite3" / "xmls" / "Lite3.xml"
)
assert LITE3_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(LITE3_XML))


##
# Actuator config.
##

# Parameters from DeepRobotics Lite3 Isaac Lab config:
#   Hip:  effort_limit=24.0, velocity_limit=26.2, stiffness=30.0, damping=1.0
#   Knee: effort_limit=36.0, velocity_limit=17.3, stiffness=30.0, damping=1.0
#
# Note: BuiltinPositionActuatorCfg in mjlab does not need velocity_limit here,
# so we keep the effort/stiffness/damping/armature values.
LITE3_HIP_STIFFNESS = 30.0
LITE3_HIP_DAMPING = 3.0
LITE3_HIP_EFFORT_LIMIT = 24.0
LITE3_HIP_ARMATURE = 0.0

LITE3_KNEE_STIFFNESS = 30.0
LITE3_KNEE_DAMPING = 3.0
LITE3_KNEE_EFFORT_LIMIT = 36.0
LITE3_KNEE_ARMATURE = 0.0

LITE3_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HipX_joint", ".*_HipY_joint"),
  stiffness=LITE3_HIP_STIFFNESS,
  damping=LITE3_HIP_DAMPING,
  effort_limit=LITE3_HIP_EFFORT_LIMIT,
  armature=LITE3_HIP_ARMATURE,
)

LITE3_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_joint",),
  stiffness=LITE3_KNEE_STIFFNESS,
  damping=LITE3_KNEE_DAMPING,
  effort_limit=LITE3_KNEE_EFFORT_LIMIT,
  armature=LITE3_KNEE_ARMATURE,
)

##
# Initial state.
##

# From DeepRobotics Lite3 Isaac Lab config:
#   pos=(0.0, 0.0, 0.375)
#   HipX=0.0, HipY=-0.65, Knee=1.3
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.35),
  joint_pos={
    ".*HipX_joint": 0.0,
    ".*HipY_joint": -0.75,
    ".*Knee_joint": 1.3,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^(FL|FR|HL|HR)_FOOT_collision$"

# This enables all collisions.
# Foot collisions are given custom condim and friction.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(
  "^(FL|FR|HL|HR)_FOOT_collision$",
  "^(FL|FR|HL|HR)_SHANK_collision$",
  "^(FL|FR|HL|HR)_THIGH_collision$",
  "^(FL|FR|HL|HR)_HIP_collision$",
  ),
  solref=(0.01, 1),
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


LITE3_ACTION_SCALE: dict[str, float] = {}
for a in LITE3_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    LITE3_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_lite3_robot_cfg())
  viewer.launch(robot.spec.compile())