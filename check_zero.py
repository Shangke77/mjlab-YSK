import mujoco

# 加载模型
model = mujoco.MjModel.from_xml_path(
  "src/mjlab/asset_zoo/robots/deeprobotics_lite3/xmls/Lite3.xml"
)

print("====================================================")
print("   绝影 Lite3 关节绝对零位（Zero State）与限位排查")
print("====================================================")

# 遍历所有关节，打印出它们的初始值和物理限位
for i in range(model.njnt):
  # 修正了这里的枚举属性错误名
  jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
  jnt_type = model.jnt_type[i]

  # 3 是 HINGE（旋转关节）的常数代号
  if jnt_type == 3:
    jnt_range = model.jnt_range[i]
    print(
      f"关节名称: {jnt_name:<20} | 零位默认值: 0.0 | 物理限位范围: [{jnt_range[0]:.3f}, {jnt_range[1]:.3f}]"
    )

print("====================================================")
