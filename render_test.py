# render_test.py — 验证 offscreen 渲染
import mujoco
import numpy as np
import os

SCENE_XML = os.path.join(os.path.dirname(__file__), "Simulation", "SO101", "scene.xml")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

# 设置渲染分辨率
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 480

# 创建 offscreen 渲染器
renderer = mujoco.Renderer(model, height=480, width=640)

# 设置初始关节角度
mujoco.mj_resetData(model, data)
data.qpos[:6] = [0, 0, 0, 0, 0, 0]
mujoco.mj_forward(model, data)

# 渲染
renderer.update_scene(data)
image = renderer.render()
print(f"Image shape: {image.shape}")  # (480, 640, 3)
print(f"Image dtype: {image.dtype}")  # uint8
print("[OK] Offscreen rendering OK")
