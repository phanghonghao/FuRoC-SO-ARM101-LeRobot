"""SO-101 MuJoCo 仿真查看器 — 双击或 python sim_viewer.py 启动"""

import mujoco
import mujoco.viewer
import numpy as np
import os

SCENE_XML = os.path.join(os.path.dirname(__file__), "Simulation", "SO101", "scene.xml")


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 打印关节信息
    print("=" * 50)
    print("SO-101 MuJoCo Simulation")
    print("=" * 50)
    print(f"关节数: {model.njnt}")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  [{i}] {name:20s}  range: {model.jnt_range[i]}")
    print(f"\n Bodies: {model.nbody}")
    print(f" DOF:    {model.nv}")
    print("=" * 50)
    print("\n操作说明:")
    print("  左键拖拽  → 旋转视角")
    print("  右键拖拽  → 平移视角")
    print("  滚轮      → 缩放")
    print("  Ctrl+左键 → 施加外力")
    print("  双击关节  → 选中并控制")
    print("  ESC       → 退出")
    print()

    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 重置到初始位置
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
