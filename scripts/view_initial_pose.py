"""Launch MuJoCo viewer to inspect SO-ARM101 initial pose.

Joint ranges (degrees):
  shoulder_pan :  -110.0 ~ +110.0
  shoulder_lift:  -100.0 ~ +100.0
  elbow_flex   :   -96.8 ~  +96.8
  wrist_flex   :   -95.0 ~  +95.0
  wrist_roll   :  -157.2 ~ +162.8
  gripper      :   -10.0 ~ +100.0
"""

import math
import mujoco
import mujoco.viewer
import os

SCENE_XML = os.path.join(
    os.path.dirname(__file__), "..", "SO-ARM101-LeRobot", "Simulation", "SO101", "scene.xml"
)

# Set an initial joint configuration (6 DOF), in RADIANS.
# Change these values to test different poses.
INITIAL_QPOS = [0, -0.5, 1, -0.5, 0, 1]


def main():
    model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE_XML))
    data = mujoco.MjData(model)

    # Set initial joint positions
    nq = min(len(INITIAL_QPOS), model.nq)
    data.qpos[:nq] = INITIAL_QPOS[:nq]
    mujoco.mj_forward(model, data)

    # Print joint info
    print(f"Model: {model.nq} joints, {model.nu} actuators")
    print(f"\n{'Joint':<15} {'Rad':>8} {'Deg':>8} {'Range (deg)':>20}")
    print("-" * 55)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(nq)]
    for i, name in enumerate(joint_names):
        rad = data.qpos[i]
        deg = math.degrees(rad)
        lo = math.degrees(model.jnt_range[i][0])
        hi = math.degrees(model.jnt_range[i][1])
        print(f"{name:<15} {rad:>8.3f} {deg:>7.1f}\u00b0  [{lo:.1f}\u00b0, {hi:.1f}\u00b0]")
    print()
    print("Drag slider in Control panel to move joints.")
    print("Press Esc to quit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
