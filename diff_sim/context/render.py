import time
import mujoco.viewer
import argparse

# Parse the path to the model file.
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the model file.")
args = parser.parse_args()

m = mujoco.MjModel.from_xml_path(args.path)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  while viewer.is_running():
    step_start = time.time()
    mujoco.mj_step(m, d)
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
