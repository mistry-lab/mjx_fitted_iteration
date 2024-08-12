import time
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjtStage, mjtSensor


# pass in a trajectory of pos and vels and animate it
# trajectory is an array of shape (n_timesteps, 2, qpos_dim)
def animate_trajectory(trajectory: np.ndarray, d: mujoco.MjData, m: mujoco.MjModel):
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        for i in range(trajectory.shape[0]):
            step_start = time.time()
            qpos = trajectory[i, 0, :]
            qvel = trajectory[i, 1, :]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
