import time
import numpy as np
import mujoco
import mujoco.viewer

def animate_trajectory(trajectory: np.ndarray, d: mujoco.MjData, m: mujoco.MjModel):
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        for b in range(trajectory.shape[0]):
            for i in range(trajectory.shape[1]):
                step_start = time.time()
                qpos = trajectory[b, i, :m.nq]
                qvel = trajectory[b, i, m.nq:]
                d.qpos[:] = qpos
                d.qvel[:] = qvel
                mujoco.mj_forward(m, d)
                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)