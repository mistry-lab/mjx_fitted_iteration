import mujoco
from mujoco import viewer, mj_step
from diff_sim.utils.mj_data_manager import create_data_manager
import jax
import time
import numpy as np 

def interactive_viewer(model, data, dxs, ctx):
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            for b in range(dxs.qpos.shape[0]):
                data.qpos = dxs.qpos[b]
                data.mocap_pos = np.array(dxs.mocap_pos[b])
                data.mocap_quat = np.array(dxs.mocap_quat[b])
                mj_step(model, data)
                v.sync()
                time.sleep(0.2)


def check_init_data(ctx, batch=0):
    model = ctx.gen_model()
    data = mujoco.MjData(model)
    init_key = jax.random.PRNGKey(ctx.seed)
    data_manager = create_data_manager()
    dxs = data_manager.create_data(ctx, init_key, custom_batch=batch)
    interactive_viewer(model, data, dxs, ctx)
