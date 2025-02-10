import mujoco
import numpy
from mujoco import viewer, mj_step
from diff_sim.utils.mj_data_manager import create_data_manager
import jax
import time
from diff_sim.context.tasks import ctxs
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="task name", default="snake")
    args = parser.parse_args()
    
    ctx = ctxs[args.task]
    model = ctx.cfg.gen_model()
    data = mujoco.MjData(model)

    init_key = jax.random.PRNGKey(ctx.cfg.seed)
    data_manager = create_data_manager()
    dxs = data_manager.create_data(ctx.cfg.mx, ctx, ctx.cfg.batch, init_key)

    # Pass dxs to the viewer function
    interactive_viewer(model, data, dxs, ctx)