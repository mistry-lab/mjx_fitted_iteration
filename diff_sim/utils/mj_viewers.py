import time
import numpy as np
import mujoco
from mujoco import viewer
import jax
from diff_sim.context.meta_context import Context
from diff_sim.simulate import controlled_simulate_fd as controlled_simulate
from diff_sim.nn.base_nn import Network
import equinox as eqx
from diff_sim.utils.mj_data_manager import create_data_manager

data_manager = create_data_manager()

# TODO pass the 2 dxs for the simulation and create themn in runner. 
def visualise_policy(
        d: mujoco.MjData, m: mujoco.MjModel, viewer: mujoco.viewer.Handle,
        ctx: Context, net: Network, key: jax.random.PRNGKey
):
    key, xkey, tkey, user_key = jax.random.split(key, num=4)
    dxs = data_manager.create_data(ctx.cfg.mx, ctx, 3, xkey)
    dxs, x, _, _, _, term_mask = eqx.filter_jit(controlled_simulate)(dxs, ctx, net, tkey, ctx.cfg.ntotal)
    x = jax.vmap(jax.vmap(ctx.cbs.state_decoder))(x)
    x = np.array(x)[:3].reshape(1, ctx.cfg.ntotal, -1)
    for b in range(x.shape[0]):
        for i in range(x.shape[1]):
            step_start = time.time()
            qpos = x[b, i, :m.nq]
            qvel = x[b, i, -m.nv:]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            d.mocap_pos = np.array(dxs.mocap_pos[b])
            d.mocap_quat = np.array(dxs.mocap_quat[b])
            mujoco.mj_forward(m, d)
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def visualise_traj(
        x: jax.numpy.ndarray, d: mujoco.MjData, m: mujoco.MjModel, viewer: mujoco.viewer.Handle
):
    x = np.array(x)
    for b in range(x.shape[0]):
        for i in range(x.shape[1]):
            step_start = time.time()
            qpos = x[b, i, :m.nq]
            qvel = x[b, i, m.nq:]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)



def visualise_traj_generic(
        x, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01
):

    with viewer.launch_passive(m, d) as v:
        x = np.array(x)
        for b in range(x.shape[0]):
            for i in range(x.shape[1]):
                step_start = time.time()
                qpos = x[b, i, :m.nq]
                qvel = x[b, i, m.nq:m.nq + m.nv]
                d.qpos[:] = qpos
                d.qvel[:] = qvel
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(sleep)
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)