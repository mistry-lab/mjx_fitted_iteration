import time
import numpy as np
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import jax
from diff_sim.context.meta_context import Context
from diff_sim.simulate import controlled_simulate
from diff_sim.nn.base_nn import Network

def set_init(x, mx):
    dx = mjx.make_data(mx)
    # TODO: Decode x_init here.
    qpos = dx.qpos.at[:].set(x[:mx.nq])
    qvel = dx.qvel.at[:].set(x[mx.nq:])
    dx = dx.replace(qpos=qpos, qvel=qvel)
    return mjx.step(mx, dx)
set_init_vmap = jax.jit(jax.vmap(set_init,in_axes=(0, None)))

# TODO pass the 2 dxs for the simulation and create themn in runner. 
def visualise_policy(
        d: mujoco.MjData, m: mujoco.MjModel, viewer: mujoco.viewer.Handle,
        ctx: Context, net: Network, key: jax.random.PRNGKey
):
    key, xkey, tkey, user_key = jax.random.split(key, num=4)
    x_inits = ctx.cbs.init_gen(2, xkey)
    dxs = set_init_vmap(x_inits, ctx.cfg.mx)
    _, x, _, _, _, _ = controlled_simulate(dxs, ctx, net, tkey)
    x = jax.vmap(jax.vmap(ctx.cbs.state_decoder))(x)
    x = np.array(x.squeeze())
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