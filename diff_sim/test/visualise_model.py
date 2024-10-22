from mujoco import mjx
from jax import config
import time
import numpy as np
import mujoco
import mujoco.viewer
import os

config.update('jax_default_matmul_precision', 'high')
model_path = os.path.join(os.path.dirname(__file__), '../xmls/point_mass_tendon.xml')
mx = mjx.put_model(mujoco.MjModel.from_xml_path(model_path))
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

import jax
import jax.numpy as jnp

def step(carry, _):
    dx, key = carry
    noise = 0.1 * jax.random.normal(key, (mx.nu,))
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(noise))
    dx = mjx.step(mx, dx)# Dynamics function
    x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
    return (dx, key), jnp.concatenate([x, dx.ctrl], axis=0)

@jax.jit
def rollout():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    dx = mjx.make_data(mx)
    _, res = jax.lax.scan(step, (dx, key), None, length=500)
    x, u = res[...,:-mx.nu], res[...,-mx.nu:]
    return x, u


def visualise_policy(viewer: mujoco.viewer.Handle):
    x, u = rollout()
    print(x.shape, u.shape)
    x = np.array(x.squeeze())
    for i in range(x.shape[0]):
        step_start = time.time()
        qpos = x[i, :m.nq]
        qvel = x[i, m.nq:]
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(m, d)
        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

import sys
from mujoco import viewer
viewer_context = viewer.launch_passive(m, d)

with viewer_context as viewer:
    visualise_policy(viewer)

sys.exit(0)

