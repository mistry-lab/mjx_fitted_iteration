import os
import time
import numpy as np
import mujoco
from mujoco import mjx
from mujoco import viewer
import jax
from jax import config
import jax.numpy as jnp

config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)
model_path = os.path.join(os.path.dirname(__file__), '../xmls/point_mass_tendon.xml')
mx = mjx.put_model(mujoco.MjModel.from_xml_path(model_path))
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

def step(carry, _):
    dx, key = carry
    dx = mjx.step(mx, dx)
    x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
    return (dx, key), jnp.concatenate([x, dx.ctrl], axis=0)

@jax.jit
def rollout():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    dx = mjx.make_data(mx)
    _, res = jax.lax.scan(step, (dx, key), None, length=50000)
    x, u = res[...,:-mx.nu], res[...,-mx.nu:]
    return x, u

def visualise_policy(viewer: mujoco.viewer.Handle):
    x, u = rollout()
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

viewer_context = viewer.launch_passive(m, d)

with viewer_context as viewer:
    visualise_policy(viewer)