import jax
import mujoco
from mujoco import mjx
import equinox as eqx
from equinox import nn
from jax import numpy as jnp
from hjb_controller import ValueFunc, Controller

devices = jax.devices()
x_key, model_key = jax.random.split(jax.random.PRNGKey(0))
is_gpu = next((d for d in devices if 'gpu' in d.device_kind.lower()), None)
model = mujoco.MjModel.from_xml_path('/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml')
mx = mjx.put_model(model)


def calc_ctrl(m, dx, ctrl):
    x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
    u = ctrl(x)
    return u

def set_crtl(dx, u):
    u = dx.ctrl.at[:].set(u)
    dx = dx.replace(ctrl=u)
    return dx

def sim_step(dx, ctrl):
    u = calc_ctrl(mx, dx, ctrl)
    dx = set_crtl(dx, u)
    dx = mjx.step(mx, dx)
    return dx, jnp.stack([dx.qpos, dx.qvel])

def sim(x_init, mx, vf):
    dx = mjx.make_data(mx)
    dx = mjx.step(mx, dx)
    ctrl = Controller(vf, mx, dx)
    qp = x_init[:mx.nq]
    qv = x_init[mx.nq:]
    qpos = dx.qpos.at[:].set(qp)
    qvel = dx.qvel.at[:].set(qv)
    dx = dx.replace(
        qpos=qpos, qvel=qvel
    )

    return sim_step(dx, ctrl)

n_simulations = 10
vf = ValueFunc([4, 64, 64, 1], model_key, jax.nn.softplus)

pole = jax.random.normal(x_key, shape=(n_simulations,)) * 0.1
cart = jnp.zeros(n_simulations)
qpos = jnp.stack([cart, pole], axis=-1)
qvel = jnp.zeros((n_simulations, 2))
x_init = jnp.concatenate([qpos, qvel], axis=-1)
mapped_sim = jax.vmap(sim, in_axes=(0, None, None))
res = mapped_sim(x_init, mx, vf)
