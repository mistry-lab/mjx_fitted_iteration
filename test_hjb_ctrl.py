import jax
import mujoco
from mujoco import mjx
import equinox as eqx
from equinox import nn
from jax import numpy as jnp

devices = jax.devices()
is_gpu = next((d for d in devices if 'gpu' in d.device_kind.lower()), None)
key = jax.random.PRNGKey(0)
model = mujoco.MjModel.from_xml_path('/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml')
mx = mjx.put_model(model)


def gen_vf(dims: list, act, key):
    # use dims to define an eqx Sequential model with linear layers use act as activation
    _net = []
    for d in dims[:-2]:
        _net.append(eqx.nn.Linear(d, dims[dims.index(d) + 1], key=key))
        _net.append(act)

    _net.append(eqx.nn.Linear(dims[-2], dims[-1], key=key))
    print(f"Net structure \n {_net}")
    return nn.Sequential(_net)


vf = gen_vf([4, 64, 64, 1], jax.nn.softplus, key)


def calc_ctrl(m, dx, vf):
    x = jnp.stack([dx.qpos, dx.qvel])
    u = vf(x)
    return u

def set_crtl(dx, u):
    u = dx.ctrl.at[:].set(u)
    dx = dx.replace(ctrl=u)
    return dx

def sim_step(dx, vf):
    u = calc_ctrl(mx, dx, vf)
    dx = set_crtl(dx, u)
    dx = mjx.step(mx, dx)
    return dx, jnp.stack([dx.qpos, dx.qvel])

def sim(x_init, mx, vf):
    dx = mjx.make_data(mx)
    qp = x_init[:mx.nq]
    qv = x_init[mx.nq:]
    qpos = dx.qpos.at[:].set(qp)
    qvel = dx.qvel.at[:].set(qv)
    dx = dx.replace(
        qpos=qpos, qvel=qvel
    )

    sim_step(dx, vf)

n_simulations = 10
pole = jax.random.normal(key, shape=(n_simulations,)) * 0.1
cart = jnp.zeros(n_simulations)

# the position
qpos = jnp.stack([cart, pole], axis=-1)
qvel = jnp.zeros((n_simulations, 2))
x_init = jnp.concatenate([qpos, qvel], axis=-1)

mapped_sim = jax.vmap(sim, in_axes=(0, None, None))
res = mapped_sim(x_init, mx, vf)
