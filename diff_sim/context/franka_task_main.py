import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.context.di import model_path
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network


model_path = os.path.join(os.path.dirname(__file__), '../xmls/franka_panda/mjx_scene.xml')

def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

_cfg = Config(
    lr=4e-3,
    num_gpu=1,
    seed=0,
    nsteps=16,
    ntotal=512,
    epochs=1000,
    batch=2,
    samples=1,
    eval=10,
    dt=0.005,
    mx= mjx.put_model(gen_model()),
    gen_model=gen_model,
)

class Policy(Network):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(
            dims[i], dims[i + 1], key=keys[i], use_bias=True
        ) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x, t):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x).squeeze()

def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    return jnp.concatenate([dx.qpos, dx.qvel], axis=-1)

def policy(net: Network, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    kpkv = net(x, t)
    kp = kpkv[...,:mx.nq]
    kv = kpkv[...,mx.nq:]
    u = -kp * dx.qpos - kv * dx.qvel
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
    return dx, u

def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    Q = jnp.diag(jnp.ones_like(x))
    return jnp.dot(x.T, jnp.dot(Q, x))

def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    Q = jnp.diag(jnp.ones_like(x))
    return jnp.dot(x.T, jnp.dot(Q, x)) * _cfg.dt

def control_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    u = dx.ctrl
    R = jnp.diag(jnp.ones_like(u))
    return jnp.dot(u.T, jnp.dot(R, u))
