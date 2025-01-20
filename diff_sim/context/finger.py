## context for finger spinner environment where the spinner is the last body with one hinge joint and the finger is the first body with one hinge joint
## with 2 hinge joint the finger is the only body that is controlled

import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network


model_path = os.path.join(os.path.dirname(__file__), '../xmls/finger_mjx.xml')
def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

_cfg = Config(
    lr=2.e-3,
    seed=4,
    batch=4,
    samples=1,
    epochs=1000,
    eval=5,
    num_gpu=1,
    dt=0.0075,
    ntotal=300,
    nsteps=300,
    mx=mjx.put_model(gen_model()),
    eps = 1.0e-6,
    gen_model=gen_model
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
        x = self.layers[-1](x).squeeze()
        x = jnp.tanh(x) * 1.
        return x


def policy(net: Network, mx: mjx.Model, dx: mjx.Data, x: jnp.ndarray) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t)
    return u

# should be defined by the user and controler should simply return u
def set_control(dx: mjx.Data,u: jnp.ndarray) -> mjx.Data:
    return dx.replace(ctrl=dx.ctrl.at[:].set(u))

def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    return jnp.concatenate([dx.qpos, dx.qvel])

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def control_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = dx.ctrl
    return jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([0., 0.])), x)
    )

# def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
#     pos = state_encoder(mx, dx)[2]
#     vel = state_encoder(mx, dx)[5]
#     x = jnp.array([pos, vel])
#     return jnp.dot(
#         x.T, jnp.dot(jnp.diag(jnp.array([1, 0])), x)
#     )

def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    pos_finger = dx.qpos[2]
    u = dx.ctrl
    return 0.002 * jnp.sum(u ** 2) + 0.002 * pos_finger ** 2


# def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
#     pos = state_encoder(mx, dx)[2]
#     vel = state_encoder(mx, dx)[5]
#     x = jnp.array([pos, vel])
#     return jnp.dot(
#         x.T, jnp.dot(jnp.diag(jnp.array([1, 0])), x)
#     )

def terminal_cost(mx: mjx.Model, dx: mjx.Data)-> jnp.ndarray:
    pos_finger = dx.qpos[2]
    return 8. * pos_finger ** 2

def set_data(mx: mjx.Model, dx: mjx.Data, ctx: Context, key: jnp.ndarray) -> mjx.Data:
    theta1 = jax.random.uniform(key, (1,), minval=0.75, maxval=1.1)
    theta2 = jnp.array([0.])
    _, key = jax.random.split(key)
    theta3 = jax.random.uniform(key, (1,), minval=-.85, maxval=-.6)
    qpos = jnp.concatenate([theta1, theta2, theta3])
    qvel = jnp.array([0., 0., 0.])
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos), qvel=dx.qvel.at[:].set(qvel))
    return dx

def gen_network(n: int) -> Network:
    key = jax.random.PRNGKey(0)
    return Policy([6, 128, 128, 2], key)

def is_terminal(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    time_limit = (dx.time/mx.opt.timestep) > (_cfg.ntotal - 1)
    # check if the first and second joint do 1 full loop (2.pi)
    # limit1 = jnp.abs(dx.qpos[0]) > 2*jnp.pi
    # limit2 = jnp.abs(dx.qpos[1]) > 2*jnp.pi
    # # use lax.cond to check if the limits are reached
    # def true_fn(): return jnp.array([True])
    # def time_fn(): return jnp.array([time_limit])
    # cond = jax.lax.cond(
    #     jnp.logical_or(limit1, limit2),
    #     true_fn,
    #     time_fn,
    # )
    return jnp.array([time_limit])

ctx = Context(
    _cfg,
    Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        set_data=set_data,
        state_encoder=state_encoder,
        gen_network=gen_network,
        controller=policy,
        set_control=set_control,
        loss_func=loss_fn_policy_det,
        is_terminal=is_terminal,
        state_decoder=state_decoder
    )
)