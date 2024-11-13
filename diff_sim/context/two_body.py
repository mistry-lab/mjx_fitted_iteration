import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

model_path = os.path.join(os.path.dirname(__file__), '../xmls/two_body.xml')

def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

_cfg = Config(
    lr=5e-3,
    seed=4,
    batch=512,
    samples=1,
    epochs=1000,
    eval=50,
    num_gpu=1,
    dt=0.01,
    ntotal=256,
    nsteps=16,
    mx=mjx.put_model(gen_model()),
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
        x = jnp.tanh(x) * .01
        return x

def policy(net: Network, mx: mjx.Model, dx: mjx.Data, x: jnp.ndarray) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t)
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(u))
    return dx, u

def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    # Combine position, orientation (quaternion), linear velocity, and angular velocity
    pos1 = dx.qpos[0]  # x position of the first point mass
    pos2 = dx.qpos[7]  # x position of the second point mass
    vel1 = dx.qvel[0]  # Linear velocity of the first point mass
    vel2 = dx.qvel[6]  # Linear velocity of the second point mass
    return jnp.array([pos1, pos2, vel1, vel2])

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def control_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = dx.qfrc_applied[0]
    return x**2 * 0.001

def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([100, 100, 1, 1])), x))

def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([100, 100, 1, 1])), x)) * _cfg.dt

def set_data(mx: mjx.Model, dx: mjx.Data, ctx: Context, key: jnp.ndarray) -> mjx.Data:
    # Initialize qpos and qvel with random values for the two point masses
    pos1 = jax.random.uniform(key, (1,), minval=-1.0, maxval=-.5) # Position for the first point mass
    _, key = jax.random.split(key)
    pos2 = jax.random.uniform(key, (1,), minval=-.5, maxval=0.)  # Position for the second point mass
    pos = jnp.concatenate(
        [pos1, jnp.array([0, 0]), jnp.array([1, 0, 0, 0]), pos2, jnp.array([0, 0]), jnp.array([1, 0, 0, 0])]
    )
    dx = dx.replace(qpos=dx.qpos.at[:].set(pos))
    return dx

def gen_network(n: int) -> Network:
    key = jax.random.PRNGKey(0)
    return Policy([4, 64, 64, 1], key)  # Adjust input and output dimensions

def is_terminal(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    time_limit = (dx.time / mx.opt.timestep) > (_cfg.ntotal - 1)
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
        loss_func=loss_fn_policy_det,
        is_terminal=is_terminal,
        state_decoder=state_decoder
    )
)
