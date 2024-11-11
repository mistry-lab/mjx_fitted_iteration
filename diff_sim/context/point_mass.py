
import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

def op_and_split(f, key):
    key, subkey = jax.random.split(key)
    return f(subkey), key

model_path = os.path.join(os.path.dirname(__file__), '../xmls/point_mass.xml')
def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

_cfg = Config(
    lr=4e-3,
    seed=4,
    batch=240,
    samples=1,
    epochs=1000,
    eval=50,
    num_gpu=1,
    dt=0.01,
    ntotal=512,
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
        # tanh the first 2 dimensions with a scale of 0.5 and the last dimension with a scale of 0.1
        x = jnp.concatenate([jnp.tanh(x[:2]) * 0.1, jnp.tanh(x[2:]) * 0.01], axis=0)
        return x

def policy(net: Network, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t)
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:].set(u))
    return dx, u

def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
    return x

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def control_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = dx.qfrc_applied
    return jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])), x)
    )

def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([1000, 1000, 1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1])), x)
    )

def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return _cfg.dt * jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([1000, 1000, 1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1])), x)
    )

def set_data(mx: mjx.Model, dx: mjx.Data, ctx: Context, key: jnp.ndarray) -> mjx.Data:
    ball_xy, key = op_and_split(lambda k: jax.random.uniform(k, minval=-0.5, maxval=0.5, shape=(2,)), key)
    ball_z = jnp.array([0.])
    ball_quat = jnp.array([1., 0., 0., 0.])
    ball_vel = jnp.array([0., 0., 0.])
    ball_ang_vel = jnp.array([0., 0., 0.])
    qpos = jnp.concatenate([ball_xy, ball_z, ball_quat], axis=0)
    qvel = jnp.concatenate([ball_vel, ball_ang_vel], axis=0)
    target_pos = jnp.array([0., 0., 0.])
    qpos = dx.qpos.at[:].set(qpos)
    qvel = dx.qvel.at[:].set(qvel)
    mocap_pos = dx.mocap_pos.at[:].set(target_pos)
    return dx.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)

def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([13, 32, 32, _cfg.mx.nv], key)

def is_terminal(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    time_limit =  (dx.time/ mx.opt.timestep) > (_cfg.ntotal - 1)
    return jnp.array([time_limit])

ctx = Context(
    _cfg,
    Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        set_data=set_data,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_det,
        is_terminal=is_terminal
    )
)


