
import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

model_path = os.path.join(os.path.dirname(__file__), '../xmls/point_mass_tendon.xml')
def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

_cfg = Config(
    lr=4e-3,
    seed=4,
    batch=200,
    samples=1,
    epochs=1000,
    eval=50,
    num_gpu=1,
    dt=0.0005,
    ntotal=2560,
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
        # t = t if t.ndim == 1 else t.reshape(1)
        # x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x).squeeze()
        x = jnp.tanh(x) * 1
        return x


def policy(net: Network, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t) + dx.qpos[:3]
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
    return dx, u

def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    return jnp.concatenate([dx.qpos, dx.qvel], axis=0)

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def control_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = dx.ctrl
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0.0001, 0.0001, 0.0001])), x))

def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0, 0, 0, 1000, 1000, 0.1, 0.1, 0.1, 1, 1])), x))

def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)
    return 0.01*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0, 0, 0, 1000, 100, 0.1, 0.1, 0.1, 1, 1])), x))

def init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
    angs = jax.random.uniform(key, (total_batch, 3), minval=-0.0, maxval=0.0)
    obj_x = jax.random.uniform(key, (total_batch, 1), minval=.1, maxval=0.1)
    obj_y = jax.random.uniform(key, (total_batch, 1), minval=0.1, maxval=0.6)
    qpos = jnp.concatenate([angs, obj_x, obj_y], axis=1)
    qvel = jax.random.uniform(key, (total_batch, _cfg.mx.nv), minval=-0.1, maxval=0.1)
    return jnp.concatenate([qpos, qvel], axis=1)

def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([_cfg.mx.nq + _cfg.mx.nv, 6, 6, _cfg.mx.nu], key)

def is_terminal(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    time_limit =  (dx.time/ mx.opt.timestep) > (_cfg.ntotal - 1)
    return jnp.array([time_limit])

ctx = Context(
    _cfg,
    Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        init_gen=init_gen,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_det,
        is_terminal=is_terminal
    )
)


