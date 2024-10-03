import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

model_path = os.path.join(os.path.dirname(__file__), '../xmls/cartpole.xml')


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
        t = t if t.ndim == 1 else t.reshape(1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


def policy(
        x: jnp.ndarray, t: jnp.ndarray, net: Network, cfg: Config, mx: mjx.Model, dx: mjx.Data
) -> jnp.ndarray:
    # returns the control action
    return net(x, t)

def run_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q x
    return  jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0., 0., 0., 0])), x))

def terminal_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q_f x
    return 10*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([25, 100, 0.25, 1])), x))

def control_cost(x: jnp.ndarray) -> jnp.ndarray:
    # u^T R u
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0.001])), x))

def init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
    xinits = jnp.concatenate([
        jax.random.uniform(key, (total_batch, 1), minval=-0.3, maxval=0.3),
        jax.random.uniform(key, (total_batch, 1), minval=jnp.pi+0.3, maxval=jnp.pi-0.3),
        jax.random.uniform(key, (total_batch, 1), minval=-0.1, maxval=0.1),
        jax.random.uniform(key, (total_batch, 1), minval=-0.1, maxval=0.1)
    ], axis=1).squeeze()
    return xinits


def coder(x: jnp.ndarray) -> jnp.ndarray:
    # encode and decode the state. Do nothing in this case
    return x

def gen_network(seed: int) -> Network:
    return Policy([5, 128, 128, 1], jax.random.PRNGKey(seed))


ctx = Context(
    cfg=Config(
        lr=4e-3,
        seed=0,
        nsteps=125,
        epochs=400,
        batch=1000,
        samples=1,
        vis=10,
        dt=0.01,
        path=model_path,
        mx=mjx.put_model(mujoco.MjModel.from_xml_path(model_path))),
    cbs=Callbacks(
        run_cost= run_cost,
        terminal_cost= terminal_cost,
        control_cost= control_cost,
        init_gen= init_gen,
        state_encoder= coder,
        state_decoder= coder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_stoch
    )
)
