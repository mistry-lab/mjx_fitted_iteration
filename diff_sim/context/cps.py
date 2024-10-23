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


def policy(net: Network, mx: mjx.Model, dx: mjx.Data, key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx,dx)
    t = jnp.expand_dims(dx.time, axis=0)
    # returns the updated data and the control
    u = net(x, t)
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
    return dx, u

def run_cost(mx: mjx.Model,dx:mjx.Data) -> jnp.ndarray:
    # x^T Q x
    x = state_encoder(mx,dx)
    return  jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0., 0., 0., 0])), x))

def terminal_cost(mx: mjx.Model,dx:mjx.Data) -> jnp.ndarray:
    # x^T Q_f x
    x = state_encoder(mx,dx)
    return 10*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([25, 100, 0.25, 1])), x))

def control_cost(mx: mjx.Model,dx:mjx.Data) -> jnp.ndarray:
    # u^T R u
    x = dx.ctrl
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0.001])), x))

def init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
    xinits = jnp.concatenate([
        jax.random.uniform(key, (total_batch, 1), minval=-0.3, maxval=0.3),
        jax.random.uniform(key, (total_batch, 1), minval=jnp.pi+0.3, maxval=jnp.pi-0.3),
        jax.random.uniform(key, (total_batch, 1), minval=-0.1, maxval=0.1),
        jax.random.uniform(key, (total_batch, 1), minval=-0.1, maxval=0.1)
    ], axis=1).squeeze()
    return xinits


def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    return jnp.concatenate([dx.qpos, dx.qvel], axis=0)

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def gen_network(seed: int) -> Network:
    return Policy([5, 128, 128, 1], jax.random.PRNGKey(seed))

def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

ctx = Context(
    cfg=Config(
        lr=4e-3,
        num_gpu=1,
        seed=0,
        nsteps=125,
        epochs=400,
        batch=64,
        samples=1,
        eval=10,
        dt=0.01,
        mx= mjx.put_model(gen_model()),
        gen_model=gen_model,
    ),
    cbs=Callbacks(
        run_cost= run_cost,
        terminal_cost= terminal_cost,
        control_cost= control_cost,
        init_gen= init_gen,
        state_encoder= state_encoder,
        state_decoder= state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_stoch
    )
)
