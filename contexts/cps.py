import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from .meta_context import Config, Callbacks, Context

model_path = os.path.join(os.path.dirname(__file__), '../xmls/cartpole.xml')

try:
    base_path = os.path.dirname(__file__)
except NameError:
    base_path = os.getcwd()

base_path = os.path.join(base_path, '../xmls')


class Policy(eqx.Module):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], use_bias=True) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x, t):
        t = t if t.ndim == 1 else t.reshape(1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

    @staticmethod
    @eqx.filter_jit
    def make_step(optim, model, state, loss, x_init, ctx):
        params, static = eqx.partition(model, eqx.is_array)
        loss_value, grads = jax.value_and_grad(loss)(params, static, x_init, ctx)
        updates, state = optim.update(grads, state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, loss_value

    @staticmethod
    def loss_fn(params, static, x, y):
        model = eqx.combine(params, static)
        pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
        y = y.reshape(-1, 1)
        return jnp.mean(jnp.square(pred - y))

def policy(
        x: jnp.ndarray, t: jnp.ndarray, net: eqx.Module, cfg: Config, mx: mjx.Model, dx: mjx.Data
) -> jnp.ndarray:
    return net(x, t)

def run_cost(x: jnp.ndarray) -> jnp.ndarray:
    return  jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0., 0., 0., 0])), x))

def terminal_cost(x: jnp.ndarray) -> jnp.ndarray:
    return 10*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([25, 100, 0.25, 1])), x))

def control_cost(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0.001])), x))

def init_gen(batch: int, key: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([
        jax.random.uniform(key, (batch, 1), minval=-0.3, maxval=0.3),
        jax.random.uniform(key, (batch, 1), minval=jnp.pi+0.3, maxval=jnp.pi-0.3),
        jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1),
        jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1)
    ], axis=1).squeeze()

def coder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def gen_network() -> eqx.Module:
    return Policy([5, 128, 128, 1], jax.random.PRNGKey(0))


ctx = Context(
    cfg=Config(
        lr=4e-3,
        seed=0,
        nsteps=120,
        epochs=400,
        batch=1000,
        vis=10,
        dt=0.01,
        R=jnp.array([[1]]),
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
        controller=policy
    )
)
