import jax
from jax import numpy as jnp
import equinox as eqx
from .meta_context import Config, Callbacks, Context
import os

try:
    # This works when __file__ is defined (e.g., in regular scripts)
    base_path = os.path.dirname(__file__)
except NameError:
    # Fallback to current working directory (e.g., in interactive sessions)
    base_path = os.getcwd()

base_path = os.path.join(base_path, '../xmls')

class ValueFunc(eqx.Module):
    layers: list
    act: lambda x: jax.nn.relu(x)

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(len(dims) - 1)]
        self.act = jax.nn.softplus

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


ctx = Context(cfg=Config(
    model_path=os.path.join(base_path, 'cartpole.xml'),
    dims=[4, 64, 64, 1],
    lr=4e-3,
    seed=0,
    nsteps=100,
    epochs=400,
    batch=1000,
    vis=10,
    dt=0.01,
    R=jnp.array([[1]]),
    horizon=jnp.arange(0, 100 + 0.01, 0.01) + 0.01
    ),cbs=Callbacks(
        run_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([0, 0, 0, 0])), x),
        terminal_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([25, 100, 0.25, 1])), x),
        control_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.array([[1]]), x).at[..., -1].set(0),
        init_gen= lambda batch, key: jnp.concatenate([
            jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1),
            jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1),
            jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1),
            jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1)
        ], axis=1).squeeze(),
    state_encoder=lambda x: x,
    gen_network=lambda :ValueFunc([5, 64, 64, 1], jax.random.PRNGKey(0))
    )
)