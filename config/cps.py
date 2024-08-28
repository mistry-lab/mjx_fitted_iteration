import jax
from .config import Config
from jax import numpy as jnp

def _run_cst(x): return jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([0, 0, 0, 0])), x)
def _ctrl_cst(u): u = jnp.einsum('...ti,ij,...tj->...t', u, jnp.array([[1]]), u); u = u.at[..., -1].set(0); return u
def _terminal_cost(x): return jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([25, 100, 0.25, 1])), x)
def init_gen(batch, key):
    qp = jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1)
    qc = jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1)
    vc = jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1)
    vp = jax.random.uniform(key, (batch, 1), minval=-0.1, maxval=0.1)
    return jnp.concatenate([qc, qp, vc, vp], axis=1)


cartpole_cfg = Config(
    model_path='/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml',
    dims=[4, 64, 64, 1],
    lr=1e-3,
    seed=0,
    nsteps=100,
    epochs=100,
    batch=1000,
    vis=10,
    R=jnp.array([[1]]),
    act=jax.nn.softplus,
    run_cst=_run_cst,
    terminal_cst=_terminal_cost,
    ctrl_cst=_ctrl_cst,
    init_gen=init_gen,
    state_encoder=lambda x: x
)
