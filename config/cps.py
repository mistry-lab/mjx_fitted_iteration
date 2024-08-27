import jax
from config import Config
from jax import numpy as jnp

def _run_cst(x): return 0
def _ctrl_cst(u): return jnp.einsum('...ti,ij,...tj->...t', u, jnp.array([[1, 0], [0, 0.1]]), u)
def _terminal_cost(x): return jnp.einsum('...ti,ij,...tj->...t', x, jnp.array([[100, 0], [0, 1]]), x)
def _init_gen(key):
    qp = jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1)
    qc = jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1)
    vc = jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1)
    vp = jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1)
    return jnp.concatenate([qc, qp, vc, vp], axis=0)


cartpole_cfg = Config(
    model_path='/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml',
    dims=[4, 64, 64, 1],
    lr=1e-3,
    seed=0,
    nsteps=100,
    epochs=1000,
    act=jax.nn.softplus,
    run_cst=_run_cst,
    terminal_cst=_terminal_cost,
    ctrl_cst=_ctrl_cst,
    init_gen=_init_gen
)