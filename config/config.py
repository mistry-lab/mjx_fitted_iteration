from typing import Callable, List, NamedTuple
from dataclasses import dataclass
from functools import partial
from jax import numpy as jnp
import jax.tree_util
import equinox as eqx

@partial(jax.tree_util.register_dataclass,
         data_fields=['R', 'horizon'],
         meta_fields=['model_path', 'dims', 'lr', 'seed', 'nsteps', 'epochs', 'batch', 'vis', 'dt'])
@dataclass(frozen=True)
class Config:
    model_path: str
    dims: List[int]
    lr: float
    seed: int
    nsteps: int
    epochs: int
    batch: int
    vis: int
    dt: float
    R: jnp.ndarray
    horizon: jnp.ndarray


class Callbacks:
    def __init__(
            self,
            run_cost: Callable[[jnp.ndarray], jnp.ndarray],
            terminal_cost: Callable[[jnp.ndarray], jnp.ndarray],
            control_cost: Callable[[jnp.ndarray], jnp.ndarray],
            init_gen: Callable[[int, jnp.ndarray], jnp.ndarray],
            state_encoder: Callable[[jnp.ndarray], jnp.ndarray],
            net: eqx.Module
    ):
        self.run_cost = run_cost
        self.terminal_cost = terminal_cost
        self.control_cost = control_cost
        self.init_gen = init_gen
        self.state_encoder = state_encoder
        self.net = net

class Context:
    def __init__(self, cfg: Config, cbs: Callbacks):
        self.cfg = cfg
        self.cbs = cbs

        assert self.cfg.horizon[0] > 0, (
            "First time step should be above 0 as mj_step should be called to init data (this increments time)."
        )
        assert jnp.all(jnp.linalg.eigh(self.cfg.R)[0] > 0), (
            "R should be positive definite."
        )

