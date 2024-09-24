from typing import Callable, List, NamedTuple
from dataclasses import dataclass
from functools import partial
from mujoco import mjx
from jax import numpy as jnp
import jax.tree_util
import equinox as eqx

@partial(jax.tree_util.register_dataclass,
         data_fields=['R', 'horizon', 'mx'],
         meta_fields=['lr', 'seed', 'nsteps', 'epochs', 'batch', 'vis', 'dt', 'path'])
@dataclass(frozen=True)
class Config:
    lr: float
    seed: int
    nsteps: int
    epochs: int
    batch: int
    vis: int
    dt: float
    path: str
    R: jnp.ndarray
    mx: mjx.Model

class Callbacks:
    def __init__(
            self,
            run_cost: Callable[[jnp.ndarray], jnp.ndarray],
            terminal_cost: Callable[[jnp.ndarray], jnp.ndarray],
            control_cost: Callable[[jnp.ndarray], jnp.ndarray],
            init_gen: Callable[[int, jnp.ndarray], jnp.ndarray],
            state_encoder: Callable[[jnp.ndarray], jnp.ndarray],
            state_decoder: Callable[[jnp.ndarray], jnp.ndarray],
            gen_network: Callable[[None], eqx.Module],
            controller: Callable[[jnp.ndarray, jnp.ndarray,eqx.Module, Config, mjx.Model, mjx.Data], jnp.ndarray]
    ):
        self.run_cost = run_cost
        self.terminal_cost = terminal_cost
        self.control_cost = control_cost
        self.init_gen = init_gen
        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.gen_network = gen_network
        self.controller = controller


class Context:
    def __init__(self, cfg: Config, cbs: Callbacks):
        self.cfg = cfg
        self.cbs = cbs
        assert jnp.all(jnp.linalg.eigh(self.cfg.R)[0] > 0), (
            "R should be positive definite."
        )
