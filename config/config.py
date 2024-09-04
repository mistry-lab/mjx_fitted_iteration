from typing import Callable, List, NamedTuple
from dataclasses import dataclass
from functools import partial
from jax import numpy as jnp
import jax.tree_util

@partial(jax.tree_util.register_dataclass,
         data_fields=['R'],
         meta_fields=['model_path', 'dims', 'lr', 'seed', 'nsteps', 'epochs', 'batch', 'vis'])
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
    R: jnp.ndarray

class Callbacks(NamedTuple):
    run_cost: Callable[[jnp.ndarray], jnp.ndarray]
    terminal_cost: Callable[[jnp.ndarray], jnp.ndarray]
    control_cost: Callable[[jnp.ndarray], jnp.ndarray]
    init_gen: Callable[[int, jnp.ndarray], jnp.ndarray]
    state_encoder: Callable[[jnp.ndarray], jnp.ndarray]
    net: Callable[[jnp.ndarray], jnp.ndarray]

Context = NamedTuple('Context', [('cfg', Config), ('cbs', Callbacks)])