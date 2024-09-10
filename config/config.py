from typing import Callable, List, NamedTuple
from dataclasses import dataclass
from functools import partial
from jax import numpy as jnp
import jax.tree_util
import equinox as eqx

@partial(jax.tree_util.register_dataclass,
         data_fields=['R'],
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

# class Callbacks(NamedTuple):
#     run_cost: Callable[[jnp.ndarray], jnp.ndarray]
#     terminal_cost: Callable[[jnp.ndarray], jnp.ndarray]
#     control_cost: Callable[[jnp.ndarray], jnp.ndarray]
#     init_gen: Callable[[int, jnp.ndarray], jnp.ndarray]
#     state_encoder: Callable[[jnp.ndarray], jnp.ndarray]
#     net: Callable[[jnp.ndarray], jnp.ndarray]

# Callbacks class with mutable `net`
class Callbacks:
    def __init__(self, 
                 run_cost: Callable[[jnp.ndarray], jnp.ndarray],
                 terminal_cost: Callable[[jnp.ndarray], jnp.ndarray],
                 control_cost: Callable[[jnp.ndarray], jnp.ndarray],
                 init_gen: Callable[[int, jnp.ndarray], jnp.ndarray],
                 state_encoder: Callable[[jnp.ndarray], jnp.ndarray],
                 net: eqx.Module):
        self.run_cost = run_cost
        self.terminal_cost = terminal_cost
        self.control_cost = control_cost
        self.init_gen = init_gen
        self.state_encoder = state_encoder
        self.net = net  # `net` is mutable because it's an Equinox module

# Context class to hold Config and Callbacks
class Context:
    def __init__(self, cfg: Config, cbs: Callbacks):
        self.cfg = cfg  # Immutable Config
        self.cbs = cbs  # Callbacks with a mutable `net`

# Context = NamedTuple('Context', [('cfg', Config), ('cbs', Callbacks)])