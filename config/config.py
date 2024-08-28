from jax import numpy as jnp
from dataclasses import dataclass

@dataclass
class Config:
    model_path: str
    dims: list
    lr: float
    seed: int
    nsteps: int
    epochs: int
    batch: int
    vis: int
    R: jnp.ndarray
    act: callable
    run_cst: callable
    terminal_cst: callable
    ctrl_cst: callable
    init_gen: callable
    state_encoder: callable
