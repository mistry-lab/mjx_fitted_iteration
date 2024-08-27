import jax
from dataclasses import dataclass

@dataclass
class Config:
    model_path: str
    dims: list
    lr: float
    seed: int
    nsteps: int
    epochs: int
    act: callable
    run_cst: callable
    terminal_cst: callable
    ctrl_cst: callable
    init_gen: callable