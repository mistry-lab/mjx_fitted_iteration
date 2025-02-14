import jax
import jax.numpy as jnp
import mujoco
from dataclasses import dataclass
import equinox as eqx
from jaxtyping import PyTree
from typing import Callable, Optional, Set
from mujoco import mjx

@jax.tree_util.register_static
@dataclass(frozen=True)
class Context:
    # Configuration
    lr: float                         # learning rate
    num_gpu: int                      # number of devices
    seed: int                         # random seed
    nsteps: int                       # mini-episode steps
    ntotal: int                       # total episode length
    epochs: int                       # training epochs
    batch: int                        # batch size
    samples: int                      # number of simulations per initial state
    eval: int                         # visualization/checkpoints frequency
    mx: mjx.Model                     # MJX model
    # Callbacks
    gen_model: Callable[[], mujoco.MjModel]
    run_cost : Callable[[mjx.Model, mjx.Data], jnp.ndarray]
    terminal_cost : Callable[[mjx.Model, mjx.Data], jnp.ndarray]
    control_cost : Callable[[mjx.Model, mjx.Data], jnp.ndarray]
    set_data : Callable[[mjx.Model, mjx.Data, jnp.ndarray], mjx.Data]
    gen_network : Callable[[int], eqx.Module]
    is_terminal : Callable[[mjx.Model, mjx.Data], jnp.ndarray]
    set_control : Callable[[mjx.Data, jnp.ndarray], mjx.Data]
    controller : Callable[
        [eqx.Module, mjx.Model, mjx.Data, jnp.ndarray],
        tuple[mjx.Data, jnp.ndarray]
    ]
    ctrl_dim: Optional[int]                  # Dimension of the control (not necessarily dx.ctrl)
    target_fields: Optional[Set[str]] = None # Target fields for finite differences 
    eps: Optional[float] = 1e-6              # Eps for finite differences 

    def __post_init__(self):
        assert self.num_gpu <= jax.device_count(), \
            "num_gpu cannot exceed number of available devices."
        assert (self.batch * self.samples) % self.num_gpu == 0, \
            "(batch*samples) must be divisible by num_gpu."
