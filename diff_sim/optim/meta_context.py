import jax
import jax.numpy as jnp
import mujoco
from dataclasses import dataclass
from typing import Callable, Optional, Set
from mujoco import mjx


@jax.tree_util.register_static
@dataclass(frozen=True)
class Context:
    # Callbacks
    gen_model: Callable[[], mujoco.MjModel]
    running_cost: Callable[[mjx.Data], jnp.ndarray]
    terminal_cost: Callable[[mjx.Data], jnp.ndarray]
    set_control: Callable[[mjx.Data, jnp.ndarray], mjx.Data]
    set_target: Callable[[mjx.Data], jnp.ndarray]

    # Configuration
    lr: float  # learning rate
    nsteps: int  # total episode length
    epochs: int  # training epochs
    mx: mjx.Model  # MJX model
    num_gpu: Optional[int] = 1  # number of devices
    seed: Optional[int] = 0  # random seed
    batch: Optional[int] = 1  # batch size

    # fd parameters
    ctrl_dim: Optional[int] = 0  # Dimension of the control (not necessarily dx.ctrl)
    target_fields: Optional[Set[str]] = None  # Target fields for finite differences
    eps: Optional[float] = 1e-6  # Eps for finite differences

    # ilqr parameters
    ddp: Optional[int] = False  # DDP (use 2nd derivatives)
    reg: Optional[float] = 1e-6  # Regularisation term to Q_uu

    def __post_init__(self):
        assert (
            self.num_gpu <= jax.device_count()
        ), "num_gpu cannot exceed number of available devices."
