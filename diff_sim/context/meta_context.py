from __future__ import annotations

import collections.abc
from typing import Callable, get_type_hints, get_args, get_origin, Optional, Set, List
from inspect import signature, Parameter
from dataclasses import dataclass
from functools import partial
import mujoco
from mujoco import mjx
from jax import numpy as jnp
import jax.tree_util
from jaxtyping import PyTree
from diff_sim.nn.base_nn import Network
from jax.flatten_util import ravel_pytree
import numpy as np
from jax._src.util import unzip2

@partial(jax.tree_util.register_dataclass,
         data_fields=['mx'],
         meta_fields=['lr', 'num_gpu', 'seed', 'nsteps', 'ntotal', 'epochs', 'batch', 'samples', 'eval', 'dt','eps', 'gen_model'])
@dataclass(frozen=True)
class Config:
    lr: float     # learning rate
    num_gpu: int  # number of devices
    seed: int     # random seed
    nsteps: int   # mini-episode steps
    ntotal: int   # total episode length
    epochs: int   # training epochs
    batch: int    # batch size (number simulations)
    samples: int  # number of simulations per initial state
    eval: int     # visualization and saving checkpoints frequency
    dt: float     # simulation time step
    mx: mjx.Model # MJX model
    eps: float    # FD sensitivity parameter
    gen_model: Callable[[], mujoco.MjModel]  # Genereate Mujoco MjModel

class Callbacks:
    def __init__(
            self,
            run_cost: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            terminal_cost: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            control_cost: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            set_data: Callable[[mjx.Model,mjx.Data,Context,jnp.ndarray], mjx.Data],
            state_encoder: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            state_decoder: Callable[[jnp.ndarray], jnp.ndarray],
            gen_network: Callable[[int], Network],
            controller: Callable[[Network, mjx.Model, mjx.Data, jnp.ndarray], jnp.ndarray],
            set_control:Callable[[mjx.Data, jnp.ndarray],mjx.Data],
            loss_func: Callable[[PyTree, PyTree, mjx.Data, Context, jnp.ndarray],
            tuple[jnp.ndarray, tuple[jnp.ndarray, mjx.Data, jnp.ndarray, jnp.ndarray]]],
            is_terminal: Callable[[mjx.Model, mjx.Data], jnp.ndarray]
    ):
        self.run_cost = run_cost           # running cost for trajectories
        self.terminal_cost = terminal_cost # terminal cost for trajectories
        self.control_cost = control_cost   # control cost for trajectories
        self.set_data = set_data           # initial state generator
        self.state_encoder = state_encoder # state encoder that is applied to all states
        self.state_decoder = state_decoder # state decoder that is applied for visualization
        self.gen_network = gen_network     # generates the neural network (policy or value). This can be a checkpoint
        self.controller = controller       # controller function that is called at each time step
        self.set_control = set_control     # apply command u from controller to the state dx
        self.loss_func = loss_func         # loss function for the over all learning problem
        self.is_terminal = is_terminal     # terminal condition for the episode
        self._validate_callbacks()         # type check the callbacks


    def _validate_callbacks(self):
        hints = get_type_hints(self.__init__)
        for attr_name, expected_type in hints.items():
            if attr_name == 'return':
                continue  # Skip return type

            func = getattr(self, attr_name)
            if get_origin(expected_type) is not collections.abc.Callable:
                raise TypeError(
                    f"Expected type of attribute '{attr_name}' is not a callable type."
                )

            expected_args_types, expected_return_type = get_args(expected_type)

            func_signature = signature(func)
            func_params = list(func_signature.parameters.values())

            # Check number of parameters
            if len(func_params) != len(expected_args_types):
                raise TypeError(
                    f"Function '{attr_name}' expects {len(expected_args_types)} parameters, "
                    f"but received {len(func_params)}."
                )

            # Check parameter types
            for param, expected_arg_type in zip(func_params, expected_args_types):
                # Handle cases where parameter has no annotation
                if param.annotation is Parameter.empty:
                    raise TypeError(
                        f"Parameter '{param.name}' in function '{attr_name}' lacks a type annotation."
                    )
                if param.annotation != expected_arg_type:
                    raise TypeError(
                        f"Parameter '{param.name}' in function '{attr_name}' has type {param.annotation}, "
                        f"expected {expected_arg_type}."
                    )

            # Check return type
            if func_signature.return_annotation is Parameter.empty:
                raise TypeError(
                    f"Function '{attr_name}' lacks a return type annotation."
                )
            if func_signature.return_annotation != expected_return_type:
                raise TypeError(
                    f"Function '{attr_name}' has return type {func_signature.return_annotation}, "
                    f"expected {expected_return_type}."
                )


# -------------------------------------------------------------
# Finite-difference cache
# -------------------------------------------------------------
@dataclass(frozen=True)
class FDCache:
    """Holds all the precomputed info needed by the custom FD-based backward pass."""
    unravel_dx: Callable[[jnp.ndarray], mjx.Data]
    sensitivity_mask: jnp.ndarray
    inner_idx: jnp.ndarray
    dx_size: int
    num_u_dims: int
    eps: float = 1e-6
 

class Context:
    def __init__(self, cfg: Config, cbs: Callbacks, target_fields: Optional[Set[str]] = None):
        self.cfg = cfg
        self.cbs = cbs
        self.fd_cache = self._build_fd_cache(target_fields)
        assert jnp.isclose(cfg.dt, cfg.mx.opt.timestep, atol=1e-6)
        assert cfg.num_gpu <= jax.device_count()
        assert (cfg.batch * cfg.samples) % cfg.num_gpu == 0

    def _build_fd_cache(
        self,
        target_fields: Optional[Set[str]] = None
    ) -> FDCache:
        """
        Build a cache containing:
        - Flatten/unflatten for dx_ref
        - The mask for relevant FD indices (e.g. qpos, qvel, ctrl)
        - The shape info for control
        """
        dx_ref = mjx.make_data(self.cfg.mx)
        u_ref = jnp.zeros((self.cfg.mx.nu,))
        if target_fields is None:
            target_fields = {"qpos", "qvel", "ctrl"}
    
        # Flatten dx
        dx_array, unravel_dx = ravel_pytree(dx_ref)
        dx_size = dx_array.shape[0]
    
        # Flatten control
        u_ref_flat = u_ref.ravel()
        num_u_dims = u_ref_flat.shape[0]
    
        # Gather leaves for qpos, qvel, ctrl
        leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_ref))
        sizes, _ = unzip2((jnp.size(leaf), jnp.shape(leaf)) for (_, leaf) in leaves_with_path)
        indices = tuple(np.cumsum(sizes))
    
        idx_target_state = []
        for i, (path, leaf_val) in enumerate(leaves_with_path):
            # Check if any level in the path has a 'name' that is in target_fields
            name_matches = any(
                getattr(level, 'name', None) in target_fields
                for level in path
            )
            if name_matches:
                idx_target_state.append(i)
    
        def leaf_index_range(leaf_idx):
            start = 0 if leaf_idx == 0 else indices[leaf_idx - 1]
            end = indices[leaf_idx]
            return np.arange(start, end)
    
        # Combine all relevant leaf sub-ranges
        inner_idx_list = []
        for i in idx_target_state:
            inner_idx_list.append(leaf_index_range(i))
        inner_idx = np.concatenate(inner_idx_list, axis=0)
        inner_idx = jnp.array(inner_idx, dtype=jnp.int32)
    
        # Build the sensitivity mask
        sensitivity_mask = jnp.zeros_like(dx_array).at[inner_idx].set(1.0)

        # Delete mjx data
        del dx_ref
    
        return FDCache(
            unravel_dx = unravel_dx,
            sensitivity_mask = sensitivity_mask,
            inner_idx = inner_idx,
            dx_size = dx_size,
            num_u_dims = num_u_dims,
            eps = self.cfg.eps
        )
