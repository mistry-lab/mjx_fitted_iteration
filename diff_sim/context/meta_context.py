from __future__ import annotations

import collections.abc
from typing import Callable, get_type_hints, get_args, get_origin
from inspect import signature, Parameter
from dataclasses import dataclass
from functools import partial
import mujoco
from mujoco import mjx
from jax import numpy as jnp
import jax.tree_util
from jaxtyping import PyTree
from diff_sim.nn.base_nn import Network

@partial(jax.tree_util.register_dataclass,
         data_fields=['mx'],
         meta_fields=['lr', 'num_gpu', 'seed', 'nsteps', 'epochs', 'batch', 'samples', 'eval', 'dt', "gen_model"])
@dataclass(frozen=True)
class Config:
    lr: float     # learning rate
    num_gpu: int  # number of devices
    seed: int     # random seed
    nsteps: int   # simulation steps
    epochs: int   # training epochs
    batch: int    # batch size (number simulations)
    samples: int  # number of simulations per initial state
    eval: int     # visualization and saving checkpoints frequency
    dt: float     # simulation time step
    mx: mjx.Model # MJX model
    gen_model: Callable[[], mujoco.MjModel]  # Genereate Mujoco MjModel

class Callbacks:
    def __init__(
            self,
            run_cost: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            terminal_cost: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            control_cost: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            init_gen: Callable[[int, jnp.ndarray], jnp.ndarray],
            state_encoder: Callable[[mjx.Model,mjx.Data], jnp.ndarray],
            state_decoder: Callable[[jnp.ndarray], jnp.ndarray],
            gen_network: Callable[[int], Network],
            controller: Callable[[Network, mjx.Model, mjx.Data, jnp.ndarray],
            tuple[mjx.Data, jnp.ndarray]],
            loss_func: Callable[[PyTree, PyTree, jnp.ndarray, Context, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
    ):
        self.run_cost = run_cost           # running cost for trajectories
        self.terminal_cost = terminal_cost # terminal cost for trajectories
        self.control_cost = control_cost   # control cost for trajectories
        self.init_gen = init_gen           # initial state generator
        self.state_encoder = state_encoder # state encoder that is applied to all states
        self.state_decoder = state_decoder # state decoder that is applied for visualization
        self.gen_network = gen_network     # generates the neural network (policy or value). This can be a checkpoint
        self.controller = controller       # controller function that is called at each time step
        self.loss_func = loss_func         # loss function for the over all learning problem
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


class Context:
    def __init__(self, cfg: Config, cbs: Callbacks):
        self.cfg = cfg
        self.cbs = cbs
        assert jnp.isclose(cfg.dt, cfg.mx.opt.timestep, atol=1e-6)
        assert cfg.num_gpu <= jax.device_count()
        assert (cfg.batch * cfg.samples) % cfg.num_gpu == 0

