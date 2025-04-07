import time
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.math import quat_integrate, quat_sub
from dataclasses import dataclass
from typing import Callable, Optional, Set
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax._src.util import unzip2
from mujoco import mjx

from diff_sim.optim.simulation.fd_cache import FDCache
from diff_sim.optim.meta_context import Context

def make_loss_fn(
        qpos_init: jnp.ndarray, #  TODO: include inside Context.
        step_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
        ctx = Context
):
    running_cost_fn = ctx.running_cost
    terminal_cost_fn = ctx.terminal_cost
    mx = ctx.mx

    @jax.jit
    def simulate_trajectory(U: jnp.ndarray):
        dx0 = mjx.make_data(mx)
        dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
        dx0 = mjx.step(mx, dx0)  # initial sync

        def scan_body(dx, u):
            dx_next = step_fn(dx, u)
            cost_t = running_cost_fn(dx_next)
            state_t = jnp.concatenate([dx_next.qpos, dx_next.qvel])
            return dx_next, (state_t, cost_t)

        dx_final, (states, costs) = jax.lax.scan(scan_body, dx0, U)
        # costs = costs.at[-1].set(0.)
        total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
        return states, total_cost

    def loss(U: jnp.ndarray):
        state, total_cost = simulate_trajectory(U)
        return total_cost, state

    return loss


@dataclass
class PMP:
    """
    A gradient-based optimizer for the FD-based MuJoCo trajectory problem.
    """
    loss: Callable[[jnp.ndarray], float]

    def grad_loss(self, U: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(self.loss)(U)

    def solve(
            self,
            U0: jnp.ndarray,
            learning_rate: float = 1e-2,
            tol: float = 1e-6,
            max_iter: int = 100
    ):
        U = U0
        for i in range(max_iter):
            now = time.time()
            g = self.grad_loss(U)
            U_new = U - learning_rate * g
            cost_val = self.loss(U_new)
            print(f"Time: {time.time() - now}")
            print(f"\n--- Iteration {i} ---")
            print(f"Cost={cost_val}")
            print(f"||grad||={jnp.linalg.norm(g)}")
            # Check for convergence
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                print(f"Converged at iteration {i}.")
                return U_new
            U = U_new
        return U
