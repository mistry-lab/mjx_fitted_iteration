import jax
from jaxtyping import PyTree
import jax.numpy as jnp
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable
import equinox
from diff_sim.nn.base_nn import Network

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, params, static, length):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        qpos_init: initial positions (array)
        set_control_fn: fn(dx, u) -> dx to apply controls
        running_cost_fn: fn(dx, u) -> cost (float)
        terminal_cost_fn: fn(dx) -> cost (float)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states
        total_cost: scalar total cost
    """
    model = equinox.combine(params, static)

    def step_fn(dx, _):
        dx = set_control_fn(dx, model)
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, length=length)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost


def make_loss(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, length):
    """
    Create a loss function that only takes U as input.
    """
    def loss(params, static):
        _, total_cost = simulate_trajectory(
            mx, qpos_init,
            set_control_fn, running_cost_fn, terminal_cost_fn,
            params, static, length
        )
        return total_cost
    return loss


@dataclass
class Policy:
    loss: Callable[[PyTree, PyTree], float]
    grad_loss: Callable[[PyTree, PyTree], jnp.ndarray]

    def solve(self, model: equinox.Module, optim, state, max_iter=100):
        """
        Gradient descent on the control trajectory.

        U0: initial guess (N, nu)
        Returns: optimized U
        """
        for i in range(max_iter):
            params, static = equinox.partition(model, equinox.is_array)
            g = self.grad_loss(params, static)
            f_val = self.loss(params, static)
            updates, state = optim.update(g, state, model)
            model = equinox.apply_updates(model, updates)
            print(f"Iteration {i}: cost={f_val}")
        return model

