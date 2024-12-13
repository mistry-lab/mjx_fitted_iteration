import equinox
import jax
import jax.numpy as jnp
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable

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

def set_control(dx, u):
    # u = jnp.tanh(u) * 0.5
    return dx.replace(ctrl=dx.ctrl.at[:].set(u))


@jax.custom_vjp
def step_dynamics(mx, dx, ctrl):
    """
    Black-box MuJoCo step that returns the next dx.
    Args:
      mx: MuJoCo model
      dx: current MuJoCo data (qpos, qvel, etc.)
      ctrl: controls to apply
      set_control_fn: function that writes 'ctrl' into dx.ctrl
    Returns:
      dx_next (MuJoCo data)
    """
    dx_applied = set_control(dx, ctrl)
    dx_next = mjx.step(mx, dx_applied)
    return dx_next


def step_dynamics_fwd(mx, dx, ctrl):
    dx_next = step_dynamics(mx, dx, ctrl)
    # Save (dx, ctrl) for the backward pass
    return dx_next, (mx, dx, ctrl)


def step_dynamics_bwd(res, g):
    """
    Custom backward for the MuJoCo step. The 'g' is the cotangent w.r.t. dx_next.
    We want to define partial derivatives only for dx, ctrl via finite differences.
    """
    mx_in, dx_in, ctrl_in = res  # from forward pass
    g_dx_next = g  # same structure as dx_next
    # We'll do finite differences on dx_in and ctrl_in to figure out
    # how dx_next changes w.r.t. those inputs.
    eps = 1e-5
    # Flatten dx_in into e.g. qpos and qvel if needed, or treat them as single objects.
    # For simplicity, let's assume we only do FD on qpos, qvel, ctrl.
    # If dx has sensors, contact, etc., you'd handle them likewise.
    # We'll separate logic for dx vs. ctrl.
    # 1) Finite difference wrt ctrl_in
    ctrl_shape = ctrl_in.shape
    ctrl_flat = ctrl_in.ravel()

    def fd_ctrl_plusminus(idx):
        eps = 1e-5
        # Create the epsilon-perturbed control
        e = jnp.zeros_like(ctrl_flat).at[idx].set(eps)
        ctrl_plus = (ctrl_flat + e).reshape(ctrl_shape)

        # Apply control & step
        dx_plus = set_control(dx_in, ctrl_plus)
        dx_next_plus = mjx.step(mx_in, dx_plus)

        # 1) Forward difference across all fields of dx
        dx_diff = jax.tree_map(lambda new, old: (new - old) / eps, dx_next_plus, dx_in)

        def multiply_conditionally(diff_tree, grad_tree):
            def maybe_multiply(d_leaf, g_leaf):
                # Check if g_leaf has float0 dtype (meaning zero-sized gradient)
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    jnp.zeros_like(d_leaf)
                else:
                    return d_leaf * g_leaf

            return jax.tree_map(maybe_multiply, diff_tree, grad_tree)

        diff_times_g = multiply_conditionally(dx_diff, g_dx_next)

        return diff_times_g  # shape ()

    ctrl_indices = jnp.arange(ctrl_flat.size)
    d_ctrl = jax.vmap(fd_ctrl_plusminus)(ctrl_indices)
    d_dx_in = jax.tree_map(jnp.ones_like, dx_in)  # ignoring gradient wrt dx
    return (None, d_dx_in, d_ctrl)

step_dynamics.defvjp(step_dynamics_fwd, step_dynamics_bwd)

@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, running_cost_fn, terminal_cost_fn, U):
    """
    Simulate a trajectory using the custom FD-based dynamics for partial derivatives,
    but normal AD for the cost function.
    """
    def step_fn(dx, ctrl):
        dx_next = step_dynamics(mx, dx, ctrl)
        c = running_cost_fn(dx_next)   # normal AD logic
        state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return dx_next, (state, c)
    # Initialize MuJoCo data
    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost

def make_loss(mx, qpos_init, running_cost_fn, terminal_cost_fn):
    """
    The loss function calls simulate_trajectory,
    which uses FD for dynamics partial derivatives,
    but normal AD for cost.
    """
    def loss(U):
        _, total_cost = simulate_trajectory(
            mx, qpos_init,
            running_cost_fn, terminal_cost_fn,
            U
        )
        return total_cost
    return loss


@dataclass
class PMP:
    loss: Callable[[jnp.ndarray], float]
    def grad_loss(self, U: jnp.ndarray) -> jnp.ndarray:
        # Let JAX do its magic. Internally it uses the custom_vjp for step_dynamics.
        return jax.grad(self.loss)(U)
    def solve(self, U0: jnp.ndarray, learning_rate=1e-2, tol=1e-6, max_iter=100):
        U = U0
        for i in range(max_iter):
            g = self.grad_loss(U)
            U_new = U - learning_rate * g
            f_val = self.loss(U_new)
            print(f"Iteration {i}: cost={f_val}")
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                return U_new
            U = U_new
        return U