import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from typing import Callable
from dataclasses import dataclass
from mujoco import mjx
import equinox
from jax._src.util import safe_zip, unzip2
import numpy as np


class GetAttrKey:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"GetAttrKey(name='{self.name}')"


def filter_state_data(dx: mjx.Data):
    """Select a subset of the mjx.Data fields (qpos, qvel, qacc, sensordata, etc.)
       that you want to include in your derivative.
       Modify this function as needed.
    """
    return (
        dx.qpos,
        dx.qvel,
        dx.qacc,
        dx.sensordata,
        dx.mocap_pos,
        dx.mocap_quat
    )


def make_step_fn(
        mx,
        set_control_fn: Callable,
        eps: float = 1e-6
):
    """
    Create a custom_vjp step function that takes a single argument u.
    The function 'set_control_fn' is a user-defined way of writing u into dx.
    Finite differences (FD) are performed w.r.t. u *across all elements of dx*.
    """

    @jax.custom_vjp
    def step_fn(dx: jnp.ndarray, u: jnp.ndarray):
        """
        Single-argument step function:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
          3) Returns dx_next (which is also a pytree).
        """
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
        return dx_next

    def step_fn_fwd(dx, u):
        dx_next = step_fn(dx, u)
        return dx_next, (dx, u, dx_next)

    def step_fn_bwd(res, g):
        """
        FD-based backward pass. We approximate ∂ dx_next / ∂ u and
        ∂ dx_next / ∂ dx, then do the chain rule with the cotangent 'g'.
        """
        dx_in, u_in, dx_out = res

        # Map float0 leaves in 'g' to zeros, which is a standard JAX trick
        def map_g_to_dinput(diff_tree, grad_tree):
            def fix_leaf(d_leaf, g_leaf):
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    return jnp.zeros_like(d_leaf)
                else:
                    return g_leaf

            return jax.tree_map(fix_leaf, diff_tree, grad_tree)

        # Filter out float0 leaves in the cotangent
        mapped_g = map_g_to_dinput(dx_in, g)

        # Flatten dx_in, dx_out, and the cotangent g
        dx_array, unravel_dx = ravel_pytree(dx_in)
        dx_out_array, _ = ravel_pytree(dx_out)
        g_array, _ = ravel_pytree(mapped_g)

        # Flatten the input control
        u_in_flat = u_in.ravel()
        num_u_dims = u_in_flat.shape[0]

        # Identify the indices in dx_array that you actually care about
        # (e.g. qpos, qvel, ctrl). This is your "sensitivity subset."
        leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_in))
        target_fields = {'qpos', 'qvel', 'ctrl'}

        # We'll gather the size and shape for each leaf
        sizes, shapes = unzip2(
            (jnp.size(leaf), jnp.shape(leaf)) for (_, leaf) in leaves_with_path
        )
        indices = tuple(np.cumsum(sizes))

        # Find which leaf indices correspond to qpos, qvel, ctrl, etc.
        # E.g. if the path is (someKey, 'qpos'), then we want it.
        idx_target_state = []
        for i, (path, leaf_val) in enumerate(leaves_with_path):
            # If any level in path is named in target_fields, we keep it
            name_matches = any(
                getattr(level, 'name', None) in target_fields
                for level in path
            )
            if name_matches:
                idx_target_state.append(i)

        # Flatten out the "inner" indices within each leaf
        # so we get a 1D list of indices in dx_array that matter.
        def leaf_index_range(leaf_idx):
            # For leaf i, the flattened slice is [start_i : end_i]
            start = 0 if leaf_idx == 0 else indices[leaf_idx - 1]
            end = indices[leaf_idx]
            return np.arange(start, end)

        inner_idx_list = []
        for i in idx_target_state:
            inner_idx_list.append(leaf_index_range(i))
        # Combine all relevant sub-ranges into one 1D array
        inner_idx = np.concatenate(inner_idx_list, axis=0)
        inner_idx = jnp.array(inner_idx, dtype=jnp.int32)

        # Our "sensitivity mask" is 1.0 for these relevant indices, 0.0 otherwise
        sensitivity_mask = jnp.zeros_like(dx_array).at[inner_idx].set(1.0)

        # ====== Derivative wrt. control U (finite differences) ======
        def fdu_plus(i):
            """Compute [dx_next(u + e_i * eps) - dx_next(u)] / eps in flattened form."""
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn(dx_in, u_in_eps)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # Vectorized over each dimension of u
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))  # shape = (num_u_dims, dx_dim)

        # ====== Derivative wrt. state dx_in (finite differences) ======
        # A *big* speedup is to only vmap over the subset of indices we need (inner_idx),
        # instead of all of dx_array.shape[0].

        def fdx_for_index(idx):
            """Compute [dx_next(dx_in + e_idx) - dx_next(dx_in)] / eps in flattened form,
               only for the selected index in dx_array."""
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # vmap over the small subset "inner_idx" (the rows we actually need)
        Jx_rows = jax.vmap(fdx_for_index)(inner_idx)  # shape = (# of relevant, dx_dim)

        # Next, we need to "scatter" these rows back into a full (dx_dim, dx_dim) matrix,
        # where only the rows in inner_idx get data, the other rows are zero.
        # This is easily done by a simple scatter operation:
        def scatter_rows(
                subset_rows: jnp.ndarray, subset_indices: jnp.ndarray, full_shape: tuple
        ):
            """Places subset_rows (len(subset_indices), N) back into a (N,N) array,
               where each row in subset_indices goes to its place, and others are zero."""
            # full_shape is (dx_dim, dx_dim)
            # subset_rows is shape (K, dx_dim)
            # subset_indices is shape (K,)
            nrows, ncols = full_shape
            base = jnp.zeros((nrows, ncols), dtype=subset_rows.dtype)
            # We can just do a lax.scatter
            return base.at[subset_indices].set(subset_rows)

        Jx_array = scatter_rows(Jx_rows, inner_idx, (dx_array.size, dx_array.size))
        # shape = (dx_dim, dx_dim)

        # ============== Combine with the cotangent ==================
        # dL/du = g_array^T @ Ju_array, shape => (num_u_dims,)
        d_u_flat = Ju_array @ g_array  # shape = (num_u_dims,)
        # dL/dx = g_array^T @ Jx_array, shape => (dx_dim,)
        d_x_flat = Jx_array @ g_array

        # Unravel back to the shape of dx_in
        d_x = unravel_dx(d_x_flat)
        # Reshape d_u to the shape of u_in
        d_u = d_u_flat.reshape(u_in.shape)

        return (d_x, d_u)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


@equinox.filter_jit
def simulate_trajectory(
        mx,
        qpos_init: jnp.ndarray,
        step_fn: Callable[[jnp.ndarray], mjx.Data],
        running_cost_fn: Callable[[mjx.Data], float],
        terminal_cost_fn: Callable[[mjx.Data], float],
        U: jnp.ndarray
):
    """
    Rolls out a trajectory under a sequence of controls U using step_fn(dx, u).
    """

    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    # An initial step might be needed to sync internal data structures
    dx0 = mjx.step(mx, dx0)

    def scan_body(dx, u):
        dx_next = step_fn(dx, u)
        cost_t = running_cost_fn(dx_next)
        state_t = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return dx_next, (state_t, cost_t)

    dx_final, (states, costs) = jax.lax.scan(scan_body, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost


def make_loss_fn(
        mx,
        qpos_init: jnp.ndarray,
        set_ctrl_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
        running_cost_fn: Callable[[mjx.Data], float],
        terminal_cost_fn: Callable[[mjx.Data], float],
        eps: float = 1e-6
):
    """
    Builds a function loss(U) that:
      1) Creates the FD-based step function with custom_vjp
      2) Simulates a trajectory
      3) Returns the total cost
    """

    # Build the single-argument step function
    single_arg_step_fn = make_step_fn(
        mx=mx,
        set_control_fn=set_ctrl_fn,
        eps=eps
    )

    def loss(U: jnp.ndarray):
        _, total_cost = simulate_trajectory(
            mx,
            qpos_init,
            single_arg_step_fn,
            running_cost_fn,
            terminal_cost_fn,
            U
        )
        return total_cost

    return loss


@dataclass
class PMP:
    """
    A gradient-based optimizer for the FD-based MuJoCo trajectory problem.
    """
    loss: Callable[[jnp.ndarray], float]

    def grad_loss(self, U: jnp.ndarray) -> jnp.ndarray:
        """
        JAX will see the custom_vjp inside make_step_fn and
        perform FD-based partial derivatives with respect to U.
        """
        return jax.grad(self.loss)(U)

    def solve(
            self,
            U0: jnp.ndarray,
            learning_rate: float = 1e-2,
            tol: float = 1e-6,
            max_iter: int = 100
    ):
        """
        Performs a simple gradient descent on the trajectory controls.
        """
        U = U0
        for i in range(max_iter):
            g = self.grad_loss(U)
            U_new = U - learning_rate * g
            cost_val = self.loss(U_new)
            print(f"\n--- Iteration {i} ---")
            print(f"Cost={cost_val}")
            print(f"||grad||={jnp.linalg.norm(g)}")
            # Check for convergence or NaNs
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                print(f"Converged at iteration {i}.")
                return U_new
            U = U_new

        return U