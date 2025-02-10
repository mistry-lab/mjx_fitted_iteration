import jax
from jaxtyping import PyTree
import jax.numpy as jnp
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable, Optional, Set
import equinox
from jax.flatten_util import ravel_pytree
import numpy as np
from jax._src.util import unzip2
import time

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

def upscale(x):
    if 'dtype' in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

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


def build_fd_cache(
        dx_ref: mjx.Data,
        target_fields: Optional[Set[str]] = None,
        eps: float = 1e-6
) -> FDCache:
    """
    Build a cache containing:
      - Flatten/unflatten for dx_ref
      - The mask for relevant FD indices (e.g. qpos, qvel, ctrl)
      - The shape info for control
    """
    if target_fields is None:
        target_fields = {"qpos", "qvel", "ctrl"}

    # Flatten dx
    dx_array, unravel_dx = ravel_pytree(dx_ref)
    dx_size = dx_array.shape[0]
    num_u_dims = dx_ref.ctrl.shape[0]

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

    return FDCache(
        unravel_dx=unravel_dx,
        sensitivity_mask=sensitivity_mask,
        inner_idx=inner_idx,
        dx_size=dx_size,
        num_u_dims=num_u_dims,
        eps=eps
    )


# -------------------------------------------------------------
# Step function with custom FD-based derivative
# -------------------------------------------------------------
def make_step_fn(
        mx,
        set_control_fn: Callable,
        fd_cache: FDCache
):
    """
    Create a custom_vjp step function that takes (dx, u) and returns dx_next.
    We do finite differences (FD) in the backward pass using the info in fd_cache.
    """

    @jax.custom_vjp
    def step_fn(dx: mjx.Data, u: jnp.ndarray):
        """
        Forward pass:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
        """
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
        return dx_next

    def step_fn_fwd(dx, u):
        dx_next = step_fn(dx, u)
        return dx_next, (dx, u, dx_next)

    def step_fn_bwd(res, g):
        """
        FD-based backward pass. We approximate d(dx_next)/d(dx,u) and chain-rule with g.
        Uses the cached flatten/unflatten info in fd_cache.
        """
        dx_in, u_in, dx_out = res

        # Convert float0 leaves in 'g' to zeros
        def map_g_to_dinput(diff_tree, grad_tree):
            def fix_leaf(d_leaf, g_leaf):
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    return jnp.zeros_like(d_leaf)
                else:
                    return g_leaf

            return jax.tree_map(fix_leaf, diff_tree, grad_tree)

        mapped_g = map_g_to_dinput(dx_in, g)
        # jax.debug.print(f"mapped_g: {mapped_g}")
        g_array, _ = ravel_pytree(mapped_g)

        # Flatten dx_in, dx_out, and controls
        dx_array, _ = ravel_pytree(dx_in)
        dx_out_array, _ = ravel_pytree(dx_out)
        u_in_flat = u_in.ravel()

        # Grab cached info
        unravel_dx = fd_cache.unravel_dx
        sensitivity_mask = fd_cache.sensitivity_mask
        inner_idx = fd_cache.inner_idx
        num_u_dims = fd_cache.num_u_dims
        eps = fd_cache.eps

        # =====================================================
        # =============== FD wrt control (u) ==================
       # =====================================================
        def fdu_plus(i):
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn(dx_in, u_in_eps)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            # Only keep relevant dims
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # shape = (num_u_dims, dx_dim)
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))

        # =====================================================
        # ================ FD wrt state (dx) ==================
        # =====================================================
        # We only FD over "inner_idx" (subset of the state: qpos, qvel, ctrl, etc.)
        def fdx_for_index(idx):
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            # Only keep relevant dims
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # shape = (len(inner_idx), dx_dim)
        Jx_rows = jax.vmap(fdx_for_index)(inner_idx)

        # -----------------------------------------------------
        # Instead of scattering rows into a (dx_dim, dx_dim) matrix,
        # multiply Jx_rows directly with g_array[inner_idx].
        # This avoids building a large dense Jacobian in memory.
        # -----------------------------------------------------
        # Jx_rows[i, :] is derivative w.r.t. dx_array[inner_idx[i]].
        # We want sum_i [ Jx_rows[i] * g_array[inner_idx[i]] ].
        # => shape (dx_dim,)
        # Scatter those rows back to a full (dx_dim, dx_dim) matrix
        def scatter_rows(subset_rows, subset_indices, full_shape):
            base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)

        dx_dim = dx_array.size

        # Solution 2 : Reduced size multiplication (inner_idx, inner_idx) @ (inner_idx,)
        d_x_flat_sub = Jx_rows[:, inner_idx] @ g_array[inner_idx]
        d_x_flat = scatter_rows(d_x_flat_sub, inner_idx, (dx_dim,))

        d_u = Ju_array[:, inner_idx] @ g_array[inner_idx]
        d_x = unravel_dx(d_x_flat)
        return (d_x, d_u)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


# -------------------------------------------------------------
# Simulate a trajectory
# -------------------------------------------------------------
@equinox.filter_jit
def simulate_trajectories(
        mx,
        qpos_inits,  # shape (B, nq) for example
        qvel_inits,  # shape (B, nqvel)
        running_cost_fn,
        terminal_cost_fn,
        step_fn,
        params,
        static,
        length,
        keys
):
    """
    Simulate a *batch* of trajectories (B of them) with a single policy.

    Args:
      mx: MuJoCo model container.
      qpos_inits: shape (B, n_qpos)
      qvel_inits: shape (B, n_qvel)
      running_cost_fn, terminal_cost_fn: same cost structure as before.
      step_fn: the custom FD-based step function returned by make_step_fn.
      params, static: your policy parameters & static parts (from equinox.partition).
      length: number of steps per trajectory.
    Returns:
      - states_batched: shape (B, length, 2 * n_qpos) (if that’s your total state dimension)
      - total_cost: a scalar cost (mean or sum across the batch).
    """
    # Combine the param & static into the actual model (same as simulate_trajectory).
    model = equinox.combine(params, static)

    def single_trajectory(qpos_init, qvel_init, key):
        """Simulate one trajectory given a single (qpos_init, qvel_init)."""
        # Build the initial MuJoCo data
        dx0 = mjx.make_data(mx)
        dx0 = jax.tree_map(upscale, dx0)
        dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
        dx0 = dx0.replace(qvel=dx0.qvel.at[:].set(qvel_init))

        # Define the scanning function for a single rollout
        def scan_step_fn(carry, _):
            dx, key = carry
            key, subkey = jax.random.split(key)

            x = jnp.concatenate([dx.qpos, dx.qvel, dx.sensordata ])
            
            # Add noise to the control  
            noise = 0. * jax.random.normal(subkey, mx.nu)
            # jax.debug.print("noise : {}", noise)
            u = model(x, dx.time) + noise # policy output

            dx = step_fn(dx, u)  # FD-based MuJoCo step
            c = running_cost_fn(dx)
            state = jnp.concatenate([dx.qpos, dx.qvel, dx.sensordata])
            return (dx,key), (state, c)

        key, subkey = jax.random.split(key)
        (dx_final, _), (states, costs) = jax.lax.scan(scan_step_fn, (dx0,subkey), length=length)
        total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
        return states, total_cost

    # vmap across the batch dimension
    states_batched, costs_batched = jax.vmap(single_trajectory)(qpos_inits, qvel_inits, keys)

    return states_batched, costs_batched


# -------------------------------------------------------------
# Build the loss
# -------------------------------------------------------------
def make_loss_multi_init(
    mx,
    qpos_inits,         # shape (B, n_qpos)
    qvel_inits,         # shape (B, n_qvel)
    set_control_fn,
    running_cost_fn,
    terminal_cost_fn,
    length,
    batch_size,
    sample_size
):
    """
    Create a loss function for *multiple initial conditions*.
    The returned function takes 'params' and 'static',
    and returns cost aggregated across all initial conditions.
    """

    dx_ref = mjx.make_data(mx)

    # Build an FD cache once, as usual
    fd_cache = build_fd_cache(
        dx_ref,
        target_fields={"qpos", "qvel", "ctrl", "sensordata"},
        eps=1e-6
    )

    # FD-based custom VJP
    step_fn = make_step_fn(mx, set_control_fn, fd_cache)

    def multi_init_loss(params, static, keys):
        _, costs_batched = simulate_trajectories(
            mx, qpos_inits, qvel_inits,
            running_cost_fn, terminal_cost_fn, step_fn,
            params, static, length,keys
        )
        total_cost = jnp.mean(costs_batched) # no samples
        return total_cost
    
    def multi_init_loss_stoch(params, static, keys):
        _, costs_batched = simulate_trajectories(
            mx, qpos_inits, qvel_inits,
            running_cost_fn, terminal_cost_fn, step_fn,
            params, static, length,keys
        )
        costs = costs_batched.reshape(batch_size, sample_size)
        exp_sum_costs = jnp.mean(costs, axis=-1)
        total_cost = jnp.mean(exp_sum_costs)
        # jax.debug.print()
        return total_cost

    return multi_init_loss_stoch


# -------------------------------------------------------------
# The Policy class for gradient-based optimization
# -------------------------------------------------------------
@dataclass
class Policy:
    loss: Callable[[PyTree, PyTree], float]

    def solve(self, model: equinox.Module, optim, state, batch_size, max_iter=100):
        """
        Generic gradient descent loop on your policy parameters.
        """

        grad_loss = equinox.filter_jit(jax.jacrev(self.loss))
        opt_model = None
        key = jax.random.PRNGKey(10) 
        for i in range(max_iter):
            now = time.time()
            params, static = equinox.partition(model, equinox.is_array)
            # Keys for random noise
            key, subkey = jax.random.split(key, num=(2,)) 
            key_batch = jax.random.split(subkey, num=(batch_size,))
            g = grad_loss(params, static,key_batch)
            f_val = self.loss(params, static,key_batch)
            updates, state = optim.update(g, state, model)
            model = equinox.apply_updates(model, updates)
            opt_model = model
            print(f"Iteration {i}: cost={f_val}, time={time.time() - now}")
            # print(f"Iteration {i}: cost={f_val}")
        return model

