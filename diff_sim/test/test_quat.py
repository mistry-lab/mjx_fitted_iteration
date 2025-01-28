import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import os
from jax import config
from diff_sim.utils.math_helper import sub_quat

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
        target_fields = {"qpos", "qvel", "ctrl", "qfrc_applied"}

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


############################################################################

def running_cost(dx):
    # return sub_quat()
    quat_ref = jnp.array([1., 0., 0., 0.])
    quat_diff = sub_quat(quat_ref, dx.qpos[3:7])

    # return 0.01*jnp.array([jnp.sum(quat_diff**2)]) + 0.00001*dx.qfrc_applied[5]**2
    return 0.00001 * jnp.array([dx.qfrc_applied[5] ** 2])


def step_scan_mjx(carry, u):
    dx = carry
    # Dynamics function
    dx = step_fn(dx, u)
    t = jnp.expand_dims(dx.time, axis=0)
    cost = running_cost(dx)
    # dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:].set(jnp.zeros_like(dx.qfrc_applied)))
    return (dx), jnp.concatenate([dx.qpos, dx.qvel, cost, t])


# from functools import partial
# @partial(jax.vmap, in_axes=(0, 0))  # Batch over both `qpos_init` and `U`
def simulate_trajectory_mjx(qpos_init, U):
    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    # dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:].set(u))
    (dx), res = jax.lax.scan(step_scan_mjx, (dx), U, length=Nlength)
    res, cost, t = res[..., :-2], res[..., -2], res[..., -1]
    return res, cost, t


vmap_simulate_trajectory_mjx = jax.vmap(lambda qpos_init, U: simulate_trajectory_mjx(qpos_init, U), in_axes=(0, 0))


# def compute_trajectory_costs(qpos_init, U):
#     res, cost, t = vmap_simulate_trajectory_mjx(qpos_init, U)
#     return cost, res, t

def simulate_trajectory_mj(qpos_init, u):
    d = mujoco.MjData(model)
    d.qpos = qpos_init
    d.qfrc_applied[0] = u
    qq = []
    qv = []
    t = []
    for k in range(Nlength):
        mujoco.mj_step(model, d)
        qq.append(d.qpos[:].copy().tolist())
        qv.append(d.qvel[:].copy().tolist())
        t.append(d.time)
        d.qfrc_applied[0] = 0.
    return np.array(qq), np.array(qv), np.array(t)


def visualise(qpos, qvel):
    import time
    from mujoco import viewer
    data = mujoco.MjData(model)
    data.qpos = idata.qpos

    with viewer.launch_passive(model, data) as viewer:
        for i in range(qpos.shape[0]):
            step_start = time.time()
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                # time.sleep(time_until_next_step)
                time.sleep(0.075)


def visu_u(U):
    qpos = jnp.expand_dims(jnp.array(idata.qpos), axis=0)
    qpos = jnp.repeat(qpos, 1, axis=0)

    # u = set_u(jnp.array([u0]))
    res, cost, t = vmap_simulate_trajectory_mjx(qpos, U)
    qpos_mjx, qvel_mjx = res[0, :, :model.nq], res[0, :, model.nq:]
    visualise(qpos_mjx, qvel_mjx)


@jax.jit
def compute_loss_grad(qpos_init, u):
    jac_fun = jax.jacrev(lambda x: loss_funct(qpos_init, x))
    ad_grad = jac_fun(u)
    return ad_grad


def loss_funct(qpos_init, U):
    # u = set_u(u)
    costs = simulate_trajectory_mjx(qpos_init, U)[0]
    costs = jnp.sum(costs, axis=1)
    costs = jnp.mean(costs)
    return jnp.array([costs])


# def set_u(u0):
#     u = jnp.zeros_like(idata.qfrc_applied)
#     u = jnp.expand_dims(u, axis=0)
#     u = jnp.repeat(u, u0.shape[0], axis=0)
#     u = u.at[:,5].set(u0)
#     return u

# @jax.jit
# def get_traj_grad(qpos_init,u):
#     jac_fun = jax.jacrev(lambda x: compute_trajectory_costs(qpos_init,set_u(x))[0])
#     ad_grad = jac_fun(jnp.array(u))
#     return ad_grad

# Gradient descent
def gradient_descent(qpos, x0, learning_rate=0.1, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        # x =jnp.round(x,5)
        grad = compute_loss_grad(qpos, x)[0]
        grad = compute_loss_grad(qpos, x)[0]
        x_new = x - learning_rate * grad  # Gradient descent update

        print(f"Iteration {i}: x = {x}, f(x) = {loss_funct(qpos, x)}")

        # Check for convergence
        if abs(x_new - x) < tol:
            break
        x = x_new

    return x


def set_control(dx, u):
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[5].set(u[0]))
    return dx


if __name__ == "__main__":
    # Load mj and mjx model
    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), '../xmls/two_body.xml'))
    mx = mjx.put_model(model)
    dx_ref = mjx.make_data(mx)

    # Build an FD cache once, as usual
    fd_cache = build_fd_cache(
        dx_ref,
        target_fields={"qpos", "qvel", "ctrl", "sensordata"},
        eps=1e-6
    )

    # FD-based custom VJP
    step_fn = make_step_fn(mx, set_control, fd_cache)

    idata = mujoco.MjData(model)

    qx0, qz0, qx1 = -0., 0.25, -0.2  # Inititial positions
    idata.qpos[0], idata.qpos[2], idata.qpos[7] = qx0, qz0, qx1
    Nlength = 40  # horizon lenght

    u0 = jnp.array([0.002])
    batch = 1
    # U0 = jnp.repeat(jnp.array([u0]), Nlength)
    U0 = jnp.repeat(jnp.array([[u0]]), Nlength, axis=1)  # Commands
    U0 = U0.repeat(batch, axis=0)  # Batch size = 1

    qpos = jnp.expand_dims(jnp.array(idata.qpos), axis=0)
    qpos = jnp.repeat(qpos, batch, axis=0)
    # u = set_u(jnp.array([u0]))

    # Simulate trajectory
    # res, cost, t = simulate_trajectory_mjx(qpos, u)
    # qpos_mjx, qvel_mjx = res[0,:,:model.nq], res[0,:,model.nq:]

    # Visualise a trajectory
    # visu_u(U0)

    # Initial guess
    x0 = U0[0]
    # Initial position
    qpos0 = qpos[0]
    # loss_funct(qpos0,x0)
    compute_loss_grad(qpos0, x0)

    # optimal_x = gradient_descent(qpos, x0, learning_rate=0.01)
    # grad = compute_loss_grad(qpos, x0)[0]