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


def upscale(x):
    if 'dtype' in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


@dataclass(frozen=True)
class FDCache:
    """Holds all the precomputed info needed by the custom FD-based backward pass."""
    unravel_dx: Callable[[jnp.ndarray], mjx.Data]
    num_u_dims: int
    eps: float
    sensitivity_mask: jnp.ndarray
    dx_flat_all_idx: Optional[jnp.ndarray]  # All indices (including free/ball quaternion)
    dx_flat_no_quat_idx: jnp.ndarray  # "normal" FD indices (excludes free/ball quaternion)
    dx_flat_quat_idx: Optional[jnp.ndarray] = None # subset that also lies in target_fields
    dx_flat_init_quat_idx: Optional[jnp.ndarray] = None # first index of each quaternion
    qpos_init_quat_idx_rep: Optional[jnp.ndarray] = None # repeated by ijk axes * quat joints
    quat_ijk_idx_rep: Optional[jnp.ndarray] = None # repeated by ijk axes * quat joints

def build_fd_cache(
    mx,                   # MuJoCo model wrapper, for jnt_type, jnt_qposadr
    dx_ref: mjx.Data,     # reference data object
    target_fields: Optional[Set[str]] = None,
    ctrl_dim: Optional[int] = None,
    eps: float = 1e-6
) -> FDCache:
    """
    Build an FDCache that:
      1) Finds the global flatten range of qpos within dx_ref.
      2) Converts MuJoCo's local qpos indices for free/ball joints
         into *global* flatten indices (quat_idx).
      3) Gathers all target_fields into 'candidate_idx_full'.
      4) Splits out 'quat_inner_idx' from 'candidate_idx_full', removing them
         so 'inner_idx' contains only non-quaternion states.
    """
    if target_fields is None:
        target_fields = {"qpos", "qvel"}

    # Flatten dx_ref
    dx_array, unravel_dx = ravel_pytree(dx_ref)
    dx_size = dx_array.shape[0]

    # If ctrl_dim not specified, get from dx_ref.ctrl
    if ctrl_dim is None:
        ctrl_dim = dx_ref.ctrl.shape[0]
    num_u_dims = ctrl_dim

    # ----------------------------------------------------------------
    # A) Identify the global flatten range for the "qpos" leaf
    # ----------------------------------------------------------------
    # We assume there's exactly one leaf whose path has name=='qpos'
    leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_ref))
    sizes, _ = unzip2((jnp.size(leaf), jnp.shape(leaf)) for (_, leaf) in leaves_with_path)
    offsets = np.cumsum(sizes)  # The end offsets for each leaf in the flatten array

    # We'll find which leaf index i is 'qpos'
    qpos_leaf_idx = None
    running_start = 0
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        leaf_end = offsets[i]
        # Check if this leaf is "qpos"
        name_matches = any(getattr(p, 'name', None) == 'qpos' for p in path)
        if name_matches:
            qpos_leaf_idx = i
            qpos_leaf_start = running_start
            break
        running_start = leaf_end

    # If we can't find qpos leaf, raise an error or handle differently
    if qpos_leaf_idx is None:
        raise RuntimeError("Could not find a 'qpos' leaf in dx_ref to map quaternion indices.")

    # So the global flatten range for qpos is [qpos_leaf_start : qpos_leaf_start + qpos_leaf_size]
    # local qpos index i in [0..(nq-1)] maps to global index (qpos_leaf_start + i)

    # ----------------------------------------------------------------
    # B) Build "quat_idx_full" in global flatten space
    # ----------------------------------------------------------------
    # For each free or ball joint, we add the last 4 or all 4 local qpos indices,
    # then shift them by qpos_leaf_start to get global indices.
    # e.g. for FREE => local [3..6] => global [qpos_leaf_start+3..+6]
    # for BALL => local [0..3], etc.
    local_quat_indices = []
    for j, jtype in enumerate(mx.jnt_type):
        if jtype == JointType.FREE:
            start = mx.jnt_qposadr[j]     # local qpos index for that joint
            local_quat_indices.append(np.arange(start+3, start+7))
        elif jtype == JointType.BALL:
            start = mx.jnt_qposadr[j]
            local_quat_indices.append(np.arange(start, start+4))
        # else HINGE/SLIDE => skip

    if len(local_quat_indices) == 0:
        quat_idx_full = np.array([], dtype=int)
        quat_idx = None
    else:
        # flatten them
        loc_concat = np.concatenate(local_quat_indices, axis=0)
        quat_idx = jnp.array(loc_concat, dtype=jnp.int32)
        # shift by qpos_leaf_start
        quat_idx_full = loc_concat + qpos_leaf_start

    # ----------------------------------------------------------------
    # C) Gather "candidate_idx_full" for target_fields
    #     i.e. flatten indices for qpos, qvel, ctrl, etc.
    # ----------------------------------------------------------------
    def leaf_index_range(leaf_idx):
        # [start, end)
        start_ = 0 if leaf_idx == 0 else offsets[leaf_idx-1]
        end_ = offsets[leaf_idx]
        return np.arange(start_, end_)

    candidate_subsets = []
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        name_matches = any(getattr(p, 'name', None) in target_fields for p in path)
        if name_matches:
            candidate_subsets.append(leaf_index_range(i))
    if len(candidate_subsets) > 0:
        dx_flat_all_idx = np.concatenate(candidate_subsets, axis=0)
    else:
        dx_flat_all_idx = np.array([], dtype=int)

    # ----------------------------------------------------------------
    # D) Build "dx_flat_quat_idx" = intersection of dx_flat_all_idx & quat_idx_full
    #    Then remove them from dx_flat_all_idx => "non_quat_inner_idx"
    # ----------------------------------------------------------------
    if quat_idx_full.size == 0 or dx_flat_all_idx.size == 0:
        quat_inner = np.array([], dtype=int)
    else:
        # intersect
        quat_inner = np.intersect1d(quat_idx_full, dx_flat_all_idx)

    # Remove them from candidate
    non_quat_inner = np.setdiff1d(dx_flat_all_idx, quat_inner)

    # Convert to jnp
    if quat_inner.size == 0:
        dx_flat_quat_idx = np.array([], dtype=int)
    else:
        dx_flat_quat_idx = jnp.array(quat_inner, dtype=jnp.int32)

    dx_flat_no_quat_idx = jnp.array(non_quat_inner, dtype=jnp.int32)

    # ----------------------------------------------------------------
    # E) Build sensitivity_mask for the final "dx_flat_no_quat_idx" only
    # ----------------------------------------------------------------
    sensitivity_mask = jnp.zeros_like(dx_array)
    sensitivity_mask = sensitivity_mask.at[dx_flat_no_quat_idx].set(1.0)
    sensitivity_mask = sensitivity_mask.at[dx_flat_quat_idx].set(1.0)

    # quat by inner size by 2
    # get befginning indexes of the quats
    if dx_flat_quat_idx.size != 0 and quat_idx.size != 0:
        dx_flat_init_quat_idx = jnp.repeat(dx_flat_quat_idx[::4],3)
        qpos_init_quat_idx_rep = jnp.repeat(quat_idx[::4],3)
    else:
        dx_flat_init_quat_idx = jnp.array([], dtype=jnp.int32)
        qpos_init_quat_idx_rep = jnp.array([], dtype=jnp.int32)

    # FOr each index above,
    quat_ijk_idx_rep = jnp.tile(jnp.array([0, 1, 2], dtype=jnp.int32), len(dx_flat_quat_idx) // 4)

    # ----------------------------------------------------------------
    # F) Return FDCache
    # ----------------------------------------------------------------
    return FDCache(
        unravel_dx = unravel_dx,
        sensitivity_mask = sensitivity_mask,
        dx_flat_no_quat_idx= dx_flat_no_quat_idx,
        dx_flat_all_idx= dx_flat_all_idx,
        num_u_dims = num_u_dims,
        eps = eps,
        qpos_init_quat_idx_rep= qpos_init_quat_idx_rep,
        dx_flat_quat_idx= dx_flat_quat_idx,
        dx_flat_init_quat_idx= dx_flat_init_quat_idx,
        quat_ijk_idx_rep= quat_ijk_idx_rep
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
        dx_flat_all_idx = fd_cache.dx_flat_all_idx
        # Quat indices
        dx_flat_quat_idx = fd_cache.dx_flat_quat_idx
        quat_ijk_idx_rep = fd_cache.quat_ijk_idx_rep
        dx_flat_no_quat_idx = fd_cache.dx_flat_no_quat_idx
        qpos_init_quat_idx_rep = fd_cache.qpos_init_quat_idx_rep
        dx_flat_init_quat_idx = jnp.array([], dtype=jnp.int32)
        if dx_flat_quat_idx.size != 0:
            dx_flat_init_quat_idx = dx_flat_quat_idx[::4]

        num_u_dims = fd_cache.num_u_dims
        eps = fd_cache.eps

        def assign_quat(array_in,q_idx):
            quat_ = jnp.zeros(4)
            quat_ = quat_.at[0].set(array_in[q_idx])
            quat_ = quat_.at[1].set(array_in[q_idx+1])
            quat_ = quat_.at[2].set(array_in[q_idx+2])
            quat_ = quat_.at[3].set(array_in[q_idx+3])
            return quat_

        def assign_inplace_quat_array(array_in, args):
            quat, q_idx = args
            array_in = array_in.at[q_idx].set(quat[0])
            array_in = array_in.at[q_idx+1].set(quat[1])
            array_in = array_in.at[q_idx+2].set(quat[2])
            array_in = array_in.at[q_idx+3].set(quat[3])
            return array_in, None

        def assign_inplace_quat_pytree(pytree_in, quat, q_idx):
            pytree_in = pytree_in.replace(qpos=pytree_in.qpos.at[q_idx].set(quat[0]))
            pytree_in = pytree_in.replace(qpos=pytree_in.qpos.at[q_idx+1].set(quat[1]))
            pytree_in = pytree_in.replace(qpos=pytree_in.qpos.at[q_idx+2].set(quat[2]))
            pytree_in = pytree_in.replace(qpos=pytree_in.qpos.at[q_idx+3].set(quat[3]))
            return pytree_in

        def diff_quat(array0,array1,q_idx):
            quat0 = assign_quat(array0, q_idx)
            quat1 = assign_quat(array1, q_idx)
            # Return [0.,vel_x, vel_y, vel_z] to fit the dimension of dx_in
            return jnp.insert(quat_sub(quat0, quat1), 0, 0.0)
        diff_quat_vmap = jax.vmap(diff_quat, in_axes=(None,None,0))

        def state_diff_quat(array0, array1):
            vels = diff_quat_vmap(array0, array1, dx_flat_init_quat_idx)
            diff_array = array0 - array1
            diff_array, _ = jax.lax.scan(assign_inplace_quat_array, diff_array, (vels, dx_flat_init_quat_idx))
            diff_array = diff_array / eps
            return diff_array

        def state_diff_no_quat(array0, array1):
            diff_array = array0 - array1
            return diff_array/eps

        state_diff = state_diff_no_quat if dx_flat_quat_idx.size == 0 else state_diff_quat

        # =====================================================
        # =============== FD wrt control (u) ==================
        # =====================================================
        def fdu_plus(i):
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn(dx_in, u_in_eps)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * state_diff(dx_perturbed_array, dx_out_array)

        # =====================================================
        # ================ FD wrt state (dx) ==================
        # =====================================================
        # We only FD over "inner_idx" (subset of the state: qpos, qvel, ctrl, etc.)
        def fdx_for_index(idx):
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * state_diff(dx_perturbed_array, dx_out_array)

        def fdx_for_quat(q_idx, a_idx):
            axe_perturbed = jnp.zeros(3).at[a_idx].set(1.0)
            dx_in_perturbed = dx_in
            quat_= assign_quat(dx_in.qpos, q_idx)
            quat_perturbed = quat_integrate(quat_, axe_perturbed, jnp.array(eps))
            dx_in_perturbed = assign_inplace_quat_pytree(dx_in_perturbed, quat_perturbed, q_idx)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * state_diff(dx_perturbed_array, dx_out_array)

        def insert_zeros_every_4_rows(X):
            num_rows, num_cols = X.shape
            num_new_rows = num_rows + (num_rows // 4) + 1  # Extra rows for zeros
            X_padded = jnp.zeros((num_new_rows, num_cols), dtype=X.dtype) # Output array filled with zeros
            idxs = jnp.arange(num_rows) + (jnp.arange(num_rows) // 3) + 1 # Indices to insiert original rows
            X_padded = X_padded.at[idxs].set(X) # update padded with original values
            return X_padded

        def scatter_rows(subset_rows, subset_indices, full_shape, base=None):
            if base is None:
                base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)

        dx_dim = dx_array.size
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))
        Jx_rows = jax.vmap(fdx_for_index)(dx_flat_no_quat_idx)
        d_x_flat_sub = Jx_rows[:, dx_flat_all_idx] @ g_array[dx_flat_all_idx]
        d_x_flat = scatter_rows(d_x_flat_sub, dx_flat_no_quat_idx, (dx_dim,)) # inner_idx : qithout quaternions

        if dx_flat_quat_idx.size != 0:
            Jxq_rows = jax.vmap(fdx_for_quat)(qpos_init_quat_idx_rep, quat_ijk_idx_rep)
            Jxq_rows = insert_zeros_every_4_rows(Jxq_rows)
            d_x_flat_q_sub = Jxq_rows[:, dx_flat_all_idx] @ g_array[dx_flat_all_idx]
            d_x_flat = scatter_rows(d_x_flat_q_sub, dx_flat_quat_idx, (dx_dim,), d_x_flat)

        d_u = Ju_array[:, dx_flat_all_idx] @ g_array[dx_flat_all_idx]
        d_x = unravel_dx(d_x_flat)

        # jax.debug.print("\n\ng_array_full: {val1}", val1=g_array[dx_flat_all_idx])
        # # jax.debug.print("Ju_array_quat: {val1}", val1=Ju_array[:, dx_flat_quat_idx])
        # jax.debug.print("Ju_array_quat_full: {val1}", val1=Ju_array[:, dx_flat_all_idx])

        return (d_x, d_u)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


def make_loss_fn(
        mx,
        qpos_init: jnp.ndarray,
        set_ctrl_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
        running_cost_fn: Callable[[mjx.Data], float],
        terminal_cost_fn: Callable[[mjx.Data], float],
        fd_cache: FDCache,
):
    # Build the step function with custom_vjp + FD
    single_arg_step_fn = make_step_fn(
        mx=mx,
        set_control_fn=set_ctrl_fn,
        fd_cache=fd_cache,
    )

    @jax.jit
    def simulate_trajectory(U: jnp.ndarray):
        dx0 = mjx.make_data(mx)
        dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
        dx0 = mjx.step(mx, dx0)  # initial sync

        def scan_body(dx, u):
            dx_next = single_arg_step_fn(dx, u)
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
