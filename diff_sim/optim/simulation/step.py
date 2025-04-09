from mujoco.mjx._src.math import quat_integrate, quat_sub
import jax
import jax.numpy as jnp
from jax._src.ad_util import Zero
from jax.flatten_util import ravel_pytree
from diff_sim.optim.simulation.fd_cache import build_fd_cache
from diff_sim.context.meta_context import Context
from mujoco import mjx


def convert_one_leaf(p_leaf, t_leaf):
    # If the tangent leaf is a SymbolicZero, build a zeros array
    # that matches the primal leaf's shape and dtype.
    if isinstance(t_leaf, Zero):
        return jnp.zeros_like(p_leaf)
    else:
        # If it's already a normal array, leave it as-is.
        return t_leaf

def make_zero(p_leaf):
        return jnp.zeros_like(p_leaf)

def float0_to_zeros(p_leaf):
    if jax.dtypes.result_type(p_leaf) == jax.dtypes.float0:
        jax.debug.print("float0_to_zeros: {p_leaf}", p_leaf=p_leaf)
        return jnp.zeros_like(p_leaf)
    else:
        return p_leaf

def _upscale(x):
    if "dtype" in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

# -------------------------------------------------------------
# Step function with automatic derivative
# -------------------------------------------------------------
def make_step_fn(ctx: Context):
    mx = ctx.mx
    set_control_fn = ctx.set_control
    def step_fn(dx: mjx.Data, u: jnp.ndarray):
        """
        Forward pass:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
        """
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
        return dx_next

    return step_fn


# -------------------------------------------------------------
# Step function with custom FD-based derivative
# -------------------------------------------------------------
def make_step_fn_fd(ctx: Context):
    """
    Create a custom_vjp step function that takes (dx, u) and returns dx_next.
    We do finite differences (FD) in the backward pass using the info in fd_cache.
    """
    mx = ctx.mx
    set_control_fn = ctx.set_control
    dx_tmp = mjx.make_data(ctx.mx) # Temporary pytree data
    dx_tmp = jax.tree.map(_upscale, dx_tmp)
    fd_cache = build_fd_cache(ctx.mx, dx_tmp, ctx.target_fields, ctx.ctrl_dim, ctx.eps)
    del dx_tmp

    @jax.custom_jvp
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

    def step_fn_bwd_vjp(res, g):
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

        def assign_quat(array_in, q_idx):
            quat_ = jnp.zeros(4)
            quat_ = quat_.at[0].set(array_in[q_idx])
            quat_ = quat_.at[1].set(array_in[q_idx + 1])
            quat_ = quat_.at[2].set(array_in[q_idx + 2])
            quat_ = quat_.at[3].set(array_in[q_idx + 3])
            return quat_

        def assign_inplace_quat_array(array_in, args):
            quat, q_idx = args
            array_in = array_in.at[q_idx].set(quat[0])
            array_in = array_in.at[q_idx + 1].set(quat[1])
            array_in = array_in.at[q_idx + 2].set(quat[2])
            array_in = array_in.at[q_idx + 3].set(quat[3])
            return array_in, None

        def assign_inplace_quat_pytree(pytree_in, quat, q_idx):
            pytree_in = pytree_in.replace(qpos=pytree_in.qpos.at[q_idx].set(quat[0]))
            pytree_in = pytree_in.replace(
                qpos=pytree_in.qpos.at[q_idx + 1].set(quat[1])
            )
            pytree_in = pytree_in.replace(
                qpos=pytree_in.qpos.at[q_idx + 2].set(quat[2])
            )
            pytree_in = pytree_in.replace(
                qpos=pytree_in.qpos.at[q_idx + 3].set(quat[3])
            )
            return pytree_in

        def diff_quat(array0, array1, q_idx):
            quat0 = assign_quat(array0, q_idx)
            quat1 = assign_quat(array1, q_idx)
            # Return [0.,vel_x, vel_y, vel_z] to fit the dimension of dx_in
            return jnp.insert(quat_sub(quat0, quat1), 0, 0.0)

        diff_quat_vmap = jax.vmap(diff_quat, in_axes=(None, None, 0))

        def state_diff_quat(array0, array1):
            vels = diff_quat_vmap(array0, array1, dx_flat_init_quat_idx)
            diff_array = array0 - array1
            diff_array, _ = jax.lax.scan(
                assign_inplace_quat_array, diff_array, (vels, dx_flat_init_quat_idx)
            )
            diff_array = diff_array / eps
            return diff_array

        def state_diff_no_quat(array0, array1):
            diff_array = array0 - array1
            return diff_array / eps

        state_diff = (
            state_diff_no_quat if dx_flat_quat_idx.size == 0 else state_diff_quat
        )

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
            quat_ = assign_quat(dx_in.qpos, q_idx)
            quat_perturbed = quat_integrate(quat_, axe_perturbed, jnp.array(eps))
            dx_in_perturbed = assign_inplace_quat_pytree(
                dx_in_perturbed, quat_perturbed, q_idx
            )
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * state_diff(dx_perturbed_array, dx_out_array)

        def insert_zeros_every_4_rows(X):
            num_rows, num_cols = X.shape
            num_new_rows = num_rows + (num_rows // 4) + 1  # Extra rows for zeros
            X_padded = jnp.zeros(
                (num_new_rows, num_cols), dtype=X.dtype
            )  # Output array filled with zeros
            idxs = (
                jnp.arange(num_rows) + (jnp.arange(num_rows) // 3) + 1
            )  # Indices to insiert original rows
            X_padded = X_padded.at[idxs].set(X)  # update padded with original values
            return X_padded

        def scatter_rows(subset_rows, subset_indices, full_shape, base=None):
            if base is None:
                base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)

        dx_dim = dx_array.size
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))
        Jx_rows = jax.vmap(fdx_for_index)(dx_flat_no_quat_idx)
        d_x_flat_sub = Jx_rows[:, dx_flat_all_idx] @ g_array[dx_flat_all_idx]
        d_x_flat = scatter_rows(
            d_x_flat_sub, dx_flat_no_quat_idx, (dx_dim,)
        )  # inner_idx : without quaternions

        if dx_flat_quat_idx.size != 0:
            Jxq_rows = jax.vmap(fdx_for_quat)(qpos_init_quat_idx_rep, quat_ijk_idx_rep)
            Jxq_rows = insert_zeros_every_4_rows(Jxq_rows)
            d_x_flat_q_sub = Jxq_rows[:, dx_flat_all_idx] @ g_array[dx_flat_all_idx]
            d_x_flat = scatter_rows(
                d_x_flat_q_sub, dx_flat_quat_idx, (dx_dim,), d_x_flat
            )

        d_u = Ju_array[:, dx_flat_all_idx] @ g_array[dx_flat_all_idx]
        d_x = unravel_dx(d_x_flat)

        return (d_x, d_u)

    def step_fn_bwd_jvp(primal_args, tangent_args):
        """
        JVP-based backward pass. We approximate d(dx_next)/d(dx,u) and chain-rule with g.
        Uses the cached flatten/unflatten info in fd_cache.
        """
        dx_in, u_in = primal_args
        d_dx_in, d_u_in = tangent_args
        d_dx_in = jax.tree.map(float0_to_zeros, d_dx_in)
        dx_out = step_fn(dx_in, u_in)

        # Flatten dx_in, dx_out, and controls
        d_dx_in_array, _ = ravel_pytree(d_dx_in)
        d_u_in_array, _ = ravel_pytree(d_u_in)
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

        def assign_quat(array_in, q_idx):
            quat_ = jnp.zeros(4)
            quat_ = quat_.at[0].set(array_in[q_idx])
            quat_ = quat_.at[1].set(array_in[q_idx + 1])
            quat_ = quat_.at[2].set(array_in[q_idx + 2])
            quat_ = quat_.at[3].set(array_in[q_idx + 3])
            return quat_

        def assign_inplace_quat_array(array_in, args):
            quat, q_idx = args
            array_in = array_in.at[q_idx].set(quat[0])
            array_in = array_in.at[q_idx + 1].set(quat[1])
            array_in = array_in.at[q_idx + 2].set(quat[2])
            array_in = array_in.at[q_idx + 3].set(quat[3])
            return array_in

        def assign_inplace_quat_pytree(pytree_in, quat, q_idx):
            pytree_in = pytree_in.replace(qpos=pytree_in.qpos.at[q_idx].set(quat[0]))
            pytree_in = pytree_in.replace(
                qpos=pytree_in.qpos.at[q_idx + 1].set(quat[1])
            )
            pytree_in = pytree_in.replace(
                qpos=pytree_in.qpos.at[q_idx + 2].set(quat[2])
            )
            pytree_in = pytree_in.replace(
                qpos=pytree_in.qpos.at[q_idx + 3].set(quat[3])
            )
            return pytree_in

        def diff_quat(array0, array1, q_idx):
            quat0 = assign_quat(array0, q_idx)
            quat1 = assign_quat(array1, q_idx)
            # Return [0.,vel_x, vel_y, vel_z] to fit the dimension of dx_in
            return jnp.insert(quat_sub(quat0, quat1), 0, 0.0)

        diff_quat_vmap = jax.vmap(diff_quat, in_axes=(None, None, 0))

        def state_diff_quat(array0, array1):
            vels = diff_quat_vmap(array0, array1, dx_flat_init_quat_idx)
            diff_array = array0 - array1
            diff_array = jax.lax.scan(
                assign_inplace_quat_array, diff_array, (vels, dx_flat_init_quat_idx)
            )
            diff_array = diff_array / eps
            return diff_array

        def state_diff_no_quat(array0, array1):
            diff_array = array0 - array1
            return diff_array / eps

        state_diff = (
            state_diff_no_quat if dx_flat_quat_idx.size == 0 else state_diff_quat
        )

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
            quat_ = assign_quat(dx_in.qpos, q_idx)
            quat_perturbed = quat_integrate(quat_, axe_perturbed, jnp.array(eps))
            dx_in_perturbed = assign_inplace_quat_pytree(
                dx_in_perturbed, quat_perturbed, q_idx
            )
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * state_diff(dx_perturbed_array, dx_out_array)

        def insert_zeros_every_4_rows(X):
            num_rows, num_cols = X.shape
            num_new_rows = num_rows + (num_rows // 4) + 1  # Extra rows for zeros
            X_padded = jnp.zeros(
                (num_new_rows, num_cols), dtype=X.dtype
            )  # Output array filled with zeros
            idxs = (
                    jnp.arange(num_rows) + (jnp.arange(num_rows) // 3) + 1
            )  # Indices to insiert original rows
            X_padded = X_padded.at[idxs].set(X)  # update padded with original values
            return X_padded

        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))
        Jx_rows = jax.vmap(fdx_for_index)(dx_flat_no_quat_idx)
        x_tangent = Jx_rows.T @ d_dx_in_array[dx_flat_all_idx]

        if dx_flat_quat_idx.size != 0:
            Jxq_rows = jax.vmap(fdx_for_quat)(qpos_init_quat_idx_rep, quat_ijk_idx_rep)
            Jxq_rows = insert_zeros_every_4_rows(Jxq_rows)
            x_tangent = x_tangent + Jxq_rows.T @ d_dx_in_array[dx_flat_all_idx]


        u_tangent = Ju_array.T @ d_u_in_array # [dx by 2] @ 2
        dx_out_array_tangent = x_tangent + u_tangent # dx
        dx_out_tangent = unravel_dx(dx_out_array_tangent)
        # jax.debug.print("dx_out_tangent shape 0: {dx_out_tangent}", dx_out_tangent=dx_out_array_tangent.shape)
        # dx_out_tangent = jax.tree_map(lambda l1: jnp.zeros_like(l1), dx_out)
        # jax.debug.print("dx_out_tangent shape 1: {dx_out_tangent}", dx_out_tangent=dx_out_array_tangent.shape)
        return dx_out, dx_out_tangent # dx

    step_fn.defjvp(step_fn_bwd_jvp)
    step_fn.defvjp(step_fn_fwd, step_fn_bwd_vjp)
    return step_fn
