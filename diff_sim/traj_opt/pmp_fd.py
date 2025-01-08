import jax
import jax.numpy as jnp
from jax import config
from jax.flatten_util import ravel_pytree
from typing import Callable
from dataclasses import dataclass

# Import MuJoCo Python bindings
from mujoco import mjx

import equinox
from jax._src.flatten_util import _ravel_list
from jax._src.util import safe_zip, unzip2
import numpy as np

# Mock representation of GetAttrKey for illustration
class GetAttrKey:
    def __init__(self, name):
        self.name = name
 
    def __repr__(self):
        return f"GetAttrKey(name='{self.name}')"
 

def filter_state_data(dx: mjx.Data):
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
    eps: float = 1e-4
):
    """
    Create a custom_vjp step function that takes a single argument u.
    The function 'set_control_fn' is a user-defined way of writing u into dx.
    Finite differences (FD) are performed w.r.t. u *across all elements of dx*.

    Args:
      mx: MuJoCo model.
      dx_init: MuJoCo data you want to step from. (e.g. "state" to be updated)
      set_control_fn: user-defined function that sets dx.ctrl from u
      eps: small step size for FD.

    Returns:
      step_fn(dx, u) -> dx_next, with a custom_vjp that approximates
      ∂ dx_next / ∂ u by FD across all elements filtered in dx_next's pytree.
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

    def step_fn_fwd(dx,u):
        dx_next = step_fn(dx,u)
        # We'll save (u, dx_next) for backward pass
        return dx_next, (dx, u, dx_next)

    def step_fn_bwd(res, g):
        """
        res: (dx, u) from the forward pass
        g: the cotangent of dx_next (same pytree structure as dx)

        We want to approximate ∂dx_next/∂u by finite differences, and
        then do a dot product with ct_dx_next (also flattened).
        """
        dx_in, u_in, result = res

        def safe_ravel_pytree(py):
            """Flatten the given pytree, but first replace float0 leaves to avoid errors."""

            def fix_leaf(x):
                # If x has float0 dtype, replace with a 0.0 float64 scalar or empty array
                if jax.dtypes.result_type(x) == jax.dtypes.float0:
                    return jnp.zeros((), dtype=jnp.float64)
                else:
                    return x

            fixed_py = jax.tree_map(fix_leaf, py)
            flat, unravel_fn = ravel_pytree(fixed_py)
            return flat, unravel_fn
        
        def safe_flatten_pytree(py):
            """Flatten the given pytree, but first replace float0 leaves to avoid errors."""

            def fix_leaf(x):
                # If x has float0 dtype, replace with a 0.0 float64 scalar or empty array
                if jax.dtypes.result_type(x) == jax.dtypes.float0:
                    return jnp.zeros((), dtype=jnp.float64)
                else:
                    return x

            fixed_py = jax.tree_map(fix_leaf, py)
            # leaves, treedef = jax.tree_util.tree_flatten(fixed_py)
            return jax.tree_util.tree_flatten(fixed_py)

        # 1) Filter & flatten dx_next
        # dx_filtered = filter_state_data(dx_in)
        dx_flat, unravel_fn = safe_ravel_pytree(dx_in)

        dx_in_leaves, dx_treedef = safe_flatten_pytree(dx_in)
        # flat, unravel_list = _ravel_list(dx_in_leaves)
        # jax.debug.print(unravel_list)        

        # jax.debug.print("dx_flat : {}", dx_flat)
        # jax.debug.print("dx_flat : {}", unravel_fn(dx_flat).qpos)

        # Prepare the perturbed state
        leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_in))
        target_fields = {'qpos', 'qvel'}
        
        idx_target_state = [
            idx for idx, (path, leaf) in enumerate(leaves_with_path)
            if any(getattr(node, 'name', None) in target_fields for node in path)
        ]

        sizes, shapes = unzip2((jnp.size(x), jnp.shape(x)) for x in dx_in_leaves)
        indices = tuple(np.cumsum(sizes))
        # jax.debug.print("sizes : {}", sizes)
        # jax.debug.print("shapes : {}", shapes)
        # jax.debug.print("indices : {}", indices)

        # inner_idx = [list(range(dx_in_leaves[id_].shape[0])) for id_ in idx_target_state]
        # Create combinations of (outer, inner) pairs using itertools.product
        # pairs = [[state, elt] for state, inner in zip(idx_target_state, inner_idx) for elt in inner]

        inner_idx = [(np.arange(dx_in_leaves[id_].shape[0]) + indices[id_-1]).tolist() for id_ in idx_target_state]
        inner_idx = np.ravel(inner_idx)
        inner_idx = jnp.array(inner_idx)
        jax.debug.print("inner_idx : {}", inner_idx)


        # (nb_to_be_vmaps  ,   idx_in_tree,    idx in array)
        # indices = [(0,2,0),(1,2,1),(2,2,2), (3,3,0), (4,3,1),(5,3,2)]
        # indices = [dx_in_leaves[idx].shape[0] for idx in target_indices]
        
        # jax.debug.print("inner_idx : {}", inner_idx)
        # jax.debug.print("dx_in_leaves : {}", dx_in_leaves[2])

        # 2) Filter & flatten result
        result_filtered = filter_state_data(result)
        result_flat, _ = safe_ravel_pytree(result_filtered)
        result_flat_unfiltered, _ = safe_ravel_pytree(result)


        # 3) Filter & flatten the cotangent
        g_filtered = filter_state_data(g)
        g_flat, _  = safe_ravel_pytree(g_filtered)
        g_flat_unfiltered, un_flatten_g  = safe_ravel_pytree(g)

        # Flatten input u as well
        u_in_flat = u_in.ravel()
        num_u_dims = u_in_flat.shape[0]
        # num_dx_dims = dx_flat.shape[0]

        # We'll define a small helper that returns the difference
        # [dx_perturbed(u + e_i*eps) - dx_next] / eps  in flattened form.
        def fdu_plus(i):
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn(dx_in, u_in_eps)
            dx_perturbed_filtered = filter_state_data(dx_perturbed)
            dx_perturbed_flat, _ = safe_ravel_pytree(dx_perturbed_filtered)
            # jax.debug.print("Computed dx_perturbed: {dx_perturbed_flat}", dx_perturbed_flat=dx_perturbed_flat)
            # jax.debug.print("Computed dx_next: {dx_next_flat}", dx_next_flat=dx_next_flat)
            return (dx_perturbed_flat - result_flat) / eps
        
        # Can be optimised on qvel/qpos. 
        # Can be optimised with a single vmap.
        def fdx_element(idx):
            # print("idx_coord : ", idx_coord)
            # jax.debug.print("idx_coord : {}", idx_coord)
            # dx_in_eps = jax.tree_util.tree_map(lambda x: x, dx_in_leaves)
            # dx_in_eps = dx_in_leaves.copy()
            # perturbation = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), dx_in_leaves)
            # Create a perturbation mask
            perturbation = jnp.zeros_like(dx_flat)
            perturbation = dx_flat.at[idx].set(eps)
            # dx_in_eps += perturbation
            dx_in_leaves_eps = dx_flat + perturbation
            dx_eps = unravel_fn(dx_in_leaves_eps)
            # dx_eps = jax.tree_unflatten(dx_treedef,dx_in_zeros)
            dx_perturbed = step_fn(dx_eps, u_in)
            dx_perturbed_filtered = filter_state_data(dx_perturbed)
            dx_perturbed_flat, _ = safe_ravel_pytree(dx_perturbed_filtered)
            # dx_perturbed_flat, _ = safe_ravel_pytree(dx_perturbed)
            return (dx_perturbed_flat - result_flat) / eps

        # e = jnp.zeros_like(dx_flat).at[i].set(eps)
        # dx_in_eps = (dx_flat + e).reshape(dx_flat.shape)

        # Get i.
        # Apply e to dx_in
        # dx_flat[i] --> data.qpos[i] ? 
        # dx_flat[i] --> data.qvel[i + 10] ?
        # 
        #  Function perturb_state(data,i) --> data_perturb
        # Filtered info : qpos, qvel ... 
        # Filtered infos : [(0, data.qpos[0]), (1,data.qpos[1] ... (nq,data.qvel[0]), ...]


        # dx_perturbed = step_fn(dx_in_eps, u_in)
        # dx_perturbed_filtered = filter_state_data(dx_perturbed)
        # dx_perturbed_flat, _ = safe_ravel_pytree(dx_perturbed_filtered)
        # jax.debug.print("Computed dx_perturbed: {dx_perturbed_flat}", dx_perturbed_flat=dx_perturbed_flat)
        # jax.debug.print("Computed dx_next: {dx_next_flat}", dx_next_flat=dx_next_flat)
        # return (dx_perturbed_flat - result_flat) / eps

        # J shape = (num_u_dims, size_of_dx_next)
        Ju = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))
        # jax.vmap(fdx_state)(jnp.array(idx_target_state))
        Jx = jax.vmap(fdx_element)(inner_idx)

        # Now multiply J by the flattened cotangent
        # => dL/du = (∂L/∂dx_next) · (∂dx_next/∂u) = J @ ct_dx_next_flat
        d_u_flat = Ju @ g_flat
        d_x_flat = Jx @ g_flat

        dx_zero = jnp.zeros_like(dx_flat)
        dx_zero = dx_zero.at[2:5].set(d_x_flat[:3])
        dx_zero = dx_zero.at[5:8].set(d_x_flat[3:])
        d_x = unravel_fn(dx_zero)
        # print(g)

        # jax.debug.print("Computed J: {Ju}", Ju=Ju)
        # jax.debug.print("Computed J: {Jx}", Ju=Jx)
        # jax.debug.print("Computed g_flat: {g_flat}", g_flat=g_flat)
        # jax.debug.print("Computed d_u: {du}", du=d_u)

        # Reshape back to original shape of u
        d_u = d_u_flat.reshape(u_in.shape)
        # d_x = unravel_fn(d_x_flat)
        # d_x = jax.tree_unflatten(dx_tree_def,d_x_flat)

        # step_fn has exactly one input: u
        return (d_x,d_u)

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
    Rolls out a trajectory under a sequence of controls U using `step_fn(u)`.

    Args:
      mx, qpos_init: define the MuJoCo model and initial state
      step_fn: the single-argument FD-based step function
      running_cost_fn, terminal_cost_fn: user-specified costs
      U: array of controls [T, nu], for T steps

    Returns:
      states: array of shape [T, nq + nv] (or your chosen representation)
      total_cost: scalar
    """

    # Prepare the initial dx
    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))

    def scan_body(dx, u):
        dx_next = step_fn(dx, u)
        cost_t = running_cost_fn(dx_next)
        # Some users only care about qpos+qvel, but you could flatten all of dx
        # if you want a bigger "state" output. Here, we just do qpos+qvel:
        state_t = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return dx_next, (state_t, cost_t)

    dx_final, (states, costs) = jax.lax.scan(scan_body, dx0, U)
    jax.debug.print("qpos diff {qpos_diff}", qpos_diff=dx_final.qpos - dx0.qpos)
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
      1) Creates a single-argument FD-based step function from create_single_arg_step_fn
      2) Simulates a trajectory
      3) Returns the total cost

    Args:
      mx, qpos_init: MuJoCo model and initial generalized coords
      set_ctrl_fn: user logic for writing 'u' into dx.ctrl
      running_cost_fn, terminal_cost_fn: cost definitions
      eps: FD step size

    Returns:
      loss(U) -> total_cost
    """

    # Build the single-argument step function.
    dx_init = mjx.make_data(mx)
    dx_init = dx_init.replace(qpos=dx_init.qpos.at[:].set(qpos_init))

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
        JAX will see the custom_vjp inside create_single_arg_step_fn
        and perform FD-based partial derivatives with respect to u.
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

        Args:
          U0: initial guess for control trajectory (shape [T, nu])
          learning_rate: step size
          tol: convergence threshold
          max_iter: maximum allowed iterations

        Returns:
          The optimized control trajectory U that locally minimizes the cost.
        """
        U = U0
        for i in range(max_iter):
            g = self.grad_loss(U)
            U_new = U - learning_rate * g
            cost_val = self.loss(U_new)
            print("\n\n\n ----")
            print(f"Iteration {i}, cost={cost_val}")

            # Check for convergence or NaNs
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                print(f"Converged: {jnp.linalg.norm(U_new - U) < tol}, NaN: {jnp.isnan(g).any()}")
                return U_new
            U = U_new

        return U
