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
    eps: float = 1e-6
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
        dx_in, u_in, dx_out = res

        def map_g_to_dinput(diff_tree, grad_tree):
            def fix_leaf(d_leaf, g_leaf):
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    return jnp.zeros_like(d_leaf)
                else:
                    return g_leaf

            mapped_g = jax.tree_map(fix_leaf, diff_tree, grad_tree )
            return mapped_g

        mapped_g = map_g_to_dinput(dx_in,g)      

        # 1) Filter & flatten dx_next.
        # dx_filtered = filter_state_data(dx_in)
        dx_array, unravel_fn = ravel_pytree(dx_in)

        # 2) Get indices for vmap.   
        leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_in))
        
        # jax.debug.print("leaves with path {}", leaves_with_path)
        target_fields = {'qpos','qvel','ctrl'}

        idx_target_state = [
            idx for idx, (path, _) in enumerate(leaves_with_path)
            if any(getattr(node, 'name', None) in target_fields for node in path)
        ]

        # jax.debug.print("index_target_state {}", idx_target_state)
        

        sizes, _ = unzip2((jnp.size(leaf) if jnp.size(leaf) > 0 else 0, jnp.shape(leaf)) for _,leaf in leaves_with_path)
        indices = tuple(np.cumsum(sizes))
        # L = [(np.arange(leaves_with_path[id_][1].shape[0]) + indices[id_ - 1]).tolist()
        #         for id_ in idx_target_state ]

        inner_idx = jnp.array(
            np.hstack([
                (np.arange(leaves_with_path[id_][1].shape[0]) + indices[id_ - 1]).tolist()
                for id_ in idx_target_state
            ])
        )
        # inner_idx = jnp.array([2,3,4,5])

        # target_fields2 = {'qpos','qvel', 'ctrl'}

        # idx_target_state2 = [
        #     idx for idx, (path, _) in enumerate(leaves_with_path)
        #     if any(getattr(node, 'name', None) in target_fields2 for node in path)
        # ]

        # Sensitivity mask
        # Apply a mask to filter the perturbed elements
        sensitivity_mask = jnp.zeros_like(dx_array)  # Define the mask
        sensitivity_mask = sensitivity_mask.at[inner_idx].set(1.)  # Apply the mask only for `mask_indices`


        # 2) Filter & flatten result
        dx_out_filtered = filter_state_data(dx_out)
        dx_out_array_filtered, _ = ravel_pytree(dx_out_filtered)
        dx_out_array, _ = ravel_pytree(dx_out)

        # 3) Filter & flatten the cotangent
        g_filtered = filter_state_data(mapped_g)
        g_array_filtered, _  = ravel_pytree(g_filtered)
        g_array, unravel_g_fn  = ravel_pytree(mapped_g)

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
            # dx_perturbed = step_fn(dx_perturbed, u_in_eps)
            # dx_perturbed_filtered = filter_state_data(dx_perturbed)
            # dx_perturbed_array_filtered, _ = ravel_pytree(dx_perturbed_filtered)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            # jax.debug.print("idx : {i}, dx_perturbed_array = {dx}", i=i, dx = dx_perturbed_array[:10] )
            # jax.debug.print("idx : {i}, dx_out_array = {dx}", i=i, dx = dx_out_array[:10] )
            # jax.debug.print("u_in : {u_in}", u_in= u_in)
            # jax.debug.print("u_in_eps : {u_in_eps}", u_in_eps= u_in_eps)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps 
        
        
        
        # Can be optimised on qvel/qpos. 
        # Can be optimised with a single vmap.
        def fdx_element(idx):
            def compute_if_in_inner_idx(_):
                # Create a perturbation mask
                perturbation_array = jnp.zeros_like(dx_array)
                perturbation_array = perturbation_array.at[idx].set(eps)
                dx_array_eps = dx_array + perturbation_array
                dx_eps = unravel_fn(dx_array_eps)
                # dx_eps.timestep = 0.001
                # dx_perturbed = step_fn(dx_eps, jnp.zeros_like(u_in))
                dx_perturbed = step_fn(dx_eps, u_in)
                dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
                # jax.debug.print("idx : {idx}, dx_perturbed_array = {dx}", idx= idx, dx = dx_perturbed_array[:7] )
                return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

            def return_zeros(_):
                # jax.debug.print("ZEROS CONDITIONS: {idx}", idx= idx)
                return jnp.zeros_like(dx_array)

            # Use lax.cond with jnp.any to check if idx is in inner_idx
            is_in_inner_idx = jnp.any(inner_idx == idx)
            # jax.debug.print("Condition result: {cond}", cond=is_in_inner_idx)

            return jax.lax.cond(
                is_in_inner_idx,  # Condition
                compute_if_in_inner_idx,  # Compute perturbation if True
                return_zeros,  # Return zeros if False
                operand=None  # No additional operand needed
            )


        # J shape = (num_u_dims, size_of_dx_next)
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))
        # jax.vmap(fdx_state)(jnp.array(idx_target_state))

        # Jx_array = jax.vmap(fdx_element)(inner_idx)
        Jx_array = jax.vmap(fdx_element)(jnp.arange(dx_array.shape[0]))

        # jax.debug.print("Time {time} [s] ; Jx_array: {Jx_array}",time=dx_in.time,  Jx_array= Jx_array[2:8,2:8])
        # jax.debug.breakpoint()
        # Now multiply J by the flattened cotangent
        # => dL/du = (∂L/∂dx_next) · (∂dx_next/∂u) = g @ Ju

        # Jx_of_dx = unravel_fn(Jx_array)
        batched_unravel = jax.vmap(unravel_fn)
        result = batched_unravel(Jx_array)
        # jax.debug.breakpoint()

         
        # g = unravel_g_fn(g)

        d_u_flat = Ju_array @ g_array
        d_x_flat = Jx_array @ g_array

        # import jax
        # jax.debug.breakpoint()

        dx_zero = jnp.zeros_like(dx_array)
        # dx_zero = dx_zero.at[2:5].set(d_x_flat[:3])
        # dx_zero = dx_zero.at[5:8].set(d_x_flat[3:])
        d_x = unravel_fn(d_x_flat)
        # print(g)

        # jax.debug.print("Computed J: {Ju}", Ju=Ju)
        # jax.debug.print("Computed J: {Jx}", Ju=Jx)
        # jax.debug.print("Computed g_flat: {g_flat}", g_flat=g_flat)
        # jax.debug.print("Computed d_u: {du}", du=d_u)

        # Reshape back to original shape of u
        d_u = d_u_flat.reshape(u_in.shape)
        # d_x = unravel_fn(d_x_flat)
        # d_x = jax.tree_unflatten(dx_tree_def,d_x_flat)

        # jax.debug.breakpoint()

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
    dx0 = mjx.step(mx,dx0)
    # dx0 = mjx.step(mx,dx0)
    # dx0 = mjx.step(mx,dx0)
    # dx0 = mjx.step(mx,dx0)
    def scan_body(dx, u):
        dx_next = step_fn(dx, u)
        cost_t = running_cost_fn(dx_next)
        # Some users only care about qpos+qvel, but you could flatten all of dx
        # if you want a bigger "state" output. Here, we just do qpos+qvel:
        state_t = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return dx_next, (state_t, cost_t)

    dx_final, (states, costs) = jax.lax.scan(scan_body, dx0, U)
    # jax.debug.print("qpos diff {qpos_diff}", qpos_diff=dx_final.qpos - dx0.qpos)
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
    dx_init = mjx.step(mx,dx_init)

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
            print(f"Initial U : {U[:10]}")
            print(f"gradient = {g[:10]}")
            print(f"New U : {U}")

            # Check for convergence or NaNs
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                print(f"Converged: {jnp.linalg.norm(U_new - U) < tol}, NaN: {jnp.isnan(g).any()}")
                return U_new
            U = U_new

        return U
