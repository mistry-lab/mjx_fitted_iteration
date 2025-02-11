import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx
from diff_sim.context.meta_context import Context
from diff_sim.nn.base_nn import Network
from typing import Callable
from diff_sim.context.meta_context import FDCache
from jax.flatten_util import ravel_pytree

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
        g_array, _ = ravel_pytree(mapped_g)
 
        # Flatten dx_in and dx_out just once
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
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps
 
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))  # shape = (num_u_dims, dx_dim)
 
        # =====================================================
        # ================ FD wrt state (dx) ==================
        # =====================================================
        def fdx_for_index(idx):
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps
 
        # Only do FD over "inner_idx" (subset of the state: qpos, qvel, ctrl, etc.)
        Jx_rows = jax.vmap(fdx_for_index)(inner_idx)
 
        # Scatter those rows back to a full (dx_dim, dx_dim) matrix
        def scatter_rows(subset_rows, subset_indices, full_shape):
            base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)
 
        dx_dim = dx_array.size
        Jx_array = scatter_rows(Jx_rows, inner_idx, (dx_dim, dx_dim))
 
        # =====================================================
        # ================== Combine with g ====================
        # =====================================================
        d_u_flat = Ju_array @ g_array     # shape = (num_u_dims,)
        d_x_flat = Jx_array @ g_array     # shape = (dx_dim,)
 
        d_x = unravel_dx(d_x_flat)
        d_u = d_u_flat.reshape(u_in.shape)
 
        return (d_x, d_u)
 
    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


@eqx.filter_jit
def controlled_simulate_fd(dxs:mjx.Data, ctx: Context, net: Network, key: jnp.ndarray, ntime: int):
    mx = ctx.cfg.mx

    step_fn = make_step_fn(mx, ctx.cbs.set_control, ctx.fd_cache)

    def cat_pos_vel(dx):
        return jnp.concatenate([dx.qpos, dx.qvel], axis=0)

    def cost_fn(mx:mjx.Model , dx:mjx.Data):
        ucost = ctx.cbs.control_cost(mx,dx)
        xcost = ctx.cbs.run_cost(mx,dx)
        return jnp.array([xcost + ucost])

    # TODO: append a flag with x that signifies wether you should terminate or not
    def step(carry, _):
        dx, key = carry
        key, subkey = jax.random.split(key)
        u = ctx.cbs.controller(net, mx, dx, subkey)
        dx = ctx.cbs.set_control(dx,u) # To get the ctrl inside dx for the cost. TODO: optimise this.
        cost = cost_fn(mx, dx)
        dx = step_fn(dx, u)
        # dx = mjx.step(mx, dx) # Dynamics function
        terminated = ctx.cbs.is_terminal(mx, dx)
        x = cat_pos_vel(dx)
        t = jnp.expand_dims(dx.time, axis=0)
        return (dx, key), jnp.concatenate([x, dx.ctrl, cost, t, terminated], axis=0)

    @jax.vmap
    def rollout(dx):
        x_init = cat_pos_vel(dx)
        (dx,_), res = jax.lax.scan(step, (dx, key), None, length=ntime-1)
        x, u, costs, ts, terminated = res[...,:-mx.nu-3], res[...,-mx.nu-3:-3], res[...,-3], res[...,-2], res[...,-1]
        x = jnp.concatenate([x_init.reshape(1,-1), x], axis=0)
        t = jnp.concatenate([jnp.array([ctx.cfg.dt]), ts], axis=0)

        # If the last time step is equal to the total time steps, then it is a terminal state
        # else it is not a terminal state. Compute the costs accordingly
        is_terminal = jnp.isclose((ts[-1]/ mx.opt.timestep), (ctx.cfg.ntotal - 1))
        def t_cost(): return ctx.cbs.terminal_cost(mx, dx);
        def r_cost(): return ctx.cbs.run_cost(mx, dx);
        term_cost = jax.lax.cond(is_terminal, t_cost, r_cost)
        zeros = jnp.zeros_like(term_cost)
        costs = jnp.concatenate([costs, zeros.reshape(-1)], axis=0)

        # Mask the gradients of the costs that are after the termination
        termination_mask = jnp.concatenate([
            jnp.array([False]),  # Ignore the first cost
            jnp.cumsum(terminated) > 0  # True from first termination onward
        ], axis=0)
        costs = costs * jnp.logical_not(termination_mask)
        return dx, x, u, costs, t, jnp.any(termination_mask)

    return rollout(dxs)


@eqx.filter_jit
def controlled_simulate(dxs:mjx.Data, ctx: Context, net: Network, key: jnp.ndarray, ntime: int):
    mx = ctx.cfg.mx

    def cat_pos_vel(dx):
        return jnp.concatenate([dx.qpos, dx.qvel], axis=0)

    def cost_fn(mx:mjx.Model , dx:mjx.Data):
        ucost = ctx.cbs.control_cost(mx,dx)
        xcost = ctx.cbs.run_cost(mx,dx)
        return jnp.array([xcost + ucost])

    # TODO: append a flag with x that signifies wether you should terminate or not
    def step(carry, _):
        dx, key = carry
        key, subkey = jax.random.split(key)
        dx, u = ctx.cbs.controller(net, mx, dx, subkey)
        cost = cost_fn(mx, dx)
        dx = mjx.step(mx, dx) # Dynamics function
        terminated = ctx.cbs.is_terminal(mx, dx)
        x = cat_pos_vel(dx)
        t = jnp.expand_dims(dx.time, axis=0)
        return (dx, key), jnp.concatenate([x, dx.ctrl, cost, t, terminated], axis=0)

    @jax.vmap
    def rollout(dx):
        x_init = cat_pos_vel(dx)
        (dx,_), res = jax.lax.scan(step, (dx, key), None, length=ntime-1)
        x, u, costs, ts, terminated = res[...,:-mx.nu-3], res[...,-mx.nu-3:-3], res[...,-3], res[...,-2], res[...,-1]
        x = jnp.concatenate([x_init.reshape(1,-1), x], axis=0)
        t = jnp.concatenate([jnp.array([ctx.cfg.dt]), ts], axis=0)

        # If the last time step is equal to the total time steps, then it is a terminal state
        # else it is not a terminal state. Compute the costs accordingly
        is_terminal = jnp.isclose((ts[-1]/ mx.opt.timestep), (ctx.cfg.ntotal - 1))
        def t_cost(): return ctx.cbs.terminal_cost(mx, dx);
        def r_cost(): return ctx.cbs.run_cost(mx, dx);
        # term_cost = jax.lax.cond(is_terminal, t_cost, r_cost)
        term_cost = t_cost()
        zeros = jnp.zeros_like(term_cost)
        costs = jnp.concatenate([costs, zeros.reshape(-1)], axis=0)

        # Mask the gradients of the costs that are after the termination

        termination_mask = jnp.concatenate([
            jnp.array([False]),  # Ignore the first cost
            jnp.cumsum(terminated) > 0  # True from first termination onward
        ], axis=0)
        # costs = costs * jnp.logical_not(termination_mask)
        return dx, x, u, costs, t, jnp.any(termination_mask)

    return rollout(dxs)
