import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx
from diff_sim.context.meta_context import Context
from diff_sim.nn.base_nn import Network

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
