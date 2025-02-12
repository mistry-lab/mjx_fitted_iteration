import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx
from diff_sim.context.meta_context import Context
from diff_sim.nn.base_nn import Network
from typing import Callable
from diff_sim.simulation.step import make_step_fn, make_step_fn_fd
from jaxtyping import PyTree

# TODO: Shall we keep jaxtyping for PyTree ? 
# Note : It seems that net (quinox NN) could be fully passed when created the simulation function
# as the update rely on tree_map function, the weight changes are properly tracked.
# For now, static part is passed for the creation and only params part is used during execution 
# which seems more logical for the gradient.
# TODO: to check.
def _simulate_fn(ctx: Context, static: PyTree, ntime: int, make_step_fn=Callable):
    step_fn = make_step_fn(ctx)
    mx = ctx.mx
    dt = mx.opt.timestep
    ctx = ctx
    static = static

    # TODO: Is ntime not part of ctx ? (Need to re-jit the entire function anyway if changed)
    # TODO: Should we create only a single run_cost function ?
    def cost_fn(mx:mjx.Model , dx:mjx.Data):
        ucost = ctx.control_cost(mx,dx)
        xcost = ctx.run_cost(mx,dx)
        return jnp.array([xcost + ucost])

    # TODO: append a flag with x that signifies wether you should terminate or not
    # TODO: make a state encoder to dertermine whether concatenate more infos
    # TODO: What if ctrl is shape 0 and we use directly forces for example
    def step(carry, _):
        dx, key, params = carry
        net = eqx.combine(params, static)
        key, subkey = jax.random.split(key)
        u = ctx.controller(net, mx, dx, subkey)
        dx = ctx.set_control(dx,u) # To get the ctrl inside dx for the cost. TODO: optimise this.
        cost = cost_fn(mx, dx)
        dx = step_fn(dx, u)
        terminated = ctx.is_terminal(mx, dx)
        x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        t = jnp.expand_dims(dx.time, axis=0)
        return (dx, key, params), jnp.concatenate([x, dx.ctrl, cost, t, terminated], axis=0)

    def rollout(dx, key, params):
        x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        (dx,_,_), res = jax.lax.scan(step, (dx, key, params), None, length=ntime-1)
        x, u, costs, ts, terminated = res[...,:-mx.nu-3], res[...,-mx.nu-3:-3], res[...,-3], res[...,-2], res[...,-1]
        x = jnp.concatenate([x_init.reshape(1,-1), x], axis=0)
        t = jnp.concatenate([jnp.array([dt]), ts], axis=0) # TODO : Shall we use dx.time instead here ? 

        # If the last time step is equal to the total time steps, then it is a terminal state
        # else it is not a terminal state. Compute the costs accordingly
        is_terminal = jnp.isclose((ts[-1]/ mx.opt.timestep), (ctx.ntotal - 1))
        def t_cost(): return ctx.terminal_cost(mx, dx);
        def r_cost(): return ctx.run_cost(mx, dx);
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

    return jax.vmap(rollout, in_axes=(0,0, None))


@eqx.filter_jit
def make_simulate_fn_fd(ctx: Context, static: PyTree, ntime: int):
    return _simulate_fn(ctx, static, ntime, make_step_fn_fd)

@eqx.filter_jit
def make_simulate_fn(ctx: Context, static: PyTree, ntime: int):
    return _simulate_fn(ctx, static, ntime, make_step_fn)