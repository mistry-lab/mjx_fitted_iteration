import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx
from diff_sim.context.meta_context import Context
from typing import Callable
from diff_sim.simulation.step import make_step_fn, make_step_fn_fd


# TODO: Shall we keep jaxtyping for PyTree ?
def _simulate_fn(ctx: Context, make_step_fn=Callable):
    step_fn = make_step_fn(ctx)  # Create step function

    # TODO: Should we create only a single run_cost function ? YES
    # TODO: make a state encoder to dertermine whether concatenate more infos (pending)
    # TODO: What if ctrl is shape 0 and we use directly forces for example
    def simulate(dxs, key, net):
        def cost_fn(mx: mjx.Model, dx: mjx.Data):
            rcost = ctx.run_cost(mx, dx)
            return jnp.array([rcost])

        def step(carry, _):
            dx, key, params = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            dx, u = ctx.controller(model, ctx.mx, dx, subkey) # Fix input/output
            dx = ctx.set_control(
                dx, u
            )  # To get the ctrl inside dx for the cost. TODO: optimise this.
            cost = cost_fn(ctx.mx, dx)
            dx = step_fn(dx, u)
            terminated = ctx.is_terminal(ctx.mx, dx)
            x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            t = jnp.expand_dims(dx.time, axis=0)
            return (dx, key, params), jnp.concatenate([x, dx.ctrl, cost, t, terminated], axis=0)

        def rollout(dx, key, params):
            x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            (dx, _, _), res = jax.lax.scan(step, (dx, key, params), None, length=ctx.nsteps - 1)
            x, u, costs, ts, terminated = (
                res[..., : -ctx.mx.nu - 3],
                res[..., -ctx.mx.nu - 3 : -3],
                res[..., -3],
                res[..., -2],
                res[..., -1],
            )
            x = jnp.concatenate([x_init.reshape(1, -1), x], axis=0)
            t = jnp.concatenate([jnp.array([ctx.mx.opt.timestep]), ts], axis=0)

            # If the last time step is equal to the total time steps, then it is a terminal state
            # else it is not a terminal state. Compute the costs accordingly
            is_terminal = jnp.isclose((ts[-1] / ctx.mx.opt.timestep), (ctx.ntotal - 1))

            def t_cost():
                return ctx.terminal_cost(ctx.mx, dx)

            def r_cost():
                return ctx.run_cost(ctx.mx, dx)

            term_cost = jax.lax.cond(is_terminal, t_cost, r_cost)
            zeros = jnp.zeros_like(term_cost)
            costs = jnp.concatenate([costs, zeros.reshape(-1)], axis=0)

            # Mask the gradients of the costs that are after the termination
            termination_mask = jnp.concatenate(
                [
                    jnp.array([False]),  # Ignore the first cost
                    jnp.cumsum(terminated) > 0,  # True from first termination onward
                ],
                axis=0,
            )
            costs = costs * jnp.logical_not(termination_mask)
            return dx, x, u, costs, t, jnp.any(termination_mask)

        params, static = eqx.partition(net, eqx.is_array)
        return jax.vmap(rollout, in_axes=(0, 0, None))(dxs, key, params)

    return simulate

def make_simulate_fn_fd(ctx: Context):
    return _simulate_fn(ctx, make_step_fn_fd)

def make_simulate_fn(ctx: Context):
    return _simulate_fn(ctx, make_step_fn)
