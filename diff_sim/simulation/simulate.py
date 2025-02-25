import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx
from diff_sim.context.meta_context import Context
from typing import Callable
from diff_sim.simulation.step import make_step_fn, make_step_fn_fd

# TODO: Shall we keep jaxtyping for PyTree ?
# Remove nets from simulate, since no grads computed 
def _simulate_fn_mppi(ctx: Context, make_step_fn=Callable):
    step_fn = make_step_fn(ctx)  # Create step function

    # TODO: make a state encoder to dertermine whether concatenate more infos (pending)
    # TODO: What if ctrl is shape 0 and we use directly forces for example
    def simulate(dxs, key, nets):
        def step(carry, _):
            dx, key, params = carry
            params_p, params_v = params
            net_p = eqx.combine(params_p, static_p)
            net_v = eqx.combine(params_v, static_v)
            key, subkey = jax.random.split(key)
            dx, u = ctx.controller((net_p, net_v), ctx.mx, dx, subkey)  # Fix input/output
            dx = ctx.set_control(
                dx, u
            )  # To get the ctrl inside dx for the cost. TODO: optimise this.
            cost_r = ctx.run_cost(ctx.mx, dx)
            cost_t = ctx.terminal_cost(ctx.mx, dx)
            dx = step_fn(dx, u)

            terminated_s = ctx.is_terminal(ctx.mx, dx)  # State termination
            x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            # t = jnp.expand_dims(dx.time, axis=0)
            return (dx, key, params), (
                x,
                dx.ctrl,
                cost_r,
                cost_t,
                dx.time,
                terminated_s,
            )

        def rollout(dx, key, params):
            t0 = dx.time
            x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            (dx, _, _), res = jax.lax.scan(
                step, (dx, key, params), None, length=ctx.nsteps
            )
            x, u, costs_r, costs_t, ts, terminated_state = res

            # TODO: Rethink the cost mechanism. Could be improve
            x = jnp.concatenate([x_init.reshape(1, -1), x], axis=0)
            # ts = jnp.concatenate([jnp.array([ctx.mx.opt.timestep]), ts], axis=0)
            ts = jnp.concatenate([jnp.array([t0]), ts], axis=0)
            costs_r = jnp.concatenate(
                [costs_r, jnp.expand_dims(ctx.run_cost(ctx.mx, dx), axis=0)], axis=0
            )
            costs_t = jnp.concatenate(
                [costs_t, jnp.expand_dims(ctx.terminal_cost(ctx.mx, dx), axis=0)],
                axis=0,
            )

            terminated_time_mask = jax.vmap(
                lambda t: (round(t / (3*ctx.mx.opt.timestep)) >= (ctx.ntotal))
            )(ts)
            # Replace cost_r with cost_t values when necessary
            costs_r = jnp.where(terminated_time_mask, costs_t, costs_r)

            # Replace first True value with False
            idx = jnp.argmax(terminated_time_mask)
            terminated_time_mask_cost = terminated_time_mask.at[idx].set(False)

            # Mask the gradients of the costs that are after the termination
            termination_state_mask = jnp.concatenate(
                [
                    jnp.array([False]),  # Ignore the first cost
                    jnp.cumsum(terminated_state)
                    > 0,  # True from first termination onward
                ],
                axis=0,
            )
            termination_mask = termination_state_mask | terminated_time_mask_cost
            terminated = termination_state_mask | terminated_time_mask
            costs_r = costs_r * jnp.logical_not(termination_mask)
            return dx, x, u, costs_r, ts, jnp.any(terminated)

        net_p, net_v = nets
        params_p, static_p = eqx.partition(net_p, eqx.is_array)
        params_v, static_v = eqx.partition(net_v, eqx.is_array)
        keys = jax.random.split(key, num=dxs.qpos.shape[0])
        return jax.vmap(rollout, in_axes=(0, 0, None))(dxs, keys, (params_p, params_v))

    return simulate



# TODO: Shall we keep jaxtyping for PyTree ?
def _simulate_fn(ctx: Context, make_step_fn=Callable):
    step_fn = make_step_fn(ctx)  # Create step function

    # TODO: make a state encoder to dertermine whether concatenate more infos (pending)
    # TODO: What if ctrl is shape 0 and we use directly forces for example
    def simulate(dxs, key, net):
        def step(carry, _):
            dx, key, params = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            dx, u = ctx.controller(model, ctx.mx, dx, subkey)  # Fix input/output
            dx = ctx.set_control(
                dx, u
            )  # To get the ctrl inside dx for the cost. TODO: optimise this.
            cost_r = ctx.run_cost(ctx.mx, dx)
            cost_t = ctx.terminal_cost(ctx.mx, dx)
            dx = step_fn(dx, u)
            dx = ctx.set_control(dx, u)
            dx = step_fn(dx, u)
            dx = ctx.set_control(dx, u)
            dx = step_fn(dx, u)
            terminated_s = ctx.is_terminal(ctx.mx, dx)  # State termination
            x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            # t = jnp.expand_dims(dx.time, axis=0)
            return (dx, key, params), (
                x,
                dx.ctrl,
                cost_r,
                cost_t,
                dx.time,
                terminated_s,
            )

        def rollout(dx, key, params):
            t0 = dx.time
            x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            (dx, _, _), res = jax.lax.scan(
                step, (dx, key, params), None, length=ctx.nsteps
            )
            x, u, costs_r, costs_t, ts, terminated_state = res

            # TODO: Rethink the cost mechanism. Could be improve
            x = jnp.concatenate([x_init.reshape(1, -1), x], axis=0)
            # ts = jnp.concatenate([jnp.array([ctx.mx.opt.timestep]), ts], axis=0)
            ts = jnp.concatenate([jnp.array([t0]), ts], axis=0)
            costs_r = jnp.concatenate(
                [costs_r, jnp.expand_dims(ctx.run_cost(ctx.mx, dx), axis=0)], axis=0
            )
            costs_t = jnp.concatenate(
                [costs_t, jnp.expand_dims(ctx.terminal_cost(ctx.mx, dx), axis=0)],
                axis=0,
            )

            terminated_time_mask = jax.vmap(
                lambda t: (round(t / (3*ctx.mx.opt.timestep)) >= (ctx.ntotal))
            )(ts)
            # Replace cost_r with cost_t values when necessary
            costs_r = jnp.where(terminated_time_mask, costs_t, costs_r)

            # Replace first True value with False
            idx = jnp.argmax(terminated_time_mask)
            terminated_time_mask_cost = terminated_time_mask.at[idx].set(False)

            # Mask the gradients of the costs that are after the termination
            termination_state_mask = jnp.concatenate(
                [
                    jnp.array([False]),  # Ignore the first cost
                    jnp.cumsum(terminated_state)
                    > 0,  # True from first termination onward
                ],
                axis=0,
            )
            termination_mask = termination_state_mask | terminated_time_mask_cost
            terminated = termination_state_mask | terminated_time_mask
            costs_r = costs_r * jnp.logical_not(termination_mask)
            return dx, x, u, costs_r, ts, jnp.any(terminated)

        params, static = eqx.partition(net, eqx.is_array)
        keys = jax.random.split(key, num=dxs.qpos.shape[0])
        return jax.lax.stop_gradient(jax.vmap(rollout, in_axes=(0, 0, None))(dxs, keys, params))

    return simulate


def _simulate_simple(ctx: Context, make_step_fn=Callable):
    step_fn = make_step_fn(ctx)  # Create step function

    def simulate(dxs, key, net):
        def step(carry, _):
            dx, key, params = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            dx, u = ctx.controller(model, ctx.mx, dx, subkey)  # Fix input/output
            dx = ctx.set_control(dx, u)
            # Collect intermediate states
            states = []
            for _ in range(3):
                dx = ctx.set_control(dx, u)
                dx = step_fn(dx, u)
                states.append(jnp.concatenate([dx.qpos, dx.qvel], axis=0))
            x = jnp.stack(states, axis=0)
            return (dx, key, params), x

        def rollout(dx, key, params):
            x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            (_, _, _), x = jax.lax.scan(step, (dx, key, params), None, length=ctx.ntotal)
            # jax.debug.breakpoint()

            x_flat = x.reshape(-1, x.shape[-1])  # Shape: (ctx.ntotal*3, state_dim)
            # Prepend the initial state so the final output has shape 
            # (1 + ctx.ntotal*3, state_dim)
            x = jnp.concatenate([x_init[None, :], x_flat], axis=0)
            return x

        params, static = eqx.partition(net, eqx.is_array)
        keys = jax.random.split(key, num=dxs.qpos.shape[0])
        return jax.vmap(rollout, in_axes=(0, 0, None))(dxs, keys, params)

    return simulate

def _simulate_simple_mppi(ctx: Context, make_step_fn=Callable):
    step_fn = make_step_fn(ctx)  # Create step function

    def simulate(dxs, key, nets):
        def step(carry, _):
            dx, key, params = carry
            params_p, params_v = params
            net_p = eqx.combine(params_p, static_p)
            net_v = eqx.combine(params_v, static_v)
            key, subkey = jax.random.split(key)
            dx, u = ctx.controller((net_p,net_v), ctx.mx, dx, subkey)  # Fix input/output
            dx = ctx.set_control(dx, u)
            # Collect intermediate states
            dx = ctx.set_control(dx, u)
            dx = step_fn(dx, u)
            x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            # states = []
            # for _ in range(3):
            #     dx = ctx.set_control(dx, u)
            #     dx = step_fn(dx, u)
            #     states.append(jnp.concatenate([dx.qpos, dx.qvel], axis=0))
            # x = jnp.stack(states, axis=0)
            return (dx, key, params), x

        def rollout(dx, key, params):
            x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
            (_, _, _), x = jax.lax.scan(step, (dx, key, params), None, length=ctx.ntotal)
            # jax.debug.breakpoint()

            x_flat = x.reshape(-1, x.shape[-1])  # Shape: (ctx.ntotal*3, state_dim)
            # Prepend the initial state so the final output has shape 
            # (1 + ctx.ntotal*3, state_dim)
            x = jnp.concatenate([x_init[None, :], x_flat], axis=0)
            return x

        net_p, net_v = nets
        params_p, static_p = eqx.partition(net_p, eqx.is_array)
        params_v, static_v = eqx.partition(net_v, eqx.is_array)
        keys = jax.random.split(key, num=dxs.qpos.shape[0])
        return jax.vmap(rollout, in_axes=(0, 0, None))(dxs, keys, (params_p, params_v))

    return simulate

def make_simulate_fn_fd(ctx: Context):
    return _simulate_fn(ctx, make_step_fn_fd)

def make_simulate_fn(ctx: Context):
    return _simulate_fn(ctx, make_step_fn)

def make_simulate_fn_simple(ctx: Context):
    return _simulate_simple(ctx, make_step_fn)

def make_simulate_fn_mppi(ctx: Context):
    return _simulate_fn_mppi(ctx, make_step_fn)

def make_simulate_simple_mppi(ctx: Context):
    return _simulate_simple_mppi(ctx, make_step_fn)