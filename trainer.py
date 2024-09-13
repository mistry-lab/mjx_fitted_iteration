import jax
import jax.numpy as jnp
import equinox as eqx
from jax.experimental.array_api import reshape

from config.cps import Context

def gen_targets_mapped(x, u, ctx:Context):
    xs, xt = x[:-1], x[-1]
    xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
    ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
    xcst = ctx.cbs.run_cost(xs) * 0.01
    tcst = ctx.cbs.terminal_cost(xt)
    xcost = jnp.concatenate([xcst, tcst])
    # xcost = jnp.concatenate([xcst, tcst]) + ucost * ctx.cfg.dt
    costs = jnp.flip(xcost)
    terminal_cost = costs[0]

    targets = jnp.flip(jnp.cumsum(costs))
    total_cost = targets[0]    
    
    return targets, total_cost, terminal_cost

def gen_traj_cost(x, u, ctx:Context):
    xs, xt = x[:-1], x[-1]
    xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
    ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
    xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
    tcst = ctx.cbs.terminal_cost(xt)
    cost = jnp.concatenate([xcst, tcst]) + ucost
    return cost, None, tcst

def make_step(optim, model, state, loss, x, times, y):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x, times, y)
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value

def loss_fn_target(params, static, x, times, y):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]), times.reshape(-1,1))
    y = y.reshape(-1, 1)
    return jnp.mean(jnp.square(pred - y))

def loss_fn_td(params, static, x, times, cost):
    model = eqx.combine(params, static)
    B, T, nx = x.shape

    def v_diff(x, t):
        v_seq = jax.vmap(model)(x, t)
        v0, v1 = v_seq[0:-1], v_seq[1:]
        return v0 - v1, v_seq[-1]

    def v_r_cost(v_diff, v_term, cost):
        v_diff_cost = v_diff - cost[:-1]
        v_term_cost = v_term - cost[-1]
        return jnp.sum(jnp.square(v_diff_cost) + jnp.square(v_term_cost))

    v_diff, v_term = jax.vmap(v_diff)(x, times)
    cost = jax.vmap(v_r_cost)(v_diff.reshape(B, T-1), v_term.reshape(B, 1), cost)
    return jnp.mean(cost)