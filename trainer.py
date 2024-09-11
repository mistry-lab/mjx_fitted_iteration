import jax
import jax.numpy as jnp
import equinox as eqx
from config.cps import Context

def gen_targets_mapped(x, u, ctx:Context):
    xs, xt = x[:-1], x[-1]
    xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
    ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
    xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
    tcst = ctx.cbs.terminal_cost(xt)
    xcost = jnp.concatenate([xcst, tcst]) + ucost
    costs = jnp.flip(xcost)
    targets = jnp.flip(jnp.cumsum(costs))
    cost = targets[0]
    return targets, cost

def get_traj_cost(x, u, ctx:Context):
    xs, xt = x[:-1], x[-1]
    xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
    ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
    xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
    tcst = ctx.cbs.terminal_cost(xt)
    cost = jnp.concatenate([xcst, tcst]) + ucost
    return cost

def make_step(optim, model, state, loss, x, y, times):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x, y, times)
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value

def loss_fn_tagret(params, static, x, y):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
    y = y.reshape(-1, 1)
    return jnp.mean(jnp.square(pred - y))

def loss_fn_td(params, static, x, cost, times):
    model = eqx.combine(params, static)
    def value_diff(x, t, cost):
        map_model = jax.vmap(model, in_axes=(0, 0))
        x0, x1 = x[0:-1], x[1:]
        t0, t1 = t[0:-1], t[1:]
        cost_run = cost[0:-1]
        cost_term = cost[-1]
        v0, v1 = map_model(x0, t0), map_model(x1, t1)
        return v0 - v1 - cost_run, v1 - cost_term

    x, times, cost = x.reshape(-1, x.shape[-1]), times.reshape(-1, 1), cost.reshape(-1, 1)
    diff, term = jax.vmap(value_diff, in_axes=(0, 0, 0))(x, times, cost)
    return jnp.mean(jnp.square(diff) + jnp.square(term))