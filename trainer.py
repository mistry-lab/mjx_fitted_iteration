import jax
import jax.numpy as jnp
import equinox as eqx
from contexts.cps import Context

def gen_traj_targets(x, u, ctx:Context):
    xs, xt = x[:-1], x[-1]
    xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
    ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
    xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
    tcst = ctx.cbs.terminal_cost(xt)
    xcost = jnp.concatenate([xcst, tcst]) + ucost
    costs = jnp.flip(xcost)
    targets = jnp.flip(jnp.cumsum(costs))
    total_cost = targets[0]
    return targets, total_cost, tcst

def gen_traj_cost(x, u, ctx:Context):
    xs, xt = x[:-1], x[-1]
    xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
    ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
    xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
    tcst = ctx.cbs.terminal_cost(xt)
    cost = jnp.concatenate([xcst, tcst]) + ucost
    return cost, jnp.sum(ucost), tcst

def make_step(optim, model, state, loss, x, times, y):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x, times, y)
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value

def loss_fn_target(params, static, x, times, y):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]), times.reshape(-1,1))
    pred = pred.reshape(y.shape)
    loss = jnp.sum(jnp.square(pred - y), axis=-1)
    return jnp.mean(loss)

def loss_fn_td(params, static, x, times, cost):
    model = eqx.combine(params, static)
    B, T, nx = x.shape
    return jnp.mean(jnp.sum(cost, axis=-1))

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