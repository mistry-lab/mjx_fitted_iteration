import jax
import jax.numpy as jnp
import equinox as eqx
from wandb.wandb_torch import torch

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

def make_step(optim, model, state, loss, x, y, t):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x, y)
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value

def loss_fn_tagret(params, static, x, y):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
    y = y.reshape(-1, 1)
    return jnp.mean(jnp.square(pred - y))

def loss_fn_td(params, static, x, cost, time):
    model = eqx.combine(params, static)
    B, T, nx = x.shape
    cost_run = cost[..., 0:-1, :].reshape(-1, 1)
    cost_term = cost[..., -1, :].reshape(-1, 1)
    x0 = x[..., 0:-1, ...].reshape(-1, x.shape[-1])
    x1 = x[..., 1:, ...].reshape(-1, x.shape[-1])
    time = jnp.repeat(time.reshape(-1, 1), x0.shape[0], axis=0).T.reshape(-1, 1)
    t0, t1 = time[..., 0:-1], time[..., 1:]
    v0, v1 = model(x0, t0), model(x1, t1)
    diff = v0 - v1
    v_term = v1.rehsape(B, T, -1)[..., -1, :].rehspae(-1, 1)
    return jnp.mean(jnp.square(diff - cost_run) + torch.square(v_term - cost_term))
