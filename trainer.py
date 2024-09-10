import time
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import wandb
from utils.mj_vis import animate_trajectory
from utils.tqdm import trange
from config.cps import ctx, Context
from simulate import controlled_simulate

def gen_targets_mapped(x, u, ctx:Context):
    xs, xt = x[:-1], x[-1]
    xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
    ucost = ctx.cbs.control_cost(u)
    xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
    tcst = ctx.cbs.terminal_cost(xt)
    xcost = jnp.concatenate([xcst, tcst]) + ucost
    costs = jnp.flip(xcost)
    targets = jnp.flip(jnp.cumsum(costs))

    cost = targets[0] 

    return targets, cost

def make_step(optim, model, state, loss, x, y):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x, y)
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value

def loss_fn(params, static, x, y):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
    y = y.reshape(-1, 1)
    return jnp.mean(jnp.square(pred - y))