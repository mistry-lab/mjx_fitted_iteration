import jax
from dataclasses import dataclass
from typing import Callable
from mujoco import mjx
import jax.numpy as jnp

@jax.tree_util.register_static
@dataclass(frozen=True)
class ParamtersMPPI():
    horizon: int
    nrollout: int
    temp: float
    mx: mjx.Model
    run_cost : Callable[[mjx.Model, mjx.Data], jnp.ndarray]
    terminal_cost : Callable[[mjx.Model, mjx.Data], jnp.ndarray]
    set_control : Callable[[mjx.Data, jnp.ndarray], mjx.Data]

def mppi(dx,key,net_mppi,ctx:ParamtersMPPI):

    # Clone dx nrollout times
    dxs = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], ctx.nrollout, axis=0), dx)

    def step(carry, u0):
        dx, key = carry
        key, subkey = jax.random.split(key) 
        du = jax.random.normal(subkey)
        u = u0 + du
        dx = ctx.set_control(dx, u)
        cost_r = ctx.run_cost(ctx.mx, dx)
        dx = mjx.step(ctx.mx,dx)
        x = jnp.concatenate([dx.qpos, dx.qvel])
        return (dx, key), (x,cost_r,du)
    
    def terminal_cost(mx, dx, key):
        return net_mppi(dx,key)

    def rollout(dx, key):
        x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        (dx, _), (xs,costs_r,dus) = jax.lax.scan(step, (dx, key), us)

        xs = jnp.concatenate([x_init.reshape(1, -1), xs], axis=0)
        costs_r = jnp.concatenate(
                [costs_r, jnp.expand_dims(terminal_cost(ctx.mx, dx, key), axis=0)], axis=0
            )
        return (xs,costs_r,dus)

    rkey, ukey = jax.random.split(key)
    rkeys = jax.random.split(rkey, num= ctx.nrollout)
    us = jax.random.normal(ukey, shape=ctx.horizon)
    _,cost_r,dus = jax.vmap(rollout, in_axes=(0, 0, None))(dxs, rkeys)
    # average the total trajecotry costs over K. cost_r = (B x T)
    csts = jnp.sum(cost_r, axis=-1)   # (B x)
    min_cst = jnp.min(csts)
    scale_csts = (csts - min_cst) * (-1/ctx.temp)
    exp_csts = jnp.exp(scale_csts)
    norm_cst = jnp.sum(exp_csts)
    prob = jnp.expand_dims(exp_csts / norm_cst, axis=-1)
    res = jnp.sum(prob * dus, axis=0) + us 
    return res[0]