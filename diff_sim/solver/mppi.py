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

def mppi(dx,key,net_p_fn, net_v_fn,ctx:ParamtersMPPI):

    # Clone dx nrollout times
    dxs = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], ctx.nrollout, axis=0), dx)

    def get_initial_trajectory(net_p_fn, key):
        def step(carry, _):
            dx, key = carry
            key, subkey = jax.random.split(key)
            u0 = net_p_fn(dx,subkey)
            dx = ctx.set_control(dx, u0)
            dx = mjx.step(ctx.mx,dx)
            return (dx, key),u0
        dx0 = jax.tree_util.tree_map(lambda x: x, dx)
        (_,_), u = jax.lax.scan(step, (dx0, key), length=ctx.horizon)
        return u

    def step(carry, u0):
        dx, key = carry
        key, key_du = jax.random.split(key) 
        du = jax.random.normal(key_du, shape=u0.shape)
        u = u0 + du
        dx = ctx.set_control(dx, u)
        cost_r = ctx.run_cost(ctx.mx, dx)
        dx = mjx.step(ctx.mx,dx)
        x = jnp.concatenate([dx.qpos, dx.qvel])
        return (dx, key), (x,cost_r,u,du)
    
    def terminal_cost(mx, dx, key):
        return net_v_fn(dx,key)

    def rollout(dx, key,u0s):
        x_init = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        (dx, _), (x,cost_r,u,du) = jax.lax.scan(step, (dx, key), u0s)

        x = jnp.concatenate([x_init.reshape(1, -1), x], axis=0)
        costs = jnp.concatenate(
                [cost_r, jnp.expand_dims(terminal_cost(ctx.mx, dx, key), axis=0)], axis=0
            )
        return (x,costs,u,du)


    rkey, ukey = jax.random.split(key)
    rkeys = jax.random.split(rkey, num= ctx.nrollout)
    u0s = get_initial_trajectory(net_p_fn, ukey)
    _,costs,_,dus = jax.vmap(rollout, in_axes=(0,0,None))(dxs, rkeys, u0s)
    # average the total trajecotry costs over K. cost_r = (B x T)
    csts = jnp.sum(costs, axis=-1)   # (B x)
    min_cst = jnp.min(csts)
    scale_csts = (csts - min_cst) * (-1/ctx.temp)
    exp_csts = jnp.exp(scale_csts)
    norm_cst = jnp.sum(exp_csts)
    # prob = jnp.expand_dims(exp_csts / norm_cst, axis=-1)
    prob = exp_csts / norm_cst
    shape = tuple([prob.shape[0]] + [1] * len(u0s.shape))
    prob = jnp.reshape(exp_csts / norm_cst, shape=shape)
    res = jnp.sum(prob * dus, axis=0) + u0s 
    return res[0]