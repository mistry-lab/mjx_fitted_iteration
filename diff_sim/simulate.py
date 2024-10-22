import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx
from diff_sim.context.meta_context import Context
from diff_sim.nn.base_nn import Network

# TODO: This function is jitted twice here and in make_step. This makes debugging break due tracing issues
# TODO: this was done because in visualise_policy we need to call this function and we jitted here for speed
# TODO: so best option is to not jit here and jit in visualise_policy separately or jit here and jit the other components
# TODO: in make_step separately but simplest is to jit in visualise_policy and not here
def controlled_simulate(x_inits:jnp.ndarray, ctx: Context, net: Network, user_key: jnp.ndarray):
    mx = ctx.cfg.mx
    keys = jax.random.split(user_key, x_inits.shape[0])

    def cost_fn(x, u):
        ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
        xcst = ctx.cbs.run_cost(x) * ctx.cfg.dt
        return jnp.array([ucost + xcst])

    def set_init(x):
        dx = mjx.make_data(mx)
        qpos = dx.qpos.at[:].set(x[:mx.nq])
        qvel = dx.qvel.at[:].set(x[mx.nq:])
        dx = dx.replace(qpos=qpos, qvel=qvel)
        return mjx.step(mx, dx)

    # TODO: make cost function take mx, dx and return cost
    # TODO: Remove the first encoding of state. If user needs encoded state they should do it in controller
    # TODO: state encoder mx, dx and returns x
    def step(carry, _):
        dx, key = carry
        key, subkey = jax.random.split(key)
        x = ctx.cbs.state_encoder(jnp.concatenate([dx.qpos, dx.qvel], axis=0))
        t = jnp.expand_dims(dx.time, axis=0)
        dx, u = ctx.cbs.controller(x, t, net, ctx.cfg, mx, dx, subkey)
        cost = cost_fn(x, u)
        dx = mjx.step(mx, dx)
        x = ctx.cbs.state_encoder(jnp.concatenate([dx.qpos, dx.qvel], axis=0))
        return (dx, key), jnp.concatenate([x, dx.ctrl, cost, t], axis=0)

    @jax.vmap
    def rollout(x_init, key):
        dx = set_init(x_init)
        x_init = ctx.cbs.state_encoder(x_init)
        _, res = jax.lax.scan(step, (dx, key), None, length=ctx.cfg.nsteps-1)
        x, u, costs, ts = res[...,:-mx.nu-2], res[...,-mx.nu-2:-2], res[...,-2], res[...,-1]
        x = jnp.concatenate([x_init.reshape(1,-1), x], axis=0)
        tcost = ctx.cbs.terminal_cost(x[-1]) # Terminal cost
        costs = jnp.concatenate([costs, tcost.reshape(-1)], axis=0)
        t = jnp.concatenate([ts,jnp.array([ts[-1] + ctx.cfg.dt])], axis=0)
        return x, u, costs, t

    return rollout(x_inits, keys)
