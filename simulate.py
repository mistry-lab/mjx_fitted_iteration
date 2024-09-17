import jax
import jax.numpy as jnp
from mujoco import mjx
from contexts.di import Context

def cost_fn(x, u, ctx:Context):
    ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
    xcst = ctx.cbs.run_cost(x) * ctx.cfg.dt
    return jnp.array([xcst + ucost])

def controlled_simulate(x_inits, mx, ctx, net, PD=False):
    def set_init(x):
        dx = mjx.make_data(mx)
        qpos = dx.qpos.at[:].set(x[:mx.nq])
        qvel = dx.qvel.at[:].set(x[mx.nq:])
        dx = dx.replace(qpos=qpos, qvel=qvel)
        return mjx.step(mx, dx)

    # def get_ctrl(mx, args):
    #     dx, vf = args
    #     x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
    #     t = jnp.expand_dims(dx.time, axis=0)

    #     # act_id = mx.actuator_trnid[:, 0]
    #     # M = mjx.full_m(mx, dx)
    #     # invM = jnp.linalg.inv(M)
    #     # dvdx = jax.jacrev(vf,0)(x, t)
    #     # G = jnp.vstack([jnp.zeros_like(invM), invM])
    #     # invR = jnp.linalg.inv(ctx.cfg.R)

    #     # u = (-1/2 * invR @ G.T[act_id, :] @ dvdx.T).squeeze()
    #     u = vf(x,t)
    #     # Compute running cost.
    #     cost = ctx.cbs.control_cost(u) * ctx.cfg.dt + ctx.cbs.run_cost(x) * ctx.cfg.dt

    #     ctrl = dx.ctrl.at[:].set(u)
    #     dx = dx.replace(ctrl=ctrl)
    #     return dx, cost

    def step(carry, _):
        dx = carry
        x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        t = jnp.expand_dims(dx.time, axis=0)
        u = net(x,t) # Policy function
        # u = -1.*dx.qpos - 0.05*dx.qvel
        cost = cost_fn(x, u, ctx) # Cost function

        ctrl = dx.ctrl.at[:].set(u)
        dx = dx.replace(ctrl=ctrl)
        dx = mjx.step(mx, dx) # Dynamics function
        return dx, jnp.concatenate([dx.qpos, dx.qvel, dx.ctrl, cost], axis=0)

    dx = set_init(x_inits)
    _, res = jax.lax.scan(step, dx, None, length=ctx.cfg.nsteps)
    x, u, costs = res[...,:-mx.nu-1], res[...,-mx.nu-1:-1], res[...,-1]
    u = u.at[-1].set(0.)
    costs = jnp.sum(costs) + ctx.cbs.terminal_cost(x[-1]) # Sum and add terminal cost
    # costs = jnp.sum(costs)
    return costs




def compute_traj(x_inits, mx, ctx, net, PD=False):
    def set_init(x):
        dx = mjx.make_data(mx)
        qpos = dx.qpos.at[:].set(x[:mx.nq])
        qvel = dx.qvel.at[:].set(x[mx.nq:])
        dx = dx.replace(qpos=qpos, qvel=qvel)
        return mjx.step(mx, dx)

    def step(carry, _):
        dx = carry
        x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        t = jnp.expand_dims(dx.time, axis=0)
        u = net(x,t) # Policy function
        # u = -1.*dx.qpos - 0.05*dx.qvel
        cost = cost_fn(x, u, ctx) # Cost function

        ctrl = dx.ctrl.at[:].set(u)
        dx = dx.replace(ctrl=ctrl)
        dx = mjx.step(mx, dx) # Dynamics function

        return dx, jnp.concatenate([dx.qpos, dx.qvel, dx.ctrl, cost], axis=0)

    dx = set_init(x_inits)
    _, res = jax.lax.scan(step, dx, None, length=ctx.cfg.nsteps)
    x, u, costs = res[...,:-mx.nu-1], res[...,-mx.nu:-1], res[...,:-1]
    u = u.at[-1].set(0.)
    costs = jnp.sum(costs) + ctx.cbs.terminal_cost(x[-1]) # Sum and add terminal cost
    return x, u
