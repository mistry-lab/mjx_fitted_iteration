import jax
import jax.numpy as jnp
from mujoco import mjx

def controlled_simulate(x_inits, mx, ctx, v, PD=False):
    def set_init(x):
        dx = mjx.make_data(mx)
        qpos = dx.qpos.at[:].set(x[:mx.nq])
        qvel = dx.qvel.at[:].set(x[mx.nq:])
        dx = dx.replace(qpos=qpos, qvel=qvel)
        return mjx.step(mx, dx)

    def get_ctrl(mx, dx):
        x = jnp.concatenate([dx.qpos, dx.qvel], axis=0)
        t = jnp.expand_dims(dx.time, axis=0)
        act_id = mx.actuator_trnid[:, 0]
        M = mjx.full_m(mx, dx)
        invM = jnp.linalg.inv(M)
        dvdx = jax.jacrev(v,0)(x, t)
        G = jnp.vstack([jnp.zeros_like(invM), invM])
        invR = jnp.linalg.inv(ctx.cfg.R)
        u = (-1/2 * invR @ G.T[act_id, :] @ dvdx.T).squeeze()

        # PD Controller
        if PD:
            u = -1.3*dx.qpos -1.*dx.qvel 

        ctrl = dx.ctrl.at[:].set(u)
        dx = dx.replace(ctrl=ctrl)
        return dx

    def mjx_step(dx, _):
        dx = get_ctrl(mx, dx)
        dx = jax.lax.stop_gradient(mjx.step(mx, dx))
        return dx, jnp.concatenate([dx.qpos, dx.qvel, dx.ctrl], axis=0)

    dx = set_init(x_inits)
    _, batched_traj = jax.lax.scan(mjx_step, dx, None, length=ctx.cfg.nsteps)
    x, u = batched_traj[..., :-mx.nu], batched_traj[..., -mx.nu:]
    u = u.at[-1].set(0.)
    return x, u
