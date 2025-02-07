import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.pmp_fd_indexes import PMP, make_loss_fn, build_fd_cache
from diff_sim.utils.mj import visualise_traj_generic


def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("../../xmls/finger_mjx.xml")
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    d = mujoco.MjData(model)
    qpos_init = jnp.array([-.8, 0, -.8])
    d.qpos = np.array(qpos_init)
    Nsteps, nu = 300, 2
    #make random control sequence
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2 * 0.2
    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 4 * pos_finger ** 2

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    fd_cache = build_fd_cache(
        mx, dx, ('qpos', 'qvel', 'ctrl'), mx.nu
    )

    # 3) Build the loss function with the new step fn
    loss_fn = make_loss_fn(
        mx=mx,
        qpos_init=qpos_init,
        set_ctrl_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        fd_cache=fd_cache,
    )


    l, x = loss_fn(U0)
    # 4) Optimize
    pmp = PMP(loss=lambda u: loss_fn(u)[0])
    optimal_U = pmp.solve(U0=U0, learning_rate=0.5, max_iter=50)

    l, x = loss_fn(optimal_U)

    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
