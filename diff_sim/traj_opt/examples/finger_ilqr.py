import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.ilqr import make_ilqr_step, simulate_trajectory_ilqr, ILQR


def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
    path = "../../xmls/finger_mjx.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    qpos_init = jnp.array([1, 0, -.8])
    Nsteps, nu = 300, 1
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

    def set_control(dx, u):
        # u = jnp.tanh(u) * 0.5
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        pos_finger = (dx.qpos[2] - jnp.pi * 2)
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 + jnp.sum(u ** 2)

    def terminal_cost(dx):
        pos_finger = (dx.qpos[2] - jnp.pi * 2)
        return 1 * pos_finger ** 2

    ilqr_step = make_ilqr_step(
        mx=mx,
        qpos_init=qpos_init,
        set_control_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        alpha=0.1,
        reg=1e-6
    )

    optimizer = ILQR(ilqr_step=ilqr_step)
    U_opt, cost = optimizer.solve(U0, max_iter=50)

    from diff_sim.utils.mj_viewers import visualise_traj_generic

    d = mujoco.MjData(model)
    x, _, _ = simulate_trajectory_ilqr(mx, qpos_init, set_control, running_cost, terminal_cost, U_opt)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
