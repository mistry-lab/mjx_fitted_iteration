import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.pmp_fd import PMP, make_loss_fn, make_step_fn

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


if __name__ == "__main__":
    path = "../../xmls/cartpole.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    qpos_init = jnp.array([0.0, 3.14])
    Nsteps, nu = 300, 1
    U0 = 0.2*jax.random.normal(jax.random.PRNGKey(5), (Nsteps, nu))
    # U0 = 0.1 * jnp.ones((Nsteps, nu))


    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        u = dx.ctrl
        return 1e-3 * jnp.sum(u ** 2) + 0.001 * jnp.sum(dx.qpos ** 2)
    
    def terminal_cost(dx):
        pos_pole = dx.qpos
        return 0.1 * jnp.sum(pos_pole ** 2)

    loss_fn = make_loss_fn(mx, qpos_init, set_control, running_cost, terminal_cost)
    optimizer = PMP(loss=loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.5, max_iter=100)

    from diff_sim.utils.mj import visualise_traj_generic
    from diff_sim.traj_opt.pmp_fd import simulate_trajectory
    import mujoco

    d = mujoco.MjData(model)
    step_func = make_step_fn(mx, set_control)
    x, cost = simulate_trajectory(mx, qpos_init, step_func, running_cost, terminal_cost, optimal_U)


    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
