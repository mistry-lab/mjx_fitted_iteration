import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
import equinox
from diff_sim.traj_opt.pmp import PMP, make_loss

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
    qpos_init = jnp.array([.1, 0, -.8])
    Nsteps, nu = 300, 2
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

    def set_control(dx, u):
        # u = jnp.tanh(u) * 0.5
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 + 0.01 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 1 * pos_finger ** 2

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.2, max_iter=50)

    from diff_sim.utils.mj_viewers import visualise_traj_generic
    from diff_sim.traj_opt.pmp import simulate_trajectory
    import mujoco

    d = mujoco.MjData(model)
    x, cost = simulate_trajectory(mx, qpos_init, set_control, running_cost, terminal_cost, optimal_U)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
