import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import equinox
from diff_sim.traj_opt.pmp import PMP, make_loss


if __name__ == "__main__":
    path = "../../xmls/cartpole.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    qpos_init = jnp.array([0.0, 3.14])
    Nsteps, nu = 300, 1
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu))

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        u = dx.ctrl
        return 1e-3 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        return 1 * jnp.sum(dx.qpos ** 2)

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.2, max_iter=100)

    from diff_sim.utils.mj_viewers import visualise_traj_generic
    from diff_sim.traj_opt.pmp import simulate_trajectory
    import mujoco

    d = mujoco.MjData(model)
    x, cost = simulate_trajectory(mx, qpos_init, set_control, running_cost, terminal_cost, optimal_U)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
