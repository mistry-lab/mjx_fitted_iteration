import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import equinox
from diff_sim.traj_opt.pmp import PMP, make_loss


if __name__ == "__main__":
    MODEL_XML = """
    <mujoco model="example">
        <option timestep="0.01" />
        <worldbody>
            <body name="object" pos="0 0 0">
                <joint type="free"/>
                <geom type="sphere" size="0.05"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    qpos_init = jnp.array([-0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    Nsteps, nu = 100, 1
    # make random U0 normally distributed
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu))

    def set_control(dx, u):
        return dx.replace(qfrc_applied=jnp.zeros_like(dx.qfrc_applied).at[0].set(u[0]))

    def running_cost(dx):
        pos_error = dx.qpos[0] - 0.0
        u = dx.qfrc_applied[0]
        return pos_error ** 2 + 1e-3 * u ** 2

    def terminal_cost(dx):
        return 10.0 * (dx.qpos[0] - 0.5) ** 2

    # Create the loss and grad_loss functions
    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    # Create the optimizer
    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.1, max_iter=100)

    from diff_sim.utils.mj import visualise_traj_generic
    from diff_sim.traj_opt.pmp import simulate_trajectory
    import mujoco
    from mujoco import viewer

    d = mujoco.MjData(model)
    x, cost = simulate_trajectory(mx, qpos_init, set_control, running_cost, terminal_cost, optimal_U)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model, viewer.launch_passive(model, d))
