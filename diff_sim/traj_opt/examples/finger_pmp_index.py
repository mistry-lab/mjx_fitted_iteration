import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.pmp_fd_indexes import PMP, make_loss_fn, make_step_fn, prepare_sensitivity

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
    qpos_init = jnp.array([-.8, 0, -.8])
    Nsteps, nu = 300, 2
    U0 = 0.2*jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 4 * pos_finger ** 2

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    # 2) Precompute flatten/unflatten + sensitivity info
    fd_cache = build_fd_cache(
        mx,
        dx_ref,
        target_fields={"qpos", "qvel", "ctrl"},
        ctrl_dim=1,
        eps=1e-6
    )

    # 3) Build the loss function with the new step fn
    loss_fn = make_loss_fn(
        mx=mx,
        qpos_init=qpos_init,
        set_ctrl_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        unravel_dx=unravel_dx,
        inner_idx=inner_idx,
        sensitivity_mask=sensitivity_mask,
        eps=1e-6
    )

    # 4) Optimize
    pmp = PMP(loss=loss_fn)
    optimal_U = pmp.solve(U0=jnp.zeros((Nsteps, nu)), learning_rate=0.5, max_iter=50)

    from diff_sim.utils.mj import visualise_traj_generic
    from diff_sim.traj_opt.pmp_fd import simulate_trajectory
    import mujoco

    d = mujoco.MjData(model)
    step_func = make_step_fn(mx, set_control, unravel_dx, inner_idx, sensitivity_mask)
    x, cost = simulate_trajectory(mx, qpos_init, step_func, running_cost, terminal_cost, optimal_U)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
