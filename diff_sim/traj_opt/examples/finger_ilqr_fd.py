import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx

# Import the FD-based iLQR code from above (or place it inline)
from diff_sim.traj_opt.ilqr_fd import (
    ILQR,
    make_ilqr_step_fd,
    simulate_trajectory_ilqr_fd
)

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":

    import mujoco
    from mujoco import mjx

    # Load model
    model = mujoco.MjModel.from_xml_path("../../xmls/finger_mjx.xml")
    mx = mjx.put_model(model)

    # Create one-time data template
    dx_template = mjx.make_data(mx)
    # Possibly promote to double precision
    dx_template = jax.tree_map(lambda x: x.astype(jnp.float64) if hasattr(x, 'dtype') else x, dx_template)

    # Define initial qpos, horizon, control dimension
    qpos_init = jnp.array([-0.5, 0.0, -1])
    Nsteps, nu = 300, 2
    U0 = 0.2 * jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2


    # Same user-defined cost
    def running_cost_fn(dx):
        pos_finger = dx.qpos[2]
        ctrl = dx.ctrl
        return 0.002 * jnp.sum(ctrl ** 2) + 0.001 * (pos_finger ** 2)


    def terminal_cost_fn(dx):
        pos_finger = dx.qpos[2]
        return 4.0 * (pos_finger ** 2)


    # Same user-defined control setter
    def set_control_fn(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))


    # Create the iLQR-step function
    ilqr_step = make_ilqr_step_fd(
        dx_template=dx_template,
        mx=mx,
        qpos_init=qpos_init,
        set_control_fn=set_control_fn,
        running_cost_fn=running_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        alpha=0.05,
        reg=1e-6
    )

    # Build the solver
    solver = ILQR(ilqr_step=ilqr_step)

    # Solve
    U_opt, final_cost = solver.solve(U0, tol=1e-5, max_iter=50)
    print("Optimal cost:", final_cost)

    from diff_sim.utils.mj import visualise_traj_generic
    import mujoco

    d = mujoco.MjData(model)
    x, cost = simulate_trajectory_ilqr_fd(dx_template, mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U_opt)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)