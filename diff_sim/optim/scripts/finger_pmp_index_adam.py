import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
import mujoco
from mujoco import mjx
from diff_sim.utils.mj_viewers import visualise_traj_generic
from diff_sim.optim.meta_context import Context
from diff_sim.optim.simulation.step import make_step_fn
from diff_sim.utils.math_helper import angle_axis_to_quaternion, quaternion_to_angle_axis
from diff_sim.optim.pmp_fd_indexes_adam import make_pmp_step, PMP

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# ---------------------------------------------------------------------
# Setup Model
# ---------------------------------------------------------------------
model_path = os.path.join(os.path.dirname(__file__), "../xmls/finger_mjx.xml")

def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):

        # Define cost functions
        def running_cost(dx):
            pos_finger = dx.qpos[2]
            quat_goal = dx.mocap_quat[0]
            pos_ref = - quaternion_to_angle_axis(quat_goal)[1]
            c_ang = 0.002 * (pos_ref - pos_finger)**2
            u = dx.ctrl - dx.qpos[:2]
            c_vel = 0.0001 * jnp.sum(dx.qvel**2)
            return 0.002 * jnp.sum(u**2) + c_ang + c_vel

        def terminal_cost(dx):
            pos_finger = dx.qpos[2]
            quat_goal = dx.mocap_quat[0]
            pos_ref = - quaternion_to_angle_axis(quat_goal)[1]
            c_ang = 4.0 * (pos_ref - pos_finger)**2
            return c_ang

        def set_control(dx, u):
            return dx.replace(ctrl=dx.ctrl.at[:].set(u))

        def set_target(dx, xdes):
            return dx.replace(
                mocap_quat=dx.mocap_quat.at[:].set(xdes[3:])
            )

        # Create context
        ctx = Context(
            lr=0.01,
            nsteps=300,
            epochs=30,
            batch=4,
            mx=mjx.put_model(gen_model()),
            gen_model=gen_model,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
            set_control=set_control,
            set_target=set_target,
            ctrl_dim=2,
            target_fields={"qpos", "qvel", "ctrl"},
            eps=1e-6,
            reg=1e-6,
            ddp=False
        )

        # Example initial states (batch=4)
        qpos_init = jnp.array([
            [-0.7,  0., -0.8],
            [-0.8,  0., -0.7],
            [-0.9,  0., -0.55],
            [-1.0,  0., -0.8]
        ])

        # Example desired states (mocap quat)
        # We'll define xdes as [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        # Here we fix pos ~ [-.2, 0, -0.35] and vary angle about Y
        def get_xdes(y_angle):
            pos = jnp.array([-0.2, 0., -0.35])
            quat = angle_axis_to_quaternion(jnp.array([0., y_angle, 0.]))
            return jnp.concatenate([pos, quat])

        xdes = jax.vmap(get_xdes)(jnp.array([0.5, -0.4, 0.3, -2.6]))

        # Create a step function: FD or implicit
        step_fn = make_step_fn(ctx)  # example with finite-difference

        # Initial guess for controls (batch, T, nu) = (4, 300, 2)
        key = jax.random.PRNGKey(0)
        U0 = 0.01 * jax.random.normal(key, (ctx.batch, ctx.nsteps, ctx.ctrl_dim))

        pmp_step = make_pmp_step(step_fn, ctx)
        pmp_solver = PMP(pmp_step=pmp_step)

        U_opt, final_cost = pmp_solver.solve(
            X0=qpos_init,
            U0=U0,
            xdes=xdes,
            lr=0.1
        )

        # If you want to visualize a single trajectory from the batch:
        model = ctx.gen_model()
        d = mujoco.MjData(model)

        from diff_sim.optim.ilqr import simulate_trajectory_ilqr

        d = mujoco.MjData(model)
        x, _, _ = jax.vmap(simulate_trajectory_ilqr, in_axes=(0, 0, 0, None, None))(
            qpos_init, U_opt, xdes, step_fn, ctx
        )
        visualise_traj_generic(x, d, model, mocap_targets=xdes)