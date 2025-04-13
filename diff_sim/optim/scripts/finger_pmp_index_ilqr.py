import os
import jax
import jax.numpy as jnp
# TODO: we internally handle this in make data but if they don't have our fix we need to upscale
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
import mujoco
from mujoco import mjx
from diff_sim.utils.mj_viewers import visualise_traj_generic
from diff_sim.optim.meta_context import Context
from diff_sim.optim.simulation.step import make_step_fn, make_step_fn_fd
from diff_sim.optim.ilqr import ILQR, make_ilqr_step, simulate_trajectory_ilqr
from diff_sim.utils.math_helper import angle_axis_to_quaternion, quaternion_to_angle_axis
# Rotation matrix cost
# from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat

# Compilation option
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
#     "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
# )


model_path = os.path.join(os.path.dirname(__file__), "../xmls/finger_mjx.xml")
def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)

# TODO : change xdes name --> x_goal ?
# Make random init for a certain batch size, for now tested with 4
# include vel in the init

if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):

        def running_cost(dx):
            pos_finger = dx.qpos[2]
            quat_goal = dx.mocap_quat[0]

            # Quaternion difference cost, failure
            # quat_spinner = angle_axis_to_quaternion(jnp.array([0.,-pos_finger,0.]))
            # c_ang = 0.002 * jnp.sum(quaternion_difference(quat_spinner,quat_goal)**2)
            # Rotation matrix cost, working
            # quat_spinner = axis_angle_to_quat(jnp.array([0.,1.,0.]), jnp.array([-pos_finger]))
            # costR = 0.002 * jnp.sum((quat_to_mat(quat_spinner)  - quat_to_mat(quat_goal))**2)
            # Angular position cost
            pos_ref = - quaternion_to_angle_axis(quat_goal)[1] # Minus sign between visualisation and mocap ?
            c_ang = 0.002*(pos_ref - pos_finger)**2

            # u = dx.ctrl # Torque control
            u = dx.ctrl - dx.qpos[:2]  # Position control
            c_vel = 0.0001 * jnp.sum(dx.qvel**2)
            return 0.002 * jnp.sum(u**2) + c_ang  + c_vel

        def terminal_cost(dx):
            pos_finger = dx.qpos[2]
            quat_goal = dx.mocap_quat[0]

            # Quaternion difference cost, failure
            # quat_spinner = angle_axis_to_quaternion(jnp.array([0.,-pos_finger,0.]))
            # c_ang = 4. * jnp.sum(quaternion_difference(quat_spinner,quat_goal)**2)
            # Rotation matrix cost, working
            # quat_spinner = axis_angle_to_quat(jnp.array([0.,1.,0.]), jnp.array([-pos_finger]))
            # c_ang = 4. * jnp.sum((quat_to_mat(quat_spinner)  - quat_to_mat(quat_goal))**2)
            # Angular position costs
            pos_ref = - quaternion_to_angle_axis(quat_goal)[1]
            c_ang = 4.*(pos_ref - pos_finger)**2

            return c_ang

        def set_control(dx, u):
            return dx.replace(ctrl=dx.ctrl.at[:].set(u))

        def set_target(dx, xdes):
            return dx.replace(
                # mocap_pos=dx.mocap_pos.at[:].set(xdes[:3]), 
                mocap_quat=dx.mocap_quat.at[:].set(xdes[3:])
                )

        # 1) General context for the optimisation
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
            reg = 1e-6,
            ddp=False
        )

        model = ctx.gen_model()
        d = mujoco.MjData(model)
        # qpos_init = jnp.array([-0.8, 0, -3.1])

        qpos_init =  jnp.array([[-0.7,  0. , -0.8],
                                [-0.8,  0. , -0.7],
                                [-0.9,  0. , -0.55],
                                [-1.0,  0. , -0.8]])

        def get_xdes(y_angle):
            pos = jnp.array([-.2, 0, -.35])
            return jnp.concatenate([pos, angle_axis_to_quaternion(jnp.array([0., y_angle, 0.]))])
        
        xdes = jax.vmap(get_xdes)(jnp.array([0.5,-0.4,0.3,-2.6]))

        Nsteps, nu = 300, 2

        # 2) Select a step function (Implicit, FD or AD)
        # step_fn = make_step_fn(ctx)
        step_fn = make_step_fn_fd(ctx)

        # 4.3: Create the batch module
        ilqr_step = make_ilqr_step(
                step_fn=step_fn,
                jac_fn=jax.jacfwd,
                ctx=ctx
        )

        # init_controls = 0.1 * jax.random.normal(key, (B, T, nu))
        # U0 = jax.random.normal(jax.random.PRNGKey(0), (ctx.nsteps, nu)) * 10
        U0 = 10. * jax.random.normal(jax.random.PRNGKey(ctx.seed), (ctx.batch, ctx.nsteps, ctx.ctrl_dim))

        ilqr = ILQR(ilqr_step)
        U_opt, cost = ilqr.solve(X0 = qpos_init, U0= U0, xdes=xdes)

        from diff_sim.utils.mj_viewers import visualise_traj_generic

        d = mujoco.MjData(model)
        x, _, _ = jax.vmap(simulate_trajectory_ilqr, in_axes=(0,0,0,None,None))(qpos_init, U_opt, xdes, step_fn, ctx)
        visualise_traj_generic(x, d, model, mocap_targets=xdes)
