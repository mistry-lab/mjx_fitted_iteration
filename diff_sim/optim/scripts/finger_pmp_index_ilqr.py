import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from diff_sim.optim.ilqr import make_ilqr_step, ILQR, simulate_trajectory_ilqr
from diff_sim.utils.mj_viewers import visualise_traj_generic

# Jax compilation flags
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
    with (jax.default_device(jax.devices("cpu")[0])):
        model = mujoco.MjModel.from_xml_path("xmls/finger_mjx.xml")
        mx = mjx.put_model(model)
        dx = mjx.make_data(mx)
        dx = jax.tree.map(upscale, dx)
        d = mujoco.MjData(model)
        fd_cache = None  # or build_fd_cache(mx, dx, ("qpos","qvel"), 2)
    
        # 4.2: Problem setup
        B = 4        # batch size
        T = 300        # number of steps
        nu = 2         # control dimension
        key = jax.random.PRNGKey(0)
    
        # qpos_init = jnp.tile(jnp.array([-.8, 0, -.8]),B).reshape((B,3))
        # qpos_init =  jnp.array([[-0.7,  0. , -0.8],
        #                         [-0.8,  0. , -0.7],
        #                         [-0.9,  0. , -0.55],
        #                         [-1.0,  0. , -0.8]])
        qpos_init = jnp.array([-0.8, 0, -0.8])
    
        def set_control(dx, u):
            return dx.replace(ctrl=dx.ctrl.at[:].set(u))

        def running_cost(dx):
            pos_finger = dx.qpos[2]
            u = dx.ctrl
            return 0.01 * pos_finger ** 2 +  0.01 * jnp.sum(u ** 2)

        def terminal_cost(dx):
            pos_finger = dx.qpos[2]
            return 1 * pos_finger ** 2
    
        # 4.3: Create the batch module
        ilqr_step = make_ilqr_step(
                mx=mx,
                qpos_init=qpos_init,
                set_control_fn=set_control,
                running_cost_fn=running_cost,
                terminal_cost_fn=terminal_cost,
                alpha=0.1,
                reg=1e-6
            )
        
        # init_controls = 0.1 * jax.random.normal(key, (B, T, nu))
        U0 = jax.random.normal(jax.random.PRNGKey(0), (300, nu)) * 10

        ilqr = ILQR(ilqr_step)
        U_opt, cost = ilqr.solve(U0= U0)
        # 4.4: Train with Adam
        # trained_model = train_batch_trajectories(
        #     model_module=batch_model,
        #     num_steps=30,      # or whichever you like
        #     lr=0.01
        # )
    
        # # Example usage:
        # states_b0 = simulate_trajectory_for_b(trained_model, b_idx=0)
        # # If you have a visualise function:
        # visualise_traj_generic(jnp.expand_dims(states_b0, axis=0), d, model)
        # # etc.
    
        # print("Done training.")
        from diff_sim.utils.mj_viewers import visualise_traj_generic

        d = mujoco.MjData(model)
        x, _, _ = simulate_trajectory_ilqr(mx, qpos_init, set_control, running_cost, terminal_cost, U_opt)
        visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)