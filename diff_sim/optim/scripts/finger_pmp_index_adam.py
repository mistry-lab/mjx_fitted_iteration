import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
# from diff_sim.optim.pmp_fd_indexes import PMP, make_loss_fn, build_fd_cache, make_loss_fn_accfd
from diff_sim.optim.pmp_fd_indexes_adam import make_batch_loss_module, BatchTrajectory, train_batch_trajectories, simulate_trajectory_for_b
from diff_sim.utils.mj_viewers import visualise_traj_generic

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
# jax.devices("cpu")[0]  # Get the CPU device


def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
 
    # 4.1: Build model, data, fd_cache, etc.
    model = mujoco.MjModel.from_xml_path("xmls/finger_mjx.xml")

    with (jax.default_device(jax.devices("cpu")[0])):
        mx = mjx.put_model(model)
        dx = mjx.make_data(mx)
        # dx = jax.tree_map(upscale, dx)
        d = mujoco.MjData(model)
        fd_cache = None  # or build_fd_cache(mx, dx, ("qpos","qvel"), 2)
    
        # 4.2: Problem setup
        B = 1        # batch size
        T = 300        # number of steps
        nu = 2         # control dimension
        key = jax.random.PRNGKey(0)
    
        qpos_init = jnp.array([-.8, 0, -.8])
    
        def running_cost(dx):
            pos_finger = dx.qpos[2]
            u = dx.ctrl
            return 0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2
    
        def terminal_cost(dx):
            pos_finger = dx.qpos[2]
            return 4 * pos_finger ** 2
    
        def set_control(dx, u):
            return dx.replace(ctrl=dx.ctrl.at[:].set(u))
    
        # 4.3: Create the batch module
        batch_model = make_batch_loss_module(
            mx=mx,
            qpos_init=qpos_init,
            set_ctrl_fn=set_control,
            running_cost_fn=running_cost,
            terminal_cost_fn=terminal_cost,
            fd_cache=fd_cache,
            B=B,
            T=T,
            nu=nu,
            key=key,
        )
    
        # 4.4: Train with Adam
        trained_model = train_batch_trajectories(
            model_module=batch_model,
            num_steps=30,      # or whichever you like
            lr=0.01
        )
    
        # Example usage:
        states_b0 = simulate_trajectory_for_b(trained_model, b_idx=0)
        # If you have a visualise function:
        visualise_traj_generic(jnp.expand_dims(states_b0, axis=0), d, model)
        # etc.
    
        print("Done training.")