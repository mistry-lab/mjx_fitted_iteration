import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
from diff_sim.optim.pmp_fd_indexes import PMP, make_loss_fn, build_fd_cache, make_loss_fn_accfd
from diff_sim.utils.mj_viewers import visualise_traj_generic

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
    model = mujoco.MjModel.from_xml_path("xmls/finger_mjx.xml")
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    d = mujoco.MjData(model)

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

    fd_cache = build_fd_cache(
        mx, dx, ('qpos', 'qvel'), 2
    )

    # 3) Build the loss function with the new step fn
    loss_fn = make_loss_fn_accfd(
        mx=mx,
        qpos_init=qpos_init,
        set_ctrl_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        fd_cache=fd_cache,
    )

    l, x = loss_fn(U0)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
    print("Initial loss: ", l)

    # 4) Optimize
    pmp = PMP(loss=lambda u: loss_fn(u)[0])
    optimal_U = pmp.solve(U0=U0, learning_rate=0.5, max_iter=10)

    l, x = loss_fn(optimal_U)

    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)