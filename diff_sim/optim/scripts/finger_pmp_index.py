import os
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
import mujoco
from mujoco import mjx
from diff_sim.optim.pmp_fd_indexes import PMP, make_loss_fn
from diff_sim.utils.mj_viewers import visualise_traj_generic
from diff_sim.optim.meta_context import Context
from diff_sim.optim.simulation.step import make_step_fn, make_step_fn_fd

# Compilation option
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


model_path = os.path.join(os.path.dirname(__file__), "../xmls/finger_mjx.xml")
def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):

        def running_cost(dx):
            pos_finger = dx.qpos[2]
            u = dx.ctrl
            return 0.002 * jnp.sum(u**2) + 0.001 * pos_finger**2

        def terminal_cost(dx):
            pos_finger = dx.qpos[2]
            return 4 * pos_finger**2

        def set_control(dx, u):
            return dx.replace(ctrl=dx.ctrl.at[:].set(u))

        # 1) General context for the optimisation
        ctx = Context(
            lr=0.5,
            nsteps=300,
            epochs=10,
            mx=mjx.put_model(gen_model()),
            gen_model=gen_model,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
            set_control=set_control,
            ctrl_dim=2,
            target_fields={"qpos", "qvel"},
            eps=1e-6,
        )

        model = ctx.gen_model()
        d = mujoco.MjData(model)
        qpos_init = jnp.array([-0.8, 0, -0.8])
        Nsteps, nu = ctx.nsteps, 2
        U0 = 0.2 * jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

        # 2) Select a step function (Implicit, FD or AD)
        step_fn = make_step_fn(ctx) # Implicit
        # step_fn = make_step_fn_fd(ctx)  # FD
        # TODO : AD

        # 3) Build the loss function with the new step fn
        loss_fn = make_loss_fn(
            qpos_init=qpos_init,
            step_fn=step_fn,
            ctx=ctx,
        )

        # l, x = loss_fn(U0)
        # visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
        # print("Initial loss: ", l)

        # 4) Optimize
        pmp = PMP(loss=lambda u: loss_fn(u)[0])
        optimal_U = pmp.solve(U0=U0, learning_rate=ctx.lr, max_iter=ctx.epochs)

        l, x = loss_fn(optimal_U)

        visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
