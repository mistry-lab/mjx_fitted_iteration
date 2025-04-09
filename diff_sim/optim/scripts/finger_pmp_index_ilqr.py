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
            lr=0.01,
            nsteps=300,
            epochs=30,
            batch=4,
            mx=mjx.put_model(gen_model()),
            gen_model=gen_model,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
            set_control=set_control,
            ctrl_dim=2,
            target_fields={"qpos", "qvel", "ctrl"},
            eps=1e-6,
            reg = 1e-6,
            ddp=False
        )

        model = ctx.gen_model()
        d = mujoco.MjData(model)
        qpos_init = jnp.array([-0.8, 0, -3.1])
        Nsteps, nu = 300, 2

        # 2) Select a step function (Implicit, FD or AD)
        # step_fn = make_step_fn(ctx) # Implicit
        step_fn = make_step_fn_fd(ctx)# FD, TODO: does not work due to custom_vjp
        # TODO : AD

        # 4.3: Create the batch module
        ilqr_step = make_ilqr_step(
                qpos_init=qpos_init,
                step_fn=step_fn,
                jac_fn=jax.jacfwd,
                ctx=ctx
        )
        
        # init_controls = 0.1 * jax.random.normal(key, (B, T, nu))
        U0 = jax.random.normal(jax.random.PRNGKey(0), (ctx.nsteps, nu)) * 10

        ilqr = ILQR(ilqr_step)
        U_opt, cost = ilqr.solve(U0= U0)
        
        from diff_sim.utils.mj_viewers import visualise_traj_generic

        d = mujoco.MjData(model)
        x, _, _ = simulate_trajectory_ilqr(qpos_init, U_opt, step_fn, ctx)
        visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)