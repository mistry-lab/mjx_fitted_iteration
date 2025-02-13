import wandb
import mujoco
from mujoco import viewer
import jax.numpy as jnp
import jax
import contextlib
import equinox as eqx
from diff_sim.utils.mj_data_manager import create_data_manager
from diff_sim.utils.tqdm import trange
from diff_sim.nn.base_nn import step_single_gpu, step_multi_gpu

# from diff_sim.context.tasks import ctxs
# from diff_sim.utils.tqdm import trangee
# from diff_sim.utils.mj_viewers import visualise_policy, visualise_traj
# from diff_sim.utils.generic_helpers import save_model
# from diff_sim.utils.mj_data_manager import create_data_manager
# from diff_sim.simulation.simulate import controlled_simulate, controlled_simulate_fd

def run(ctx, optimiser, simulate_fn, headless=False, wb_project="default", gpu_id=0):
    try:
        # Initialize wandb
        wandb.init(anonymous="allow", mode='offline', project=wb_project)

        # Initial keys for random number generation
        key, subkey = jax.random.split(jax.random.PRNGKey(ctx.seed))

        # Model and data for rendering (CPU side)
        model = ctx.gen_model()
        data = mujoco.MjData(model)
        viewer_context = contextlib.nullcontext() if headless else viewer.launch_passive(model, data)

        # Notes:
        # TODO: Remove step from base_nn (rename base_nn as well, put a BaseNetwork inside as well)
        # TODO: Pass only ctx in data manager.

        # Start the training loop in JAX default device
        with (jax.default_device(jax.devices()[gpu_id])), viewer_context as view:
            net = ctx.gen_network(ctx.seed)
            opt_state = optimiser.init(eqx.filter(net, eqx.is_array))
            step = step_multi_gpu if ctx.num_gpu > 1 else step_single_gpu

            key, init_key = jax.random.split(key)
            # data_manager = create_data_manager()
            # dxs = data_manager.create_data(ctx.mx, ctx, ctx.batch*ctx.samples, init_key)
            # dxs = jax.vmap(lambda x: mjx.make_data(mx), in_axes=(0,))(jnp.arange(N))
            sum_loss, sum_cost, sum_reset, iter = 0, 0, 0, ctx.ntotal//ctx.nsteps
            # init data
            for e in (es := trange(ctx.epochs)):
                key, xkey, tkey, user_key = jax.random.split(key, num = 4)

                user_keys = jax.random.split(user_key,num = ctx.batch)
                net, opt_state, loss_value, res = step(dxs, optimiser, net, opt_state, ctx, user_keys, simulate_fn)
                traj_cost, dxs, terminated, _ = res

                print("ok")
            
                # dxs = data_manager.reset_data(ctx.mx, dxs, ctx, tkey, terminated)
                # sum_loss += loss_value.item()
                # sum_cost += traj_cost.item()
                # sum_reset += jnp.sum(terminated).item()

                # if e % iter == 0:
                #     dx_vis = data_manager.create_data(ctx.mx, ctx, 2, xkey)
                #     user_keys = jax.random.split(tkey,num = 2)
                #     _, _, _, costs, _, _ = simulate_fn(dx_vis, user_keys, model)
                #     log_data = {
                #         "Loss avg": round(sum_loss/iter, 3),
                #         "Traj Cost avg": jnp.mean(jnp.sum(costs, axis=-1)),
                #         "nreset avg": sum_reset
                #     }

                #     wandb.log(log_data)
                #     es.set_postfix(log_data)
                #     sum_loss, sum_cost, sum_reset = 0, 0, 0

                # if e % ctx.eval == 0 or e == ctx.epochs - 1:
                #     # Only visualize policy if not in headless mode
                #     if not headless:
                #         visualise_policy(data, model, viewer, ctx, net, key)
                #     name = f"{args.task}_checkpoint_{e}"
                #     save_model(net, args.task, name)
                #     log_data["latest_model"] = name
                #     es.set_postfix(log_data)

    except KeyboardInterrupt:
        print("Exiting wandb...")
        wandb.finish()