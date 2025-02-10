import argparse
import time

import wandb
import mujoco
from mujoco import viewer
import equinox as eqx
import optax
import jax
from jax import config
config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import contextlib  # Added for handling headless mode
from diff_sim.context.tasks import ctxs
from diff_sim.utils.tqdm import trange
from diff_sim.utils.mj_viewers import visualise_policy, visualise_traj
from diff_sim.utils.generic_helpers import save_model
from diff_sim.utils.mj_data_manager import create_data_manager
from diff_sim.simulate import controlled_simulate, controlled_simulate_fd


# stop when you hit NaNs in jax
# jax.config.update("jax_debug_nans", True)
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", help="task name", default="finger")
        parser.add_argument("--ad", action="store_false", help="Uses automatic differentiation")
        parser.add_argument("--wb_project", help="wandb project name", default="not_named")
        parser.add_argument("--headless", action="store_true", help="Disable visualization")
        parser.add_argument(
            "--gpu_id", type=int, default=0,
            help="Use jax.devices() and nvidia-smi to select the least busy GPU"
        )
        args = parser.parse_args()
        ctx = ctxs[args.task]

        # Finite difference or automatic differentiation
        simulate = controlled_simulate_fd if args.ad else controlled_simulate

        # Initialize wandb
        wandb.init(anonymous="allow", mode='offline') if args.wb_project is None else (
            wandb.init(project=args.wb_project, anonymous="allow")
        )

        # Initial keys for random number generation; these will be split for each iteration
        init_key = jax.random.PRNGKey(ctx.cfg.seed)
        #array of keys

        key, subkey = jax.random.split(init_key)

        # Model and data for rendering (CPU side)
        model = ctx.cfg.gen_model()
        data = mujoco.MjData(model)
        viewer_context = contextlib.nullcontext() if args.headless else viewer.launch_passive(model, data)


        # Start the training loop in JAX default device
        with (jax.default_device(jax.devices()[args.gpu_id])), viewer_context as viewer:
            net, optim = ctx.cbs.gen_network(ctx.cfg.seed), optax.adamw(ctx.cfg.lr)
            opt_state = optim.init(eqx.filter(net, eqx.is_array))
            make_step = net.make_step_multi_gpu if ctx.cfg.num_gpu > 1 else net.make_step

            # Run through the epochs and log the loss
            # make_data that give batch of dx
            key, init_key = jax.random.split(key)
            data_manager = create_data_manager()
            dxs = data_manager.create_data(ctx.cfg.mx, ctx, ctx.cfg.batch*ctx.cfg.samples, init_key)
            sum_loss, sum_cost, sum_reset, iter = 0, 0, 0, ctx.cfg.ntotal//ctx.cfg.nsteps
            # init data
            for e in (es := trange(ctx.cfg.epochs)):
                key, xkey, tkey, user_key = jax.random.split(key, num = 4)
                net, opt_state, loss_value, res = make_step(dxs, optim, net, opt_state, ctx, user_key)
                traj_cost, dxs, terminated, x = res
                # visualise_traj(x, data, model, viewer, ctx)
                # print("Waiting for 10 seconds before resetting data...")
                # time.sleep(0.5)
                dxs = data_manager.reset_data(ctx.cfg.mx, dxs, ctx, tkey, terminated)
                sum_loss += loss_value.item()
                sum_cost += traj_cost.item()
                sum_reset += jnp.sum(terminated).item()

                if e % iter == 0:
                    key, xkey, tkey = jax.random.split(key, num = 3)
                    dx_vis = data_manager.create_data(ctx.cfg.mx, ctx, 2, xkey)
                    _, _, _, costs, _, _ = eqx.filter_jit(simulate)(dx_vis, ctx, net, tkey, ctx.cfg.ntotal)
                    log_data = {
                        "Loss avg": round(sum_loss/iter, 3),
                        "Traj Cost avg": jnp.mean(jnp.sum(costs, axis=-1)),
                        "nreset avg": sum_reset
                    }

                    wandb.log(log_data)
                    es.set_postfix(log_data)
                    sum_loss, sum_cost, sum_reset = 0, 0, 0

                if e % ctx.cfg.eval == 0 or e == ctx.cfg.epochs - 1:
                    # Only visualize policy if not in headless mode
                    if not args.headless:
                        visualise_policy(data, model, viewer, ctx, net, key)
                    name = f"{args.task}_checkpoint_{e}"
                    save_model(net, args.task, name)
                    log_data["latest_model"] = name
                    es.set_postfix(log_data)

    except KeyboardInterrupt:
        print("Exiting wandb...")
        wandb.finish()
