import argparse
import wandb
import mujoco
from mujoco import viewer
import equinox as eqx
import optax
from jax import config
import jax
import contextlib  # Added for handling headless mode
from diff_sim.context.tasks import ctxs
from diff_sim.utils.tqdm import trange
from diff_sim.utils.mj import visualise_policy
from diff_sim.utils.generic import save_model

config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", help="task name", default="double_integrator")
        parser.add_argument("--wb_project", help="wandb project name", default="not_named")
        parser.add_argument("--headless", action="store_true", help="Disable visualization")
        parser.add_argument(
            "--gpu_id", type=int, default=0,
            help="Use jax.devices() and nvidia-smi to select the least busy GPU"
        )
        args = parser.parse_args()
        ctx = ctxs[args.task]

        # Initialize wandb
        wandb.init(anonymous="allow", mode='offline') if args.wb_project is None else (
            wandb.init(project=args.wb_project, anonymous="allow")
        )

        # Initial keys for random number generation; these will be split for each iteration
        init_key = jax.random.PRNGKey(ctx.cfg.seed)
        key, subkey = jax.random.split(init_key)

        # Model and data for rendering (CPU side)
        model = mujoco.MjModel.from_xml_path(ctx.cfg.path)
        data = mujoco.MjData(model)
        viewer_context = contextlib.nullcontext() if args.headless else viewer.launch_passive(model, data)


        # Start the training loop in JAX default device
        with (jax.default_device(jax.devices()[args.gpu_id])), viewer_context as viewer:
            net, optim = ctx.cbs.gen_network(ctx.cfg.seed), optax.adamw(ctx.cfg.lr)
            opt_state = optim.init(eqx.filter(net, eqx.is_array))
            make_step = net.make_step_multi_gpu if ctx.cfg.num_gpu > 1 else net.make_step

            # Run through the epochs and log the loss
            for e in (es := trange(ctx.cfg.epochs)):
                key, xkey, tkey, user_key = jax.random.split(key, num = 4)
                x_inits = ctx.cbs.init_gen(ctx.cfg.batch * ctx.cfg.samples, xkey)
                net, opt_state, loss_value, traj_cost = net.make_step(optim, net, opt_state, x_inits, ctx, user_key)
                log_data = {"loss": round(loss_value.item(), 3), "Traj Cost": round(traj_cost.item(), 3)}
                wandb.log(log_data)
                es.set_postfix(log_data)

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
