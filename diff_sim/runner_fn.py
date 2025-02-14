import wandb
import mujoco
from mujoco import viewer
import jax
import jax.numpy as jnp
import contextlib
import equinox as eqx

from diff_sim.utils.tqdm import trange
from diff_sim.training.train_step import step_single_gpu, step_multi_gpu
from diff_sim.utils.mj_data_manager import create_data_manager, create
from diff_sim.utils.mj_viewers import visualise_policy
from diff_sim.utils.generic_helpers import save_model

def run(ctx, optimiser, simulate_fn, loss_fn, headless=False, wb_project="default", gpu_id=0):
    """
    Runs a training loop using JAX for multi-GPU or single-GPU stepping,
    logs metrics to W&B, and optionally visualizes policy in MuJoCo.
    """
    try:
        # Initialize wandb
        wandb.init(anonymous="allow", mode='offline', project=wb_project)

        # Initial random keys
        key, subkey = jax.random.split(jax.random.PRNGKey(ctx.seed))

        # Create MuJoCo model (CPU side) and optional viewer
        model = ctx.gen_model()
        data = mujoco.MjData(model)
        viewer_context = contextlib.nullcontext() if headless else viewer.launch_passive(model, data)

        # Choose which device to run on (GPU or CPU fallback)
        with jax.default_device(jax.devices()[gpu_id]), viewer_context as view:
            # Create network and optimizer state
            net = ctx.gen_network(ctx.seed)
            opt_state = optimiser.init(eqx.filter(net, eqx.is_array))

            # Single-GPU or multi-GPU step function
            step_fn = step_multi_gpu if ctx.num_gpu > 1 else step_single_gpu

            # Prepare data manager and initial dataset
            key, init_key = jax.random.split(key)
            data_manager = create_data_manager()
            dxs = data_manager.create_data(ctx.mx, ctx, ctx.batch * ctx.samples, init_key)

            # Stats tracking
            stats = {"loss": 0.0, "cost": 0.0, "reset": 0.0}
            # How often to log stats
            log_interval = max(1, ctx.ntotal // ctx.nsteps)

            # Helper function to handle logging
            def log_stats_and_reset(iteration):
                """Logs training stats, evaluates policy, then resets stats."""
                # Evaluate the policy on a small validation batch
                dx_vis = data_manager.create_data(ctx.mx, ctx, 2, xkey)
                user_keys_eval = jax.random.split(tkey, num=2)

                # Example: simulate_fn returns (.., costs, ..) in 4th position
                # Adjust to match your actual signature
                _, _, _, costs, _, _ = simulate_fn(dx_vis, user_keys_eval, model)

                # Construct log data
                log_data = {
                    "Iteration": iteration,
                    "Loss avg": round(stats["loss"] / log_interval, 3),
                    "Traj Cost avg": float(jnp.mean(jnp.sum(costs, axis=-1))),
                    "nreset avg": stats["reset"],  # could also be an average if desired
                }
                wandb.log(log_data)
                es.set_postfix(log_data)

                # Reset stats
                stats["loss"] = 0.0
                stats["cost"] = 0.0
                stats["reset"] = 0.0

            # Main training loop
            for e in (es := trange(ctx.epochs)):
                # Generate random keys for this epoch
                key, xkey, tkey, user_key = jax.random.split(key, num=4)
                user_keys = jax.random.split(user_key, num=ctx.batch)

                # One training step
                net, opt_state, loss_value, res = step_fn(
                    dxs, optimiser, net, opt_state, ctx, user_keys, simulate_fn, loss_fn
                )
                traj_cost, dxs, terminated, _ = res

                # Accumulate stats
                stats["loss"] += float(loss_value)
                stats["cost"] += float(traj_cost)
                stats["reset"] += float(jnp.sum(terminated))

                # Reset data for next iteration if needed
                dxs = data_manager.reset_data(ctx.mx, dxs, ctx, tkey, terminated)

                # Periodically log statistics
                if (e + 1) % log_interval == 0:
                    log_stats_and_reset(e + 1)

                # Check for evaluation/visualization
                if (e + 1) % ctx.eval == 0 or e == ctx.epochs - 1:
                    if not headless:
                        visualise_policy(data, model, view, ctx, net, key)
                    # Save model checkpoint
                    task_name = getattr(ctx, "task", "model")
                    checkpoint_name = f"{task_name}_checkpoint_{e}"
                    save_model(net, checkpoint_name)

                    # Log the checkpoint name
                    wandb.log({"latest_model": checkpoint_name})
                    es.set_postfix({"latest_model": checkpoint_name})

    except KeyboardInterrupt:
        print("Exiting wandb...")
        wandb.finish()
