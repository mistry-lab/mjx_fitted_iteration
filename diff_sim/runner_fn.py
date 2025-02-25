import time
import wandb
import mujoco
from mujoco import viewer
import jax
import jax.numpy as jnp
import contextlib
import equinox as eqx

from diff_sim.simulation.simulate import make_simulate_fn_simple
from diff_sim.utils.tqdm import trange
from diff_sim.train_step import step_single_gpu, step_multi_gpu
from diff_sim.utils.mj_data_manager import create_data_manager
from diff_sim.utils.mj_viewers import visualise_policy
from diff_sim.utils.generic_helpers import save_model


class WandBLogger:
    def __init__(self, project="default", anonymous="allow", mode="online"):
        self.project = project
        self.anonymous = anonymous
        self.mode = mode
        self.run = None

    def __enter__(self):
        self.run = wandb.init(
            project=self.project, anonymous=self.anonymous, mode=self.mode
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If something goes wrong or finishes normally, ensure W&B is closed properly
        if self.run is not None:
            wandb.finish()

    def log(self, data: dict):
        """Log a dictionary of metrics to WandB."""
        wandb.log(data)


def run(
    ctx, optimiser, simulate_fn, loss_fn, headless=False, wb_project="default", gpu_id=0
):
    """
    Runs a training loop using JAX for multi-GPU or single-GPU stepping,
    logs metrics to W&B, and optionally visualizes the policy in MuJoCo.
    """
    try:
        # Initial random keys
        key_main = jax.random.PRNGKey(ctx.seed)

        # Create MuJoCo model (CPU side) and optional viewer
        model = ctx.gen_model()
        data = mujoco.MjData(model)
        viewer_context = (
            contextlib.nullcontext() if headless else viewer.launch_passive(model, data)
        )

        # Choose which device to run on (GPU or CPU fallback), init wandb logger
        with WandBLogger(project=wb_project) as logger, jax.default_device(
            jax.devices()[gpu_id]
        ), viewer_context as view:
            # Create network and optimizer state
            net = ctx.gen_network(ctx.seed)
            simulate_fn_visu = eqx.filter_jit(make_simulate_fn_simple(ctx))
            visualise_policy(data, model, view, ctx, net, key_main, simulate_fn_visu)
            opt_state = optimiser.init(eqx.filter(net, eqx.is_array))

            # Determine single-GPU or multi-GPU stepping
            step_fn = step_multi_gpu if ctx.num_gpu > 1 else step_single_gpu

            # Initial dataset
            key_main, key_data, key_sim = jax.random.split(key_main, num=3)
            data_manager = create_data_manager()
            dxs = data_manager.create_data(ctx, jax.random.PRNGKey(ctx.seed))

            # Stats tracking
            stats = {"loss": 0.0, "cost": 0.0, "reset": 0.0}
            # How often to log stats
            log_interval = max(1, ctx.ntotal // ctx.nsteps)

            # Helper function to handle logging
            # def log_stats_and_reset(iteration, key_log):
            #     """Logs training stats, evaluates policy, then resets stats."""
            #     key_data_log, key_sim_log = jax.random.split(key_log,num=2)
            #     # Evaluate the policy on a small validation batch
            #     dxs_log = data_manager.create_data(ctx, key_data_log, custom_batch=2)

            #     # Example: simulate_fn returns (.., costs, ..) in 4th position
            #     # Adjust to match your actual signature
            #     _, _, _, costs, _, _ = simulate_fn(dxs_log, key_sim_log, net)

            #     # Construct log data
            #     log_data = {
            #         "Iteration": iteration,
            #         "Loss avg": round(stats["loss"] / log_interval, 3),
            #         "Traj Cost avg": float(jnp.mean(jnp.sum(costs, axis=-1))),
            #         "nreset avg": stats["reset"],  # could also be an average if desired
            #     }
            #     wandb.log(log_data)
            #     es.set_postfix(log_data)

            #     # Reset stats
            #     stats["loss"] = 0.0
            #     stats["cost"] = 0.0
            #     stats["reset"] = 0.0

            # Main training loop
            for e in (es := trange(ctx.epochs)):
                # Generate random keys for this epoch
                key_main, key_sim, key_data, key_vis, key_log = jax.random.split(
                    key_main, num=5
                )

                # One training step
                t0 = time.perf_counter_ns()
                net, opt_state, loss_value, res = step_fn(
                    dxs, net, ctx, key_sim, opt_state, optimiser, simulate_fn, loss_fn
                )
                t1 = time.perf_counter_ns()
                _, dxs, terminated, _ = res

                # (Re)Create data for the next iteration if needed
                key_main, key_data = jax.random.split(key_main)
                dxs = data_manager.reset_data(dxs, ctx, key_data, terminated=terminated)

                # Log step metrics
                logger.log({"loss": float(loss_value), "time_ms": (t1 - t0) * 1e-6})
                es.set_postfix({"loss":float(loss_value)})
                # Accumulate stats
                # stats["loss"] += float(loss_value)
                # stats["cost"] += float(traj_cost)
                # stats["reset"] += float(jnp.sum(terminated))

                # Check for evaluation/visualization
                if (e + 1) % ctx.eval == 0 or e == ctx.epochs - 1:
                    if not headless:
                        visualise_policy(
                            data, model, view, ctx, net, key_vis, simulate_fn_visu
                        )
                    # Save model checkpoint
                    # task_name = getattr(ctx, "task", "model")
                    # checkpoint_name = f"{task_name}_checkpoint_{e}"
                    # save_model(net, checkpoint_name)
                    # logger.log({"latest_model": checkpoint_name})

    except KeyboardInterrupt:
        print("Exiting due to user interrupt...")
        # WandB is finished via the context manager __exit__
