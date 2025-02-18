import time
import wandb
import mujoco
from mujoco import viewer
import jax
import jax.numpy as jnp
import contextlib
import equinox as eqx

from diff_sim.utils.tqdm import trange
from diff_sim.training.train_step import step_single_gpu, step_multi_gpu
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
        self.run = wandb.init(project=self.project, anonymous=self.anonymous, mode=self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If something goes wrong or finishes normally, ensure W&B is closed properly
        if self.run is not None:
            wandb.finish()

    def log(self, data: dict):
        """Log a dictionary of metrics to WandB."""
        wandb.log(data)


def run(ctx, optimiser, simulate_fn, loss_fn, headless=False, wb_project="default", gpu_id=0):
    """
    Runs a training loop using JAX for multi-GPU or single-GPU stepping,
    logs metrics to W&B, and optionally visualizes the policy in MuJoCo.
    """
    try:
        # --- WandB Logging Context ---
        with WandBLogger(project=wb_project) as logger:
            # Initial random keys
            key_main = jax.random.PRNGKey(ctx.seed)

            # Create MuJoCo model (CPU side) and optional viewer
            model = ctx.gen_model()
            data = mujoco.MjData(model)
            viewer_context = contextlib.nullcontext() if headless else viewer.launch_passive(model, data)

            # Choose which device to run on (GPU or CPU fallback)
            with jax.default_device(jax.devices()[gpu_id]), viewer_context as view:
                # Create network and optimizer state
                net = ctx.gen_network(ctx.seed)
                opt_state = optimiser.init(eqx.filter(net, eqx.is_array))

                # (Optional) Visualize the initial policy
                visualise_policy(data, model, view, ctx, net, key_main, simulate_fn)

                # Determine single-GPU or multi-GPU stepping
                step_fn = step_multi_gpu if ctx.num_gpu > 1 else step_single_gpu

                # Create initial dataset
                key_main, key_data = jax.random.split(key_main)
                data_manager = create_data_manager()
                dxs = data_manager.create_data(ctx, key_data)

                # Main training loop
                for epoch in trange(ctx.epochs):
                    # Split keys
                    key_main, key_sim = jax.random.split(key_main)

                    # One training step
                    t0 = time.perf_counter_ns()
                    net, opt_state, loss_value, res = step_fn(
                        dxs, net, ctx, key_sim, opt_state, optimiser, simulate_fn, loss_fn
                    )
                    t1 = time.perf_counter_ns()

                    # Log step metrics
                    logger.log({"loss": float(loss_value), "time_ms": (t1 - t0) * 1e-6})

                    # Unpack results
                    _, dxs, terminated, _ = res
                    # Optionally accumulate stats here if you want a custom aggregator

                    # (Re)Create data for the next iteration if needed
                    key_main, key_data = jax.random.split(key_main)
                    dxs = data_manager.create_data(ctx, key_data)

                    # Periodic evaluation/visualization
                    if (epoch + 1) % ctx.eval == 0 or epoch == ctx.epochs - 1:
                        if not headless:
                            key_main, key_vis = jax.random.split(key_main)
                            visualise_policy(data, model, view, ctx, net, key_vis, simulate_fn)

                        # Optionally save or log model checkpoint
                        # checkpoint_name = f"checkpoint_{epoch}"
                        # save_model(net, checkpoint_name)
                        # logger.log({"latest_model": checkpoint_name})

    except KeyboardInterrupt:
        print("Exiting due to user interrupt...")
        # WandB is finished via the context manager __exit__
