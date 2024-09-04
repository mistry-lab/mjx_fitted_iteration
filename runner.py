import argparse
import time
import mujoco
from mujoco import mjx
import jax
import wandb
from config.cps import ctx as cp_ctx
from simulate import controlled_simulate
import equinox as eqx

wandb.init(project="fvi", entity="lonephd")
configs = {"cartpole_swing_up": cp_ctx}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="task name", default="cartpole_swing_up")
    args = parser.parse_args()
    ctx = configs[args.task]
    with jax.default_device(jax.devices()[0]):
        x_inits = ctx.cbs.init_gen(ctx.cfg.batch, jax.random.PRNGKey(ctx.cfg.seed)).squeeze()
        mx = mjx.put_model(mujoco.MjModel.from_xml_path(ctx.cfg.model_path))
        sim = eqx.filter_jit(jax.vmap(controlled_simulate, in_axes=(0, None, None)))
        for _ in range(10):
            time_init = time.time()
            x, u = sim(x_inits, mx, ctx)
            print(f"Time taken: {time.time() - time_init}")
        # runner.train()
