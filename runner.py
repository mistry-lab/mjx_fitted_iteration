import argparse
import time
import mujoco
from mujoco import mjx
import jax
import wandb
from config.cps import ctx as cp_ctx
from config.di import ctx as di_ctx
from simulate import controlled_simulate
from trainer import gen_targets_mapped, make_step, loss_fn
import equinox as eqx
import optax
import copy
import numpy as np
from utils.mj_vis import animate_trajectory

wandb.init(project="fvi", anonymous="allow")
configs = {"cartpole_swing_up": cp_ctx, "double_integrator":di_ctx }


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", help="task name", default="cartpole_swing_up")
        args = parser.parse_args()
        ctx = configs[args.task]

        # Optimiser
        optim = optax.adamw(ctx.cfg.lr)
        opt_state = optim.init(eqx.filter(ctx.cbs.net, eqx.is_array))

        # Random number generator
        init_key = jax.random.PRNGKey(ctx.cfg.seed)
        key, subkey = jax.random.split(init_key)

        with jax.default_device(jax.devices()[0]):
            # Model definition
            model = mujoco.MjModel.from_xml_path(ctx.cfg.model_path)
            model.opt.timestep = ctx.cfg.dt # Setting timestep from Context
            data = mujoco.MjData(model)
            mx = mjx.put_model(model)
            sim = eqx.filter_jit(jax.vmap(controlled_simulate, in_axes=(0, None, None)))
            # sim = jax.vmap(controlled_simulate, in_axes=(0, None, None))
            f_target = eqx.filter_jit(jax.vmap(gen_targets_mapped, in_axes=(0,0,None)))
            f_make_step = eqx.filter_jit(make_step)
            for e in range(ctx.cfg.epochs):
                key, xkey, tkey = jax.random.split(key, num = 3)
                x_inits = ctx.cbs.init_gen(ctx.cfg.batch, xkey)
                # time_init = time.time()
                x, u = sim(x_inits, mx, ctx)
                target  = f_target(x,u, ctx)
                # ctx.cbs.net, opt_state, loss_value = make_step(optim, ctx.cbs.net,opt_state, loss_fn,x, target)
                # wandb.log({"loss": loss_value, "net": ctx.cbs.net})
                # print(f"Time taken: {time.time() - time_init}")

                if e % ctx.cfg.vis == 0 and e > 0:
                    xs = x[jax.random.randint(tkey, (5,), 0, ctx.cfg.nsteps)]
                    term_cst = ctx.cbs.terminal_cost(xs[...,-1,:])
                    print(f"Terminal cost is: {term_cst}")
                    x = np.array(xs.squeeze())
                    print()
                    animate_trajectory(x, data, model)

    except KeyboardInterrupt:
        print("Exit wandb")
        wandb.finish()
