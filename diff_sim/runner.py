import argparse
import wandb
import numpy as np
import mujoco
from mujoco import viewer
import equinox as eqx
import optax
import jax.debug
from jax import config
from diff_sim.context.tasks import ctxs
from diff_sim.simulate import controlled_simulate
from diff_sim.utils.tqdm import trange
from diff_sim.utils.mj_vis import animate_trajectory

config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", help="task name", default="cartpole_swing_up")
        parser.add_argument("--wb_project", help="wandb project name", default="diff_sim")
        args = parser.parse_args()
        ctx = ctxs[args.task]

        # Initialize wandb
        wandb.init(anonymous="allow", mode='offline') if args.wb_project is None else (
            wandb.init(project=args.wb_project, anonymous="allow", mode='offline')
        )

        # initial keys for random number generation these will be split for each iteration
        init_key = jax.random.PRNGKey(ctx.cfg.seed)
        key, subkey = jax.random.split(init_key)

        # model and data for rendering (CPU side)
        model = mujoco.MjModel.from_xml_path(ctx.cfg.path)
        data = mujoco.MjData(model)

        # start the training loop in jax default device and launch a passive viewer
        with (jax.default_device(jax.devices()[0])) and viewer.launch_passive(model, data) as viewer:
            net, optim = ctx.cbs.gen_network(), optax.adamw(ctx.cfg.lr)
            opt_state = optim.init(eqx.filter(net, eqx.is_array))
            es = trange(ctx.cfg.epochs)

            # run through the epochs and log the loss
            for e in es:
                key, xkey, tkey = jax.random.split(key, num = 3)
                x_inits = ctx.cbs.init_gen(ctx.cfg.batch, xkey)
                key, xkey, tkey = jax.random.split(key, num = 3)
                net, opt_state, loss_value = net.make_step(optim, net, opt_state, x_inits, ctx)
                wandb.log({"loss": loss_value})
                es.set_postfix({"Loss": loss_value})

                if e % ctx.cfg.vis == 0 or e == ctx.cfg.epochs - 1:
                    key, xkey, tkey = jax.random.split(key, num = 3)
                    x_inits = x_inits[jax.random.randint(tkey, (2,), 0, ctx.cfg.batch)]
                    x, _,_,_ = controlled_simulate(x_inits, ctx, net)
                    x = jax.vmap(jax.vmap(ctx.cbs.state_decoder))(x)
                    x = np.array(x.squeeze())
                    animate_trajectory(x, data, model, viewer)

    except KeyboardInterrupt:
        print("Exit wandb")
        wandb.finish()
