import argparse
import mujoco
from mujoco import mjx
import wandb
from contexts.contexts import ctxs
from simulate import controlled_simulate
import equinox as eqx
import optax
from utils.tqdm import trange
import numpy as np
from utils.mj_vis import animate_trajectory
import jax.debug
from trainer import make_step, loss_fn, loss_fn_td, loss_fn_target
from jax import config

config.update('jax_default_matmul_precision', jax.lax.Precision.HIGH)

wandb.init(project="fvi", anonymous="allow", mode='offline')

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", help="task name", default="cartpole_swing_up")
        parser.add_argument("--loss_type", help="loss type", default="td")
        args = parser.parse_args()
        ctx = ctxs[args.task]
        # Random number generator
        init_key = jax.random.PRNGKey(ctx.cfg.seed)
        key, subkey = jax.random.split(init_key)

        with jax.default_device(jax.devices()[0]):
            # Model definition
            model = mujoco.MjModel.from_xml_path(ctx.cfg.path)
            data = mujoco.MjData(model)
            # mx = mjx.put_model(model)
            net = ctx.cbs.gen_network()
            optim = optax.adamw(ctx.cfg.lr)
            opt_state = optim.init(eqx.filter(net, eqx.is_array))
            es = trange(ctx.cfg.epochs)
            for e in es:
                key, xkey, tkey = jax.random.split(key, num = 3)
                x_inits = ctx.cbs.init_gen(ctx.cfg.batch, xkey)
                key, xkey, tkey = jax.random.split(key, num = 3)
                net, opt_state, loss_value = net.make_step(optim, net, opt_state, loss_fn, x_inits, ctx)
                wandb.log({"loss": loss_value})
                es.set_postfix({"Loss": loss_value})

                if e % ctx.cfg.vis == 0 or e == ctx.cfg.epochs - 1:
                    key, xkey, tkey = jax.random.split(key, num = 3)
                    x_inits = x_inits[jax.random.randint(tkey, (2,), 0, ctx.cfg.batch)]
                    x, _,_,_ = controlled_simulate(x_inits, ctx, net)
                    x = jax.vmap(jax.vmap(ctx.cbs.state_decoder))(x)
                    x = np.array(x.squeeze())
                    animate_trajectory(x, data, model)

    except KeyboardInterrupt:
        print("Exit wandb")
        wandb.finish()
