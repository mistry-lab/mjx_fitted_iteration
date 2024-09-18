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
import jax.numpy as jnp
from utils.mj_vis import animate_trajectory
import jax.debug
from trainer import gen_traj_targets, make_step, loss_fn_td, loss_fn_target, gen_traj_cost


wandb.init(project="fvi", anonymous="allow", mode='online')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", help="task name", default="double_integrator")
        parser.add_argument("--loss_type", help="loss type", default="td")
        args = parser.parse_args()
        ctx = ctxs[args.task]
        # Random number generator
        init_key = jax.random.PRNGKey(ctx.cfg.seed)
        key, subkey = jax.random.split(init_key)

        with jax.default_device(jax.devices()[0]):
            # Model definition
            model = mujoco.MjModel.from_xml_path(ctx.cfg.model_path)
            model.opt.timestep = ctx.cfg.dt # Setting timestep from Context
            data = mujoco.MjData(model)
            mx = mjx.put_model(model)
            net = ctx.cbs.gen_network()
            optim = optax.adamw(ctx.cfg.lr)
            opt_state = optim.init(eqx.filter(net, eqx.is_array))
            times = jnp.repeat(ctx.cfg.horizon.reshape(1,ctx.cfg.horizon.shape[-1]), ctx.cfg.batch, axis=0)
            sim = eqx.filter_jit(jax.vmap(controlled_simulate, in_axes=(0, None, None, None, None)))
            truths = gen_traj_cost if args.loss_type == "td" else gen_traj_targets
            loss = loss_fn_td if args.loss_type == "td" else loss_fn_target
            truths = eqx.filter_jit(jax.vmap(truths, in_axes=(0, 0, None)))

            for e in trange(ctx.cfg.epochs):
                key, xkey, tkey = jax.random.split(key, num = 3)
                x_inits = ctx.cbs.init_gen(ctx.cfg.batch, xkey)
                x, u = sim(x_inits, mx, ctx, net, PD=False)
                td_res = truths(x, u, ctx)
                net, opt_state, loss_value = make_step(optim, net, opt_state, loss, x, times, td_res[0])
                wandb.log({"loss": loss_value, "cost": jnp.mean(td_res[1])})
                print(f"Epoch {e}, Loss: {loss_value.item()}")

                if e % ctx.cfg.vis == 0 or e == ctx.cfg.epochs - 1:
                    xs = x[jax.random.randint(tkey, (5,), 0, ctx.cfg.nsteps)]
                    x = np.array(xs.squeeze())
                    animate_trajectory(x, data, model)

    except KeyboardInterrupt:
        print("Exit wandb")
        wandb.finish()
