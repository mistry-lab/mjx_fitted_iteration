import argparse
import wandb
import mujoco
from mujoco import viewer
import equinox as eqx
import optax
import jax.debug
from jax import config
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
        args = parser.parse_args()
        ctx = ctxs[args.task]

        # Initialize wandb
        wandb.init(anonymous="allow", mode='offline') if args.wb_project is None else (
            wandb.init(project=args.wb_project, anonymous="allow")
        )

        # initial keys for random number generation these will be split for each iteration
        init_key = jax.random.PRNGKey(ctx.cfg.seed)
        key, subkey = jax.random.split(init_key)

        # model and data for rendering (CPU side)
        model = mujoco.MjModel.from_xml_path(ctx.cfg.path)
        data = mujoco.MjData(model)

        # start the training loop in jax default device and launch a passive viewer
        with viewer.launch_passive(model, data) as viewer:
            net, optim = ctx.cbs.gen_network(ctx.cfg.seed), optax.adamw(ctx.cfg.lr)
            opt_state = optim.init(eqx.filter(net, eqx.is_array))

            # run through the epochs and log the loss
            for e in (es := trange(ctx.cfg.epochs)):
                key, xkey, tkey = jax.random.split(key, num = 3)
                x_inits = ctx.cbs.init_gen(ctx.cfg.batch * ctx.cfg.samples, xkey)
                net, opt_state, loss_value, traj_cost = net.make_step(optim, net, opt_state, x_inits, ctx)
                log_data = {"loss": round(loss_value.item(), 3), "Traj Cost": round(traj_cost.item(), 3)}
                wandb.log(log_data)
                es.set_postfix(log_data)

                if e % ctx.cfg.eval == 0 or e == ctx.cfg.epochs - 1:
                    visualise_policy(data, model, viewer, ctx, net, key)
                    save_model(net, f"{args.task}_checkpoint_{e}.pkl")

    except KeyboardInterrupt:
        print("Exit wandb")
        wandb.finish()


