import argparse
import mujoco
from mujoco import mjx
import wandb
from config.cps import ctx as cp_ctx
from config.di import ctx as di_ctx
from simulate import controlled_simulate
# from trainer import gen_targets_mapped, make_step, loss_fn_tagret
import equinox as eqx
import optax
import numpy as np
import jax.numpy as jnp
from utils.mj_vis import animate_trajectory
import jax.debug
from trainer import gen_targets_mapped, make_step, loss_fn_td, loss_fn_target, gen_traj_cost

wandb.init(project="fvi", anonymous="allow", mode='online')
configs = {
    "cartpole_swing_up": cp_ctx,
    "double_integrator":di_ctx
}


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
            times = jnp.repeat(di_ctx.cfg.horizon.reshape(1,di_ctx.cfg.horizon.shape[-1]), 10000, axis=0) 

            sim = eqx.filter_jit(jax.vmap(controlled_simulate, in_axes=(0, None, None)))
            f_target = eqx.filter_jit(jax.vmap(gen_targets_mapped, in_axes=(0, 0, None)))
            f_make_step = eqx.filter_jit(make_step)
            for e in range(100):
                key, xkey, tkey = jax.random.split(key, num = 3)
                x_inits = ctx.cbs.init_gen(ctx.cfg.batch, xkey)
                # time_init = time.time()
                x, u = sim(x_inits, mx, ctx)
                target, total_cost, terminal_cost =  f_target(x,u, ctx)

                for k in range(5):
                    di_ctx.cbs.net, opt_state, loss_value = make_step(optim, di_ctx.cbs.net,opt_state, loss_fn_td, x, times, target)
                    wandb.log({"loss": loss_value, "cost": jnp.mean(loss_value)})

                if e % ctx.cfg.vis == 0 :
                    xs = x[jax.random.randint(tkey, (5,), 0, ctx.cfg.nsteps)]
                    term_cst = ctx.cbs.terminal_cost(xs[...,-1,:])
                    print(f"Terminal cost is: {term_cst}")
                    x = np.array(xs.squeeze())
                    print()
                    animate_trajectory(x, data, model)

                    # plot the network over discrete states as meshgrid
                    x = np.linspace(-1, 1, 100)
                    y = np.linspace(-.5, .5, 100)
                    xx, yy = np.meshgrid(x, y)
                    z = np.zeros_like(xx)
                    for i in range(100):
                        for j in range(100):
                            z[i, j] = ctx.cbs.net(jnp.array([xx[i, j], yy[i, j]])).item()

                    # plot using matplotlib
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(xx, yy, z, cmap='viridis')
                    plt.show()

    except KeyboardInterrupt:
        print("Exit wandb")
        wandb.finish()
