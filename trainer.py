import time
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import wandb
from utils.mj_vis import animate_trajectory
from utils.tqdm import trange
from config.cps import ctx
from simulate import controlled_simulate
from hjb_controller import Controller

class Runner(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self._m = mujoco.MjModel.from_xml_path(self._cfg.model_path)
        self._d = mujoco.MjData(self._m)
        self.x_key, self.traj_select, self.model_key = jax.random.split(jax.random.PRNGKey(self._cfg.seed), num=3)

    def _gen_targets_mapped(self, x, u):
        @jax.jit
        def csts(x, u):
            xs, xt = x[:-1], x[-1]
            xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
            ucost = self._cfg.ctrl_cst(u)
            xcst = self._cfg.run_cst(xs)
            tcst = self._cfg.terminal_cst(xt)
            xcost = jnp.concatenate([xcst, tcst])
            return xcost + ucost

        costs = jnp.flip(csts(x, u))
        targets = jnp.flip(jnp.cumsum(costs))
        return targets

    def train(self):
        mx = mjx.put_model(self._m)
        optim = optax.adamw(self._cfg.lr)
        ctrl = Controller(self._cfg.dims, self._cfg.act, self.model_key, self._cfg.R)
        opt_state = optim.init(eqx.filter(ctrl, eqx.is_array))

        @eqx.filter_jit
        def make_step(model, state, loss, x, y):
            params, static = eqx.partition(model, eqx.is_array)
            loss_value, grads = jax.value_and_grad(loss)(params, static, x, y)
            updates, state = optim.update(grads, state, model)
            model = eqx.apply_updates(model, updates)
            return model, state, loss_value

        def loss_fn(params, static, x, y):
            model = eqx.combine(params, static)
            pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
            y = y.reshape(-1, 1)
            return jnp.mean(jnp.square(pred - y))

        for e in trange(self._cfg.epochs, desc='Epochs completed'):
            start_time = time.time()
            x_inits = self._init_gen(self._cfg.batch, self.x_key).squeeze()
            x, u = jax.jit(
                jax.vmap(controlled_simulate, in_axes=(0, None, None)), static_argnums=2
            )(x_inits, mx, self._cfg)
            # targets = jax.vmap(self._gen_targets_mapped)(x, u)
            # self._ctrl.vf, opt_state, lv = make_step(self._ctrl.vf, opt_state, loss_fn, x, targets)
            print(f"Epoch {e} took {time.time() - start_time} seconds")
            # wandb.log({"loss": lv, "net": self._ctrl.vf})
            if e % self._cfg.vis == 0 and e > 0:
                x = np.array(x[jax.random.randint(self.traj_select, (5,), 0, self._cfg.nsteps)].squeeze())
                animate_trajectory(x, self._d, self._m)
