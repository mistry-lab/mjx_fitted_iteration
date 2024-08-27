import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import wandb
from mj_vis import animate_trajectory
from utils.tqdm import trange
from hjb_controller import Controller
from config.cps import cartpole_cfg

#init wandb
wandb.init(project="fvi", entity="danielhfn")
cfg = {"cartpole": cartpole_cfg}

class Runner(object):
    def __init__(self, name):
        self._cfg = cfg[name]
        is_gpu = next((d for d in  jax.devices() if 'gpu' in d.device_kind.lower()), None)
        self._m = mujoco.MjModel.from_xml_path(self._cfg.model_path)
        self._d = mujoco.MjData(self._m)
        self._mx = mjx.put_model(self._m, device=is_gpu)
        self._opt = optax.adam(self._cfg.lr)
        self.x_key, self.model_key = jax.random.split(jax.random.PRNGKey(self._cfg.seed))
        self._ctrl = Controller(self._cfg.dims, self._cfg.act, self.model_key)
        self._init_gen = self._cfg.init_gen

    def _simulate(self, nsteps):
        def set_init(dx, x):
            qpos = dx.qpos.at[:].set(x[:self._mx.nq])
            qvel = dx.qvel.at[:].set(x[self._mx.nq:])
            return dx.replace(qpos=qpos, qvel=qvel)

        def mjx_step(dx, _):
            dx = self._ctrl(self._mx, dx)
            mjx_data = mjx.step(self._mx, dx)
            return mjx_data, jnp.stack([mjx_data.qpos, mjx_data.qvel, mjx_data.ctrl]).squeeze()

        dx = mjx.make_data(self._mx)
        x_inits = self._init_gen(self.x_key)
        dx = set_init(dx, x_inits)

        _, traj = jax.lax.scan(mjx_step, dx, None, length=nsteps)
        return traj

    def _gen_targets(self, traj):
        def csts(traj):
            us, xs, xt = traj[:, -self._mx.nu:], traj[:, :-self._mx.nu], traj[:, :-self._mx.nu]
            ucst = self._cfg.ctrl_cst(us)
            xcst = self._cfg.run_cst(xs)
            tcst = self._cfg.terminal_cst(xt)
            return jnp.sum(ucst + xcst + tcst, axis=0)

        costs = jnp.flip(csts(traj) , axis=0)
        targets = jnp.flip(jnp.cumsum(costs, axis=0), axis=0)
        return targets


    def train(self):
        optim = optax.adamw(self._cfg.lr)
        opt_state = optim.init(eqx.filter(self._ctrl.vf, eqx.is_array))

        @eqx.filter_jit
        def make_step(model, state, loss, x, y):
            params, static = eqx.partition(model, eqx.is_array)
            loss_value, grads = jax.value_and_grad(loss)(params, static, x, y)
            updates, state = optim.update(grads, state, model)
            model = eqx.apply_updates(model, updates)
            return model, state, loss_value

        def loss_fn(params, static, x, y):
            model = eqx.combine(params, static)
            return jnp.mean(jnp.square(model(x) - y))

        for e in trange(self._cfg.epochs, desc='Epochs completed'):
            traj = self._simulate(self._cfg.nsteps)
            targets = self._gen_targets(traj)
            self._ctrl.vf, opt_state, loss_value = make_step(self._ctrl.vf, opt_state, loss_fn, traj, targets)
            wandb.log({"loss": loss_value})

            if e % 100 == 0:
                traj = np.array(traj[jax.random.randint(self.x_key, (5,), 0, self._cfg.nsteps)].squeeze())
                animate_trajectory(traj, self._d, self._m)

