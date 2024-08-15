import yaml
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import optax
from hjb_controller import ValueFunction, Controller


class Runner(object):
    def __init__(self, path_to_cfg):
        def load_cfg(path, class_name='Config'):
            with open(path, 'r') as file:
                config_data = yaml.safe_load(file)

            class_attributes = {"_" + key: value for key, value in config_data.items()}
            return type(class_name, (object,), class_attributes)

        self._cfg = load_cfg(path_to_cfg)
        self._m = mujoco.MjModel.from_xml_path(self._cfg._model_path)
        self._d = mujoco.MjData(self._m)
        self._opt = optax.adam(self._cfg._lr)
        is_gpu = next((d for d in  jax.devices() if 'gpu' in d.device_kind.lower()), None)
        self._mx = mujoco.mjx.put_model(self._m, is_gpu)
        self._vf = ValueFunction(self._cfg._dims, self._cfg._act)
        self._ctrl = Controller(self._vf)


    def _simulate(self, x_inits, m, nsteps):
        dx = mjx.make_data(m)
        qp = x_inits[:m.nq]
        qv = x_inits[m.nq:]
        qpos = dx.qpos.at[:].set(qp)
        qvel = dx.qvel.at[:].set(qv)
        dx_n = dx.replace(qpos=qpos, qvel=qvel)

        def mjx_step(dx, _):
            mjx_data = mjx.step(m, dx)
            return mjx_data, jnp.stack([mjx_data.qpos, mjx_data.qvel])

        _, traj = jax.lax.scan(mjx_step, dx_n, None, length=nsteps)
        return traj


    def _gen_targets(self, traj, cst_func):
        costs = cst_func(traj)
        costs = jnp.flip(costs, axis=0)
        targets = jnp.flip(jnp.cumsum(costs, axis=0), axis=0)
        return targets


    def train(self, x_inits, cst_func):
        traj = self._simulate(x_inits, self._m, self._cfg._nsteps)
        targets = self._gen_targets(traj, cst_func)
        self._opt = self._opt.init(self._vf.net)

        def loss_fn(params):
            return jnp.mean(jnp.square(self._vf.net(params, traj) - targets))

        for i in range(self._cfg._epochs):
            grad = jax.grad(loss_fn)(self._opt.target)
            updates, opt_state = self._opt.update(grad, self._opt.state)
            self._opt = self._opt.replace(state=opt_state)
            self._vf.net = optax.apply_updates(self._vf.net, updates)

