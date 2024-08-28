# test gpu usage
import jax
import mujoco
from mujoco import mjx

mjx_model = mjx.put_model(mujoco.MjModel.from_xml_path('/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml'))

# mjx_model = mjx.put_model(model, device=jax.devices()[0])

@jax.vmap
def batched_step(vel):
  mjx_data = mjx.make_data(mjx_model)
  qvel = mjx_data.qvel.at[0].set(vel)
  mjx_data = mjx_data.replace(qvel=qvel)
  pos = mjx.step(mjx_model, mjx_data).qpos[0]
  return pos

vel = jax.numpy.arange(0.0, 1.0, 0.0000001)
pos = jax.jit(batched_step)(vel)
#
# from jax import numpy as jnp
# from jax import random
# import jax
# import mujoco
# from mujoco import mjx
#
#
# def _init_gen(batch, key):
#   key = random.split(key, 4)
#   qp = jax.random.uniform(key[0], (batch, 1), minval=-0.1, maxval=0.1)
#   qc = jax.random.uniform(key[1], (batch, 1), minval=-0.1, maxval=0.1)
#   vc = jax.random.uniform(key[2], (batch, 1), minval=-0.1, maxval=0.1)
#   vp = jax.random.uniform(key[3], (batch, 1), minval=-0.1, maxval=0.1)
#   return jnp.concatenate([qc, qp, vc, vp], axis=1)
#
#
# def _simulate(x_inits):
#   _mx = mjx.put_model(mujoco.MjModel.from_xml_path('/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml'))
#
#   def set_init(dx, x):
#     qpos = dx.qpos.at[:].set(x[:_mx.nq])
#     qvel = dx.qvel.at[:].set(x[_mx.nq:])
#     dx = dx.replace(qpos=qpos, qvel=qvel)
#     return mjx.step(_mx, dx)
#
#   def _ctrl(dx):
#     # generate random control
#     ctrl = random.normal(random.PRNGKey(0), (1,))
#     ctrl = dx.ctrl.at[:].set(ctrl)
#     dx = dx.replace(ctrl=ctrl)
#     return dx
#
#   def mjx_step(dx, _):
#     dx = _ctrl(dx)
#     dx = mjx.step(_mx, dx)
#     return dx, jnp.concatenate([dx.qpos, dx.qvel, dx.ctrl], axis=0)
#
#   dx = mjx.make_data(_mx)
#   batched_dx = jax.vmap(set_init, in_axes=(None, 0))(dx, x_inits)
#
#   @jax.jit
#   def scan_fn(dx, _):
#     # do not compute gradients through the simulation
#     return jax.lax.scan(mjx_step, dx, None, length=2)
#
#   _, batched_traj = jax.vmap(scan_fn)(batched_dx, None)
#   x, u = batched_traj[..., :-_mx.nu], batched_traj[..., -_mx.nu:]
#   return x, u
#
#
# if __name__ == "__main__":
#   x_inits = _init_gen(100000, random.PRNGKey(0))
#   x, u = _simulate(x_inits)
