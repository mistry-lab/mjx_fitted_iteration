import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from chex import register_dataclass_type_with_jax_tree_util
from jax.lib.xla_extension import jax_jit
from mujoco import mjx
import os
#
# def comp_grads():
#     model_path = os.path.join(os.path.dirname(__file__), '../xmls/two_body.xml')
#     m = mujoco.MjModel.from_xml_path(model_path)
#     d = mujoco.MjData(m)
#     mx = mjx.put_model(m)
#
#     # Initialize MuJoCo state
#     d.qpos[0], d.qpos[7] = 0.1, 0.5
#     mujoco.mj_forward(m, d)
#
#     # Define a function for finite differences
#     def step_mj(x):
#         d.qfrc_applied[0] = x
#         mujoco.mj_step(m, d)
#         return np.array([d.qpos[0], d.qpos[7]])
#
#     # Finite difference gradient
#     x, eps = 0.1, 1e-6
#     fd_grad = ((x + eps) - step_mj(x - eps)) / (2 * eps)
#
#     dx = mjx.make_data(mx)
#     qpos = jnp.array(d.qpos)
#     dx = dx.replace(qpos=dx.qpos.at[:].set(qpos))
#     dx = mjx.step(mx, dx)
#     print(dx)
#
#     # JAX-based step function
#     def step_mjx(x):
#         dx = mjx.make_data(mx)
#         # qpos = jnp.array(d.qpos)
#         # dx = dx.replace(qpos=dx.qpos.at[:].set(qpos))
#         # dx = dx.replace(qpos=dx.qpos.at[:].set(jnp.array(d.qpos)))
#         # dx = dx.replace(qvel=dx.qvel.at[:].set(jnp.array(d.qvel)))
#         # dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(x))
#         # qpos = jnp.array([])
#         dx = mjx.step(mx, dx)
#         return dx
#
#     # add leading dimension to x
#     # x = jnp.array(0.1)
#     #
#     # # Gradient calculation with JAX
#     # res = step_mjx(x)
# #     # print(res)
#     grad_fn = jax.grad(lambda x: step_mjx(x))  # Gradient with respect to qpos[0]
#     jax_grad = grad_fn(jnp.array(0.1))
#     return fd_grad
#
# if __name__ == '__main__':
#     fd_grad, jax_grad = comp_grads()
#     print(f'Finite Difference Grad: {fd_grad}')
#     print(f'Jax Grad: {jax_grad}')


model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), '../xmls/two_body.xml'))
mx = mjx.put_model(model)
dx = mjx.make_data(mx)
d = mujoco.MjData(model)
d.qpos[0], d.qpos[7] = 0.1, 0.175

@jax.jit
@jax.vmap
def gen_and_step_batched(qpos):
    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos))
    dx = mjx.step(mx, dx)
    return dx

# if you remove the jit below this function throws segfault
@jax.jit
def gen_and_step(qpos):
    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos))
    dx = mjx.step(mx, dx)
    return dx

qpos0 = jnp.array(d.qpos)
# qpos1 = jnp.repeat(jnp.array([d.qpos]), 1, axis=0)
# qpos2 = jnp.repeat(jnp.array([d.qpos]), 2, axis=0)
# dx2 = gen_and_step_batched(qpos2)
# print(dx2.qpos.shape)
# dx1 = gen_and_step_batched(qpos1)
# print(dx1.qpos.shape)
dx = gen_and_step(qpos0)
print(dx.qpos.shape)


def step_mj(u):
    d.qfrc_applied[0] = u
    mujoco.mj_step(model, d)
    return np.array([d.qpos[0], d.qpos[7]])

def step_mjx(u):
    def init_data():
        dx = mjx.make_data(mx)
        dx = dx.replace(qpos=dx.qpos.at[:].set(jnp.array(d.qpos)))
        return dx

    dx = init_data()
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(u))
    dx = mjx.step(mx, dx)
    return jnp.array([dx.qpos[0], dx.qpos[7]])


### finite difference mj
u, eps = 0.0, 1e-6
fd_grad = (step_mj(u + eps) - step_mj(u - eps)) / (2 * eps)
fd_grad_mjx = (step_mjx(jnp.array(u) + eps) - step_mjx(jnp.array(u) - eps)) / (2 * eps)
jac_fun = jax.jacrev(lambda x: step_mjx(x))
ad_grad = jac_fun(jnp.array(u))

# from the results looks like the gradients between mjx and mj are not the same
# primarily because the response to the qfrcs_applied is not the same
print(f"FD: {fd_grad}")
print(f"FD MJX {fd_grad_mjx}")
def mjx_to_mj_step(u):
    def init_data():
        dx = mjx.make_data(mx)
        dx = dx.replace(qpos=dx.qpos.at[:].set(jnp.array(d.qpos)))
        return dx

    dx = init_data()
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(u))
    dx = mjx.step(mx, dx)
    return jnp.array([dx.qpos[0], dx.qpos[7]])
print(f"AD: {ad_grad}")

