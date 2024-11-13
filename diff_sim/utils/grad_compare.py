import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import os

def comp_grads():
    model_path = os.path.join(os.path.dirname(__file__), '../xmls/two_body.xml')
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mx = mjx.put_model(m)

    # Initialize MuJoCo state
    d.qpos[0], d.qpos[7] = 0.1, 0.25
    mujoco.mj_forward(m, d)

    # Define a function for finite differences
    def step_mj(x):
        d.qfrc_applied[0] = x
        mujoco.mj_step(m, d)
        return np.array([d.qpos[0], d.qpos[7]])

    # Finite difference gradient
    x, eps = 0.1, 1e-6
    fd_grad = (step_mj(x + eps) - step_mj(x - eps)) / (2 * eps)

    # JAX-based step function
    def step_mjx(x):
        dx = mjx.make_data(mx)
        dx = dx.replace(qpos=dx.qpos.at[:].set(jnp.array(d.qpos)))
        dx = dx.replace(qvel=dx.qvel.at[:].set(jnp.array(d.qvel)))
        dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(x))
        dx = mjx.step(mx, dx)
        return jnp.array([dx.qpos[0], dx.qpos[7]])

    # Gradient calculation with JAX
    grad_fn = jax.grad(lambda x: step_mjx(x))  # Gradient with respect to qpos[0]
    jax_grad = grad_fn(jnp.array(0.1))
    return fd_grad, jax_grad


if __name__ == '__main__':
    fd_grad, jax_grad = comp_grads()
    print(f'Finite Difference Grad: {fd_grad}')
    print(f'Jax Grad: {jax_grad}')
