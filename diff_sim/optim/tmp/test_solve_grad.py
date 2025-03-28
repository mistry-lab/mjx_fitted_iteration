import jax
import jax.numpy as jnp
from mujoco.mjx._src.dataclasses import PyTreeNode
import mujoco
import mujoco.mjx as mjx
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

model = mujoco.MjModel.from_xml_path("xmls/finger_mjx.xml")
mx = mjx.put_model(model)
dx = mjx.make_data(mx)

qpos_init = jnp.array([-.8, 0, -.8])
Nsteps, nu = 3, 2
U0 = 0.2*jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2


def running_cost(dx):
    pos_finger = dx.qpos[2]
    u = dx.ctrl
    return 0. * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2

def set_control(dx, u):
    return dx.replace(ctrl=dx.ctrl.at[:].set(u))

@jax.jit
def loss(U, dx):
    cost = 0.
    for i in range(2):
        dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
        dx = set_control(dx, U[i])  # Fix: properly update dx with set_control
        dx = mjx.step(mx,dx)
        cost += running_cost(dx)
    return cost

# Compute loss
loss_value = loss(U0, dx)

# Compute gradient correctly: Pass `loss` as a function
grad_loss = jax.grad(loss, argnums=0)(U0, dx)  # Differentiate w.r.t. U0

print("Loss:", loss_value)
print("Gradient:", grad_loss)