import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from chex import register_dataclass_type_with_jax_tree_util
from jax.lib.xla_extension import jax_jit
from mujoco import mjx
import os
from copy import copy
from jax import config
config.update('jax_default_matmul_precision', 'high')
config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)

model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), '../xmls/two_body.xml'))
mx = mjx.put_model(model)
idata = mujoco.MjData(model)

qx0, qz0, qx1 = -0.375, 0., -0.2
# qx0, qz0, qx1 = -0.1, 0.2, 0.175 
Nlength = 100
# qx0, qx1 = 0.1, 0.175  # NONAN
u0 = 1.
idata.qpos[0],idata.qpos[2], idata.qpos[7] = qx0, qz0, qx1

# @jax.jit
# @jax.vmap
# def gen_and_step_batched(qpos):
#     dx = mjx.make_data(mx)
#     dx = dx.replace(qpos=dx.qpos.at[:].set(qpos))
#     dx = mjx.step(mx, dx)
#     return dx

# # if you remove the jit below this function throws segfault
# @jax.jit
# def gen_and_step(qpos):
#     dx = mjx.make_data(mx)
#     dx = dx.replace(qpos=dx.qpos.at[:].set(qpos))
#     dx = mjx.step(mx, dx)
#     return dx

# qpos0 = jnp.array(d.qpos)
# qpos1 = jnp.repeat(jnp.array([d.qpos]), 1, axis=0)
# qpos2 = jnp.repeat(jnp.array([d.qpos]), 2, axis=0)
# dx2 = gen_and_step_batched(qpos2)
# print(dx2.qpos.shape)
# dx1 = gen_and_step_batched(qpos1)
# print(dx1.qpos.shape)
# dx = gen_and_step(qpos0)
# print(dx.qpos.shape)


# def step_mj(u):
#     d.qfrc_applied[0] = u
#     mujoco.mj_step(model, d)
#     return np.array([d.qpos[0], d.qpos[7]])



# def step_mjx(u):    
#     dx = init_data()
#     dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(u))
#     dx = mjx.step(mx, dx)
#     return jnp.array([dx.qpos[0], dx.qpos[7]])


### finite difference mj
# u, eps = 0.1, 1e-6
# fd_grad = (step_mj(u + eps) - step_mj(u)) / (eps)
# fd_grad_mjx = (step_mjx(jnp.array(u) + eps) - step_mjx(jnp.array(u))) / ( eps)
# jac_fun = jax.jacrev(lambda x: step_mjx(x))
# ad_grad = jac_fun(jnp.array(u))

# # from the results looks like the gradients between mjx and mj are not the same
# # primarily because the response to the qfrcs_applied is not the same
# print(f"FD: {fd_grad}")
# print(f"FD MJX {fd_grad_mjx}")
# print(f"AD: {ad_grad}")

def init_data():
    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(jnp.array(idata.qpos)))
    return dx


def running_cost(dx):
    return jnp.array([dx.qpos[7]**2 + 0.00001*dx.qfrc_applied[0]**2])


def step_scan_mjx(carry, _):
    dx = carry
    dx = mjx.step(mx, dx) # Dynamics function
    t = jnp.expand_dims(dx.time, axis=0)
    cost = running_cost(dx)
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(0.))
    return (dx), jnp.concatenate([dx.qpos, dx.qvel, cost, t])


def simulate_trajectory_mjx(u):
    dx = init_data()
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[0].set(u))
    (dx), res = jax.lax.scan(step_scan_mjx, (dx), None, length=Nlength)
    res, cost, t = res[...,:-2], res[...,-2], res[...,-1]
    return res, cost, t

def compute_trajectory_costs(u):
    res, cost, t = simulate_trajectory_mjx(u)
    return cost, res, t

def simulate_trajectory_mj(u):
    d = mujoco.MjData(model)
    d.qpos = idata.qpos
    d.qfrc_applied[0] = u
    qq = []
    qv = []
    t = []
    for k in range(Nlength):
        mujoco.mj_step(model, d)
        qq.append(d.qpos[:].copy().tolist())
        qv.append(d.qvel[:].copy().tolist())
        t.append(d.time)
        d.qfrc_applied[0] = 0.
    return np.array(qq), np.array(qv), np.array(t)



Nlength = 100
res, cost, t = simulate_trajectory_mjx(u0)
qpos_mjx, qvel_mjx = res[:,:model.nq], res[:,model.nq:]
qpos_mj,qvel_mj, t = simulate_trajectory_mj(u0)



def visualise(qpos, qvel):
    import time
    from mujoco import viewer
    data = mujoco.MjData(model)
    data.qpos = idata.qpos

    with viewer.launch_passive(model, data) as viewer:
        for i in range(qpos.shape[0]):
            step_start = time.time()
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                # time.sleep(time_until_next_step)
                time.sleep(0.05)


# launch visualisation diff threads
# Nlength = 100
# res, cost, t = simulate_trajectory_mjx(u0)
# qpos_mjx, qvel_mjx = res[:,:model.nq], res[:,model.nq:]
# visualise(qpos_mjx, qvel_mjx)

# @jax.jit
# @jax.vmap
# def compute_grad(u):
#     jac_fun = jax.jacrev(lambda x: simulate_trajectory_mjx(x)[0][:,[0,7]])
#     ad_grad = jac_fun(jnp.array(u))
#     return ad_grad

@jax.vmap
def compute_loss_grad(u):
    jac_fun = jax.jacrev(lambda x: loss_funct(x))
    ad_grad = jac_fun(jnp.array(u))
    return ad_grad


cost, res, t = compute_trajectory_costs(u0)

def loss_funct(u):
    costs = compute_trajectory_costs(u)[0]
    return jnp.sum(costs, axis=-1)

# Gradient descent
def gradient_descent(x0, learning_rate=0.1, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        # x =jnp.round(x,5)
        grad = compute_loss_grad(jnp.array([x, 1.,0.2]))[0]
        x_new = x - learning_rate * grad  # Gradient descent update
        
        print(f"Iteration {i}: x = {x}, f(x) = {loss_funct(x)}")
        
        # Check for convergence
        if abs(x_new - x) < tol:
            break
        x = x_new

    return x

def visu_u(u):
    Nlength = 100
    res, cost, t = simulate_trajectory_mjx(u)
    qpos_mjx, qvel_mjx = res[:,:model.nq], res[:,model.nq:]
    visualise(qpos_mjx, qvel_mjx)


# Initial guess
x0 = 2.
# Run gradient descent
# optimal_x = gradient_descent(x0)
# print(f"Optimal x: {optimal_x}, f(x): {loss_funct(optimal_x)}")


# with jax.disable_jit():
    # print("Computing grad..")
    # compute_loss_grad(jnp.array([4.5631613, 1.1,0.25]))[0]


# nan values
# compute_loss_grad(jnp.array([4.5631613, 1.1,0.25]))[0]

# u = 4.563199 # working
# u = 4.5631999 # not working



## finite difference mj
# print(compute_grad(jnp.array([0.1, 0.11, 0.12, 0.5 , 0.12])))




# from the results looks like the gradients between mjx and mj are not the same
# primarily because the response to the qfrcs_applied is not the same


# import matplotlib.pyplot as plt 
# plt.ion()
# ax = plt.subplot(311)
# ax.set_title("Positions")
# ax.plot(t, qpos_mjx[:,0], label="mjx_q0")
# ax.plot(t, qpos_mjx[:,7], label="mjx_q1")

# ax.plot(t, qpos_mj[:,0], label="mj_q0")
# ax.plot(t, qpos_mj[:,7], label="mj_q1")
# ax.legend()

# ax = plt.subplot(312)
# ax.set_title("Velocities")
# ax.plot(t, qvel_mjx[:,0], label="mjx_q0")
# ax.plot(t, qvel_mjx[:,6], label="mjx_q1")

# ax.plot(t, qvel_mj[:,0], label="mj_q0")
# ax.plot(t, qvel_mj[:,6], label="mj_q1")

# ax = plt.subplot(313)
# ax.plot(t, ad_grad[:,0], label="grad_mjx0")
# ax.plot(t, ad_grad[:,1], label="grad_mjx1")
# ax.legend()
# plt.show()

