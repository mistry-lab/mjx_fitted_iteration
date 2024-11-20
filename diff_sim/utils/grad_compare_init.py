import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import os
from jax import config

# Jax configs
config.update('jax_default_matmul_precision', 'high')
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)

def running_cost(dx):
    return jnp.array([dx.qpos[7]**2 + 0.00001*dx.qfrc_applied[0]**2])

def step_scan_mjx(carry, _):
    dx = carry
    dx = mjx.step(mx, dx) # Dynamics function
    t = jnp.expand_dims(dx.time, axis=0)
    cost = running_cost(dx)
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:].set(jnp.zeros_like(dx.qfrc_applied)))
    return (dx), jnp.concatenate([dx.qpos, dx.qvel, cost, t])

@jax.vmap
def simulate_trajectory_mjx(qpos_init, u):
    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:].set(u))
    (dx), res = jax.lax.scan(step_scan_mjx, (dx), None, length=Nlength)
    res, cost, t = res[...,:-2], res[...,-2], res[...,-1]
    return res, cost, t

def compute_trajectory_costs(qpos_init, u):
    res, cost, t = simulate_trajectory_mjx(qpos_init, u)
    return cost, res, t

def simulate_trajectory_mj(qpos_init, u):
    d = mujoco.MjData(model)
    d.qpos = qpos_init
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
                time.sleep(0.075)

def visu_u(u0):
    qpos = jnp.expand_dims(jnp.array(idata.qpos), axis=0)
    qpos = jnp.repeat(qpos,1, axis = 0)

    u = set_u(jnp.array([u0]))
    res, cost, t = simulate_trajectory_mjx(qpos, u)
    qpos_mjx, qvel_mjx = res[0,:,:model.nq], res[0,:,model.nq:]
    visualise(qpos_mjx, qvel_mjx)

@jax.jit
def compute_loss_grad(qpos_init, u):
    jac_fun = jax.jacrev(lambda x: loss_funct(qpos_init,x))
    ad_grad = jac_fun(jnp.array(u))
    return ad_grad


def loss_funct(qpos_init, u):
    u = set_u(u)
    costs = compute_trajectory_costs(qpos_init,u)[0]
    costs = jnp.sum(costs, axis=1)
    costs = jnp.mean(costs)
    return costs

def set_u(u0):
    u = jnp.zeros_like(idata.qfrc_applied)
    u = jnp.expand_dims(u, axis=0)
    u = jnp.repeat(u, u0.shape[0], axis=0)
    u = u.at[:,0].set(u0)
    return u

@jax.jit
def get_traj_grad(qpos_init,u):
    jac_fun = jax.jacrev(lambda x: compute_trajectory_costs(qpos_init,set_u(x))[0])
    ad_grad = jac_fun(jnp.array(u))
    return ad_grad

# Gradient descent
def gradient_descent(qpos, x0, learning_rate=0.1, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        # x =jnp.round(x,5)
        grad = compute_loss_grad(qpos, x)[0]
        x_new = x - learning_rate * grad  # Gradient descent update
        
        print(f"Iteration {i}: x = {x}, f(x) = {loss_funct(qpos, x)}")

        # Check for convergence
        if abs(x_new - x) < tol:
            break
        x = x_new

    return x

if __name__ == "__main__":

    # Load mj and mjx model
    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), '../xmls/two_body.xml'))
    mx = mjx.put_model(model)
    idata = mujoco.MjData(model)

    qx0, qz0, qx1 = -0.375, 0.1, -0.2 # Inititial positions
    idata.qpos[0],idata.qpos[2], idata.qpos[7] = qx0, qz0, qx1
    Nlength = 250 # use smaller nlength to show nan behaviour

    u0 = 15.
    batch = 1
    qpos = jnp.expand_dims(jnp.array(idata.qpos), axis=0)
    qpos = jnp.repeat(qpos,batch, axis = 0)
    # u = set_u(jnp.array([u0]))

    # Simulate trajectory
    # res, cost, t = simulate_trajectory_mjx(qpos, u)
    # qpos_mjx, qvel_mjx = res[0,:,:model.nq], res[0,:,model.nq:]

    # Visualise a trajectory
    visu_u(u0)

    # Initial guess
    x0 = jnp.array([u0])
    # Initial position
    qpos = jnp.expand_dims(jnp.array(idata.qpos), axis=0)
    # Run gradient descent
    optimal_x = gradient_descent(qpos, x0, learning_rate=1., max_iter=10)

    # Check the gradient along the trajectory
    grads = get_traj_grad(qpos, x0)
    print(grads)
    # Observation
    # Box -- Sphere example. 
    # Nan values directly on the first call. (unless u = 0.)
    # jax.jit does not change the nan values.
    # qx0, qz0, qx1 = -0.375, 0.1, -0.2  
    # u = anything. (15. for dt = 0.001 for example)
    # Nlenght = 250


    # Sphere -- Sphere example.
    # 2 balls in collision, less nan. Only specific numbers. 
    # dt = 0.01, Nlenght = 100
    # qx0, qz0, qx1 = -0.375, 0., -0.2
    # u = jnp.array([4.5631613])
    # dt = 0.001, Nlenght = 500
    # Nans goes away. u ~ 45.