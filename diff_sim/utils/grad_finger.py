import jax
from jax import config
config.update('jax_default_matmul_precision', 'high')
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mujoco
from mujoco import mjx

def running_cost(dx):
    return jnp.array([dx.qpos[7] ** 2 + 0.1 * dx.qfrc_applied[0] ** 2])


@jax.vmap
def simulate_trajectory_mjx(qpos_init, u):
    """ Simulate the impulse of the force. Return states and costs."""

    def step_scan_mjx(carry, _):
        dx = carry
        dx = mjx.step(mx, dx)  # Dynamics function
        t = jnp.expand_dims(dx.time, axis=0)
        cost = running_cost(dx)
        dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:].set(jnp.zeros_like(dx.qfrc_applied)))
        return (dx), jnp.concatenate([dx.qpos, dx.qvel, cost, t])

    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=dx.qpos.at[:].set(qpos_init))
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:].set(u))
    (dx), res = jax.lax.scan(step_scan_mjx, (dx), None, length=Nlength)
    res, cost, t = res[..., :-2], res[..., -2], res[..., -1]
    return res, cost, t


def compute_trajectory_costs(qpos_init, u):
    """ Wrapper function to compute the gradient wrt costs."""
    res, cost, t = simulate_trajectory_mjx(qpos_init, u)
    return cost, res, t


def visu_u(u0):
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

    qpos = jnp.expand_dims(jnp.array(idata.qpos), axis=0)
    qpos = jnp.repeat(qpos, 1, axis=0)

    u = set_u(jnp.array([u0]))
    res, _, _ = simulate_trajectory_mjx(qpos, u)
    qpos_mjx, qvel_mjx = res[0, :, :model.nq], res[0, :, model.nq:]
    visualise(qpos_mjx, qvel_mjx)


@jax.jit
def compute_loss_grad(qpos_init, u):
    """ Compute gradient of loss wrt u"""
    jac_fun = jax.jacrev(lambda x: loss(qpos_init, x), has_aux=True)
    ad_grad, costs = jac_fun(jnp.array(u))
    return ad_grad, costs


@jax.jit
def loss(qpos_init, u):
    """ Sum of running costs"""
    u = set_u(u)
    costs = compute_trajectory_costs(qpos_init, u)[0]
    costs = jnp.sum(costs, axis=1)
    costs = jnp.mean(costs)
    return costs, costs


def set_u(u0):
    """ Utility function to initialize data"""
    u = jnp.zeros_like(idata.qfrc_applied)
    u = jnp.expand_dims(u, axis=0)
    u = jnp.repeat(u, u0.shape[0], axis=0)
    u = u.at[:, 0].set(u0)
    return u


@jax.jit
def compute_traj_grad_wrt_u(qpos_init, u):
    """ Gradient vector to debug the NaN occurence."""
    jac_fun = jax.jacrev(lambda x: compute_trajectory_costs(qpos_init, set_u(x))[0])
    ad_grad = jac_fun(jnp.array(u))
    return ad_grad


# Gradient descent
def gradient_descent(qpos, x0, learning_rate=0.1, tol=1e-6, max_iter=100):
    """ Optimise initial force."""
    x = x0
    for i in range(max_iter):
        grad, costs = compute_loss_grad(qpos, x)
        x_new = x - learning_rate * grad[0]  # Gradient descent update

        print(f"Iteration {i}: x = {x_new}, f(x) = {loss(qpos, x_new)}, Costs: {costs}")

        # Check for convergence
        if abs(x_new - x) < tol or jnp.isnan(grad):
            # print if it has converged or not
            print(f"Converged: {abs(x_new - x) < tol}")
            print(f"Gradient is NaN: {jnp.isnan(grad)}")
            break
        x = x_new

    return x


# The XML model as a string
model_xml = """
<mujoco model="planar point mass">
    <visual>
        <quality shadowsize="2048" />
        <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1" />
    </visual>

  <option impratio="10" iterations="1" ls_iterations="4" timestep="0.01">
    <flag eulerdamp="disable"/>
  </option>

  <custom>
    <numeric data="15" name="max_contact_points"/>
    <numeric data="15" name="max_geom_pairs"/>
  </custom>

    <asset>
        <texture type="skybox" name="skybox" builtin="gradient" mark="random" rgb1="0.4 0.6 0.8" rgb2="0 0 0" markrgb="1 1 1" width="800" height="4800" />
        <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" markrgb="0.2 0.3 0.4" width="300" height="300" />
        <material name="grid" texture="grid" texuniform="true" reflectance="0.2" />
    </asset>

    <worldbody>
        <geom name="ground" size="1 1 0.1" pos="-0.0 0 -0.013" type="plane" rgba=".123 .140 .28 1" contype="1" conaffinity="2"/>

        <body name="object1" pos="0.0 0.0 0.035">
            <joint type="free" damping="0.0001"/>
            <camera name="cam1" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.7 0.7"/>
            <!-- <geom name="pointmass1" type="sphere" size=".042" material="grid" mass=".01" condim="3"  group="2" solimp="0.1 0.95 0.1 0.5 2" solref="0.02 1.0"/> -->
            <geom name="pointmass1" type="capsule" size=".042 0.01" material="grid" mass=".01" condim="3" contype="7"  group="2" solref="0.01 1.0"/>
        </body>

        <body name="object2" pos="0.0 0.0 0.035">
            <joint type="free" damping="0.0001"/>
            <camera name="cam2" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.7 0.7"/>
            <geom name="pointmass2" type="sphere" size=".042" material="grid" mass=".01" condim="3" contype="7"  friction="1.1 0.01 0.003" group="2" solref="0.01 1.0"/>
        </body>
        <body mocap="true" name="mocap_target">
          <geom type="sphere" size="0.025" rgba="1 0 0 1" contype="0" conaffinity="0"/>
        </body>
    </worldbody>
</mujoco>
"""

# Load mj and mjx model
model = mujoco.MjModel.from_xml_string(model_xml)
mx = mjx.put_model(model)
idata = mujoco.MjData(model)

dx = mjx.make_data(mx)


# # Convert everything to the 64bit versions.
# def upscale(x):
#     if 'dtype' in dir(x):
#         if x.dtype == jnp.int32:
#             return jnp.int64(x)
#         elif x.dtype == jnp.float32:
#             return jnp.float64(x)
#             return jnp.float64(x)
#     return x
#
#
# data_init = jax.tree.map(upscale, dx)

qx0, qz0, qx1 = -0.375, 0.1, -0.2  # Inititial conditions
idata.qpos[0], idata.qpos[2], idata.qpos[7] = qx0, qz0, qx1
Nlength = 100  # horizon lenght

u0, batch = 2., 1  # Initial guess
u0_jnp = jnp.array([u0])
qpos = jnp.expand_dims(jnp.array(idata.qpos), axis=0)
qpos = jnp.repeat(qpos, batch, axis=0)

# Visualise guess
# visu_u(u0)

# Run gradient descent
optimal_u = gradient_descent(qpos, u0_jnp, learning_rate=0.5).squeeze()
visu_u(optimal_u)
# Check the gradient along the trajectory, debugging
# print(compute_traj_grad_wrt_u(qpos, jnp.array([2.8]))) # Working
# print(compute_traj_grad_wrt_u(qpos, jnp.array([2.9]))) # NaNs

