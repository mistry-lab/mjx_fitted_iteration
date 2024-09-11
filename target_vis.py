from simulate import controlled_simulate
from config.di import ctx as di_ctx
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from trainer import gen_targets_mapped

# get keys and generate init conditions
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
# generate a grid of initial conditions using a meshgrid
x = jnp.linspace(-1, 1, 100)
xv, yv = jnp.meshgrid(x, x)
x_inits = jnp.stack([xv.ravel(), yv.ravel()], axis=1)
# rollout the dynamics using these initial conditions with mujoco double integrator
model = mujoco.MjModel.from_xml_path(di_ctx.cfg.model_path)
model.opt.timestep = di_ctx.cfg.dt
data = mujoco.MjData(model)
mx = mjx.put_model(model)
sim = jax.vmap(controlled_simulate, in_axes=(0, None, None))
f_tagert = jax.vmap(gen_targets_mapped, in_axes=(0, 0, None))
x, u = sim(x_inits, mx, di_ctx)
# calculate the target values from out function
target, costs = f_tagert(x, u, di_ctx)
# plot the targets as a meshgrid with their associated states the first target is associated with the first state
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
v = target[:, 0, ...].reshape(100, 100)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, v, cmap='viridis')
plt.show()



